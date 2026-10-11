//! Massive Multi-Word Bit-Parallel GPU BFS Push Dispatcher (Milestone M12).
//!
//! Scaled for massive multi-source batches: S in {128, 256, 512, 1024, 2048}.
//! Uses 2D warp-tiled Push traversal with bounded out-degrees (max 255 on `web-Stanford`),
//! CSR edge reuse across word dimensions, dynamic words-per-vertex selection (K in {4, 8, 16, 32, 64}),
//! and automatic macro-chunking fallback when hardware storage buffer limits require it.

use anyhow::Result;
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;
use wgpu::util::DeviceExt;

use crate::{
    buffer::read_buffer_u32_blocking,
    context::GpuContext,
    dispatch::multi_word_bp::determine_macro_chunk_size,
    pipeline::PipelineCache,
};

/// Maximum steps recorded speculatively per command encoder submission.
const MAX_CHUNK_STEPS: usize = 48;

/// Push traversal uniform parameters matching `multi_word_bp_push.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushParams {
    pub num_vertices: u32,
    pub words_per_vertex: u32,
    pub current_level: u32,
    pub dispatch_x: u32,
}

/// Filter traversal uniform parameters matching `multi_word_bp_filter.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilterParams {
    pub num_vertices: u32,
    pub words_per_vertex: u32,
    pub current_level: u32,
    pub chunk_start: u32,
    pub total_sources: u32,
    pub dispatch_x: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Dynamically selects the word count K in {4, 8, 16, 32, 64} for a given source count.
pub fn select_words_per_vertex(num_sources: usize) -> u32 {
    let raw_k = (num_sources + 31) / 32;
    if raw_k <= 4 {
        4
    } else if raw_k <= 8 {
        8
    } else if raw_k <= 16 {
        16
    } else if raw_k <= 32 {
        32
    } else {
        64
    }
}

/// Executes Multi-Source BFS using the Massive Multi-Word Bit-Parallel Push pipeline.
pub fn run_multi_word_bp_push(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    if num_sources == 0 || num_vertices == 0 {
        return Ok(Vec::new());
    }

    let device = ctx.device();
    let queue = ctx.queue();
    let nv = num_vertices as u32;

    // 1. Upload static CSR topology
    let row_ptr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MWBP Push CSR Row Offsets"),
        contents: bytemuck::cast_slice(&graph.row_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dummy_col = [0u32];
    let col_slice: &[u32] = if graph.col_indices.is_empty() {
        &dummy_col
    } else {
        &graph.col_indices
    };
    let col_ind_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MWBP Push CSR Col Indices"),
        contents: bytemuck::cast_slice(col_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // 2. Determine macro-chunking sizing based on adapter limits (capped at 512 sources)
    let chunk_size = determine_macro_chunk_size(device, num_vertices, num_sources).min(512);
    let mut result = vec![u32::MAX; num_sources * num_vertices];

    // 3. Process sources in macro-chunks
    for (macro_idx, source_chunk) in sources.chunks(chunk_size).enumerate() {
        let chunk_len = source_chunk.len();
        let chunk_start_src = macro_idx * chunk_size;

        // Dynamic word count K in {4, 8, 16, 32, 64}
        let words_k = select_words_per_vertex(chunk_len);
        let state_words = num_vertices * (words_k as usize);
        let state_bytes = ((state_words * std::mem::size_of::<u32>()) as u64).max(4);
        let dist_chunk_words = chunk_len * num_vertices;
        let dist_chunk_bytes = ((dist_chunk_words * std::mem::size_of::<u32>()) as u64).max(4);

        // Allocate working buffers for this chunk
        let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MWBP Push Frontier A"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MWBP Push Frontier B"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let candidate_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MWBP Push Candidate"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MWBP Push Visited"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MWBP Push Distance"),
            size: dist_chunk_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Counter buffers for per-hop activity detection
        let mut counter_bufs = Vec::with_capacity(MAX_CHUNK_STEPS);
        for i in 0..MAX_CHUNK_STEPS {
            counter_bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("MWBP Push Counter {}", i)),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MWBP Push Staging Counter"),
            size: (MAX_CHUNK_STEPS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MWBP Push Staging Distance"),
            size: dist_chunk_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-allocate parameters uniform buffers and bind groups
        let push_layout = pipelines.multi_word_push_layout();
        let filter_layout = pipelines.multi_word_filter_layout();

        let mut push_params_bufs = Vec::with_capacity(MAX_CHUNK_STEPS);
        let mut filter_params_bufs = Vec::with_capacity(MAX_CHUNK_STEPS);

        let mut push_bgs_even = Vec::with_capacity(MAX_CHUNK_STEPS);
        let mut push_bgs_odd = Vec::with_capacity(MAX_CHUNK_STEPS);
        let mut filter_bgs_even = Vec::with_capacity(MAX_CHUNK_STEPS);
        let mut filter_bgs_odd = Vec::with_capacity(MAX_CHUNK_STEPS);

        for step in 0..MAX_CHUNK_STEPS {
            let p_push = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("MWBP Push Params {}", step)),
                size: std::mem::size_of::<PushParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let p_filter = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("MWBP Filter Params {}", step)),
                size: std::mem::size_of::<FilterParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Push Pass:
            // Even step: reads frontier_a, writes next_candidate
            let push_bg_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MWBP Push BG Even {}", step)),
                layout: push_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: p_push.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: row_ptr_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: col_ind_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: frontier_buf_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: candidate_buf.as_entire_binding(),
                    },
                ],
            });

            // Odd step: reads frontier_b, writes next_candidate
            let push_bg_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MWBP Push BG Odd {}", step)),
                layout: push_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: p_push.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: row_ptr_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: col_ind_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: frontier_buf_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: candidate_buf.as_entire_binding(),
                    },
                ],
            });

            // Filter Pass:
            // Even step: reads next_candidate, writes frontier_b
            let filter_bg_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MWBP Filter BG Even {}", step)),
                layout: filter_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: p_filter.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: candidate_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: visited_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: frontier_buf_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: distance_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: counter_bufs[step].as_entire_binding(),
                    },
                ],
            });

            // Odd step: reads next_candidate, writes frontier_a
            let filter_bg_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MWBP Filter BG Odd {}", step)),
                layout: filter_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: p_filter.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: candidate_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: visited_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: frontier_buf_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: distance_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: counter_bufs[step].as_entire_binding(),
                    },
                ],
            });

            push_params_bufs.push(p_push);
            filter_params_bufs.push(p_filter);
            push_bgs_even.push(push_bg_even);
            push_bgs_odd.push(push_bg_odd);
            filter_bgs_even.push(filter_bg_even);
            filter_bgs_odd.push(filter_bg_odd);
        }

        // Pack initial bitmasks and distance arrays
        let mut init_frontier = vec![0u32; state_words];
        let mut init_visited = vec![0u32; state_words];
        let mut init_distance = vec![u32::MAX; dist_chunk_words];

        let wk = words_k as usize;
        for (s_idx, &src) in source_chunk.iter().enumerate() {
            let src_idx = src as usize;
            let word_idx = s_idx / 32;
            let bit_pos = s_idx % 32;
            let mask = 1u32 << bit_pos;

            init_frontier[src_idx * wk + word_idx] |= mask;
            init_visited[src_idx * wk + word_idx] |= mask;
            init_distance[s_idx * num_vertices + src_idx] = 0;
        }

        queue.write_buffer(&frontier_buf_a, 0, bytemuck::cast_slice(&init_frontier));
        queue.write_buffer(
            &frontier_buf_b,
            0,
            bytemuck::cast_slice(&vec![0u32; state_words]),
        );
        queue.write_buffer(
            &candidate_buf,
            0,
            bytemuck::cast_slice(&vec![0u32; state_words]),
        );
        queue.write_buffer(&visited_buf, 0, bytemuck::cast_slice(&init_visited));
        queue.write_buffer(&distance_buf, 0, bytemuck::cast_slice(&init_distance));

        // Compute dispatch geometry: 2D Warp-Tiled (W_k = 8, W_v = 16 => 128 threads)
        let total_wgs = (nv + 15) / 16;
        let (dispatch_x, dispatch_y) = if total_wgs <= 65535 {
            (total_wgs.max(1), 1)
        } else {
            let dx = 256u32;
            let dy = (total_wgs + dx - 1) / dx;
            (dx, dy)
        };

        let push_pipeline = pipelines.multi_word_push_pipeline();
        let filter_pipeline = pipelines.multi_word_filter_pipeline();

        // Host-Device speculative traversal loop
        let mut level = 0u32;
        let mut cur_chunk_steps = 1usize;

        while level <= nv {
            let chunk_len_steps = cur_chunk_steps
                .min(MAX_CHUNK_STEPS)
                .min((nv - level + 1) as usize);

            // 1. Write uniform parameters and clear counters
            for k in 0..chunk_len_steps {
                let cur_lvl = level + k as u32;
                let push_p = PushParams {
                    num_vertices: nv,
                    words_per_vertex: words_k,
                    current_level: cur_lvl,
                    dispatch_x,
                };
                let filter_p = FilterParams {
                    num_vertices: nv,
                    words_per_vertex: words_k,
                    current_level: cur_lvl,
                    chunk_start: 0, // relative to this chunk
                    total_sources: chunk_len as u32,
                    dispatch_x,
                    _pad0: 0,
                    _pad1: 0,
                };
                queue.write_buffer(&push_params_bufs[k], 0, bytemuck::bytes_of(&push_p));
                queue.write_buffer(&filter_params_bufs[k], 0, bytemuck::bytes_of(&filter_p));
                queue.write_buffer(&counter_bufs[k], 0, bytemuck::bytes_of(&0u32));
            }

            // 2. Record speculative chunk passes into single command encoder
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MWBP Push Macro-Batch Encoder"),
            });

            for k in 0..chunk_len_steps {
                let cur_lvl = level + k as u32;
                let is_even = cur_lvl.is_multiple_of(2);

                // Pass 1: Push Traversal
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("MWBP Push Compute Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(push_pipeline);
                    pass.set_bind_group(0, if is_even { &push_bgs_even[k] } else { &push_bgs_odd[k] }, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                // Pass 2: Filter & Distance Update
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("MWBP Filter Compute Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(filter_pipeline);
                    pass.set_bind_group(0, if is_even { &filter_bgs_even[k] } else { &filter_bgs_odd[k] }, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                // Copy active counter to staging buffer
                encoder.copy_buffer_to_buffer(
                    &counter_bufs[k],
                    0,
                    &staging_counter_buf,
                    (k * std::mem::size_of::<u32>()) as u64,
                    4,
                );
            }

            // 3. Submit single command buffer for the entire speculative chunk
            queue.submit(Some(encoder.finish()));

            // 4. Synchronous readback of counter slice
            let counts = read_buffer_u32_blocking(device, &staging_counter_buf, chunk_len_steps)?;

            let mut all_active = true;
            for (step_offset, &c) in counts.iter().enumerate() {
                if c == 0 {
                    level += step_offset as u32;
                    all_active = false;
                    break;
                }
            }

            if !all_active {
                break;
            }

            level += chunk_len_steps as u32;
            cur_chunk_steps = (cur_chunk_steps * 2).min(MAX_CHUNK_STEPS);
        }

        // 5. Read back distance array for this chunk
        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MWBP Push Distance Readback Encoder"),
        });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist_buf, 0, dist_chunk_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_chunk = read_buffer_u32_blocking(device, &staging_dist_buf, dist_chunk_words)?;
        let start_dst = chunk_start_src * num_vertices;
        result[start_dst..start_dst + dist_chunk_words].copy_from_slice(&dist_chunk);
    }

    Ok(result)
}
