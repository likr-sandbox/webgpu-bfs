//! Massive Multi-Word Bit-Parallel GPU BFS dispatcher (Milestone M8).
//!
//! Scaled for massive multi-source batches: S in {256, 512, 1024, 2048}.
//! Uses 2D warp-tiled Pull traversal, single-pass level updates without atomics,
//! dynamic words-per-vertex selection (K in {8, 16, 32, 64}), and automatic
//! macro-chunking fallback when hardware storage buffer limits require it.

use anyhow::Result;
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;

use crate::{
    buffer::{read_buffer_u32_blocking, PushPullGraphBuffers},
    context::GpuContext,
    pipeline::PipelineCache,
};

/// Maximum steps recorded speculatively per command encoder submission.
const MAX_CHUNK_STEPS: usize = 48;

/// Uniform parameters matching `multi_word_bp_pull.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MultiWordBpParams {
    pub num_vertices: u32,
    pub words_per_vertex: u32,
    pub current_level: u32,
    pub total_sources: u32,
    pub dispatch_x: u32,
    pub _padding: [u32; 3],
}

/// Dynamically selects the word count K in {8, 16, 32, 64} for a given source count.
pub fn select_words_per_vertex(num_sources: usize) -> u32 {
    let raw_k = (num_sources + 31) / 32;
    if raw_k <= 8 {
        8
    } else if raw_k <= 16 {
        16
    } else if raw_k <= 32 {
        32
    } else {
        64
    }
}

/// Determines the maximum sources executable per macro-chunk based on device buffer limits.
pub fn determine_macro_chunk_size(
    device: &wgpu::Device,
    num_vertices: usize,
    total_sources: usize,
) -> usize {
    let limits = device.limits();
    let max_binding = limits.max_storage_buffer_binding_size as u64;
    let max_buffer = limits.max_buffer_size;
    let effective_limit = max_binding.min(max_buffer);

    let bytes_per_source = (num_vertices * std::mem::size_of::<u32>()) as u64;
    if bytes_per_source == 0 {
        return total_sources.max(1);
    }

    let max_s_by_buffer = (effective_limit / bytes_per_source) as usize;
    if max_s_by_buffer >= total_sources {
        return total_sources.max(1);
    }

    let max_allowed = max_s_by_buffer.min(2048);
    if max_allowed >= 2048 {
        2048
    } else if max_allowed >= 1024 {
        1024
    } else if max_allowed >= 512 {
        512
    } else if max_allowed >= 256 {
        256
    } else if max_allowed >= 32 {
        (max_allowed / 32) * 32
    } else {
        max_allowed.max(1)
    }
}

/// Runner struct for Massive Multi-Word Bit-Parallel BFS.
pub struct MultiWordBitParallelBfs;

impl MultiWordBitParallelBfs {
    /// Executes Multi-Source BFS across arbitrary sources up to 2048.
    pub fn run(
        ctx: &Arc<GpuContext>,
        pipelines: &Arc<PipelineCache>,
        graph: &CsrGraph,
        sources: &[u32],
    ) -> Result<Vec<u32>> {
        run_multi_word_bp(ctx, pipelines, graph, sources)
    }
}

/// Executes Multi-Source BFS using the Massive Multi-Word Bit-Parallel Pull pipeline.
pub fn run_multi_word_bp(
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

    // 1. Upload graph topology (CSC col_offsets and row_indices for Pull traversal)
    let graph_buffers = PushPullGraphBuffers::from_graph(ctx, graph);
    let row_offsets_buf = &graph_buffers.csc_col_offsets;
    let col_indices_buf = &graph_buffers.csc_row_indices;

    // 2. Determine macro-chunking sizing based on adapter limits
    let chunk_size = determine_macro_chunk_size(device, num_vertices, num_sources);
    let mut result = vec![u32::MAX; num_sources * num_vertices];

    // 3. Process sources in macro-chunks (single chunk when adapter limits permit)
    for (macro_idx, source_chunk) in sources.chunks(chunk_size).enumerate() {
        let chunk_len = source_chunk.len();
        let chunk_start_src = macro_idx * chunk_size;

        // Dynamic word count K in {8, 16, 32, 64}
        let words_k = select_words_per_vertex(chunk_len);
        let state_words = num_vertices * (words_k as usize);
        let state_bytes = (state_words * std::mem::size_of::<u32>()) as u64;
        let dist_chunk_words = chunk_len * num_vertices;
        let dist_chunk_bytes = (dist_chunk_words * std::mem::size_of::<u32>()) as u64;

        // Allocate working buffers for this chunk
        let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Frontier A"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Frontier B"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Visited"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Distance"),
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
                label: Some(&format!("MultiWord Counter {}", i)),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Staging Counter"),
            size: (MAX_CHUNK_STEPS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Staging Distance"),
            size: dist_chunk_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-allocate parameters uniform buffers and bind groups
        let pipeline_layout = pipelines.multi_word_bp_layout();
        let mut params_bufs = Vec::with_capacity(MAX_CHUNK_STEPS);
        let mut bgs_even = Vec::with_capacity(MAX_CHUNK_STEPS);
        let mut bgs_odd = Vec::with_capacity(MAX_CHUNK_STEPS);

        for step in 0..MAX_CHUNK_STEPS {
            let pbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("MultiWord Params {}", step)),
                size: std::mem::size_of::<MultiWordBpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Even steps: reads frontier_buf_a, writes frontier_buf_b
            let bg_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MultiWord BG Even {}", step)),
                layout: pipeline_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: row_offsets_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: col_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: frontier_buf_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: visited_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: frontier_buf_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: distance_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: counter_bufs[step].as_entire_binding(),
                    },
                ],
            });

            // Odd steps: reads frontier_buf_b, writes frontier_buf_a
            let bg_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MultiWord BG Odd {}", step)),
                layout: pipeline_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: row_offsets_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: col_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: frontier_buf_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: visited_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: frontier_buf_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: distance_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: counter_bufs[step].as_entire_binding(),
                    },
                ],
            });

            params_bufs.push(pbuf);
            bgs_even.push(bg_even);
            bgs_odd.push(bg_odd);
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
        queue.write_buffer(&visited_buf, 0, bytemuck::cast_slice(&init_visited));
        queue.write_buffer(&distance_buf, 0, bytemuck::cast_slice(&init_distance));

        // Compute dispatch geometry: 2D Warp-Tiled (32 words x 4 vertices = 128 threads)
        // Avoid exceeding WebGPU max_compute_workgroups_per_dimension (65535) via 2D grid
        let total_wgs = (nv + 3) / 4;
        let (dispatch_x, dispatch_y) = if total_wgs <= 65535 {
            (total_wgs.max(1), 1)
        } else {
            let dx = 256u32;
            let dy = (total_wgs + dx - 1) / dx;
            (dx, dy)
        };
        let compute_pipeline = pipelines.multi_word_bp_pipeline();

        // Host-Device speculative traversal loop
        let mut level = 0u32;
        let mut cur_chunk_steps = 1usize;

        while level <= nv {
            let chunk_len_steps = cur_chunk_steps
                .min(MAX_CHUNK_STEPS)
                .min((nv - level + 1) as usize);

            // 1. Write uniform parameters and clear counters
            for k in 0..chunk_len_steps {
                let p = MultiWordBpParams {
                    num_vertices: nv,
                    words_per_vertex: words_k,
                    current_level: level + k as u32,
                    total_sources: chunk_len as u32,
                    dispatch_x,
                    _padding: [0; 3],
                };
                queue.write_buffer(&params_bufs[k], 0, bytemuck::bytes_of(&p));
                queue.write_buffer(&counter_bufs[k], 0, bytemuck::bytes_of(&0u32));
            }

            // 2. Encode compute passes
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MultiWord Hop Batch Encoder"),
            });

            for k in 0..chunk_len_steps {
                let cur_step = level + k as u32;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(compute_pipeline);
                let bg = if cur_step % 2 == 0 {
                    &bgs_even[k]
                } else {
                    &bgs_odd[k]
                };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // 3. Copy per-hop counters into staging buffer
            for k in 0..chunk_len_steps {
                encoder.copy_buffer_to_buffer(
                    &counter_bufs[k],
                    0,
                    &staging_counter_buf,
                    (k * 4) as u64,
                    4,
                );
            }

            // 4. Submit and poll activity
            queue.submit(Some(encoder.finish()));

            let counts = read_buffer_u32_blocking(device, &staging_counter_buf, chunk_len_steps)?;
            if let Some(_zero_pos) = counts.iter().position(|&c| c == 0) {
                // BFS frontier has completely drained
                break;
            }

            level += chunk_len_steps as u32;
            cur_chunk_steps = (cur_chunk_steps * 2).min(MAX_CHUNK_STEPS);
        }

        // 5. Read back chunk distance matrix
        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MultiWord Distance Read Encoder"),
        });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist_buf, 0, dist_chunk_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_chunk = read_buffer_u32_blocking(device, &staging_dist_buf, dist_chunk_words)?;
        let start_dst = chunk_start_src * num_vertices;
        result[start_dst..start_dst + dist_chunk_words].copy_from_slice(&dist_chunk);
    }

    Ok(result)
}
