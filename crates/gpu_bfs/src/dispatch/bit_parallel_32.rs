//! 32-source Bit-Parallel GPU MS-BFS dispatcher.

use anyhow::Result;
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;
use wgpu::util::DeviceExt;

use crate::{buffer::read_buffer_u32_blocking, context::GpuContext, pipeline::PipelineCache};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushParams {
    num_vertices: u32,
    current_level: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FilterParams {
    num_vertices: u32,
    current_level: u32,
    chunk_start: u32,
    pad: u32,
}

/// Executes Multi-Source BFS using 32-source bit-parallel packed frontiers.
///
/// Sources are processed in chunks of up to 32, supporting arbitrary source slice sizes.
pub fn run_bit_parallel_32(
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

    // 1. Static graph buffers
    let row_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSR Row Offsets"),
        contents: bytemuck::cast_slice(&graph.row_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dummy_col = [0u32];
    let col_slice: &[u32] = if graph.col_indices.is_empty() {
        &dummy_col
    } else {
        &graph.col_indices
    };
    let col_indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSR Col Indices"),
        contents: bytemuck::cast_slice(col_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // 2. Working storage buffers (allocated once, reused across 32-source chunks)
    let v_bytes = (num_vertices * std::mem::size_of::<u32>()) as u64;
    let dist_batch_bytes = (32 * num_vertices * std::mem::size_of::<u32>()) as u64;

    let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Frontier A"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Frontier B"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let candidate_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Candidate"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Visited"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Distances"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let push_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Push Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    const CHUNK_STEPS: usize = 16;

    // 3. Staging buffers
    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Staging Distances"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Staging Counter"),
        size: (CHUNK_STEPS * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 4. Pre-create Push and Filter bind groups
    let push_layout = pipelines.bp_push_layout();
    let filter_layout = pipelines.bp_filter_layout();

    let push_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BP32 Push BG A"),
        layout: push_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: push_params_buf.as_entire_binding(),
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
                resource: candidate_buf.as_entire_binding(),
            },
        ],
    });

    let push_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BP32 Push BG B"),
        layout: push_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: push_params_buf.as_entire_binding(),
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
                resource: candidate_buf.as_entire_binding(),
            },
        ],
    });

    let mut filter_params_bufs = Vec::with_capacity(CHUNK_STEPS);
    let mut counter_bufs = Vec::with_capacity(CHUNK_STEPS);
    let mut filter_bgs_to_b = Vec::with_capacity(CHUNK_STEPS);
    let mut filter_bgs_to_a = Vec::with_capacity(CHUNK_STEPS);

    for _ in 0..CHUNK_STEPS {
        let fbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BP32 Filter Params"),
            size: std::mem::size_of::<FilterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BP32 Step Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BP32 Filter BG -> B"),
            layout: filter_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fbuf.as_entire_binding(),
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
                    resource: cbuf.as_entire_binding(),
                },
            ],
        });

        let bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BP32 Filter BG -> A"),
            layout: filter_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fbuf.as_entire_binding(),
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
                    resource: cbuf.as_entire_binding(),
                },
            ],
        });

        filter_params_bufs.push(fbuf);
        counter_bufs.push(cbuf);
        filter_bgs_to_b.push(bg_b);
        filter_bgs_to_a.push(bg_a);
    }

    let push_pipeline = pipelines.bp32_push_pipeline();
    let filter_pipeline = pipelines.bp32_filter_pipeline();
    let workgroups_x = nv.div_ceil(128);

    let mut result = vec![u32::MAX; num_sources * num_vertices];

    // 5. Execute in chunks of up to 32 sources
    for (batch_idx, batch_sources) in sources.chunks(32).enumerate() {
        let batch_size = batch_sources.len();
        let batch_start = batch_idx * 32;

        let mut init_frontier = vec![0u32; num_vertices];
        let mut init_visited = vec![0u32; num_vertices];
        let mut init_distance = vec![u32::MAX; 32 * num_vertices];

        for (s_idx, &src) in batch_sources.iter().enumerate() {
            let mask = 1u32 << s_idx;
            init_frontier[src as usize] |= mask;
            init_visited[src as usize] |= mask;
            init_distance[s_idx * num_vertices + src as usize] = 0;
        }

        queue.write_buffer(&frontier_buf_a, 0, bytemuck::cast_slice(&init_frontier));
        queue.write_buffer(
            &frontier_buf_b,
            0,
            bytemuck::cast_slice(&vec![0u32; num_vertices]),
        );
        queue.write_buffer(&visited_buf, 0, bytemuck::cast_slice(&init_visited));
        queue.write_buffer(&distance_buf, 0, bytemuck::cast_slice(&init_distance));
        queue.write_buffer(
            &candidate_buf,
            0,
            bytemuck::cast_slice(&vec![0u32; num_vertices]),
        );

        let p_push = PushParams {
            num_vertices: nv,
            current_level: 0,
            pad0: 0,
            pad1: 0,
        };
        queue.write_buffer(&push_params_buf, 0, bytemuck::bytes_of(&p_push));

        let mut level = 0u32;
        let mut cur_chunk_steps = 1usize;
        while level <= nv {
            let chunk_len = cur_chunk_steps.min(CHUNK_STEPS).min((nv - level + 1) as usize);

            for k in 0..chunk_len {
                let p_filter = FilterParams {
                    num_vertices: nv,
                    current_level: level + k as u32,
                    chunk_start: 0,
                    pad: 0,
                };
                queue.write_buffer(&filter_params_bufs[k], 0, bytemuck::bytes_of(&p_filter));
                queue.write_buffer(&counter_bufs[k], 0, bytemuck::bytes_of(&0u32));
            }

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            });

            for k in 0..chunk_len {
                let cur_step = level + k as u32;
                // Pass 1: Push Traversal
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(push_pipeline);
                    let push_bg = if cur_step.is_multiple_of(2) {
                        &push_bg_a
                    } else {
                        &push_bg_b
                    };
                    pass.set_bind_group(0, push_bg, &[]);
                    pass.dispatch_workgroups(workgroups_x, 1, 1);
                }

                // Pass 2: Reachability Filter & Distance Update
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(filter_pipeline);
                    let filter_bg = if cur_step.is_multiple_of(2) {
                        &filter_bgs_to_b[k]
                    } else {
                        &filter_bgs_to_a[k]
                    };
                    pass.set_bind_group(0, filter_bg, &[]);
                    pass.dispatch_workgroups(workgroups_x, 1, 1);
                }

                encoder.copy_buffer_to_buffer(&counter_bufs[k], 0, &staging_counter_buf, (k * 4) as u64, 4);
            }

            queue.submit(Some(encoder.finish()));

            let counts = read_buffer_u32_blocking(device, &staging_counter_buf, chunk_len)?;
            if let Some(_zero_pos) = counts.iter().position(|&c| c == 0) {
                break;
            }

            level += chunk_len as u32;
            cur_chunk_steps = (cur_chunk_steps * 2).min(CHUNK_STEPS);
        }

        // Readback distance matrix for this batch
        let batch_elements = batch_size * num_vertices;
        let batch_bytes = (batch_elements * std::mem::size_of::<u32>()) as u64;

        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BP32 Distances Readback"),
        });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist_buf, 0, batch_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_batch = read_buffer_u32_blocking(device, &staging_dist_buf, batch_elements)?;
        let start_dst = batch_start * num_vertices;
        result[start_dst..start_dst + batch_elements].copy_from_slice(&dist_batch);
    }

    Ok(result)
}
