//! 3-Stage Boolean Reachability GPU MS-BFS Dispatcher.

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
struct ReachParams {
    num_vertices: u32,
    current_level: u32,
    chunk_start: u32,
    pad: u32,
}

/// Executes Multi-Source BFS using the 3-stage pipelined Boolean Reachability engine.
///
/// Supports arbitrary source counts by chunking sources into 32-source words and writing
/// directly into the global distance buffer via `chunk_start`.
pub fn run_boolean_reachability(
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
    let total_elements = num_sources * num_vertices;
    let matrix_bytes = (total_elements * std::mem::size_of::<u32>()) as u64;
    let v_bytes = (num_vertices * std::mem::size_of::<u32>()) as u64;

    // 1. Static CSR Graph Buffers
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

    // 2. Global Distance Buffer [S * |V|] initialized to u32::MAX, with source distances = 0
    let mut initial_dist = vec![u32::MAX; total_elements];
    for (s_idx, &src) in sources.iter().enumerate() {
        initial_dist[s_idx * num_vertices + src as usize] = 0;
    }

    let distances_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Global Distance Buffer"),
        contents: bytemuck::cast_slice(&initial_dist),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // 3. Reusable State Buffers for Bit-Parallel Traversals [|V| x u32]
    let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Frontier A"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Frontier B"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let candidate_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Candidate Buffer"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Visited Buffer"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    const CHUNK_STEPS: usize = 16;
    let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Counter"),
        size: (CHUNK_STEPS * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let push_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Push Params Buffer"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 4. Pre-create Bind Groups
    let push_layout = pipelines.bool_push_layout();
    let push_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Push BG A"),
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
        label: Some("Push BG B"),
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

    let reach_layout = pipelines.bool_reach_layout();
    let mut reach_params_bufs = Vec::with_capacity(CHUNK_STEPS);
    let mut counter_bufs = Vec::with_capacity(CHUNK_STEPS);
    let mut reach_bgs_to_b = Vec::with_capacity(CHUNK_STEPS);
    let mut reach_bgs_to_a = Vec::with_capacity(CHUNK_STEPS);

    for _ in 0..CHUNK_STEPS {
        let rbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reach Params"),
            size: std::mem::size_of::<ReachParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reach Step Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reach BG to B"),
            layout: reach_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rbuf.as_entire_binding(),
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
                    resource: distances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cbuf.as_entire_binding(),
                },
            ],
        });

        let bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reach BG to A"),
            layout: reach_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rbuf.as_entire_binding(),
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
                    resource: distances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cbuf.as_entire_binding(),
                },
            ],
        });

        reach_params_bufs.push(rbuf);
        counter_bufs.push(cbuf);
        reach_bgs_to_b.push(bg_b);
        reach_bgs_to_a.push(bg_a);
    }

    let push_pipeline = pipelines.bool_push_pipeline();
    let reach_pipeline = pipelines.bool_reach_pipeline();
    let workgroups_x = nv.div_ceil(128);

    // 5. Process Batches in Chunks of up to 32 Sources
    for (chunk_idx, chunk) in sources.chunks(32).enumerate() {
        let chunk_start = (chunk_idx * 32) as u32;

        let mut initial_frontier = vec![0u32; num_vertices];
        let mut initial_visited = vec![0u32; num_vertices];
        for (k, &src) in chunk.iter().enumerate() {
            let mask = 1u32 << k;
            initial_frontier[src as usize] |= mask;
            initial_visited[src as usize] |= mask;
        }

        queue.write_buffer(&frontier_buf_a, 0, bytemuck::cast_slice(&initial_frontier));
        queue.write_buffer(
            &frontier_buf_b,
            0,
            bytemuck::cast_slice(&vec![0u32; num_vertices]),
        );
        queue.write_buffer(&visited_buf, 0, bytemuck::cast_slice(&initial_visited));
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
                let p_reach = ReachParams {
                    num_vertices: nv,
                    current_level: level + k as u32,
                    chunk_start,
                    pad: 0,
                };
                queue.write_buffer(&reach_params_bufs[k], 0, bytemuck::bytes_of(&p_reach));
                queue.write_buffer(&counter_bufs[k], 0, bytemuck::bytes_of(&0u32));
            }

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            });

            for k in 0..chunk_len {
                let cur_step = level + k as u32;
                // STAGE 1: Traversal Pass (bit_parallel_push)
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(push_pipeline);
                    let bg = if cur_step.is_multiple_of(2) {
                        &push_bg_a
                    } else {
                        &push_bg_b
                    };
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(workgroups_x, 1, 1);
                }

                // STAGES 2 & 3: Filter & Distance Update Pass (boolean_reachability)
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(reach_pipeline);
                    let bg = if cur_step.is_multiple_of(2) {
                        &reach_bgs_to_b[k]
                    } else {
                        &reach_bgs_to_a[k]
                    };
                    pass.set_bind_group(0, bg, &[]);
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
    }

    // 6. Single Final Readback of the Entire Distance Matrix
    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Distances Readback"),
        size: matrix_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Final Distance Matrix Readback Encoder"),
    });
    read_enc.copy_buffer_to_buffer(&distances_buf, 0, &staging_dist_buf, 0, matrix_bytes);
    queue.submit(Some(read_enc.finish()));

    read_buffer_u32_blocking(device, &staging_dist_buf, total_elements)
}
