//! Single-Pass Bit-Parallel 64-source GPU BFS dispatcher.
//!
//! Uses a single compute pass per hop via reverse (Pull) traversal,
//! eliminating the intermediate candidate buffer and global atomics while
//! halving the host command recording overhead.

use anyhow::Result;
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;

use crate::{
    buffer::{read_buffer_u32_blocking, PushPullGraphBuffers},
    context::GpuContext,
    pipeline::PipelineCache,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Pull64Params {
    num_vertices: u32,
    current_level: u32,
    chunk_start: u32,
    mask_low: u32,
    mask_high: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Executes Multi-Source BFS using the Single-Pass Bit-Parallel 64 Pull pipeline.
pub fn run_bit_parallel_64_pull(
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

    // 1. Upload graph topology (CSC for Pull traversal)
    let graph_buffers = PushPullGraphBuffers::from_graph(ctx, graph);
    let row_offsets_buf = &graph_buffers.csc_col_offsets;
    let col_indices_buf = &graph_buffers.csc_row_indices;

    // 2. Allocate state buffers
    let state_words = 2 * num_vertices;
    let state_bytes = (state_words * std::mem::size_of::<u32>()) as u64;
    let dist_batch_bytes = (64 * num_vertices * std::mem::size_of::<u32>()) as u64;

    let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64Pull Frontier A"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64Pull Frontier B"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64Pull Visited"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64Pull Distance"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64Pull Active Counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64Pull Staging Counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64Pull Staging Distance"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 3. Pre-allocate speculative chunk execution resources
    const CHUNK_STEPS: usize = 48;
    let pull_layout = pipelines.bp64_pull_layout();
    let mut params_bufs = Vec::with_capacity(CHUNK_STEPS);
    let mut bgs_even = Vec::with_capacity(CHUNK_STEPS);
    let mut bgs_odd = Vec::with_capacity(CHUNK_STEPS);

    for _ in 0..CHUNK_STEPS {
        let pbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BP64Pull Params"),
            size: std::mem::size_of::<Pull64Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Even steps: reads frontier_buf_a, writes frontier_buf_b
        let bg_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: pull_layout,
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
                    resource: counter_buf.as_entire_binding(),
                },
            ],
        });

        // Odd steps: reads frontier_buf_b, writes frontier_buf_a
        let bg_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: pull_layout,
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
                    resource: counter_buf.as_entire_binding(),
                },
            ],
        });

        params_bufs.push(pbuf);
        bgs_even.push(bg_even);
        bgs_odd.push(bg_odd);
    }

    let pull_pipeline = pipelines.bp64_pull_pipeline();
    let workgroups_x = nv.div_ceil(128);

    let mut result = vec![u32::MAX; num_sources * num_vertices];

    // 4. Execute in chunks of up to 64 sources
    for (batch_idx, batch_sources) in sources.chunks(64).enumerate() {
        let batch_size = batch_sources.len();
        let batch_start = batch_idx * 64;

        let mask_low = if batch_size >= 32 {
            u32::MAX
        } else {
            (1u32 << batch_size) - 1u32
        };
        let mask_high = if batch_size > 32 {
            if batch_size == 64 {
                u32::MAX
            } else {
                (1u32 << (batch_size - 32)) - 1u32
            }
        } else {
            0u32
        };

        let mut init_frontier = vec![0u32; state_words];
        let mut init_visited = vec![0u32; state_words];
        let mut init_distance = vec![u32::MAX; 64 * num_vertices];

        for (s_idx, &src) in batch_sources.iter().enumerate() {
            let src_idx = src as usize;
            if s_idx < 32 {
                let mask = 1u32 << s_idx;
                init_frontier[src_idx * 2] |= mask;
                init_visited[src_idx * 2] |= mask;
            } else {
                let mask = 1u32 << (s_idx - 32);
                init_frontier[src_idx * 2 + 1] |= mask;
                init_visited[src_idx * 2 + 1] |= mask;
            }
            init_distance[s_idx * num_vertices + src_idx] = 0;
        }

        queue.write_buffer(&frontier_buf_a, 0, bytemuck::cast_slice(&init_frontier));
        queue.write_buffer(&frontier_buf_b, 0, bytemuck::cast_slice(&vec![0u32; state_words]));
        queue.write_buffer(&visited_buf, 0, bytemuck::cast_slice(&init_visited));
        queue.write_buffer(&distance_buf, 0, bytemuck::cast_slice(&init_distance));

        let mut level = 0u32;
        let mut cur_chunk_steps = 1usize;

        while level <= nv {
            let chunk_len = cur_chunk_steps.min(CHUNK_STEPS).min((nv - level + 1) as usize);

            for k in 0..chunk_len {
                let p = Pull64Params {
                    num_vertices: nv,
                    current_level: level + k as u32,
                    chunk_start: 0,
                    mask_low,
                    mask_high,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                };
                queue.write_buffer(&params_bufs[k], 0, bytemuck::bytes_of(&p));
            }
            queue.write_buffer(&counter_buf, 0, bytemuck::bytes_of(&0u32));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            });

            // Single pass per hop!
            for k in 0..chunk_len {
                let cur_step = level + k as u32;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(pull_pipeline);
                let bg = if cur_step.is_multiple_of(2) {
                    &bgs_even[k]
                } else {
                    &bgs_odd[k]
                };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_counter_buf, 0, 4);

            queue.submit(Some(encoder.finish()));

            let active = read_buffer_u32_blocking(device, &staging_counter_buf, 1)?[0];

            if active == 0 {
                break;
            }

            level += chunk_len as u32;
            cur_chunk_steps = (cur_chunk_steps * 2).min(CHUNK_STEPS);
        }

        // Readback distance matrix for this batch
        let batch_elements = batch_size * num_vertices;
        let batch_bytes = (batch_elements * std::mem::size_of::<u32>()) as u64;

        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist_buf, 0, batch_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_batch = read_buffer_u32_blocking(device, &staging_dist_buf, batch_elements)?;
        let start_dst = batch_start * num_vertices;
        result[start_dst..start_dst + batch_elements].copy_from_slice(&dist_batch);
    }

    Ok(result)
}
