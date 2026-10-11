//! Naive sequential single-source GPU BFS loop dispatcher.

use anyhow::Result;
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;
use wgpu::util::DeviceExt;

use crate::{buffer::read_buffer_u32_blocking, context::GpuContext, pipeline::PipelineCache};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NaiveParams {
    num_vertices: u32,
    current_level: u32,
    pad0: u32,
    pad1: u32,
}

pub fn run_naive_sequential(
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

    // 2. Working storage buffers (allocated once, reused across sources)
    let frontier_bytes = (num_vertices * std::mem::size_of::<u32>()) as u64;
    let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Frontier A"),
        size: frontier_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Frontier B"),
        size: frontier_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let distances_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Distances"),
        size: frontier_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Active Counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Naive Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 3. Staging buffers
    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Distances"),
        size: frontier_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 4. Pre-create ping-pong bind groups
    let layout = pipelines.naive_spmv_layout();
    let bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Naive BG A->B"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: row_offsets_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: col_indices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: frontier_buf_a.as_entire_binding(),
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
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: counter_buf.as_entire_binding(),
            },
        ],
    });
    let bind_group_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Naive BG B->A"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: row_offsets_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: col_indices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: frontier_buf_b.as_entire_binding(),
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
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: counter_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline = pipelines.naive_spmv_pipeline();
    let workgroups_x = nv.div_ceil(128);
    let mut result = vec![u32::MAX; num_sources * num_vertices];

    let mut initial_dist = vec![u32::MAX; num_vertices];
    let mut initial_frontier = vec![0u32; num_vertices];

    for (s_idx, &src) in sources.iter().enumerate() {
        // Reset buffers for current source
        initial_dist[src as usize] = 0;
        queue.write_buffer(&distances_buf, 0, bytemuck::cast_slice(&initial_dist));
        initial_dist[src as usize] = u32::MAX; // reset template

        initial_frontier[src as usize] = 1;
        queue.write_buffer(&frontier_buf_a, 0, bytemuck::cast_slice(&initial_frontier));
        initial_frontier[src as usize] = 0; // reset template

        // Ping-pong level loop
        let mut level = 0u32;
        loop {
            let params = NaiveParams {
                num_vertices: nv,
                current_level: level,
                pad0: 0,
                pad1: 0,
            };
            queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
            queue.write_buffer(&counter_buf, 0, bytemuck::bytes_of(&0u32));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Naive Level Step"),
            });

            // Clear target write frontier buffer
            let target_buf = if level.is_multiple_of(2) {
                &frontier_buf_b
            } else {
                &frontier_buf_a
            };
            encoder.clear_buffer(target_buf, 0, None);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Naive Compute Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                let bg = if level.is_multiple_of(2) {
                    &bind_group_a_to_b
                } else {
                    &bind_group_b_to_a
                };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            // Copy counter to staging buffer for level termination check
            encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_counter_buf, 0, 4);
            queue.submit(Some(encoder.finish()));

            // Check termination
            let active = read_buffer_u32_blocking(device, &staging_counter_buf, 1)?[0];
            if active == 0 {
                break;
            }

            level += 1;
            if level > nv {
                break; // safety break to prevent infinite loops
            }
        }

        // Readback distances for current source
        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Distances Readback"),
        });
        read_enc.copy_buffer_to_buffer(&distances_buf, 0, &staging_dist_buf, 0, frontier_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_s = read_buffer_u32_blocking(device, &staging_dist_buf, num_vertices)?;
        let start = s_idx * num_vertices;
        result[start..start + num_vertices].copy_from_slice(&dist_s);
    }

    Ok(result)
}
