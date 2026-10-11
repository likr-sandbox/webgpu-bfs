//! Single-source Sparse Frontier Queue GPU BFS dispatcher.
//!
//! Employs dynamic ping-pong active vertex queues and atomic distance claim
//! in WGSL, achieving strict O(|F| + |E(F)|) work complexity per level.

use anyhow::Result;
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;
use wgpu::util::DeviceExt;

use crate::{buffer::read_buffer_u32_blocking, context::GpuContext, pipeline::PipelineCache};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SparseQueueParams {
    current_level: u32,
    num_vertices: u32,
    pad0: u32,
    pad1: u32,
}

/// Standalone buffer-pooled Single-Source Sparse Queue BFS executor.
pub struct SparseQueueBfs {
    ctx: Arc<GpuContext>,
    pipelines: Arc<PipelineCache>,
    #[allow(dead_code)]
    row_offsets_buf: wgpu::Buffer,
    #[allow(dead_code)]
    col_indices_buf: wgpu::Buffer,
    queue_buf_a: wgpu::Buffer,
    #[allow(dead_code)]
    queue_buf_b: wgpu::Buffer,
    distances_buf: wgpu::Buffer,
    queue_sizes_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    staging_dist_buf: wgpu::Buffer,
    staging_size_buf: wgpu::Buffer,
    bind_group_a_to_b: wgpu::BindGroup,
    bind_group_b_to_a: wgpu::BindGroup,
    num_vertices: usize,
}

impl SparseQueueBfs {
    /// Initializes GPU storage buffers and bind groups for the given graph topology.
    pub fn new(
        ctx: Arc<GpuContext>,
        pipelines: Arc<PipelineCache>,
        graph: &CsrGraph,
    ) -> Result<Self> {
        let device = ctx.device();
        let num_vertices = graph.num_vertices;

        let queue_bytes = ((num_vertices * std::mem::size_of::<u32>()) as u64).max(4);

        let row_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SparseQueue Row Offsets"),
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
            label: Some("SparseQueue Col Indices"),
            contents: bytemuck::cast_slice(col_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let queue_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Queue A"),
            size: queue_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let queue_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Queue B"),
            size: queue_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let distances_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Distances"),
            size: queue_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 4 x u32: [in_size, out_size, cand_size, reserved]
        let queue_sizes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Queue Sizes"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Queue Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Sparse Distances"),
            size: queue_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_size_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Sparse Queue Size"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = pipelines.sparse_queue_layout();

        let bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SparseQueue BG A->B"),
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
                    resource: queue_buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: queue_buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: distances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: queue_sizes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let bind_group_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SparseQueue BG B->A"),
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
                    resource: queue_buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: queue_buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: distances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: queue_sizes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            ctx,
            pipelines,
            row_offsets_buf,
            col_indices_buf,
            queue_buf_a,
            queue_buf_b,
            distances_buf,
            queue_sizes_buf,
            params_buf,
            staging_dist_buf,
            staging_size_buf,
            bind_group_a_to_b,
            bind_group_b_to_a,
            num_vertices,
        })
    }

    /// Executes sequential BFS across all provided source vertices.
    pub fn run(&self, sources: &[u32]) -> Result<Vec<u32>> {
        let num_sources = sources.len();
        if num_sources == 0 || self.num_vertices == 0 {
            return Ok(Vec::new());
        }

        let device = self.ctx.device();
        let queue = self.ctx.queue();
        let nv = self.num_vertices as u32;
        let dist_bytes = (self.num_vertices * std::mem::size_of::<u32>()) as u64;

        let pipeline = self.pipelines.sparse_queue_pipeline();
        let mut result = vec![u32::MAX; num_sources * self.num_vertices];
        let mut initial_dist = vec![u32::MAX; self.num_vertices];

        for (s_idx, &src) in sources.iter().enumerate() {
            // 1. Reset distances: dist[src] = 0, all others = INF
            initial_dist[src as usize] = 0;
            queue.write_buffer(&self.distances_buf, 0, bytemuck::cast_slice(&initial_dist));
            initial_dist[src as usize] = u32::MAX;

            // 2. Enqueue source vertex into queue_buf_a at index 0
            queue.write_buffer(&self.queue_buf_a, 0, bytemuck::cast_slice(&[src]));

            // 3. Initialize queue sizes: [in_size = 1, out_size = 0, cand = 0, pad = 0]
            let mut in_size = 1u32;
            queue.write_buffer(
                &self.queue_sizes_buf,
                0,
                bytemuck::cast_slice(&[in_size, 0u32, 0u32, 0u32]),
            );

            let mut level = 0u32;
            loop {
                let params = SparseQueueParams {
                    current_level: level,
                    num_vertices: nv,
                    pad0: 0,
                    pad1: 0,
                };
                queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

                let workgroups_x = in_size.div_ceil(128);

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Sparse Queue Level Step"),
                });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Sparse Queue Compute Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    let bg = if level.is_multiple_of(2) {
                        &self.bind_group_a_to_b
                    } else {
                        &self.bind_group_b_to_a
                    };
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(workgroups_x, 1, 1);
                }

                // Copy queue_sizes[1] (out_size at offset 4) to staging buffer
                encoder.copy_buffer_to_buffer(&self.queue_sizes_buf, 4, &self.staging_size_buf, 0, 4);
                queue.submit(Some(encoder.finish()));

                // Readback out_size to check termination
                let out_size = read_buffer_u32_blocking(device, &self.staging_size_buf, 1)?[0];
                if out_size == 0 {
                    break;
                }

                // Prepare next level: in_size becomes previous out_size, reset out_size to 0
                in_size = out_size;
                queue.write_buffer(
                    &self.queue_sizes_buf,
                    0,
                    bytemuck::cast_slice(&[in_size, 0u32, 0u32, 0u32]),
                );

                level += 1;
                if level > nv {
                    break; // safety guard
                }
            }

            // Readback distance array for current source
            let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Distances Readback"),
            });
            read_enc.copy_buffer_to_buffer(
                &self.distances_buf,
                0,
                &self.staging_dist_buf,
                0,
                dist_bytes,
            );
            queue.submit(Some(read_enc.finish()));

            let dist_s = read_buffer_u32_blocking(device, &self.staging_dist_buf, self.num_vertices)?;
            let start = s_idx * self.num_vertices;
            result[start..start + self.num_vertices].copy_from_slice(&dist_s);
        }

        Ok(result)
    }
}

/// Executes Single-Source Sparse Queue BFS on the GPU.
pub fn run_sparse_queue(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>> {
    let runner = SparseQueueBfs::new(ctx.clone(), pipelines.clone(), graph)?;
    runner.run(sources)
}
