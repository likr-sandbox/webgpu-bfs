//! WebGPU Indirect Dispatch & Macro-Batching BFS Dispatcher (Milestone M9).
//!
//! Eliminates CPU-GPU host synchronization stalls and command recording overhead
//! on high-diameter graphs (e.g. `luxembourg_osm`) via:
//! 1. GPU-side autonomous hop iteration (`prep_hop.wgsl`).
//! 2. Speculative macro-batching (default 64 hops per command submission).
//! 3. Native WebGPU `dispatch_workgroups_indirect` with hardware zero-workgroup no-op exit.
//! 4. 4-byte `terminated` flag readback per chunk with instant host early termination.

use anyhow::Result;
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;

use crate::{
    buffer::{read_buffer_u32_blocking, PushPullGraphBuffers},
    context::GpuContext,
    pipeline::PipelineCache,
};

/// Number of speculative hops pre-recorded per command encoder submission.
pub const DEFAULT_CHUNK_HOPS: usize = 64;

/// GPU Indirect Dispatch Argument Layout matching WebGPU / Vulkan specification.
/// Exact size: 12 bytes. Alignment: 4 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndirectArgs {
    pub workgroups_x: u32,
    pub workgroups_y: u32,
    pub workgroups_z: u32,
}

/// GPU Iteration State Layout matching `IterationState` in WGSL.
/// Exact size: 16 bytes. Alignment: 4 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IterationState {
    pub current_level: u32,    // offset 0
    pub active_count: u32,     // offset 4 (atomic<u32> in WGSL)
    pub terminated: u32,       // offset 8
    pub max_workgroups_x: u32, // offset 12
}

/// Static parameters passed via uniform buffer to the traversal shader.
/// Exact size: 32 bytes (16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndirectStaticParams {
    pub num_vertices: u32,
    pub chunk_start: u32,
    pub mask_low: u32,
    pub mask_high: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

/// Buffer-pooled executor for Indirect Macro-Batched Multi-Source BFS.
pub struct IndirectMacroBatchBfs {
    ctx: Arc<GpuContext>,
    pipelines: Arc<PipelineCache>,
    #[allow(dead_code)]
    row_offsets_buf: wgpu::Buffer,
    #[allow(dead_code)]
    col_indices_buf: wgpu::Buffer,
    frontier_buf_a: wgpu::Buffer,
    frontier_buf_b: wgpu::Buffer,
    visited_buf: wgpu::Buffer,
    distance_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    indirect_args_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    staging_status_buf: wgpu::Buffer,
    staging_dist_buf: wgpu::Buffer,
    prep_bind_group: wgpu::BindGroup,
    bind_group_even: wgpu::BindGroup,
    bind_group_odd: wgpu::BindGroup,
    num_vertices: usize,
    chunk_hops: usize,
}

impl IndirectMacroBatchBfs {
    /// Initializes GPU storage buffers, indirect argument buffers, and static bind groups.
    pub fn new(
        ctx: Arc<GpuContext>,
        pipelines: Arc<PipelineCache>,
        graph: &CsrGraph,
        chunk_hops: usize,
    ) -> Result<Self> {
        let device = ctx.device();
        let num_vertices = graph.num_vertices;

        // 1. Upload CSC topology for Pull traversal
        let graph_buffers = PushPullGraphBuffers::from_graph(&ctx, graph);
        let row_offsets_buf = graph_buffers.csc_col_offsets;
        let col_indices_buf = graph_buffers.csc_row_indices;

        // 2. Allocate state buffers
        let state_words = 2 * num_vertices;
        let state_bytes = ((state_words * std::mem::size_of::<u32>()) as u64).max(4);
        let dist_batch_bytes = ((64 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);

        let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Frontier A"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Frontier B"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Visited"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Distances"),
            size: dist_batch_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Indirect Dispatch Arguments Buffer: INDIRECT | STORAGE | COPY_DST
        let indirect_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Args Buffer"),
            size: std::mem::size_of::<IndirectArgs>() as u64, // 12 bytes
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 4. IterationState Buffer: STORAGE | COPY_SRC | COPY_DST
        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Iteration State Buffer"),
            size: std::mem::size_of::<IterationState>() as u64, // 16 bytes
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 5. Static Parameters Uniform Buffer
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Static Params Buffer"),
            size: std::mem::size_of::<IndirectStaticParams>() as u64, // 32 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 6. Staging Buffers for Readback
        let staging_status_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Status Buffer"),
            size: 4, // 4-byte terminated flag
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Distance Buffer"),
            size: dist_batch_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 7. Bind Group for prep_hop (Binding 0 = IterationState, Binding 1 = IndirectArgs)
        let prep_layout = pipelines.prep_hop_layout();
        let prep_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Prep Hop BG"),
            layout: prep_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_args_buf.as_entire_binding(),
                },
            ],
        });

        // 8. Static Traversal Ping-Pong Bind Groups
        let trav_layout = pipelines.indirect_bp_pull_layout();

        // Even hop: reads frontier A, writes frontier B
        let bind_group_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Indirect Traversal BG Even"),
            layout: trav_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
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
                    resource: state_buf.as_entire_binding(),
                },
            ],
        });

        // Odd hop: reads frontier B, writes frontier A
        let bind_group_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Indirect Traversal BG Odd"),
            layout: trav_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
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
                    resource: state_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            ctx,
            pipelines,
            row_offsets_buf,
            col_indices_buf,
            frontier_buf_a,
            frontier_buf_b,
            visited_buf,
            distance_buf,
            state_buf,
            indirect_args_buf,
            params_buf,
            staging_status_buf,
            staging_dist_buf,
            prep_bind_group,
            bind_group_even,
            bind_group_odd,
            num_vertices,
            chunk_hops: chunk_hops.max(1),
        })
    }

    /// Executes Indirect Macro-Batched MS-BFS across all provided sources.
    pub fn run(&self, sources: &[u32]) -> Result<Vec<u32>> {
        let num_sources = sources.len();
        if num_sources == 0 || self.num_vertices == 0 {
            return Ok(Vec::new());
        }

        let device = self.ctx.device();
        let queue = self.ctx.queue();
        let nv = self.num_vertices as u32;
        let max_workgroups_x = nv.div_ceil(128);
        let state_words = 2 * self.num_vertices;

        let prep_pipeline = self.pipelines.prep_hop_pipeline();
        let trav_pipeline = self.pipelines.indirect_bp_pull_pipeline();

        let mut result = vec![u32::MAX; num_sources * self.num_vertices];

        // Process sources in chunks of up to 64
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

            // 1. Initialize static parameters uniform buffer (written ONCE)
            let params = IndirectStaticParams {
                num_vertices: nv,
                chunk_start: 0,
                mask_low,
                mask_high,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                pad3: 0,
            };
            queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            // 2. Initialize frontier, visited, and distance buffers
            let mut init_frontier = vec![0u32; state_words];
            let mut init_visited = vec![0u32; state_words];
            let mut init_distance = vec![u32::MAX; 64 * self.num_vertices];

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
                init_distance[s_idx * self.num_vertices + src_idx] = 0;
            }

            queue.write_buffer(&self.frontier_buf_a, 0, bytemuck::cast_slice(&init_frontier));
            queue.write_buffer(
                &self.frontier_buf_b,
                0,
                bytemuck::cast_slice(&vec![0u32; state_words]),
            );
            queue.write_buffer(&self.visited_buf, 0, bytemuck::cast_slice(&init_visited));
            queue.write_buffer(&self.distance_buf, 0, bytemuck::cast_slice(&init_distance));

            // 3. Initialize IterationState: active_count = batch_size, level = 0, terminated = 0
            let initial_state = IterationState {
                current_level: 0,
                active_count: batch_size as u32,
                terminated: 0,
                max_workgroups_x,
            };
            queue.write_buffer(&self.state_buf, 0, bytemuck::bytes_of(&initial_state));

            // 4. Macro-Batch Loop
            let mut level = 0u32;
            let max_hops = nv;

            while level < max_hops {
                let chunk_len = (self.chunk_hops as u32).min(max_hops - level) as usize;

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Indirect Macro-Batch Encoder"),
                });

                // Pre-record speculative hops
                for k in 0..chunk_len {
                    let cur_step = level + k as u32;
                    let is_even = cur_step.is_multiple_of(2);

                    // Pass 1: Prepare Hop (1 workgroup, 1 thread)
                    {
                        let mut prep_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("Prep Hop Compute Pass"),
                                timestamp_writes: None,
                            });
                        prep_pass.set_pipeline(prep_pipeline);
                        prep_pass.set_bind_group(0, &self.prep_bind_group, &[]);
                        prep_pass.dispatch_workgroups(1, 1, 1);
                    }

                    // Pass 2: Traversal Pass (Indirect Dispatch)
                    {
                        let mut trav_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("Traversal Indirect Compute Pass"),
                                timestamp_writes: None,
                            });
                        trav_pass.set_pipeline(trav_pipeline);
                        let bg = if is_even {
                            &self.bind_group_even
                        } else {
                            &self.bind_group_odd
                        };
                        trav_pass.set_bind_group(0, bg, &[]);
                        trav_pass.dispatch_workgroups_indirect(&self.indirect_args_buf, 0);
                    }
                }

                // Copy terminated flag (offset 8 in IterationState) to staging buffer
                encoder.copy_buffer_to_buffer(&self.state_buf, 8, &self.staging_status_buf, 0, 4);

                // Single submission per macro-batch chunk
                queue.submit(Some(encoder.finish()));

                // Read back 4-byte terminated flag
                let terminated = read_buffer_u32_blocking(device, &self.staging_status_buf, 1)?[0];
                if terminated != 0 {
                    break; // Instant early exit! All frontiers drained.
                }

                level += chunk_len as u32;
            }

            // 5. Read back final distance matrix for this batch
            let dist_words = batch_size * self.num_vertices;
            let dist_bytes = (dist_words * std::mem::size_of::<u32>()) as u64;

            let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Distance Read Encoder"),
            });
            read_enc.copy_buffer_to_buffer(
                &self.distance_buf,
                0,
                &self.staging_dist_buf,
                0,
                dist_bytes,
            );
            queue.submit(Some(read_enc.finish()));

            let dist_batch =
                read_buffer_u32_blocking(device, &self.staging_dist_buf, dist_words)?;
            let dst_offset = batch_start * self.num_vertices;
            result[dst_offset..dst_offset + dist_words].copy_from_slice(&dist_batch);
        }

        Ok(result)
    }
}

/// Convenience entry point executing Indirect Macro-Batched BFS.
pub fn run_indirect_macro_batch(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>> {
    let runner =
        IndirectMacroBatchBfs::new(ctx.clone(), pipelines.clone(), graph, DEFAULT_CHUNK_HOPS)?;
    runner.run(sources)
}
