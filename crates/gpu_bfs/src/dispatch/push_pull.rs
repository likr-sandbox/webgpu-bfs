//! Adaptive Push/Pull direction-optimizing GPU BFS dispatcher.

use anyhow::Result;
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;

use crate::{
    buffer::{read_buffer_u32_blocking, PushPullGraphBuffers},
    context::GpuContext,
    pipeline::PipelineCache,
};

/// Direction state for the adaptive BFS controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    Push,
    Pull,
}

/// Dynamic switching configuration thresholds.
#[derive(Debug, Clone, Copy)]
pub struct PushPullConfig {
    /// Threshold parameter $\alpha$: switch from Push to Pull when $m_f > |E| / \alpha$.
    pub alpha: u32,
    /// Threshold parameter $\beta$: switch from Pull to Push when $n_f < |V| / \beta$.
    pub beta: u32,
}

impl Default for PushPullConfig {
    fn default() -> Self {
        Self {
            alpha: 14,
            beta: 24,
        }
    }
}

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
struct PullParams {
    num_vertices: u32,
    all_sources_mask: u32,
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

/// Executes Multi-Source BFS using the adaptive Push/Pull direction-optimizing pipeline.
/// Supports arbitrary source counts by chunking sources into batches of <= 32.
pub fn run_push_pull_adaptive(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    config: PushPullConfig,
) -> Result<Vec<u32>> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    if num_sources == 0 || num_vertices == 0 {
        return Ok(Vec::new());
    }

    let device = ctx.device();
    let queue = ctx.queue();
    let nv = num_vertices as u32;
    let num_edges = graph.num_edges;

    // 1. Upload graph topology (CSR & CSC)
    let graph_buffers = PushPullGraphBuffers::from_graph(ctx, graph);

    // 2. Working storage buffers (allocated once, reused across 32-source chunks)
    let v_bytes = (num_vertices * std::mem::size_of::<u32>()) as u64;
    let dist_batch_bytes = (32 * num_vertices * std::mem::size_of::<u32>()) as u64;

    let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Frontier A"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Frontier B"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let candidate_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Candidate"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Visited"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Distance Matrix"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Active Counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Staging buffers
    let staging_counter = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Staging Counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_dist = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Staging Distance"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Uniform buffers
    let push_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Push Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pull_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pull Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let filter_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Bool Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 3. Pre-create Push, Pull, and Filter bind groups
    let push_layout = pipelines.push_layout();
    let pull_layout = pipelines.pull_layout();
    let filter_layout = pipelines.bp_filter_layout();

    // Push: A -> Cand, B -> Cand
    let push_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PP Push BG A"),
        layout: push_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: push_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: graph_buffers.csr_row_offsets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: graph_buffers.csr_col_indices.as_entire_binding(),
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
        label: Some("PP Push BG B"),
        layout: push_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: push_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: graph_buffers.csr_row_offsets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: graph_buffers.csr_col_indices.as_entire_binding(),
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

    // Pull: A -> Cand, B -> Cand
    let pull_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PP Pull BG A"),
        layout: pull_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pull_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: graph_buffers.csc_col_offsets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: graph_buffers.csc_row_indices.as_entire_binding(),
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
                resource: candidate_buf.as_entire_binding(),
            },
        ],
    });
    let pull_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PP Pull BG B"),
        layout: pull_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pull_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: graph_buffers.csc_col_offsets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: graph_buffers.csc_row_indices.as_entire_binding(),
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
                resource: candidate_buf.as_entire_binding(),
            },
        ],
    });

    // Filter: Cand -> writes B, Cand -> writes A
    let filter_bg_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PP Filter BG -> B"),
        layout: filter_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: filter_params_buf.as_entire_binding(),
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
                resource: counter_buf.as_entire_binding(),
            },
        ],
    });
    let filter_bg_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PP Filter BG -> A"),
        layout: filter_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: filter_params_buf.as_entire_binding(),
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
                resource: counter_buf.as_entire_binding(),
            },
        ],
    });

    let workgroups_x = nv.div_ceil(128);
    let push_threshold_edges = (num_edges as u64) / (config.alpha as u64).max(1);
    let pull_threshold_vertices = (num_vertices as u64) / (config.beta as u64).max(1);
    let avg_degree = if num_vertices > 0 {
        num_edges as f64 / num_vertices as f64
    } else {
        0.0
    };

    let mut result = vec![u32::MAX; num_sources * num_vertices];

    // 4. Process in chunks of up to 32 sources
    for (chunk_idx, source_chunk) in sources.chunks(32).enumerate() {
        let chunk_size = source_chunk.len();
        let batch_start = chunk_idx * 32;
        let all_sources_mask = if chunk_size == 32 {
            u32::MAX
        } else {
            (1u32 << chunk_size) - 1u32
        };

        let mut init_visited = vec![0u32; num_vertices];
        let mut init_frontier = vec![0u32; num_vertices];
        let mut init_distance = vec![u32::MAX; 32 * num_vertices];

        for (s_idx, &src) in source_chunk.iter().enumerate() {
            let bit = 1u32 << s_idx;
            init_visited[src as usize] |= bit;
            init_frontier[src as usize] |= bit;
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

        let mut current_direction = TraversalDirection::Push;
        let mut level = 0u32;

        loop {
            let push_params = PushParams {
                num_vertices: nv,
                current_level: level,
                pad0: 0,
                pad1: 0,
            };
            let pull_params = PullParams {
                num_vertices: nv,
                all_sources_mask,
                pad0: 0,
                pad1: 0,
            };
            let filter_params = FilterParams {
                num_vertices: nv,
                current_level: level,
                chunk_start: 0,
                pad: 0,
            };

            queue.write_buffer(&push_params_buf, 0, bytemuck::bytes_of(&push_params));
            queue.write_buffer(&pull_params_buf, 0, bytemuck::bytes_of(&pull_params));
            queue.write_buffer(&filter_params_buf, 0, bytemuck::bytes_of(&filter_params));
            queue.write_buffer(&counter_buf, 0, bytemuck::bytes_of(&0u32));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PP Step"),
            });

            // Stage 1: Traversal Pass (Push or Pull)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PP Traversal Pass"),
                    timestamp_writes: None,
                });

                match current_direction {
                    TraversalDirection::Push => {
                        pass.set_pipeline(pipelines.push_pipeline());
                        let bg = if level.is_multiple_of(2) {
                            &push_bg_a
                        } else {
                            &push_bg_b
                        };
                        pass.set_bind_group(0, bg, &[]);
                    }
                    TraversalDirection::Pull => {
                        pass.set_pipeline(pipelines.pull_pipeline());
                        let bg = if level.is_multiple_of(2) {
                            &pull_bg_a
                        } else {
                            &pull_bg_b
                        };
                        pass.set_bind_group(0, bg, &[]);
                    }
                }
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            // Stage 2: Boolean Reachability & Distance Update Pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PP Boolean Filter Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.bool_reach_pipeline());
                let bg = if level.is_multiple_of(2) {
                    &filter_bg_to_b
                } else {
                    &filter_bg_to_a
                };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_counter, 0, 4);
            queue.submit(Some(encoder.finish()));

            let active_bits = read_buffer_u32_blocking(device, &staging_counter, 1)?[0];
            if active_bits == 0 {
                break;
            }

            // Direction controller decision for next level
            let n_f = (active_bits as u64).div_ceil(chunk_size as u64);
            let estimated_m_f = (n_f as f64 * avg_degree) as u64;

            match current_direction {
                TraversalDirection::Push => {
                    if estimated_m_f > push_threshold_edges {
                        current_direction = TraversalDirection::Pull;
                    }
                }
                TraversalDirection::Pull => {
                    if n_f < pull_threshold_vertices {
                        current_direction = TraversalDirection::Push;
                    }
                }
            }

            level += 1;
            if level > nv {
                break;
            }
        }

        // Readback distances for this chunk
        let batch_elements = chunk_size * num_vertices;
        let batch_bytes = (batch_elements * std::mem::size_of::<u32>()) as u64;

        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PP Readback Distances"),
        });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist, 0, batch_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_batch = read_buffer_u32_blocking(device, &staging_dist, batch_elements)?;
        let start_dst = batch_start * num_vertices;
        result[start_dst..start_dst + batch_elements].copy_from_slice(&dist_batch);
    }

    Ok(result)
}
