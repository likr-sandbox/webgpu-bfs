//! 64-source Multi-Source Bit-Parallel Sparse Queue BFS dispatcher.

use anyhow::Result;
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;
use wgpu::util::DeviceExt;

use crate::{buffer::read_buffer_u32_blocking, context::GpuContext, pipeline::PipelineCache};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushParams {
    pub num_vertices: u32,
    pub current_level: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilterParams {
    pub num_vertices: u32,
    pub current_level: u32,
    pub chunk_start: u32,
    pub pad: u32,
}

/// Executes Multi-Source BFS using 64-source bit-parallel sparse queues.
///
/// Dispatches only over active vertices in Pass 1 and touched candidates in Pass 2,
/// achieving strict O(|F| + |E(F)|) complexity per hop with zero O(|V|) global sweeps.
pub fn run_sparse_bit_parallel_64(
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
        label: Some("Sparse BP CSR Row Offsets"),
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
        label: Some("Sparse BP CSR Col Indices"),
        contents: bytemuck::cast_slice(col_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // 2. Working storage buffers (allocated once, reused across 64-source chunks)
    let state_words = 2 * num_vertices;
    let state_bytes = ((state_words * std::mem::size_of::<u32>()) as u64).max(4);
    let v_bytes = ((num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
    let dist_batch_bytes = ((64 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);

    // Ping-pong frontier buffers (holds bitmask vec2<u32>)
    let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Frontier A"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Frontier B"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Ping-pong vertex queues (holds vertex IDs u32)
    let queue_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Queue A"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let queue_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Queue B"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Candidate queue and deduplication structures
    let candidate_queue_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Candidate Queue"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let candidate_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Next Candidate"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let touched_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Touched"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Visited"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Distances"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Queue sizes: [in_size, out_size, cand_size, reserved]
    let queue_sizes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Queue Sizes"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Uniform param buffers
    let push_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Push Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let filter_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Filter Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Staging readback buffers
    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Staging Distances"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_sizes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sparse BP Staging Queue Sizes"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 3. Pre-create Bind Groups
    let push_layout = pipelines.sparse_bp_push_layout();
    let filter_layout = pipelines.sparse_bp_filter_layout();

    // Push BG A: queue_in = queue_a, curr_frontier = frontier_a
    let push_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Sparse BP Push BG A"),
        layout: push_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: push_params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: row_offsets_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: col_indices_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: queue_buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: frontier_buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: candidate_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: touched_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: candidate_queue_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: queue_sizes_buf.as_entire_binding() },
        ],
    });

    // Push BG B: queue_in = queue_b, curr_frontier = frontier_b
    let push_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Sparse BP Push BG B"),
        layout: push_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: push_params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: row_offsets_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: col_indices_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: queue_buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: frontier_buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: candidate_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: touched_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: candidate_queue_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: queue_sizes_buf.as_entire_binding() },
        ],
    });

    // Filter BG A -> B: next_frontier = frontier_b, queue_out = queue_b
    let filter_bg_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Sparse BP Filter BG -> B"),
        layout: filter_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: filter_params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: candidate_queue_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: candidate_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: visited_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: frontier_buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: distance_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: touched_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: queue_buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: queue_sizes_buf.as_entire_binding() },
        ],
    });

    // Filter BG B -> A: next_frontier = frontier_a, queue_out = queue_a
    let filter_bg_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Sparse BP Filter BG -> A"),
        layout: filter_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: filter_params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: candidate_queue_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: candidate_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: visited_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: frontier_buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: distance_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: touched_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: queue_buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: queue_sizes_buf.as_entire_binding() },
        ],
    });

    let push_pipeline = pipelines.sparse_bp_push_pipeline();
    let filter_pipeline = pipelines.sparse_bp_filter_pipeline();

    let mut result = vec![u32::MAX; num_sources * num_vertices];
    let zeros_state = vec![0u32; state_words];
    let zeros_v = vec![0u32; num_vertices];

    // 4. Execute in chunks of up to 64 sources
    for (batch_idx, batch_sources) in sources.chunks(64).enumerate() {
        let batch_size = batch_sources.len();
        let batch_start = batch_idx * 64;

        let mut init_frontier = vec![0u32; state_words];
        let mut init_visited = vec![0u32; state_words];
        let mut init_distance = vec![u32::MAX; 64 * num_vertices];
        let mut init_queue: Vec<u32> = Vec::with_capacity(batch_size);

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
            if !init_queue.contains(&src) {
                init_queue.push(src);
            }
        }

        let initial_in_size = init_queue.len() as u32;

        // Upload initial state
        queue.write_buffer(&frontier_buf_a, 0, bytemuck::cast_slice(&init_frontier));
        queue.write_buffer(&frontier_buf_b, 0, bytemuck::cast_slice(&zeros_state));
        queue.write_buffer(&visited_buf, 0, bytemuck::cast_slice(&init_visited));
        queue.write_buffer(&distance_buf, 0, bytemuck::cast_slice(&init_distance));
        queue.write_buffer(&queue_buf_a, 0, bytemuck::cast_slice(&init_queue));
        queue.write_buffer(&touched_buf, 0, bytemuck::cast_slice(&zeros_v));
        queue.write_buffer(&candidate_buf, 0, bytemuck::cast_slice(&zeros_state));
        queue.write_buffer(&queue_sizes_buf, 0, bytemuck::cast_slice(&[initial_in_size, 0u32, 0u32, 0u32]));

        let mut current_in_size = initial_in_size;
        let mut level = 0u32;

        while current_in_size > 0 && level <= nv {
            let is_even = level.is_multiple_of(2);
            let push_bg = if is_even { &push_bg_a } else { &push_bg_b };
            let filter_bg = if is_even { &filter_bg_to_b } else { &filter_bg_to_a };

            // Dynamic Pass 1 Workgroup Count
            let push_wgs = current_in_size.div_ceil(128);

            // Update PushParams
            let p_push = PushParams {
                num_vertices: nv,
                current_level: level,
                pad0: 0,
                pad1: 0,
            };
            queue.write_buffer(&push_params_buf, 0, bytemuck::bytes_of(&p_push));

            // Update FilterParams for current level
            let p_filter = FilterParams {
                num_vertices: nv,
                current_level: level,
                chunk_start: 0,
                pad: 0,
            };
            queue.write_buffer(&filter_params_buf, 0, bytemuck::bytes_of(&p_filter));

            // Execute Pass 1 (Push Expansion)
            let mut enc1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(push_pipeline);
                pass.set_bind_group(0, push_bg, &[]);
                pass.dispatch_workgroups(push_wgs, 1, 1);
            }
            enc1.copy_buffer_to_buffer(&queue_sizes_buf, 0, &staging_sizes_buf, 0, 16);
            queue.submit(Some(enc1.finish()));

            // Readback cand_size to size Pass 2 dynamically
            let sizes = read_buffer_u32_blocking(device, &staging_sizes_buf, 4)?;
            let cand_size = sizes[2];
            if cand_size == 0 {
                break;
            }

            // Dynamic Pass 2 Workgroup Count
            let filter_wgs = cand_size.div_ceil(128);

            // Execute Pass 2 (Filter & Distance Update)
            let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(filter_pipeline);
                pass.set_bind_group(0, filter_bg, &[]);
                pass.dispatch_workgroups(filter_wgs, 1, 1);
            }
            enc2.copy_buffer_to_buffer(&queue_sizes_buf, 0, &staging_sizes_buf, 0, 16);
            queue.submit(Some(enc2.finish()));

            // Readback out_size for next hop and termination check
            let sizes2 = read_buffer_u32_blocking(device, &staging_sizes_buf, 4)?;
            let out_size = sizes2[1];
            if out_size == 0 {
                break;
            }

            // Setup sizes for next hop: in_size = out_size, out_size = 0, cand_size = 0
            current_in_size = out_size;
            queue.write_buffer(&queue_sizes_buf, 0, bytemuck::cast_slice(&[current_in_size, 0u32, 0u32, 0u32]));

            level += 1;
        }

        // Readback distance matrix for this 64-source batch
        let batch_elements = batch_size * num_vertices;
        let batch_bytes = (batch_elements * std::mem::size_of::<u32>()) as u64;

        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist_buf, 0, batch_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_batch = read_buffer_u32_blocking(device, &staging_dist_buf, batch_elements)?;
        let start_dst = batch_start * num_vertices;
        result[start_dst..start_dst + batch_elements].copy_from_slice(&dist_batch);
    }

    Ok(result)
}

/// Alias for `run_sparse_bit_parallel_64`.
pub fn run_sparse_bp_64(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>> {
    run_sparse_bit_parallel_64(ctx, pipelines, graph, sources)
}
