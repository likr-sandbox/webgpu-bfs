//! Single-Dispatch Autonomous Workgroup-Persistent GPU MS-BFS Dispatcher (Milestone M26).
//!
//! Features:
//! - 1 Workgroup per source, single compute dispatch for all hops (0 host roundtrips).
//! - 100% on-chip SRAM visited bitmask and frontier queue (<= 16,384 bytes shared memory).
//! - Direct bit-inverted half-word distance encoding (`packed_dist: array<atomic<u32>>`).
//! - Spatial source sorting with zero-cost permutation inversion.
//! - Double-buffered asynchronous DMA pipelining for S > 512.
//! - Parallel multithreaded Rayon distance unpacking and reachability bitmask scatter.

use anyhow::{Context, Result};
use std::sync::Arc;
use webgpu_bfs_core::{CsrGraph, ReachabilityBitmask};

use crate::{
    buffer::PushPullGraphBuffers,
    context::GpuContext,
    pipeline::PipelineCache,
};

/// Maximum WebGPU storage buffer binding size default limit (128 MiB).
pub const MAX_STORAGE_BINDING_BYTES: u64 = 128 * 1024 * 1024;

/// Workgroup shared memory limit on WebGPU (16 KiB).
pub const MAX_WORKGROUP_SHARED_BYTES: usize = 16384;

/// Maximum vertices supported in on-chip SRAM visited bitmask (3584 u32 words * 32 = 114,688).
pub const SRAM_MAX_VERTICES: usize = 114_688;
pub const SRAM_VISITED_WORDS: usize = 3584;

/// Uniform parameters matching `workgroup_persistent_bfs.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorkgroupPersistentParams {
    pub num_vertices: u32,
    pub num_sources: u32,
    pub use_sram_visited: u32,
    pub reachability_only: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

/// Dynamically calculates max sources per chunk respecting WebGPU 128 MB storage limit.
pub fn calculate_chunk_size(device: &wgpu::Device, num_vertices: usize) -> usize {
    let limits = device.limits();
    let max_binding = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
    let words_per_source = num_vertices.div_ceil(2) as u64;
    let bytes_per_source = (words_per_source * 4).max(4);
    let max_sources = (max_binding / bytes_per_source) as usize;
    max_sources.clamp(1, 512)
}

/// Pre-allocated, reusable GPU storage buffers for Workgroup Persistent BFS.
pub struct PersistentWorkgroupBfsBuffers {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub max_sources_per_chunk: usize,
    pub words_per_source: usize,
    pub visited_words_per_source: usize,

    pub row_offsets: Arc<wgpu::Buffer>,
    pub col_indices: Arc<wgpu::Buffer>,

    pub params_buf: wgpu::Buffer,
    pub sources_buf: wgpu::Buffer,
    pub packed_dist_buf: wgpu::Buffer,
    pub vram_visited_buf: wgpu::Buffer,
    pub visited_out_buf: wgpu::Buffer,
    pub overflow_buf: wgpu::Buffer,

    pub staging_dist_bufs: [wgpu::Buffer; 2],
    pub staging_visited_bufs: [wgpu::Buffer; 2],
    pub staging_overflow_bufs: [wgpu::Buffer; 2],

    pub bind_group: wgpu::BindGroup,
}

impl PersistentWorkgroupBfsBuffers {
    pub fn new(
        ctx: &GpuContext,
        pipelines: &PipelineCache,
        graph: &CsrGraph,
    ) -> Self {
        let device = ctx.device();
        let num_vertices = graph.num_vertices;
        let num_edges = graph.num_edges;

        // 1. Upload graph topology (CSR format for forward push)
        let graph_buffers = PushPullGraphBuffers::from_symmetric_graph(ctx, graph);
        let row_offsets = graph_buffers.csr_row_offsets;
        let col_indices = graph_buffers.csr_col_indices;

        let words_per_source = num_vertices.div_ceil(2);
        let visited_words_per_source = num_vertices.div_ceil(32);

        let max_sources_per_chunk = calculate_chunk_size(device, num_vertices);

        // 2. Traversal parameter and source buffers
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent BFS Params"),
            size: std::mem::size_of::<WorkgroupPersistentParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent BFS Sources"),
            size: (max_sources_per_chunk * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Packed distance buffer (half-word bit-inverted encoding)
        let dist_bytes = (max_sources_per_chunk * words_per_source * std::mem::size_of::<u32>()) as u64;
        let packed_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent BFS Packed Distances"),
            size: dist_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 4. VRAM visited fallback buffer (for |V| > 114,688)
        let vram_visited_bytes = if num_vertices > SRAM_MAX_VERTICES {
            (max_sources_per_chunk * visited_words_per_source * std::mem::size_of::<u32>()) as u64
        } else {
            4u64
        };
        let vram_visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent BFS VRAM Visited Fallback"),
            size: vram_visited_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 5. Reachability visited out buffer (flushed from SRAM or VRAM)
        let visited_out_words = max_sources_per_chunk * visited_words_per_source.max(SRAM_VISITED_WORDS);
        let visited_out_bytes = (visited_out_words * std::mem::size_of::<u32>()) as u64;
        let visited_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent BFS Visited Out"),
            size: visited_out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 6. Double-buffered staging buffers
        let staging_dist_bufs: [wgpu::Buffer; 2] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Persistent BFS Staging Dist {i}")),
                size: dist_bytes.max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let staging_visited_bufs: [wgpu::Buffer; 2] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Persistent BFS Staging Visited {i}")),
                size: visited_out_bytes.max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // 6. Overflow status buffer (1 u32 per source in chunk)
        let overflow_bytes = ((max_sources_per_chunk * std::mem::size_of::<u32>()) as u64).max(4);
        let overflow_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent BFS Overflow Buffer"),
            size: overflow_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_overflow_bufs: [wgpu::Buffer; 2] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Persistent BFS Staging Overflow {i}")),
                size: overflow_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // 7. Bind group matching workgroup_persistent_bfs.wgsl layout:
        // 0: params
        // 1: sources
        // 2: row_ptr
        // 3: col_ind
        // 4: packed_dist
        // 5: vram_visited
        // 6: visited_out_buf
        // 7: overflow_buf
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Persistent BFS Traversal BG"),
            layout: pipelines.workgroup_persistent_bfs_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sources_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: packed_dist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: vram_visited_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: visited_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: overflow_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            num_vertices,
            num_edges,
            max_sources_per_chunk,
            words_per_source,
            visited_words_per_source,
            row_offsets,
            col_indices,
            params_buf,
            sources_buf,
            packed_dist_buf,
            vram_visited_buf,
            visited_out_buf,
            overflow_buf,
            staging_dist_bufs,
            staging_visited_bufs,
            staging_overflow_bufs,
            bind_group,
        }
    }
}

/// Executes Multi-Source BFS using Workgroup Persistent BFS.
pub fn run_workgroup_persistent_bfs(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>> {
    run_workgroup_persistent_bfs_pooled(ctx, pipelines, graph, sources, None)
}

/// Executes Multi-Source BFS with persistent buffer pooling.
pub fn run_workgroup_persistent_bfs_pooled(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    pool: Option<&mut Option<PersistentWorkgroupBfsBuffers>>,
) -> Result<Vec<u32>> {
    let total = sources.len() * graph.num_vertices;
    if total == 0 {
        return Ok(Vec::new());
    }
    let mut result = vec![u32::MAX; total];
    run_workgroup_persistent_bfs_into(ctx, pipelines, graph, sources, pool, &mut result)?;
    Ok(result)
}

/// Executes Multi-Source BFS directly into a caller-provided destination slice.
pub fn run_workgroup_persistent_bfs_into(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    mut pool: Option<&mut Option<PersistentWorkgroupBfsBuffers>>,
    result: &mut [u32],
) -> Result<()> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    if num_sources == 0 || num_vertices == 0 {
        return Ok(());
    }

    let total = num_sources * num_vertices;
    anyhow::ensure!(
        result.len() >= total,
        "Destination slice length {} is smaller than required {}",
        result.len(),
        total
    );

    // Retrieve or allocate pooled buffers
    let mut local_storage;
    let bufs: &mut PersistentWorkgroupBfsBuffers = match pool {
        Some(ref mut opt) => {
            let needs_alloc = match opt.as_ref() {
                Some(cached) => {
                    cached.num_vertices != num_vertices || cached.num_edges != graph.num_edges
                }
                None => true,
            };
            if needs_alloc {
                **opt = Some(PersistentWorkgroupBfsBuffers::new(ctx, pipelines, graph));
            }
            opt.as_mut().unwrap()
        }
        None => {
            local_storage = Some(PersistentWorkgroupBfsBuffers::new(ctx, pipelines, graph));
            local_storage.as_mut().unwrap()
        }
    };

    // 1. Spatial source sorting: permute indices by source vertex ID
    let mut perm: Vec<usize> = (0..num_sources).collect();
    perm.sort_unstable_by_key(|&i| sources[i]);

    let chunk_size = bufs.max_sources_per_chunk;
    let num_chunks = num_sources.div_ceil(chunk_size);

    if num_chunks <= 1 {
        // Single chunk execution
        let chunk_sources: Vec<u32> = perm.iter().map(|&i| sources[i]).collect();
        execute_persistent_chunk(
            ctx,
            pipelines,
            graph,
            bufs,
            &chunk_sources,
            &perm,
            num_vertices,
            0,
            result,
        )?;
    } else {
        // Double-buffered asynchronous pipelining
        execute_persistent_pipelined(
            ctx,
            pipelines,
            graph,
            bufs,
            sources,
            &perm,
            num_vertices,
            chunk_size,
            result,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_persistent_chunk(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    bufs: &PersistentWorkgroupBfsBuffers,
    chunk_sources: &[u32],
    chunk_perm: &[usize],
    num_vertices: usize,
    staging_idx: usize,
    result: &mut [u32],
) -> Result<()> {
    let device = ctx.device();
    let queue = ctx.queue();
    let active_sources = chunk_sources.len();
    if active_sources == 0 {
        return Ok(());
    }

    let use_sram_visited = if num_vertices <= SRAM_MAX_VERTICES { 1u32 } else { 0u32 };
    let params = WorkgroupPersistentParams {
        num_vertices: num_vertices as u32,
        num_sources: active_sources as u32,
        use_sram_visited,
        reachability_only: 0,
        pad0: 0,
        pad1: 0,
        pad2: 0,
        pad3: 0,
    };

    queue.write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));
    queue.write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(chunk_sources));

    let active_bytes = (active_sources * bufs.words_per_source * 4) as u64;
    let overflow_bytes = (active_sources * std::mem::size_of::<u32>()) as u64;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Persistent BFS Compute Encoder"),
    });

    encoder.clear_buffer(&bufs.packed_dist_buf, 0, Some(active_bytes));
    encoder.clear_buffer(&bufs.overflow_buf, 0, Some(overflow_bytes.max(4)));
    if use_sram_visited == 0 {
        let vram_visited_bytes = (active_sources * bufs.visited_words_per_source * 4) as u64;
        encoder.clear_buffer(&bufs.vram_visited_buf, 0, Some(vram_visited_bytes));
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Persistent BFS Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipelines.workgroup_persistent_bfs_pipeline());
        pass.set_bind_group(0, &bufs.bind_group, &[]);
        pass.dispatch_workgroups(active_sources as u32, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &bufs.packed_dist_buf,
        0,
        &bufs.staging_dist_bufs[staging_idx],
        0,
        active_bytes,
    );
    encoder.copy_buffer_to_buffer(
        &bufs.overflow_buf,
        0,
        &bufs.staging_overflow_bufs[staging_idx],
        0,
        overflow_bytes,
    );

    queue.submit(Some(encoder.finish()));

    // Map staging buffer
    let slice = bufs.staging_dist_bufs[staging_idx].slice(..active_bytes);
    let (tx_dist, rx_dist) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_dist.send(res);
    });

    let ovf_slice = bufs.staging_overflow_bufs[staging_idx].slice(..overflow_bytes);
    let (tx_ovf, rx_ovf) = futures_channel::oneshot::channel();
    ovf_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_ovf.send(res);
    });

    #[cfg(not(target_arch = "wasm32"))]
    {
        device.poll(wgpu::Maintain::Wait);

        pollster::block_on(rx_dist)
            .context("Staging dist buffer receiver dropped")?
            .context("Failed to map staging dist buffer")?;
        pollster::block_on(rx_ovf)
            .context("Staging overflow buffer receiver dropped")?
            .context("Failed to map staging overflow buffer")?;

        {
            let view = slice.get_mapped_range();
            let packed_u32: &[u32] = bytemuck::cast_slice(&view);
            let packed_u16: &[u16] = bytemuck::cast_slice(packed_u32);

            unpack_chunk_distances(packed_u16, chunk_perm, num_vertices, bufs.words_per_source, result);

            drop(view);
            bufs.staging_dist_bufs[staging_idx].unmap();
        }

        let mut overflowed_indices = Vec::new();
        {
            let ovf_view = ovf_slice.get_mapped_range();
            let ovf_data: &[u32] = bytemuck::cast_slice(&ovf_view);
            for (k, &flag) in ovf_data[..active_sources].iter().enumerate() {
                if flag != 0 {
                    overflowed_indices.push(k);
                }
            }
            drop(ovf_view);
            bufs.staging_overflow_bufs[staging_idx].unmap();
        }

        if !overflowed_indices.is_empty() {
            let fallback_sources: Vec<u32> = overflowed_indices.iter().map(|&k| chunk_sources[k]).collect();
            let mut fallback_dist = vec![u32::MAX; fallback_sources.len() * num_vertices];
            crate::dispatch::multi_word_bp::run_multi_word_bp_into(
                ctx,
                pipelines,
                graph,
                &fallback_sources,
                None,
                &mut fallback_dist,
            )?;

            for (fb_idx, &k) in overflowed_indices.iter().enumerate() {
                let orig_idx = chunk_perm[k];
                let fb_slice = &fallback_dist[fb_idx * num_vertices..(fb_idx + 1) * num_vertices];
                let dst_slice = &mut result[orig_idx * num_vertices..(orig_idx + 1) * num_vertices];
                dst_slice.copy_from_slice(fb_slice);
            }
        }

        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = (rx_dist, rx_ovf, slice, ovf_slice, chunk_perm, result, graph);
        anyhow::bail!("Synchronous buffer mapping is not supported on wasm32; use run_workgroup_persistent_bfs_async instead");
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_persistent_pipelined(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    bufs: &PersistentWorkgroupBfsBuffers,
    sources: &[u32],
    perm: &[usize],
    num_vertices: usize,
    chunk_size: usize,
    result: &mut [u32],
) -> Result<()> {
    let chunks: Vec<&[usize]> = perm.chunks(chunk_size).collect();
    #[cfg(not(target_arch = "wasm32"))]
    let num_chunks = chunks.len();

    #[cfg(target_arch = "wasm32")]
    {
        for (chunk_idx, &chunk_perm) in chunks.iter().enumerate() {
            let chunk_sources: Vec<u32> = chunk_perm.iter().map(|&i| sources[i]).collect();
            let staging_idx = chunk_idx % 2;
            execute_persistent_chunk(
                ctx,
                pipelines,
                graph,
                bufs,
                &chunk_sources,
                chunk_perm,
                num_vertices,
                staging_idx,
                result,
            )?;
        }
        return Ok(());
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        struct PendingGpuMap {
            chunk_perm: Vec<usize>,
            chunk_sources: Vec<u32>,
            active_bytes: u64,
            overflow_bytes: u64,
            staging_idx: usize,
            rx_dist: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
            rx_ovf: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
        }

        struct PendingWorker<'scope> {
            handle: std::thread::ScopedJoinHandle<'scope, Result<wgpu::BufferView<'scope>>>,
            staging_idx: usize,
            overflowed_indices: Vec<usize>,
            chunk_sources: Vec<u32>,
            chunk_perm: Vec<usize>,
        }

        let total_elements = sources.len() * num_vertices;
        let result_ptr = result.as_mut_ptr() as usize;

        // Submits GPU traversal and DMA copies for a chunk, initiating map_async
        let submit_gpu_chunk = |chunk_perm: &[usize], staging_idx: usize| -> Result<PendingGpuMap> {
            let chunk_sources: Vec<u32> = chunk_perm.iter().map(|&i| sources[i]).collect();
            let active_sources = chunk_sources.len();
            let active_bytes = (active_sources * bufs.words_per_source * 4) as u64;
            let overflow_bytes = (active_sources * std::mem::size_of::<u32>()) as u64;

            let use_sram_visited = if num_vertices <= SRAM_MAX_VERTICES { 1u32 } else { 0u32 };
            let params = WorkgroupPersistentParams {
                num_vertices: num_vertices as u32,
                num_sources: active_sources as u32,
                use_sram_visited,
                reachability_only: 0,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                pad3: 0,
            };

            ctx.queue().write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));
            ctx.queue().write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(&chunk_sources));

            let mut encoder = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Persistent BFS Compute Encoder (Pipelined)"),
            });

            encoder.clear_buffer(&bufs.packed_dist_buf, 0, Some(active_bytes));
            encoder.clear_buffer(&bufs.overflow_buf, 0, Some(overflow_bytes.max(4)));
            if use_sram_visited == 0 {
                let vram_visited_bytes = (active_sources * bufs.visited_words_per_source * 4) as u64;
                encoder.clear_buffer(&bufs.vram_visited_buf, 0, Some(vram_visited_bytes));
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Persistent BFS Compute Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.workgroup_persistent_bfs_pipeline());
                pass.set_bind_group(0, &bufs.bind_group, &[]);
                pass.dispatch_workgroups(active_sources as u32, 1, 1);
            }

            encoder.copy_buffer_to_buffer(
                &bufs.packed_dist_buf,
                0,
                &bufs.staging_dist_bufs[staging_idx],
                0,
                active_bytes,
            );
            encoder.copy_buffer_to_buffer(
                &bufs.overflow_buf,
                0,
                &bufs.staging_overflow_bufs[staging_idx],
                0,
                overflow_bytes,
            );

            ctx.queue().submit(Some(encoder.finish()));

            let slice = bufs.staging_dist_bufs[staging_idx].slice(..active_bytes);
            let (tx_dist, rx_dist) = futures_channel::oneshot::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx_dist.send(res);
            });

            let ovf_slice = bufs.staging_overflow_bufs[staging_idx].slice(..overflow_bytes);
            let (tx_ovf, rx_ovf) = futures_channel::oneshot::channel();
            ovf_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx_ovf.send(res);
            });

            Ok(PendingGpuMap {
                chunk_perm: chunk_perm.to_vec(),
                chunk_sources,
                active_bytes,
                overflow_bytes,
                staging_idx,
                rx_dist,
                rx_ovf,
            })
        };

        std::thread::scope(|s| -> Result<()> {
            if num_chunks == 0 {
                return Ok(());
            }

            // 1. Prologue: Submit Chunk 0 GPU pass and initiate map_async
            let mut pending_gpu = submit_gpu_chunk(chunks[0], 0)?;
            let mut pending_worker: Option<PendingWorker> = None;

            // 2. Double-buffered overlapping loop: Over Chunk 1..num_chunks
            for chunk_idx in 1..num_chunks {
                let staging_idx = chunk_idx % 2;

                // A. Join previous unpack worker using staging_idx (from chunk_idx - 2) before reusing staging buffer
                if let Some(worker) = pending_worker.take() {
                    let view = worker
                        .handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("CPU unpack worker panicked"))??;
                    drop(view);
                    bufs.staging_dist_bufs[worker.staging_idx].unmap();

                    if !worker.overflowed_indices.is_empty() {
                        let fallback_sources: Vec<u32> = worker.overflowed_indices.iter().map(|&k| worker.chunk_sources[k]).collect();
                        let mut fallback_dist = vec![u32::MAX; fallback_sources.len() * num_vertices];
                        crate::dispatch::multi_word_bp::run_multi_word_bp_into(
                            ctx,
                            pipelines,
                            graph,
                            &fallback_sources,
                            None,
                            &mut fallback_dist,
                        )?;
                        let res_slice = unsafe {
                            std::slice::from_raw_parts_mut(result_ptr as *mut u32, total_elements)
                        };
                        for (fb_idx, &k) in worker.overflowed_indices.iter().enumerate() {
                            let orig_idx = worker.chunk_perm[k];
                            let fb_slice = &fallback_dist[fb_idx * num_vertices..(fb_idx + 1) * num_vertices];
                            let dst_slice = &mut res_slice[orig_idx * num_vertices..(orig_idx + 1) * num_vertices];
                            dst_slice.copy_from_slice(fb_slice);
                        }
                    }
                }

                // B. Submit chunk_idx GPU traversal and DMA copy to staging_dist_bufs[staging_idx]
                let next_gpu = submit_gpu_chunk(chunks[chunk_idx], staging_idx)?;

                // C. Concurrently, while chunk_idx is executing on the GPU:
                // Await chunk_idx - 1 mapping, get mapped range, and spawn Rayon unpack worker thread
                ctx.device().poll(wgpu::Maintain::Wait);
                pollster::block_on(pending_gpu.rx_dist)
                    .context("Staging dist buffer receiver dropped")?
                    .context("Failed to map staging dist buffer")?;
                pollster::block_on(pending_gpu.rx_ovf)
                    .context("Staging overflow buffer receiver dropped")?
                    .context("Failed to map staging overflow buffer")?;

                let ovf_slice = bufs.staging_overflow_bufs[pending_gpu.staging_idx].slice(..pending_gpu.overflow_bytes);
                let mut overflowed_indices = Vec::new();
                {
                    let ovf_view = ovf_slice.get_mapped_range();
                    let ovf_data: &[u32] = bytemuck::cast_slice(&ovf_view);
                    for (k, &flag) in ovf_data[..pending_gpu.chunk_sources.len()].iter().enumerate() {
                        if flag != 0 {
                            overflowed_indices.push(k);
                        }
                    }
                    drop(ovf_view);
                    bufs.staging_overflow_bufs[pending_gpu.staging_idx].unmap();
                }

                let slice = bufs.staging_dist_bufs[pending_gpu.staging_idx].slice(..pending_gpu.active_bytes);
                let view = slice.get_mapped_range();

                let words_per_source = bufs.words_per_source;
                let chunk_perm_for_worker = pending_gpu.chunk_perm.clone();

                let handle = s.spawn(move || -> Result<wgpu::BufferView<'_>> {
                    let packed_u32: &[u32] = bytemuck::cast_slice(&view);
                    let packed_u16: &[u16] = bytemuck::cast_slice(packed_u32);
                    let res_slice = unsafe {
                        std::slice::from_raw_parts_mut(result_ptr as *mut u32, total_elements)
                    };
                    unpack_chunk_distances(
                        packed_u16,
                        &chunk_perm_for_worker,
                        num_vertices,
                        words_per_source,
                        res_slice,
                    );
                    Ok(view)
                });

                pending_worker = Some(PendingWorker {
                    handle,
                    staging_idx: pending_gpu.staging_idx,
                    overflowed_indices,
                    chunk_sources: pending_gpu.chunk_sources,
                    chunk_perm: pending_gpu.chunk_perm,
                });

                pending_gpu = next_gpu;
            }

            // 3. Epilogue: Drain remaining worker thread (Chunk num_chunks - 2)
            if let Some(worker) = pending_worker.take() {
                let view = worker
                    .handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("CPU unpack worker panicked"))??;
                drop(view);
                bufs.staging_dist_bufs[worker.staging_idx].unmap();

                if !worker.overflowed_indices.is_empty() {
                    let fallback_sources: Vec<u32> = worker.overflowed_indices.iter().map(|&k| worker.chunk_sources[k]).collect();
                    let mut fallback_dist = vec![u32::MAX; fallback_sources.len() * num_vertices];
                    crate::dispatch::multi_word_bp::run_multi_word_bp_into(
                        ctx,
                        pipelines,
                        graph,
                        &fallback_sources,
                        None,
                        &mut fallback_dist,
                    )?;
                    let res_slice = unsafe {
                        std::slice::from_raw_parts_mut(result_ptr as *mut u32, total_elements)
                    };
                    for (fb_idx, &k) in worker.overflowed_indices.iter().enumerate() {
                        let orig_idx = worker.chunk_perm[k];
                        let fb_slice = &fallback_dist[fb_idx * num_vertices..(fb_idx + 1) * num_vertices];
                        let dst_slice = &mut res_slice[orig_idx * num_vertices..(orig_idx + 1) * num_vertices];
                        dst_slice.copy_from_slice(fb_slice);
                    }
                }
            }

            // 4. Drain final chunk (Chunk num_chunks - 1)
            ctx.device().poll(wgpu::Maintain::Wait);
            pollster::block_on(pending_gpu.rx_dist)
                .context("Staging dist buffer receiver dropped")?
                .context("Failed to map staging dist buffer")?;
            pollster::block_on(pending_gpu.rx_ovf)
                .context("Staging overflow buffer receiver dropped")?
                .context("Failed to map staging overflow buffer")?;

            let ovf_slice = bufs.staging_overflow_bufs[pending_gpu.staging_idx].slice(..pending_gpu.overflow_bytes);
            let mut overflowed_indices = Vec::new();
            {
                let ovf_view = ovf_slice.get_mapped_range();
                let ovf_data: &[u32] = bytemuck::cast_slice(&ovf_view);
                for (k, &flag) in ovf_data[..pending_gpu.chunk_sources.len()].iter().enumerate() {
                    if flag != 0 {
                        overflowed_indices.push(k);
                    }
                }
                drop(ovf_view);
                bufs.staging_overflow_bufs[pending_gpu.staging_idx].unmap();
            }

            let slice = bufs.staging_dist_bufs[pending_gpu.staging_idx].slice(..pending_gpu.active_bytes);
            let view = slice.get_mapped_range();
            let packed_u32: &[u32] = bytemuck::cast_slice(&view);
            let packed_u16: &[u16] = bytemuck::cast_slice(packed_u32);

            let res_slice = unsafe {
                std::slice::from_raw_parts_mut(result_ptr as *mut u32, total_elements)
            };
            unpack_chunk_distances(
                packed_u16,
                &pending_gpu.chunk_perm,
                num_vertices,
                bufs.words_per_source,
                res_slice,
            );

            drop(view);
            bufs.staging_dist_bufs[pending_gpu.staging_idx].unmap();

            if !overflowed_indices.is_empty() {
                let fallback_sources: Vec<u32> = overflowed_indices.iter().map(|&k| pending_gpu.chunk_sources[k]).collect();
                let mut fallback_dist = vec![u32::MAX; fallback_sources.len() * num_vertices];
                crate::dispatch::multi_word_bp::run_multi_word_bp_into(
                    ctx,
                    pipelines,
                    graph,
                    &fallback_sources,
                    None,
                    &mut fallback_dist,
                )?;
                for (fb_idx, &k) in overflowed_indices.iter().enumerate() {
                    let orig_idx = pending_gpu.chunk_perm[k];
                    let fb_slice = &fallback_dist[fb_idx * num_vertices..(fb_idx + 1) * num_vertices];
                    let dst_slice = &mut res_slice[orig_idx * num_vertices..(orig_idx + 1) * num_vertices];
                    dst_slice.copy_from_slice(fb_slice);
                }
            }

            Ok(())
        })?;

        Ok(())
    }
}

/// Unpacks 16-bit half-words into 32-bit distance values directly into original source slots.
pub fn unpack_chunk_distances(
    u16_data: &[u16],
    chunk_perm: &[usize],
    num_vertices: usize,
    words_per_source: usize,
    result: &mut [u32],
) {
    let u16_per_source = words_per_source * 2;

    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        let result_ptr = result.as_mut_ptr() as usize;

        chunk_perm.par_iter().enumerate().for_each(|(k, &orig_idx)| {
            let src_u16 = &u16_data[k * u16_per_source..k * u16_per_source + num_vertices];
            let dst_ptr = (result_ptr + orig_idx * num_vertices * std::mem::size_of::<u32>()) as *mut u32;
            let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst_ptr, num_vertices) };

            for (v, &raw) in src_u16.iter().enumerate() {
                dst_slice[v] = if raw == 0 {
                    u32::MAX
                } else {
                    (0xFFFFu16 - raw) as u32
                };
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (k, &orig_idx) in chunk_perm.iter().enumerate() {
            let src_u16 = &u16_data[k * u16_per_source..k * u16_per_source + num_vertices];
            let dst_slice = &mut result[orig_idx * num_vertices..(orig_idx + 1) * num_vertices];
            for (v, &raw) in src_u16.iter().enumerate() {
                dst_slice[v] = if raw == 0 {
                    u32::MAX
                } else {
                    (0xFFFFu16 - raw) as u32
                };
            }
        }
    }
}

/// Executes Multi-Source Reachability query using Workgroup Persistent BFS.
pub fn run_workgroup_persistent_reachability(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    mut pool: Option<&mut Option<PersistentWorkgroupBfsBuffers>>,
) -> Result<ReachabilityBitmask> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    if num_sources == 0 || num_vertices == 0 {
        return Ok(ReachabilityBitmask::zeroed(num_vertices, num_sources));
    }

    let mut local_storage;
    let bufs: &mut PersistentWorkgroupBfsBuffers = match pool {
        Some(ref mut opt) => {
            let needs_alloc = match opt.as_ref() {
                Some(cached) => {
                    cached.num_vertices != num_vertices || cached.num_edges != graph.num_edges
                }
                None => true,
            };
            if needs_alloc {
                **opt = Some(PersistentWorkgroupBfsBuffers::new(ctx, pipelines, graph));
            }
            opt.as_mut().unwrap()
        }
        None => {
            local_storage = Some(PersistentWorkgroupBfsBuffers::new(ctx, pipelines, graph));
            local_storage.as_mut().unwrap()
        }
    };

    let mut perm: Vec<usize> = (0..num_sources).collect();
    perm.sort_unstable_by_key(|&i| sources[i]);

    #[allow(unused_mut)]
    let mut bitmask = ReachabilityBitmask::zeroed(num_vertices, num_sources);
    let chunk_size = bufs.max_sources_per_chunk;
    let wpv = bufs.visited_words_per_source;

    for (chunk_idx, chunk_perm) in perm.chunks(chunk_size).enumerate() {
        let chunk_sources: Vec<u32> = chunk_perm.iter().map(|&i| sources[i]).collect();
        let active_sources = chunk_sources.len();
        if active_sources == 0 {
            continue;
        }

        let staging_idx = chunk_idx % 2;
        let use_sram_visited = if num_vertices <= SRAM_MAX_VERTICES { 1u32 } else { 0u32 };
        let params = WorkgroupPersistentParams {
            num_vertices: num_vertices as u32,
            num_sources: active_sources as u32,
            use_sram_visited,
            reachability_only: 1, // Reachability mode enabled
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };

        ctx.queue().write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));
        ctx.queue().write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(&chunk_sources));

        let copy_words = active_sources * wpv;
        let copy_bytes = (copy_words * 4) as u64;
        let overflow_bytes = (active_sources * std::mem::size_of::<u32>()) as u64;

        let mut encoder = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Persistent BFS Reachability Encoder"),
        });

        encoder.clear_buffer(&bufs.visited_out_buf, 0, Some(copy_bytes));
        encoder.clear_buffer(&bufs.overflow_buf, 0, Some(overflow_bytes.max(4)));
        if use_sram_visited == 0 {
            let vram_visited_bytes = (active_sources * wpv * 4) as u64;
            encoder.clear_buffer(&bufs.vram_visited_buf, 0, Some(vram_visited_bytes));
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Persistent BFS Reachability Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipelines.workgroup_persistent_bfs_pipeline());
            pass.set_bind_group(0, &bufs.bind_group, &[]);
            pass.dispatch_workgroups(active_sources as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &bufs.visited_out_buf,
            0,
            &bufs.staging_visited_bufs[staging_idx],
            0,
            copy_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &bufs.overflow_buf,
            0,
            &bufs.staging_overflow_bufs[staging_idx],
            0,
            overflow_bytes,
        );

        ctx.queue().submit(Some(encoder.finish()));

        let slice = bufs.staging_visited_bufs[staging_idx].slice(..copy_bytes);
        let (tx_vis, rx_vis) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_vis.send(res);
        });

        let ovf_slice = bufs.staging_overflow_bufs[staging_idx].slice(..overflow_bytes);
        let (tx_ovf, rx_ovf) = futures_channel::oneshot::channel();
        ovf_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_ovf.send(res);
        });

        #[cfg(not(target_arch = "wasm32"))]
        {
            ctx.device().poll(wgpu::Maintain::Wait);

            pollster::block_on(rx_vis)
                .context("Staging visited buffer receiver dropped")?
                .context("Failed to map staging visited buffer")?;
            pollster::block_on(rx_ovf)
                .context("Staging overflow buffer receiver dropped")?
                .context("Failed to map staging overflow buffer")?;

            let view = slice.get_mapped_range();
            let visited_data: &[u32] = bytemuck::cast_slice(&view);

            unpack_reachability_scatter(visited_data, chunk_perm, num_vertices, wpv, &mut bitmask);

            drop(view);
            bufs.staging_visited_bufs[staging_idx].unmap();

            let mut overflowed_indices = Vec::new();
            {
                let ovf_view = ovf_slice.get_mapped_range();
                let ovf_data: &[u32] = bytemuck::cast_slice(&ovf_view);
                for (k, &flag) in ovf_data[..active_sources].iter().enumerate() {
                    if flag != 0 {
                        overflowed_indices.push(k);
                    }
                }
                drop(ovf_view);
                bufs.staging_overflow_bufs[staging_idx].unmap();
            }

            if !overflowed_indices.is_empty() {
                let fallback_sources: Vec<u32> = overflowed_indices.iter().map(|&k| chunk_sources[k]).collect();
                let fallback_mask = crate::dispatch::multi_word_bp::run_multi_word_bp_reachability(
                    ctx.as_ref(),
                    pipelines.as_ref(),
                    graph,
                    &fallback_sources,
                )?;

                for (fb_idx, &k) in overflowed_indices.iter().enumerate() {
                    let orig_s = chunk_perm[k];
                    for v in 0..num_vertices {
                        let is_reach = fallback_mask.is_reachable(fb_idx, v);
                        if is_reach {
                            bitmask.set_reachable(orig_s, v);
                        }
                    }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = (rx_vis, rx_ovf, slice, ovf_slice, chunk_perm, graph);
            anyhow::bail!("Synchronous buffer mapping is not supported on wasm32; use run_workgroup_persistent_reachability_async instead");
        }
    }

    Ok(bitmask)
}

fn unpack_reachability_scatter(
    visited_data: &[u32],
    chunk_perm: &[usize],
    _num_vertices: usize,
    words_per_source: usize,
    bitmask: &mut ReachabilityBitmask,
) {
    let wpv = bitmask.words_per_vertex;

    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        bitmask.words
            .par_chunks_mut(wpv)
            .enumerate()
            .for_each(|(v, v_words)| {
                let v_word_idx = v / 32;
                let v_bit = 1u32 << (v % 32);

                for (k, &orig_idx) in chunk_perm.iter().enumerate() {
                    let s_word = visited_data[k * words_per_source + v_word_idx];
                    if (s_word & v_bit) != 0 {
                        v_words[orig_idx / 32] |= 1u32 << (orig_idx % 32);
                    }
                }
            });
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (v, v_words) in bitmask.words.chunks_mut(wpv).enumerate() {
            let v_word_idx = v / 32;
            let v_bit = 1u32 << (v % 32);

            for (k, &orig_idx) in chunk_perm.iter().enumerate() {
                let s_word = visited_data[k * words_per_source + v_word_idx];
                if (s_word & v_bit) != 0 {
                    v_words[orig_idx / 32] |= 1u32 << (orig_idx % 32);
                }
            }
        }
    }
}

/// Asynchronously executes a single chunk traversal on WebGPU (compatible with wasm32 and native).
pub async fn execute_persistent_chunk_async(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    bufs: &PersistentWorkgroupBfsBuffers,
    chunk_sources: &[u32],
    chunk_perm: &[usize],
    num_vertices: usize,
    staging_idx: usize,
    result: &mut [u32],
) -> Result<()> {
    let active_sources = chunk_sources.len();
    if active_sources == 0 {
        return Ok(());
    }

    let use_sram_visited = if num_vertices <= SRAM_MAX_VERTICES { 1u32 } else { 0u32 };
    let params = WorkgroupPersistentParams {
        num_vertices: num_vertices as u32,
        num_sources: active_sources as u32,
        use_sram_visited,
        reachability_only: 0,
        pad0: 0,
        pad1: 0,
        pad2: 0,
        pad3: 0,
    };

    ctx.queue().write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));
    ctx.queue().write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(chunk_sources));

    let active_bytes = (active_sources * bufs.words_per_source * 4) as u64;
    let overflow_bytes = (active_sources * std::mem::size_of::<u32>()) as u64;

    let mut encoder = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Persistent BFS Compute Encoder (Async)"),
    });

    encoder.clear_buffer(&bufs.packed_dist_buf, 0, Some(active_bytes));
    encoder.clear_buffer(&bufs.overflow_buf, 0, Some(overflow_bytes.max(4)));
    if use_sram_visited == 0 {
        let vram_visited_bytes = (active_sources * bufs.visited_words_per_source * 4) as u64;
        encoder.clear_buffer(&bufs.vram_visited_buf, 0, Some(vram_visited_bytes));
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Persistent BFS Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipelines.workgroup_persistent_bfs_pipeline());
        pass.set_bind_group(0, &bufs.bind_group, &[]);
        pass.dispatch_workgroups(active_sources as u32, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &bufs.packed_dist_buf,
        0,
        &bufs.staging_dist_bufs[staging_idx],
        0,
        active_bytes,
    );
    encoder.copy_buffer_to_buffer(
        &bufs.overflow_buf,
        0,
        &bufs.staging_overflow_bufs[staging_idx],
        0,
        overflow_bytes,
    );

    ctx.queue().submit(Some(encoder.finish()));

    let slice = bufs.staging_dist_bufs[staging_idx].slice(..active_bytes);
    let (tx_dist, rx_dist) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_dist.send(res);
    });

    let ovf_slice = bufs.staging_overflow_bufs[staging_idx].slice(..overflow_bytes);
    let (tx_ovf, rx_ovf) = futures_channel::oneshot::channel();
    ovf_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_ovf.send(res);
    });

    #[cfg(not(target_arch = "wasm32"))]
    ctx.device().poll(wgpu::Maintain::Wait);

    rx_dist
        .await
        .context("Staging dist buffer receiver dropped")?
        .context("Failed to map staging dist buffer")?;
    rx_ovf
        .await
        .context("Staging overflow buffer receiver dropped")?
        .context("Failed to map staging overflow buffer")?;

    {
        let view = slice.get_mapped_range();
        let packed_u32: &[u32] = bytemuck::cast_slice(&view);
        let packed_u16: &[u16] = bytemuck::cast_slice(packed_u32);

        unpack_chunk_distances(packed_u16, chunk_perm, num_vertices, bufs.words_per_source, result);

        drop(view);
        bufs.staging_dist_bufs[staging_idx].unmap();
    }

    let mut overflowed_indices = Vec::new();
    {
        let ovf_view = ovf_slice.get_mapped_range();
        let ovf_data: &[u32] = bytemuck::cast_slice(&ovf_view);
        for (k, &flag) in ovf_data[..active_sources].iter().enumerate() {
            if flag != 0 {
                overflowed_indices.push(k);
            }
        }
        drop(ovf_view);
        bufs.staging_overflow_bufs[staging_idx].unmap();
    }

    if !overflowed_indices.is_empty() {
        let fallback_sources: Vec<u32> = overflowed_indices.iter().map(|&k| chunk_sources[k]).collect();
        let mut fallback_dist = vec![u32::MAX; fallback_sources.len() * num_vertices];
        crate::dispatch::multi_word_bp::run_multi_word_bp_into(
            ctx,
            pipelines,
            graph,
            &fallback_sources,
            None,
            &mut fallback_dist,
        )?;

        for (fb_idx, &k) in overflowed_indices.iter().enumerate() {
            let orig_idx = chunk_perm[k];
            let fb_slice = &fallback_dist[fb_idx * num_vertices..(fb_idx + 1) * num_vertices];
            let dst_slice = &mut result[orig_idx * num_vertices..(orig_idx + 1) * num_vertices];
            dst_slice.copy_from_slice(fb_slice);
        }
    }

    Ok(())
}

/// Asynchronously runs Multi-Source BFS using Workgroup Persistent BFS.
pub async fn run_workgroup_persistent_bfs_async(
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

    let mut result = vec![u32::MAX; num_sources * num_vertices];
    let bufs = PersistentWorkgroupBfsBuffers::new(ctx, pipelines, graph);

    let mut perm: Vec<usize> = (0..num_sources).collect();
    perm.sort_unstable_by_key(|&i| sources[i]);

    let chunk_size = bufs.max_sources_per_chunk;
    for (chunk_idx, chunk_perm) in perm.chunks(chunk_size).enumerate() {
        let chunk_sources: Vec<u32> = chunk_perm.iter().map(|&i| sources[i]).collect();
        let staging_idx = chunk_idx % 2;
        execute_persistent_chunk_async(
            ctx,
            pipelines,
            graph,
            &bufs,
            &chunk_sources,
            chunk_perm,
            num_vertices,
            staging_idx,
            &mut result,
        )
        .await?;
    }

    Ok(result)
}

/// Asynchronously runs Multi-Source Reachability query using Workgroup Persistent BFS.
pub async fn run_workgroup_persistent_reachability_async(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<ReachabilityBitmask> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();
    if num_sources == 0 || num_vertices == 0 {
        return Ok(ReachabilityBitmask::zeroed(num_vertices, num_sources));
    }

    let bufs = PersistentWorkgroupBfsBuffers::new(ctx, pipelines, graph);
    let mut perm: Vec<usize> = (0..num_sources).collect();
    perm.sort_unstable_by_key(|&i| sources[i]);

    let mut bitmask = ReachabilityBitmask::zeroed(num_vertices, num_sources);
    let chunk_size = bufs.max_sources_per_chunk;
    let wpv = bufs.visited_words_per_source;

    for (chunk_idx, chunk_perm) in perm.chunks(chunk_size).enumerate() {
        let chunk_sources: Vec<u32> = chunk_perm.iter().map(|&i| sources[i]).collect();
        let active_sources = chunk_sources.len();
        if active_sources == 0 {
            continue;
        }

        let staging_idx = chunk_idx % 2;
        let use_sram_visited = if num_vertices <= SRAM_MAX_VERTICES { 1u32 } else { 0u32 };
        let params = WorkgroupPersistentParams {
            num_vertices: num_vertices as u32,
            num_sources: active_sources as u32,
            use_sram_visited,
            reachability_only: 1,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };

        ctx.queue().write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));
        ctx.queue().write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(&chunk_sources));

        let copy_words = active_sources * wpv;
        let copy_bytes = (copy_words * 4) as u64;
        let overflow_bytes = (active_sources * std::mem::size_of::<u32>()) as u64;

        let mut encoder = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Persistent BFS Reachability Encoder (Async)"),
        });

        encoder.clear_buffer(&bufs.visited_out_buf, 0, Some(copy_bytes));
        encoder.clear_buffer(&bufs.overflow_buf, 0, Some(overflow_bytes.max(4)));
        if use_sram_visited == 0 {
            let vram_visited_bytes = (active_sources * wpv * 4) as u64;
            encoder.clear_buffer(&bufs.vram_visited_buf, 0, Some(vram_visited_bytes));
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Persistent BFS Reachability Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipelines.workgroup_persistent_bfs_pipeline());
            pass.set_bind_group(0, &bufs.bind_group, &[]);
            pass.dispatch_workgroups(active_sources as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &bufs.visited_out_buf,
            0,
            &bufs.staging_visited_bufs[staging_idx],
            0,
            copy_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &bufs.overflow_buf,
            0,
            &bufs.staging_overflow_bufs[staging_idx],
            0,
            overflow_bytes,
        );

        ctx.queue().submit(Some(encoder.finish()));

        let slice = bufs.staging_visited_bufs[staging_idx].slice(..copy_bytes);
        let (tx_vis, rx_vis) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_vis.send(res);
        });

        let ovf_slice = bufs.staging_overflow_bufs[staging_idx].slice(..overflow_bytes);
        let (tx_ovf, rx_ovf) = futures_channel::oneshot::channel();
        ovf_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_ovf.send(res);
        });

        #[cfg(not(target_arch = "wasm32"))]
        ctx.device().poll(wgpu::Maintain::Wait);

        rx_vis
            .await
            .context("Staging visited buffer receiver dropped")?
            .context("Failed to map staging visited buffer")?;
        rx_ovf
            .await
            .context("Staging overflow buffer receiver dropped")?
            .context("Failed to map staging overflow buffer")?;

        let view = slice.get_mapped_range();
        let visited_data: &[u32] = bytemuck::cast_slice(&view);

        unpack_reachability_scatter(visited_data, chunk_perm, num_vertices, wpv, &mut bitmask);

        drop(view);
        bufs.staging_visited_bufs[staging_idx].unmap();

        let mut overflowed_indices = Vec::new();
        {
            let ovf_view = ovf_slice.get_mapped_range();
            let ovf_data: &[u32] = bytemuck::cast_slice(&ovf_view);
            for (k, &flag) in ovf_data[..active_sources].iter().enumerate() {
                if flag != 0 {
                    overflowed_indices.push(k);
                }
            }
            drop(ovf_view);
            bufs.staging_overflow_bufs[staging_idx].unmap();
        }

        if !overflowed_indices.is_empty() {
            let fallback_sources: Vec<u32> = overflowed_indices.iter().map(|&k| chunk_sources[k]).collect();
            let fallback_mask = crate::dispatch::multi_word_bp::run_multi_word_bp_reachability(
                ctx.as_ref(),
                pipelines.as_ref(),
                graph,
                &fallback_sources,
            )?;

            for (fb_idx, &k) in overflowed_indices.iter().enumerate() {
                let orig_s = chunk_perm[k];
                for v in 0..num_vertices {
                    let is_reach = fallback_mask.is_reachable(fb_idx, v);
                    if is_reach {
                        bitmask.set_reachable(orig_s, v);
                    }
                }
            }
        }
    }

    Ok(bitmask)
}
