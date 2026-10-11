//! WebGPU context management and asynchronous compute dispatch.

use std::sync::Arc;
use wasm_bindgen::prelude::*;
use webgpu_bfs_core::CsrGraph;
use webgpu_bfs_gpu::{GpuContext, PipelineCache};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, Device};

use crate::{
    now_ms,
    types::{calc_mteps, count_traversed_edges, count_visited_vertices, WasmBfsResult},
    WasmGraph,
};

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
struct BatchedParams {
    num_vertices: u32,
    num_sources: u32,
    current_level: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NaiveParams {
    num_vertices: u32,
    current_level: u32,
    pad0: u32,
    pad1: u32,
}

#[wasm_bindgen]
pub struct WasmGpuContext {
    pub(crate) context: Arc<GpuContext>,
    pub(crate) pipelines: Arc<PipelineCache>,
}

#[wasm_bindgen]
impl WasmGpuContext {
    /// Asynchronously requests a WebGPU adapter, device, and initializes compute pipelines.
    pub async fn create() -> Result<WasmGpuContext, JsValue> {
        let context = GpuContext::new_async()
            .await
            .map_err(|e| JsValue::from_str(&format!("WebGPU initialization failed: {:#}", e)))?;
        let context = Arc::new(context);

        let pipelines = PipelineCache::new(context.device())
            .map_err(|e| JsValue::from_str(&format!("Pipeline creation failed: {:#}", e)))?;
        let pipelines = Arc::new(pipelines);

        Ok(Self { context, pipelines })
    }

    pub fn adapter_name(&self) -> String {
        self.context.info().name.clone()
    }

    pub fn adapter_type(&self) -> String {
        format!("{:?}", self.context.info().device_type)
    }

    pub fn adapter_backend(&self) -> String {
        format!("{:?}", self.context.info().backend)
    }

    pub fn adapter_driver(&self) -> String {
        self.context.info().driver.clone()
    }
}

/// Asynchronously reads back `count` u32 words from a mappable staging buffer.
pub async fn read_buffer_u32_async(
    device: &Device,
    staging_buffer: &Buffer,
    count: usize,
) -> Result<Vec<u32>, JsValue> {
    let byte_size = (count * std::mem::size_of::<u32>()) as u64;
    let slice = staging_buffer.slice(..byte_size);

    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });

    let _ = device;
    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::Maintain::Wait);

    receiver
        .await
        .map_err(|_| JsValue::from_str("Buffer map channel dropped"))?
        .map_err(|e| JsValue::from_str(&format!("Failed to map buffer: {:?}", e)))?;

    let view = slice.get_mapped_range();
    let data: &[u32] = bytemuck::cast_slice(&view);
    let result = data.to_vec();
    drop(view);
    staging_buffer.unmap();

    Ok(result)
}

/// Asynchronously runs GPU Multi-Source BFS.
#[wasm_bindgen]
pub async fn run_gpu_bfs(
    ctx: &WasmGpuContext,
    graph: &WasmGraph,
    sources: &[u32],
    method: Option<String>,
) -> Result<WasmBfsResult, JsValue> {
    let method_str = method.as_deref().unwrap_or("BitParallel64");
    let t0 = now_ms();

    let distances = run_gpu_dispatch_async(ctx, &graph.inner, sources, method_str).await?;
    let elapsed_ms = (now_ms() - t0).max(0.001);

    let traversed_edges = count_traversed_edges(&graph.inner, &distances, sources.len());
    let visited_vertices = count_visited_vertices(&distances);
    let mteps = calc_mteps(traversed_edges, elapsed_ms);

    Ok(WasmBfsResult::new(
        distances,
        graph.inner.num_vertices,
        sources.len(),
        elapsed_ms,
        mteps,
        traversed_edges,
        visited_vertices,
        method_str.to_string(),
    ))
}

pub async fn run_gpu_dispatch_async(
    ctx: &WasmGpuContext,
    graph: &CsrGraph,
    sources: &[u32],
    method: &str,
) -> Result<Vec<u32>, JsValue> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    if num_sources == 0 || num_vertices == 0 {
        return Ok(Vec::new());
    }

    for (i, &s) in sources.iter().enumerate() {
        if s as usize >= num_vertices {
            return Err(JsValue::from_str(&format!(
                "Source {} at index {} exceeds vertex count {}",
                s, i, num_vertices
            )));
        }
    }

    match method {
        "BitParallel32" => async_dispatch_bp32(ctx, graph, sources).await,
        "BitParallel64" => async_dispatch_bp64(ctx, graph, sources).await,
        "BooleanReachability" => async_dispatch_bool_reach(ctx, graph, sources).await,
        "BatchedDenseSpMM" => async_dispatch_batched(ctx, graph, sources).await,
        "NaiveSequential" => async_dispatch_naive(ctx, graph, sources).await,
        "PushPullAdaptive" => async_dispatch_push_pull(ctx, graph, sources).await,
        other => Err(JsValue::from_str(&format!(
            "Unknown GPU BFS method: {}",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// 1. BitParallel32 (async)
// ---------------------------------------------------------------------------
async fn async_dispatch_bp32(
    ctx: &WasmGpuContext,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>, JsValue> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();
    let nv = num_vertices as u32;

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
    let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Active Counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Staging Counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Staging Distances"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let push_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Push Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let filter_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP32 Filter Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let push_layout = ctx.pipelines.bp_push_layout();
    let filter_layout = ctx.pipelines.bp_filter_layout();

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

    let filter_bg_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BP32 Filter BG -> B"),
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
        label: Some("BP32 Filter BG -> A"),
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

    let push_pipeline = ctx.pipelines.bp32_push_pipeline();
    let filter_pipeline = ctx.pipelines.bp32_filter_pipeline();
    let workgroups_x = nv.div_ceil(128);

    let mut result = vec![u32::MAX; num_sources * num_vertices];

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
            init_distance[s_idx * num_vertices + (src as usize)] = 0;
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

        let mut level = 0u32;
        loop {
            let p_push = PushParams {
                num_vertices: nv,
                current_level: level,
                pad0: 0,
                pad1: 0,
            };
            let p_filter = FilterParams {
                num_vertices: nv,
                current_level: level,
                chunk_start: 0,
                pad: 0,
            };
            queue.write_buffer(&push_params_buf, 0, bytemuck::bytes_of(&p_push));
            queue.write_buffer(&filter_params_buf, 0, bytemuck::bytes_of(&p_filter));
            queue.write_buffer(&counter_buf, 0, bytemuck::bytes_of(&0u32));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BP32 Step"),
            });

            // Pass 1: Push Traversal
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BP32 Push Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(push_pipeline);
                let push_bg = if level.is_multiple_of(2) {
                    &push_bg_a
                } else {
                    &push_bg_b
                };
                pass.set_bind_group(0, push_bg, &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            // Pass 2: Filter & Distance Update
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BP32 Filter Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(filter_pipeline);
                let filter_bg = if level.is_multiple_of(2) {
                    &filter_bg_to_b
                } else {
                    &filter_bg_to_a
                };
                pass.set_bind_group(0, filter_bg, &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_counter_buf, 0, 4);
            queue.submit(Some(encoder.finish()));

            let active = read_buffer_u32_async(device, &staging_counter_buf, 1).await?[0];
            if active == 0 {
                break;
            }

            level += 1;
            if level > nv {
                break;
            }
        }

        let batch_elements = batch_size * num_vertices;
        let batch_bytes = (batch_elements * std::mem::size_of::<u32>()) as u64;

        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BP32 Distances Readback"),
        });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist_buf, 0, batch_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_batch = read_buffer_u32_async(device, &staging_dist_buf, batch_elements).await?;
        let start_dst = batch_start * num_vertices;
        result[start_dst..start_dst + batch_elements].copy_from_slice(&dist_batch);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// 2. BitParallel64 (async)
// ---------------------------------------------------------------------------
async fn async_dispatch_bp64(
    ctx: &WasmGpuContext,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>, JsValue> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();
    let nv = num_vertices as u32;

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

    let state_words = 2 * num_vertices;
    let state_bytes = (state_words * std::mem::size_of::<u32>()) as u64;
    let dist_batch_bytes = (64 * num_vertices * std::mem::size_of::<u32>()) as u64;

    let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Frontier A"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Frontier B"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let candidate_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Candidate"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Visited"),
        size: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Distances"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Active Counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Staging Counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Staging Distances"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let push_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Push Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let filter_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BP64 Filter Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let push_layout = ctx.pipelines.bp_push_layout();
    let filter_layout = ctx.pipelines.bp_filter_layout();

    let push_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BP64 Push BG A"),
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
        label: Some("BP64 Push BG B"),
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

    let filter_bg_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BP64 Filter BG -> B"),
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
        label: Some("BP64 Filter BG -> A"),
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

    let push_pipeline = ctx.pipelines.bp64_push_pipeline();
    let filter_pipeline = ctx.pipelines.bp64_filter_pipeline();
    let workgroups_x = nv.div_ceil(128);

    let mut result = vec![u32::MAX; num_sources * num_vertices];

    for (batch_idx, batch_sources) in sources.chunks(64).enumerate() {
        let batch_size = batch_sources.len();
        let batch_start = batch_idx * 64;

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
        queue.write_buffer(
            &frontier_buf_b,
            0,
            bytemuck::cast_slice(&vec![0u32; state_words]),
        );
        queue.write_buffer(&visited_buf, 0, bytemuck::cast_slice(&init_visited));
        queue.write_buffer(&distance_buf, 0, bytemuck::cast_slice(&init_distance));
        queue.write_buffer(
            &candidate_buf,
            0,
            bytemuck::cast_slice(&vec![0u32; state_words]),
        );

        let mut level = 0u32;
        loop {
            let p_push = PushParams {
                num_vertices: nv,
                current_level: level,
                pad0: 0,
                pad1: 0,
            };
            let p_filter = FilterParams {
                num_vertices: nv,
                current_level: level,
                chunk_start: 0,
                pad: 0,
            };
            queue.write_buffer(&push_params_buf, 0, bytemuck::bytes_of(&p_push));
            queue.write_buffer(&filter_params_buf, 0, bytemuck::bytes_of(&p_filter));
            queue.write_buffer(&counter_buf, 0, bytemuck::bytes_of(&0u32));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BP64 Step"),
            });

            // Pass 1: Push Traversal
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BP64 Push Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(push_pipeline);
                let push_bg = if level.is_multiple_of(2) {
                    &push_bg_a
                } else {
                    &push_bg_b
                };
                pass.set_bind_group(0, push_bg, &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            // Pass 2: Filter & Distance Update
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BP64 Filter Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(filter_pipeline);
                let filter_bg = if level.is_multiple_of(2) {
                    &filter_bg_to_b
                } else {
                    &filter_bg_to_a
                };
                pass.set_bind_group(0, filter_bg, &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_counter_buf, 0, 4);
            queue.submit(Some(encoder.finish()));

            let active = read_buffer_u32_async(device, &staging_counter_buf, 1).await?[0];
            if active == 0 {
                break;
            }

            level += 1;
            if level > nv {
                break;
            }
        }

        let batch_elements = batch_size * num_vertices;
        let batch_bytes = (batch_elements * std::mem::size_of::<u32>()) as u64;

        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BP64 Distances Readback"),
        });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist_buf, 0, batch_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_batch = read_buffer_u32_async(device, &staging_dist_buf, batch_elements).await?;
        let start_dst = batch_start * num_vertices;
        result[start_dst..start_dst + batch_elements].copy_from_slice(&dist_batch);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// 3. BooleanReachability (async)
// ---------------------------------------------------------------------------
async fn async_dispatch_bool_reach(
    ctx: &WasmGpuContext,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>, JsValue> {
    // BooleanReachability uses the 32-bit parallel frontier with boolean reachability filter
    async_dispatch_bp32(ctx, graph, sources).await
}

// ---------------------------------------------------------------------------
// 4. BatchedDenseSpMM (async)
// ---------------------------------------------------------------------------
async fn async_dispatch_batched(
    ctx: &WasmGpuContext,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>, JsValue> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();
    let nv = num_vertices as u32;
    let ns = num_sources as u32;
    let total_elements = num_sources * num_vertices;
    let matrix_bytes = (total_elements * std::mem::size_of::<u32>()) as u64;

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

    let mut initial_dist = vec![u32::MAX; total_elements];
    let mut initial_frontier = vec![0u32; total_elements];
    for (s_idx, &src) in sources.iter().enumerate() {
        initial_dist[s_idx * num_vertices + (src as usize)] = 0;
        initial_frontier[s_idx * num_vertices + (src as usize)] = 1;
    }

    let dist_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Batched Distance Buffer"),
        contents: bytemuck::cast_slice(&initial_dist),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let frontier_buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Batched Frontier A"),
        contents: bytemuck::cast_slice(&initial_frontier),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Batched Frontier B"),
        size: matrix_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Batched Counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Batched Staging Counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Batched Staging Distances"),
        size: matrix_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Batched Params Buffer"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = ctx.pipelines.batched_spmm_layout();
    let bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Batched BG A->B"),
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
                resource: dist_buf.as_entire_binding(),
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

    let bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Batched BG B->A"),
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
                resource: dist_buf.as_entire_binding(),
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

    let workgroups_x = nv.div_ceil(128);
    let workgroups_y = ns;
    let mut level = 0u32;

    loop {
        queue.write_buffer(
            &params_buf,
            0,
            bytemuck::bytes_of(&BatchedParams {
                num_vertices: nv,
                num_sources: ns,
                current_level: level,
                pad: 0,
            }),
        );
        queue.write_buffer(&counter_buf, 0, bytemuck::bytes_of(&0u32));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Batched Step Encoder"),
        });

        let target_buf = if level.is_multiple_of(2) {
            &frontier_buf_b
        } else {
            &frontier_buf_a
        };
        encoder.clear_buffer(target_buf, 0, None);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched SpMM Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(ctx.pipelines.batched_spmm_pipeline());
            let bg = if level.is_multiple_of(2) {
                &bg_a_to_b
            } else {
                &bg_b_to_a
            };
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_counter_buf, 0, 4);
        queue.submit(Some(encoder.finish()));

        let active = read_buffer_u32_async(device, &staging_counter_buf, 1).await?[0];
        if active == 0 {
            break;
        }

        level += 1;
        if level > nv {
            break;
        }
    }

    let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Batched Read Encoder"),
    });
    read_enc.copy_buffer_to_buffer(&dist_buf, 0, &staging_dist_buf, 0, matrix_bytes);
    queue.submit(Some(read_enc.finish()));

    read_buffer_u32_async(device, &staging_dist_buf, total_elements).await
}

// ---------------------------------------------------------------------------
// 5. NaiveSequential (async)
// ---------------------------------------------------------------------------
async fn async_dispatch_naive(
    ctx: &WasmGpuContext,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>, JsValue> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let num_vertices = graph.num_vertices;
    let nv = num_vertices as u32;
    let v_bytes = (num_vertices * std::mem::size_of::<u32>()) as u64;

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

    let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Naive Frontier A"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Naive Frontier B"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Naive Distance Buffer"),
        size: v_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Naive Counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Naive Staging Counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Naive Staging Distances"),
        size: v_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Naive Params Buffer"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = ctx.pipelines.naive_spmv_layout();
    let bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                resource: dist_buf.as_entire_binding(),
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

    let bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                resource: dist_buf.as_entire_binding(),
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

    let workgroups_x = nv.div_ceil(128);
    let mut result = vec![u32::MAX; sources.len() * num_vertices];

    for (s_idx, &src) in sources.iter().enumerate() {
        let mut initial_dist = vec![u32::MAX; num_vertices];
        initial_dist[src as usize] = 0;
        queue.write_buffer(&dist_buf, 0, bytemuck::cast_slice(&initial_dist));

        let mut initial_frontier = vec![0u32; num_vertices];
        initial_frontier[src as usize] = 1;
        queue.write_buffer(&frontier_buf_a, 0, bytemuck::cast_slice(&initial_frontier));
        queue.write_buffer(
            &frontier_buf_b,
            0,
            bytemuck::cast_slice(&vec![0u32; num_vertices]),
        );

        let mut level = 0u32;
        loop {
            queue.write_buffer(
                &params_buf,
                0,
                bytemuck::bytes_of(&NaiveParams {
                    num_vertices: nv,
                    current_level: level,
                    pad0: 0,
                    pad1: 0,
                }),
            );
            queue.write_buffer(&counter_buf, 0, bytemuck::bytes_of(&0u32));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Naive Step Encoder"),
            });

            let target_buf = if level.is_multiple_of(2) {
                &frontier_buf_b
            } else {
                &frontier_buf_a
            };
            encoder.clear_buffer(target_buf, 0, None);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Naive SpMV Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(ctx.pipelines.naive_spmv_pipeline());
                let bg = if level.is_multiple_of(2) {
                    &bg_a_to_b
                } else {
                    &bg_b_to_a
                };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_counter_buf, 0, 4);
            queue.submit(Some(encoder.finish()));

            let active = read_buffer_u32_async(device, &staging_counter_buf, 1).await?[0];
            if active == 0 {
                break;
            }

            level += 1;
            if level > nv {
                break;
            }
        }

        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Naive Read Encoder"),
        });
        read_enc.copy_buffer_to_buffer(&dist_buf, 0, &staging_dist_buf, 0, v_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_src = read_buffer_u32_async(device, &staging_dist_buf, num_vertices).await?;
        let start = s_idx * num_vertices;
        result[start..start + num_vertices].copy_from_slice(&dist_src);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// 6. PushPullAdaptive (async)
// ---------------------------------------------------------------------------
async fn async_dispatch_push_pull(
    ctx: &WasmGpuContext,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>, JsValue> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let num_vertices = graph.num_vertices;
    let num_edges = graph.num_edges;
    let num_sources = sources.len();
    let nv = num_vertices as u32;

    let csr_row_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSR Row Offsets"),
        contents: bytemuck::cast_slice(&graph.row_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dummy_col = [0u32];
    let csr_col_slice: &[u32] = if graph.col_indices.is_empty() {
        &dummy_col
    } else {
        &graph.col_indices
    };
    let csr_col_indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSR Col Indices"),
        contents: bytemuck::cast_slice(csr_col_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // CSC buffers for Pull traversal
    let csc = if graph.is_symmetric() {
        None
    } else {
        Some(graph.to_csc())
    };

    let csc_col_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSC Col Offsets"),
        contents: bytemuck::cast_slice(match &csc {
            Some(c) => &c.col_offsets,
            None => &graph.row_offsets,
        }),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let csc_row_slice = match &csc {
        Some(c) if !c.row_indices.is_empty() => &c.row_indices[..],
        Some(_) => &dummy_col,
        None => csr_col_slice,
    };
    let csc_row_indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSC Row Indices"),
        contents: bytemuck::cast_slice(csc_row_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });

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
    let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Staging Counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Staging Distance"),
        size: dist_batch_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let push_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Push Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pull_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Pull Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let filter_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PP Filter Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let push_layout = ctx.pipelines.push_layout();
    let pull_layout = ctx.pipelines.pull_layout();
    let filter_layout = ctx.pipelines.bp_filter_layout();

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
                resource: csr_row_offsets_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: csr_col_indices_buf.as_entire_binding(),
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
                resource: csr_row_offsets_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: csr_col_indices_buf.as_entire_binding(),
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
                resource: csc_col_offsets_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: csc_row_indices_buf.as_entire_binding(),
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
                resource: csc_col_offsets_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: csc_row_indices_buf.as_entire_binding(),
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

    let push_pipeline = ctx.pipelines.push_pipeline();
    let pull_pipeline = ctx.pipelines.pull_pipeline();
    let filter_pipeline = ctx.pipelines.bp32_filter_pipeline();
    let workgroups_x = nv.div_ceil(128);

    let push_threshold_edges = (num_edges as u64) / 14u64;
    let pull_threshold_vertices = (num_vertices as u64) / 24u64;
    let avg_degree = if num_vertices > 0 {
        num_edges as f64 / num_vertices as f64
    } else {
        0.0
    };

    let mut result = vec![u32::MAX; num_sources * num_vertices];

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
            init_distance[s_idx * num_vertices + (src as usize)] = 0;
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

        let mut is_pull = false;
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
                if !is_pull {
                    pass.set_pipeline(push_pipeline);
                    let bg = if level.is_multiple_of(2) {
                        &push_bg_a
                    } else {
                        &push_bg_b
                    };
                    pass.set_bind_group(0, bg, &[]);
                } else {
                    pass.set_pipeline(pull_pipeline);
                    let bg = if level.is_multiple_of(2) {
                        &pull_bg_a
                    } else {
                        &pull_bg_b
                    };
                    pass.set_bind_group(0, bg, &[]);
                }
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            // Stage 2: Boolean Filter & Distance Update Pass (ALWAYS EXECUTED)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PP Filter Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(filter_pipeline);
                let bg = if level.is_multiple_of(2) {
                    &filter_bg_to_b
                } else {
                    &filter_bg_to_a
                };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_counter_buf, 0, 4);
            queue.submit(Some(encoder.finish()));

            let active_bits = read_buffer_u32_async(device, &staging_counter_buf, 1).await?[0];
            if active_bits == 0 {
                break;
            }

            let n_f = (active_bits as u64).div_ceil(chunk_size as u64);
            let estimated_m_f = (n_f as f64 * avg_degree) as u64;

            if !is_pull {
                if estimated_m_f > push_threshold_edges {
                    is_pull = true;
                }
            } else if n_f < pull_threshold_vertices {
                is_pull = false;
            }

            level += 1;
            if level > nv {
                break;
            }
        }

        let batch_elements = chunk_size * num_vertices;
        let batch_bytes = (batch_elements * std::mem::size_of::<u32>()) as u64;

        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PP Readback Distances"),
        });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist_buf, 0, batch_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_batch = read_buffer_u32_async(device, &staging_dist_buf, batch_elements).await?;
        let start_dst = batch_start * num_vertices;
        result[start_dst..start_dst + batch_elements].copy_from_slice(&dist_batch);
    }

    Ok(result)
}
