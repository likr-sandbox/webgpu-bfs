//! Massive Multi-Word Bit-Parallel GPU BFS dispatcher (Milestones M8 & M19).
//!
//! Milestone M19:
//! - 8 Multi-Bank Distance Storage Buffers (64 sources each, 512 total per pass)
//! - In-GPU u16 quantization via `pack_distances_u32_to_u16.wgsl` and single-encoder DMA readback
//! - 4 macro-chunk passes for S=2048 (reduced 8x from 32 passes)
//! - Persistent buffer pooling (`PersistentPullBuffers512`)
//! - Dedicated fast path for S <= 64 preserving <= 233 ms latency

use anyhow::{Context, Result};
use std::sync::Arc;
use webgpu_bfs_core::{CsrGraph, ReachabilityBitmask};

use crate::{
    buffer::{read_buffer_u32_blocking, PushPullGraphBuffers},
    context::GpuContext,
    dispatch::indirect_sparse_push::{
        calculate_bank_shift_and_mask, calculate_bank_sources, unpack_u16_words_to_u32,
        InitSourcesParams, PackParams, ResetParams,
    },
    pipeline::PipelineCache,
};

/// Maximum steps recorded speculatively per command encoder submission.
const MAX_CHUNK_STEPS: usize = 64;

/// Uniform parameters matching `multi_word_bp_pull.wgsl` and `multi_word_bp_pull_512.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MultiWordBpParams {
    pub num_vertices: u32,
    pub words_per_vertex: u32,
    pub current_level: u32,
    pub total_sources: u32,
    pub dispatch_x: u32,
    pub bank_shift: u32,
    pub bank_mask: u32,
    pub pad0: u32,
}

/// Dynamically selects the word count K in {8, 16, 32, 64} for a given source count.
pub fn select_words_per_vertex(num_sources: usize) -> u32 {
    let raw_k = (num_sources + 31) / 32;
    if raw_k <= 8 {
        8
    } else if raw_k <= 16 {
        16
    } else if raw_k <= 32 {
        32
    } else {
        64
    }
}

/// Determines the maximum sources executable per macro-chunk based on device buffer limits.
pub fn determine_macro_chunk_size(
    device: &wgpu::Device,
    num_vertices: usize,
    total_sources: usize,
) -> usize {
    let limits = device.limits();
    let max_binding = limits.max_storage_buffer_binding_size as u64;
    let max_buffer = limits.max_buffer_size;
    let effective_limit = max_binding.min(max_buffer);

    let bytes_per_source = (num_vertices * std::mem::size_of::<u32>()) as u64;
    if bytes_per_source == 0 {
        return total_sources.max(1);
    }

    let max_s_by_buffer = (effective_limit / bytes_per_source) as usize;
    if max_s_by_buffer >= total_sources {
        return total_sources.max(1);
    }

    let max_allowed = max_s_by_buffer.min(2048);
    if max_allowed >= 2048 {
        2048
    } else if max_allowed >= 1024 {
        1024
    } else if max_allowed >= 512 {
        512
    } else if max_allowed >= 256 {
        256
    } else if max_allowed >= 32 {
        (max_allowed / 32) * 32
    } else {
        max_allowed.max(1)
    }
}

/// Pre-allocated, reusable GPU storage buffers and bind groups for 512-source Pull Traversal (Milestone M19).
pub struct PersistentPullBuffers512 {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub bank_sources: usize,

    pub csc_col_offsets: Arc<wgpu::Buffer>,
    pub csc_row_indices: Arc<wgpu::Buffer>,

    pub frontier_buf_a: wgpu::Buffer,
    pub frontier_buf_b: wgpu::Buffer,
    pub visited_buf: wgpu::Buffer,
    pub staging_visited_buf: wgpu::Buffer,

    pub dist_banks: [wgpu::Buffer; 8],
    pub packed_u16_banks: [wgpu::Buffer; 8],
    pub staging_packed_buf: wgpu::Buffer,

    pub sources_buf: wgpu::Buffer,
    pub reset_params_buf: wgpu::Buffer,
    pub init_sources_params_buf: wgpu::Buffer,
    pub pack_params_buf: wgpu::Buffer,

    pub counter_bufs: Vec<wgpu::Buffer>,
    pub staging_counter_buf: wgpu::Buffer,
    pub params_bufs: Vec<wgpu::Buffer>,

    pub reset_dist_bgs: [wgpu::BindGroup; 8],
    pub init_sources_bg: wgpu::BindGroup,
    pub pack_dist_bgs: [wgpu::BindGroup; 8],
    pub bgs_even: Vec<wgpu::BindGroup>,
    pub bgs_odd: Vec<wgpu::BindGroup>,
}

impl PersistentPullBuffers512 {
    pub fn new(
        ctx: &GpuContext,
        pipelines: &PipelineCache,
        graph: &CsrGraph,
    ) -> Self {
        let device = ctx.device();
        let num_vertices = graph.num_vertices;
        let num_edges = graph.num_edges;

        // 1. Upload graph topology (CSC col_offsets and row_indices for Pull traversal)
        let graph_buffers = PushPullGraphBuffers::from_symmetric_graph(ctx, graph);
        let csc_col_offsets = graph_buffers.csc_col_offsets;
        let csc_row_indices = graph_buffers.csc_row_indices;

        // 2. Traversal state buffers (16 words per vertex = 512 bits)
        let state_words = num_vertices * 16;
        let state_bytes = ((state_words * std::mem::size_of::<u32>()) as u64).max(4);

        let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Frontier A"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Frontier B"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Visited"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Staging Visited"),
            size: state_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Multi-bank distance buffers: 8 banks of scaled sources (obeying 128 MB limit)
        let bank_sources = calculate_bank_sources(num_vertices);
        let bank_words = bank_sources * num_vertices;
        let bank_bytes = ((bank_words * std::mem::size_of::<u32>()) as u64).max(4);
        let dist_banks: [wgpu::Buffer; 8] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pull512 Distance Bank {i}")),
                size: bank_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // 4. In-GPU u16 packed distance banks & consolidated staging buffer
        let packed_words = (bank_words + 1) / 2;
        let packed_bytes = ((packed_words * std::mem::size_of::<u32>()) as u64).max(4);
        let packed_u16_banks: [wgpu::Buffer; 8] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pull512 Packed u16 Bank {i}")),
                size: packed_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let staging_packed_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Staging Packed Distance Buffer"),
            size: 8 * packed_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 5. Source initialization and uniform parameter buffers
        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Sources Buffer"),
            size: (512 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reset_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Reset Params Buffer"),
            size: std::mem::size_of::<ResetParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_sources_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Init Sources Params Buffer"),
            size: std::mem::size_of::<InitSourcesParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pack_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Pack Params Buffer"),
            size: std::mem::size_of::<PackParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 6. Hop counter and parameter buffers
        let mut counter_bufs = Vec::with_capacity(MAX_CHUNK_STEPS);
        for i in 0..MAX_CHUNK_STEPS {
            counter_bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pull512 Counter {i}")),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pull512 Staging Counter Buffer"),
            size: (MAX_CHUNK_STEPS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut params_bufs = Vec::with_capacity(MAX_CHUNK_STEPS);
        let mut bgs_even = Vec::with_capacity(MAX_CHUNK_STEPS);
        let mut bgs_odd = Vec::with_capacity(MAX_CHUNK_STEPS);

        let pull_layout = pipelines.multi_word_bp_pull_512_layout();
        for step in 0..MAX_CHUNK_STEPS {
            let pbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pull512 Params {step}")),
                size: std::mem::size_of::<MultiWordBpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bg_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Pull512 BG Even {step}")),
                layout: pull_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: csc_col_offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: csc_row_indices.as_entire_binding(),
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
                        resource: counter_bufs[step].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: dist_banks[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: dist_banks[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: dist_banks[2].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: dist_banks[3].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: dist_banks[4].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: dist_banks[5].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: dist_banks[6].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: dist_banks[7].as_entire_binding(),
                    },
                ],
            });

            let bg_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Pull512 BG Odd {step}")),
                layout: pull_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: csc_col_offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: csc_row_indices.as_entire_binding(),
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
                        resource: counter_bufs[step].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: dist_banks[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: dist_banks[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: dist_banks[2].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: dist_banks[3].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: dist_banks[4].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: dist_banks[5].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: dist_banks[6].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: dist_banks[7].as_entire_binding(),
                    },
                ],
            });

            params_bufs.push(pbuf);
            bgs_even.push(bg_even);
            bgs_odd.push(bg_odd);
        }

        // 7. Bind groups for reset, init_sources, and pack_distances
        let reset_dist_bgs: [wgpu::BindGroup; 8] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Pull512 Reset Dist BG {i}")),
                layout: pipelines.reset_dist_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: reset_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dist_banks[i].as_entire_binding(),
                    },
                ],
            })
        });

        let init_sources_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pull512 Init Sources BG"),
            layout: pipelines.init_sparse_sources_512_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: init_sources_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sources_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frontier_buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visited_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dist_banks[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dist_banks[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dist_banks[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: dist_banks[3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: dist_banks[4].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: dist_banks[5].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: dist_banks[6].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: dist_banks[7].as_entire_binding(),
                },
            ],
        });

        let pack_dist_bgs: [wgpu::BindGroup; 8] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Pull512 Pack Dist BG {i}")),
                layout: pipelines.pack_distances_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pack_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dist_banks[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: packed_u16_banks[i].as_entire_binding(),
                    },
                ],
            })
        });

        Self {
            num_vertices,
            num_edges,
            bank_sources,
            csc_col_offsets,
            csc_row_indices,
            frontier_buf_a,
            frontier_buf_b,
            visited_buf,
            staging_visited_buf,
            dist_banks,
            packed_u16_banks,
            staging_packed_buf,
            sources_buf,
            reset_params_buf,
            init_sources_params_buf,
            pack_params_buf,
            counter_bufs,
            staging_counter_buf,
            params_bufs,
            reset_dist_bgs,
            init_sources_bg,
            pack_dist_bgs,
            bgs_even,
            bgs_odd,
        }
    }
}

/// Runner struct for Massive Multi-Word Bit-Parallel BFS.
pub struct MultiWordBitParallelBfs;

impl MultiWordBitParallelBfs {
    /// Executes Multi-Source BFS across arbitrary sources up to 2048.
    pub fn run(
        ctx: &Arc<GpuContext>,
        pipelines: &Arc<PipelineCache>,
        graph: &CsrGraph,
        sources: &[u32],
    ) -> Result<Vec<u32>> {
        run_multi_word_bp(ctx, pipelines, graph, sources)
    }
}

/// Executes Multi-Source BFS using the Massive Multi-Word Bit-Parallel Pull pipeline.
pub fn run_multi_word_bp(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>> {
    run_multi_word_bp_pooled(ctx, pipelines, graph, sources, None)
}

/// Executes Multi-Source BFS using Massive Multi-Word Bit-Parallel Pull with persistent buffer pooling.
pub fn run_multi_word_bp_pooled(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    pool: Option<&mut Option<PersistentPullBuffers512>>,
) -> Result<Vec<u32>> {
    let total_elements = sources.len() * graph.num_vertices;
    if total_elements == 0 {
        return Ok(Vec::new());
    }
    let mut result = vec![u32::MAX; total_elements];
    run_multi_word_bp_into(ctx, pipelines, graph, sources, pool, &mut result)?;
    Ok(result)
}

/// Executes Multi-Source BFS using Massive Multi-Word Bit-Parallel Pull directly into caller-provided memory.
pub fn run_multi_word_bp_into(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    mut pool: Option<&mut Option<PersistentPullBuffers512>>,
    result: &mut [u32],
) -> Result<()> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    if num_sources == 0 || num_vertices == 0 {
        return Ok(());
    }

    let total_elements = num_sources * num_vertices;
    anyhow::ensure!(
        result.len() >= total_elements,
        "Output buffer length {} is smaller than required {}",
        result.len(),
        total_elements
    );

    // Fallback: If hardware cannot bind 15 storage buffers, use legacy multi_word_bp
    if ctx.device().limits().max_storage_buffers_per_shader_stage < 15 {
        let res = run_multi_word_bp_legacy(ctx, pipelines, graph, sources)?;
        result[..total_elements].copy_from_slice(&res);
        return Ok(());
    }

    let nv = num_vertices as u32;
    let mut local_storage;
    let bufs: &mut PersistentPullBuffers512 = match pool {
        Some(ref mut opt) => {
            let needs_alloc = match opt.as_ref() {
                Some(cached) => {
                    cached.num_vertices != num_vertices || cached.num_edges != graph.num_edges
                }
                None => true,
            };
            if needs_alloc {
                **opt = Some(PersistentPullBuffers512::new(ctx, pipelines, graph));
            }
            opt.as_mut().unwrap()
        }
        None => {
            local_storage = Some(PersistentPullBuffers512::new(ctx, pipelines, graph));
            local_storage.as_mut().unwrap()
        }
    };

    // Process in 512-source macro chunks (exactly 4 passes for S=2048)
    for (chunk_idx, chunk_sources) in sources.chunks(512).enumerate() {
        let chunk_start = chunk_idx * 512;
        run_multi_word_bp_512_chunk(
            ctx,
            pipelines,
            bufs,
            chunk_sources,
            nv,
            chunk_start,
            result,
        )?;
    }

    Ok(())
}

fn run_multi_word_bp_512_chunk(
    ctx: &GpuContext,
    pipelines: &PipelineCache,
    bufs: &PersistentPullBuffers512,
    chunk_sources: &[u32],
    nv: u32,
    chunk_start: usize,
    result: &mut [u32],
) -> Result<()> {
    let device = ctx.device();
    let queue = ctx.queue();
    let active_sources = chunk_sources.len();
    if active_sources == 0 {
        return Ok(());
    }
    let bank_sources = bufs.bank_sources;
    let num_banks = (active_sources + bank_sources - 1) / bank_sources;

    queue.write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(chunk_sources));

    let bank_words = bank_sources * (nv as usize);
    let total_reset_wgs = ((bank_words + 255) / 256) as u32;
    let (reset_dx, reset_dy) = if total_reset_wgs <= 65535 {
        (total_reset_wgs.max(1), 1u32)
    } else {
        let dx = 256u32;
        let dy = (total_reset_wgs + dx - 1) / dx;
        (dx, dy)
    };

    let reset_p = ResetParams {
        total_elements: if num_banks == 1 {
            (active_sources * (nv as usize)) as u32
        } else {
            bank_words as u32
        },
        dispatch_x: reset_dx,
    };
    queue.write_buffer(&bufs.reset_params_buf, 0, bytemuck::bytes_of(&reset_p));

    let (bank_shift, bank_mask) = calculate_bank_shift_and_mask(nv as usize);
    let init_src_p = InitSourcesParams {
        num_vertices: nv,
        num_sources: active_sources as u32,
        chunk_start: 0,
        pad0: 0x100 | bank_shift,
    };
    queue.write_buffer(&bufs.init_sources_params_buf, 0, bytemuck::bytes_of(&init_src_p));

    let bank_packed_words = ((bank_words + 1) / 2) as u32;
    let bank_packed_bytes = (bank_packed_words as u64) * 4;

    let pack_wgs = ((bank_packed_words + 255) / 256) as u32;
    let (pack_dx, pack_dy) = if pack_wgs <= 65535 {
        (pack_wgs.max(1), 1u32)
    } else {
        let dx = 256u32;
        let dy = (pack_wgs + dx - 1) / dx;
        (dx, dy)
    };

    let pack_p = PackParams {
        total_elements: bank_words as u32,
        total_packed_words: bank_packed_words,
        dispatch_x: pack_dx,
        pad0: 0,
    };
    queue.write_buffer(&bufs.pack_params_buf, 0, bytemuck::bytes_of(&pack_p));

    // Traversal workgroup dispatch geometry: (W_k=16, W_v=8)
    let total_wgs = (nv + 7) / 8;
    let (dispatch_x, dispatch_y) = if total_wgs <= 65535 {
        (total_wgs.max(1), 1u32)
    } else {
        let dx = 256u32;
        let dy = (total_wgs + dx - 1) / dx;
        (dx, dy)
    };

    let compute_pipeline = pipelines.multi_word_bp_pull_512_pipeline();

    let mut level = 0u32;
    let mut cur_chunk_steps = 4usize;

    while level <= nv {
        let chunk_len_steps = cur_chunk_steps
            .min(MAX_CHUNK_STEPS)
            .min((nv - level + 1) as usize);

        for k in 0..chunk_len_steps {
            let p = MultiWordBpParams {
                num_vertices: nv,
                words_per_vertex: 16,
                current_level: level + k as u32,
                total_sources: active_sources as u32,
                dispatch_x,
                bank_shift,
                bank_mask,
                pad0: 1,
            };
            queue.write_buffer(&bufs.params_bufs[k], 0, bytemuck::bytes_of(&p));
        }
        let last_k = chunk_len_steps - 1;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pull512 Hop Batch Encoder"),
        });
        encoder.clear_buffer(&bufs.counter_bufs[last_k], 0, Some(4));

        if level == 0 {
            encoder.clear_buffer(&bufs.frontier_buf_a, 0, None);
            encoder.clear_buffer(&bufs.frontier_buf_b, 0, None);
            encoder.clear_buffer(&bufs.visited_buf, 0, None);

            for b in 0..num_banks {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Pull512 Reset Distance Bank Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.reset_dist_pipeline());
                pass.set_bind_group(0, &bufs.reset_dist_bgs[b], &[]);
                pass.dispatch_workgroups(reset_dx, reset_dy, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Pull512 Init Sources Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.init_sparse_sources_512_pipeline());
                pass.set_bind_group(0, &bufs.init_sources_bg, &[]);
                let init_wgs = ((active_sources as u32 + 255) / 256).max(1);
                pass.dispatch_workgroups(init_wgs, 1, 1);
            }
        }

        for k in 0..chunk_len_steps {
            let cur_step = level + k as u32;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(compute_pipeline);
            let bg = if cur_step % 2 == 0 {
                &bufs.bgs_even[k]
            } else {
                &bufs.bgs_odd[k]
            };
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            &bufs.counter_bufs[last_k],
            0,
            &bufs.staging_counter_buf,
            0,
            4,
        );

        queue.submit(Some(encoder.finish()));

        let count = read_buffer_u32_blocking(device, &bufs.staging_counter_buf, 1)?[0];
        if count == 0 {
            break;
        }

        level += chunk_len_steps as u32;
        cur_chunk_steps = (cur_chunk_steps * 2).min(MAX_CHUNK_STEPS);
    }

    execute_pull_readback_512(
        ctx,
        pipelines,
        bufs,
        num_banks,
        bank_packed_bytes,
        bank_packed_words,
        pack_dx,
        pack_dy,
        chunk_start,
        active_sources,
        nv as usize,
        result,
    )?;

    Ok(())
}

fn execute_pull_readback_512(
    ctx: &GpuContext,
    pipelines: &PipelineCache,
    bufs: &PersistentPullBuffers512,
    num_banks: usize,
    bank_packed_bytes: u64,
    bank_packed_words: u32,
    pack_dx: u32,
    pack_dy: u32,
    chunk_start: usize,
    active_sources: usize,
    nv: usize,
    result: &mut [u32],
) -> Result<()> {
    let device = ctx.device();
    let queue = ctx.queue();

    let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Pull512 Distance Pack & Read Encoder"),
    });

    for b in 0..num_banks {
        {
            let mut pass = read_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Pull512 Pack Distances Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipelines.pack_distances_pipeline());
            pass.set_bind_group(0, &bufs.pack_dist_bgs[b], &[]);
            pass.dispatch_workgroups(pack_dx, pack_dy, 1);
        }

        read_enc.copy_buffer_to_buffer(
            &bufs.packed_u16_banks[b],
            0,
            &bufs.staging_packed_buf,
            (b as u64) * bank_packed_bytes,
            bank_packed_bytes,
        );
    }

    queue.submit(Some(read_enc.finish()));

    let total_packed_bytes = (num_banks as u64) * bank_packed_bytes;
    let slice_packed = bufs.staging_packed_buf.slice(..total_packed_bytes);
    let (d_tx, d_rx) = futures_channel::oneshot::channel();
    slice_packed.map_async(wgpu::MapMode::Read, move |res| {
        let _ = d_tx.send(res);
    });

    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::Maintain::Wait);

    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(d_rx)
        .context("Staging packed dist buffer map receiver dropped")?
        .context("Failed to map staging packed dist buffer")?;

    let packed_view = slice_packed.get_mapped_range();
    let packed_data: &[u32] = bytemuck::cast_slice(&packed_view);

    let start_dst = chunk_start * nv;
    let active_dst = &mut result[start_dst..start_dst + active_sources * nv];
    let words_per_bank = bank_packed_words as usize;

    #[cfg(not(target_arch = "wasm32"))]
    std::thread::scope(|s| {
        for (dst_bank, src_bank) in active_dst
            .chunks_mut(bufs.bank_sources * nv)
            .zip(packed_data[..num_banks * words_per_bank].chunks(words_per_bank))
        {
            s.spawn(move || {
                let needed_words = (dst_bank.len() + 1) / 2;
                unpack_u16_words_to_u32(&src_bank[..needed_words], dst_bank);
            });
        }
    });

    #[cfg(target_arch = "wasm32")]
    for (dst_bank, src_bank) in active_dst
        .chunks_mut(bufs.bank_sources * nv)
        .zip(packed_data[..num_banks * words_per_bank].chunks(words_per_bank))
    {
        let needed_words = (dst_bank.len() + 1) / 2;
        unpack_u16_words_to_u32(&src_bank[..needed_words], dst_bank);
    }

    drop(packed_view);
    bufs.staging_packed_buf.unmap();
    Ok(())
}

/// Fallback legacy multi_word_bp for downlevel hardware with fewer than 15 storage buffers per stage.
fn run_multi_word_bp_legacy(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    let device = ctx.device();
    let queue = ctx.queue();
    let nv = num_vertices as u32;

    let graph_buffers = PushPullGraphBuffers::from_graph(ctx, graph);
    let row_offsets_buf = &graph_buffers.csc_col_offsets;
    let col_indices_buf = &graph_buffers.csc_row_indices;

    let chunk_size = determine_macro_chunk_size(device, num_vertices, num_sources);
    let mut result = vec![u32::MAX; num_sources * num_vertices];

    for (macro_idx, source_chunk) in sources.chunks(chunk_size).enumerate() {
        let chunk_len = source_chunk.len();
        let chunk_start_src = macro_idx * chunk_size;

        let words_k = select_words_per_vertex(chunk_len);
        let state_words = num_vertices * (words_k as usize);
        let state_bytes = (state_words * std::mem::size_of::<u32>()) as u64;
        let dist_chunk_words = chunk_len * num_vertices;
        let dist_chunk_bytes = (dist_chunk_words * std::mem::size_of::<u32>()) as u64;

        let frontier_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Frontier A"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frontier_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Frontier B"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Visited"),
            size: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Distance"),
            size: dist_chunk_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut counter_bufs = Vec::with_capacity(48);
        for i in 0..48 {
            counter_bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("MultiWord Counter {}", i)),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let staging_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Staging Counter"),
            size: (48 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiWord Staging Distance"),
            size: dist_chunk_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_layout = pipelines.multi_word_bp_layout();
        let mut params_bufs = Vec::with_capacity(48);
        let mut bgs_even = Vec::with_capacity(48);
        let mut bgs_odd = Vec::with_capacity(48);

        for step in 0..48 {
            let pbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("MultiWord Params {}", step)),
                size: std::mem::size_of::<MultiWordBpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bg_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MultiWord BG Even {}", step)),
                layout: pipeline_layout,
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
                        resource: counter_bufs[step].as_entire_binding(),
                    },
                ],
            });

            let bg_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MultiWord BG Odd {}", step)),
                layout: pipeline_layout,
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
                        resource: counter_bufs[step].as_entire_binding(),
                    },
                ],
            });

            params_bufs.push(pbuf);
            bgs_even.push(bg_even);
            bgs_odd.push(bg_odd);
        }

        let mut init_frontier = vec![0u32; state_words];
        let mut init_visited = vec![0u32; state_words];
        let mut init_distance = vec![u32::MAX; dist_chunk_words];

        let wk = words_k as usize;
        for (s_idx, &src) in source_chunk.iter().enumerate() {
            let src_idx = src as usize;
            let word_idx = s_idx / 32;
            let bit_pos = s_idx % 32;
            let mask = 1u32 << bit_pos;

            init_frontier[src_idx * wk + word_idx] |= mask;
            init_visited[src_idx * wk + word_idx] |= mask;
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

        let total_wgs = (nv + 3) / 4;
        let (dispatch_x, dispatch_y) = if total_wgs <= 65535 {
            (total_wgs.max(1), 1)
        } else {
            let dx = 256u32;
            let dy = (total_wgs + dx - 1) / dx;
            (dx, dy)
        };
        let compute_pipeline = pipelines.multi_word_bp_pipeline();

        let mut level = 0u32;
        let mut cur_chunk_steps = 1usize;

        while level <= nv {
            let chunk_len_steps = cur_chunk_steps
                .min(48)
                .min((nv - level + 1) as usize);

            for k in 0..chunk_len_steps {
                let p = MultiWordBpParams {
                    num_vertices: nv,
                    words_per_vertex: words_k,
                    current_level: level + k as u32,
                    total_sources: chunk_len as u32,
                    dispatch_x,
                    bank_shift: 0,
                    bank_mask: 0,
                    pad0: 0,
                };
                queue.write_buffer(&params_bufs[k], 0, bytemuck::bytes_of(&p));
            }
            let last_k = chunk_len_steps - 1;
            queue.write_buffer(&counter_bufs[last_k], 0, bytemuck::bytes_of(&0u32));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MultiWord Hop Batch Encoder"),
            });

            for k in 0..chunk_len_steps {
                let cur_step = level + k as u32;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(compute_pipeline);
                let bg = if cur_step % 2 == 0 {
                    &bgs_even[k]
                } else {
                    &bgs_odd[k]
                };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            encoder.copy_buffer_to_buffer(
                &counter_bufs[last_k],
                0,
                &staging_counter_buf,
                0,
                4,
            );

            queue.submit(Some(encoder.finish()));

            let count = read_buffer_u32_blocking(device, &staging_counter_buf, 1)?[0];
            if count == 0 {
                break;
            }

            level += chunk_len_steps as u32;
            cur_chunk_steps = (cur_chunk_steps * 2).min(48);
        }

        let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MultiWord Distance Read Encoder"),
        });
        read_enc.copy_buffer_to_buffer(&distance_buf, 0, &staging_dist_buf, 0, dist_chunk_bytes);
        queue.submit(Some(read_enc.finish()));

        let dist_chunk = read_buffer_u32_blocking(device, &staging_dist_buf, dist_chunk_words)?;
        let start_dst = chunk_start_src * num_vertices;
        result[start_dst..start_dst + dist_chunk_words].copy_from_slice(&dist_chunk);
    }

    Ok(result)
}

/// Executes Multi-Source Reachability query using MultiWordBitParallel Pull traversal.
pub fn run_multi_word_bp_reachability_pooled(
    ctx: &GpuContext,
    pipelines: &PipelineCache,
    graph: &CsrGraph,
    sources: &[u32],
    pool: &mut Option<PersistentPullBuffers512>,
) -> Result<ReachabilityBitmask> {
    let num_sources = sources.len();
    let num_vertices = graph.num_vertices;
    if num_sources == 0 || num_vertices == 0 {
        return Ok(ReachabilityBitmask::zeroed(num_vertices, num_sources));
    }

    let needs_alloc = match pool.as_ref() {
        Some(cached) => {
            cached.num_vertices != num_vertices || cached.num_edges != graph.num_edges
        }
        None => true,
    };
    if needs_alloc {
        *pool = Some(PersistentPullBuffers512::new(ctx, pipelines, graph));
    }
    let bufs = pool.as_mut().unwrap();

    let mut bitmask = ReachabilityBitmask::zeroed(num_vertices, num_sources);
    let nv = num_vertices as u32;

    for (chunk_idx, chunk_sources) in sources.chunks(512).enumerate() {
        let chunk_start = chunk_idx * 512;
        run_multi_word_bp_reachability_chunk(
            ctx,
            pipelines,
            bufs,
            chunk_sources,
            nv,
            chunk_start,
            &mut bitmask,
        )?;
    }

    Ok(bitmask)
}

pub fn run_multi_word_bp_reachability(
    ctx: &GpuContext,
    pipelines: &PipelineCache,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<ReachabilityBitmask> {
    let mut pool = None;
    run_multi_word_bp_reachability_pooled(ctx, pipelines, graph, sources, &mut pool)
}

fn run_multi_word_bp_reachability_chunk(
    ctx: &GpuContext,
    pipelines: &PipelineCache,
    bufs: &PersistentPullBuffers512,
    chunk_sources: &[u32],
    nv: u32,
    chunk_start: usize,
    bitmask: &mut ReachabilityBitmask,
) -> Result<()> {
    let device = ctx.device();
    let queue = ctx.queue();
    let active_sources = chunk_sources.len();
    if active_sources == 0 {
        return Ok(());
    }
    let bank_sources = bufs.bank_sources;
    let num_banks = (active_sources + bank_sources - 1) / bank_sources;

    queue.write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(chunk_sources));

    let bank_words = bank_sources * (nv as usize);
    let total_reset_wgs = ((bank_words + 255) / 256) as u32;
    let (reset_dx, reset_dy) = if total_reset_wgs <= 65535 {
        (total_reset_wgs.max(1), 1u32)
    } else {
        let dx = 256u32;
        let dy = (total_reset_wgs + dx - 1) / dx;
        (dx, dy)
    };

    let reset_p = ResetParams {
        total_elements: if num_banks == 1 {
            (active_sources * (nv as usize)) as u32
        } else {
            bank_words as u32
        },
        dispatch_x: reset_dx,
    };
    queue.write_buffer(&bufs.reset_params_buf, 0, bytemuck::bytes_of(&reset_p));

    let (bank_shift, bank_mask) = calculate_bank_shift_and_mask(nv as usize);
    let init_src_p = InitSourcesParams {
        num_vertices: nv,
        num_sources: active_sources as u32,
        chunk_start: 0,
        pad0: 0x100 | bank_shift,
    };
    queue.write_buffer(&bufs.init_sources_params_buf, 0, bytemuck::bytes_of(&init_src_p));

    let total_wgs = (nv + 7) / 8;
    let (dispatch_x, dispatch_y) = if total_wgs <= 65535 {
        (total_wgs.max(1), 1u32)
    } else {
        let dx = 256u32;
        let dy = (total_wgs + dx - 1) / dx;
        (dx, dy)
    };

    let compute_pipeline = pipelines.multi_word_bp_pull_512_pipeline();

    let mut level = 0u32;
    let mut cur_chunk_steps = 4usize;

    while level <= nv {
        let chunk_len_steps = cur_chunk_steps
            .min(MAX_CHUNK_STEPS)
            .min((nv - level + 1) as usize);
        if chunk_len_steps == 0 {
            break;
        }

        for k in 0..chunk_len_steps {
            let p = MultiWordBpParams {
                num_vertices: nv,
                words_per_vertex: 16,
                current_level: level + k as u32,
                total_sources: active_sources as u32,
                dispatch_x,
                bank_shift,
                bank_mask,
                pad0: 1,
            };
            queue.write_buffer(&bufs.params_bufs[k], 0, bytemuck::bytes_of(&p));
        }
        let last_k = chunk_len_steps - 1;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pull512 Reachability Hop Batch Encoder"),
        });
        encoder.clear_buffer(&bufs.counter_bufs[last_k], 0, Some(4));

        if level == 0 {
            encoder.clear_buffer(&bufs.frontier_buf_a, 0, None);
            encoder.clear_buffer(&bufs.frontier_buf_b, 0, None);
            encoder.clear_buffer(&bufs.visited_buf, 0, None);

            for b in 0..num_banks {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Pull512 Reset Distance Bank Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.reset_dist_pipeline());
                pass.set_bind_group(0, &bufs.reset_dist_bgs[b], &[]);
                pass.dispatch_workgroups(reset_dx, reset_dy, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Pull512 Init Sources Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.init_sparse_sources_512_pipeline());
                pass.set_bind_group(0, &bufs.init_sources_bg, &[]);
                let init_wgs = ((active_sources as u32 + 255) / 256).max(1);
                pass.dispatch_workgroups(init_wgs, 1, 1);
            }
        }

        for k in 0..chunk_len_steps {
            let cur_step = level + k as u32;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(compute_pipeline);
            let bg = if cur_step % 2 == 0 {
                &bufs.bgs_even[k]
            } else {
                &bufs.bgs_odd[k]
            };
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            &bufs.counter_bufs[last_k],
            0,
            &bufs.staging_counter_buf,
            0,
            4,
        );

        queue.submit(Some(encoder.finish()));

        let count = read_buffer_u32_blocking(device, &bufs.staging_counter_buf, 1)?[0];
        if count == 0 {
            break;
        }

        level += chunk_len_steps as u32;
        cur_chunk_steps = (cur_chunk_steps * 2).min(MAX_CHUNK_STEPS);
    }

    let visited_bytes = (nv as u64) * 16 * 4;
    let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Pull512 Visited Readback Encoder"),
    });
    read_enc.copy_buffer_to_buffer(
        &bufs.visited_buf,
        0,
        &bufs.staging_visited_buf,
        0,
        visited_bytes,
    );
    queue.submit(Some(read_enc.finish()));

    let slice = bufs.staging_visited_buf.slice(..visited_bytes);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });

    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::Maintain::Wait);

    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(rx)
        .context("Staging visited buffer receiver dropped")?
        .context("Failed to map staging visited buffer")?;

    let view = slice.get_mapped_range();
    let visited_slice: &[u32] = bytemuck::cast_slice(&view);

    unpack_pull_reachability(visited_slice, chunk_start, active_sources, nv as usize, bitmask);

    drop(view);
    bufs.staging_visited_buf.unmap();
    Ok(())
}

fn unpack_pull_reachability(
    visited_slice: &[u32],
    chunk_start: usize,
    active_sources: usize,
    _num_vertices: usize,
    bitmask: &mut ReachabilityBitmask,
) {
    let dst_wpv = bitmask.words_per_vertex;
    let src_wpv = 16;
    let start_word = chunk_start / 32;
    let num_chunk_words = (active_sources + 31) / 32;

    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        bitmask
            .words
            .par_chunks_mut(dst_wpv)
            .enumerate()
            .for_each(|(v, v_words)| {
                for k in 0..num_chunk_words {
                    let mask = if (k + 1) * 32 <= active_sources {
                        !0u32
                    } else {
                        let rem = active_sources - k * 32;
                        (1u32 << rem) - 1
                    };
                    v_words[start_word + k] |= visited_slice[v * src_wpv + k] & mask;
                }
            });
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (v, v_words) in bitmask.words.chunks_mut(dst_wpv).enumerate() {
            for k in 0..num_chunk_words {
                let mask = if (k + 1) * 32 <= active_sources {
                    !0u32
                } else {
                    let rem = active_sources - k * 32;
                    (1u32 << rem) - 1
                };
                v_words[start_word + k] |= visited_slice[v * src_wpv + k] & mask;
            }
        }
    }
}
