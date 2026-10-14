//! Fused Indirect Sparse Queue Push BFS dispatcher (Milestone M12 / M14).
//!
//! Designed specifically for directed power-law graphs (such as `web-Stanford`):
//! 1. Fused single-pass traversal: CSR expansion + multi-source visited claims +
//!    distance updates + deduplicated queue insertion in 1 compute pass per hop.
//! 2. Sub-warp cooperative expansion (stride 4, 100% coalesced 16-byte edge reads)
//!    and workgroup shared queue aggregation (reducing global atomics by 128x).
//! 3. GPU-autonomous hop iteration: `prep_sparse_push_hop.wgsl` computes indirect args,
//!    manages ping-pong queues, advances level, and sets hardware zero-workgroup early exit.
//! 4. Speculative macro-batching: 128 hops recorded per CommandEncoder with only 1 status read per chunk.
//! 5. Distance Quantization (u16) & DMA Readback: compresses distance buffer from 72 MB to 36 MB
//!    on GPU before DMA readback, slashing PCIe WC read latency.
//! 6. Persistent Buffer Pooling: reuses preallocated working buffers and bind groups across iterations.

use anyhow::{Context, Result};
use std::sync::Arc;
use webgpu_bfs_core::CsrGraph;
use wgpu::util::DeviceExt;

use crate::{
    context::GpuContext,
    pipeline::PipelineCache,
};

/// Initial speculative hops pre-recorded on level 0 (covers power-law / scale-free diameters in 1 pass).
pub const DEFAULT_MACRO_BATCH_HOPS: usize = 160;

/// Speculative hops pre-recorded for intermediate levels (160 <= level < 480).
pub const ADAPTIVE_MACRO_BATCH_HOPS_MED: usize = 320;

/// Speculative hops pre-recorded for deep levels (level >= 480) on high-diameter graphs (e.g. road networks).
pub const ADAPTIVE_MACRO_BATCH_HOPS_HIGH: usize = 512;

/// Computes the speculative macro-batch chunk size adaptively based on the current BFS exploration depth.
#[inline]
pub fn determine_macro_chunk_hops(level: u32) -> usize {
    if level == 0 {
        DEFAULT_MACRO_BATCH_HOPS
    } else if level < 480 {
        ADAPTIVE_MACRO_BATCH_HOPS_MED
    } else {
        ADAPTIVE_MACRO_BATCH_HOPS_HIGH
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushUniforms {
    pub num_vertices: u32,
    pub chunk_start: u32,
    pub hub_threshold: u32,
    pub num_words: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IterationState {
    pub level: u32,
    pub terminated: u32,
    pub ping_pong_in: u32,
    pub ping_pong_out: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndirectArgs {
    pub workgroups_x: u32,
    pub workgroups_y: u32,
    pub workgroups_z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResetParams {
    pub total_elements: u32,
    pub dispatch_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InitSourcesParams {
    pub num_vertices: u32,
    pub num_sources: u32,
    pub chunk_start: u32,
    pub pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackParams {
    pub total_elements: u32,
    pub total_packed_words: u32,
    pub dispatch_x: u32,
    pub pad0: u32,
}

/// Pre-allocated, reusable GPU storage buffers and bind groups for ISQP (Milestone M14).
pub struct PersistentSparsePushBuffers {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub max_batch_size: usize,

    // Static CSR buffers
    pub row_ptr_buf: wgpu::Buffer,
    pub col_ind_buf: wgpu::Buffer,

    // Reusable traversal state buffers
    pub queues_buf: wgpu::Buffer,
    pub frontiers_buf: wgpu::Buffer,
    pub visited_buf: wgpu::Buffer,
    pub distance_buf: wgpu::Buffer,
    pub packed_u16_buf: wgpu::Buffer,
    pub enqueued_buf: wgpu::Buffer,
    pub queue_sizes_buf: wgpu::Buffer,
    pub state_buf: wgpu::Buffer,
    pub indirect_args_buf: wgpu::Buffer,
    pub uniforms_buf: wgpu::Buffer,
    pub sources_buf: wgpu::Buffer,
    pub reset_params_buf: wgpu::Buffer,
    pub init_sources_params_buf: wgpu::Buffer,
    pub pack_params_buf: wgpu::Buffer,
    pub staging_status_buf: wgpu::Buffer,
    pub staging_packed_buf: wgpu::Buffer,

    // Cached bind groups
    pub reset_dist_bg: wgpu::BindGroup,
    pub init_sources_bg: wgpu::BindGroup,
    pub prep_bg: wgpu::BindGroup,
    pub trav_bg: wgpu::BindGroup,
    pub pack_dist_bg: wgpu::BindGroup,

    // Milestone M17: Dedicated 256 and 512 multi-word pools
    pub pool_256: Option<PersistentSparsePushBuffers256>,
    pub pool_512: Option<PersistentSparsePushBuffers512>,
}

impl PersistentSparsePushBuffers {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &PipelineCache,
        graph: &CsrGraph,
        max_batch_size: usize,
    ) -> Self {
        let num_vertices = graph.num_vertices;
        let num_edges = graph.num_edges;

        // 1. Static CSR buffers
        let row_ptr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ISQP CSR Row Pointers"),
            contents: bytemuck::cast_slice(&graph.row_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dummy_col = [0u32];
        let col_slice: &[u32] = if graph.col_indices.is_empty() {
            &dummy_col
        } else {
            &graph.col_indices
        };
        let col_ind_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ISQP CSR Col Indices"),
            contents: bytemuck::cast_slice(col_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 2. Working state buffers
        let queues_bytes = ((2 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let queues_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Queues Buffer"),
            size: queues_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frontiers_bytes = ((4 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let frontiers_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Frontiers Buffer"),
            size: frontiers_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visited_bytes = ((2 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Visited Buffer"),
            size: visited_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let batch_cap = max_batch_size.max(64);
        let dist_batch_words = batch_cap * num_vertices;
        let dist_batch_bytes = ((dist_batch_words * std::mem::size_of::<u32>()) as u64).max(4);
        let distance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Distance Buffer"),
            size: dist_batch_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let packed_words = (dist_batch_words + 1) / 2;
        let packed_bytes = ((packed_words * std::mem::size_of::<u32>()) as u64).max(4);
        let packed_u16_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Packed u16 Distance Buffer"),
            size: packed_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let enqueued_bytes = ((num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let enqueued_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Enqueued Buffer"),
            size: enqueued_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let queue_sizes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Queue Sizes Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Iteration State Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Indirect Args Buffer"),
            size: 12,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Push Uniforms Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Batch Sources Buffer"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reset_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Reset Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_sources_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Init Sources Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pack_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Pack Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_status_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Staging Status Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_packed_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP Staging Packed Distance Buffer"),
            size: packed_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Static bind groups
        let reset_dist_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ISQP Reset Dist BG"),
            layout: pipelines.reset_dist_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reset_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: distance_buf.as_entire_binding(),
                },
            ],
        });

        let init_sources_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ISQP Init Sources BG"),
            layout: pipelines.init_sparse_sources_layout(),
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
                    resource: frontiers_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visited_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: distance_buf.as_entire_binding(),
                },
            ],
        });

        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ISQP Prep Bind Group"),
            layout: pipelines.prep_sparse_push_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_args_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: queue_sizes_buf.as_entire_binding(),
                },
            ],
        });

        let trav_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ISQP Traversal Bind Group"),
            layout: pipelines.indirect_sparse_push_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_ptr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: col_ind_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: queues_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: frontiers_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: visited_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: distance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: enqueued_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: queue_sizes_buf.as_entire_binding(),
                },
            ],
        });

        let pack_dist_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ISQP Pack Dist BG"),
            layout: pipelines.pack_distances_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pack_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: distance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: packed_u16_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            num_vertices,
            num_edges,
            max_batch_size: batch_cap,
            row_ptr_buf,
            col_ind_buf,
            queues_buf,
            frontiers_buf,
            visited_buf,
            distance_buf,
            packed_u16_buf,
            enqueued_buf,
            queue_sizes_buf,
            state_buf,
            indirect_args_buf,
            uniforms_buf,
            sources_buf,
            reset_params_buf,
            init_sources_params_buf,
            pack_params_buf,
            staging_status_buf,
            staging_packed_buf,
            reset_dist_bg,
            init_sources_bg,
            prep_bg,
            trav_bg,
            pack_dist_bg,
            pool_256: None,
            pool_512: None,
        }
    }
}

/// Pre-allocated, reusable GPU storage buffers and bind groups for 256-source ISQP (Milestone M17).
pub struct PersistentSparsePushBuffers256 {
    pub num_vertices: usize,
    pub num_edges: usize,

    pub row_ptr_buf: wgpu::Buffer,
    pub col_ind_buf: wgpu::Buffer,

    pub queues_buf: wgpu::Buffer,
    pub frontiers_buf: wgpu::Buffer,
    pub visited_buf: wgpu::Buffer,
    pub enqueued_buf: wgpu::Buffer,
    pub queue_sizes_buf: wgpu::Buffer,
    pub state_buf: wgpu::Buffer,
    pub indirect_args_buf: wgpu::Buffer,
    pub uniforms_buf: wgpu::Buffer,
    pub sources_buf: wgpu::Buffer,
    pub reset_params_buf: wgpu::Buffer,
    pub init_sources_params_buf: wgpu::Buffer,
    pub pack_params_buf: wgpu::Buffer,
    pub staging_status_buf: wgpu::Buffer,

    pub dist_banks: [wgpu::Buffer; 4],
    pub packed_u16_banks: [wgpu::Buffer; 4],
    pub staging_packed_buf: wgpu::Buffer,

    pub reset_dist_bgs: [wgpu::BindGroup; 4],
    pub init_sources_bg: wgpu::BindGroup,
    pub prep_bg: wgpu::BindGroup,
    pub trav_bg: wgpu::BindGroup,
    pub pack_dist_bgs: [wgpu::BindGroup; 4],
}

impl PersistentSparsePushBuffers256 {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &PipelineCache,
        graph: &CsrGraph,
    ) -> Self {
        let num_vertices = graph.num_vertices;
        let num_edges = graph.num_edges;

        let row_ptr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ISQP 256 CSR Row Pointers"),
            contents: bytemuck::cast_slice(&graph.row_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dummy_col = [0u32];
        let col_slice: &[u32] = if graph.col_indices.is_empty() {
            &dummy_col
        } else {
            &graph.col_indices
        };
        let col_ind_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ISQP 256 CSR Col Indices"),
            contents: bytemuck::cast_slice(col_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let queues_bytes = ((2 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let queues_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Queues Buffer"),
            size: queues_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 8 words * 2 ping-pong slots = 16 words per vertex
        let frontiers_bytes = ((16 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let frontiers_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Frontiers Buffer"),
            size: frontiers_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 8 words per vertex
        let visited_bytes = ((8 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Visited Buffer"),
            size: visited_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let enqueued_bytes = ((num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let enqueued_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Enqueued Buffer"),
            size: enqueued_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let queue_sizes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Queue Sizes Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Iteration State Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Indirect Args Buffer"),
            size: 12,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Push Uniforms Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Batch Sources Buffer"),
            size: 256 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reset_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Reset Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_sources_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Init Sources Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pack_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Pack Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_status_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Staging Status Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bank_words = 64 * num_vertices;
        let bank_bytes = ((bank_words * std::mem::size_of::<u32>()) as u64).max(4);
        let dist_banks: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ISQP 256 Distance Bank {i}")),
                size: bank_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let packed_words = (bank_words + 1) / 2;
        let packed_bytes = ((packed_words * std::mem::size_of::<u32>()) as u64).max(4);
        let packed_u16_banks: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ISQP 256 Packed u16 Bank {i}")),
                size: packed_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let staging_packed_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 256 Staging Packed Distance Buffer"),
            size: 4 * packed_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reset_dist_bgs: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("ISQP 256 Reset Dist BG {i}")),
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
            label: Some("ISQP 256 Init Sources BG"),
            layout: pipelines.init_sparse_sources_256_layout(),
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
                    resource: frontiers_buf.as_entire_binding(),
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
            ],
        });

        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ISQP 256 Prep BG"),
            layout: pipelines.prep_sparse_push_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_args_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: queue_sizes_buf.as_entire_binding(),
                },
            ],
        });

        let trav_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ISQP 256 Traversal BG"),
            layout: pipelines.indirect_sparse_push_256_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_ptr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: col_ind_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: queues_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: frontiers_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: visited_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: enqueued_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: queue_sizes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: dist_banks[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: dist_banks[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: dist_banks[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: dist_banks[3].as_entire_binding(),
                },
            ],
        });

        let pack_dist_bgs: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("ISQP 256 Pack Dist BG {i}")),
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
            row_ptr_buf,
            col_ind_buf,
            queues_buf,
            frontiers_buf,
            visited_buf,
            enqueued_buf,
            queue_sizes_buf,
            state_buf,
            indirect_args_buf,
            uniforms_buf,
            sources_buf,
            reset_params_buf,
            init_sources_params_buf,
            pack_params_buf,
            staging_status_buf,
            dist_banks,
            packed_u16_banks,
            staging_packed_buf,
            reset_dist_bgs,
            init_sources_bg,
            prep_bg,
            trav_bg,
            pack_dist_bgs,
        }
    }
}

/// Pre-allocated, reusable GPU storage buffers and bind groups for 512-source ISQP (Milestone M17).
pub struct PersistentSparsePushBuffers512 {
    pub num_vertices: usize,
    pub num_edges: usize,

    pub row_ptr_buf: wgpu::Buffer,
    pub col_ind_buf: wgpu::Buffer,

    pub queues_buf: wgpu::Buffer,
    pub frontiers_buf: wgpu::Buffer,
    pub visited_buf: wgpu::Buffer,
    pub enqueued_buf: wgpu::Buffer,
    pub queue_sizes_buf: wgpu::Buffer,
    pub state_buf: wgpu::Buffer,
    pub indirect_args_buf: wgpu::Buffer,
    pub uniforms_buf: wgpu::Buffer,
    pub sources_buf: wgpu::Buffer,
    pub reset_params_buf: wgpu::Buffer,
    pub init_sources_params_buf: wgpu::Buffer,
    pub pack_params_buf: wgpu::Buffer,
    pub staging_status_buf: wgpu::Buffer,

    pub dist_banks: [wgpu::Buffer; 8],
    pub packed_u16_banks: [wgpu::Buffer; 8],
    pub staging_packed_buf: wgpu::Buffer,

    pub reset_dist_bgs: [wgpu::BindGroup; 8],
    pub init_sources_bg: wgpu::BindGroup,
    pub prep_bg: wgpu::BindGroup,
    pub trav_bg: wgpu::BindGroup,
    pub pack_dist_bgs: [wgpu::BindGroup; 8],
}

impl PersistentSparsePushBuffers512 {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &PipelineCache,
        graph: &CsrGraph,
    ) -> Self {
        let num_vertices = graph.num_vertices;
        let num_edges = graph.num_edges;

        let row_ptr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ISQP 512 CSR Row Pointers"),
            contents: bytemuck::cast_slice(&graph.row_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dummy_col = [0u32];
        let col_slice: &[u32] = if graph.col_indices.is_empty() {
            &dummy_col
        } else {
            &graph.col_indices
        };
        let col_ind_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ISQP 512 CSR Col Indices"),
            contents: bytemuck::cast_slice(col_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let queues_bytes = ((2 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let queues_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Queues Buffer"),
            size: queues_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 16 words * 2 ping-pong slots = 32 words per vertex
        let frontiers_bytes = ((32 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let frontiers_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Frontiers Buffer"),
            size: frontiers_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 16 words per vertex
        let visited_bytes = ((16 * num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let visited_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Visited Buffer"),
            size: visited_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let enqueued_bytes = ((num_vertices * std::mem::size_of::<u32>()) as u64).max(4);
        let enqueued_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Enqueued Buffer"),
            size: enqueued_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let queue_sizes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Queue Sizes Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Iteration State Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Indirect Args Buffer"),
            size: 12,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Push Uniforms Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Batch Sources Buffer"),
            size: 512 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reset_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Reset Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_sources_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Init Sources Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pack_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Pack Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_status_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Staging Status Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bank_words = 64 * num_vertices;
        let bank_bytes = ((bank_words * std::mem::size_of::<u32>()) as u64).max(4);
        let dist_banks: [wgpu::Buffer; 8] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ISQP 512 Distance Bank {i}")),
                size: bank_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let packed_words = (bank_words + 1) / 2;
        let packed_bytes = ((packed_words * std::mem::size_of::<u32>()) as u64).max(4);
        let packed_u16_banks: [wgpu::Buffer; 8] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ISQP 512 Packed u16 Bank {i}")),
                size: packed_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let staging_packed_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ISQP 512 Staging Packed Distance Buffer"),
            size: 8 * packed_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reset_dist_bgs: [wgpu::BindGroup; 8] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("ISQP 512 Reset Dist BG {i}")),
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
            label: Some("ISQP 512 Init Sources BG"),
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
                    resource: frontiers_buf.as_entire_binding(),
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

        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ISQP 512 Prep BG"),
            layout: pipelines.prep_sparse_push_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_args_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: queue_sizes_buf.as_entire_binding(),
                },
            ],
        });

        let trav_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ISQP 512 Traversal BG"),
            layout: pipelines.indirect_sparse_push_512_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_ptr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: col_ind_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: queues_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: frontiers_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: visited_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: enqueued_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: queue_sizes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: dist_banks[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: dist_banks[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: dist_banks[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: dist_banks[3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: dist_banks[4].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: dist_banks[5].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: dist_banks[6].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: dist_banks[7].as_entire_binding(),
                },
            ],
        });

        let pack_dist_bgs: [wgpu::BindGroup; 8] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("ISQP 512 Pack Dist BG {i}")),
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
            row_ptr_buf,
            col_ind_buf,
            queues_buf,
            frontiers_buf,
            visited_buf,
            enqueued_buf,
            queue_sizes_buf,
            state_buf,
            indirect_args_buf,
            uniforms_buf,
            sources_buf,
            reset_params_buf,
            init_sources_params_buf,
            pack_params_buf,
            staging_status_buf,
            dist_banks,
            packed_u16_banks,
            staging_packed_buf,
            reset_dist_bgs,
            init_sources_bg,
            prep_bg,
            trav_bg,
            pack_dist_bgs,
        }
    }
}

fn execute_readback_512(
    ctx: &GpuContext,
    pipelines: &PipelineCache,
    bufs: &PersistentSparsePushBuffers512,
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
        label: Some("ISQP 512 Final Distance Pack & Read Encoder"),
    });

    for b in 0..num_banks {
        {
            let mut pass = read_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ISQP 512 Pack Distances Pass"),
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
    if num_banks == 1 {
        let needed_words = (active_dst.len() + 1) / 2;
        unpack_u16_words_to_u32_parallel(&packed_data[..needed_words], active_dst);
    } else {
        std::thread::scope(|s| {
            for (dst_bank, src_bank) in active_dst
                .chunks_mut(64 * nv)
                .zip(packed_data[..num_banks * words_per_bank].chunks(words_per_bank))
            {
                s.spawn(move || {
                    let needed_words = (dst_bank.len() + 1) / 2;
                    unpack_u16_words_to_u32(&src_bank[..needed_words], dst_bank);
                });
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    for (dst_bank, src_bank) in active_dst
        .chunks_mut(64 * nv)
        .zip(packed_data[..num_banks * words_per_bank].chunks(words_per_bank))
    {
        let needed_words = (dst_bank.len() + 1) / 2;
        unpack_u16_words_to_u32(&src_bank[..needed_words], dst_bank);
    }

    drop(packed_view);
    bufs.staging_packed_buf.unmap();
    Ok(())
}

fn run_indirect_sparse_push_512_chunk(
    ctx: &GpuContext,
    pipelines: &PipelineCache,
    bufs: &PersistentSparsePushBuffers512,
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
    let num_banks = (active_sources + 63) / 64;

    let mut unique_sources = Vec::with_capacity(active_sources);
    for &src in chunk_sources {
        if !unique_sources.contains(&src) {
            unique_sources.push(src);
        }
    }
    let initial_in_count = unique_sources.len() as u32;

    queue.write_buffer(&bufs.queues_buf, 0, bytemuck::cast_slice(&unique_sources));
    queue.write_buffer(
        &bufs.queue_sizes_buf,
        0,
        bytemuck::cast_slice(&[initial_in_count, 0u32, 0u32, 0u32]),
    );
    queue.write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(chunk_sources));

    let bank_words = (64 * nv) as usize;
    let total_reset_wgs = ((bank_words + 255) / 256) as u32;
    let (reset_dx, reset_dy) = if total_reset_wgs <= 65535 {
        (total_reset_wgs.max(1), 1u32)
    } else {
        let dx = 256u32;
        let dy = (total_reset_wgs + dx - 1) / dx;
        (dx, dy)
    };

    let reset_p = ResetParams {
        total_elements: bank_words as u32,
        dispatch_x: reset_dx,
    };
    queue.write_buffer(&bufs.reset_params_buf, 0, bytemuck::bytes_of(&reset_p));

    let init_src_p = InitSourcesParams {
        num_vertices: nv,
        num_sources: active_sources as u32,
        chunk_start: chunk_start as u32,
        pad0: 0,
    };
    queue.write_buffer(&bufs.init_sources_params_buf, 0, bytemuck::bytes_of(&init_src_p));

    let active_words = ((active_sources + 31) / 32) as u32;
    let push_uniforms = PushUniforms {
        num_vertices: nv,
        chunk_start: chunk_start as u32,
        hub_threshold: 128,
        num_words: active_words,
    };
    queue.write_buffer(&bufs.uniforms_buf, 0, bytemuck::bytes_of(&push_uniforms));

    let init_state = IterationState {
        level: 0,
        terminated: 0,
        ping_pong_in: 0,
        ping_pong_out: 1,
    };
    queue.write_buffer(&bufs.state_buf, 0, bytemuck::bytes_of(&init_state));

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

    let mut level = 0u32;
    let max_hops = nv;
    let mut distance_read = false;

    while level < max_hops {
        let max_chunk = determine_macro_chunk_hops(level);
        let chunk_len = (max_chunk as u32).min(max_hops - level) as usize;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ISQP 512 Macro-Batch Encoder"),
        });

        if level == 0 {
            encoder.clear_buffer(&bufs.frontiers_buf, 0, None);
            encoder.clear_buffer(&bufs.visited_buf, 0, None);
            encoder.clear_buffer(&bufs.enqueued_buf, 0, None);

            for b in 0..num_banks {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ISQP 512 Reset Distance Bank Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.reset_dist_pipeline());
                pass.set_bind_group(0, &bufs.reset_dist_bgs[b], &[]);
                pass.dispatch_workgroups(reset_dx, reset_dy, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ISQP 512 Init Sources Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.init_sparse_sources_512_pipeline());
                pass.set_bind_group(0, &bufs.init_sources_bg, &[]);
                let init_wgs = ((active_sources as u32 + 255) / 256).max(1);
                pass.dispatch_workgroups(init_wgs, 1, 1);
            }
        }

        for _ in 0..chunk_len {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ISQP 512 Prep Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.prep_sparse_push_pipeline());
                pass.set_bind_group(0, &bufs.prep_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ISQP 512 Traversal Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.indirect_sparse_push_512_pipeline());
                pass.set_bind_group(0, &bufs.trav_bg, &[]);
                pass.dispatch_workgroups_indirect(&bufs.indirect_args_buf, 0);
            }
        }

        encoder.copy_buffer_to_buffer(&bufs.state_buf, 0, &bufs.staging_status_buf, 0, 16);
        queue.submit(Some(encoder.finish()));

        let slice_status = bufs.staging_status_buf.slice(..16);
        let (s_tx, s_rx) = futures_channel::oneshot::channel();
        slice_status.map_async(wgpu::MapMode::Read, move |res| {
            let _ = s_tx.send(res);
        });

        #[cfg(not(target_arch = "wasm32"))]
        device.poll(wgpu::Maintain::Wait);

        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(s_rx)
            .context("Staging status buffer map receiver dropped")?
            .context("Failed to map staging status buffer")?;

        let st_view = slice_status.get_mapped_range();
        let st: &[u32] = bytemuck::cast_slice(&st_view);
        let terminated = st[1];
        drop(st_view);
        bufs.staging_status_buf.unmap();

        if terminated != 0 {
            execute_readback_512(ctx, pipelines, bufs, num_banks, bank_packed_bytes, bank_packed_words, pack_dx, pack_dy, chunk_start, active_sources, nv as usize, result)?;
            distance_read = true;
            break;
        }

        level += chunk_len as u32;
    }

    if !distance_read {
        execute_readback_512(ctx, pipelines, bufs, num_banks, bank_packed_bytes, bank_packed_words, pack_dx, pack_dy, chunk_start, active_sources, nv as usize, result)?;
    }

    Ok(())
}

fn execute_readback_256(
    ctx: &GpuContext,
    pipelines: &PipelineCache,
    bufs: &PersistentSparsePushBuffers256,
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
        label: Some("ISQP 256 Final Distance Pack & Read Encoder"),
    });

    for b in 0..num_banks {
        {
            let mut pass = read_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ISQP 256 Pack Distances Pass"),
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
    if num_banks == 1 {
        let needed_words = (active_dst.len() + 1) / 2;
        unpack_u16_words_to_u32_parallel(&packed_data[..needed_words], active_dst);
    } else {
        std::thread::scope(|s| {
            for (dst_bank, src_bank) in active_dst
                .chunks_mut(64 * nv)
                .zip(packed_data[..num_banks * words_per_bank].chunks(words_per_bank))
            {
                s.spawn(move || {
                    let needed_words = (dst_bank.len() + 1) / 2;
                    unpack_u16_words_to_u32(&src_bank[..needed_words], dst_bank);
                });
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    for (dst_bank, src_bank) in active_dst
        .chunks_mut(64 * nv)
        .zip(packed_data[..num_banks * words_per_bank].chunks(words_per_bank))
    {
        let needed_words = (dst_bank.len() + 1) / 2;
        unpack_u16_words_to_u32(&src_bank[..needed_words], dst_bank);
    }

    drop(packed_view);
    bufs.staging_packed_buf.unmap();
    Ok(())
}

fn run_indirect_sparse_push_256_chunk(
    ctx: &GpuContext,
    pipelines: &PipelineCache,
    bufs: &PersistentSparsePushBuffers256,
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
    let num_banks = (active_sources + 63) / 64;

    let mut unique_sources = Vec::with_capacity(active_sources);
    for &src in chunk_sources {
        if !unique_sources.contains(&src) {
            unique_sources.push(src);
        }
    }
    let initial_in_count = unique_sources.len() as u32;

    queue.write_buffer(&bufs.queues_buf, 0, bytemuck::cast_slice(&unique_sources));
    queue.write_buffer(
        &bufs.queue_sizes_buf,
        0,
        bytemuck::cast_slice(&[initial_in_count, 0u32, 0u32, 0u32]),
    );
    queue.write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(chunk_sources));

    let bank_words = (64 * nv) as usize;
    let total_reset_wgs = ((bank_words + 255) / 256) as u32;
    let (reset_dx, reset_dy) = if total_reset_wgs <= 65535 {
        (total_reset_wgs.max(1), 1u32)
    } else {
        let dx = 256u32;
        let dy = (total_reset_wgs + dx - 1) / dx;
        (dx, dy)
    };

    let reset_p = ResetParams {
        total_elements: bank_words as u32,
        dispatch_x: reset_dx,
    };
    queue.write_buffer(&bufs.reset_params_buf, 0, bytemuck::bytes_of(&reset_p));

    let init_src_p = InitSourcesParams {
        num_vertices: nv,
        num_sources: active_sources as u32,
        chunk_start: chunk_start as u32,
        pad0: 0,
    };
    queue.write_buffer(&bufs.init_sources_params_buf, 0, bytemuck::bytes_of(&init_src_p));

    let active_words = ((active_sources + 31) / 32) as u32;
    let push_uniforms = PushUniforms {
        num_vertices: nv,
        chunk_start: chunk_start as u32,
        hub_threshold: 128,
        num_words: active_words,
    };
    queue.write_buffer(&bufs.uniforms_buf, 0, bytemuck::bytes_of(&push_uniforms));

    let init_state = IterationState {
        level: 0,
        terminated: 0,
        ping_pong_in: 0,
        ping_pong_out: 1,
    };
    queue.write_buffer(&bufs.state_buf, 0, bytemuck::bytes_of(&init_state));

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

    let mut level = 0u32;
    let max_hops = nv;
    let mut distance_read = false;

    while level < max_hops {
        let max_chunk = determine_macro_chunk_hops(level);
        let chunk_len = (max_chunk as u32).min(max_hops - level) as usize;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ISQP 256 Macro-Batch Encoder"),
        });

        if level == 0 {
            encoder.clear_buffer(&bufs.frontiers_buf, 0, None);
            encoder.clear_buffer(&bufs.visited_buf, 0, None);
            encoder.clear_buffer(&bufs.enqueued_buf, 0, None);

            for b in 0..num_banks {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ISQP 256 Reset Distance Bank Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.reset_dist_pipeline());
                pass.set_bind_group(0, &bufs.reset_dist_bgs[b], &[]);
                pass.dispatch_workgroups(reset_dx, reset_dy, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ISQP 256 Init Sources Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.init_sparse_sources_256_pipeline());
                pass.set_bind_group(0, &bufs.init_sources_bg, &[]);
                let init_wgs = ((active_sources as u32 + 255) / 256).max(1);
                pass.dispatch_workgroups(init_wgs, 1, 1);
            }
        }

        for _ in 0..chunk_len {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ISQP 256 Prep Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.prep_sparse_push_pipeline());
                pass.set_bind_group(0, &bufs.prep_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ISQP 256 Traversal Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.indirect_sparse_push_256_pipeline());
                pass.set_bind_group(0, &bufs.trav_bg, &[]);
                pass.dispatch_workgroups_indirect(&bufs.indirect_args_buf, 0);
            }
        }

        encoder.copy_buffer_to_buffer(&bufs.state_buf, 0, &bufs.staging_status_buf, 0, 16);
        queue.submit(Some(encoder.finish()));

        let slice_status = bufs.staging_status_buf.slice(..16);
        let (s_tx, s_rx) = futures_channel::oneshot::channel();
        slice_status.map_async(wgpu::MapMode::Read, move |res| {
            let _ = s_tx.send(res);
        });

        #[cfg(not(target_arch = "wasm32"))]
        device.poll(wgpu::Maintain::Wait);

        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(s_rx)
            .context("Staging status buffer map receiver dropped")?
            .context("Failed to map staging status buffer")?;

        let st_view = slice_status.get_mapped_range();
        let st: &[u32] = bytemuck::cast_slice(&st_view);
        let terminated = st[1];
        drop(st_view);
        bufs.staging_status_buf.unmap();

        if terminated != 0 {
            execute_readback_256(ctx, pipelines, bufs, num_banks, bank_packed_bytes, bank_packed_words, pack_dx, pack_dy, chunk_start, active_sources, nv as usize, result)?;
            distance_read = true;
            break;
        }

        level += chunk_len as u32;
    }

    if !distance_read {
        execute_readback_256(ctx, pipelines, bufs, num_banks, bank_packed_bytes, bank_packed_words, pack_dx, pack_dy, chunk_start, active_sources, nv as usize, result)?;
    }

    Ok(())
}

/// Executes Multi-Source BFS using 256-source Fused Indirect Sparse Queue Push into a caller-provided reusable destination buffer.
pub fn run_indirect_sparse_push_256_into(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    mut pool: Option<&mut Option<PersistentSparsePushBuffers256>>,
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
    let device = ctx.device();
    let nv = num_vertices as u32;

    let mut local_storage;
    let bufs: &mut PersistentSparsePushBuffers256 = match pool {
        Some(ref mut opt) => {
            let needs_alloc = match opt.as_ref() {
                Some(cached) => {
                    cached.num_vertices != num_vertices || cached.num_edges != graph.num_edges
                }
                None => true,
            };
            if needs_alloc {
                **opt = Some(PersistentSparsePushBuffers256::new(device, pipelines, graph));
            }
            opt.as_mut().unwrap()
        }
        None => {
            local_storage = Some(PersistentSparsePushBuffers256::new(device, pipelines, graph));
            local_storage.as_mut().unwrap()
        }
    };

    for (chunk_idx, chunk_sources) in sources.chunks(256).enumerate() {
        let chunk_start = chunk_idx * 256;
        run_indirect_sparse_push_256_chunk(
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

/// Executes Multi-Source BFS using 512-source Fused Indirect Sparse Queue Push into a caller-provided reusable destination buffer.
pub fn run_indirect_sparse_push_512_into(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    mut pool: Option<&mut Option<PersistentSparsePushBuffers512>>,
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
    let device = ctx.device();
    let nv = num_vertices as u32;

    let mut local_storage;
    let bufs: &mut PersistentSparsePushBuffers512 = match pool {
        Some(ref mut opt) => {
            let needs_alloc = match opt.as_ref() {
                Some(cached) => {
                    cached.num_vertices != num_vertices || cached.num_edges != graph.num_edges
                }
                None => true,
            };
            if needs_alloc {
                **opt = Some(PersistentSparsePushBuffers512::new(device, pipelines, graph));
            }
            opt.as_mut().unwrap()
        }
        None => {
            local_storage = Some(PersistentSparsePushBuffers512::new(device, pipelines, graph));
            local_storage.as_mut().unwrap()
        }
    };

    for (chunk_idx, chunk_sources) in sources.chunks(512).enumerate() {
        let chunk_start = chunk_idx * 512;
        run_indirect_sparse_push_512_chunk(
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

/// Executes Multi-Source BFS using Fused Indirect Sparse Queue Push with default local buffers.
pub fn run_indirect_sparse_push(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
) -> Result<Vec<u32>> {
    run_indirect_sparse_push_pooled(ctx, pipelines, graph, sources, None)
}

/// Executes Multi-Source BFS using Fused Indirect Sparse Queue Push with optional persistent buffer pooling.
pub fn run_indirect_sparse_push_pooled(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    pool: Option<&mut Option<PersistentSparsePushBuffers>>,
) -> Result<Vec<u32>> {
    let total_elements = sources.len() * graph.num_vertices;
    if total_elements == 0 {
        return Ok(Vec::new());
    }
    let mut result = vec![0u32; total_elements];
    run_indirect_sparse_push_into(ctx, pipelines, graph, sources, pool, &mut result)?;
    Ok(result)
}

/// Executes Multi-Source BFS using Fused Indirect Sparse Queue Push into a caller-provided reusable destination buffer.
pub fn run_indirect_sparse_push_into(
    ctx: &Arc<GpuContext>,
    pipelines: &Arc<PipelineCache>,
    graph: &CsrGraph,
    sources: &[u32],
    mut pool: Option<&mut Option<PersistentSparsePushBuffers>>,
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

    let device = ctx.device();
    let queue = ctx.queue();
    let nv = num_vertices as u32;

    // Buffer acquisition: reuse from pool if available, otherwise allocate
    let mut local_storage;
    let bufs: &mut PersistentSparsePushBuffers = match pool {
        Some(ref mut opt) => {
            let needs_alloc = match opt.as_ref() {
                Some(cached) => {
                    cached.num_vertices != num_vertices || cached.num_edges != graph.num_edges
                }
                None => true,
            };
            if needs_alloc {
                **opt = Some(PersistentSparsePushBuffers::new(device, pipelines, graph, 64));
            }
            opt.as_mut().unwrap()
        }
        None => {
            local_storage = Some(PersistentSparsePushBuffers::new(device, pipelines, graph, 64));
            local_storage.as_mut().unwrap()
        }
    };

    if num_sources > 256 {
        let pool_512 = match bufs.pool_512.as_mut() {
            Some(p) if p.num_vertices == num_vertices && p.num_edges == graph.num_edges => p,
            _ => {
                bufs.pool_512 = Some(PersistentSparsePushBuffers512::new(device, pipelines, graph));
                bufs.pool_512.as_mut().unwrap()
            }
        };

        for (chunk_idx, chunk_sources) in sources.chunks(512).enumerate() {
            let chunk_start = chunk_idx * 512;
            run_indirect_sparse_push_512_chunk(
                ctx,
                pipelines,
                pool_512,
                chunk_sources,
                nv,
                chunk_start,
                result,
            )?;
        }
        return Ok(());
    } else if num_sources > 64 {
        let pool_256 = match bufs.pool_256.as_mut() {
            Some(p) if p.num_vertices == num_vertices && p.num_edges == graph.num_edges => p,
            _ => {
                bufs.pool_256 = Some(PersistentSparsePushBuffers256::new(device, pipelines, graph));
                bufs.pool_256.as_mut().unwrap()
            }
        };

        for (chunk_idx, chunk_sources) in sources.chunks(256).enumerate() {
            let chunk_start = chunk_idx * 256;
            run_indirect_sparse_push_256_chunk(
                ctx,
                pipelines,
                pool_256,
                chunk_sources,
                nv,
                chunk_start,
                result,
            )?;
        }
        return Ok(());
    }

    let reset_dist_pipeline = pipelines.reset_dist_pipeline();
    let init_sources_pipeline = pipelines.init_sparse_sources_pipeline();
    let prep_pipeline = pipelines.prep_sparse_push_pipeline();
    let trav_pipeline = pipelines.indirect_sparse_push_pipeline();
    let pack_pipeline = pipelines.pack_distances_pipeline();

    // Iterate over sources in chunks of up to 64
    for (batch_idx, batch_sources) in sources.chunks(64).enumerate() {
        let batch_size = batch_sources.len();
        let batch_start = batch_idx * 64;

        // Reset grid geometry: 2D dispatch for exact active elements (batch_size * nv)
        let total_reset_elements = (batch_size * num_vertices) as usize;
        let total_reset_wgs = ((total_reset_elements + 255) / 256) as u32;
        let (reset_dx, reset_dy) = if total_reset_wgs <= 65535 {
            (total_reset_wgs.max(1), 1u32)
        } else {
            let dx = 256u32;
            let dy = (total_reset_wgs + dx - 1) / dx;
            (dx, dy)
        };

        // Build list of unique source vertices for the initial active queue
        let mut unique_sources = Vec::with_capacity(batch_size);
        for &src in batch_sources {
            if !unique_sources.contains(&src) {
                unique_sources.push(src);
            }
        }
        let initial_in_count = unique_sources.len() as u32;

        // Write small initialization buffers (total upload < 1 KB across PCIe!)
        queue.write_buffer(&bufs.queues_buf, 0, bytemuck::cast_slice(&unique_sources));
        queue.write_buffer(&bufs.queue_sizes_buf, 0, bytemuck::cast_slice(&[initial_in_count, 0u32, 0u32, 0u32]));
        queue.write_buffer(&bufs.sources_buf, 0, bytemuck::cast_slice(batch_sources));

        let reset_p = ResetParams {
            total_elements: total_reset_elements as u32,
            dispatch_x: reset_dx,
        };
        queue.write_buffer(&bufs.reset_params_buf, 0, bytemuck::bytes_of(&reset_p));

        let init_src_p = InitSourcesParams {
            num_vertices: nv,
            num_sources: batch_size as u32,
            chunk_start: 0,
            pad0: 0,
        };
        queue.write_buffer(&bufs.init_sources_params_buf, 0, bytemuck::bytes_of(&init_src_p));

        let active_words = if batch_size <= 32 { 1u32 } else { 2u32 };
        let push_uniforms = PushUniforms {
            num_vertices: nv,
            chunk_start: 0,
            hub_threshold: 128,
            num_words: active_words,
        };
        queue.write_buffer(&bufs.uniforms_buf, 0, bytemuck::bytes_of(&push_uniforms));

        let init_state = IterationState {
            level: 0,
            terminated: 0,
            ping_pong_in: 0,
            ping_pong_out: 1,
        };
        queue.write_buffer(&bufs.state_buf, 0, bytemuck::bytes_of(&init_state));

        // Speculative Macro-Batching Loop
        let mut level = 0u32;
        let max_hops = nv;
        let batch_elements = batch_size * num_vertices;
        let total_packed_words = ((batch_elements + 1) / 2) as u32;
        let packed_bytes = (total_packed_words as u64) * 4;

        let pack_wgs = (total_packed_words + 255) / 256;
        let (pack_dx, pack_dy) = if pack_wgs <= 65535 {
            (pack_wgs.max(1), 1u32)
        } else {
            let dx = 256u32;
            let dy = (pack_wgs + dx - 1) / dx;
            (dx, dy)
        };

        let pack_p = PackParams {
            total_elements: batch_elements as u32,
            total_packed_words,
            dispatch_x: pack_dx,
            pad0: 0,
        };
        queue.write_buffer(&bufs.pack_params_buf, 0, bytemuck::bytes_of(&pack_p));

        let mut distance_read = false;

        while level < max_hops {
            let max_chunk = determine_macro_chunk_hops(level);
            let chunk_len = (max_chunk as u32).min(max_hops - level) as usize;

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ISQP Macro-Batch Encoder"),
            });

            // Fast on-GPU initialization at step 0:
            // 1) Clear frontiers, visited, and enqueued
            // 2) Reset distance buffer to u32::MAX
            // 3) Initialize source distances and bitmasks
            if level == 0 {
                encoder.clear_buffer(&bufs.frontiers_buf, 0, None);
                encoder.clear_buffer(&bufs.visited_buf, 0, None);
                encoder.clear_buffer(&bufs.enqueued_buf, 0, None);

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ISQP Reset Distance Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(reset_dist_pipeline);
                    pass.set_bind_group(0, &bufs.reset_dist_bg, &[]);
                    pass.dispatch_workgroups(reset_dx, reset_dy, 1);
                }

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ISQP Init Sources Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(init_sources_pipeline);
                    pass.set_bind_group(0, &bufs.init_sources_bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }

            // Pre-record speculative hops
            for _ in 0..chunk_len {
                // Pass 1: Prepare Hop (1 workgroup, 1 thread)
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ISQP Prep Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(prep_pipeline);
                    pass.set_bind_group(0, &bufs.prep_bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }

                // Pass 2: Fused Traversal (Indirect Dispatch)
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ISQP Traversal Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(trav_pipeline);
                    pass.set_bind_group(0, &bufs.trav_bg, &[]);
                    pass.dispatch_workgroups_indirect(&bufs.indirect_args_buf, 0);
                }
            }

            // Copy ONLY 16-byte IterationState to staging status buffer
            encoder.copy_buffer_to_buffer(&bufs.state_buf, 0, &bufs.staging_status_buf, 0, 16);

            let cmd_buf = encoder.finish();

            // Single command buffer submission for the entire speculative chunk
            queue.submit(Some(cmd_buf));

            // Synchronous map of ONLY the 16-byte staging status buffer
            let slice_status = bufs.staging_status_buf.slice(..16);
            let (s_tx, s_rx) = futures_channel::oneshot::channel();
            slice_status.map_async(wgpu::MapMode::Read, move |res| {
                let _ = s_tx.send(res);
            });

            #[cfg(not(target_arch = "wasm32"))]
            device.poll(wgpu::Maintain::Wait);

            #[cfg(not(target_arch = "wasm32"))]
            pollster::block_on(s_rx)
                .context("Staging status buffer map receiver dropped")?
                .context("Failed to map staging status buffer")?;

            let st_view = slice_status.get_mapped_range();
            let st: &[u32] = bytemuck::cast_slice(&st_view);
            let terminated = st[1];
            drop(st_view);
            bufs.staging_status_buf.unmap();

            if terminated != 0 {
                // Dedicated read pass: pack u32 -> u16 on GPU (0.38 ms) and copy half the size across PCIe
                let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ISQP Final Distance Pack & Read Encoder"),
                });

                {
                    let mut pass = read_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ISQP Pack Distances Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pack_pipeline);
                    pass.set_bind_group(0, &bufs.pack_dist_bg, &[]);
                    pass.dispatch_workgroups(pack_dx, pack_dy, 1);
                }

                read_enc.copy_buffer_to_buffer(&bufs.packed_u16_buf, 0, &bufs.staging_packed_buf, 0, packed_bytes);
                queue.submit(Some(read_enc.finish()));

                let slice_packed = bufs.staging_packed_buf.slice(..packed_bytes);
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
                let start_dst = batch_start * num_vertices;

                unpack_u16_words_to_u32_parallel(packed_data, &mut result[start_dst..start_dst + batch_elements]);

                drop(packed_view);
                bufs.staging_packed_buf.unmap();
                distance_read = true;
                break;
            }

            level += chunk_len as u32;
        }

        if !distance_read {
            let mut read_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ISQP Final Distance Pack & Read Encoder (Max Hops Fallback)"),
            });

            {
                let mut pass = read_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ISQP Pack Distances Pass (Fallback)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pack_pipeline);
                pass.set_bind_group(0, &bufs.pack_dist_bg, &[]);
                pass.dispatch_workgroups(pack_dx, pack_dy, 1);
            }

            read_enc.copy_buffer_to_buffer(&bufs.packed_u16_buf, 0, &bufs.staging_packed_buf, 0, packed_bytes);
            queue.submit(Some(read_enc.finish()));

            let slice_packed = bufs.staging_packed_buf.slice(..packed_bytes);
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
            let start_dst = batch_start * num_vertices;

            unpack_u16_words_to_u32_parallel(packed_data, &mut result[start_dst..start_dst + batch_elements]);

            drop(packed_view);
            bufs.staging_packed_buf.unmap();
        }
    }

    Ok(())
}

/// Ultra-fast branchless host unpacking from u16 packed words to u32 canonical distance format.
/// Preserves exact bitwise parity: unreachable 0xFFFF maps to u32::MAX (0xFFFFFFFF).
#[inline]
pub fn unpack_u16_words_to_u32(packed_words: &[u32], dst: &mut [u32]) {
    let num_pairs = dst.len() / 2;
    let pairs: &mut [[u32; 2]] = bytemuck::cast_slice_mut(&mut dst[..num_pairs * 2]);
    for (i, &word) in packed_words[..num_pairs].iter().enumerate() {
        let p0 = word & 0xFFFF;
        let p1 = word >> 16;
        let mask0 = ((p0 == 0xFFFF) as u32).wrapping_neg();
        let mask1 = ((p1 == 0xFFFF) as u32).wrapping_neg();
        pairs[i] = [p0 | mask0, p1 | mask1];
    }
    if dst.len() % 2 == 1 {
        let last_word = packed_words[num_pairs];
        let p0 = last_word & 0xFFFF;
        let mask0 = ((p0 == 0xFFFF) as u32).wrapping_neg();
        dst[num_pairs * 2] = p0 | mask0;
    }
}

/// Multi-threaded host unpacking across all CPU cores.
/// Partitions dst into even-length chunks so that packed word indices remain aligned.
pub fn unpack_u16_words_to_u32_parallel(packed_words: &[u32], dst: &mut [u32]) {
    #[cfg(target_arch = "wasm32")]
    {
        unpack_u16_words_to_u32(packed_words, dst);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let total_len = dst.len();
        if total_len <= 32768 {
            unpack_u16_words_to_u32(packed_words, dst);
            return;
        }

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .min(16);

        // Ensure chunk length is an even number of u32 elements
        let raw_chunk = (total_len + num_threads - 1) / num_threads;
        let chunk_elements = ((raw_chunk + 1) / 2) * 2;
        let words_per_chunk = chunk_elements / 2;

        std::thread::scope(|s| {
            for (chunk_idx, dst_chunk) in dst.chunks_mut(chunk_elements).enumerate() {
                let word_start = chunk_idx * words_per_chunk;
                let needed_words = (dst_chunk.len() + 1) / 2;
                let src_slice = &packed_words[word_start..word_start + needed_words];
                s.spawn(move || {
                    unpack_u16_words_to_u32(src_slice, dst_chunk);
                });
            }
        });
    }
}
