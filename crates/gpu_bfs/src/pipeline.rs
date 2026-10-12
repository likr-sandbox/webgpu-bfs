//! Compute pipeline and bind group layout caching for GPU BFS traversal.

use anyhow::Result;
use webgpu_bfs_shaders::{
    BATCHED_SPMM_WGSL, BIT_PARALLEL_64_PULL_WGSL, BIT_PARALLEL_64_PUSH_WGSL, BIT_PARALLEL_64_WGSL,
    BIT_PARALLEL_PULL_WGSL, BIT_PARALLEL_PUSH_WGSL, BOOLEAN_REACHABILITY_WGSL,
    INDIRECT_BP_PULL_WGSL, INDIRECT_SPARSE_PUSH_WGSL, INIT_SPARSE_PUSH_SOURCES_WGSL,
    MULTI_WORD_BP_FILTER_WGSL, MULTI_WORD_BP_PULL_WGSL, MULTI_WORD_BP_PUSH_WGSL, NAIVE_SPMV_WGSL,
    PREP_HOP_WGSL, PREP_SPARSE_PUSH_HOP_WGSL, RESET_BUFFER_U32_MAX_WGSL, SPARSE_BP_FILTER_WGSL,
    SPARSE_BP_PUSH_WGSL, SPARSE_QUEUE_PUSH_WGSL, PACK_DISTANCES_U32_TO_U16_WGSL,
    INDIRECT_SPARSE_PUSH_256_WGSL, INDIRECT_SPARSE_PUSH_512_WGSL,
    INIT_SPARSE_PUSH_SOURCES_256_WGSL, INIT_SPARSE_PUSH_SOURCES_512_WGSL,
};
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

/// Type alias for `PipelineCache` matching architectural specifications.
pub type BfsPipelines = PipelineCache;

/// Cache of precompiled compute pipelines and bind group layouts.
#[derive(Debug)]
pub struct PipelineCache {
    naive_layout: BindGroupLayout,
    naive_pipeline: ComputePipeline,
    batched_layout: BindGroupLayout,
    batched_pipeline: ComputePipeline,

    // Milestone 3: Bit-Parallel Push (32 & 64)
    bp_push_layout: BindGroupLayout,
    bp32_push_pipeline: ComputePipeline,
    bp64_push_pipeline: ComputePipeline,

    // Milestone 3: Bit-Parallel Pull (32)
    pull_layout: BindGroupLayout,
    pull_pipeline: ComputePipeline,

    // Milestone 3: Reachability Filter & Distance Update (32 & 64)
    bp_filter_layout: BindGroupLayout,
    bp32_filter_pipeline: ComputePipeline,
    bp64_filter_pipeline: ComputePipeline,

    // Single-Pass 64-source Pull Pipeline
    bp64_pull_layout: BindGroupLayout,
    bp64_pull_pipeline: ComputePipeline,

    // Milestone 7: Single-Source Sparse Queue Push
    sparse_queue_layout: BindGroupLayout,
    sparse_queue_pipeline: ComputePipeline,

    // Milestone 7: Bit-Parallel Sparse Push (Pass 1)
    sparse_bp_push_layout: BindGroupLayout,
    sparse_bp_push_pipeline: ComputePipeline,

    // Milestone 7: Bit-Parallel Sparse Filter (Pass 2)
    sparse_bp_filter_layout: BindGroupLayout,
    sparse_bp_filter_pipeline: ComputePipeline,

    // Milestone 8: Massive Multi-Word Bit-Parallel Pull Pipeline
    multi_word_bp_layout: BindGroupLayout,
    multi_word_bp_pipeline: ComputePipeline,

    // Milestone 9: Indirect Macro-Batch Preparation & Traversal Pipelines
    prep_hop_layout: BindGroupLayout,
    prep_hop_pipeline: ComputePipeline,
    indirect_bp_pull_layout: BindGroupLayout,
    indirect_bp_pull_pipeline: ComputePipeline,

    // Milestone 12: Fused Indirect Sparse Queue Push
    prep_sparse_push_layout: BindGroupLayout,
    prep_sparse_push_pipeline: ComputePipeline,
    indirect_sparse_push_layout: BindGroupLayout,
    indirect_sparse_push_pipeline: ComputePipeline,

    // Milestone 12: Multi-Word Bit-Parallel Push
    multi_word_push_layout: BindGroupLayout,
    multi_word_push_pipeline: ComputePipeline,
    multi_word_filter_layout: BindGroupLayout,
    multi_word_filter_pipeline: ComputePipeline,

    // Milestone 12: Fast GPU Buffer Reset & Source Initialization
    reset_dist_layout: BindGroupLayout,
    reset_dist_pipeline: ComputePipeline,
    init_sparse_sources_layout: BindGroupLayout,
    init_sparse_sources_pipeline: ComputePipeline,

    // Milestone 14: Distance Quantization (u32 -> u16)
    pack_distances_layout: BindGroupLayout,
    pack_distances_pipeline: ComputePipeline,

    // Milestone 17: Multi-Word Sparse Push Pipelines (256 & 512 width)
    indirect_sparse_push_256_layout: BindGroupLayout,
    indirect_sparse_push_256_pipeline: ComputePipeline,
    init_sparse_sources_256_layout: BindGroupLayout,
    init_sparse_sources_256_pipeline: ComputePipeline,

    indirect_sparse_push_512_layout: BindGroupLayout,
    indirect_sparse_push_512_pipeline: ComputePipeline,
    init_sparse_sources_512_layout: BindGroupLayout,
    init_sparse_sources_512_pipeline: ComputePipeline,
}

impl PipelineCache {
    /// Creates and compiles all compute pipelines for GPU BFS methods.
    pub fn new(device: &Device) -> Result<Self> {
        let naive_layout = Self::create_traversal_bind_group_layout(device, "naive_spmv_bgl");
        let batched_layout = Self::create_traversal_bind_group_layout(device, "batched_spmm_bgl");

        let naive_pipeline = Self::create_compute_pipeline(
            device,
            "naive_spmv_pipeline",
            &naive_layout,
            NAIVE_SPMV_WGSL,
        );
        let batched_pipeline = Self::create_compute_pipeline(
            device,
            "batched_spmm_pipeline",
            &batched_layout,
            BATCHED_SPMM_WGSL,
        );

        // M3 Bit-Parallel Push Layout (5 entries)
        let bp_push_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bp_push_bgl"),
            entries: &[
                Self::bgl_uniform(0),
                Self::bgl_storage_ro(1),
                Self::bgl_storage_ro(2),
                Self::bgl_storage_ro(3),
                Self::bgl_storage_rw(4),
            ],
        });

        let bp32_push_pipeline = Self::create_compute_pipeline(
            device,
            "bp32_push_pipeline",
            &bp_push_layout,
            BIT_PARALLEL_PUSH_WGSL,
        );
        let bp64_push_pipeline = Self::create_compute_pipeline(
            device,
            "bp64_push_pipeline",
            &bp_push_layout,
            BIT_PARALLEL_64_PUSH_WGSL,
        );

        // M3 Bit-Parallel Pull Layout (6 entries)
        let pull_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("pull_bgl"),
            entries: &[
                Self::bgl_uniform(0),
                Self::bgl_storage_ro(1),
                Self::bgl_storage_ro(2),
                Self::bgl_storage_ro(3),
                Self::bgl_storage_ro(4),
                Self::bgl_storage_rw(5),
            ],
        });

        let pull_pipeline = Self::create_compute_pipeline(
            device,
            "pull_pipeline",
            &pull_layout,
            BIT_PARALLEL_PULL_WGSL,
        );

        // M3 Bit-Parallel Filter & Distance Update Layout (6 entries)
        let bp_filter_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bp_filter_bgl"),
            entries: &[
                Self::bgl_uniform(0),
                Self::bgl_storage_rw(1),
                Self::bgl_storage_rw(2),
                Self::bgl_storage_rw(3),
                Self::bgl_storage_rw(4),
                Self::bgl_storage_rw(5),
            ],
        });

        let bp32_filter_pipeline = Self::create_compute_pipeline(
            device,
            "bp32_filter_pipeline",
            &bp_filter_layout,
            BOOLEAN_REACHABILITY_WGSL,
        );
        let bp64_filter_pipeline = Self::create_compute_pipeline(
            device,
            "bp64_filter_pipeline",
            &bp_filter_layout,
            BIT_PARALLEL_64_WGSL,
        );

        // M3 Bit-Parallel 64 Single-Pass Pull Layout (8 entries)
        let bp64_pull_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bp64_pull_bgl"),
            entries: &[
                Self::bgl_uniform(0),
                Self::bgl_storage_ro(1),
                Self::bgl_storage_ro(2),
                Self::bgl_storage_ro(3),
                Self::bgl_storage_rw(4),
                Self::bgl_storage_rw(5),
                Self::bgl_storage_rw(6),
                Self::bgl_storage_rw(7),
            ],
        });

        let bp64_pull_pipeline = Self::create_compute_pipeline(
            device,
            "bp64_pull_pipeline",
            &bp64_pull_layout,
            BIT_PARALLEL_64_PULL_WGSL,
        );

        // Milestone 7: Single-Source Sparse Queue Push Layout (7 entries)
        let sparse_queue_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sparse_queue_bgl"),
            entries: &[
                Self::bgl_storage_ro(0), // row_offsets
                Self::bgl_storage_ro(1), // col_indices
                Self::bgl_storage_ro(2), // queue_in
                Self::bgl_storage_rw(3), // queue_out
                Self::bgl_storage_rw(4), // distances
                Self::bgl_storage_rw(5), // queue_sizes
                Self::bgl_uniform(6),    // params
            ],
        });

        let sparse_queue_pipeline = Self::create_compute_pipeline(
            device,
            "sparse_queue_pipeline",
            &sparse_queue_layout,
            SPARSE_QUEUE_PUSH_WGSL,
        );

        // Milestone 7: Multi-Source Bit-Parallel Sparse Push Layout (Pass 1 - 9 entries)
        let sparse_bp_push_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sparse_bp_push_bgl"),
            entries: &[
                Self::bgl_uniform(0),     // params: PushParams
                Self::bgl_storage_ro(1),  // row_ptr
                Self::bgl_storage_ro(2),  // col_ind
                Self::bgl_storage_ro(3),  // queue_in
                Self::bgl_storage_rw(4),  // curr_frontier
                Self::bgl_storage_rw(5),  // next_candidate
                Self::bgl_storage_rw(6),  // touched
                Self::bgl_storage_rw(7),  // candidate_queue
                Self::bgl_storage_rw(8),  // queue_sizes
            ],
        });

        let sparse_bp_push_pipeline = Self::create_compute_pipeline(
            device,
            "sparse_bp_push_pipeline",
            &sparse_bp_push_layout,
            SPARSE_BP_PUSH_WGSL,
        );

        // Milestone 7: Multi-Source Bit-Parallel Sparse Filter Layout (Pass 2 - 9 entries)
        let sparse_bp_filter_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sparse_bp_filter_bgl"),
            entries: &[
                Self::bgl_uniform(0),     // params: FilterParams
                Self::bgl_storage_ro(1),  // candidate_queue
                Self::bgl_storage_rw(2),  // next_candidate
                Self::bgl_storage_rw(3),  // visited
                Self::bgl_storage_rw(4),  // next_frontier
                Self::bgl_storage_rw(5),  // distance
                Self::bgl_storage_rw(6),  // touched
                Self::bgl_storage_rw(7),  // queue_out
                Self::bgl_storage_rw(8),  // queue_sizes
            ],
        });

        let sparse_bp_filter_pipeline = Self::create_compute_pipeline(
            device,
            "sparse_bp_filter_pipeline",
            &sparse_bp_filter_layout,
            SPARSE_BP_FILTER_WGSL,
        );

        // Milestone 8: Massive Multi-Word Bit-Parallel Pull Layout (8 entries)
        let multi_word_bp_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("multi_word_bp_bgl"),
            entries: &[
                Self::bgl_uniform(0),     // params: MultiWordBpParams
                Self::bgl_storage_ro(1),  // row_offsets (CSC col offsets)
                Self::bgl_storage_ro(2),  // col_indices (CSC row indices)
                Self::bgl_storage_ro(3),  // curr_frontier
                Self::bgl_storage_rw(4),  // visited
                Self::bgl_storage_rw(5),  // next_frontier
                Self::bgl_storage_rw(6),  // distance
                Self::bgl_storage_rw(7),  // active_count
            ],
        });

        let multi_word_bp_pipeline = Self::create_compute_pipeline(
            device,
            "multi_word_bp_pull_pipeline",
            &multi_word_bp_layout,
            MULTI_WORD_BP_PULL_WGSL,
        );

        // Milestone 9: Indirect Macro-Batch Preparation Layout (2 entries)
        let prep_hop_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prep_hop_bgl"),
            entries: &[
                Self::bgl_storage_rw(0), // binding 0: IterationState
                Self::bgl_storage_rw(1), // binding 1: IndirectArgs
            ],
        });

        let prep_hop_pipeline = Self::create_compute_pipeline(
            device,
            "prep_hop_pipeline",
            &prep_hop_layout,
            PREP_HOP_WGSL,
        );

        // Milestone 9: Indirect Bit-Parallel Pull Layout (8 entries)
        let indirect_bp_pull_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("indirect_bp_pull_bgl"),
            entries: &[
                Self::bgl_uniform(0),     // params: IndirectStaticParams
                Self::bgl_storage_ro(1),  // row_ptr
                Self::bgl_storage_ro(2),  // col_ind
                Self::bgl_storage_ro(3),  // curr_frontier
                Self::bgl_storage_rw(4),  // visited
                Self::bgl_storage_rw(5),  // next_frontier
                Self::bgl_storage_rw(6),  // distance
                Self::bgl_storage_rw(7),  // state: IterationState
            ],
        });

        let indirect_bp_pull_pipeline = Self::create_compute_pipeline(
            device,
            "indirect_bp_pull_pipeline",
            &indirect_bp_pull_layout,
            INDIRECT_BP_PULL_WGSL,
        );

        // Milestone 12: Fused Indirect Sparse Queue Push
        let prep_sparse_push_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prep_sparse_push_bgl"),
            entries: &[
                Self::bgl_storage_rw(0), // state: IterationState
                Self::bgl_storage_rw(1), // indirect_args: IndirectArgs
                Self::bgl_storage_rw(2), // queue_sizes: array<atomic<u32>>
            ],
        });

        let prep_sparse_push_pipeline = Self::create_compute_pipeline(
            device,
            "prep_sparse_push_pipeline",
            &prep_sparse_push_layout,
            PREP_SPARSE_PUSH_HOP_WGSL,
        );

        let indirect_sparse_push_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("indirect_sparse_push_bgl"),
            entries: &[
                Self::bgl_uniform(0),    // uniforms: PushUniforms
                Self::bgl_uniform(1),    // state: IterationState
                Self::bgl_storage_ro(2), // row_ptr
                Self::bgl_storage_ro(3), // col_ind
                Self::bgl_storage_rw(4), // queues
                Self::bgl_storage_rw(5), // frontiers
                Self::bgl_storage_rw(6), // visited
                Self::bgl_storage_rw(7), // distance
                Self::bgl_storage_rw(8), // enqueued
                Self::bgl_storage_rw(9), // queue_sizes
            ],
        });

        let indirect_sparse_push_pipeline = Self::create_compute_pipeline(
            device,
            "indirect_sparse_push_pipeline",
            &indirect_sparse_push_layout,
            INDIRECT_SPARSE_PUSH_WGSL,
        );

        // Milestone 12: Multi-Word Bit-Parallel Push
        let multi_word_push_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("multi_word_push_bgl"),
            entries: &[
                Self::bgl_uniform(0),    // params: PushParams
                Self::bgl_storage_ro(1), // row_ptr
                Self::bgl_storage_ro(2), // col_ind
                Self::bgl_storage_ro(3), // curr_frontier
                Self::bgl_storage_rw(4), // next_candidate
            ],
        });

        let multi_word_push_pipeline = Self::create_compute_pipeline(
            device,
            "multi_word_push_pipeline",
            &multi_word_push_layout,
            MULTI_WORD_BP_PUSH_WGSL,
        );

        let multi_word_filter_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("multi_word_filter_bgl"),
            entries: &[
                Self::bgl_uniform(0),    // params: FilterParams
                Self::bgl_storage_rw(1), // next_candidate
                Self::bgl_storage_rw(2), // visited
                Self::bgl_storage_rw(3), // next_frontier
                Self::bgl_storage_rw(4), // distance
                Self::bgl_storage_rw(5), // active_count
            ],
        });

        let multi_word_filter_pipeline = Self::create_compute_pipeline(
            device,
            "multi_word_filter_pipeline",
            &multi_word_filter_layout,
            MULTI_WORD_BP_FILTER_WGSL,
        );

        // Milestone 12: Fast GPU Buffer Reset & Source Initialization
        let reset_dist_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("reset_dist_bgl"),
            entries: &[
                Self::bgl_uniform(0),    // params: ResetParams
                Self::bgl_storage_rw(1), // data
            ],
        });

        let reset_dist_pipeline = Self::create_compute_pipeline(
            device,
            "reset_dist_pipeline",
            &reset_dist_layout,
            RESET_BUFFER_U32_MAX_WGSL,
        );

        let init_sparse_sources_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("init_sparse_sources_bgl"),
            entries: &[
                Self::bgl_uniform(0),    // params: InitSourcesParams
                Self::bgl_storage_ro(1), // sources
                Self::bgl_storage_rw(2), // frontiers
                Self::bgl_storage_rw(3), // visited
                Self::bgl_storage_rw(4), // distance
            ],
        });

        let init_sparse_sources_pipeline = Self::create_compute_pipeline(
            device,
            "init_sparse_sources_pipeline",
            &init_sparse_sources_layout,
            INIT_SPARSE_PUSH_SOURCES_WGSL,
        );

        // Milestone 14: Distance Quantization (u32 -> u16)
        let pack_distances_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("pack_distances_bgl"),
            entries: &[
                Self::bgl_uniform(0),    // params: PackParams
                Self::bgl_storage_ro(1), // dist_u32
                Self::bgl_storage_rw(2), // dist_u16_packed
            ],
        });

        let pack_distances_pipeline = Self::create_compute_pipeline(
            device,
            "pack_distances_pipeline",
            &pack_distances_layout,
            PACK_DISTANCES_U32_TO_U16_WGSL,
        );

        // Milestone 17: Multi-Word Sparse Push Pipelines (256 & 512 width)
        let indirect_sparse_push_256_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("indirect_sparse_push_256_bgl"),
                entries: &[
                    Self::bgl_uniform(0),     // uniforms: PushUniforms
                    Self::bgl_uniform(1),     // state: IterationState
                    Self::bgl_storage_ro(2),  // row_ptr
                    Self::bgl_storage_ro(3),  // col_ind
                    Self::bgl_storage_rw(4),  // queues
                    Self::bgl_storage_rw(5),  // frontiers
                    Self::bgl_storage_rw(6),  // visited
                    Self::bgl_storage_rw(7),  // enqueued
                    Self::bgl_storage_rw(8),  // queue_sizes
                    Self::bgl_storage_rw(9),  // dist_bank0
                    Self::bgl_storage_rw(10), // dist_bank1
                    Self::bgl_storage_rw(11), // dist_bank2
                    Self::bgl_storage_rw(12), // dist_bank3
                ],
            });

        let indirect_sparse_push_256_pipeline = Self::create_compute_pipeline(
            device,
            "indirect_sparse_push_256_pipeline",
            &indirect_sparse_push_256_layout,
            INDIRECT_SPARSE_PUSH_256_WGSL,
        );

        let init_sparse_sources_256_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("init_sparse_sources_256_bgl"),
                entries: &[
                    Self::bgl_uniform(0),    // params: InitSourcesParams
                    Self::bgl_storage_ro(1), // sources
                    Self::bgl_storage_rw(2), // frontiers
                    Self::bgl_storage_rw(3), // visited
                    Self::bgl_storage_rw(4), // dist_bank0
                    Self::bgl_storage_rw(5), // dist_bank1
                    Self::bgl_storage_rw(6), // dist_bank2
                    Self::bgl_storage_rw(7), // dist_bank3
                ],
            });

        let init_sparse_sources_256_pipeline = Self::create_compute_pipeline(
            device,
            "init_sparse_sources_256_pipeline",
            &init_sparse_sources_256_layout,
            INIT_SPARSE_PUSH_SOURCES_256_WGSL,
        );

        let indirect_sparse_push_512_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("indirect_sparse_push_512_bgl"),
                entries: &[
                    Self::bgl_uniform(0),     // uniforms: PushUniforms
                    Self::bgl_uniform(1),     // state: IterationState
                    Self::bgl_storage_ro(2),  // row_ptr
                    Self::bgl_storage_ro(3),  // col_ind
                    Self::bgl_storage_rw(4),  // queues
                    Self::bgl_storage_rw(5),  // frontiers
                    Self::bgl_storage_rw(6),  // visited
                    Self::bgl_storage_rw(7),  // enqueued
                    Self::bgl_storage_rw(8),  // queue_sizes
                    Self::bgl_storage_rw(9),  // dist_bank0
                    Self::bgl_storage_rw(10), // dist_bank1
                    Self::bgl_storage_rw(11), // dist_bank2
                    Self::bgl_storage_rw(12), // dist_bank3
                    Self::bgl_storage_rw(13), // dist_bank4
                    Self::bgl_storage_rw(14), // dist_bank5
                    Self::bgl_storage_rw(15), // dist_bank6
                    Self::bgl_storage_rw(16), // dist_bank7
                ],
            });

        let indirect_sparse_push_512_pipeline = Self::create_compute_pipeline(
            device,
            "indirect_sparse_push_512_pipeline",
            &indirect_sparse_push_512_layout,
            INDIRECT_SPARSE_PUSH_512_WGSL,
        );

        let init_sparse_sources_512_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("init_sparse_sources_512_bgl"),
                entries: &[
                    Self::bgl_uniform(0),     // params: InitSourcesParams
                    Self::bgl_storage_ro(1),  // sources
                    Self::bgl_storage_rw(2),  // frontiers
                    Self::bgl_storage_rw(3),  // visited
                    Self::bgl_storage_rw(4),  // dist_bank0
                    Self::bgl_storage_rw(5),  // dist_bank1
                    Self::bgl_storage_rw(6),  // dist_bank2
                    Self::bgl_storage_rw(7),  // dist_bank3
                    Self::bgl_storage_rw(8),  // dist_bank4
                    Self::bgl_storage_rw(9),  // dist_bank5
                    Self::bgl_storage_rw(10), // dist_bank6
                    Self::bgl_storage_rw(11), // dist_bank7
                ],
            });

        let init_sparse_sources_512_pipeline = Self::create_compute_pipeline(
            device,
            "init_sparse_sources_512_pipeline",
            &init_sparse_sources_512_layout,
            INIT_SPARSE_PUSH_SOURCES_512_WGSL,
        );

        Ok(Self {
            naive_layout,
            naive_pipeline,
            batched_layout,
            batched_pipeline,
            bp_push_layout,
            bp32_push_pipeline,
            bp64_push_pipeline,
            pull_layout,
            pull_pipeline,
            bp_filter_layout,
            bp32_filter_pipeline,
            bp64_filter_pipeline,
            bp64_pull_layout,
            bp64_pull_pipeline,
            sparse_queue_layout,
            sparse_queue_pipeline,
            sparse_bp_push_layout,
            sparse_bp_push_pipeline,
            sparse_bp_filter_layout,
            sparse_bp_filter_pipeline,
            multi_word_bp_layout,
            multi_word_bp_pipeline,
            prep_hop_layout,
            prep_hop_pipeline,
            indirect_bp_pull_layout,
            indirect_bp_pull_pipeline,
            prep_sparse_push_layout,
            prep_sparse_push_pipeline,
            indirect_sparse_push_layout,
            indirect_sparse_push_pipeline,
            multi_word_push_layout,
            multi_word_push_pipeline,
            multi_word_filter_layout,
            multi_word_filter_pipeline,
            reset_dist_layout,
            reset_dist_pipeline,
            init_sparse_sources_layout,
            init_sparse_sources_pipeline,
            pack_distances_layout,
            pack_distances_pipeline,
            indirect_sparse_push_256_layout,
            indirect_sparse_push_256_pipeline,
            init_sparse_sources_256_layout,
            init_sparse_sources_256_pipeline,
            indirect_sparse_push_512_layout,
            indirect_sparse_push_512_pipeline,
            init_sparse_sources_512_layout,
            init_sparse_sources_512_pipeline,
        })
    }

    fn bgl_uniform(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn bgl_storage_ro(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn bgl_storage_rw(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn create_compute_pipeline(
        device: &Device,
        label: &str,
        layout: &BindGroupLayout,
        source: &str,
    ) -> ComputePipeline {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipe_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", label)),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipe_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Standard 7-slot bind group layout for baseline traversal:
    /// 0: row_offsets   (storage, read)
    /// 1: col_indices   (storage, read)
    /// 2: curr_frontier (storage, read)
    /// 3: next_frontier (storage, read_write)
    /// 4: distances     (storage, read_write)
    /// 5: params        (uniform)
    /// 6: counter       (storage, read_write)
    fn create_traversal_bind_group_layout(device: &Device, label: &str) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                Self::bgl_storage_ro(0),
                Self::bgl_storage_ro(1),
                Self::bgl_storage_ro(2),
                Self::bgl_storage_rw(3),
                Self::bgl_storage_rw(4),
                Self::bgl_uniform(5),
                Self::bgl_storage_rw(6),
            ],
        })
    }

    pub fn naive_spmv_layout(&self) -> &BindGroupLayout {
        &self.naive_layout
    }

    pub fn naive_spmv_pipeline(&self) -> &ComputePipeline {
        &self.naive_pipeline
    }

    pub fn batched_spmm_layout(&self) -> &BindGroupLayout {
        &self.batched_layout
    }

    pub fn batched_spmm_pipeline(&self) -> &ComputePipeline {
        &self.batched_pipeline
    }

    pub fn bp_push_layout(&self) -> &BindGroupLayout {
        &self.bp_push_layout
    }

    pub fn bp32_push_pipeline(&self) -> &ComputePipeline {
        &self.bp32_push_pipeline
    }

    pub fn bp64_push_pipeline(&self) -> &ComputePipeline {
        &self.bp64_push_pipeline
    }

    pub fn pull_layout(&self) -> &BindGroupLayout {
        &self.pull_layout
    }

    pub fn pull_pipeline(&self) -> &ComputePipeline {
        &self.pull_pipeline
    }

    pub fn bp_filter_layout(&self) -> &BindGroupLayout {
        &self.bp_filter_layout
    }

    pub fn bp32_filter_pipeline(&self) -> &ComputePipeline {
        &self.bp32_filter_pipeline
    }

    pub fn bp64_filter_pipeline(&self) -> &ComputePipeline {
        &self.bp64_filter_pipeline
    }

    pub fn bp64_pull_layout(&self) -> &BindGroupLayout {
        &self.bp64_pull_layout
    }

    pub fn bp64_pull_pipeline(&self) -> &ComputePipeline {
        &self.bp64_pull_pipeline
    }

    // Convenience aliases
    pub fn bool_push_layout(&self) -> &BindGroupLayout {
        &self.bp_push_layout
    }

    pub fn bool_push_pipeline(&self) -> &ComputePipeline {
        &self.bp32_push_pipeline
    }

    pub fn bool_reach_layout(&self) -> &BindGroupLayout {
        &self.bp_filter_layout
    }

    pub fn bool_reach_pipeline(&self) -> &ComputePipeline {
        &self.bp32_filter_pipeline
    }

    pub fn push_layout(&self) -> &BindGroupLayout {
        &self.bp_push_layout
    }

    pub fn push_pipeline(&self) -> &ComputePipeline {
        &self.bp32_push_pipeline
    }

    // Milestone 7: Sparse Queue & Bit-Parallel Pipelines
    pub fn sparse_queue_layout(&self) -> &BindGroupLayout {
        &self.sparse_queue_layout
    }

    pub fn sparse_queue_pipeline(&self) -> &ComputePipeline {
        &self.sparse_queue_pipeline
    }

    pub fn sparse_bp_push_layout(&self) -> &BindGroupLayout {
        &self.sparse_bp_push_layout
    }

    pub fn sparse_bp_push_pipeline(&self) -> &ComputePipeline {
        &self.sparse_bp_push_pipeline
    }

    pub fn sparse_bp_filter_layout(&self) -> &BindGroupLayout {
        &self.sparse_bp_filter_layout
    }

    pub fn sparse_bp_filter_pipeline(&self) -> &ComputePipeline {
        &self.sparse_bp_filter_pipeline
    }

    // Milestone 8: Multi-Word Bit-Parallel Pipelines
    pub fn multi_word_bp_layout(&self) -> &BindGroupLayout {
        &self.multi_word_bp_layout
    }

    pub fn multi_word_bp_pipeline(&self) -> &ComputePipeline {
        &self.multi_word_bp_pipeline
    }

    // Milestone 9: Indirect Preparation & Traversal Pipeline Accessors
    pub fn prep_hop_layout(&self) -> &BindGroupLayout {
        &self.prep_hop_layout
    }

    pub fn prep_hop_pipeline(&self) -> &ComputePipeline {
        &self.prep_hop_pipeline
    }

    pub fn indirect_bp_pull_layout(&self) -> &BindGroupLayout {
        &self.indirect_bp_pull_layout
    }

    pub fn indirect_bp_pull_pipeline(&self) -> &ComputePipeline {
        &self.indirect_bp_pull_pipeline
    }

    // Forwarding aliases specified in task description
    pub fn pipeline_sparse_queue(&self) -> &ComputePipeline {
        self.sparse_queue_pipeline()
    }

    pub fn pipeline_bit_parallel_64(&self) -> &ComputePipeline {
        self.bp64_push_pipeline()
    }

    // Milestone 12: Fused Indirect Sparse Queue Push
    pub fn prep_sparse_push_layout(&self) -> &BindGroupLayout {
        &self.prep_sparse_push_layout
    }

    pub fn prep_sparse_push_pipeline(&self) -> &ComputePipeline {
        &self.prep_sparse_push_pipeline
    }

    pub fn indirect_sparse_push_layout(&self) -> &BindGroupLayout {
        &self.indirect_sparse_push_layout
    }

    pub fn indirect_sparse_push_pipeline(&self) -> &ComputePipeline {
        &self.indirect_sparse_push_pipeline
    }

    // Milestone 12: Multi-Word Bit-Parallel Push
    pub fn multi_word_push_layout(&self) -> &BindGroupLayout {
        &self.multi_word_push_layout
    }

    pub fn multi_word_push_pipeline(&self) -> &ComputePipeline {
        &self.multi_word_push_pipeline
    }

    pub fn multi_word_filter_layout(&self) -> &BindGroupLayout {
        &self.multi_word_filter_layout
    }

    pub fn multi_word_filter_pipeline(&self) -> &ComputePipeline {
        &self.multi_word_filter_pipeline
    }

    // Milestone 12: Fast GPU Buffer Reset & Source Initialization
    pub fn reset_dist_layout(&self) -> &BindGroupLayout {
        &self.reset_dist_layout
    }

    pub fn reset_dist_pipeline(&self) -> &ComputePipeline {
        &self.reset_dist_pipeline
    }

    pub fn init_sparse_sources_layout(&self) -> &BindGroupLayout {
        &self.init_sparse_sources_layout
    }

    pub fn init_sparse_sources_pipeline(&self) -> &ComputePipeline {
        &self.init_sparse_sources_pipeline
    }

    // Milestone 14: Distance Quantization (u32 -> u16)
    pub fn pack_distances_layout(&self) -> &BindGroupLayout {
        &self.pack_distances_layout
    }

    pub fn pack_distances_pipeline(&self) -> &ComputePipeline {
        &self.pack_distances_pipeline
    }

    // Milestone 17: Multi-Word Sparse Push Pipelines (256 & 512 width)
    pub fn indirect_sparse_push_256_layout(&self) -> &BindGroupLayout {
        &self.indirect_sparse_push_256_layout
    }

    pub fn indirect_sparse_push_256_pipeline(&self) -> &ComputePipeline {
        &self.indirect_sparse_push_256_pipeline
    }

    pub fn init_sparse_sources_256_layout(&self) -> &BindGroupLayout {
        &self.init_sparse_sources_256_layout
    }

    pub fn init_sparse_sources_256_pipeline(&self) -> &ComputePipeline {
        &self.init_sparse_sources_256_pipeline
    }

    pub fn indirect_sparse_push_512_layout(&self) -> &BindGroupLayout {
        &self.indirect_sparse_push_512_layout
    }

    pub fn indirect_sparse_push_512_pipeline(&self) -> &ComputePipeline {
        &self.indirect_sparse_push_512_pipeline
    }

    pub fn init_sparse_sources_512_layout(&self) -> &BindGroupLayout {
        &self.init_sparse_sources_512_layout
    }

    pub fn init_sparse_sources_512_pipeline(&self) -> &ComputePipeline {
        &self.init_sparse_sources_512_pipeline
    }
}

