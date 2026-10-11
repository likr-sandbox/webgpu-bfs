//! Embedded WGSL compute shaders for `webgpu-bfs`.
//!
//! Provides compile-time embedded WGSL shader strings via `include_str!` and a
//! validation registry for automated testing with Naga.

/// WGSL shader source for baseline Single-Source BFS Push traversal.
pub const NAIVE_SPMV_WGSL: &str = include_str!("naive_spmv.wgsl");

/// WGSL shader source for dense Multi-Source SpMM Push traversal (2D dispatch).
pub const BATCHED_SPMM_WGSL: &str = include_str!("batched_spmm.wgsl");

/// WGSL shader source for dense Multi-Source SpMM Push traversal (2D warp-broadcast tiled).
pub const BATCHED_SPMM_2D_WGSL: &str = include_str!("batched_spmm_2d.wgsl");

/// WGSL shader source for 32-source bit-parallel Push traversal.
pub const BIT_PARALLEL_PUSH_WGSL: &str = include_str!("bit_parallel_push.wgsl");

/// WGSL shader source for 32-source bit-parallel Pull traversal.
pub const BIT_PARALLEL_PULL_WGSL: &str = include_str!("bit_parallel_pull.wgsl");

/// WGSL shader source for 32-source Boolean reachability filtering and distance update.
pub const BOOLEAN_REACHABILITY_WGSL: &str = include_str!("boolean_reachability.wgsl");

/// WGSL shader source for 64-source bit-parallel reachability and distance update.
pub const BIT_PARALLEL_64_WGSL: &str = include_str!("bit_parallel_64.wgsl");

/// WGSL shader source for 64-source bit-parallel Push traversal.
pub const BIT_PARALLEL_64_PUSH_WGSL: &str = include_str!("bit_parallel_64_push.wgsl");

/// WGSL shader source for 64-source single-pass bit-parallel Pull traversal.
pub const BIT_PARALLEL_64_PULL_WGSL: &str = include_str!("bit_parallel_64_pull.wgsl");

/// WGSL shader source for Single-Source Sparse Queue BFS Push traversal.
pub const SPARSE_QUEUE_PUSH_WGSL: &str = include_str!("sparse_queue_push.wgsl");

/// WGSL shader source for Multi-Source Bit-Parallel Sparse Push traversal.
pub const SPARSE_BP_PUSH_WGSL: &str = include_str!("sparse_bp_push.wgsl");

/// WGSL shader source for Multi-Source Bit-Parallel Sparse Filter & Distance Update.
pub const SPARSE_BP_FILTER_WGSL: &str = include_str!("sparse_bp_filter.wgsl");

/// WGSL shader source for Massive Multi-Word Bit-Parallel Pull traversal (256 - 2048 sources).
pub const MULTI_WORD_BP_PULL_WGSL: &str = include_str!("multi_word_bp_pull.wgsl");

/// WGSL shader source for Indirect Macro-Batch preparation kernel.
pub const PREP_HOP_WGSL: &str = include_str!("prep_hop.wgsl");

/// WGSL shader source for Indirect Bit-Parallel Pull traversal.
pub const INDIRECT_BP_PULL_WGSL: &str = include_str!("indirect_bp_pull.wgsl");

/// WGSL shader source for Indirect Sparse Push preparation kernel (Milestone M12).
pub const PREP_SPARSE_PUSH_HOP_WGSL: &str = include_str!("prep_sparse_push_hop.wgsl");

/// WGSL shader source for Fused Indirect Sparse Push traversal (Milestone M12).
pub const INDIRECT_SPARSE_PUSH_WGSL: &str = include_str!("indirect_sparse_push.wgsl");

/// WGSL shader source for Massive Multi-Word Bit-Parallel Push traversal (Milestone M12).
pub const MULTI_WORD_BP_PUSH_WGSL: &str = include_str!("multi_word_bp_push.wgsl");

/// WGSL shader source for Massive Multi-Word Bit-Parallel Filter & Distance Update (Milestone M12).
pub const MULTI_WORD_BP_FILTER_WGSL: &str = include_str!("multi_word_bp_filter.wgsl");

/// WGSL shader source for fast buffer reset to u32::MAX (0xFFFFFFFF).
pub const RESET_BUFFER_U32_MAX_WGSL: &str = include_str!("reset_buffer_u32_max.wgsl");

/// WGSL shader source for GPU source initialization for Sparse Push BFS.
pub const INIT_SPARSE_PUSH_SOURCES_WGSL: &str = include_str!("init_sparse_push_sources.wgsl");

/// WGSL shader source for packing u32 distances to u16 (Milestone M14).
pub const PACK_DISTANCES_U32_TO_U16_WGSL: &str = include_str!("pack_distances_u32_to_u16.wgsl");

/// Returns a slice of tuples containing (filename, shader_source) for all embedded shaders.
pub fn all_shaders() -> &'static [(&'static str, &'static str)] {
    &[
        ("naive_spmv.wgsl", NAIVE_SPMV_WGSL),
        ("batched_spmm.wgsl", BATCHED_SPMM_WGSL),
        ("batched_spmm_2d.wgsl", BATCHED_SPMM_2D_WGSL),
        ("bit_parallel_push.wgsl", BIT_PARALLEL_PUSH_WGSL),
        ("bit_parallel_pull.wgsl", BIT_PARALLEL_PULL_WGSL),
        ("boolean_reachability.wgsl", BOOLEAN_REACHABILITY_WGSL),
        ("bit_parallel_64.wgsl", BIT_PARALLEL_64_WGSL),
        ("bit_parallel_64_push.wgsl", BIT_PARALLEL_64_PUSH_WGSL),
        ("bit_parallel_64_pull.wgsl", BIT_PARALLEL_64_PULL_WGSL),
        ("sparse_queue_push.wgsl", SPARSE_QUEUE_PUSH_WGSL),
        ("sparse_bp_push.wgsl", SPARSE_BP_PUSH_WGSL),
        ("sparse_bp_filter.wgsl", SPARSE_BP_FILTER_WGSL),
        ("multi_word_bp_pull.wgsl", MULTI_WORD_BP_PULL_WGSL),
        ("prep_hop.wgsl", PREP_HOP_WGSL),
        ("indirect_bp_pull.wgsl", INDIRECT_BP_PULL_WGSL),
        ("prep_sparse_push_hop.wgsl", PREP_SPARSE_PUSH_HOP_WGSL),
        ("indirect_sparse_push.wgsl", INDIRECT_SPARSE_PUSH_WGSL),
        ("multi_word_bp_push.wgsl", MULTI_WORD_BP_PUSH_WGSL),
        ("multi_word_bp_filter.wgsl", MULTI_WORD_BP_FILTER_WGSL),
        ("reset_buffer_u32_max.wgsl", RESET_BUFFER_U32_MAX_WGSL),
        ("init_sparse_push_sources.wgsl", INIT_SPARSE_PUSH_SOURCES_WGSL),
        ("pack_distances_u32_to_u16.wgsl", PACK_DISTANCES_U32_TO_U16_WGSL),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validates that a WGSL source string can be parsed and passes Naga type-checking.
    fn validate_wgsl_shader(name: &str, source: &str) {
        let module = match naga::front::wgsl::parse_str(source) {
            Ok(module) => module,
            Err(e) => {
                panic!(
                    "Shader '{}' failed WGSL syntax parsing:\n{}",
                    name,
                    e.emit_to_string(source)
                );
            }
        };

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );

        if let Err(e) = validator.validate(&module) {
            panic!("Shader '{}' failed Naga validation:\n{:?}", name, e);
        }
    }

    #[test]
    fn test_all_embedded_shaders_compile() {
        for (name, source) in all_shaders() {
            validate_wgsl_shader(name, source);
        }
    }

    #[test]
    fn test_naive_spmv_syntax() {
        validate_wgsl_shader("naive_spmv.wgsl", NAIVE_SPMV_WGSL);
    }

    #[test]
    fn test_batched_spmm_syntax() {
        validate_wgsl_shader("batched_spmm.wgsl", BATCHED_SPMM_WGSL);
    }

    #[test]
    fn test_bit_parallel_push_syntax() {
        validate_wgsl_shader("bit_parallel_push.wgsl", BIT_PARALLEL_PUSH_WGSL);
    }

    #[test]
    fn test_bit_parallel_pull_syntax() {
        validate_wgsl_shader("bit_parallel_pull.wgsl", BIT_PARALLEL_PULL_WGSL);
    }

    #[test]
    fn test_boolean_reachability_syntax() {
        validate_wgsl_shader("boolean_reachability.wgsl", BOOLEAN_REACHABILITY_WGSL);
    }

    #[test]
    fn test_bit_parallel_64_syntax() {
        validate_wgsl_shader("bit_parallel_64.wgsl", BIT_PARALLEL_64_WGSL);
    }

    #[test]
    fn test_bit_parallel_64_push_syntax() {
        validate_wgsl_shader("bit_parallel_64_push.wgsl", BIT_PARALLEL_64_PUSH_WGSL);
    }

    #[test]
    fn test_sparse_queue_push_syntax() {
        validate_wgsl_shader("sparse_queue_push.wgsl", SPARSE_QUEUE_PUSH_WGSL);
    }

    #[test]
    fn test_sparse_bp_push_syntax() {
        validate_wgsl_shader("sparse_bp_push.wgsl", SPARSE_BP_PUSH_WGSL);
    }

    #[test]
    fn test_sparse_bp_filter_syntax() {
        validate_wgsl_shader("sparse_bp_filter.wgsl", SPARSE_BP_FILTER_WGSL);
    }

    #[test]
    fn test_multi_word_bp_pull_syntax() {
        validate_wgsl_shader("multi_word_bp_pull.wgsl", MULTI_WORD_BP_PULL_WGSL);
    }

    #[test]
    fn test_prep_hop_syntax() {
        validate_wgsl_shader("prep_hop.wgsl", PREP_HOP_WGSL);
    }

    #[test]
    fn test_indirect_bp_pull_syntax() {
        validate_wgsl_shader("indirect_bp_pull.wgsl", INDIRECT_BP_PULL_WGSL);
    }

    #[test]
    fn test_prep_sparse_push_hop_syntax() {
        validate_wgsl_shader("prep_sparse_push_hop.wgsl", PREP_SPARSE_PUSH_HOP_WGSL);
    }

    #[test]
    fn test_indirect_sparse_push_syntax() {
        validate_wgsl_shader("indirect_sparse_push.wgsl", INDIRECT_SPARSE_PUSH_WGSL);
    }

    #[test]
    fn test_multi_word_bp_push_syntax() {
        validate_wgsl_shader("multi_word_bp_push.wgsl", MULTI_WORD_BP_PUSH_WGSL);
    }

    #[test]
    fn test_multi_word_bp_filter_syntax() {
        validate_wgsl_shader("multi_word_bp_filter.wgsl", MULTI_WORD_BP_FILTER_WGSL);
    }

    #[test]
    fn test_reset_buffer_u32_max_syntax() {
        validate_wgsl_shader("reset_buffer_u32_max.wgsl", RESET_BUFFER_U32_MAX_WGSL);
    }

    #[test]
    fn test_init_sparse_push_sources_syntax() {
        validate_wgsl_shader("init_sparse_push_sources.wgsl", INIT_SPARSE_PUSH_SOURCES_WGSL);
    }

    #[test]
    fn test_pack_distances_u32_to_u16_syntax() {
        validate_wgsl_shader("pack_distances_u32_to_u16.wgsl", PACK_DISTANCES_U32_TO_U16_WGSL);
    }
}
