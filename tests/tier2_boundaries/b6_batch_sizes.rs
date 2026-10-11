use webgpu_bfs_core::generators::{
    generate_erdos_renyi_np, generate_graph500_rmat, generate_random_tree, Grid2D,
};

#[test]
fn test_t2_b6_01_batch_size_1_minimal_active_mask() {
    let graph = generate_erdos_renyi_np(60, 0.1, Some(42)).unwrap();
    let sources = [7];

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &graph, &sources);
}

#[test]
fn test_t2_b6_02_batch_sizes_3_and_17_prime_unaligned_masks() {
    let grid = Grid2D::new(6, 6).unwrap();
    let ctx = crate::common::shared_gpu_context();

    for &s_count in &[3, 17] {
        let sources: Vec<u32> = (0..s_count).map(|i| (i * 2) % 36).collect();
        crate::common::assert_cross_engine_parity(&ctx, &grid.graph, &sources);
    }
}

#[test]
fn test_t2_b6_03_batch_sizes_31_and_33_word32_boundary_crossing() {
    let graph = generate_erdos_renyi_np(60, 0.1, Some(99)).unwrap();
    let ctx = crate::common::shared_gpu_context();

    for &s_count in &[31, 33] {
        let sources: Vec<u32> = (0..s_count).map(|i| (i * 3) % 60).collect();
        crate::common::assert_cross_engine_parity(&ctx, &graph, &sources);
    }
}

#[test]
fn test_t2_b6_04_batch_sizes_63_and_65_word64_boundary_crossing() {
    let tree = generate_random_tree(80, Some(101)).unwrap();
    let ctx = crate::common::shared_gpu_context();

    for &s_count in &[63, 65] {
        let sources: Vec<u32> = (0..s_count).map(|i| (i * 5) % 80).collect();
        crate::common::assert_cross_engine_parity(&ctx, &tree, &sources);
    }
}

#[test]
fn test_t2_b6_05_batch_size_129_triple_chunk_offset_stress() {
    let rmat = generate_graph500_rmat(5, 8, Some(555)).unwrap(); // |V| = 32
    let sources: Vec<u32> = (0..129).map(|i| i % 32).collect();

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &rmat, &sources);
}
