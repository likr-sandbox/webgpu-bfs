use webgpu_bfs_core::generators::{
    generate_erdos_renyi_np, generate_graph500_rmat, generate_random_tree, Grid2D,
};

#[test]
fn test_t3_grid2d_arbitrary_batch_sweep() {
    let grid = Grid2D::new(8, 8).unwrap();
    let ctx = crate::common::shared_gpu_context();

    for &s_count in &[7, 33, 65] {
        let sources: Vec<u32> = (0..s_count).map(|i| (i * 7) % 64).collect();
        let oracle = grid.ms_bfs_oracle(&sources);
        crate::common::assert_oracle_parity(&ctx, &grid.graph, &sources, &oracle);
    }
}

#[test]
fn test_t3_random_tree_multi_chunk_sweep() {
    let tree = generate_random_tree(80, Some(42)).unwrap();
    let ctx = crate::common::shared_gpu_context();

    for &s_count in &[3, 33, 65] {
        let sources: Vec<u32> = (0..s_count).map(|i| (i * 11) % 80).collect();
        crate::common::assert_cross_engine_parity(&ctx, &tree, &sources);
    }
}

#[test]
fn test_t3_erdos_renyi_batch_cross_chunk() {
    let graph = generate_erdos_renyi_np(70, 0.08, Some(999)).unwrap();
    let ctx = crate::common::shared_gpu_context();

    for &s_count in &[17, 33, 65] {
        let sources: Vec<u32> = (0..s_count).map(|i| (i * 13) % 70).collect();
        crate::common::assert_cross_engine_parity(&ctx, &graph, &sources);
    }
}

#[test]
fn test_t3_graph500_rmat_batch_cross_chunk() {
    let rmat = generate_graph500_rmat(5, 8, Some(101)).unwrap(); // |V| = 32
    let ctx = crate::common::shared_gpu_context();

    for &s_count in &[1, 33, 65] {
        let sources: Vec<u32> = (0..s_count).map(|i| i % 32).collect();
        crate::common::assert_cross_engine_parity(&ctx, &rmat, &sources);
    }
}
