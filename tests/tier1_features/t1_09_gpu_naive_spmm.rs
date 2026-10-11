use webgpu_bfs_core::generators::{
    disconnected_cliques, generate_erdos_renyi_np, generate_random_tree, Grid2D,
};
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t1_f9_01_gpu_naive_sequential_single_source() {
    let grid = Grid2D::new(8, 8).unwrap();
    let oracle = grid.single_source_oracle(0);

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::NaiveSequential).unwrap();
    let dist = engine.run_ms_bfs(&grid.graph, &[0]).unwrap();

    assert_eq!(dist, oracle);
}

#[test]
fn test_t1_f9_02_gpu_naive_sequential_batched() {
    let graph = generate_erdos_renyi_np(50, 0.1, Some(111)).unwrap();
    let sources = [0, 5, 12, 23, 34, 45];

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_dist = engine.run_ms_bfs(&graph, &sources).unwrap();

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).unwrap();

    assert_eq!(gpu_dist, cpu_dist);
}

#[test]
fn test_t1_f9_03_gpu_batched_dense_spmm_multi_source() {
    let grid = Grid2D::new(8, 8).unwrap();
    let sources: Vec<u32> = (0..16).collect();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let gpu_dist = engine.run_ms_bfs(&grid.graph, &sources).unwrap();

    let oracle = grid.ms_bfs_oracle(&sources);
    assert_eq!(gpu_dist, oracle);
}

#[test]
fn test_t1_f9_04_gpu_batched_dense_spmm_unreachable_sentinel() {
    let graph = disconnected_cliques(&[10, 10]);
    let sources = [0, 15]; // one in clique 0, one in clique 1

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let dist = engine.run_ms_bfs(&graph, &sources).unwrap();

    // Source 0 should reach 0..10 in 1 hop, 10..20 should be u32::MAX
    for &d in &dist[10..20] {
        assert_eq!(d, u32::MAX);
    }
    // Source 15 should reach 10..20 in 1 hop, 0..10 should be u32::MAX
    for &d in &dist[20..30] {
        assert_eq!(d, u32::MAX);
    }
}

#[test]
fn test_t1_f9_05_gpu_naive_vs_spmm_pairwise_equivalence() {
    let tree = generate_random_tree(80, Some(222)).unwrap();
    let sources = [0, 10, 20, 30, 40, 50, 60, 70];

    let ctx = crate::common::shared_gpu_context();
    let naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let spmm = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();

    let dist_naive = naive.run_ms_bfs(&tree, &sources).unwrap();
    let dist_spmm = spmm.run_ms_bfs(&tree, &sources).unwrap();

    assert_eq!(dist_naive, dist_spmm);
}
