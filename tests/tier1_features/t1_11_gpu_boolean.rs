use webgpu_bfs_core::generators::{
    disconnected_cliques, generate_erdos_renyi_np, generate_random_tree, Grid2D,
};
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t1_f11_01_gpu_boolean_reachability_pipeline_parity() {
    let grid = Grid2D::new(10, 10).unwrap();
    let sources: Vec<u32> = (0..32).collect();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BooleanReachability).unwrap();
    let gpu_dist = engine.run_ms_bfs(&grid.graph, &sources).unwrap();

    let oracle = grid.ms_bfs_oracle(&sources);
    assert_eq!(gpu_dist, oracle);
}

#[test]
fn test_t1_f11_02_gpu_boolean_reachability_chunk_offset_writes() {
    let graph = generate_erdos_renyi_np(80, 0.08, Some(404)).unwrap();
    let sources: Vec<u32> = (0..48).collect(); // 32 + 16 (2 chunks)

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BooleanReachability).unwrap();
    let gpu_dist = engine.run_ms_bfs(&graph, &sources).unwrap();

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).unwrap();
    assert_eq!(gpu_dist, cpu_dist);
}

#[test]
fn test_t1_f11_03_gpu_boolean_disconnected_sentinels() {
    let graph = disconnected_cliques(&[15, 15]);
    let sources = [2, 18];

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BooleanReachability).unwrap();
    let dist = engine.run_ms_bfs(&graph, &sources).unwrap();

    // Source 2 (in clique 0) cannot reach clique 1 (15..30)
    for &d in &dist[15..30] {
        assert_eq!(d, u32::MAX);
    }
    // Source 18 (in clique 1) cannot reach clique 0 (0..15)
    for &d in &dist[30..45] {
        assert_eq!(d, u32::MAX);
    }
}

#[test]
fn test_t1_f11_04_gpu_boolean_single_source() {
    let grid = Grid2D::new(6, 6).unwrap();
    let sources = [14];

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BooleanReachability).unwrap();
    let gpu_dist = engine.run_ms_bfs(&grid.graph, &sources).unwrap();

    let oracle = grid.single_source_oracle(14);
    assert_eq!(gpu_dist, oracle);
}

#[test]
fn test_t1_f11_05_gpu_boolean_vs_cpu_bit_parallel() {
    let tree = generate_random_tree(100, Some(505)).unwrap();
    let sources: Vec<u32> = (0..32).collect();

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BooleanReachability).unwrap();
    let cpu = CpuBfsEngine::bit_parallel();

    let dist_gpu = gpu.run_ms_bfs(&tree, &sources).unwrap();
    let dist_cpu = cpu.run_ms_bfs(&tree, &sources).unwrap();
    assert_eq!(dist_gpu, dist_cpu);
}
