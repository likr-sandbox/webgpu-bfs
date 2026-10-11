use webgpu_bfs_core::generators::{generate_erdos_renyi_np, generate_graph500_rmat, Grid2D};
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t1_f10_01_gpu_bp32_exact_word_match() {
    let graph = generate_erdos_renyi_np(100, 0.08, Some(101)).unwrap();
    let sources: Vec<u32> = (0..32).collect();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel32).unwrap();
    let gpu_dist = engine.run_ms_bfs(&graph, &sources).unwrap();

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).unwrap();
    assert_eq!(gpu_dist, cpu_dist);
}

#[test]
fn test_t1_f10_02_gpu_bp64_interleaved_vec2_match() {
    let graph = generate_erdos_renyi_np(100, 0.08, Some(202)).unwrap();
    let sources: Vec<u32> = (0..64).collect();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel64).unwrap();
    let gpu_dist = engine.run_ms_bfs(&graph, &sources).unwrap();

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).unwrap();
    assert_eq!(gpu_dist, cpu_dist);
}

#[test]
fn test_t1_f10_03_gpu_bp32_multi_chunk_arbitrary_batch() {
    let grid = Grid2D::new(8, 8).unwrap();
    // 75 sources sampled from 64 vertices with wraparound
    let sources: Vec<u32> = (0..75).map(|i| i % 64).collect();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel32).unwrap();
    let gpu_dist = engine.run_ms_bfs(&grid.graph, &sources).unwrap();

    let oracle = grid.ms_bfs_oracle(&sources);
    assert_eq!(gpu_dist, oracle);
}

#[test]
fn test_t1_f10_04_gpu_bp64_multi_chunk_arbitrary_batch() {
    let grid = Grid2D::new(8, 8).unwrap();
    let sources: Vec<u32> = (0..70).map(|i| (i * 3) % 64).collect();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel64).unwrap();
    let gpu_dist = engine.run_ms_bfs(&grid.graph, &sources).unwrap();

    let oracle = grid.ms_bfs_oracle(&sources);
    assert_eq!(gpu_dist, oracle);
}

#[test]
fn test_t1_f10_05_gpu_bp32_vs_bp64_bitwise_parity() {
    let rmat = generate_graph500_rmat(6, 8, Some(303)).unwrap();
    let sources: Vec<u32> = (0..64).collect();

    let ctx = crate::common::shared_gpu_context();
    let bp32 = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::BitParallel32).unwrap();
    let bp64 = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel64).unwrap();

    let dist_32 = bp32.run_ms_bfs(&rmat, &sources).unwrap();
    let dist_64 = bp64.run_ms_bfs(&rmat, &sources).unwrap();
    assert_eq!(dist_32, dist_64);
}
