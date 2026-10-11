use webgpu_bfs_core::generators::{
    generate_erdos_renyi_np, generate_graph500_rmat, path_graph, Grid2D,
};
use webgpu_bfs_core::graph::CsrGraph;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t1_f12_01_gpu_push_pull_direction_switch_trigger() {
    let rmat = generate_graph500_rmat(7, 16, Some(42)).unwrap();
    let sources: Vec<u32> = (0..32).collect();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    let gpu_dist = engine.run_ms_bfs(&rmat, &sources).unwrap();

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&rmat, &sources).unwrap();
    assert_eq!(gpu_dist, cpu_dist);
}

#[test]
fn test_t1_f12_02_gpu_push_pull_pathological_switch_avoidance() {
    let chain = path_graph(100);
    let sources = [0, 99];

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    let gpu_dist = engine.run_ms_bfs(&chain, &sources).unwrap();

    // In a path graph, frontier size is <= 2 at every step, so it stays in Push mode
    for v in 0..100u32 {
        assert_eq!(gpu_dist[v as usize], v);
        assert_eq!(gpu_dist[100 + v as usize], 99 - v);
    }
}

#[test]
fn test_t1_f12_03_all_six_gpu_methods_complete_unification() {
    let grid = Grid2D::new(8, 8).unwrap();
    let sources: Vec<u32> = (0..32).collect();

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &grid.graph, &sources);
}

#[test]
fn test_t1_f12_04_gpu_push_pull_symmetric_buffer_sharing() {
    let er = generate_erdos_renyi_np(60, 0.2, Some(101)).unwrap();
    assert!(er.is_symmetric());
    let sources: Vec<u32> = (0..16).collect();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    let gpu_dist = engine.run_ms_bfs(&er, &sources).unwrap();

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&er, &sources).unwrap();
    assert_eq!(gpu_dist, cpu_dist);
}

#[test]
fn test_t1_f12_05_gpu_push_pull_directed_asymmetric() {
    let edges = [(0, 1), (1, 2), (2, 3), (0, 3)];
    let csr = CsrGraph::from_edge_list(4, &edges);
    assert!(!csr.is_symmetric());

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    let dist = engine.run_ms_bfs(&csr, &[0, 3]).unwrap();

    // From 0: 0->0, 1->1, 2->2, 3->1
    assert_eq!(&dist[0..4], &[0, 1, 2, 1]);
    // From 3: only 3 is reachable, others are u32::MAX
    assert_eq!(&dist[4..8], &[u32::MAX, u32::MAX, u32::MAX, 0]);
}
