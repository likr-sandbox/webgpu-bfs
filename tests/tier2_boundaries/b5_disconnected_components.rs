use webgpu_bfs_core::generators::disconnected_cliques;
use webgpu_bfs_core::graph::CsrGraph;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t2_b5_01_two_disjoint_cliques_isolation() {
    let graph = disconnected_cliques(&[5, 5]);
    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &[1]).unwrap();

    assert_eq!(dist[1], 0);
    for v in [0, 2, 3, 4] {
        assert_eq!(dist[v], 1);
    }
    for &d in &dist[5..10] {
        assert_eq!(d, u32::MAX);
    }
}

#[test]
fn test_t2_b5_02_three_disjoint_paths_cross_component_unreachable() {
    let edges = [
        // Path 1: 0-1-2-3-4
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        // Path 2: 5-6-7-8-9
        (5, 6),
        (6, 5),
        (6, 7),
        (7, 6),
        (7, 8),
        (8, 7),
        (8, 9),
        (9, 8),
        // Path 3: 10-11-12-13-14
        (10, 11),
        (11, 10),
        (11, 12),
        (12, 11),
        (12, 13),
        (13, 12),
        (13, 14),
        (14, 13),
    ];
    let csr = CsrGraph::from_edge_list(15, &edges);
    let sources = [2, 7, 12];

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &csr, &sources);
}

#[test]
fn test_t2_b5_03_singleton_island_beside_connected_component() {
    let edges = [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 0),
        (0, 3),
    ];
    // Vertex 4 has 0 edges
    let csr = CsrGraph::from_edge_list(5, &edges);

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel32).unwrap();
    let dist = gpu.run_ms_bfs(&csr, &[4, 0]).unwrap();

    // From 4: only 4 is reachable
    assert_eq!(dist[0..5], [u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0]);
    // From 0: 0..4 are reachable, 4 is unreachable
    assert_eq!(dist[5..10], [0, 1, 2, 1, u32::MAX]);
}

#[test]
fn test_t2_b5_04_multi_source_distributed_across_partitions() {
    let graph = disconnected_cliques(&[8, 8, 8, 8]);
    let sources: Vec<u32> = (0..16).map(|i| (i * 2) % 32).collect();

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &graph, &sources);
}

#[test]
fn test_t2_b5_05_disconnected_graph_push_pull_convergence() {
    let graph = disconnected_cliques(&[10, 10]);
    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    let dist = engine.run_ms_bfs(&graph, &[0, 15]).unwrap();

    assert_eq!(dist[0], 0);
    assert_eq!(dist[10], u32::MAX);
    assert_eq!(dist[35], 0);
    assert_eq!(dist[20], u32::MAX);
}
