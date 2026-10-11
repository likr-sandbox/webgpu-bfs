use webgpu_bfs_core::generators::{clique_graph, star_graph};
use webgpu_bfs_core::graph::CsrGraph;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t2_b8_01_star_graph_center_hub_broadcast() {
    let star = star_graph(101);
    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::NaiveSequential).unwrap();
    let dist = gpu.run_ms_bfs(&star, &[0]).unwrap();

    assert_eq!(dist[0], 0);
    for &d in &dist[1..=100] {
        assert_eq!(d, 1);
    }
}

#[test]
fn test_t2_b8_02_star_graph_leaf_source_hub_forwarding() {
    let star = star_graph(101);
    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel32).unwrap();
    let dist = gpu.run_ms_bfs(&star, &[5]).unwrap();

    assert_eq!(dist[5], 0);
    assert_eq!(dist[0], 1);
    for (v, &d) in dist.iter().enumerate().take(101).skip(1) {
        if v != 5 {
            assert_eq!(d, 2);
        }
    }
}

#[test]
fn test_t2_b8_03_star_graph_all_leaf_sources_hub_atomic_collision() {
    let star = star_graph(101);
    let sources: Vec<u32> = (1..=32).collect(); // 32 distinct leaves

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &star, &sources);
}

#[test]
fn test_t2_b8_04_complete_clique_k50_extreme_edge_contention() {
    let clique = clique_graph(50);
    let sources: Vec<u32> = (0..16).collect();

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &clique, &sources);
}

#[test]
fn test_t2_b8_05_double_star_inter_hub_bridge_switching() {
    // Star 1: hub 0 with leaves 1..20
    // Star 2: hub 21 with leaves 22..40
    // Bridge: 0 <-> 21
    let mut edges = Vec::new();
    for leaf in 1..=20 {
        edges.push((0, leaf));
        edges.push((leaf, 0));
    }
    for leaf in 22..=40 {
        edges.push((21, leaf));
        edges.push((leaf, 21));
    }
    edges.push((0, 21));
    edges.push((21, 0));

    let csr = CsrGraph::from_edge_list(41, &edges);
    let sources = [1, 25]; // leaf in star 1 and leaf in star 2

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &csr, &sources);
}
