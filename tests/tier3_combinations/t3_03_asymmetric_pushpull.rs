use webgpu_bfs_core::generators::generate_erdos_renyi_np;
use webgpu_bfs_core::graph::CsrGraph;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t3_symmetric_dense_pushpull_transition() {
    let graph = generate_erdos_renyi_np(80, 0.20, Some(1010)).unwrap();
    assert!(graph.is_symmetric());
    let sources: Vec<u32> = (0..32).collect();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    let gpu_dist = engine.run_ms_bfs(&graph, &sources).unwrap();

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).unwrap();
    assert_eq!(gpu_dist, cpu_dist);
}

#[test]
fn test_t3_directed_asymmetric_csc_pushpull() {
    let mut edges = Vec::new();
    // Feedforward DAG: u -> v for u < v
    for u in 0..50u32 {
        for v in (u + 1)..50u32 {
            if (u + v) % 5 == 0 {
                edges.push((u, v));
            }
        }
    }

    let graph = CsrGraph::from_edge_list(50, &edges);
    assert!(!graph.is_symmetric());

    let sources = [0, 10, 20, 30, 40, 49];
    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    let gpu_dist = engine.run_ms_bfs(&graph, &sources).unwrap();

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).unwrap();
    assert_eq!(gpu_dist, cpu_dist);

    // Source 49 has out-degree 0, can only reach itself
    let s49_row = &gpu_dist[5 * 50..6 * 50];
    for (v, &d) in s49_row.iter().enumerate().take(49) {
        assert_eq!(d, u32::MAX, "Node {} should be unreachable backwards", v);
    }
    assert_eq!(s49_row[49], 0);
}

#[test]
fn test_t3_pushpull_adaptive_vs_pure_methods() {
    let graph = generate_erdos_renyi_np(60, 0.15, Some(2020)).unwrap();
    let sources: Vec<u32> = (0..32).collect();

    let ctx = crate::common::shared_gpu_context();
    let pp = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::PushPullAdaptive).unwrap();
    let bp32 = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::BitParallel32).unwrap();
    let bp64 = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::BitParallel64).unwrap();
    let bool_reach = GpuBfsEngine::new(ctx, GpuBfsMethod::BooleanReachability).unwrap();

    let d_pp = pp.run_ms_bfs(&graph, &sources).unwrap();
    let d_bp32 = bp32.run_ms_bfs(&graph, &sources).unwrap();
    let d_bp64 = bp64.run_ms_bfs(&graph, &sources).unwrap();
    let d_bool = bool_reach.run_ms_bfs(&graph, &sources).unwrap();

    assert_eq!(d_pp, d_bp32);
    assert_eq!(d_pp, d_bp64);
    assert_eq!(d_pp, d_bool);
}
