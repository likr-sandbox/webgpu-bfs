use webgpu_bfs_core::generators::{cycle_graph, path_graph};
use webgpu_bfs_core::graph::CsrGraph;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t2_b7_01_linear_chain_depth_50() {
    let chain = path_graph(50);
    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::NaiveSequential).unwrap();
    let dist = gpu.run_ms_bfs(&chain, &[0]).unwrap();

    for v in 0..50u32 {
        assert_eq!(dist[v as usize], v);
    }
}

#[test]
fn test_t2_b7_02_linear_chain_depth_120() {
    let chain = path_graph(120);
    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel64).unwrap();
    let dist = gpu.run_ms_bfs(&chain, &[0]).unwrap();

    for v in 0..120u32 {
        assert_eq!(dist[v as usize], v);
    }
}

#[test]
fn test_t2_b7_03_linear_chain_bidirectional_endpoint_sources() {
    let chain = path_graph(60);
    let sources = [0, 59];

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &chain, &sources);
}

#[test]
fn test_t2_b7_04_cycle_graph_circular_collision() {
    let cycle = cycle_graph(60);
    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BooleanReachability).unwrap();
    let dist = gpu.run_ms_bfs(&cycle, &[0]).unwrap();

    for v in 0..60u32 {
        let expected = v.min(60 - v);
        assert_eq!(dist[v as usize], expected);
    }
}

#[test]
fn test_t2_b7_05_hairpin_snake_narrow_wavefront() {
    // Generate a snake path: 0 -> 1 -> ... -> 9, then 9 -> 19 -> 18 -> ... -> 10, then 10 -> 20 -> ...
    let mut edges = Vec::new();
    let w = 8usize;
    let h = 8usize;
    for y in 0..h {
        let is_even = y % 2 == 0;
        for x in 0..(w - 1) {
            let (u, v) = if is_even {
                (y * w + x, y * w + x + 1)
            } else {
                (y * w + (w - 1 - x), y * w + (w - 2 - x))
            };
            edges.push((u as u32, v as u32));
            edges.push((v as u32, u as u32));
        }
        if y + 1 < h {
            let u = if is_even { (y + 1) * w - 1 } else { y * w };
            let v = if is_even {
                (y + 2) * w - 1
            } else {
                (y + 1) * w
            };
            edges.push((u as u32, v as u32));
            edges.push((v as u32, u as u32));
        }
    }

    let csr = CsrGraph::from_edge_list(w * h, &edges);
    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &csr, &[0]);
}
