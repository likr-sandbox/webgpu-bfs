use webgpu_bfs_core::generators::single_vertex;
use webgpu_bfs_core::graph::CsrGraph;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t2_b2_01_single_vertex_zero_edges() {
    let graph = single_vertex();
    let cpu = CpuBfsEngine::new();
    let dist = cpu.run_ms_bfs(&graph, &[0]).unwrap();
    assert_eq!(dist, vec![0]);

    let ctx = crate::common::shared_gpu_context();
    for &method in &crate::common::ALL_GPU_METHODS {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let gpu_dist = gpu.run_ms_bfs(&graph, &[0]).unwrap();
        assert_eq!(
            gpu_dist,
            vec![0],
            "Method {:?} failed on single vertex",
            method
        );
    }
}

#[test]
fn test_t2_b2_02_single_vertex_filtered_self_loop() {
    let graph = CsrGraph::from_edge_list(1, &[(0, 0)]);
    assert_eq!(graph.num_vertices, 1);
    assert_eq!(graph.num_edges, 0);

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::NaiveSequential).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &[0]).unwrap();
    assert_eq!(dist, vec![0]);
}

#[test]
fn test_t2_b2_03_single_vertex_duplicate_sources() {
    let graph = single_vertex();
    let sources = [0, 0, 0, 0, 0];

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &sources).unwrap();
    assert_eq!(dist, vec![0, 0, 0, 0, 0]);
}

#[test]
fn test_t2_b2_04_single_vertex_batch_32_packing() {
    let graph = single_vertex();
    let sources = [0u32; 32];

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel32).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &sources).unwrap();
    assert_eq!(dist, vec![0u32; 32]);
}

#[test]
fn test_t2_b2_05_single_vertex_batch_64_packing() {
    let graph = single_vertex();
    let sources = [0u32; 64];

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel64).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &sources).unwrap();
    assert_eq!(dist, vec![0u32; 64]);
}
