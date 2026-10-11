use webgpu_bfs_core::generators::{empty_graph, Grid2D};
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t2_b1_01_empty_graph_empty_sources() {
    let graph = empty_graph();
    let cpu = CpuBfsEngine::new();
    let dist = cpu.run_ms_bfs(&graph, &[]).unwrap();
    assert_eq!(dist, Vec::<u32>::new());

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_dist = gpu.run_ms_bfs(&graph, &[]).unwrap();
    assert_eq!(gpu_dist, Vec::<u32>::new());
}

#[test]
fn test_t2_b1_02_empty_graph_nonempty_sources_error() {
    let graph = empty_graph();
    let ctx = crate::common::shared_gpu_context();
    for &method in &crate::common::ALL_GPU_METHODS {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert!(
            gpu.run_ms_bfs(&graph, &[0]).is_err(),
            "Method {:?} must return Err for source on empty graph",
            method
        );
    }
}

#[test]
fn test_t2_b1_03_nonempty_graph_empty_sources() {
    let grid = Grid2D::new(5, 5).unwrap();
    let cpu = CpuBfsEngine::new();
    let dist = cpu.run_ms_bfs(&grid.graph, &[]).unwrap();
    assert_eq!(dist, Vec::<u32>::new());

    let ctx = crate::common::shared_gpu_context();
    for &method in &crate::common::ALL_GPU_METHODS {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let gpu_dist = gpu.run_ms_bfs(&grid.graph, &[]).unwrap();
        assert_eq!(gpu_dist, Vec::<u32>::new());
    }
}

#[test]
fn test_t2_b1_04_out_of_bounds_single_source_error() {
    let grid = Grid2D::new(3, 3).unwrap(); // |V| = 9
    let cpu = CpuBfsEngine::new();
    assert!(cpu.run_ms_bfs(&grid.graph, &[9]).is_err());

    let ctx = crate::common::shared_gpu_context();
    for &method in &crate::common::ALL_GPU_METHODS {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert!(gpu.run_ms_bfs(&grid.graph, &[9]).is_err());
    }
}

#[test]
fn test_t2_b1_05_out_of_bounds_in_mixed_batch_error() {
    let grid = Grid2D::new(4, 5).unwrap(); // |V| = 20
    let sources = [0, 5, 19, 20, 1];
    let cpu = CpuBfsEngine::new();
    assert!(cpu.run_ms_bfs(&grid.graph, &sources).is_err());

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    assert!(gpu.run_ms_bfs(&grid.graph, &sources).is_err());
}
