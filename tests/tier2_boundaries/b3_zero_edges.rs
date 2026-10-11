use webgpu_bfs_core::generators::isolated_vertices;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t2_b3_01_isolated_vertices_single_source_sentinels() {
    let graph = isolated_vertices(10);
    let sources = [4];

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::NaiveSequential).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &sources).unwrap();

    assert_eq!(dist.len(), 10);
    assert_eq!(dist[4], 0);
    for (v, &d) in dist.iter().enumerate() {
        if v != 4 {
            assert_eq!(d, u32::MAX);
        }
    }
}

#[test]
fn test_t2_b3_02_isolated_vertices_multi_source_matrix() {
    let graph = isolated_vertices(20);
    let sources = [2, 7, 15];

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &sources).unwrap();

    assert_eq!(dist.len(), 60);
    for (s_idx, &src) in sources.iter().enumerate() {
        for v in 0..20 {
            let d = dist[s_idx * 20 + v];
            if v == src as usize {
                assert_eq!(d, 0);
            } else {
                assert_eq!(d, u32::MAX);
            }
        }
    }
}

#[test]
fn test_t2_b3_03_isolated_vertices_push_pull_no_div0() {
    let graph = isolated_vertices(30);
    let sources = [0, 5, 10];

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &sources).unwrap();

    assert_eq!(dist.len(), 90);
    assert_eq!(dist[0], 0);
    assert_eq!(dist[30 + 5], 0);
    assert_eq!(dist[60 + 10], 0);
}

#[test]
fn test_t2_b3_04_isolated_vertices_bit_parallel_dummy_col() {
    let graph = isolated_vertices(32);
    let sources: Vec<u32> = (0..32).collect();

    let ctx = crate::common::shared_gpu_context();
    let bp32 = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::BitParallel32).unwrap();
    let bp64 = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel64).unwrap();

    let dist32 = bp32.run_ms_bfs(&graph, &sources).unwrap();
    let dist64 = bp64.run_ms_bfs(&graph, &sources).unwrap();

    assert_eq!(dist32, dist64);
}

#[test]
fn test_t2_b3_05_isolated_vertices_dense_spmm_early_break() {
    let graph = isolated_vertices(10);
    let sources = [0, 1];

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BooleanReachability).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &sources).unwrap();

    assert_eq!(dist.len(), 20);
    assert_eq!(dist[0], 0);
    assert_eq!(dist[10 + 1], 0);
}
