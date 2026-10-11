use webgpu_bfs_core::generators::{generate_erdos_renyi_np, Grid2D};
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t3_memory_stride_and_source_major_isolation() {
    let grid = Grid2D::new(8, 8).unwrap();
    let sources = [0, 10, 25, 33, 50, 63];
    let nv = 64;

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel32).unwrap();
    let dist = gpu.run_ms_bfs(&grid.graph, &sources).unwrap();

    assert_eq!(dist.len(), sources.len() * nv);
    for (s_idx, &src) in sources.iter().enumerate() {
        assert_eq!(dist[s_idx * nv + src as usize], 0);
    }
}

#[test]
fn test_t3_duplicate_source_row_idempotence() {
    let graph = generate_erdos_renyi_np(60, 0.1, Some(555)).unwrap();
    let sources = [12, 45, 12, 12, 45, 7];
    let nv = 60;

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::BitParallel64).unwrap();
    let dist = gpu.run_ms_bfs(&graph, &sources).unwrap();

    // Rows 0, 2, 3 must be bitwise identical (source 12)
    assert_eq!(&dist[..nv], &dist[2 * nv..3 * nv]);
    assert_eq!(&dist[..nv], &dist[3 * nv..4 * nv]);

    // Rows 1 and 4 must be bitwise identical (source 45)
    assert_eq!(&dist[nv..2 * nv], &dist[4 * nv..5 * nv]);
}

#[test]
fn test_t3_single_source_vs_batched_row_equivalence() {
    let grid = Grid2D::new(7, 7).unwrap(); // |V| = 49
    let sources: Vec<u32> = (0..65).map(|i| (i * 3) % 49).collect();
    let nv = 49;

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::BitParallel64).unwrap();
    let batched_dist = gpu.run_ms_bfs(&grid.graph, &sources).unwrap();

    // Verify against individual single-source runs
    for (s_idx, &src) in sources.iter().take(10).enumerate() {
        let single_dist = gpu.run_ms_bfs(&grid.graph, &[src]).unwrap();
        let batched_row = &batched_dist[s_idx * nv..(s_idx + 1) * nv];
        assert_eq!(
            batched_row,
            single_dist.as_slice(),
            "Mismatch at source index {}",
            s_idx
        );
    }
}
