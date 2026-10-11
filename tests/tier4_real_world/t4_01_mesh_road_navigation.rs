use webgpu_bfs_core::generators::Grid2D;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_gpu::GpuBfsEngine;

#[test]
fn test_t4_01_mesh_road_navigation_manhattan_oracle() {
    let width = 25usize;
    let height = 20usize;
    let grid = Grid2D::new(width, height).expect("Failed to build 25x20 Grid2D");
    let nv = width * height;
    assert_eq!(nv, 500);

    // 64 sources spanning corners, center, and pseudo-random distributed waypoints
    let mut sources = vec![
        grid.vertex_id(0, 0),
        grid.vertex_id(width - 1, 0),
        grid.vertex_id(0, height - 1),
        grid.vertex_id(width - 1, height - 1),
        grid.vertex_id(width / 2, height / 2),
    ];
    for i in 5..64 {
        sources.push(((i * 7 + 13) % nv) as u32);
    }

    let oracle = grid.ms_bfs_oracle(&sources);

    let ctx = crate::common::shared_gpu_context();
    for &method in &crate::common::ALL_GPU_METHODS {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let dist = gpu.run_ms_bfs(&grid.graph, &sources).unwrap();

        assert_eq!(dist.len(), 64 * nv);
        assert_eq!(
            dist, oracle,
            "Method {:?} failed on Scenario 1 Mesh Road Grid",
            method
        );
    }

    // Diameter verification: from corner (0, 0) to opposite corner (width-1, height-1)
    let s0_opp = oracle[(width - 1) + (height - 1) * width];
    let expected_diam = (width - 1 + height - 1) as u32;
    assert_eq!(
        s0_opp, expected_diam,
        "Grid diameter must be exactly {}",
        expected_diam
    );
}
