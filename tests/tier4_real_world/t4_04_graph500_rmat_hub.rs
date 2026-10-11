use webgpu_bfs_core::generators::generate_graph500_rmat;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t4_04_graph500_rmat_scale9_hub_contention() {
    // Scale 9 -> 512 vertices, ~8000 directed edges
    let rmat = generate_graph500_rmat(9, 16, Some(20261010)).expect("Failed to build R-MAT graph");
    assert_eq!(rmat.num_vertices, 512);

    let hub_deg = rmat.degree(0);
    assert!(
        hub_deg > 50,
        "Hub 0 degree should exceed 50, got {}",
        hub_deg
    );

    // 32 sources including hub 0, high-degree hubs, and distributed nodes
    let mut sources = vec![0u32];
    for i in 1..32u32 {
        sources.push((i * 13) % 512);
    }

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&rmat, &sources).unwrap();

    let ctx = crate::common::shared_gpu_context();
    let methods = [
        GpuBfsMethod::PushPullAdaptive,
        GpuBfsMethod::BitParallel64,
        GpuBfsMethod::BooleanReachability,
        GpuBfsMethod::BitParallel32,
    ];

    for &method in &methods {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let gpu_dist = gpu.run_ms_bfs(&rmat, &sources).unwrap();

        assert_eq!(
            gpu_dist, cpu_dist,
            "Method {:?} failed on Scenario 4 Graph500 R-MAT",
            method
        );
    }
}
