use webgpu_bfs_core::generators::{
    generate_graph500_rmat, generate_rmat, GRAPH500_A, GRAPH500_B, GRAPH500_C, GRAPH500_D,
};
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;

#[test]
fn test_t1_f7_01_rmat_graph500_hub_degree_distribution() {
    let graph = generate_graph500_rmat(7, 16, Some(42)).expect("R-MAT generation failed");
    assert_eq!(graph.num_vertices, 128);
    assert!(graph.validate().is_ok());

    let hub_deg = graph.degree(0);
    let avg_deg = graph.avg_degree();
    assert!(
        hub_deg as f64 > 2.0 * avg_deg,
        "Hub 0 degree ({}) should be significantly larger than avg degree ({})",
        hub_deg,
        avg_deg
    );
}

#[test]
fn test_t1_f7_02_rmat_scale_sweep_cardinality() {
    for scale in [4, 5, 6] {
        let graph = generate_graph500_rmat(scale, 8, Some(100 + scale as u64)).unwrap();
        assert_eq!(graph.num_vertices, 1 << scale);
        assert!(graph.validate().is_ok());
    }
}

#[test]
fn test_t1_f7_03_rmat_directed_vs_undirected_transposition() {
    let directed = generate_rmat(
        5,
        8,
        GRAPH500_A,
        GRAPH500_B,
        GRAPH500_C,
        GRAPH500_D,
        Some(123),
        true,
    )
    .unwrap();
    let undirected = generate_rmat(
        5,
        8,
        GRAPH500_A,
        GRAPH500_B,
        GRAPH500_C,
        GRAPH500_D,
        Some(123),
        false,
    )
    .unwrap();

    assert!(
        !directed.is_symmetric(),
        "Directed R-MAT should be asymmetric"
    );
    assert!(
        undirected.is_symmetric(),
        "Undirected R-MAT should be symmetric"
    );
}

#[test]
fn test_t1_f7_04_rmat_hub_wavefront_propagation() {
    let graph = generate_graph500_rmat(6, 12, Some(999)).unwrap();
    let cpu = CpuBfsEngine::new();
    let dist = cpu.run_ms_bfs(&graph, &[0]).unwrap();

    let reachable_count = dist.iter().filter(|&&d| d != u32::MAX).count();
    let max_dist = dist
        .iter()
        .filter(|&&d| d != u32::MAX)
        .max()
        .copied()
        .unwrap_or(0);

    // R-MAT graphs have very low diameter
    assert!(
        max_dist <= 6,
        "R-MAT graph diameter should be small (<=6), got {}",
        max_dist
    );
    assert!(
        reachable_count > graph.num_vertices / 2,
        "Hub should reach majority of connected component"
    );
}

#[test]
fn test_t1_f7_05_rmat_all_gpu_cpu_bitwise_equivalence() {
    let graph = generate_graph500_rmat(6, 8, Some(456)).unwrap();
    let sources: Vec<u32> = (0..16).collect();

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &graph, &sources);
}
