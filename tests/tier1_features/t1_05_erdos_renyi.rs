use webgpu_bfs_core::generators::{generate_erdos_renyi_nm, generate_erdos_renyi_np};

#[test]
fn test_t1_f5_01_erdos_renyi_np_sparse_geometric_skipping() {
    let graph = generate_erdos_renyi_np(100, 0.05, Some(42)).expect("ER generation failed");
    assert_eq!(graph.num_vertices, 100);
    assert!(graph.validate().is_ok());
    assert!(graph.is_symmetric());
    let avg = graph.avg_degree();
    assert!(
        avg > 2.0 && avg < 8.0,
        "Average degree should be around 5.0, got {}",
        avg
    );
}

#[test]
fn test_t1_f5_02_erdos_renyi_np_dense_bernoulli() {
    let graph = generate_erdos_renyi_np(60, 0.7, Some(101)).expect("ER generation failed");
    assert_eq!(graph.num_vertices, 60);
    assert!(graph.validate().is_ok());
    assert!(graph.is_symmetric());
    let avg = graph.avg_degree();
    assert!(
        avg > 35.0 && avg < 48.0,
        "Average degree should be around 41.3, got {}",
        avg
    );
}

#[test]
fn test_t1_f5_03_erdos_renyi_nm_exact_cardinality() {
    let graph = generate_erdos_renyi_nm(50, 120, Some(999)).expect("ER nm failed");
    assert_eq!(graph.num_vertices, 50);
    assert_eq!(graph.num_edges, 240); // 2 * 120
    assert!(graph.validate().is_ok());
    assert!(graph.is_symmetric());
}

#[test]
fn test_t1_f5_04_erdos_renyi_prng_seed_reproducibility() {
    let g1 = generate_erdos_renyi_np(64, 0.1, Some(12345)).unwrap();
    let g2 = generate_erdos_renyi_np(64, 0.1, Some(12345)).unwrap();
    let g3 = generate_erdos_renyi_np(64, 0.1, Some(54321)).unwrap();

    assert_eq!(g1.row_offsets, g2.row_offsets);
    assert_eq!(g1.col_indices, g2.col_indices);
    assert_ne!(g1.col_indices, g3.col_indices);
}

#[test]
fn test_t1_f5_05_erdos_renyi_all_engines_bitwise_equivalence() {
    let graph = generate_erdos_renyi_np(80, 0.1, Some(777)).unwrap();
    let sources: Vec<u32> = (0..16).collect();

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &graph, &sources);
}
