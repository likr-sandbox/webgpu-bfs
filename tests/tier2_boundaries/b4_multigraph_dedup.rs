use std::io::Cursor;
use webgpu_bfs_core::graph::CsrGraph;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t2_b4_01_all_vertices_self_loops_to_empty_edges() {
    let edges: Vec<(u32, u32)> = (0..10).map(|i| (i, i)).collect();
    let csr = CsrGraph::from_edge_list(10, &edges);
    assert_eq!(csr.num_vertices, 10);
    assert_eq!(csr.num_edges, 0);
    assert!(csr.validate().is_ok());
}

#[test]
fn test_t2_b4_02_high_multiplicity_duplicate_edges() {
    let edges: Vec<(u32, u32)> = vec![(0, 1); 200];
    let csr = CsrGraph::from_edge_list(2, &edges);
    assert_eq!(csr.num_edges, 1);

    let ctx = crate::common::shared_gpu_context();
    let gpu = GpuBfsEngine::new(ctx, GpuBfsMethod::NaiveSequential).unwrap();
    let dist = gpu.run_ms_bfs(&csr, &[0]).unwrap();
    assert_eq!(dist, vec![0, 1]);
}

#[test]
fn test_t2_b4_03_mixed_loops_reciprocals_and_duplicates() {
    let edges = [
        (0, 0),
        (1, 1),
        (2, 2),
        (0, 1),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (1, 2),
        (0, 2),
        (2, 0),
        (0, 2),
    ];
    let csr = CsrGraph::from_edge_list(3, &edges);
    assert_eq!(csr.num_edges, 6); // complete K3
    assert!(csr.is_symmetric());

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &csr, &[0, 1, 2]);
}

#[test]
fn test_t2_b4_04_mtx_diagonal_self_loop_skipping() {
    let mtx = "\
%%MatrixMarket matrix coordinate pattern general
3 3 5
1 1
2 2
3 3
1 2
2 3
";
    let csr = CsrGraph::from_mtx_reader(Cursor::new(mtx)).unwrap();
    assert_eq!(csr.num_vertices, 3);
    assert_eq!(csr.num_edges, 2);
    assert_eq!(csr.neighbors(0), &[1]);
    assert_eq!(csr.neighbors(1), &[2]);
}

#[test]
fn test_t2_b4_05_random_edge_deduplication() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::new();
    for _ in 0..1000 {
        let u = rng.gen_range(0..20);
        let v = rng.gen_range(0..20);
        edges.push((u, v));
    }

    let csr = CsrGraph::from_edge_list(20, &edges);
    assert!(csr.validate().is_ok());

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &csr, &[0, 5, 10, 15]);
}
