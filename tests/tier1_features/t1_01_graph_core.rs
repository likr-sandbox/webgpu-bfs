use webgpu_bfs_core::CsrGraph;

#[test]
fn test_t1_f1_01_csr_canonical_construction() {
    let edges = [(3, 4), (0, 2), (0, 1), (2, 3), (1, 2), (3, 0)];
    let csr = CsrGraph::from_edge_list(5, &edges);

    assert!(csr.validate().is_ok(), "Canonical CSR must pass validation");
    assert_eq!(csr.num_vertices(), 5);
    assert_eq!(csr.num_edges(), 6);
    assert_eq!(csr.row_offsets, vec![0, 2, 3, 4, 6, 6]);
    assert_eq!(csr.col_indices, vec![1, 2, 2, 3, 0, 4]);
}

#[test]
fn test_t1_f1_02_csr_self_loop_filtering() {
    let edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (3, 3)];
    let csr = CsrGraph::from_edge_list(4, &edges);

    assert!(csr.validate().is_ok());
    assert_eq!(csr.num_edges(), 2);
    assert_eq!(csr.neighbors(0), &[1]);
    assert_eq!(csr.neighbors(1), &[2]);
    assert_eq!(csr.neighbors(2), &[]);
    assert_eq!(csr.neighbors(3), &[]);
}

#[test]
fn test_t1_f1_03_csr_duplicate_edge_dedup() {
    let edges = [(0, 1), (0, 1), (0, 1), (1, 2), (1, 2)];
    let csr = CsrGraph::from_edge_list(3, &edges);

    assert_eq!(csr.num_edges(), 2);
    assert_eq!(csr.neighbors(0), &[1]);
    assert_eq!(csr.neighbors(1), &[2]);
    assert_eq!(csr.neighbors(2), &[]);
}

#[test]
fn test_t1_f1_04_csr_degree_queries_and_invariants() {
    let edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)];
    let csr = CsrGraph::from_edge_list(6, &edges);

    assert_eq!(csr.degree(0), 5);
    for u in 1..=5 {
        assert_eq!(csr.degree(u), 0);
    }
    assert_eq!(csr.max_degree(), 5);
    assert!((csr.avg_degree() - (5.0 / 6.0)).abs() < 1e-6);
    assert_eq!(csr.get_degree(5), Some(0));
    assert_eq!(csr.get_degree(6), None);
}

#[test]
fn test_t1_f1_05_csr_neighbor_slice_ordering() {
    let mut edges = Vec::new();
    for v in (1..=20).rev() {
        edges.push((0, v));
    }
    let csr = CsrGraph::from_edge_list(21, &edges);

    let neighbors = csr.neighbors(0);
    assert_eq!(neighbors.len(), 20);
    for w in neighbors.windows(2) {
        assert!(
            w[0] < w[1],
            "Neighbors must be strictly ascending: {} < {}",
            w[0],
            w[1]
        );
    }
}
