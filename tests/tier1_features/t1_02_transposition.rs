use webgpu_bfs_core::CsrGraph;

#[test]
fn test_t1_f2_01_csc_counting_sort_transposition() {
    let edges = [(0, 1), (0, 2), (1, 2), (3, 2)];
    let csr = CsrGraph::from_edge_list(4, &edges);
    let csc = csr.to_csc();

    assert!(csc.validate().is_ok());
    assert_eq!(csc.in_degree(2), 3);
    assert_eq!(csc.predecessors(2), &[0, 1, 3]);
}

#[test]
fn test_t1_f2_02_csr_csc_roundtrip_isomorphism() {
    let edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 4),
        (2, 5),
        (3, 6),
        (4, 6),
        (4, 7),
        (5, 7),
        (6, 8),
        (7, 8),
        (0, 5),
        (1, 7),
        (2, 8),
        (3, 9),
        (4, 9),
        (5, 9),
    ];
    let csr = CsrGraph::from_edge_list(10, &edges);
    let csc = csr.to_csc();
    let csr_roundtrip = csc.to_csr();

    assert_eq!(csr.row_offsets, csr_roundtrip.row_offsets);
    assert_eq!(csr.col_indices, csr_roundtrip.col_indices);
    assert_eq!(csr, csr_roundtrip);
}

#[test]
fn test_t1_f2_03_symmetric_graph_detection() {
    let edges = [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        (4, 0),
        (0, 4),
    ];
    let csr = CsrGraph::from_edge_list(5, &edges);
    assert!(csr.is_symmetric(), "Undirected 5-cycle must be symmetric");
}

#[test]
fn test_t1_f2_04_asymmetric_graph_rejection() {
    let edges = [(0, 1), (1, 2), (2, 3)];
    let csr = CsrGraph::from_edge_list(4, &edges);
    assert!(!csr.is_symmetric(), "Directed chain must not be symmetric");
}

#[test]
fn test_t1_f2_05_in_degree_and_predecessor_integrity() {
    let mut edges = Vec::new();
    for u in 0..50 {
        edges.push((u, 50));
    }
    let csr = CsrGraph::from_edge_list(51, &edges);
    let csc = csr.to_csc();

    assert_eq!(csc.in_degree(50), 50);
    assert_eq!(csc.max_in_degree(), 50);
    let expected: Vec<u32> = (0..50).collect();
    assert_eq!(csc.predecessors(50), expected.as_slice());
}
