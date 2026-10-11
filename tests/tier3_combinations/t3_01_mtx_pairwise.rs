use std::io::Cursor;
use webgpu_bfs_core::graph::CsrGraph;

#[test]
fn test_t3_mtx_symmetric_pattern_all_methods() {
    let mtx = "\
%%MatrixMarket matrix coordinate pattern symmetric
% Undirected 4-cycle with diagonal cross links
4 4 5
1 2
2 3
3 4
4 1
1 3
";
    let csr = CsrGraph::from_mtx_reader(Cursor::new(mtx)).unwrap();
    assert_eq!(csr.num_vertices, 4);
    assert_eq!(csr.num_edges, 10);
    assert!(csr.is_symmetric());

    let ctx = crate::common::shared_gpu_context();
    let sources = [0, 1, 2, 3];
    crate::common::assert_cross_engine_parity(&ctx, &csr, &sources);
}

#[test]
fn test_t3_mtx_directed_asymmetric_dag_all_methods() {
    let mtx = "\
%%MatrixMarket matrix coordinate integer general
% Directed DAG: 1 -> 2 -> 3 -> 4, and 1 -> 4
4 4 4
1 2 1
2 3 1
3 4 1
1 4 1
";
    let csr = CsrGraph::from_mtx_reader(Cursor::new(mtx)).unwrap();
    assert_eq!(csr.num_vertices, 4);
    assert_eq!(csr.num_edges, 4);
    assert!(!csr.is_symmetric());

    let ctx = crate::common::shared_gpu_context();
    let sources = [0, 3];
    crate::common::assert_cross_engine_parity(&ctx, &csr, &sources);
}

#[test]
fn test_t3_mtx_comments_and_whitespace_all_methods() {
    let mtx = "\
%%MatrixMarket matrix coordinate pattern symmetric
% Post-banner comments
% Interleaved comments
5 5 4
1 2
2 3
3 4
4 5
";
    let csr = CsrGraph::from_mtx_reader(Cursor::new(mtx)).unwrap();
    assert_eq!(csr.num_vertices, 5);
    assert!(csr.is_symmetric());

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &csr, &[0, 2, 4]);
}
