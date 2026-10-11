use std::io::Cursor;
use webgpu_bfs_core::CsrGraph;

#[test]
fn test_t1_f3_01_mtx_coordinate_pattern_symmetric() {
    let mtx = "\
%%MatrixMarket matrix coordinate pattern symmetric
% Synthetic 4-cycle graph
4 4 4
1 2
2 3
3 4
4 1
";
    let csr = CsrGraph::from_mtx_reader(Cursor::new(mtx)).expect("MTX parsing failed");
    assert_eq!(csr.num_vertices(), 4);
    assert_eq!(csr.num_edges(), 8);
    assert!(csr.is_symmetric());
}

#[test]
fn test_t1_f3_02_mtx_coordinate_real_general() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3 2
1 2 3.5
2 3 1.2
";
    let csr = CsrGraph::from_mtx_reader(Cursor::new(mtx)).expect("MTX parsing failed");
    assert_eq!(csr.num_vertices(), 3);
    assert_eq!(csr.num_edges(), 2);
    assert!(!csr.is_symmetric());
    assert_eq!(csr.neighbors(0), &[1]);
    assert_eq!(csr.neighbors(1), &[2]);
    assert_eq!(csr.neighbors(2), &[]);
}

#[test]
fn test_t1_f3_03_mtx_comment_and_whitespace_handling() {
    let mtx = "\
%%MatrixMarket matrix coordinate pattern general
% Post-banner comment 1
% Post-banner comment 2
% Another comment line
3   3   2

  1   2   
   2   3   
";
    let csr = CsrGraph::from_mtx_reader(Cursor::new(mtx)).expect("MTX parsing failed");
    assert_eq!(csr.num_vertices(), 3);
    assert_eq!(csr.num_edges(), 2);
    assert_eq!(csr.neighbors(0), &[1]);
    assert_eq!(csr.neighbors(1), &[2]);
}

#[test]
fn test_t1_f3_04_mtx_error_handling_and_validation() {
    // 1. Invalid banner
    let invalid_banner = "Not a matrix market file\n3 3 1\n1 2\n";
    assert!(CsrGraph::from_mtx_reader(Cursor::new(invalid_banner)).is_err());

    // 2. Zero-indexed entry (MTX is 1-indexed)
    let zero_index = "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n0 1\n";
    assert!(CsrGraph::from_mtx_reader(Cursor::new(zero_index)).is_err());

    // 3. Out-of-bounds index
    let out_of_bounds = "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n4 1\n";
    assert!(CsrGraph::from_mtx_reader(Cursor::new(out_of_bounds)).is_err());
}

#[test]
fn test_t1_f3_05_mtx_luxembourg_osm_mock_pipeline() {
    let mock_osm = "\
%%MatrixMarket matrix coordinate pattern symmetric
% Mock Luxembourg road junction
6 6 6
1 2
2 3
3 4
4 5
5 6
6 1
";
    let csr = CsrGraph::from_mtx_reader(Cursor::new(mock_osm)).expect("Parsing mock OSM failed");
    assert_eq!(csr.num_vertices(), 6);
    assert_eq!(csr.num_edges(), 12);
    assert!(csr.is_symmetric());

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &csr, &[0, 2]);
}
