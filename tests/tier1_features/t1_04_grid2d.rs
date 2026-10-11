use webgpu_bfs_core::Grid2D;

#[test]
fn test_t1_f4_01_grid2d_dimension_and_edge_cardinality() {
    let cases = [(1, 10), (10, 1), (6, 7), (15, 15)];
    for (w, h) in cases {
        let grid = Grid2D::new(w, h).unwrap();
        let nv = w * h;
        let ne = 2 * ((w.saturating_sub(1) * h) + (w * h.saturating_sub(1)));
        assert_eq!(grid.graph.num_vertices, nv);
        assert_eq!(grid.graph.num_edges, ne);
        assert!(grid.graph.validate().is_ok());
    }
}

#[test]
fn test_t1_f4_02_grid2d_canonical_neighbor_ordering() {
    let grid = Grid2D::new(10, 10).unwrap();
    // Vertex (5, 5) -> 55
    let v = grid.vertex_id(5, 5);
    let neighbors = grid.graph.neighbors(v as usize);
    assert_eq!(neighbors.len(), 4);
    // (5, 4)=45, (4, 5)=54, (6, 5)=56, (5, 6)=65
    assert_eq!(neighbors, &[45, 54, 56, 65]);
    for w in neighbors.windows(2) {
        assert!(w[0] < w[1]);
    }
}

#[test]
fn test_t1_f4_03_grid2d_single_source_manhattan_oracle() {
    let grid = Grid2D::new(8, 8).unwrap();
    let s = grid.vertex_id(2, 3);
    let oracle = grid.single_source_oracle(s);
    for v in 0..64u32 {
        let (x, y) = grid.coordinates(v);
        let expected = (x as i32 - 2).unsigned_abs() + (y as i32 - 3).unsigned_abs();
        assert_eq!(oracle[v as usize], expected);
    }
}

#[test]
fn test_t1_f4_04_grid2d_multi_source_oracle_layout() {
    let grid = Grid2D::new(5, 5).unwrap();
    let sources = [0, 4, 20, 24]; // 4 corners
    let oracle = grid.ms_bfs_oracle(&sources);
    assert_eq!(oracle.len(), 4 * 25);
    for (s_idx, &s) in sources.iter().enumerate() {
        assert_eq!(oracle[s_idx * 25 + s as usize], 0);
    }
}

#[test]
fn test_t1_f4_05_grid2d_all_engines_oracle_parity() {
    let grid = Grid2D::new(10, 10).unwrap();
    let sources = [0, 9, 90, 99, 45, 54, 23, 76];
    let oracle = grid.ms_bfs_oracle(&sources);

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_oracle_parity(&ctx, &grid.graph, &sources, &oracle);
}
