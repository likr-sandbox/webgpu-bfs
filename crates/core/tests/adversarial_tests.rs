use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::io::Cursor;
use webgpu_bfs_core::error::MtxError;
use webgpu_bfs_core::generators::{
    clique_graph, cycle_graph, disconnected_cliques, empty_graph, generate_erdos_renyi,
    generate_graph500_rmat, generate_grid2d, generate_random_tree, isolated_vertices, path_graph,
    single_vertex, star_graph,
};
use webgpu_bfs_core::graph::{CscGraph, CsrGraph};
use webgpu_bfs_core::mtx::MtxReader;

// ==============================================================================
// 1. ADVERSARIAL MTX PARSER TESTS
// ==============================================================================

#[test]
fn test_adversarial_mtx_empty_and_whitespace_inputs() {
    // 1. Completely empty (0 bytes)
    let res = MtxReader::parse(Cursor::new(""));
    assert!(
        matches!(res, Err(MtxError::InvalidHeader { .. })),
        "Empty input must return InvalidHeader, got: {:?}",
        res
    );

    // 2. Only whitespace and newlines
    let whitespace_inputs = [
        "   \t  \r\n",
        "\n\n\n",
        "  \n   \n\t\t\r\n",
        "                 ",
    ];
    for input in &whitespace_inputs {
        let res = MtxReader::parse(Cursor::new(input));
        assert!(
            matches!(res, Err(MtxError::InvalidHeader { .. })),
            "Whitespace input '{}' must return InvalidHeader, got: {:?}",
            input,
            res
        );
    }
}

#[test]
fn test_adversarial_mtx_corrupted_headers() {
    let cases = [
        // Missing banner
        ("MatrixMarket matrix coordinate pattern general\n3 3 1\n1 2\n", "Missing '%%MatrixMarket'"),
        // Wrong banner
        ("%%NotMatrixMarket matrix coordinate pattern general\n3 3 1\n1 2\n", "Unknown banner"),
        // Object not matrix
        ("%%MatrixMarket vector coordinate pattern general\n3 3 1\n1 2\n", "Unsupported object"),
        ("%%MatrixMarket tensor coordinate pattern general\n3 3 1\n1 2\n", "Unsupported object"),
        // Unknown format
        ("%%MatrixMarket matrix sparse pattern general\n3 3 1\n1 2\n", "Unknown format"),
        // Unsupported format: array
        ("%%MatrixMarket matrix array real general\n3 3 1\n1 2\n", "Unsupported format array"),
        // Unknown field
        ("%%MatrixMarket matrix coordinate boolean general\n3 3 1\n1 2\n", "Unknown field"),
        ("%%MatrixMarket matrix coordinate text general\n3 3 1\n1 2\n", "Unknown field"),
        // Unknown symmetry
        ("%%MatrixMarket matrix coordinate pattern triangular\n3 3 1\n1 2\n", "Unknown symmetry"),
        ("%%MatrixMarket matrix coordinate pattern unsymmetric\n3 3 1\n1 2\n", "Unknown symmetry"),
        // Header token count mismatch (too few)
        ("%%MatrixMarket matrix coordinate\n3 3 1\n1 2\n", "Too few tokens"),
        ("%%MatrixMarket\n3 3 1\n1 2\n", "Single token"),
        // Header token count mismatch (too many)
        ("%%MatrixMarket matrix coordinate pattern general extra\n3 3 1\n1 2\n", "Too many tokens"),
        // Comment before header banner (NIST spec requires header on line 1)
        ("% Comment preceding header\n%%MatrixMarket matrix coordinate pattern general\n3 3 1\n1 2\n", "Comment before header"),
    ];

    for (raw, desc) in &cases {
        let res = MtxReader::parse(Cursor::new(raw));
        assert!(
            res.is_err(),
            "Corrupted header case '{}' must be rejected gracefully with Err, but got Ok",
            desc
        );
    }
}

#[test]
fn test_adversarial_mtx_corrupted_dimensions() {
    let cases = [
        // Missing dimension line entirely (EOF)
        "%%MatrixMarket matrix coordinate pattern general\n",
        // Dimension line with negative rows
        "%%MatrixMarket matrix coordinate pattern general\n-3 3 1\n1 2\n",
        // Dimension line with negative cols
        "%%MatrixMarket matrix coordinate pattern general\n3 -3 1\n1 2\n",
        // Dimension line with negative nnz
        "%%MatrixMarket matrix coordinate pattern general\n3 3 -1\n1 2\n",
        // Dimension line with non-numeric text
        "%%MatrixMarket matrix coordinate pattern general\nfoo 3 1\n1 2\n",
        "%%MatrixMarket matrix coordinate pattern general\n3 bar 1\n1 2\n",
        "%%MatrixMarket matrix coordinate pattern general\n3 3 baz\n1 2\n",
        // Dimension line with float
        "%%MatrixMarket matrix coordinate pattern general\n3.5 3 1\n1 2\n",
        // Dimension line with only 2 tokens
        "%%MatrixMarket matrix coordinate pattern general\n3 3\n1 2\n",
        // Dimension line with 4 tokens
        "%%MatrixMarket matrix coordinate pattern general\n3 3 1 0\n1 2\n",
        // Symmetric matrix with non-square dimensions
        "%%MatrixMarket matrix coordinate pattern symmetric\n4 5 1\n1 2\n",
    ];

    for raw in &cases {
        let res = MtxReader::parse(Cursor::new(raw));
        assert!(
            matches!(res, Err(MtxError::InvalidDimensions { .. })),
            "Expected InvalidDimensions for input:\n{}\nGot: {:?}",
            raw,
            res
        );
    }
}

#[test]
fn test_adversarial_mtx_corrupted_coordinates() {
    // 1. Zero index cases (Matrix Market is 1-based)
    let zero_cases = [
        ("3 3 1\n0 1\n", 0, 1),
        ("3 3 1\n1 0\n", 1, 0),
        ("3 3 1\n0 0\n", 0, 0),
    ];
    for (body, expected_r, expected_c) in &zero_cases {
        let raw = format!("%%MatrixMarket matrix coordinate pattern general\n{}", body);
        let res = MtxReader::parse(Cursor::new(raw));
        assert!(
            matches!(
                res,
                Err(MtxError::ZeroIndex { row, col, .. }) if row == *expected_r && col == *expected_c
            ),
            "Expected ZeroIndex({}, {}), got: {:?}",
            expected_r,
            expected_c,
            res
        );
    }

    // 2. Out of bounds index cases (> rows or > cols)
    let oob_cases = [
        ("3 3 1\n4 1\n", 4, 1),
        ("3 3 1\n1 4\n", 1, 4),
        ("3 3 1\n10 10\n", 10, 10),
    ];
    for (body, expected_r, expected_c) in &oob_cases {
        let raw = format!("%%MatrixMarket matrix coordinate pattern general\n{}", body);
        let res = MtxReader::parse(Cursor::new(raw));
        assert!(
            matches!(
                res,
                Err(MtxError::IndexOutOfBounds { row, col, .. }) if row == *expected_r && col == *expected_c
            ),
            "Expected IndexOutOfBounds({}, {}), got: {:?}",
            expected_r,
            expected_c,
            res
        );
    }

    // 3. Negative coordinates / non-integers / incomplete coordinate lines
    let invalid_line_cases = [
        "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n-1 2\n",
        "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n1 -2\n",
        "%%MatrixMarket matrix coordinate pattern general\n3 3 1\nfoo 2\n",
        "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n1 bar\n",
        "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n1.5 2\n",
        "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n1\n", // Single token
    ];
    for raw in &invalid_line_cases {
        let res = MtxReader::parse(Cursor::new(raw));
        assert!(
            matches!(res, Err(MtxError::InvalidCoordinateLine { .. })),
            "Expected InvalidCoordinateLine for input:\n{}\nGot: {:?}",
            raw,
            res
        );
    }
}

#[test]
fn test_adversarial_mtx_extreme_dimensions_and_zero_edges() {
    // 1. Dimensions 0 0 0: Empty graph with 0 vertices, 0 edges
    let raw_zero = "\
%%MatrixMarket matrix coordinate pattern general
0 0 0
";
    let graph_zero = MtxReader::parse(Cursor::new(raw_zero)).unwrap();
    assert_eq!(graph_zero.num_vertices, 0);
    assert_eq!(graph_zero.num_edges, 0);
    assert!(graph_zero.is_empty());
    assert!(graph_zero.validate().is_ok());

    // 2. Huge dimension with 0 edges: 100,000 vertices, 0 edges (isolated vertices)
    let raw_huge = "\
%%MatrixMarket matrix coordinate pattern general
100000 100000 0
";
    let graph_huge = MtxReader::parse(Cursor::new(raw_huge)).unwrap();
    assert_eq!(graph_huge.num_vertices, 100000);
    assert_eq!(graph_huge.num_edges, 0);
    assert_eq!(graph_huge.row_offsets.len(), 100001);
    assert_eq!(graph_huge.col_indices.len(), 0);
    assert_eq!(graph_huge.max_degree(), 0);
    assert!(graph_huge.validate().is_ok());

    // 3. Rectangular graph with 0 edges
    let raw_rect = "\
%%MatrixMarket matrix coordinate pattern general
50000 20000 0
";
    let graph_rect = MtxReader::parse(Cursor::new(raw_rect)).unwrap();
    assert_eq!(graph_rect.num_vertices, 50000);
    assert_eq!(graph_rect.num_edges, 0);
    assert!(graph_rect.validate().is_ok());
}

#[test]
fn test_adversarial_mtx_comments_and_formatting_stress() {
    // Interspersed comments, empty lines, CRLF, extra values/weights, trailing comments
    let raw = "\
%%MatrixMarket Matrix Coordinate Real Symmetric\r
% Line comment 1\r
%\r
%    Indented comment with symbols: !@#$%^&*()_+\r
% Line comment 2\r
\r
4 4 6\r
% Comment before first edge\r
1 2 3.14159\r
% Another comment in between\r
\r
2 3 2.71828\r
3 4 1.41421\r
% Comment at the end of file\r
%\r
";
    let graph = MtxReader::parse(Cursor::new(raw)).unwrap();
    assert_eq!(graph.num_vertices, 4);
    // 3 symmetric edges -> 6 directed edges
    assert_eq!(graph.num_edges, 6);
    assert!(graph.is_symmetric());
    assert_eq!(graph.neighbors(0), &[1]);
    assert_eq!(graph.neighbors(1), &[0, 2]);
    assert_eq!(graph.neighbors(2), &[1, 3]);
    assert_eq!(graph.neighbors(3), &[2]);
    assert!(graph.validate().is_ok());
}

// ==============================================================================
// 2. TRANSPOSITION INVARIANT (graph.to_csc().to_csr() == graph)
// ==============================================================================

#[test]
fn test_transposition_invariant_100_random_graphs() {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF_CAFE1234);

    // Test a corpus of 100 diverse random and deterministic graph topologies
    for i in 0..100 {
        let graph: CsrGraph = match i {
            // Edge case topologies
            0 => empty_graph(),
            1 => single_vertex(),
            2 => isolated_vertices(50),
            3 => path_graph(30),
            4 => cycle_graph(30),
            5 => star_graph(40),
            6 => clique_graph(20),
            7 => disconnected_cliques(&[5, 6, 7]),
            8 => generate_grid2d(10, 10),
            9 => generate_grid2d(1, 50),
            10 => generate_random_tree(50, Some(rng.gen())).unwrap(),
            11 => generate_graph500_rmat(6, 16, Some(rng.gen())).unwrap(),
            // Procedural Erdős-Rényi graphs
            _ => {
                let n = rng.gen_range(1..120);
                let p = rng.gen_range(0.005..0.4);
                let seed = rng.gen::<u64>();
                generate_erdos_renyi(n, p, seed)
            }
        };

        // 1. Validate original CSR graph
        assert!(
            graph.validate().is_ok(),
            "Graph #{} failed CSR validation",
            i
        );

        // 2. Transpose to CSC
        let csc: CscGraph = graph.to_csc();

        // 3. Validate CSC invariants
        assert!(csc.validate().is_ok(), "Graph #{} failed CSC validation", i);
        assert_eq!(
            csc.num_vertices, graph.num_vertices,
            "Graph #{}: CSC num_vertices mismatch",
            i
        );
        assert_eq!(
            csc.num_edges, graph.num_edges,
            "Graph #{}: CSC num_edges mismatch",
            i
        );

        // 4. Transpose back to CSR (Roundtrip)
        let roundtrip: CsrGraph = csc.to_csr();

        // 5. Validate roundtrip graph
        assert!(
            roundtrip.validate().is_ok(),
            "Graph #{} roundtrip failed CSR validation",
            i
        );

        // 6. Assert Exact Roundtrip Identity: graph.to_csc().to_csr() == graph
        assert_eq!(
            roundtrip.num_vertices, graph.num_vertices,
            "Graph #{}: Roundtrip num_vertices mismatch",
            i
        );
        assert_eq!(
            roundtrip.num_edges, graph.num_edges,
            "Graph #{}: Roundtrip num_edges mismatch",
            i
        );
        assert_eq!(
            roundtrip.row_offsets, graph.row_offsets,
            "Graph #{}: Roundtrip row_offsets mismatch",
            i
        );
        assert_eq!(
            roundtrip.col_indices, graph.col_indices,
            "Graph #{}: Roundtrip col_indices mismatch",
            i
        );
        assert_eq!(
            roundtrip, graph,
            "Graph #{}: Roundtrip graph != original graph",
            i
        );

        // 7. Dual Roundtrip Identity: csc.to_csr().to_csc() == csc
        let csc_roundtrip = roundtrip.to_csc();
        assert_eq!(
            csc_roundtrip.col_offsets, csc.col_offsets,
            "Graph #{}: Dual roundtrip col_offsets mismatch",
            i
        );
        assert_eq!(
            csc_roundtrip.row_indices, csc.row_indices,
            "Graph #{}: Dual roundtrip row_indices mismatch",
            i
        );
    }
}

// ==============================================================================
// 3. HANDSHAKING LEMMA FOR UNDIRECTED SYMMETRIC GRAPHS
//    \sum_{u=0}^{|V|-1} degree(u) == 2 |E_undirected| == num_edges_directed
// ==============================================================================

#[test]
fn test_handshaking_lemma_100_symmetric_graphs() {
    let mut rng = StdRng::seed_from_u64(0x594D4D45_54524943);

    for i in 0..100 {
        let graph: CsrGraph = match i {
            0 => empty_graph(),
            1 => single_vertex(),
            2 => isolated_vertices(100),
            3 => path_graph(50),
            4 => cycle_graph(50),
            5 => star_graph(60),
            6 => clique_graph(25),
            7 => disconnected_cliques(&[4, 6, 8, 10]),
            8 => generate_grid2d(12, 12),
            9 => generate_grid2d(3, 40),
            10 => generate_random_tree(80, Some(rng.gen())).unwrap(),
            11 => generate_graph500_rmat(6, 12, Some(rng.gen())).unwrap(),
            _ => {
                let n = rng.gen_range(1..100);
                let p = rng.gen_range(0.01..0.5);
                let seed = rng.gen::<u64>();
                // generate_erdos_renyi generates undirected (symmetric) graphs by default
                generate_erdos_renyi(n, p, seed)
            }
        };

        // Graph must be undirected (symmetric)
        assert!(graph.is_symmetric(), "Graph #{} must be symmetric", i);

        let v = graph.num_vertices;
        let csc = graph.to_csc();

        // 1. Sum of out-degrees
        let sum_out_degrees: usize = (0..v).map(|u| graph.degree(u)).sum();

        // 2. Sum of in-degrees
        let sum_in_degrees: usize = (0..v).map(|u| csc.in_degree(u)).sum();

        // 3. Degree of every vertex must match its in-degree
        for u in 0..v {
            assert_eq!(
                graph.degree(u),
                csc.in_degree(u),
                "Graph #{}, vertex {}: degree({}) != in_degree({})",
                i,
                u,
                graph.degree(u),
                csc.in_degree(u)
            );
        }

        // 4. In a symmetric loopless graph, num_edges is exactly 2 * |E_undirected|
        assert_eq!(
            graph.num_edges % 2,
            0,
            "Graph #{}: Directed edge count {} is not even in symmetric graph",
            i,
            graph.num_edges
        );
        let undirected_edges = graph.num_edges / 2;

        // 5. Handshaking Lemma: \sum degree(u) == 2 * |E_undirected| == directed edges
        assert_eq!(
            sum_out_degrees,
            2 * undirected_edges,
            "Graph #{}: Handshaking lemma violated: sum_out_degrees {} != 2 * {}",
            i,
            sum_out_degrees,
            undirected_edges
        );
        assert_eq!(
            sum_in_degrees,
            2 * undirected_edges,
            "Graph #{}: Handshaking lemma violated for in-degrees: sum_in_degrees {} != 2 * {}",
            i,
            sum_in_degrees,
            undirected_edges
        );
        assert_eq!(
            sum_out_degrees, graph.num_edges,
            "Graph #{}: sum_out_degrees != num_edges",
            i
        );

        // 6. Direct confirmation by counting unique unordered pairs {u, v}
        let mut unique_undirected_pairs: HashSet<(u32, u32)> = HashSet::new();
        for u in 0..v {
            for &w in graph.neighbors(u) {
                let pair = if (u as u32) < w {
                    (u as u32, w)
                } else {
                    (w, u as u32)
                };
                unique_undirected_pairs.insert(pair);
            }
        }
        assert_eq!(
            unique_undirected_pairs.len(),
            undirected_edges,
            "Graph #{}: Unique unordered pair count {} != undirected_edges {}",
            i,
            unique_undirected_pairs.len(),
            undirected_edges
        );
    }
}

// ==============================================================================
// 4. SELF-LOOP AND DUPLICATE EDGE FILTERING UNDER HIGH CONTENTION / RANDOMNESS
// ==============================================================================

#[test]
fn test_self_loops_and_duplicate_filtering_under_high_contention() {
    let mut rng = StdRng::seed_from_u64(0xCAFEBABE_98765432);

    let num_vertices = 64;
    let mut raw_edges = Vec::new();

    // 1. Flood with self-loops: every vertex u has (u, u) repeated 25 times
    for u in 0..num_vertices as u32 {
        for _ in 0..25 {
            raw_edges.push((u, u));
        }
    }

    // 2. Flood with dense duplicate edges:
    // Create random multigraph edges with 5 to 50 duplicates per edge
    for _ in 0..500 {
        let u = rng.gen_range(0..num_vertices as u32);
        let v = rng.gen_range(0..num_vertices as u32);
        let reps = rng.gen_range(5..50);
        for _ in 0..reps {
            raw_edges.push((u, v));
        }
    }

    // 3. Flood with out-of-bounds edges: endpoints >= num_vertices
    for _ in 0..200 {
        let u = rng.gen_range(num_vertices as u32..num_vertices as u32 + 50);
        let v = rng.gen_range(0..num_vertices as u32);
        raw_edges.push((u, v));
        raw_edges.push((v, u));
    }

    let total_raw_count = raw_edges.len();
    assert!(
        total_raw_count > 10000,
        "Should have > 10,000 raw input edges"
    );

    // Construct CSR graph
    let graph = CsrGraph::from_edge_list(num_vertices, &raw_edges);

    // Invariant verification:
    // A. Valid graph structure
    assert!(graph.validate().is_ok());

    // B. Zero self-loops present
    for u in 0..num_vertices {
        let neighbors = graph.neighbors(u);
        assert!(
            !neighbors.contains(&(u as u32)),
            "Vertex {} contains self-loop in neighbor list: {:?}",
            u,
            neighbors
        );
    }

    // C. Zero duplicate edges: neighbor lists are strictly monotonic
    for u in 0..num_vertices {
        let neighbors = graph.neighbors(u);
        for w in neighbors.windows(2) {
            assert!(
                w[0] < w[1],
                "Vertex {} neighbor list not strictly increasing: {} >= {}",
                u,
                w[0],
                w[1]
            );
        }
    }

    // D. In-bounds column indices
    for &col in &graph.col_indices {
        assert!((col as usize) < num_vertices);
    }
}

#[test]
fn test_mtx_streaming_parser_with_massive_duplicates_and_self_loops() {
    // Construct an MTX stream with 5,000 coordinate lines having heavy self-loops and multi-edges
    let mut mtx_buf =
        String::from("%%MatrixMarket matrix coordinate integer general\n10 10 5000\n");

    for _ in 0..1000 {
        // 1000 self-loops
        mtx_buf.push_str("3 3 1\n");
        mtx_buf.push_str("5 5 99\n");
    }
    for _ in 0..1500 {
        // 1500 duplicate edges (1, 2)
        mtx_buf.push_str("1 2 42\n");
    }
    for _ in 0..1500 {
        // 1500 duplicate edges (2, 7)
        mtx_buf.push_str("2 7 10\n");
    }

    let graph = MtxReader::parse(Cursor::new(mtx_buf)).unwrap();

    assert_eq!(graph.num_vertices, 10);
    // Only 2 unique non-self-loop directed edges: (0 -> 1) and (1 -> 6)
    assert_eq!(graph.num_edges, 2);
    assert_eq!(graph.neighbors(0), &[1]);
    assert_eq!(graph.neighbors(1), &[6]);
    for u in 2..10 {
        assert_eq!(graph.neighbors(u), &[]);
    }
    assert!(graph.validate().is_ok());
}
