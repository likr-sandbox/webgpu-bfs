#![allow(clippy::identity_op, clippy::erasing_op)]

use crate::{
    bit_parallel::ms_bfs_bit_parallel, fifo::ms_bfs_fifo, rayon_ms::ms_bfs_rayon, BfsEngine,
    CpuBfsEngine,
};
use webgpu_bfs_core::{
    generators::{generate_erdos_renyi, generate_grid2d, generate_random_tree},
    CsrGraph,
};

/// Asserts that all three CPU BFS algorithms produce 100% bitwise identical results.
fn assert_cpu_3way_equivalence(graph: &CsrGraph, sources: &[u32]) {
    let dist_fifo = ms_bfs_fifo(graph, sources).expect("FIFO BFS failed");
    let dist_rayon = ms_bfs_rayon(graph, sources).expect("Rayon BFS failed");
    let dist_bit = ms_bfs_bit_parallel(graph, sources).expect("Bit-parallel BFS failed");

    assert_eq!(
        dist_fifo.len(),
        sources.len() * graph.num_vertices,
        "Length mismatch in FIFO output"
    );
    assert_eq!(
        dist_fifo, dist_rayon,
        "Discrepancy detected between Algorithm A (FIFO) and Algorithm B (Rayon)!"
    );
    assert_eq!(
        dist_fifo, dist_bit,
        "Discrepancy detected between Algorithm A (FIFO) and Algorithm C (Bit-Parallel)!"
    );
}

#[test]
fn test_grid2d_against_manhattan_distance_oracle() {
    let width = 10;
    let height = 10;
    let graph = generate_grid2d(width, height);
    assert_eq!(graph.num_vertices, 100);

    // Sources: 4 corners and center
    let sources = vec![0, 9, 90, 99, 45];
    let engine = CpuBfsEngine::new();
    let dist = engine
        .run_ms_bfs(&graph, &sources)
        .expect("Grid2D BFS failed");

    for (s_idx, &src) in sources.iter().enumerate() {
        let sx = (src % width as u32) as i32;
        let sy = (src / width as u32) as i32;

        for v in 0..graph.num_vertices {
            let vx = (v as u32 % width as u32) as i32;
            let vy = (v as u32 / width as u32) as i32;

            let manhattan = ((sx - vx).abs() + (sy - vy).abs()) as u32;
            let computed = dist[s_idx * graph.num_vertices + v];

            assert_eq!(
                computed, manhattan,
                "Shortest path mismatch on Grid2D from ({}, {}) to ({}, {}): computed={}, expected={}",
                sx, sy, vx, vy, computed, manhattan
            );
        }
    }

    assert_cpu_3way_equivalence(&graph, &sources);
}

#[test]
fn test_random_tree_connectivity_and_triangle_inequality() {
    let n = 200;
    let graph = generate_random_tree(n, Some(42)).expect("Tree generation failed");
    assert_eq!(graph.num_vertices, n);

    let sources = vec![0, 10, 50, 100, 150];
    let dist = ms_bfs_fifo(&graph, &sources).expect("Tree BFS failed");

    for (s_idx, &src) in sources.iter().enumerate() {
        // Source distance is 0
        assert_eq!(dist[s_idx * n + src as usize], 0);

        // A tree is connected: no vertex can have distance u32::MAX
        for v in 0..n {
            let d = dist[s_idx * n + v];
            assert_ne!(
                d,
                u32::MAX,
                "Vertex {} in tree is unreachable from source {}",
                v,
                src
            );
            assert!(d < n as u32, "Tree distance exceeds vertex count");
        }
    }

    assert_cpu_3way_equivalence(&graph, &sources);
}

#[test]
fn test_erdos_renyi_batch_sizes() {
    let n = 500;
    let p = 0.05;
    let graph = generate_erdos_renyi(n, p, 12345);

    // Test varying batch sizes: S = 1, 16, 32, 64, 128
    for &s_count in &[1, 16, 32, 64, 128] {
        let sources: Vec<u32> = (0..s_count).map(|s| (s * 3) % n as u32).collect();
        assert_cpu_3way_equivalence(&graph, &sources);
    }
}

#[test]
fn test_non_power_of_two_batch_sizes() {
    let n = 300;
    let graph = generate_erdos_renyi(n, 0.04, 999);

    // Arbitrary non-power-of-two source counts
    for &s_count in &[2, 7, 33, 65] {
        let sources: Vec<u32> = (0..s_count).collect();
        assert_cpu_3way_equivalence(&graph, &sources);
    }
}

#[test]
fn test_disconnected_graph_and_unreachable_sentinel() {
    // Construct graph with 2 isolated components: {0, 1, 2} and {3, 4}
    // Component 1: 0-1, 1-2, 2-0 (triangle)
    // Component 2: 3-4
    let row_offsets = vec![0, 2, 4, 6, 7, 8];
    let col_indices = vec![
        1, 2, // 0 -> 1, 2
        0, 2, // 1 -> 0, 2
        0, 1, // 2 -> 0, 1
        4, // 3 -> 4
        3, // 4 -> 3
    ];
    let graph = CsrGraph {
        num_vertices: 5,
        num_edges: 8,
        row_offsets,
        col_indices,
    };

    let sources = vec![0, 3];
    let dist = ms_bfs_fifo(&graph, &sources).expect("Disconnected BFS failed");

    // From source 0: 0, 1, 2 reachable; 3, 4 unreachable (u32::MAX)
    assert_eq!(dist[0 * 5 + 0], 0);
    assert_eq!(dist[0 * 5 + 1], 1);
    assert_eq!(dist[0 * 5 + 2], 1);
    assert_eq!(dist[0 * 5 + 3], u32::MAX);
    assert_eq!(dist[0 * 5 + 4], u32::MAX);

    // From source 3: 3, 4 reachable; 0, 1, 2 unreachable (u32::MAX)
    assert_eq!(dist[1 * 5 + 0], u32::MAX);
    assert_eq!(dist[1 * 5 + 1], u32::MAX);
    assert_eq!(dist[1 * 5 + 2], u32::MAX);
    assert_eq!(dist[1 * 5 + 3], 0);
    assert_eq!(dist[1 * 5 + 4], 1);

    assert_cpu_3way_equivalence(&graph, &sources);
}

#[test]
fn test_topological_edge_cases() {
    // 1. Single-vertex graph (N=1, E=0)
    let g1 = CsrGraph {
        num_vertices: 1,
        num_edges: 0,
        row_offsets: vec![0, 0],
        col_indices: vec![],
    };
    assert_cpu_3way_equivalence(&g1, &[0]);
    let d1 = ms_bfs_fifo(&g1, &[0]).unwrap();
    assert_eq!(d1, vec![0]);

    // 2. Zero-edge graph (N=4, E=0)
    let g0 = CsrGraph {
        num_vertices: 4,
        num_edges: 0,
        row_offsets: vec![0, 0, 0, 0, 0],
        col_indices: vec![],
    };
    assert_cpu_3way_equivalence(&g0, &[0, 2]);
    let d0 = ms_bfs_fifo(&g0, &[0, 2]).unwrap();
    assert_eq!(
        d0,
        vec![
            0,
            u32::MAX,
            u32::MAX,
            u32::MAX, // from 0
            u32::MAX,
            u32::MAX,
            0,
            u32::MAX // from 2
        ]
    );

    // 3. High-diameter line graph (0-1-2-3-4)
    let g_line = CsrGraph {
        num_vertices: 5,
        num_edges: 8,
        row_offsets: vec![0, 1, 3, 5, 7, 8],
        col_indices: vec![1, 0, 2, 1, 3, 2, 4, 3],
    };
    assert_cpu_3way_equivalence(&g_line, &[0, 4]);
    let d_line = ms_bfs_fifo(&g_line, &[0, 4]).unwrap();
    assert_eq!(
        d_line,
        vec![
            0, 1, 2, 3, 4, // from 0
            4, 3, 2, 1, 0 // from 4
        ]
    );
}

#[test]
fn test_duplicate_sources() {
    let graph = generate_grid2d(5, 5);
    let sources = vec![7, 7, 12, 7];
    assert_cpu_3way_equivalence(&graph, &sources);
}

#[test]
fn test_out_of_bounds_source_error() {
    let graph = generate_grid2d(3, 3); // 9 vertices: 0..8
    let result = ms_bfs_fifo(&graph, &[0, 9]);
    assert!(result.is_err(), "Expected error for out-of-bounds source");
}

#[test]
fn test_empty_sources_and_empty_graph() {
    let graph = generate_grid2d(3, 3);
    let res_empty_src = ms_bfs_fifo(&graph, &[]).unwrap();
    assert!(res_empty_src.is_empty());

    let g_empty = CsrGraph {
        num_vertices: 0,
        num_edges: 0,
        row_offsets: vec![0],
        col_indices: vec![],
    };
    let res_empty_g = ms_bfs_fifo(&g_empty, &[]).unwrap();
    assert!(res_empty_g.is_empty());
}
