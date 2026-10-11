use std::time::Instant;
use webgpu_bfs_core::{
    generators::{
        clique_graph, csr_from_undirected_pairs, empty_graph, generate_erdos_renyi_np,
        generate_graph500_rmat, generate_grid2d, isolated_vertices, path_graph, single_vertex,
        Grid2D,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::{
    ms_bfs_bit_parallel, ms_bfs_fifo, ms_bfs_rayon, CpuBfsAlgorithm, CpuBfsEngine,
};

/// Batch sizes requested in the challenge specification
const BATCH_SIZES: [usize; 7] = [1, 3, 17, 32, 64, 65, 128];

/// Asserts 100% bitwise equality across Algorithm A (FIFO), Algorithm B (Rayon),
/// and Algorithm C (Bit-Parallel 64-bit), and returns execution timings.
fn assert_cpu_3way_equality(graph: &CsrGraph, sources: &[u32]) -> (Vec<u32>, f64, f64, f64) {
    let t0 = Instant::now();
    let dist_fifo = ms_bfs_fifo(graph, sources).expect("FIFO MS-BFS failed");
    let dur_fifo = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let dist_rayon = ms_bfs_rayon(graph, sources).expect("Rayon MS-BFS failed");
    let dur_rayon = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    let dist_bit = ms_bfs_bit_parallel(graph, sources).expect("Bit-Parallel MS-BFS failed");
    let dur_bit = t2.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        dist_fifo.len(),
        sources.len() * graph.num_vertices,
        "Distance vector length mismatch for sources count {}",
        sources.len()
    );

    // Bitwise equality between Algorithm A and Algorithm B
    assert_eq!(
        dist_fifo,
        dist_rayon,
        "Discrepancy between Algorithm A (FIFO) and Algorithm B (Rayon) for S={}",
        sources.len()
    );

    // Bitwise equality between Algorithm A and Algorithm C
    assert_eq!(
        dist_fifo,
        dist_bit,
        "Discrepancy between Algorithm A (FIFO) and Algorithm C (Bit-Parallel) for S={}",
        sources.len()
    );

    (dist_fifo, dur_fifo, dur_rayon, dur_bit)
}

/// Asserts the metric triangle inequality for all edges across all sources:
/// In an unweighted undirected graph, for every edge (u, v):
/// If u is reachable, v must be reachable, and |dist[s, u] - dist[s, v]| <= 1.
/// If neither is reachable, both must equal u32::MAX.
fn assert_triangle_inequality(graph: &CsrGraph, sources: &[u32], dist: &[u32]) {
    let nv = graph.num_vertices;
    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * nv;
        // Source node distance must be exactly 0
        assert_eq!(
            dist[base + src as usize],
            0,
            "Source {} at batch index {} does not have distance 0",
            src,
            s_idx
        );

        for u in 0..nv {
            let du = dist[base + u];
            let start = graph.row_offsets[u] as usize;
            let end = graph.row_offsets[u + 1] as usize;

            for &v in &graph.col_indices[start..end] {
                let dv = dist[base + v as usize];

                if du != u32::MAX {
                    assert_ne!(
                        dv,
                        u32::MAX,
                        "Edge ({}, {}) violation: node {} has dist {}, but adjacent node {} is marked unreachable",
                        u, v, u, du, v
                    );
                    let diff = du.abs_diff(dv);
                    assert!(
                        diff <= 1,
                        "Triangle inequality violated on edge ({}, {}): dist(u)={}, dist(v)={}, diff={}",
                        u, v, du, dv, diff
                    );
                } else {
                    // For undirected graphs, if u is unreachable, adjacent v must also be unreachable
                    assert_eq!(
                        dv,
                        u32::MAX,
                        "Edge ({}, {}) violation: node {} is unreachable, but adjacent node {} has dist {}",
                        u, v, u, v, dv
                    );
                }
            }
        }
    }
}

/// Helper to generate disconnected star components
fn generate_disconnected_stars(num_stars: usize, star_size: usize) -> CsrGraph {
    let total_vertices = num_stars * star_size;
    let mut edges = Vec::new();

    for star_idx in 0..num_stars {
        let hub = (star_idx * star_size) as u32;
        for leaf_offset in 1..star_size {
            let leaf = hub + leaf_offset as u32;
            edges.push((hub, leaf));
        }
    }

    csr_from_undirected_pairs(total_vertices, &edges)
}

#[test]
fn test_stress_grid2d_50x50_manhattan_oracle() {
    println!("\n=== TEST: 50x50 Grid2D vs Analytical Manhattan Oracle ===");
    let width = 50;
    let height = 50;
    let grid = Grid2D::new(width, height).expect("Failed to create 50x50 grid");
    let nv = grid.graph.num_vertices;
    assert_eq!(nv, 2500);
    assert_eq!(grid.graph.num_edges, 2 * (49 * 50 + 50 * 49)); // 9800 directed edges

    for &s_count in &BATCH_SIZES {
        // Deterministically construct sources across grid (corners, edges, center, stride)
        let sources: Vec<u32> = (0..s_count).map(|i| ((i * 19) % nv) as u32).collect();

        let (dist, dur_fifo, dur_rayon, dur_bit) = assert_cpu_3way_equality(&grid.graph, &sources);

        // Verify against analytical Manhattan distance oracle
        for (s_idx, &src) in sources.iter().enumerate() {
            let sx = (src % width as u32) as i32;
            let sy = (src / width as u32) as i32;

            for v in 0..nv {
                let vx = (v as u32 % width as u32) as i32;
                let vy = (v as u32 / width as u32) as i32;
                let expected_manhattan = ((sx - vx).abs() + (sy - vy).abs()) as u32;
                let computed = dist[s_idx * nv + v];

                assert_eq!(
                    computed, expected_manhattan,
                    "Manhattan oracle mismatch on Grid2D 50x50: source ({}, {}) to node ({}, {}): computed={}, expected={}",
                    sx, sy, vx, vy, computed, expected_manhattan
                );
            }
        }

        // Verify triangle inequality
        assert_triangle_inequality(&grid.graph, &sources, &dist);

        println!(
            "  Grid2D 50x50 [S={:>3}]: FIFO={:>6.2}ms, Rayon={:>6.2}ms, Bit-Parallel={:>6.2}ms (All Passed)",
            s_count, dur_fifo, dur_rayon, dur_bit
        );
    }
}

#[test]
fn test_stress_linear_chain_10k() {
    println!("\n=== TEST: Deep Linear Chain (N=10,000, Depth=9,999) ===");
    let n = 10_000;
    let chain = path_graph(n);
    assert_eq!(chain.num_vertices, n);
    assert_eq!(chain.num_edges, 2 * (n - 1));

    for &s_count in &BATCH_SIZES {
        // Diverse source distribution: endpoints, quarter points, stride
        let sources: Vec<u32> = (0..s_count)
            .map(|i| {
                if i == 0 {
                    0
                } else if i == 1 {
                    (n - 1) as u32
                } else {
                    ((i * 1337) % n) as u32
                }
            })
            .collect();

        let (dist, dur_fifo, dur_rayon, dur_bit) = assert_cpu_3way_equality(&chain, &sources);

        // Analytical oracle for path graph: dist[s, v] = |s - v|
        for (s_idx, &src) in sources.iter().enumerate() {
            for v in 0..n {
                let expected = (src as i32 - v as i32).unsigned_abs();
                let computed = dist[s_idx * n + v];
                assert_eq!(
                    computed, expected,
                    "Path graph oracle mismatch: src={}, v={}, computed={}, expected={}",
                    src, v, computed, expected
                );
            }
        }

        assert_triangle_inequality(&chain, &sources, &dist);

        println!(
            "  Chain 10k    [S={:>3}]: FIFO={:>6.2}ms, Rayon={:>6.2}ms, Bit-Parallel={:>6.2}ms (All Passed)",
            s_count, dur_fifo, dur_rayon, dur_bit
        );
    }
}

#[test]
fn test_stress_dense_clique_k500() {
    println!("\n=== TEST: Dense Clique K_500 (N=500, E=249,500 directed) ===");
    let n = 500;
    let clique = clique_graph(n);
    assert_eq!(clique.num_vertices, n);
    assert_eq!(clique.num_edges, n * (n - 1));

    for &s_count in &BATCH_SIZES {
        let sources: Vec<u32> = (0..s_count).map(|i| (i % n) as u32).collect();

        let (dist, dur_fifo, dur_rayon, dur_bit) = assert_cpu_3way_equality(&clique, &sources);

        // Analytical oracle for clique: dist[s, s] = 0, dist[s, v] = 1 for v != s
        for (s_idx, &src) in sources.iter().enumerate() {
            for v in 0..n {
                let expected = if v as u32 == src { 0 } else { 1 };
                let computed = dist[s_idx * n + v];
                assert_eq!(
                    computed, expected,
                    "Clique K500 oracle mismatch: src={}, v={}, computed={}, expected={}",
                    src, v, computed, expected
                );
            }
        }

        assert_triangle_inequality(&clique, &sources, &dist);

        println!(
            "  Clique K500  [S={:>3}]: FIFO={:>6.2}ms, Rayon={:>6.2}ms, Bit-Parallel={:>6.2}ms (All Passed)",
            s_count, dur_fifo, dur_rayon, dur_bit
        );
    }
}

#[test]
fn test_stress_disconnected_stars() {
    println!("\n=== TEST: Disconnected Stars (20 stars x 50 vertices = 1,000 vertices) ===");
    let num_stars = 20;
    let star_size = 50;
    let total_v = num_stars * star_size;
    let graph = generate_disconnected_stars(num_stars, star_size);
    assert_eq!(graph.num_vertices, total_v);

    for &s_count in &BATCH_SIZES {
        // Pick sources across different components, including hubs and leaves
        let sources: Vec<u32> = (0..s_count)
            .map(|i| {
                let star = i % num_stars;
                let offset = (i * 7) % star_size;
                (star * star_size + offset) as u32
            })
            .collect();

        let (dist, dur_fifo, dur_rayon, dur_bit) = assert_cpu_3way_equality(&graph, &sources);

        // Analytical oracle for disconnected stars:
        // Same component:
        //   - if s == v -> 0
        //   - if s is hub or v is hub -> 1
        //   - else (both leaves) -> 2
        // Different component:
        //   - u32::MAX
        for (s_idx, &src) in sources.iter().enumerate() {
            let src_star = src as usize / star_size;
            let src_is_hub = (src as usize).is_multiple_of(star_size);

            for v in 0..total_v {
                let v_star = v / star_size;
                let v_is_hub = v.is_multiple_of(star_size);

                let expected = if src_star != v_star {
                    u32::MAX
                } else if src as usize == v {
                    0
                } else if src_is_hub || v_is_hub {
                    1
                } else {
                    2
                };

                let computed = dist[s_idx * total_v + v];
                assert_eq!(
                    computed, expected,
                    "Disconnected stars mismatch: src={}, v={}, computed={}, expected={}",
                    src, v, computed, expected
                );
            }
        }

        assert_triangle_inequality(&graph, &sources, &dist);

        println!(
            "  Disc Stars   [S={:>3}]: FIFO={:>6.2}ms, Rayon={:>6.2}ms, Bit-Parallel={:>6.2}ms (All Passed)",
            s_count, dur_fifo, dur_rayon, dur_bit
        );
    }
}

#[test]
fn test_stress_erdos_renyi_5000() {
    println!("\n=== TEST: Large Erdős-Rényi Graph (N=5,000, p=0.002, ~25,000 edges) ===");
    let n = 5_000;
    let p = 0.002;
    let graph = generate_erdos_renyi_np(n, p, Some(424242)).expect("ER generation failed");
    println!(
        "  Constructed ER(5000, 0.002): |V|={}, |E|={}",
        graph.num_vertices, graph.num_edges
    );

    for &s_count in &BATCH_SIZES {
        let sources: Vec<u32> = (0..s_count).map(|i| ((i * 37) % n) as u32).collect();

        let (dist, dur_fifo, dur_rayon, dur_bit) = assert_cpu_3way_equality(&graph, &sources);
        assert_triangle_inequality(&graph, &sources, &dist);

        println!(
            "  ER 5000      [S={:>3}]: FIFO={:>6.2}ms, Rayon={:>6.2}ms, Bit-Parallel={:>6.2}ms (All Passed)",
            s_count, dur_fifo, dur_rayon, dur_bit
        );
    }
}

#[test]
fn test_stress_rmat_scale12() {
    println!(
        "\n=== TEST: Graph500 R-MAT Scale-Free Power-Law (Scale=12, N=4,096, EdgeFactor=16) ==="
    );
    let scale = 12;
    let edge_factor = 16;
    let graph =
        generate_graph500_rmat(scale, edge_factor, Some(888123)).expect("R-MAT generation failed");
    let n = graph.num_vertices;
    assert_eq!(n, 4096);
    println!(
        "  Constructed R-MAT Scale 12: |V|={}, |E|={}, Max Degree={}, Avg Degree={:.1}",
        n,
        graph.num_edges,
        graph.max_degree(),
        graph.avg_degree()
    );

    for &s_count in &BATCH_SIZES {
        let sources: Vec<u32> = (0..s_count).map(|i| ((i * 31) % n) as u32).collect();

        let (dist, dur_fifo, dur_rayon, dur_bit) = assert_cpu_3way_equality(&graph, &sources);
        assert_triangle_inequality(&graph, &sources, &dist);

        println!(
            "  R-MAT S12    [S={:>3}]: FIFO={:>6.2}ms, Rayon={:>6.2}ms, Bit-Parallel={:>6.2}ms (All Passed)",
            s_count, dur_fifo, dur_rayon, dur_bit
        );
    }
}

#[test]
fn test_stress_boundary_and_corner_cases() {
    println!("\n=== TEST: Boundary and Corner Topologies ===");

    // 1. Empty Graph
    let g_empty = empty_graph();
    let empty_res = assert_cpu_3way_equality(&g_empty, &[]);
    assert!(empty_res.0.is_empty());

    // 2. Single Vertex with various batch sizes
    let g_single = single_vertex();
    for &s_count in &[1, 3, 32, 64, 65, 128] {
        let sources = vec![0u32; s_count];
        let (dist, _, _, _) = assert_cpu_3way_equality(&g_single, &sources);
        assert_eq!(dist.len(), s_count);
        for &d in &dist {
            assert_eq!(d, 0);
        }
        assert_triangle_inequality(&g_single, &sources, &dist);
    }

    // 3. Isolated Vertices (N=200, E=0)
    let g_iso = isolated_vertices(200);
    for &s_count in &[1, 17, 65] {
        let sources: Vec<u32> = (0..s_count).map(|i| (i * 3) as u32).collect();
        let (dist, _, _, _) = assert_cpu_3way_equality(&g_iso, &sources);
        for (s_idx, &src) in sources.iter().enumerate() {
            for v in 0..200 {
                let expected = if v as u32 == src { 0 } else { u32::MAX };
                assert_eq!(dist[s_idx * 200 + v], expected);
            }
        }
        assert_triangle_inequality(&g_iso, &sources, &dist);
    }

    // 4. Large duplicate sources batch (S=128 identical sources)
    let g_small = generate_grid2d(4, 4);
    let dup_sources = vec![5u32; 128];
    let (dist_dup, _, _, _) = assert_cpu_3way_equality(&g_small, &dup_sources);
    assert_eq!(dist_dup.len(), 128 * 16);
    let slice0 = &dist_dup[0..16];
    for s_idx in 1..128 {
        assert_eq!(&dist_dup[s_idx * 16..(s_idx + 1) * 16], slice0);
    }
    assert_triangle_inequality(&g_small, &dup_sources, &dist_dup);

    // 5. Out-of-bounds error verification across all engines
    let g_grid = generate_grid2d(3, 3); // 9 vertices
    let bad_sources = vec![0, 9];
    assert!(ms_bfs_fifo(&g_grid, &bad_sources).is_err());
    assert!(ms_bfs_rayon(&g_grid, &bad_sources).is_err());
    assert!(ms_bfs_bit_parallel(&g_grid, &bad_sources).is_err());

    // 6. CpuBfsEngine dynamic dispatch
    let engine_fifo = CpuBfsEngine::with_algorithm(CpuBfsAlgorithm::SequentialFifo);
    let engine_rayon = CpuBfsEngine::with_algorithm(CpuBfsAlgorithm::RayonBatched);
    let engine_bit = CpuBfsEngine::with_algorithm(CpuBfsAlgorithm::BitParallel64);

    let res_e_fifo = engine_fifo.run_ms_bfs(&g_grid, &[0, 4, 8]).unwrap();
    let res_e_rayon = engine_rayon.run_ms_bfs(&g_grid, &[0, 4, 8]).unwrap();
    let res_e_bit = engine_bit.run_ms_bfs(&g_grid, &[0, 4, 8]).unwrap();
    assert_eq!(res_e_fifo, res_e_rayon);
    assert_eq!(res_e_fifo, res_e_bit);

    println!("  All boundary and corner tests passed successfully!");
}
