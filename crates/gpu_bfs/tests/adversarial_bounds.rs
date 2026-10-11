#![allow(clippy::identity_op, clippy::erasing_op)]

//! Adversarial bounds and edge case stress tests for GPU MS-BFS pipelines.
//!
//! Verified dimensions:
//! 1. Non-power-of-two batch sizes: S in {3, 5, 9, 17, 31}
//! 2. Boundary graph topologies:
//!    - Zero edges (N = 100, isolated vertices)
//!    - Single vertex (N = 1)
//!    - Linear chain (N = 500, diameter = 499)
//!    - Star graph (N = 500, hub degree = 499, diameter = 2)
//! 3. Out-of-bounds source validation (asserts Err returned)
//! 4. Duplicate sources in input slice ([0, 0, 5, 5, ...])
//! 5. Early termination loop breaking and timing verification

use std::sync::Arc;
use std::time::Instant;
use webgpu_bfs_core::{
    generators::{
        empty_graph, generate_erdos_renyi, generate_grid2d, isolated_vertices, path_graph,
        single_vertex, star_graph, Grid2D,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

/// Asserts 100% bitwise distance equivalence between GPU and CPU engines.
fn assert_gpu_cpu_equivalence(
    gpu_engine: &GpuBfsEngine,
    cpu_engine: &CpuBfsEngine,
    graph: &CsrGraph,
    sources: &[u32],
) {
    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .expect("CPU MS-BFS failed");
    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .expect("GPU MS-BFS failed");

    assert_eq!(
        gpu_dist.len(),
        cpu_dist.len(),
        "Output length mismatch for |V|={}, S={}",
        graph.num_vertices,
        sources.len()
    );

    for (s_idx, &src) in sources.iter().enumerate() {
        for v in 0..graph.num_vertices {
            let idx = s_idx * graph.num_vertices + v;
            let c = cpu_dist[idx];
            let g = gpu_dist[idx];
            if c != g {
                panic!(
                    "Distance mismatch! Method={:?}, Source idx {} (id {}), Vertex {}: CPU={}, GPU={}",
                    gpu_engine.method(),
                    s_idx,
                    src,
                    v,
                    c,
                    g
                );
            }
        }
    }
}

#[test]
fn test_bounds_non_power_of_two_batch_sizes() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    // 1. Test across Grid2D (15x15 = 225 vertices) with analytical Manhattan oracle
    let width = 15;
    let height = 15;
    let grid = Grid2D::new(width, height).expect("Grid2D creation failed");
    let grid_graph = &grid.graph;

    // Non-power-of-two batch sizes: S in {3, 5, 9, 17, 31}
    let odd_batch_sizes = [3, 5, 9, 17, 31];

    for &s_count in &odd_batch_sizes {
        let sources: Vec<u32> = (0..s_count)
            .map(|i| (i * 37 + 13) % grid_graph.num_vertices as u32)
            .collect();

        for method in [
            GpuBfsMethod::NaiveSequential,
            GpuBfsMethod::BatchedDenseSpMM,
        ] {
            let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();

            // CPU equivalence check
            assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, grid_graph, &sources);

            // Analytical Manhattan oracle check
            let gpu_dist = gpu_engine.run_ms_bfs(grid_graph, &sources).unwrap();
            assert_eq!(gpu_dist.len(), (s_count as usize) * grid_graph.num_vertices);

            for (s_idx, &src) in sources.iter().enumerate() {
                for v in 0..grid_graph.num_vertices {
                    let oracle_dist = grid.manhattan_distance(src, v as u32);
                    let computed = gpu_dist[s_idx * grid_graph.num_vertices + v];
                    assert_eq!(
                        computed, oracle_dist,
                        "Manhattan mismatch at S={}, method={:?}, src={}, v={}",
                        s_count, method, src, v
                    );
                }
            }
        }
    }

    // 2. Also test across Erdős-Rényi random graph (N=200, p=0.08)
    let er_graph = generate_erdos_renyi(200, 0.08, 99991);
    for &s_count in &odd_batch_sizes {
        let sources: Vec<u32> = (0..s_count)
            .map(|i| (i * 29 + 7) % er_graph.num_vertices as u32)
            .collect();

        for method in [
            GpuBfsMethod::NaiveSequential,
            GpuBfsMethod::BatchedDenseSpMM,
        ] {
            let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
            assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &er_graph, &sources);
        }
    }
}

#[test]
fn test_bounds_isolated_zero_edges_graph() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    // 100 vertices, 0 edges
    let graph = isolated_vertices(100);
    assert_eq!(graph.num_vertices, 100);
    assert_eq!(graph.num_edges, 0);

    let sources = vec![0, 23, 77, 99];

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);

        let gpu_dist = gpu_engine.run_ms_bfs(&graph, &sources).unwrap();
        assert_eq!(gpu_dist.len(), sources.len() * 100);

        for (s_idx, &src) in sources.iter().enumerate() {
            for v in 0..100 {
                let dist = gpu_dist[s_idx * 100 + v];
                if v as u32 == src {
                    assert_eq!(dist, 0, "Self-distance must be 0 for source {}", src);
                } else {
                    assert_eq!(
                        dist,
                        u32::MAX,
                        "Unreachable vertex {} from source {} must be u32::MAX in 0-edge graph",
                        v,
                        src
                    );
                }
            }
        }
    }
}

#[test]
fn test_bounds_single_vertex_graph() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let single_graph = single_vertex();
    assert_eq!(single_graph.num_vertices, 1);
    assert_eq!(single_graph.num_edges, 0);

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();

        // Single source [0]
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &single_graph, &[0]);
        let dist = gpu_engine.run_ms_bfs(&single_graph, &[0]).unwrap();
        assert_eq!(dist, vec![0]);

        // Duplicate sources on single vertex [0, 0, 0]
        let multi_sources = vec![0, 0, 0];
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &single_graph, &multi_sources);
        let dist_multi = gpu_engine
            .run_ms_bfs(&single_graph, &multi_sources)
            .unwrap();
        assert_eq!(dist_multi, vec![0, 0, 0]);
    }
}

#[test]
fn test_bounds_linear_chain_high_diameter() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    // Linear chain with 500 vertices (diameter = 499)
    let n = 500;
    let chain = path_graph(n);
    assert_eq!(chain.num_vertices, n);
    assert_eq!(chain.num_edges, 2 * (n - 1));

    let sources = vec![0, 249, 499];

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();

        // Equivalence with CPU
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &chain, &sources);

        // Analytical linear chain oracle check: dist(u, v) = |u - v|
        let dist = gpu_engine.run_ms_bfs(&chain, &sources).unwrap();
        assert_eq!(dist.len(), sources.len() * n);

        for (s_idx, &src) in sources.iter().enumerate() {
            for v in 0..n {
                let expected = (src as i32 - v as i32).unsigned_abs();
                let actual = dist[s_idx * n + v];
                assert_eq!(
                    actual, expected,
                    "Chain oracle mismatch at method={:?}, src={}, v={}: expected {}, got {}",
                    method, src, v, expected, actual
                );
            }
        }
    }
}

#[test]
fn test_bounds_star_graph_extreme_degree() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    // Star graph with 500 vertices (hub 0 has degree 499, leaves 1..499 have degree 1)
    let n = 500;
    let star = star_graph(n);
    assert_eq!(star.num_vertices, n);
    assert_eq!(star.num_edges, 2 * (n - 1));
    assert_eq!(star.degree(0), 499);

    let sources = vec![0, 1, 100, 499];

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();

        // Equivalence with CPU reference
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &star, &sources);

        // Analytical star graph oracle:
        // - dist(0, 0) = 0
        // - dist(0, v) = 1 for v > 0
        // - dist(L, L) = 0 for leaf L > 0
        // - dist(L, 0) = 1 for leaf L > 0
        // - dist(L, v) = 2 for leaf L > 0 and leaf v != L
        let dist = gpu_engine.run_ms_bfs(&star, &sources).unwrap();
        for (s_idx, &src) in sources.iter().enumerate() {
            for v in 0..n as u32 {
                let expected = if src == 0 {
                    if v == 0 {
                        0
                    } else {
                        1
                    }
                } else if v == src {
                    0
                } else if v == 0 {
                    1
                } else {
                    2
                };

                let actual = dist[s_idx * n + v as usize];
                assert_eq!(
                    actual, expected,
                    "Star oracle mismatch at method={:?}, src={}, v={}: expected {}, got {}",
                    method, src, v, expected, actual
                );
            }
        }
    }
}

#[test]
fn test_bounds_out_of_bounds_source_validation() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let graph = generate_grid2d(10, 10); // N = 100

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();

        // 1. Source exactly equal to N (boundary index)
        let res = gpu_engine.run_ms_bfs(&graph, &[100]);
        assert!(res.is_err(), "Source == N must return Err");
        let err_msg = res.unwrap_err().to_string();
        assert!(
            err_msg.contains("exceeds graph vertex count 100"),
            "Error message should clearly describe bound violation: {}",
            err_msg
        );

        // 2. Source strictly greater than N
        assert!(gpu_engine.run_ms_bfs(&graph, &[101]).is_err());
        assert!(gpu_engine.run_ms_bfs(&graph, &[9999]).is_err());
        assert!(gpu_engine.run_ms_bfs(&graph, &[u32::MAX]).is_err());

        // 3. Slice with one valid and one invalid source
        assert!(gpu_engine.run_ms_bfs(&graph, &[0, 50, 100]).is_err());
        assert!(gpu_engine.run_ms_bfs(&graph, &[105, 0]).is_err());

        // 4. Graph with 0 vertices
        let empty = empty_graph();
        assert!(
            gpu_engine.run_ms_bfs(&empty, &[0]).is_err(),
            "Source 0 on empty graph must return Err"
        );

        // 5. Empty sources slice should succeed and return empty Vec
        let empty_res = gpu_engine.run_ms_bfs(&graph, &[]);
        assert!(empty_res.is_ok());
        assert_eq!(empty_res.unwrap(), Vec::<u32>::new());
    }
}

#[test]
fn test_bounds_duplicate_sources_independence() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let grid = Grid2D::new(8, 8).unwrap();
    let graph = &grid.graph;
    let nv = graph.num_vertices;

    // Sources with multiple repeats: 0 appears twice, 15 appears 3 times, 63 appears twice
    let sources = vec![0, 0, 15, 15, 15, 63, 63];

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();

        // Check equivalence with CPU reference
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, graph, &sources);

        let dist = gpu_engine.run_ms_bfs(graph, &sources).unwrap();
        assert_eq!(dist.len(), sources.len() * nv);

        // Verify that duplicate source rows are completely identical
        let row_0_a = &dist[0 * nv..1 * nv];
        let row_0_b = &dist[1 * nv..2 * nv];
        assert_eq!(row_0_a, row_0_b, "Duplicate source 0 rows must match");

        let row_15_a = &dist[2 * nv..3 * nv];
        let row_15_b = &dist[3 * nv..4 * nv];
        let row_15_c = &dist[4 * nv..5 * nv];
        assert_eq!(row_15_a, row_15_b, "Duplicate source 15 rows must match");
        assert_eq!(row_15_b, row_15_c, "Duplicate source 15 rows must match");

        let row_63_a = &dist[5 * nv..6 * nv];
        let row_63_b = &dist[6 * nv..7 * nv];
        assert_eq!(row_63_a, row_63_b, "Duplicate source 63 rows must match");
    }
}

#[test]
fn test_bounds_early_termination_and_timing_verification() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));

    let n = 500;
    let star = star_graph(n);
    let chain = path_graph(n);
    let iso = isolated_vertices(n);

    // Warm-up pipeline and GPU context
    let batched_engine = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let _ = batched_engine.run_ms_bfs(&star, &[0]).unwrap();

    // 1. Isolated vertices: early termination breaks immediately after 1 iteration (counter == 0)
    let t0 = Instant::now();
    let iso_dist = batched_engine.run_ms_bfs(&iso, &[0]).unwrap();
    let t_iso = t0.elapsed();
    assert_eq!(iso_dist[0], 0);
    assert_eq!(iso_dist[1], u32::MAX);

    // 2. Star graph with center source 0: diameter is 1, breaks after 2 iterations
    let t1 = Instant::now();
    let star_dist = batched_engine.run_ms_bfs(&star, &[0]).unwrap();
    let t_star = t1.elapsed();
    assert_eq!(star_dist[0], 0);
    assert_eq!(star_dist[1], 1);
    assert_eq!(star_dist[499], 1);

    // 3. Linear chain with source 0: diameter is 499, requires 500 iterations
    let t2 = Instant::now();
    let chain_dist = batched_engine.run_ms_bfs(&chain, &[0]).unwrap();
    let t_chain = t2.elapsed();
    assert_eq!(chain_dist[0], 0);
    assert_eq!(chain_dist[499], 499);

    println!(
        "\n[TIMING EVIDENCE] Early Termination Loop Breaking on N=500 graphs (BatchedDenseSpMM):"
    );
    println!("  Isolated graph (0 edges, 1 iter):     {:?}", t_iso);
    println!("  Star graph (diameter 1, 2 iters):      {:?}", t_star);
    println!("  Linear chain (diameter 499, 500 iters): {:?}", t_chain);

    // Star graph (2 iterations) should be an order of magnitude faster than chain (500 iterations)
    assert!(
        t_star < t_chain,
        "Star graph ({:?}) must terminate significantly faster than chain ({:?})",
        t_star,
        t_chain
    );

    let speedup = t_chain.as_secs_f64() / t_star.as_secs_f64().max(1e-6);
    println!(
        "  Early termination speedup (Chain / Star): {:.2}x",
        speedup
    );
    assert!(
        speedup > 3.0,
        "Early termination should yield at least 3x speedup on diameter 1 vs 499, observed: {:.2}x",
        speedup
    );

    // 4. Test Early Termination on NaiveSequential as well
    let naive_engine = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let _ = naive_engine.run_ms_bfs(&star, &[0]).unwrap();

    let t0_naive = Instant::now();
    let iso_naive = naive_engine.run_ms_bfs(&iso, &[0]).unwrap();
    let t_iso_naive = t0_naive.elapsed();
    assert_eq!(iso_naive[0], 0);
    assert_eq!(iso_naive[1], u32::MAX);

    let t1_naive = Instant::now();
    let star_naive = naive_engine.run_ms_bfs(&star, &[0]).unwrap();
    let t_star_naive = t1_naive.elapsed();
    assert_eq!(star_naive[0], 0);
    assert_eq!(star_naive[1], 1);

    let t2_naive = Instant::now();
    let chain_naive = naive_engine.run_ms_bfs(&chain, &[0]).unwrap();
    let t_chain_naive = t2_naive.elapsed();
    assert_eq!(chain_naive[0], 0);
    assert_eq!(chain_naive[499], 499);

    println!(
        "\n[TIMING EVIDENCE] Early Termination Loop Breaking on N=500 graphs (NaiveSequential):"
    );
    println!("  Isolated graph (0 edges, 1 iter):     {:?}", t_iso_naive);
    println!(
        "  Star graph (diameter 1, 2 iters):      {:?}",
        t_star_naive
    );
    println!(
        "  Linear chain (diameter 499, 500 iters): {:?}",
        t_chain_naive
    );

    let speedup_naive = t_chain_naive.as_secs_f64() / t_star_naive.as_secs_f64().max(1e-6);
    println!(
        "  Early termination speedup (Naive Chain / Star): {:.2}x",
        speedup_naive
    );
    assert!(
        speedup_naive > 3.0,
        "Early termination on NaiveSequential should yield significant speedup, observed: {:.2}x",
        speedup_naive
    );
}
