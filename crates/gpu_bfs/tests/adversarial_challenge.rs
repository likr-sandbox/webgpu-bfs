#![allow(clippy::identity_op, clippy::erasing_op)]

//! Adversarial Empirical Correctness & Stress Harness for Milestone 2 GPU BFS.
//!
//! Evaluates both `NaiveSequential` and `BatchedDenseSpMM` GPU implementations against
//! `CpuBfsEngine` and analytical oracles across:
//! 1. 2D Grid graphs (analytical Manhattan distance oracle + CPU reference).
//! 2. Random Trees (e.g., N=500 deep diameter irregular branching).
//! 3. Erdős-Rényi random graphs (N=1,000, p=0.005, multiple components & hubs).
//! 4. Disconnected components (strictly assert u32::MAX for unreachable vertices).
//! 5. High-diameter path graphs (255 sequential compute iterations).
//! 6. Star graphs (512 vertices, high-degree hub contention).
//! 7. Graph500 R-MAT power-law scale-free graphs (scale 9, 4096 edges).
//! 8. Degree-0 isolated source vertices (early termination, zero frontier).
//! 9. Duplicate sources in batch (contention and idempotence).
//! 10. Large batch size S=64.
//! 11. Batch sizes S in {1, 2, 7, 16, 32} across all topologies.

use std::{sync::Arc, time::Instant};
use webgpu_bfs_core::{
    generators::{
        disconnected_cliques, generate_erdos_renyi_np, generate_graph500_rmat,
        generate_random_tree, isolated_vertices, path_graph, star_graph, Grid2D,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

const BATCH_SIZES: [usize; 5] = [1, 2, 7, 16, 32];

/// Result metrics collected during an adversarial 3-way test.
#[allow(dead_code)]
struct TestMetric {
    s_count: usize,
    cpu_ms: f64,
    naive_gpu_ms: f64,
    batched_gpu_ms: f64,
}

/// Executes a 3-way equivalence verification:
/// CPU Reference vs. GPU NaiveSequential vs. GPU BatchedDenseSpMM.
///
/// Asserts:
/// 1. Output length == S * |V|
/// 2. Source vertex distance == 0: dist[s_idx * |V| + sources[s_idx]] == 0
/// 3. Bitwise equality between CPU Reference and GPU NaiveSequential
/// 4. Bitwise equality between CPU Reference and GPU BatchedDenseSpMM
/// 5. Metric triangle inequality for all graph edges: |dist(u) - dist(v)| <= 1
fn assert_3way_equivalence(
    gpu_naive: &GpuBfsEngine,
    gpu_batched: &GpuBfsEngine,
    cpu_engine: &CpuBfsEngine,
    graph: &CsrGraph,
    sources: &[u32],
    tag: &str,
) -> TestMetric {
    let nv = graph.num_vertices;
    let s_count = sources.len();

    // 1. CPU Reference Execution
    let t0 = Instant::now();
    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] CPU BFS failed: {e}"));
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // 2. GPU NaiveSequential Execution
    let t1 = Instant::now();
    let naive_dist = gpu_naive
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] GPU NaiveSequential failed: {e}"));
    let naive_gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // 3. GPU BatchedDenseSpMM Execution
    let t2 = Instant::now();
    let batched_dist = gpu_batched
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] GPU BatchedDenseSpMM failed: {e}"));
    let batched_gpu_ms = t2.elapsed().as_secs_f64() * 1000.0;

    // Length assertions
    assert_eq!(
        cpu_dist.len(),
        s_count * nv,
        "[{tag}] CPU dist length mismatch"
    );
    assert_eq!(
        naive_dist.len(),
        s_count * nv,
        "[{tag}] GPU Naive dist length mismatch"
    );
    assert_eq!(
        batched_dist.len(),
        s_count * nv,
        "[{tag}] GPU Batched dist length mismatch"
    );

    // Verify bitwise equivalence and contracts
    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * nv;

        // Source vertex must have distance 0
        assert_eq!(
            cpu_dist[base + src as usize],
            0,
            "[{tag}] Source {src} distance != 0 on CPU"
        );
        assert_eq!(
            naive_dist[base + src as usize],
            0,
            "[{tag}] Source {src} distance != 0 on GPU Naive"
        );
        assert_eq!(
            batched_dist[base + src as usize],
            0,
            "[{tag}] Source {src} distance != 0 on GPU Batched"
        );

        // Bitwise comparison across all vertices
        for v in 0..nv {
            let idx = base + v;
            let c = cpu_dist[idx];
            let n = naive_dist[idx];
            let b = batched_dist[idx];

            if n != c {
                panic!(
                    "[{tag}] DISCREPANCY CPU vs NaiveGPU: S={}, s_idx={}, src={}, v={}, CPU={}, NaiveGPU={}",
                    s_count, s_idx, src, v, c, n
                );
            }
            if b != c {
                panic!(
                    "[{tag}] DISCREPANCY CPU vs BatchedGPU: S={}, s_idx={}, src={}, v={}, CPU={}, BatchedGPU={}",
                    s_count, s_idx, src, v, c, b
                );
            }
        }

        // Triangle inequality verification for all undirected edges
        for u in 0..nv {
            let start = graph.row_offsets[u] as usize;
            let end = graph.row_offsets[u + 1] as usize;
            let d_u = batched_dist[base + u];

            for &v_neighbor in &graph.col_indices[start..end] {
                let d_v = batched_dist[base + v_neighbor as usize];
                if d_u == u32::MAX {
                    assert_eq!(
                        d_v,
                        u32::MAX,
                        "[{tag}] Inconsistency: vertex {u} is unreachable but neighbor {v_neighbor} is reachable (dist={d_v})"
                    );
                } else {
                    assert_ne!(
                        d_v,
                        u32::MAX,
                        "[{tag}] Inconsistency: vertex {u} is reachable (dist={d_u}) but neighbor {v_neighbor} is unreachable"
                    );
                    let diff = (d_u as i64 - d_v as i64).abs();
                    assert!(
                        diff <= 1,
                        "[{tag}] Triangle inequality violated on edge ({u}, {v_neighbor}): dist({u})={d_u}, dist({v_neighbor})={d_v}"
                    );
                }
            }
        }
    }

    TestMetric {
        s_count,
        cpu_ms,
        naive_gpu_ms,
        batched_gpu_ms,
    }
}

/// Generates diverse source indices for a graph of size N.
fn pick_sources(n: usize, s_count: usize, seed_multiplier: usize) -> Vec<u32> {
    assert!(n > 0);
    if s_count == 1 {
        return vec![0];
    }
    let step = n / s_count;
    let mut sources = Vec::with_capacity(s_count);
    for i in 0..s_count {
        let raw = (i * step + (i * seed_multiplier) % n.max(1)) % n;
        sources.push(raw as u32);
    }
    sources
}

#[test]
fn test_adversarial_grid2d_with_manhattan_oracle() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    // 25x25 grid (625 vertices, 2400 directed edges)
    let width = 25;
    let height = 25;
    let grid = Grid2D::new(width, height).unwrap();
    let graph = &grid.graph;

    println!(
        "\n--- [ADVERSARIAL SUITE] 1. 2D Grid ({}x{}, |V|={}, |E|={}) ---",
        width, height, graph.num_vertices, graph.num_edges
    );

    for &s in &BATCH_SIZES {
        let sources = pick_sources(graph.num_vertices, s, 7);
        let metric = assert_3way_equivalence(
            &gpu_naive,
            &gpu_batched,
            &cpu_engine,
            graph,
            &sources,
            "Grid2D",
        );

        // Ground-truth analytical Manhattan distance oracle verification
        let batched_dist = gpu_batched.run_ms_bfs(graph, &sources).unwrap();
        for (s_idx, &src) in sources.iter().enumerate() {
            let base = s_idx * graph.num_vertices;
            for v in 0..graph.num_vertices {
                let computed = batched_dist[base + v];
                let analytical = grid.manhattan_distance(src, v as u32);
                assert_eq!(
                    computed, analytical,
                    "Manhattan distance oracle failure: src={src}, v={v}, computed={computed}, expected={analytical}"
                );
            }
        }

        println!(
            "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | Manhattan Oracle: PASS 100%",
            s, metric.cpu_ms, metric.naive_gpu_ms, metric.batched_gpu_ms
        );
    }
}

#[test]
fn test_adversarial_random_tree_deep_branching() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    // Random Tree N=500 (499 undirected edges, 998 directed edges)
    let n = 500;
    let graph = generate_random_tree(n, Some(20261010)).expect("Random Tree generation failed");

    println!(
        "\n--- [ADVERSARIAL SUITE] 2. Random Tree (|V|={}, |E|={}) ---",
        graph.num_vertices, graph.num_edges
    );

    for &s in &BATCH_SIZES {
        let sources = pick_sources(n, s, 13);
        let metric = assert_3way_equivalence(
            &gpu_naive,
            &gpu_batched,
            &cpu_engine,
            &graph,
            &sources,
            "RandomTree",
        );

        // Every node in a tree is reachable from any source
        let batched_dist = gpu_batched.run_ms_bfs(&graph, &sources).unwrap();
        for (s_idx, &src) in sources.iter().enumerate() {
            let base = s_idx * n;
            for v in 0..n {
                let d = batched_dist[base + v];
                assert_ne!(
                    d,
                    u32::MAX,
                    "Tree vertex {v} was unreachable from source {src}!"
                );
            }
        }

        println!(
            "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | Tree Equivalence: PASS 100%",
            s, metric.cpu_ms, metric.naive_gpu_ms, metric.batched_gpu_ms
        );
    }
}

#[test]
fn test_adversarial_erdos_renyi_sparse() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    // Erdős-Rényi N=1,000, p=0.005 (average degree ~5.0)
    let n = 1000;
    let p = 0.005;
    let graph = generate_erdos_renyi_np(n, p, Some(20261010)).expect("ER generation failed");

    println!(
        "\n--- [ADVERSARIAL SUITE] 3. Erdős-Rényi Random Graph (|V|={}, p={}, |E|={}) ---",
        graph.num_vertices, p, graph.num_edges
    );

    for &s in &BATCH_SIZES {
        let sources = pick_sources(n, s, 37);
        let metric = assert_3way_equivalence(
            &gpu_naive,
            &gpu_batched,
            &cpu_engine,
            &graph,
            &sources,
            "ErdosRenyi",
        );

        println!(
            "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | Bitwise Equivalence: PASS 100%",
            s, metric.cpu_ms, metric.naive_gpu_ms, metric.batched_gpu_ms
        );
    }
}

#[test]
fn test_adversarial_disconnected_components_sentinel() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    // 5 disconnected cliques of sizes [20, 50, 100, 150, 200] (|V| = 520)
    let clique_sizes = vec![20, 50, 100, 150, 200];
    let graph = disconnected_cliques(&clique_sizes);
    let total_v = graph.num_vertices;

    println!(
        "\n--- [ADVERSARIAL SUITE] 4. Disconnected Cliques ({:?}, |V|={}, |E|={}) ---",
        clique_sizes, total_v, graph.num_edges
    );

    // Calculate component start and end ranges
    let mut comp_ranges = Vec::new();
    let mut cur = 0;
    for &sz in &clique_sizes {
        comp_ranges.push(cur..cur + sz);
        cur += sz;
    }

    for &s in &BATCH_SIZES {
        // Pick sources distributed across the 5 disconnected components
        let mut sources = Vec::new();
        for i in 0..s {
            let comp_idx = i % clique_sizes.len();
            let range = &comp_ranges[comp_idx];
            let src = (range.start + (i * 7) % (range.end - range.start)) as u32;
            sources.push(src);
        }

        let metric = assert_3way_equivalence(
            &gpu_naive,
            &gpu_batched,
            &cpu_engine,
            &graph,
            &sources,
            "DisconnectedCliques",
        );

        // Explicitly assert that vertices outside the source's component STRICTLY receive u32::MAX
        let batched_dist = gpu_batched.run_ms_bfs(&graph, &sources).unwrap();
        for (s_idx, &src) in sources.iter().enumerate() {
            let src_u = src as usize;
            let comp_idx = comp_ranges
                .iter()
                .position(|r| r.contains(&src_u))
                .expect("Source must belong to a component");
            let valid_range = &comp_ranges[comp_idx];

            let base = s_idx * total_v;
            let mut unreachable_count = 0;
            for v in 0..total_v {
                let d = batched_dist[base + v];
                if valid_range.contains(&v) {
                    if v == src_u {
                        assert_eq!(d, 0, "Source distance must be 0");
                    } else {
                        assert_eq!(d, 1, "Distance to vertex in same clique must be 1");
                    }
                } else {
                    assert_eq!(
                        d,
                        u32::MAX,
                        "Cross-component vertex {v} (src {src} in comp {comp_idx}) MUST be u32::MAX, got {d}"
                    );
                    unreachable_count += 1;
                }
            }

            let expected_unreachable = total_v - clique_sizes[comp_idx];
            assert_eq!(
                unreachable_count, expected_unreachable,
                "Unreachable vertex count mismatch for src {src}"
            );
        }

        println!(
            "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | Sentinel u32::MAX: PASS 100%",
            s, metric.cpu_ms, metric.naive_gpu_ms, metric.batched_gpu_ms
        );
    }
}

#[test]
fn test_adversarial_deep_wavefront_path_graph() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    // Linear path graph N=256: diameter is 255. Requires 255 sequential compute steps!
    let n = 256;
    let graph = path_graph(n);

    println!(
        "\n--- [ADVERSARIAL SUITE] 5. High-Diameter Path Graph (|V|={}, Diameter={}) ---",
        graph.num_vertices,
        n - 1
    );

    // Extreme sources: endpoint 0, opposite endpoint n-1, midpoint
    let sources = vec![0, (n - 1) as u32, (n / 2) as u32];
    let metric = assert_3way_equivalence(
        &gpu_naive,
        &gpu_batched,
        &cpu_engine,
        &graph,
        &sources,
        "PathGraph",
    );

    let batched_dist = gpu_batched.run_ms_bfs(&graph, &sources).unwrap();
    // Endpoint 0 to Endpoint n-1 must equal n - 1
    assert_eq!(batched_dist[0 * n + (n - 1)], (n - 1) as u32);
    // Endpoint n-1 to Endpoint 0 must equal n - 1
    assert_eq!(batched_dist[1 * n + 0], (n - 1) as u32);

    println!(
        "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | 255 Compute Steps: PASS",
        sources.len(),
        metric.cpu_ms,
        metric.naive_gpu_ms,
        metric.batched_gpu_ms
    );
}

#[test]
fn test_adversarial_high_degree_star_graph() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    // Star graph: Hub 0 connected to 511 leaves
    let n = 512;
    let graph = star_graph(n);

    println!(
        "\n--- [ADVERSARIAL SUITE] 6. High-Degree Star Graph (|V|={}, Hub Degree={}) ---",
        graph.num_vertices,
        n - 1
    );

    // Sources: Hub (node 0) and various leaves (1, 100, 250, 500)
    let sources = vec![0, 1, 100, 250, 500];
    let metric = assert_3way_equivalence(
        &gpu_naive,
        &gpu_batched,
        &cpu_engine,
        &graph,
        &sources,
        "StarGraph",
    );

    let batched_dist = gpu_batched.run_ms_bfs(&graph, &sources).unwrap();
    // From hub (source 0), all leaves must have distance 1
    for leaf in 1..n {
        assert_eq!(
            batched_dist[0 * n + leaf],
            1,
            "Hub to leaf {leaf} distance must be 1"
        );
    }
    // From leaf 1 (source 1), hub has distance 1, all other leaves have distance 2
    assert_eq!(batched_dist[1 * n + 0], 1, "Leaf to hub distance must be 1");
    for other_leaf in 2..n {
        assert_eq!(
            batched_dist[1 * n + other_leaf],
            2,
            "Leaf to leaf distance must be 2"
        );
    }

    println!(
        "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | Hub Contention: PASS",
        sources.len(),
        metric.cpu_ms,
        metric.naive_gpu_ms,
        metric.batched_gpu_ms
    );
}

#[test]
fn test_adversarial_isolated_degree_zero_sources() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    // Graph with 50 completely isolated vertices (0 edges)
    let n = 50;
    let graph = isolated_vertices(n);

    println!(
        "\n--- [ADVERSARIAL SUITE] 7. Completely Isolated Graph (|V|={}, |E|=0) ---",
        graph.num_vertices
    );

    let sources = vec![0, 10, 25, 49];
    let metric = assert_3way_equivalence(
        &gpu_naive,
        &gpu_batched,
        &cpu_engine,
        &graph,
        &sources,
        "IsolatedGraph",
    );

    let batched_dist = gpu_batched.run_ms_bfs(&graph, &sources).unwrap();
    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * n;
        for v in 0..n {
            let d = batched_dist[base + v];
            if v == src as usize {
                assert_eq!(d, 0, "Source distance must be 0");
            } else {
                assert_eq!(
                    d,
                    u32::MAX,
                    "Isolated vertex {v} must be unreachable (u32::MAX)"
                );
            }
        }
    }

    println!(
        "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | Zero Edge Termination: PASS",
        sources.len(),
        metric.cpu_ms,
        metric.naive_gpu_ms,
        metric.batched_gpu_ms
    );
}

#[test]
fn test_adversarial_duplicate_sources_idempotence() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    // Grid 15x15 with batch of 7 identical sources [42, 42, 42, 42, 42, 42, 42]
    let grid = Grid2D::new(15, 15).unwrap();
    let graph = &grid.graph;
    let nv = graph.num_vertices;

    println!("\n--- [ADVERSARIAL SUITE] 8. Duplicate Sources Idempotence (S=7, all src=42) ---");

    let sources = vec![42u32; 7];
    let metric = assert_3way_equivalence(
        &gpu_naive,
        &gpu_batched,
        &cpu_engine,
        graph,
        &sources,
        "DuplicateSources",
    );

    let batched_dist = gpu_batched.run_ms_bfs(graph, &sources).unwrap();
    // All 7 rows must be identical
    let row0 = &batched_dist[0..nv];
    for s_idx in 1..7 {
        let row_s = &batched_dist[s_idx * nv..(s_idx + 1) * nv];
        assert_eq!(
            row0, row_s,
            "Row {s_idx} diverges from row 0 under duplicate source dispatch"
        );
    }

    println!(
        "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | Cross-Row Invariance: PASS",
        sources.len(),
        metric.cpu_ms,
        metric.naive_gpu_ms,
        metric.batched_gpu_ms
    );
}

#[test]
fn test_adversarial_graph500_rmat_powerlaw() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    // Graph500 R-MAT scale 9 (|V| = 512, edge_factor 8 -> ~4096 edges)
    let graph = generate_graph500_rmat(9, 8, Some(20261010)).expect("R-MAT generation failed");

    println!(
        "\n--- [ADVERSARIAL SUITE] 9. Graph500 R-MAT Power-Law (|V|={}, |E|={}) ---",
        graph.num_vertices, graph.num_edges
    );

    for &s in &[1, 7, 32] {
        let sources = pick_sources(graph.num_vertices, s, 19);
        let metric = assert_3way_equivalence(
            &gpu_naive,
            &gpu_batched,
            &cpu_engine,
            &graph,
            &sources,
            "Graph500Rmat",
        );

        println!(
            "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | Power-Law Equivalence: PASS 100%",
            s, metric.cpu_ms, metric.naive_gpu_ms, metric.batched_gpu_ms
        );
    }
}

#[test]
fn test_adversarial_large_batch_size_64() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let gpu_naive = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::NaiveSequential).unwrap();
    let gpu_batched = GpuBfsEngine::new(ctx, GpuBfsMethod::BatchedDenseSpMM).unwrap();
    let cpu_engine = CpuBfsEngine::new();

    let grid = Grid2D::new(16, 16).unwrap();
    let graph = &grid.graph;
    let s = 64;
    let sources = pick_sources(graph.num_vertices, s, 11);

    println!(
        "\n--- [ADVERSARIAL SUITE] 10. Large Batch Size S=64 (|V|={}, S={}) ---",
        graph.num_vertices, s
    );

    let metric = assert_3way_equivalence(
        &gpu_naive,
        &gpu_batched,
        &cpu_engine,
        graph,
        &sources,
        "LargeBatch64",
    );

    println!(
        "  S={:2} | CPU: {:7.2} ms | Naive GPU: {:7.2} ms | Batched GPU: {:7.2} ms | Batch-64 Equivalence: PASS 100%",
        s, metric.cpu_ms, metric.naive_gpu_ms, metric.batched_gpu_ms
    );
}
