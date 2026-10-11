#![allow(clippy::identity_op, clippy::erasing_op)]

//! Comprehensive Empirical Adversarial Correctness & Equivalence Test Suite for Milestone 3.
//!
//! Evaluates ALL 6 WebGPU BFS methods:
//! 1. `NaiveSequential`
//! 2. `BatchedDenseSpMM`
//! 3. `BitParallel32`
//! 4. `BitParallel64`
//! 5. `BooleanReachability`
//! 6. `PushPullAdaptive`
//!
//! against `CpuBfsEngine` ground truth and analytical Manhattan distance oracles across:
//! - Large Grid2D (30x30, 900 vertices, 3480 directed edges)
//! - Random Trees (N=1000, 1998 directed edges, irregular deep branching)
//! - Erdős-Rényi random graphs (N=2000, p=0.004, multiple components and hubs)
//! - Disconnected components (5 disjoint cliques of sizes [50, 100, 150, 200, 250], N=750)
//! - Dense cliques (N=150 fully connected, 22,350 directed edges)
//!
//! Across sweep batch sizes:
//!   S in {1, 3, 17, 32, 33, 64, 65, 128}
//!
//! Strictly asserts:
//! - 100% bitwise distance equality across all 6 GPU methods and CPU baseline.
//! - 0 metric triangle inequality violations on reachable edges (|d(u) - d(v)| <= 1).
//! - Source self-distance invariant (d(s) == 0).
//! - Disconnected unreachable sentinel invariant (d(v) == u32::MAX).

use std::sync::Arc;
use std::time::Instant;
use webgpu_bfs_core::{
    generators::{
        clique_graph, disconnected_cliques, generate_erdos_renyi_np, generate_random_tree, Grid2D,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

const SWEEP_BATCH_SIZES: [usize; 8] = [1, 3, 17, 32, 33, 64, 65, 128];

const ALL_SIX_METHODS: [GpuBfsMethod; 6] = [
    GpuBfsMethod::NaiveSequential,
    GpuBfsMethod::BatchedDenseSpMM,
    GpuBfsMethod::BitParallel32,
    GpuBfsMethod::BitParallel64,
    GpuBfsMethod::BooleanReachability,
    GpuBfsMethod::PushPullAdaptive,
];

/// Deterministically generates S diverse source indices for graph of size N.
fn pick_sources(n: usize, s_count: usize, seed_multiplier: usize) -> Vec<u32> {
    assert!(n > 0);
    if s_count == 1 {
        return vec![0];
    }
    let mut sources = Vec::with_capacity(s_count);
    let step = (n / s_count.min(n)).max(1);
    for i in 0..s_count {
        let raw = (i * step + (i * seed_multiplier)) % n;
        sources.push(raw as u32);
    }
    sources
}

/// Statistics collected during an equivalence test run.
#[derive(Debug, Default)]
struct VerificationStats {
    total_distance_assertions: usize,
    total_triangle_checks: usize,
    total_oracle_checks: usize,
}

/// Runs and validates all 6 GPU BFS methods against CPU ground truth and checks:
/// 1. Length equality
/// 2. Bitwise distance equality across all 6 GPU methods and CPU reference
/// 3. Source self-distance == 0
/// 4. Metric triangle inequality for all reachable edges: |d(u) - d(v)| <= 1
/// 5. Reachability consistency: u reachable <=> v reachable for edge (u, v)
/// 6. Optional analytical oracle verification
fn verify_all_methods_on_graph<F>(
    ctx: &Arc<GpuContext>,
    cpu_engine: &CpuBfsEngine,
    graph: &CsrGraph,
    sources: &[u32],
    topology_tag: &str,
    oracle: Option<F>,
    stats: &mut VerificationStats,
) where
    F: Fn(u32, u32) -> u32,
{
    let nv = graph.num_vertices;
    let s_count = sources.len();

    // 1. CPU Reference Ground Truth
    let t_cpu = Instant::now();
    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{topology_tag}] CPU BFS failed: {e}"));
    let cpu_elapsed = t_cpu.elapsed();

    assert_eq!(
        cpu_dist.len(),
        s_count * nv,
        "[{topology_tag}] CPU distance length mismatch"
    );

    println!(
        "\n[{topology_tag}] S={:<3} (|V|={}, |E|={}) | CPU: {:>8.2} ms",
        s_count,
        nv,
        graph.num_edges,
        cpu_elapsed.as_secs_f64() * 1000.0
    );

    // 2. Execute each of the 6 GPU BFS methods
    for &method in &ALL_SIX_METHODS {
        let engine = GpuBfsEngine::new(ctx.clone(), method)
            .unwrap_or_else(|e| panic!("Failed to instantiate {:?}: {e}", method));

        let t_gpu = Instant::now();
        let gpu_dist = engine
            .run_ms_bfs(graph, sources)
            .unwrap_or_else(|e| panic!("[{topology_tag}] {:?} failed: {e}", method));
        let gpu_elapsed = t_gpu.elapsed();

        assert_eq!(
            gpu_dist.len(),
            s_count * nv,
            "[{topology_tag}] Method {:?} distance length mismatch",
            method
        );

        // Verify source distances and bitwise equality against CPU
        for (s_idx, &src) in sources.iter().enumerate() {
            let base = s_idx * nv;

            // Source self-distance must be 0
            assert_eq!(
                gpu_dist[base + src as usize],
                0,
                "[{topology_tag}] Method {:?} source self-distance != 0 for src {}",
                method,
                src
            );

            // Bitwise equality against CPU baseline
            for v in 0..nv {
                let idx = base + v;
                let c = cpu_dist[idx];
                let g = gpu_dist[idx];

                if g != c {
                    panic!(
                        "[{topology_tag}] BITWISE MISMATCH! Method: {:?}, S: {}, s_idx: {}, src: {}, v: {}, CPU: {}, GPU: {}",
                        method, s_count, s_idx, src, v, c, g
                    );
                }
                stats.total_distance_assertions += 1;

                // Optional analytical oracle check
                if let Some(ref oracle_fn) = oracle {
                    let expected = oracle_fn(src, v as u32);
                    if g != expected {
                        panic!(
                            "[{topology_tag}] ORACLE MISMATCH! Method: {:?}, src: {}, v: {}, GPU: {}, Oracle: {}",
                            method, src, v, g, expected
                        );
                    }
                    stats.total_oracle_checks += 1;
                }
            }

            // Triangle inequality verification:
            // For every edge (u, v), |dist(u) - dist(v)| <= 1 for reachable vertices,
            // and reachability is invariant across edges.
            for u in 0..nv {
                let start = graph.row_offsets[u] as usize;
                let end = graph.row_offsets[u + 1] as usize;
                let d_u = gpu_dist[base + u];

                for &v_neighbor in &graph.col_indices[start..end] {
                    let d_v = gpu_dist[base + v_neighbor as usize];
                    stats.total_triangle_checks += 1;

                    if d_u == u32::MAX {
                        assert_eq!(
                            d_v,
                            u32::MAX,
                            "[{topology_tag}] Method {:?}: Edge ({}, {}) reachability violation: u is unreachable but v has dist {}",
                            method, u, v_neighbor, d_v
                        );
                    } else {
                        assert_ne!(
                            d_v,
                            u32::MAX,
                            "[{topology_tag}] Method {:?}: Edge ({}, {}) reachability violation: u is reachable (dist {}) but v is unreachable",
                            method, u, v_neighbor, d_u
                        );
                        let diff = (d_u as i64 - d_v as i64).abs();
                        assert!(
                            diff <= 1,
                            "[{topology_tag}] Method {:?}: TRIANGLE INEQUALITY VIOLATION on edge ({}, {}): dist(u)={}, dist(v)={}, |diff|={}",
                            method, u, v_neighbor, d_u, d_v, diff
                        );
                    }
                }
            }
        }

        println!(
            "  -> {:<20}: {:>8.2} ms | Bitwise Equal: PASS | Triangle Ineq: PASS",
            format!("{:?}", method),
            gpu_elapsed.as_secs_f64() * 1000.0
        );
    }
}

/// 1. Large 2D Grid (30x30, 900 vertices, 3480 directed edges)
///    Analytical Manhattan distance oracle comparison + 6-way GPU + CPU.
#[test]
fn test_m3_adversarial_grid2d_30x30_sweep() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let width = 30;
    let height = 30;
    let grid = Grid2D::new(width, height).expect("Grid2D creation failed");
    let graph = &grid.graph;

    println!("\n========================================================");
    println!(" [ADVERSARIAL STRESS 1] Grid2D 30x30 (|V|=900, |E|=3480)");
    println!("========================================================");

    let mut stats = VerificationStats::default();

    for &s in &SWEEP_BATCH_SIZES {
        let sources = pick_sources(graph.num_vertices, s, 11);
        verify_all_methods_on_graph(
            &ctx,
            &cpu_engine,
            graph,
            &sources,
            "Grid2D-30x30",
            Some(|src, v| grid.manhattan_distance(src, v)),
            &mut stats,
        );
    }

    println!(
        "\n[Grid2D Summary] Total distance asserts: {}, Total triangle checks: {}, Total oracle checks: {}",
        stats.total_distance_assertions, stats.total_triangle_checks, stats.total_oracle_checks
    );
}

/// 2. Random Trees (N=1000, 1998 directed edges, deep diameter)
///    All vertices are reachable (0 unreachable nodes), irregular tree branching.
#[test]
fn test_m3_adversarial_random_tree_1000_sweep() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let n = 1000;
    let graph = generate_random_tree(n, Some(20261010)).expect("Random Tree generation failed");

    println!("\n========================================================");
    println!(" [ADVERSARIAL STRESS 2] Random Tree N=1000 (|V|=1000, |E|=1998)");
    println!("========================================================");

    let mut stats = VerificationStats::default();

    for &s in &SWEEP_BATCH_SIZES {
        let sources = pick_sources(graph.num_vertices, s, 29);
        verify_all_methods_on_graph(
            &ctx,
            &cpu_engine,
            &graph,
            &sources,
            "RandomTree-1000",
            None::<fn(u32, u32) -> u32>,
            &mut stats,
        );
    }

    println!(
        "\n[RandomTree Summary] Total distance asserts: {}, Total triangle checks: {}",
        stats.total_distance_assertions, stats.total_triangle_checks
    );
}

/// 3. Erdős-Rényi random graphs (N=2000, p=0.004, ~16000 directed edges)
///    Dense connectivity, multiple local hubs, small diameter.
#[test]
fn test_m3_adversarial_erdos_renyi_2000_sweep() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let n = 2000;
    let p = 0.004;
    let graph = generate_erdos_renyi_np(n, p, Some(20261010)).expect("ER generation failed");

    println!("\n========================================================");
    println!(
        " [ADVERSARIAL STRESS 3] Erdős-Rényi N=2000, p={} (|V|=2000, |E|={})",
        p, graph.num_edges
    );
    println!("========================================================");

    let mut stats = VerificationStats::default();

    for &s in &SWEEP_BATCH_SIZES {
        let sources = pick_sources(graph.num_vertices, s, 43);
        verify_all_methods_on_graph(
            &ctx,
            &cpu_engine,
            &graph,
            &sources,
            "ErdosRenyi-2000",
            None::<fn(u32, u32) -> u32>,
            &mut stats,
        );
    }

    println!(
        "\n[ErdosRenyi Summary] Total distance asserts: {}, Total triangle checks: {}",
        stats.total_distance_assertions, stats.total_triangle_checks
    );
}

/// 4. Disconnected components with multiple disjoint components
///    5 disjoint cliques of sizes [50, 100, 150, 200, 250] (|V|=750)
///    Strictly asserts u32::MAX for all cross-component vertices.
#[test]
fn test_m3_adversarial_disconnected_components_sweep() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let clique_sizes = [50, 100, 150, 200, 250];
    let graph = disconnected_cliques(&clique_sizes);
    let total_v = graph.num_vertices;

    println!("\n========================================================");
    println!(
        " [ADVERSARIAL STRESS 4] Disconnected Components ({:?}, |V|={}, |E|={})",
        clique_sizes, total_v, graph.num_edges
    );
    println!("========================================================");

    // Component ranges
    let mut comp_ranges = Vec::new();
    let mut cur = 0;
    for &sz in &clique_sizes {
        comp_ranges.push(cur..cur + sz);
        cur += sz;
    }

    let mut stats = VerificationStats::default();

    for &s in &SWEEP_BATCH_SIZES {
        // Distribute sources across disjoint components
        let mut sources = Vec::with_capacity(s);
        for i in 0..s {
            let comp_idx = i % clique_sizes.len();
            let range = &comp_ranges[comp_idx];
            let offset = (i * 7) % (range.end - range.start);
            sources.push((range.start + offset) as u32);
        }

        verify_all_methods_on_graph(
            &ctx,
            &cpu_engine,
            &graph,
            &sources,
            "DisconnectedCliques",
            None::<fn(u32, u32) -> u32>,
            &mut stats,
        );

        // Additional explicit check: verify unreachable count matches expected
        let bp64_engine = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::BitParallel64).unwrap();
        let dist = bp64_engine.run_ms_bfs(&graph, &sources).unwrap();

        for (s_idx, &src) in sources.iter().enumerate() {
            let src_u = src as usize;
            let comp_idx = comp_ranges
                .iter()
                .position(|r| r.contains(&src_u))
                .expect("Source must be inside one of the components");
            let comp_range = &comp_ranges[comp_idx];

            let base = s_idx * total_v;
            let mut unreachable_count = 0;
            for v in 0..total_v {
                let d = dist[base + v];
                if comp_range.contains(&v) {
                    if v == src_u {
                        assert_eq!(d, 0);
                    } else {
                        assert_eq!(d, 1);
                    }
                } else {
                    assert_eq!(
                        d,
                        u32::MAX,
                        "Cross-component vertex {v} must be unreachable from source {src}"
                    );
                    unreachable_count += 1;
                }
            }
            let expected_unreachable = total_v - clique_sizes[comp_idx];
            assert_eq!(unreachable_count, expected_unreachable);
        }
    }

    println!(
        "\n[Disconnected Summary] Total distance asserts: {}, Total triangle checks: {}",
        stats.total_distance_assertions, stats.total_triangle_checks
    );
}

/// 5. Dense Cliques (N=150 fully connected graph, 22,350 directed edges)
///    Extreme edge density (every vertex connected to all others), diameter 1.
#[test]
fn test_m3_adversarial_dense_cliques_sweep() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let n = 150;
    let graph = clique_graph(n);

    println!("\n========================================================");
    println!(
        " [ADVERSARIAL STRESS 5] Dense Clique N=150 (|V|={}, |E|={})",
        graph.num_vertices, graph.num_edges
    );
    println!("========================================================");

    let mut stats = VerificationStats::default();

    for &s in &SWEEP_BATCH_SIZES {
        let sources = pick_sources(n, s, 17);
        verify_all_methods_on_graph(
            &ctx,
            &cpu_engine,
            &graph,
            &sources,
            "DenseClique-150",
            Some(|src, v| if src == v { 0 } else { 1 }),
            &mut stats,
        );
    }

    println!(
        "\n[DenseClique Summary] Total distance asserts: {}, Total triangle checks: {}, Total oracle checks: {}",
        stats.total_distance_assertions, stats.total_triangle_checks, stats.total_oracle_checks
    );
}
