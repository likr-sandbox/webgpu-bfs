#![allow(clippy::identity_op, clippy::erasing_op)]

//! Adversarial Stress and Topological Corner Case Verification Suite for Milestone M7:
//! 1. `GpuBfsMethod::SparseQueue` (Single-Source Sparse Queue BFS)
//! 2. `GpuBfsMethod::SparseBitParallel64` (64-Source Bit-Parallel Sparse Queue BFS)
//!
//! Evaluates strictly against `CpuBfsEngine` reference and analytical ground-truth oracles across:
//! 1. Extreme diameters: Linear line graphs (N=5000, diameter=4999).
//! 2. Massive hub / star graphs (N=2000, 1 hub + 1999 leaves, massive contention).
//! 3. Empty edge graphs (N=100, E=0 isolated vertices).
//! 4. Single-node graphs (N=1, E=0 boundary condition).
//! 5. High-diameter grid sweeps (40x40, N=1600, diameter=78).
//! 6. Dense cliques (N=200, 39,800 directed edges, extreme neighbor contention).
//! 7. Multi-chunk (S=65, S=128) and duplicate source stress tests.

use std::sync::Arc;
use std::time::Instant;

use webgpu_bfs_core::{
    generators::{
        clique_graph, generate_random_tree, isolated_vertices, path_graph, single_vertex,
        star_graph, Grid2D,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Helper asserting 100% bitwise correctness against CPU baseline, analytical oracle,
/// and metric invariants (triangle inequality, self-distance, reachability).
fn verify_adversarial_case<F>(
    ctx: &Arc<GpuContext>,
    cpu_engine: &CpuBfsEngine,
    graph: &CsrGraph,
    sources: &[u32],
    method: GpuBfsMethod,
    tag: &str,
    oracle: Option<F>,
) where
    F: Fn(u32, u32) -> u32,
{
    let nv = graph.num_vertices;
    let s_count = sources.len();

    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .expect("CPU BFS failed");
    assert_eq!(
        cpu_dist.len(),
        s_count * nv,
        "[{tag}] CPU distance length mismatch"
    );

    let gpu_engine = GpuBfsEngine::new(ctx.clone(), method)
        .unwrap_or_else(|e| panic!("Failed to instantiate {:?}: {e}", method));

    let t0 = Instant::now();
    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] {:?} failed: {e}", method));
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        gpu_dist.len(),
        s_count * nv,
        "[{tag}] GPU distance length mismatch"
    );

    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * nv;

        // Invariant 1: Source self-distance == 0
        assert_eq!(
            gpu_dist[base + src as usize],
            0,
            "[{tag}] {:?} source self-distance != 0 for src {}",
            method,
            src
        );

        // Invariant 2: Bitwise parity against CPU reference and analytical oracle
        for v in 0..nv {
            let idx = base + v;
            let c = cpu_dist[idx];
            let g = gpu_dist[idx];
            if g != c {
                panic!(
                    "[{tag}] BITWISE MISMATCH! Method: {:?}, s_idx: {}, src: {}, v: {}, CPU: {}, GPU: {}",
                    method, s_idx, src, v, c, g
                );
            }

            if let Some(ref oracle_fn) = oracle {
                let expected = oracle_fn(src, v as u32);
                if g != expected {
                    panic!(
                        "[{tag}] ORACLE MISMATCH! Method: {:?}, s_idx: {}, src: {}, v: {}, GPU: {}, Oracle: {}",
                        method, s_idx, src, v, g, expected
                    );
                }
            }
        }

        // Invariant 3: Metric triangle inequality on all edges: |d(u) - d(v)| <= 1
        for u in 0..nv {
            let start = graph.row_offsets[u] as usize;
            let end = graph.row_offsets[u + 1] as usize;
            let d_u = gpu_dist[base + u];

            for edge in start..end {
                let v = graph.col_indices[edge] as usize;
                let d_v = gpu_dist[base + v];

                if d_u != u32::MAX && d_v != u32::MAX {
                    let diff = (d_u as i64 - d_v as i64).abs();
                    assert!(
                        diff <= 1,
                        "[{tag}] Triangle inequality violated! Method: {:?}, edge ({}, {}), d(u)={}, d(v)={}",
                        method, u, v, d_u, d_v
                    );
                } else {
                    assert_eq!(
                        d_u == u32::MAX,
                        d_v == u32::MAX,
                        "[{tag}] Reachability mismatch across edge ({}, {}): d(u)={}, d(v)={}",
                        u, v, d_u, d_v
                    );
                }
            }
        }
    }

    println!(
        "  -> {:<22} (S={:<2}) : {:>8.2} ms | Bitwise Equal: PASS | Triangle Ineq: PASS",
        format!("{:?}", method),
        s_count,
        gpu_ms
    );
}

#[test]
fn test_m7_extreme_diameter_line_graph() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    const N: usize = 5000;
    let graph = path_graph(N);
    assert_eq!(graph.num_vertices, N);
    assert_eq!(graph.num_edges, 2 * (N - 1));

    let oracle = |s: u32, v: u32| -> u32 { (s as i64 - v as i64).unsigned_abs() as u32 };

    println!("\n=== [M7 Adversarial] Linear Line Graph (N=5000, Diameter=4999) ===");

    // 1. SparseQueue single-source endpoints and middle
    for &src in &[0u32, 2500u32, 4999u32] {
        verify_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &[src],
            GpuBfsMethod::SparseQueue,
            &format!("Line5000-SparseQueue-src{}", src),
            Some(oracle),
        );
    }

    // 2. SparseQueue multi-source S=5
    let sources_s5 = vec![0, 1250, 2500, 3750, 4999];
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &sources_s5,
        GpuBfsMethod::SparseQueue,
        "Line5000-SparseQueue-S5",
        Some(oracle),
    );

    // 3. SparseBitParallel64 single-source
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        GpuBfsMethod::SparseBitParallel64,
        "Line5000-SparseBP64-src0",
        Some(oracle),
    );
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[2500],
        GpuBfsMethod::SparseBitParallel64,
        "Line5000-SparseBP64-src2500",
        Some(oracle),
    );

    // 4. SparseBitParallel64 S=64
    let sources_s64: Vec<u32> = (0..64).map(|i| (i * 78) as u32).collect();
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &sources_s64,
        GpuBfsMethod::SparseBitParallel64,
        "Line5000-SparseBP64-S64",
        Some(oracle),
    );
}

#[test]
fn test_m7_massive_hub_star_graph() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    const N: usize = 2000;
    let graph = star_graph(N);
    assert_eq!(graph.num_vertices, N);
    assert_eq!(graph.num_edges, 2 * (N - 1));

    let oracle = |s: u32, v: u32| -> u32 {
        if s == v {
            0
        } else if s == 0 || v == 0 {
            1
        } else {
            2
        }
    };

    println!("\n=== [M7 Adversarial] Massive Hub Star Graph (N=2000, 1 Hub + 1999 Leaves) ===");

    // Hub source (s=0)
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        GpuBfsMethod::SparseQueue,
        "Star2000-SparseQueue-Hub",
        Some(oracle),
    );
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        GpuBfsMethod::SparseBitParallel64,
        "Star2000-SparseBP64-Hub",
        Some(oracle),
    );

    // Leaf source (s=1)
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[1],
        GpuBfsMethod::SparseQueue,
        "Star2000-SparseQueue-Leaf",
        Some(oracle),
    );
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[1],
        GpuBfsMethod::SparseBitParallel64,
        "Star2000-SparseBP64-Leaf",
        Some(oracle),
    );

    // S=3 mixed sources
    let sources_s3 = vec![0, 42, 1999];
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &sources_s3,
        GpuBfsMethod::SparseQueue,
        "Star2000-SparseQueue-S3",
        Some(oracle),
    );
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &sources_s3,
        GpuBfsMethod::SparseBitParallel64,
        "Star2000-SparseBP64-S3",
        Some(oracle),
    );

    // S=64 LEAF sources: All 64 leaves push to hub 0 at the exact same hop!
    // Maximum atomic collision stress test.
    let leaf_sources_s64: Vec<u32> = (1..=64).collect();
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &leaf_sources_s64,
        GpuBfsMethod::SparseBitParallel64,
        "Star2000-SparseBP64-64LeafContention",
        Some(oracle),
    );
}

#[test]
fn test_m7_empty_edge_isolated_vertices() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    const N: usize = 100;
    let graph = isolated_vertices(N);
    assert_eq!(graph.num_vertices, N);
    assert_eq!(graph.num_edges, 0);

    let oracle = |s: u32, v: u32| -> u32 {
        if s == v {
            0
        } else {
            u32::MAX
        }
    };

    println!("\n=== [M7 Adversarial] Empty Edge Graph (N=100, E=0) ===");

    // SparseQueue
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        GpuBfsMethod::SparseQueue,
        "Isolated100-SparseQueue-src0",
        Some(oracle),
    );
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[42],
        GpuBfsMethod::SparseQueue,
        "Isolated100-SparseQueue-src42",
        Some(oracle),
    );

    let sources_s10: Vec<u32> = (0..10).map(|i| i * 10).collect();
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &sources_s10,
        GpuBfsMethod::SparseQueue,
        "Isolated100-SparseQueue-S10",
        Some(oracle),
    );

    // SparseBitParallel64
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        GpuBfsMethod::SparseBitParallel64,
        "Isolated100-SparseBP64-src0",
        Some(oracle),
    );

    let sources_s64: Vec<u32> = (0..64).collect();
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &sources_s64,
        GpuBfsMethod::SparseBitParallel64,
        "Isolated100-SparseBP64-S64",
        Some(oracle),
    );
}

#[test]
fn test_m7_single_node_graph() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    let graph = single_vertex();
    assert_eq!(graph.num_vertices, 1);
    assert_eq!(graph.num_edges, 0);

    let oracle = |_s: u32, _v: u32| -> u32 { 0 };

    println!("\n=== [M7 Adversarial] Single Node Graph (N=1, E=0) ===");

    // S=1
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        GpuBfsMethod::SparseQueue,
        "SingleNode-SparseQueue-S1",
        Some(oracle),
    );
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        GpuBfsMethod::SparseBitParallel64,
        "SingleNode-SparseBP64-S1",
        Some(oracle),
    );

    // S=64 duplicate sources
    let sources_s64 = vec![0u32; 64];
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &sources_s64,
        GpuBfsMethod::SparseBitParallel64,
        "SingleNode-SparseBP64-S64Dup",
        Some(oracle),
    );
}

#[test]
fn test_m7_high_diameter_grid_sweep() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    let grid = Grid2D::new(40, 40).expect("Grid40x40 creation failed");
    let graph = &grid.graph;
    assert_eq!(graph.num_vertices, 1600);
    assert_eq!(graph.num_edges, 6240);

    let oracle = |s: u32, v: u32| -> u32 { grid.manhattan_distance(s, v) };

    println!("\n=== [M7 Adversarial] High-Diameter Grid 40x40 (N=1600, Diameter=78) ===");

    // Corners and center for SparseQueue
    for &src in &[0u32, 820u32, 1599u32] {
        verify_adversarial_case(
            &ctx,
            &cpu,
            graph,
            &[src],
            GpuBfsMethod::SparseQueue,
            &format!("Grid40x40-SparseQueue-src{}", src),
            Some(oracle),
        );
    }

    // S=5 for SparseQueue
    let sources_s5 = vec![0, 39, 820, 1560, 1599];
    verify_adversarial_case(
        &ctx,
        &cpu,
        graph,
        &sources_s5,
        GpuBfsMethod::SparseQueue,
        "Grid40x40-SparseQueue-S5",
        Some(oracle),
    );

    // S=1, 17, 32, 64 for SparseBitParallel64
    for &s_count in &[1usize, 17usize, 32usize, 64usize] {
        let sources: Vec<u32> = (0..s_count as u32).map(|i| (i * 23) % 1600).collect();
        verify_adversarial_case(
            &ctx,
            &cpu,
            graph,
            &sources,
            GpuBfsMethod::SparseBitParallel64,
            &format!("Grid40x40-SparseBP64-S{}", s_count),
            Some(oracle),
        );
    }
}

#[test]
fn test_m7_dense_clique_contention() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    const N: usize = 200;
    let graph = clique_graph(N);
    assert_eq!(graph.num_vertices, N);
    assert_eq!(graph.num_edges, N * (N - 1));

    let oracle = |s: u32, v: u32| -> u32 {
        if s == v {
            0
        } else {
            1
        }
    };

    println!("\n=== [M7 Adversarial] Dense Clique (N=200, |E|=39,800) ===");

    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        GpuBfsMethod::SparseQueue,
        "Clique200-SparseQueue-S1",
        Some(oracle),
    );
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        GpuBfsMethod::SparseBitParallel64,
        "Clique200-SparseBP64-S1",
        Some(oracle),
    );

    let sources_s64: Vec<u32> = (0..64).collect();
    verify_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &sources_s64,
        GpuBfsMethod::SparseBitParallel64,
        "Clique200-SparseBP64-S64",
        Some(oracle),
    );
}

#[test]
fn test_m7_multichunk_and_duplicate_sources() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    let graph = generate_random_tree(500, Some(1337)).expect("Tree generation failed");
    assert_eq!(graph.num_vertices, 500);

    println!("\n=== [M7 Adversarial] Multi-Chunk (S=65, S=128) & Duplicate Sources ===");

    // 1. S=65 spans 2 chunks (64 + 1)
    let sources_s65: Vec<u32> = (0..65).map(|i| (i * 7) % 500).collect();
    verify_adversarial_case::<fn(u32, u32) -> u32>(
        &ctx,
        &cpu,
        &graph,
        &sources_s65,
        GpuBfsMethod::SparseBitParallel64,
        "Tree500-SparseBP64-S65-ChunkCrossing",
        None,
    );

    // 2. S=128 spans exactly 2 full chunks (64 + 64)
    let sources_s128: Vec<u32> = (0..128).map(|i| (i * 3) % 500).collect();
    verify_adversarial_case::<fn(u32, u32) -> u32>(
        &ctx,
        &cpu,
        &graph,
        &sources_s128,
        GpuBfsMethod::SparseBitParallel64,
        "Tree500-SparseBP64-S128-TwoFullChunks",
        None,
    );

    // 3. Duplicate sources within same chunk and across chunks
    let dup_sources = vec![0, 0, 7, 7, 7, 42, 42, 100, 100, 100, 100];
    verify_adversarial_case::<fn(u32, u32) -> u32>(
        &ctx,
        &cpu,
        &graph,
        &dup_sources,
        GpuBfsMethod::SparseQueue,
        "Tree500-SparseQueue-Duplicates",
        None,
    );
    verify_adversarial_case::<fn(u32, u32) -> u32>(
        &ctx,
        &cpu,
        &graph,
        &dup_sources,
        GpuBfsMethod::SparseBitParallel64,
        "Tree500-SparseBP64-Duplicates",
        None,
    );
}
