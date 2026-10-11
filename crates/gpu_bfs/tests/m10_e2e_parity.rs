#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M10: Comprehensive End-to-End Automated Parity Verification Suite
//! ==============================================================================
//!
//! Validates 100% bitwise distance parity against `CpuBfsEngine` (Rayon multi-threaded)
//! and analytical ground-truth oracles across all implemented GPU methods, topologies,
//! and batch sizes.
//!
//! Evaluated Methods:
//! 1. `GpuBfsMethod::SparseQueue` (Single-Source Sparse Frontier Queue)
//! 2. `GpuBfsMethod::SparseBitParallel64` (64-Source Bit-Parallel Sparse Queue)
//! 3. `GpuBfsMethod::MultiWordBitParallel` (256, 512, 1024, 2048 Source 2D Warp-Tiled)
//! 4. `GpuBfsMethod::IndirectMacroBatch` (GPU Autonomous Indirect Dispatch)
//! 5. All 7 Milestone 1-3 Baseline Methods (`NaiveSequential`, `BatchedDenseSpMM`,
//!    `BitParallel32`, `BitParallel64`, `BitParallel64Pull`, `BooleanReachability`,
//!    `PushPullAdaptive`)
//!
//! Evaluated Topologies:
//! - Grid2D (30x30, N=900, E=3480, Planar, Manhattan Oracle)
//! - Random Trees (N=1000, E=1998, Irregular Deep Branching)
//! - Disconnected Cliques (N=1000, 5x200 disjoint cliques, Unreachable Component Sentinels)
//! - SuiteSparse `luxembourg_osm` (N=114,599, E=239,332, Diameter ~1200)
//! - Arbitrary & Non-Power-of-2 Source Counts (S in {3, 17, 33, 65, 127, 300, 750, 1500})
//!
//! Strict Invariants Enforced:
//! 1. Length & Shape Match: len == S * |V|
//! 2. Source Self-Distance: d(s, s) == 0 for all s in S
//! 3. Bitwise Parity: d_gpu(s, v) == d_cpu(s, v) for all s in S, v in V
//! 4. Analytical Oracle: d_gpu(s, v) == oracle(s, v) when oracle provided
//! 5. Metric Triangle Inequality: |d(u) - d(v)| <= 1 on all reachable graph edges
//! 6. Component Reachability Consistency: d(u) == u32::MAX <=> d(v) == u32::MAX on symmetric edges

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use webgpu_bfs_core::{
    generators::{disconnected_cliques, generate_random_tree, Grid2D},
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

/// Global test synchronization mutex preventing driver race conditions during parallel test execution.
static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Helper to locate `luxembourg_osm.mtx` across multiple workspace relative paths.
fn find_luxembourg_path() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("data/luxembourg_osm/luxembourg_osm.mtx"),
        PathBuf::from("../../data/luxembourg_osm/luxembourg_osm.mtx"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/luxembourg_osm/luxembourg_osm.mtx"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data/luxembourg_osm/luxembourg_osm.mtx"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// Core E2E Parity Verification Harness.
///
/// Runs both CPU baseline (Rayon) and the designated GPU method, asserting:
/// 1. Exact length matching
/// 2. Source self-distance == 0
/// 3. Exact 100% bitwise distance equality
/// 4. Analytical oracle equality (if provided)
/// 5. Metric triangle inequality |d(u) - d(v)| <= 1 on all edges
fn verify_e2e_parity<F>(
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
    let is_sym = graph.is_symmetric();

    // 1. CPU Rayon Ground Truth
    let t_cpu = Instant::now();
    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] CPU BFS failed: {e}"));
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        cpu_dist.len(),
        s_count * nv,
        "[{tag}] CPU distance length mismatch: expected {}, got {}",
        s_count * nv,
        cpu_dist.len()
    );

    // 2. GPU BFS Execution
    let gpu_engine = GpuBfsEngine::new(ctx.clone(), method)
        .unwrap_or_else(|e| panic!("[{tag}] Failed to instantiate {:?}: {e}", method));

    let t_gpu = Instant::now();
    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] GPU {:?} execution failed: {e}", method));
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        gpu_dist.len(),
        s_count * nv,
        "[{tag}] {:?} distance length mismatch: expected {}, got {}",
        method,
        s_count * nv,
        gpu_dist.len()
    );

    // 3. Invariants Verification
    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * nv;

        // Invariant 1: Source self-distance == 0
        assert_eq!(
            gpu_dist[base + src as usize],
            0,
            "[{tag}] {:?} Source self-distance != 0 for source {} (s_idx {})",
            method,
            src,
            s_idx
        );

        // Invariant 2 & 3: Bitwise equality with CPU and analytical oracle
        for v in 0..nv {
            let idx = base + v;
            let c = cpu_dist[idx];
            let g = gpu_dist[idx];

            if g != c {
                panic!(
                    "[{tag}] BITWISE MISMATCH! Method: {:?}, s_idx: {}, src: {}, vertex: {}, CPU: {}, GPU: {}",
                    method, s_idx, src, v, c, g
                );
            }

            if let Some(ref oracle_fn) = oracle {
                let expected = oracle_fn(src, v as u32);
                if g != expected {
                    panic!(
                        "[{tag}] ORACLE MISMATCH! Method: {:?}, src: {}, vertex: {}, GPU: {}, Oracle: {}",
                        method, src, v, g, expected
                    );
                }
            }
        }

        // Invariant 4: Metric triangle inequality |d(u) - d(v)| <= 1 on all graph edges
        for u in 0..nv {
            let start = graph.row_offsets[u] as usize;
            let end = graph.row_offsets[u + 1] as usize;
            let d_u = gpu_dist[base + u];

            for edge_idx in start..end {
                let v = graph.col_indices[edge_idx] as usize;
                let d_v = gpu_dist[base + v];

                if d_u != u32::MAX && d_v != u32::MAX {
                    let diff = (d_u as i64 - d_v as i64).abs();
                    assert!(
                        diff <= 1,
                        "[{tag}] Triangle inequality violated! Method: {:?}, edge ({}, {}), d(u)={}, d(v)={}, |diff|={}",
                        method, u, v, d_u, d_v, diff
                    );
                } else if is_sym {
                    assert_eq!(
                        d_u == u32::MAX,
                        d_v == u32::MAX,
                        "[{tag}] Reachability mismatch across symmetric edge ({}, {}): d(u)={}, d(v)={}",
                        u, v, d_u, d_v
                    );
                }
            }
        }
    }

    let speedup = cpu_ms / gpu_ms.max(0.001);
    println!(
        "  -> {:<22} (S={:<4}) | CPU: {:>8.2} ms | GPU: {:>8.2} ms | Speedup: {:>5.2}x | Parity: 100% PASS",
        format!("{:?}", method),
        s_count,
        cpu_ms,
        gpu_ms,
        speedup
    );
}

// ==============================================================================
// 1. Grid2D Topology: Phase 2 Breakthrough Methods + Analytical Oracle
// ==============================================================================

#[test]
fn test_m10_e2e_grid2d_phase2_methods() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU context init failed"));
    let cpu = CpuBfsEngine::new();
    let grid = Grid2D::new(30, 30).expect("Grid generation failed");
    let graph = &grid.graph;
    let oracle = |s: u32, v: u32| -> u32 { grid.manhattan_distance(s, v) };

    println!("\n=== [M10 E2E] Grid2D 30x30 (|V|=900, |E|=3480, D=58) ===");

    // A. SparseQueue: S in {1, 4}
    for &s in &[1usize, 4] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 23) % 900).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            graph,
            &sources,
            GpuBfsMethod::SparseQueue,
            &format!("Grid2D-SparseQueue-S{}", s),
            Some(oracle),
        );
    }

    // B. SparseBitParallel64: S in {1, 32, 64}
    for &s in &[1usize, 32, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 17) % 900).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            graph,
            &sources,
            GpuBfsMethod::SparseBitParallel64,
            &format!("Grid2D-SparseBP64-S{}", s),
            Some(oracle),
        );
    }

    // C. IndirectMacroBatch: S in {1, 32, 64}
    for &s in &[1usize, 32, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % 900).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            graph,
            &sources,
            GpuBfsMethod::IndirectMacroBatch,
            &format!("Grid2D-Indirect-S{}", s),
            Some(oracle),
        );
    }

    // D. MultiWordBitParallel: S in {256, 512, 1024, 2048}
    for &s in &[256usize, 512, 1024, 2048] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 900).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            graph,
            &sources,
            GpuBfsMethod::MultiWordBitParallel,
            &format!("Grid2D-MultiWord-S{}", s),
            Some(oracle),
        );
    }
}

// ==============================================================================
// 2. Comprehensive 11-Method Cross-Parity Suite on Planar Grid
// ==============================================================================

#[test]
fn test_m10_e2e_all_11_methods_cross_parity() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU context init failed"));
    let cpu = CpuBfsEngine::new();
    let grid = Grid2D::new(20, 20).expect("Grid generation failed");
    let graph = &grid.graph;
    let oracle = |s: u32, v: u32| -> u32 { grid.manhattan_distance(s, v) };

    let all_methods = [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
        GpuBfsMethod::BitParallel32,
        GpuBfsMethod::BitParallel64,
        GpuBfsMethod::BitParallel64Pull,
        GpuBfsMethod::BooleanReachability,
        GpuBfsMethod::PushPullAdaptive,
        GpuBfsMethod::SparseQueue,
        GpuBfsMethod::SparseBitParallel64,
        GpuBfsMethod::MultiWordBitParallel,
        GpuBfsMethod::IndirectMacroBatch,
    ];

    println!("\n=== [M10 E2E] All 11 Methods Cross-Parity (Grid2D 20x20, |V|=400) ===");
    for &method in &all_methods {
        // Use S=32 compatible with all 11 methods
        let sources: Vec<u32> = (0..32u32).map(|i| (i * 11) % 400).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            graph,
            &sources,
            method,
            &format!("AllMethods-CrossParity-S32-{:?}", method),
            Some(oracle),
        );
    }
}

// ==============================================================================
// 3. Random Trees: Deep Irregular Branching (Zero Cycles)
// ==============================================================================

#[test]
fn test_m10_e2e_random_trees_parity() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU context init failed"));
    let cpu = CpuBfsEngine::new();
    let graph = generate_random_tree(1000, Some(42)).expect("Tree generation failed");

    println!("\n=== [M10 E2E] Random Tree N=1000 (|V|=1000, |E|=1998) ===");

    // SparseQueue: S in {1, 3}
    for &s in &[1usize, 3] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 29) % 1000).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::SparseQueue,
            &format!("Tree-SparseQueue-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }

    // SparseBitParallel64: S in {1, 64}
    for &s in &[1usize, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 17) % 1000).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::SparseBitParallel64,
            &format!("Tree-SparseBP64-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }

    // IndirectMacroBatch: S in {1, 64}
    for &s in &[1usize, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % 1000).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::IndirectMacroBatch,
            &format!("Tree-Indirect-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }

    // MultiWordBitParallel: S in {256, 512, 1024, 2048}
    for &s in &[256usize, 512, 1024, 2048] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 1000).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::MultiWordBitParallel,
            &format!("Tree-MultiWord-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}

// ==============================================================================
// 4. Disconnected Cliques: Unreachable Sentinel & Hardware No-Op Early Exit
// ==============================================================================

#[test]
fn test_m10_e2e_disconnected_cliques_isolation() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU context init failed"));
    let cpu = CpuBfsEngine::new();

    // 5 disconnected cliques of size 200 (Total N=1000)
    let disconnected = disconnected_cliques(&[200, 200, 200, 200, 200]);
    println!("\n=== [M10 E2E] Disconnected Cliques (|V|=1000, 5x200) ===");

    // Sources placed only in Clique 0 (0..199) and Clique 2 (400..599).
    // Cliques 1, 3, 4 are strictly unreachable (dist == u32::MAX).
    let s_count = 64usize;
    let sources: Vec<u32> = (0..s_count as u32)
        .map(|i| if i % 2 == 0 { (i * 7) % 200 } else { 400 + (i * 7) % 200 })
        .collect();

    // Verify all Phase 2 methods on disconnected structure
    let methods = [
        GpuBfsMethod::SparseQueue,
        GpuBfsMethod::SparseBitParallel64,
        GpuBfsMethod::IndirectMacroBatch,
    ];

    for &method in &methods {
        let test_sources = if method == GpuBfsMethod::SparseQueue {
            &sources[..4]
        } else {
            &sources[..]
        };
        verify_e2e_parity(
            &ctx,
            &cpu,
            &disconnected,
            test_sources,
            method,
            &format!("Disconnected-{:?}", method),
            None::<fn(u32, u32) -> u32>,
        );
    }

    // Verify MultiWordBitParallel on Disconnected Cliques (S=256, 512)
    for &s in &[256usize, 512] {
        let mw_sources: Vec<u32> = (0..s as u32)
            .map(|i| if i % 2 == 0 { (i * 7) % 200 } else { 400 + (i * 7) % 200 })
            .collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &disconnected,
            &mw_sources,
            GpuBfsMethod::MultiWordBitParallel,
            &format!("Disconnected-MultiWord-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}

// ==============================================================================
// 5. Arbitrary & Non-Power-of-2 Batch Sizes Stress
// ==============================================================================

#[test]
fn test_m10_e2e_arbitrary_source_counts_stress() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU context init failed"));
    let cpu = CpuBfsEngine::new();
    let graph = generate_random_tree(1000, Some(99)).expect("Tree generation failed");

    println!("\n=== [M10 E2E] Arbitrary & Non-Power-of-2 Source Counts Stress ===");

    // Non-power-of-2 small/medium counts on SparseBP64 and IndirectMacroBatch
    for &s in &[3usize, 17, 33, 65, 127] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 11) % 1000).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::SparseBitParallel64,
            &format!("Arbitrary-SparseBP64-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::IndirectMacroBatch,
            &format!("Arbitrary-Indirect-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }

    // Large non-power-of-2 counts on MultiWordBitParallel
    for &s in &[300usize, 750, 1500] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % 1000).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::MultiWordBitParallel,
            &format!("Arbitrary-MultiWord-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}

// ==============================================================================
// 6. SuiteSparse `luxembourg_osm`: Sparse Queue & Indirect Macro-Batching
// ==============================================================================

#[test]
fn test_m10_e2e_luxembourg_osm_sparse_and_indirect() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let Some(lux_path) = find_luxembourg_path() else {
        eprintln!("Skipping Luxembourg test: luxembourg_osm.mtx fixture not found.");
        return;
    };

    let file = File::open(&lux_path).expect("Failed to open luxembourg_osm.mtx");
    let reader = BufReader::new(file);
    let graph = CsrGraph::from_mtx_reader(reader).expect("Failed to parse luxembourg_osm.mtx");

    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU context init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [M10 E2E] Luxembourg OSM (|V|=114,599, |E|=239,332, D~1200) Sparse & Indirect ===");

    // A. SparseQueue (Single-Source): S=1
    let sources_1 = vec![0u32];
    verify_e2e_parity(
        &ctx,
        &cpu,
        &graph,
        &sources_1,
        GpuBfsMethod::SparseQueue,
        "Luxembourg-SparseQueue-S1",
        None::<fn(u32, u32) -> u32>,
    );

    // B. SparseBitParallel64: S in {1, 64}
    for &s in &[1usize, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 1000) % 114599).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::SparseBitParallel64,
            &format!("Luxembourg-SparseBP64-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }

    // C. IndirectMacroBatch: S in {1, 64} (Validates ~19 macro-chunks of 64 hops each)
    for &s in &[1usize, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 1000) % 114599).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::IndirectMacroBatch,
            &format!("Luxembourg-Indirect-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}

// ==============================================================================
// 7. SuiteSparse `luxembourg_osm`: Massive Multi-Word Bit-Parallel Scaling
// ==============================================================================

#[test]
fn test_m10_e2e_luxembourg_osm_multi_word() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let Some(lux_path) = find_luxembourg_path() else {
        eprintln!("Skipping Luxembourg test: luxembourg_osm.mtx fixture not found.");
        return;
    };

    let file = File::open(&lux_path).expect("Failed to open luxembourg_osm.mtx");
    let reader = BufReader::new(file);
    let graph = CsrGraph::from_mtx_reader(reader).expect("Failed to parse luxembourg_osm.mtx");

    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU context init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [M10 E2E] Luxembourg OSM (|V|=114,599) Massive Multi-Word Bit-Parallel ===");

    // Determine batch sweep based on environment variable to avoid prolonged software emulation
    let full_sweep = std::env::var("M10_MASSIVE_SWEEP").is_ok()
        || std::env::var("BFS_FULL_SWEEP").is_ok()
        || !cfg!(debug_assertions);

    let batch_sizes: Vec<usize> = if full_sweep {
        vec![256, 512, 1024, 2048]
    } else {
        vec![256]
    };

    for s in batch_sizes {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 1000) % 114599).collect();
        verify_e2e_parity(
            &ctx,
            &cpu,
            &graph,
            &sources,
            GpuBfsMethod::MultiWordBitParallel,
            &format!("Luxembourg-MultiWord-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}
