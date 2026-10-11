#![allow(clippy::identity_op, clippy::erasing_op)]

//! Comprehensive Automated Equivalence & Correctness Verification for Milestone M7:
//! 1. `GpuBfsMethod::SparseQueue` (Single-Source Sparse Queue BFS)
//! 2. `GpuBfsMethod::SparseBitParallel64` (64-Source Bit-Parallel Sparse Queue BFS)
//!
//! Evaluates bitwise 100% equivalence against `CpuBfsEngine` and analytical oracles across:
//! - Grid2D (30x30, 900 vertices, 3480 directed edges, high diameter)
//! - Random Trees (N=1000, 1998 directed edges, irregular deep branching)
//! - Erdős-Rényi random graphs (N=2000, p=0.004, high degree variance)
//! - SuiteSparse `luxembourg_osm` (114,599 vertices, 239,332 edges, diameter ~1200)
//! - Disconnected components (5 disjoint cliques of sizes [50, 100, 150, 200, 250], N=750)
//!
//! Batch sizes:
//! - S in {1, 5} for `SparseQueue`
//! - S in {1, 17, 32, 64} for `SparseBitParallel64`

use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use std::time::Instant;

use webgpu_bfs_core::{
    generators::{
        disconnected_cliques, generate_erdos_renyi_np, generate_random_tree, Grid2D,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

const M7_METHODS: [GpuBfsMethod; 2] = [
    GpuBfsMethod::SparseQueue,
    GpuBfsMethod::SparseBitParallel64,
];

/// Helper asserting full bitwise correctness against CPU baseline and metric invariants.
fn verify_m7_engine_against_cpu<F>(
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

    let cpu_dist = cpu_engine.run_ms_bfs(graph, sources).expect("CPU BFS failed");
    assert_eq!(cpu_dist.len(), s_count * nv, "[{tag}] CPU distance length mismatch");

    let gpu_engine = GpuBfsEngine::new(ctx.clone(), method)
        .unwrap_or_else(|e| panic!("Failed to instantiate {:?}: {e}", method));

    let t0 = Instant::now();
    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] {:?} failed: {e}", method));
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(gpu_dist.len(), s_count * nv, "[{tag}] GPU distance length mismatch");

    // Invariant 1: Source self-distance and bitwise equality
    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * nv;
        assert_eq!(
            gpu_dist[base + src as usize],
            0,
            "[{tag}] {:?} source self-distance != 0 for src {}",
            method,
            src
        );

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
                        "[{tag}] ORACLE MISMATCH! Method: {:?}, src: {}, v: {}, GPU: {}, Oracle: {}",
                        method, src, v, g, expected
                    );
                }
            }
        }

        // Invariant 2: Metric triangle inequality on all reachable edges: |d(u) - d(v)| <= 1
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
fn test_m7_sparse_queue_grid2d_sweep() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let grid = Grid2D::new(30, 30).expect("Grid generation failed");
    let graph = &grid.graph;

    let oracle = |s: u32, v: u32| -> u32 { grid.manhattan_distance(s, v) };

    println!("\n=== [M7 Equivalence] Grid2D 30x30 (|V|=900, |E|=3480) ===");
    for &method in &M7_METHODS {
        let batch_sizes = if method == GpuBfsMethod::SparseQueue {
            vec![1, 5]
        } else {
            vec![1, 17, 32, 64]
        };

        for s in batch_sizes {
            let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % 900).collect();
            verify_m7_engine_against_cpu(
                &ctx,
                &cpu,
                graph,
                &sources,
                method,
                "Grid2D-30x30",
                Some(oracle),
            );
        }
    }
}

#[test]
fn test_m7_sparse_queue_random_tree() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let graph = generate_random_tree(1000, Some(42)).expect("Tree generation failed");

    println!("\n=== [M7 Equivalence] RandomTree N=1000 (|V|=1000, |E|=1998) ===");
    for &method in &M7_METHODS {
        let batch_sizes = if method == GpuBfsMethod::SparseQueue {
            vec![1, 3]
        } else {
            vec![1, 64]
        };
        for s in batch_sizes {
            let sources: Vec<u32> = (0..s as u32).map(|i| (i * 17) % 1000).collect();
            verify_m7_engine_against_cpu(
                &ctx,
                &cpu,
                &graph,
                &sources,
                method,
                "RandomTree-1000",
                None::<fn(u32, u32) -> u32>,
            );
        }
    }
}

#[test]
fn test_m7_sparse_queue_erdos_renyi() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let graph = generate_erdos_renyi_np(2000, 0.004, Some(99)).expect("ER generation failed");

    println!("\n=== [M7 Equivalence] Erdos-Renyi N=2000, p=0.004 ===");
    for &method in &M7_METHODS {
        let batch_sizes = if method == GpuBfsMethod::SparseQueue {
            vec![1]
        } else {
            vec![1, 64]
        };
        for s in batch_sizes {
            let sources: Vec<u32> = (0..s as u32).map(|i| (i * 23) % 2000).collect();
            verify_m7_engine_against_cpu(
                &ctx,
                &cpu,
                &graph,
                &sources,
                method,
                "ErdosRenyi-2000",
                None::<fn(u32, u32) -> u32>,
            );
        }
    }
}

#[test]
fn test_m7_sparse_queue_luxembourg_osm() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let candidates = [
        std::path::PathBuf::from("data/luxembourg_osm/luxembourg_osm.mtx"),
        std::path::PathBuf::from("../../data/luxembourg_osm/luxembourg_osm.mtx"),
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/luxembourg_osm/luxembourg_osm.mtx"),
    ];
    let lux_path = candidates.iter().find(|p| p.exists());
    let Some(lux_path) = lux_path else {
        eprintln!("Skipping Luxembourg test: data/luxembourg_osm/luxembourg_osm.mtx not found.");
        return;
    };

    let file = File::open(lux_path).expect("Failed to open luxembourg_osm.mtx");
    let reader = BufReader::new(file);
    let graph = CsrGraph::from_mtx_reader(reader).expect("Failed to parse luxembourg_osm.mtx");

    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [M7 Equivalence] Luxembourg OSM (|V|=114,599, |E|=239,332) ===");
    // S=1 for both SparseQueue and SparseBitParallel64
    let sources_1 = vec![0u32];
    for &method in &M7_METHODS {
        verify_m7_engine_against_cpu(
            &ctx,
            &cpu,
            &graph,
            &sources_1,
            method,
            "Luxembourg-OSM",
            None::<fn(u32, u32) -> u32>,
        );
    }

    // S=64 for SparseBitParallel64
    let sources_64: Vec<u32> = (0..64u32).map(|i| (i * 1000) % 114599).collect();
    verify_m7_engine_against_cpu(
        &ctx,
        &cpu,
        &graph,
        &sources_64,
        GpuBfsMethod::SparseBitParallel64,
        "Luxembourg-OSM",
        None::<fn(u32, u32) -> u32>,
    );
}

#[test]
fn test_m7_pathological_topologies() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    // Disconnected Cliques (N=750)
    let disconnected = disconnected_cliques(&[50, 100, 150, 200, 250]);
    println!("\n=== [M7 Equivalence] Disconnected Cliques (|V|=750) ===");
    for &method in &M7_METHODS {
        let sources = vec![0, 75, 200];
        verify_m7_engine_against_cpu(
            &ctx,
            &cpu,
            &disconnected,
            &sources,
            method,
            "DisconnectedCliques",
            None::<fn(u32, u32) -> u32>,
        );
    }
}
