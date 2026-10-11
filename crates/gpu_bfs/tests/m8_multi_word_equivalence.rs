#![allow(clippy::identity_op)]

//! Comprehensive Automated Equivalence & Correctness Verification for Milestone M8:
//! `GpuBfsMethod::MultiWordBitParallel`
//!
//! Validates 100% bitwise match against `CpuBfsEngine` (Rayon) and analytical oracles across:
//! - Grid2D (30x30, 900 vertices, 3480 directed edges, Manhattan distance oracle)
//! - Random Trees (N=2000, 3998 directed edges, high diameter)
//! - SuiteSparse `luxembourg_osm` (114,599 vertices, 239,332 edges, diameter ~1200)
//! - Disconnected Cliques (N=1000, unreachable component verification)
//! - Non-power-of-2 / arbitrary source counts (S in {300, 750, 1500})
//!
//! Batch sizes: S in {256, 512, 1024, 2048}.

use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use std::time::Instant;

use webgpu_bfs_core::{
    generators::{disconnected_cliques, generate_random_tree, Grid2D},
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuContext};

static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Helper asserting 100% bitwise equality against CPU Rayon baseline and metric invariants.
fn verify_m8_engine_against_cpu<F>(
    ctx: &Arc<GpuContext>,
    cpu_engine: &CpuBfsEngine,
    graph: &CsrGraph,
    sources: &[u32],
    tag: &str,
    oracle: Option<F>,
) where
    F: Fn(u32, u32) -> u32,
{
    let nv = graph.num_vertices;
    let s_count = sources.len();
    let is_sym = graph.is_symmetric();

    let t_cpu = Instant::now();
    let cpu_dist = cpu_engine.run_ms_bfs(graph, sources).expect("CPU BFS failed");
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        cpu_dist.len(),
        s_count * nv,
        "[{tag}] CPU distance length mismatch"
    );

    let gpu_engine = GpuBfsEngine::multi_word_bit_parallel(ctx.clone())
        .expect("Failed to instantiate MultiWordBitParallel engine");

    let t_gpu = Instant::now();
    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] GPU MultiWordBitParallel failed: {e}"));
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        gpu_dist.len(),
        s_count * nv,
        "[{tag}] GPU distance length mismatch"
    );

    // Invariant 1: Source self-distance and 100% bitwise equality against CPU
    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * nv;
        assert_eq!(
            gpu_dist[base + src as usize],
            0,
            "[{tag}] Source self-distance != 0 for src {}",
            src
        );

        for v in 0..nv {
            let idx = base + v;
            let c = cpu_dist[idx];
            let g = gpu_dist[idx];
            if g != c {
                panic!(
                    "[{tag}] BITWISE MISMATCH! s_idx: {}, src: {}, v: {}, CPU: {}, GPU: {}",
                    s_idx, src, v, c, g
                );
            }

            if let Some(ref oracle_fn) = oracle {
                let expected = oracle_fn(src, v as u32);
                if g != expected {
                    panic!(
                        "[{tag}] ORACLE MISMATCH! src: {}, v: {}, GPU: {}, Oracle: {}",
                        src, v, g, expected
                    );
                }
            }
        }

        // Invariant 2: Metric triangle inequality on all graph edges: |d(u) - d(v)| <= 1
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
                        "[{tag}] Triangle inequality violated! Edge ({}, {}), d(u)={}, d(v)={}",
                        u, v, d_u, d_v
                    );
                } else if is_sym {
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

    let speedup = cpu_ms / gpu_ms.max(0.001);
    println!(
        "  -> {:<24} (S={:<4}) | CPU: {:>8.2} ms | GPU: {:>8.2} ms | Speedup: {:>5.2}x | Parity: 100% PASS",
        tag, s_count, cpu_ms, gpu_ms, speedup
    );
}

#[test]
fn test_m8_multi_word_grid2d_sweep() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let grid = Grid2D::new(30, 30).expect("Grid generation failed");
    let graph = &grid.graph;

    let oracle = |s: u32, v: u32| -> u32 { grid.manhattan_distance(s, v) };

    println!("\n=== [M8 Multi-Word Equivalence] Grid2D 30x30 (|V|=900, |E|=3480) ===");
    for &s in &[256usize, 512, 1024, 2048] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 900).collect();
        verify_m8_engine_against_cpu(
            &ctx,
            &cpu,
            graph,
            &sources,
            &format!("Grid2D-S{}", s),
            Some(oracle),
        );
    }
}

#[test]
fn test_m8_multi_word_random_tree() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let graph = generate_random_tree(2000, Some(42)).expect("Tree generation failed");

    println!("\n=== [M8 Multi-Word Equivalence] RandomTree N=2000 (|V|=2000, |E|=3998) ===");
    for &s in &[256usize, 512, 1024, 2048] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % 2000).collect();
        verify_m8_engine_against_cpu(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("Tree-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}

#[test]
fn test_m8_multi_word_luxembourg_osm() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let candidates = [
        std::path::PathBuf::from("data/luxembourg_osm/luxembourg_osm.mtx"),
        std::path::PathBuf::from("../../data/luxembourg_osm/luxembourg_osm.mtx"),
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/luxembourg_osm/luxembourg_osm.mtx"),
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

    println!("\n=== [M8 Multi-Word Equivalence] Luxembourg OSM (|V|=114,599, |E|=239,332) ===");
    for &s in &[256usize, 512, 1024, 2048] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 1000) % 114599).collect();
        verify_m8_engine_against_cpu(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("Luxembourg-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}

#[test]
fn test_m8_multi_word_disconnected_cliques() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    let disconnected = disconnected_cliques(&[100, 200, 300, 400]);
    println!("\n=== [M8 Multi-Word Equivalence] Disconnected Cliques (|V|=1000) ===");
    for &s in &[256usize, 512] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 1000).collect();
        verify_m8_engine_against_cpu(
            &ctx,
            &cpu,
            &disconnected,
            &sources,
            &format!("Disconnected-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}

#[test]
fn test_m8_multi_word_arbitrary_source_counts() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let graph = generate_random_tree(1000, Some(99)).expect("Tree generation failed");

    println!("\n=== [M8 Multi-Word Equivalence] Arbitrary Counts S in {{300, 750, 1500}} ===");
    for &s in &[300usize, 750, 1500] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 11) % 1000).collect();
        verify_m8_engine_against_cpu(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("ArbitraryCount-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}
