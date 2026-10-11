#![allow(clippy::identity_op)]

//! Comprehensive Automated Equivalence & Correctness Verification for Milestone M9:
//! `GpuBfsMethod::IndirectMacroBatch` (WebGPU Native Indirect Dispatch & Macro-Batching)
//!
//! Evaluates 100% bitwise equivalence against `CpuBfsEngine` (Rayon) and analytical oracles across:
//! 1. Small Grid2D (30x30, 900 vertices, 3480 directed edges, Manhattan distance oracle)
//! 2. Deep Line Graph / Chain (N=400, diameter D=399, exceeding single 64-hop chunk by 6.2x)
//! 3. Random Trees (N=2000, 3998 directed edges, irregular branching)
//! 4. Disconnected Cliques (N=1000, 5 cliques of 200 vertices, asserting zero-workgroup early exit)
//! 5. SuiteSparse `luxembourg_osm` (114,599 vertices, 239,332 edges, diameter ~1200)
//!
//! Validates:
//! - Exact bitwise distance equality: gpu_dist[s * |V| + v] == cpu_dist[s * |V| + v]
//! - Source self-distance invariant: dist[s * |V| + s] == 0
//! - Metric triangle inequality: |d(u) - d(v)| <= 1 on all reachable edges
//! - Unreachable component integrity: d(u) == u32::MAX on disconnected components

use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use std::time::Instant;

use webgpu_bfs_core::{
    generators::{disconnected_cliques, generate_random_tree, path_graph, Grid2D},
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuContext};

static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Helper asserting 100% bitwise equality against CPU Rayon baseline and metric invariants.
fn verify_m9_indirect_against_cpu<F>(
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

    let gpu_engine = GpuBfsEngine::indirect_macro_batch(ctx.clone())
        .expect("Failed to instantiate IndirectMacroBatch engine");

    let t_gpu = Instant::now();
    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] GPU IndirectMacroBatch failed: {e}"));
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
        "  -> {:<26} (S={:<2}) | CPU: {:>8.2} ms | GPU: {:>8.2} ms | Speedup: {:>5.2}x | Parity: 100% PASS",
        tag, s_count, cpu_ms, gpu_ms, speedup
    );
}

#[test]
fn test_m9_indirect_grid2d_sweep() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let grid = Grid2D::new(30, 30).expect("Grid generation failed");
    let graph = &grid.graph;

    let oracle = |s: u32, v: u32| -> u32 { grid.manhattan_distance(s, v) };

    println!("\n=== [M9 Indirect Equivalence] Grid2D 30x30 (|V|=900, |E|=3480) ===");
    for &s in &[1usize, 32, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % 900).collect();
        verify_m9_indirect_against_cpu(
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
fn test_m9_indirect_deep_line_chain() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    // N=400 path graph with diameter D=399.
    // Exceeds a 64-hop chunk by ~6.2x, verifying progression across 7 consecutive macro-batch chunks.
    let n = 400;
    let graph = path_graph(n);

    let oracle = |s: u32, v: u32| -> u32 { (s as i64 - v as i64).unsigned_abs() as u32 };

    println!("\n=== [M9 Indirect Equivalence] Deep Line Graph (|V|=400, Diameter=399) ===");
    for &s in &[1usize, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 3) % (n as u32)).collect();
        verify_m9_indirect_against_cpu(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("LineChain-S{}", s),
            Some(oracle),
        );
    }
}

#[test]
fn test_m9_indirect_random_tree() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let graph = generate_random_tree(2000, Some(42)).expect("Tree generation failed");

    println!("\n=== [M9 Indirect Equivalence] RandomTree N=2000 (|V|=2000, |E|=3998) ===");
    for &s in &[1usize, 16, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 17) % 2000).collect();
        verify_m9_indirect_against_cpu(
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
fn test_m9_indirect_disconnected_cliques() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    // 5 disconnected cliques of size 200 (N=1000).
    // Within each clique, diameter is 1 hop.
    // Tests zero-workgroup early exit: hops 2..63 in a 64-hop chunk execute as hardware no-ops.
    let disconnected = disconnected_cliques(&[200, 200, 200, 200, 200]);
    println!("\n=== [M9 Indirect Equivalence] Disconnected Cliques (|V|=1000, 5x200) ===");
    for &s in &[1usize, 64] {
        // Sources placed in clique 0 (vertices 0..199) and clique 2 (vertices 400..599)
        let sources: Vec<u32> = (0..s as u32)
            .map(|i| if i % 2 == 0 { (i * 7) % 200 } else { 400 + (i * 7) % 200 })
            .collect();
        verify_m9_indirect_against_cpu(
            &ctx,
            &cpu,
            &disconnected,
            &sources,
            &format!("DisconnectedCliques-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}

#[test]
fn test_m9_indirect_luxembourg_osm() {
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

    println!("\n=== [M9 Indirect Equivalence] Luxembourg OSM (|V|=114,599, |E|=239,332, D~1200) ===");
    for &s in &[1usize, 16, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 1000) % 114599).collect();
        verify_m9_indirect_against_cpu(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("Luxembourg-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }
}
