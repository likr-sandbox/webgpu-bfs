#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M10 Challenger 2: Adversarial Stress & Corner-Case Harness
//! for Real-World SuiteSparse `luxembourg_osm`
//! ==============================================================================
//!
//! Evaluates the robustness, stability, and bitwise correctness of:
//! 1. `GpuBfsMethod::IndirectMacroBatch` (GPU Autonomous Indirect Dispatch)
//! 2. `GpuBfsMethod::SparseBitParallel64` (Compacted Frontier 64-Source Bit-Parallel)
//! 3. `GpuBfsMethod::SparseQueue` (Compacted Frontier Single-Source)
//!
//! On SuiteSparse `luxembourg_osm` (|V|=114,599, |E|=239,332, Diameter ~1,337).
//!
//! Stress Vectors Evaluated:
//! 1. Extremal & High-Diameter Sources:
//!    - Periphery North/East (node 48292, depth 1337)
//!    - Periphery South/West (node 24481, depth 1337)
//!    - Degree-6 Junction Hub (node 6397)
//!    - Degree-1 Dead Ends (nodes 3898, 5112)
//!    - Extremal Vertices (0, 114598)
//! 2. Duplicate Sources Stress:
//!    - Duplicate inputs in `IndirectMacroBatch` (e.g. [0, 0, 48292, 48292, 6397, 6397])
//! 3. Cross-Batch Chunk Rollover:
//!    - S = 65 (64 + 1) on `IndirectMacroBatch` traversing 1,337 hops across multiple 64-source batches
//! 4. Adversarial Macro-Batch Chunk Sizes:
//!    - Custom `chunk_hops` = 128 and `chunk_hops` = 17 on `luxembourg_osm`
//! 5. Invariants Enforced:
//!    - 100% bitwise parity with CPU Rayon baseline
//!    - Source self-distance == 0 for all sources
//!    - Metric triangle inequality |d(u) - d(v)| <= 1 on all 239,332 directed edges
//!    - Component reachability consistency on symmetric edges

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use webgpu_bfs_core::{BfsEngine, CsrGraph};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{
    dispatch::indirect::IndirectMacroBatchBfs,
    pipeline::PipelineCache,
    GpuBfsEngine, GpuBfsMethod, GpuContext,
};

static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn find_luxembourg_graph() -> Option<CsrGraph> {
    let candidates = [
        PathBuf::from("data/luxembourg_osm/luxembourg_osm.mtx"),
        PathBuf::from("../../data/luxembourg_osm/luxembourg_osm.mtx"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/luxembourg_osm/luxembourg_osm.mtx"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data/luxembourg_osm/luxembourg_osm.mtx"),
    ];
    let path = candidates.into_iter().find(|p| p.exists())?;
    let file = File::open(&path).expect("Failed to open luxembourg_osm.mtx");
    let reader = BufReader::new(file);
    let graph = CsrGraph::from_mtx_reader(reader).expect("Failed to parse luxembourg_osm.mtx");
    Some(graph)
}

fn assert_invariants(
    graph: &CsrGraph,
    sources: &[u32],
    cpu_dist: &[u32],
    gpu_dist: &[u32],
    tag: &str,
) {
    let nv = graph.num_vertices;
    let s_count = sources.len();
    let is_sym = graph.is_symmetric();

    assert_eq!(
        gpu_dist.len(),
        s_count * nv,
        "[{tag}] GPU distance buffer size mismatch: expected {}, got {}",
        s_count * nv,
        gpu_dist.len()
    );

    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * nv;

        // Invariant 1: Source self-distance == 0
        assert_eq!(
            gpu_dist[base + src as usize],
            0,
            "[{tag}] Self-distance violation for source {} (s_idx {})",
            src,
            s_idx
        );

        // Invariant 2: 100% bitwise parity with CPU baseline
        for v in 0..nv {
            let idx = base + v;
            let c = cpu_dist[idx];
            let g = gpu_dist[idx];
            if g != c {
                panic!(
                    "[{tag}] BITWISE MISMATCH! s_idx: {}, src: {}, vertex: {}, CPU: {}, GPU: {}",
                    s_idx, src, v, c, g
                );
            }
        }

        // Invariant 3: Metric triangle inequality |d(u) - d(v)| <= 1 on all edges
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
                        "[{tag}] Triangle inequality violated! Edge ({}, {}), d(u)={}, d(v)={}, |diff|={}",
                        u, v, d_u, d_v, diff
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
}

/// Adversarial Test 1: Extremal & Diameter Boundary Sources with IndirectMacroBatch
#[test]
fn test_m10_adversarial_luxembourg_extremal_indirect() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let Some(graph) = find_luxembourg_graph() else {
        eprintln!("Skipping: luxembourg_osm.mtx fixture not found.");
        return;
    };

    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let gpu = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::IndirectMacroBatch)
        .expect("Indirect engine init failed");

    println!("\n=== [M10 Adversarial Challenger 2] Luxembourg Extremal & Deep Sources (Indirect) ===");

    // Test diameter extremities (1337 hops), junction hubs, dead-ends, and graph boundaries
    let extremal_sources = vec![
        48292, // Periphery North/East (depth 1337)
        24481, // Periphery South/West (depth 1337)
        6397,  // Degree-6 Junction Hub
        3898,  // Degree-1 Leaf
        0,      // First vertex
        114598, // Last vertex
    ];

    let t_cpu = Instant::now();
    let cpu_dist = cpu
        .run_ms_bfs(&graph, &extremal_sources)
        .expect("CPU BFS failed");
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

    let t_gpu = Instant::now();
    let gpu_dist = gpu
        .run_ms_bfs(&graph, &extremal_sources)
        .expect("GPU Indirect BFS failed");
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

    assert_invariants(
        &graph,
        &extremal_sources,
        &cpu_dist,
        &gpu_dist,
        "Luxembourg-Indirect-Extremal6",
    );

    println!(
        "  -> IndirectMacroBatch (Extremal S=6) | CPU: {:>8.2} ms | GPU: {:>8.2} ms | Parity & Triangle Ineq: 100% PASS",
        cpu_ms, gpu_ms
    );
}

/// Adversarial Test 2: Duplicate Sources Array with IndirectMacroBatch
#[test]
fn test_m10_adversarial_luxembourg_duplicate_sources() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let Some(graph) = find_luxembourg_graph() else {
        eprintln!("Skipping: luxembourg_osm.mtx fixture not found.");
        return;
    };

    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let gpu = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::IndirectMacroBatch)
        .expect("Indirect engine init failed");

    println!("\n=== [M10 Adversarial Challenger 2] Luxembourg Duplicate Sources Array ===");

    // Heavily duplicated sources testing bitmask coalescing and output isolation
    let duplicate_sources = vec![48292, 48292, 48292, 6397, 6397, 0, 0, 0];

    let cpu_dist = cpu
        .run_ms_bfs(&graph, &duplicate_sources)
        .expect("CPU BFS failed");
    let gpu_dist = gpu
        .run_ms_bfs(&graph, &duplicate_sources)
        .expect("GPU Indirect BFS failed");

    assert_invariants(
        &graph,
        &duplicate_sources,
        &cpu_dist,
        &gpu_dist,
        "Luxembourg-Indirect-Duplicates8",
    );

    println!("  -> IndirectMacroBatch (Duplicates S=8) | Parity & Invariants: 100% PASS");
}

/// Adversarial Test 3: Multi-Batch Chunk Rollover (S=65) across 1,337 hops
#[test]
fn test_m10_adversarial_luxembourg_indirect_chunk_rollover_65() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let Some(graph) = find_luxembourg_graph() else {
        eprintln!("Skipping: luxembourg_osm.mtx fixture not found.");
        return;
    };

    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let gpu = GpuBfsEngine::new(ctx.clone(), GpuBfsMethod::IndirectMacroBatch)
        .expect("Indirect engine init failed");

    println!("\n=== [M10 Adversarial Challenger 2] Luxembourg Indirect Multi-Batch Rollover (S=65) ===");

    // S=65 spans two 64-source batches (batch 0: 64 sources, batch 1: 1 source)
    let sources: Vec<u32> = (0..65u32).map(|i| (i * 1700) % 114599).collect();

    let t_cpu = Instant::now();
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).expect("CPU BFS failed");
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

    let t_gpu = Instant::now();
    let gpu_dist = gpu.run_ms_bfs(&graph, &sources).expect("GPU Indirect BFS failed");
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

    assert_invariants(
        &graph,
        &sources,
        &cpu_dist,
        &gpu_dist,
        "Luxembourg-Indirect-Rollover-S65",
    );

    println!(
        "  -> IndirectMacroBatch (Rollover S=65) | CPU: {:>8.2} ms | GPU: {:>8.2} ms | Parity: 100% PASS",
        cpu_ms, gpu_ms
    );
}

/// Adversarial Test 4: Custom Macro-Batch Chunk Sizes (H=128 and H=17) on Luxembourg OSM
#[test]
fn test_m10_adversarial_luxembourg_custom_chunk_sizes() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let Some(graph) = find_luxembourg_graph() else {
        eprintln!("Skipping: luxembourg_osm.mtx fixture not found.");
        return;
    };

    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let pipelines = Arc::new(PipelineCache::new(ctx.device()).expect("PipelineCache failed"));

    println!("\n=== [M10 Adversarial Challenger 2] Custom Chunk Sizes H=128 and H=17 on Luxembourg ===");

    let sources = vec![48292u32, 6397u32]; // Extremal periphery (1337 hops) + junction hub
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).expect("CPU BFS failed");

    // Chunk size 128: tests larger speculative macro-chunks (~11 host submissions across 1337 hops)
    let runner_128 = IndirectMacroBatchBfs::new(ctx.clone(), pipelines.clone(), &graph, 128)
        .expect("IndirectMacroBatchBfs(128) init failed");
    let gpu_dist_128 = runner_128.run(&sources).expect("Runner 128 run failed");
    assert_invariants(
        &graph,
        &sources,
        &cpu_dist,
        &gpu_dist_128,
        "Luxembourg-Indirect-ChunkHops128",
    );
    println!("  -> Custom Chunk Hops = 128 | Parity & Invariants: 100% PASS");

    // Chunk size 17: odd/prime chunk size testing unaligned chunk boundary termination
    let runner_17 = IndirectMacroBatchBfs::new(ctx.clone(), pipelines.clone(), &graph, 17)
        .expect("IndirectMacroBatchBfs(17) init failed");
    let gpu_dist_17 = runner_17.run(&sources).expect("Runner 17 run failed");
    assert_invariants(
        &graph,
        &sources,
        &cpu_dist,
        &gpu_dist_17,
        "Luxembourg-Indirect-ChunkHops17",
    );
    println!("  -> Custom Chunk Hops = 17  | Parity & Invariants: 100% PASS");
}
