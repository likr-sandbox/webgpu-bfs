#![allow(clippy::identity_op, clippy::erasing_op)]

//! Milestone M7 Challenger 2: Real-World Dataset & Scalability Empirical Stress Suite.
//!
//! Specifically validates Milestone M7's implementations:
//! 1. `GpuBfsMethod::SparseQueue`
//! 2. `GpuBfsMethod::SparseBitParallel64`
//!
//! on SuiteSparse `luxembourg_osm` (114,599 vertices, 239,332 directed edges, diameter ~1337).
//!
//! Empirical Verification Points:
//! 1. Extremal & Topological Boundary Sources:
//!    - Periphery North/East (node 48292, max depth 1337)
//!    - Periphery South/West (node 24481, max depth 1337)
//!    - Topological Center (node 4993, max depth 669)
//!    - Max-degree junction hub (node 6397, deg 6)
//!    - Min-degree leaves / dead ends (node 3898, 5112, deg 1)
//!    - Boundary vertex indices (0, 114598)
//! 2. Multi-Source Batch Scaling:
//!    - Extremal 8-source set on both `SparseQueue` and `SparseBitParallel64`
//!    - Full 64-source bit-parallel batch on `SparseBitParallel64`
//!    - 65-source chunk rollover test (64 + 1 sources) on `SparseBitParallel64`
//! 3. 100% Bitwise Parity against `CpuBfsEngine` across all 114,599 vertices.
//! 4. 100% Metric Triangle Inequality (|d(u) - d(v)| <= 1) across all 239,332 edges.
//! 5. Quantitative GPU Thread Reduction analysis proving >500x elimination of idle threads.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use webgpu_bfs_core::{BfsEngine, CsrGraph};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Load the SuiteSparse luxembourg_osm graph
fn load_luxembourg_graph() -> CsrGraph {
    let candidates = [
        PathBuf::from("data/luxembourg_osm/luxembourg_osm.mtx"),
        PathBuf::from("../../data/luxembourg_osm/luxembourg_osm.mtx"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/luxembourg_osm/luxembourg_osm.mtx"),
    ];
    let lux_path = candidates
        .iter()
        .find(|p| p.exists())
        .expect("data/luxembourg_osm/luxembourg_osm.mtx not found");

    let file = File::open(lux_path).expect("Failed to open luxembourg_osm.mtx");
    let reader = BufReader::new(file);
    let graph = CsrGraph::from_mtx_reader(reader).expect("Failed to parse luxembourg_osm.mtx");
    assert_eq!(graph.num_vertices, 114599, "Expected 114,599 vertices in luxembourg_osm");
    assert_eq!(graph.num_edges(), 239332, "Expected 239,332 directed edges in luxembourg_osm");
    graph
}

/// Verifies bitwise parity and metric invariants against CpuBfsEngine
fn verify_luxembourg_sources(
    ctx: &Arc<GpuContext>,
    cpu_engine: &CpuBfsEngine,
    graph: &CsrGraph,
    sources: &[u32],
    method: GpuBfsMethod,
    label: &str,
) {
    let nv = graph.num_vertices;
    let s_count = sources.len();

    println!(
        "  [Challenger 2] Testing {:<20} | Batch S={:<2} | Label: {}",
        format!("{:?}", method),
        s_count,
        label
    );

    // 1. Run CPU Reference
    let t_cpu = Instant::now();
    let cpu_dist = cpu_engine.run_ms_bfs(graph, sources).expect("CPU BFS failed");
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(cpu_dist.len(), s_count * nv);

    // 2. Run GPU Implementation
    let gpu_engine = GpuBfsEngine::new(ctx.clone(), method)
        .unwrap_or_else(|e| panic!("Failed to create GPU engine for {:?}: {e}", method));

    let t_gpu = Instant::now();
    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{label}] GPU {:?} failed: {e}", method));
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(gpu_dist.len(), s_count * nv);

    println!(
        "    -> CPU Time: {:>8.2} ms | GPU Time: {:>8.2} ms",
        cpu_ms, gpu_ms
    );

    // 3. Bitwise Equality and Self-Distance Verification
    let mut total_distance_checks = 0usize;
    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * nv;

        // Self-distance
        assert_eq!(
            gpu_dist[base + src as usize],
            0,
            "[{label}] {:?} source self-distance != 0 for src {}",
            method,
            src
        );

        // Bitwise parity across all 114,599 vertices
        for v in 0..nv {
            let idx = base + v;
            let c = cpu_dist[idx];
            let g = gpu_dist[idx];
            if c != g {
                panic!(
                    "[{label}] BITWISE MISMATCH! Method: {:?}, s_idx: {}, src: {}, vertex: {}, CPU: {}, GPU: {}",
                    method, s_idx, src, v, c, g
                );
            }
            // All nodes in luxembourg_osm are connected, so distance must be finite
            assert_ne!(
                g,
                u32::MAX,
                "[{label}] Disconnected node {} encountered for src {} in connected graph",
                v,
                src
            );
            total_distance_checks += 1;
        }

        // 4. Metric Triangle Inequality: |d(u) - d(v)| <= 1 on all 239,332 edges
        let mut edge_count = 0usize;
        for u in 0..nv {
            let start = graph.row_offsets[u] as usize;
            let end = graph.row_offsets[u + 1] as usize;
            let d_u = gpu_dist[base + u];

            for edge_idx in start..end {
                let v = graph.col_indices[edge_idx] as usize;
                let d_v = gpu_dist[base + v];

                let diff = (d_u as i64 - d_v as i64).abs();
                assert!(
                    diff <= 1,
                    "[{label}] Triangle inequality violated! Method: {:?}, edge ({}, {}), d(u)={}, d(v)={}",
                    method, u, v, d_u, d_v
                );
                edge_count += 1;
            }
        }
        assert_eq!(edge_count, 239332);
    }

    println!(
        "    -> PASSED: {} distances checked (100% bitwise parity) | {} edges checked (100% triangle inequality)",
        total_distance_checks,
        s_count * 239332
    );
}

#[test]
fn test_m7_luxembourg_extremal_sparse_queue() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let graph = load_luxembourg_graph();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [Challenger 2] Luxembourg Extremal Single Sources (SparseQueue) ===");
    // 8 extremal sources tested individually
    let extremal_sources = [
        (48292u32, "Diameter Extremity 1 (Periphery NE, max_depth=1337)"),
        (24481u32, "Diameter Extremity 2 (Periphery SW, max_depth=1337)"),
        (4993u32, "Graph Metric Center (max_depth=669)"),
        (6397u32, "Unique Degree-6 Hub / Junction"),
        (3898u32, "Dead-End Leaf Node (Degree 1)"),
        (5112u32, "Dead-End Leaf Node (Degree 1)"),
        (0u32, "First Vertex Index (v=0)"),
        (114598u32, "Last Vertex Index (v=N-1)"),
    ];

    for (src, label) in extremal_sources {
        verify_luxembourg_sources(
            &ctx,
            &cpu,
            &graph,
            &[src],
            GpuBfsMethod::SparseQueue,
            label,
        );
    }
}

#[test]
fn test_m7_luxembourg_multisource_sparse_bp64_extremal_batch() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let graph = load_luxembourg_graph();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [Challenger 2] Luxembourg Extremal Batch S=8 (SparseBitParallel64) ===");
    let sources = vec![
        48292,  // Periphery 1
        24481,  // Periphery 2
        4993,   // Center
        6397,   // Degree 6 hub
        3898,   // Leaf 1
        5112,   // Leaf 2
        0,      // Node 0
        114598, // Node N-1
    ];

    // Test both SparseQueue (multi-source sequential) and SparseBitParallel64 (bit-parallel)
    verify_luxembourg_sources(
        &ctx,
        &cpu,
        &graph,
        &sources,
        GpuBfsMethod::SparseQueue,
        "SparseQueue-Extremal-8",
    );

    verify_luxembourg_sources(
        &ctx,
        &cpu,
        &graph,
        &sources,
        GpuBfsMethod::SparseBitParallel64,
        "SparseBitParallel64-Extremal-8",
    );
}

#[test]
fn test_m7_luxembourg_sparse_bp64_full_64_batch() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let graph = load_luxembourg_graph();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [Challenger 2] Luxembourg Full 64-Source Batch (SparseBitParallel64) ===");
    // Construct 64 diverse sources: 8 extremal + 56 distributed across graph
    let mut sources = vec![48292, 24481, 4993, 6397, 3898, 5112, 0, 114598];
    for i in 1..=56u32 {
        let s = (i * 1993) % 114599;
        if !sources.contains(&s) {
            sources.push(s);
        }
    }
    while sources.len() < 64 {
        let s = (sources.len() as u32 * 3137) % 114599;
        sources.push(s);
    }
    assert_eq!(sources.len(), 64);

    verify_luxembourg_sources(
        &ctx,
        &cpu,
        &graph,
        &sources,
        GpuBfsMethod::SparseBitParallel64,
        "Full-64-Batch-Mixed",
    );
}

#[test]
fn test_m7_luxembourg_sparse_bp64_chunk_rollover_65() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let graph = load_luxembourg_graph();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [Challenger 2] Luxembourg Multi-Chunk Rollover S=65 (SparseBitParallel64) ===");
    // S=65 forces SparseBitParallel64 to process chunk 0 (64 sources) followed by chunk 1 (1 source)
    let mut sources = Vec::with_capacity(65);
    for i in 0..65u32 {
        sources.push((i * 1753) % 114599);
    }
    assert_eq!(sources.len(), 65);

    verify_luxembourg_sources(
        &ctx,
        &cpu,
        &graph,
        &sources,
        GpuBfsMethod::SparseBitParallel64,
        "MultiChunk-Rollover-65",
    );
}

#[test]
fn test_m7_luxembourg_quantitative_thread_reduction() {
    let graph = load_luxembourg_graph();
    let cpu = CpuBfsEngine::new();

    println!("\n=== [Challenger 2] Luxembourg Quantitative Thread Reduction Analysis ===");
    let src = 48292u32;
    let dist = cpu.run_ms_bfs(&graph, &[src]).expect("CPU BFS failed");

    let nv = graph.num_vertices;
    let max_dist = *dist.iter().max().unwrap();
    assert_eq!(max_dist, 1337, "Luxembourg diameter from 48292 must be 1337 hops");

    // Compute frontier histogram per level
    let mut level_counts = vec![0usize; (max_dist + 1) as usize];
    for &d in &dist {
        if d != u32::MAX {
            level_counts[d as usize] += 1;
        }
    }

    // 1. Dense SpMV Thread Calculations:
    // Every level launches ceil(|V| / 128) workgroups = 896 workgroups = 114,688 threads
    let dense_workgroups_per_hop = nv.div_ceil(128);
    let dense_threads_per_hop = dense_workgroups_per_hop * 128;
    let dense_total_threads = (max_dist as usize) * dense_threads_per_hop;

    // 2. Sparse Queue Thread Calculations:
    // Each level launches ceil(|F_h| / 128) workgroups
    let mut sparse_total_workgroups = 0usize;
    let mut max_frontier_size = 0usize;
    for &f_size in level_counts.iter().take(max_dist as usize) {
        if f_size > max_frontier_size {
            max_frontier_size = f_size;
        }
        let wgs = f_size.div_ceil(128);
        sparse_total_workgroups += wgs;
    }
    let sparse_total_threads = sparse_total_workgroups * 128;

    let reduction_ratio = dense_total_threads as f64 / sparse_total_threads as f64;
    let avg_frontier = nv as f64 / max_dist as f64;

    println!("  Total Vertices (|V|)        : {}", nv);
    println!("  Total Directed Edges (|E|)  : {}", graph.num_edges());
    println!("  Traversal Diameter (Hops)   : {}", max_dist);
    println!("  Average Frontier Size / Hop : {:.2} vertices", avg_frontier);
    println!("  Maximum Frontier Size       : {} vertices", max_frontier_size);
    println!("  Dense SpMV Threads Launched : {:>12}", dense_total_threads);
    println!("  Sparse Queue Threads Launch : {:>12}", sparse_total_threads);
    println!("  Thread Reduction Ratio      : {:>10.2}x", reduction_ratio);

    // Assert quantitative thresholds
    assert_eq!(dense_threads_per_hop, 114688);
    assert_eq!(dense_total_threads, 153_337_856);
    assert!(max_frontier_size < 300, "Max frontier size unexpectedly large: {}", max_frontier_size);
    assert!(reduction_ratio > 500.0, "Thread reduction ratio must exceed 500x, got {:.2}x", reduction_ratio);
    assert!(reduction_ratio >= 700.0, "Expected ~706x reduction, got {:.2}x", reduction_ratio);
}
