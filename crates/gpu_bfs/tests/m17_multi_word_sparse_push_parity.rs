#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M17: Multi-Word Sparse Push & Topology Router Parity Test Suite
//! ==============================================================================
//!
//! Validates 100% bitwise parity against `CpuBfsEngine` (Rayon multi-threaded reference)
//! and analytical ground truth across large-batch scaling (S in {64, 256, 512, 1024}),
//! multi-bank distance storage buffers, persistent buffer pooling, and topology-aware
//! routing (`GpuBfsMethod::Auto`).

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;

use webgpu_bfs_core::{
    generators::{generate_graph500_rmat, star_graph, Grid2D},
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

/// Global test synchronization mutex preventing driver race conditions during parallel execution.
static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn assert_distance_parity(
    gpu_dist: &[u32],
    cpu_dist: &[u32],
    num_sources: usize,
    num_vertices: usize,
    tag: &str,
) {
    assert_eq!(
        gpu_dist.len(),
        num_sources * num_vertices,
        "[{tag}] GPU distance buffer size mismatch"
    );
    assert_eq!(
        cpu_dist.len(),
        num_sources * num_vertices,
        "[{tag}] CPU distance buffer size mismatch"
    );

    let mut mismatches = 0;
    let mut first_mismatch = None;

    for s_idx in 0..num_sources {
        for v in 0..num_vertices {
            let idx = s_idx * num_vertices + v;
            let d_gpu = gpu_dist[idx];
            let d_cpu = cpu_dist[idx];
            if d_gpu != d_cpu {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((s_idx, v, d_gpu, d_cpu));
                }
            }
        }
    }

    if let Some((s_idx, v, d_gpu, d_cpu)) = first_mismatch {
        panic!(
            "[{tag}] PARITY MISMATCH: {mismatches}/{} entries differ. First mismatch at source index {s_idx}, vertex {v}: GPU={d_gpu}, CPU={d_cpu}",
            num_sources * num_vertices
        );
    }
}

/// Test 1: Pure unit test of `select_auto_method` on diverse graph topologies.
#[test]
fn test_auto_routing_heuristic() {
    // 1. Planar low-degree mesh (Grid2D: max degree 4 <= 32)
    let grid = Grid2D::new(20, 20).unwrap();
    assert_eq!(grid.graph.max_degree(), 4);
    assert_eq!(
        GpuBfsEngine::select_auto_method(&grid.graph, 64),
        GpuBfsMethod::WorkgroupPersistentBfs
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&grid.graph, 512),
        GpuBfsMethod::WorkgroupPersistentBfs
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&grid.graph, 32),
        GpuBfsMethod::WorkgroupPersistentBfs
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&grid.graph, 16),
        GpuBfsMethod::WorkgroupPersistentBfs
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&grid.graph, 1),
        GpuBfsMethod::WorkgroupPersistentBfs
    );

    // 2. Extreme star hub (max degree 599 >= 128)
    let star = star_graph(600);
    assert_eq!(star.max_degree(), 599);
    assert_eq!(
        GpuBfsEngine::select_auto_method(&star, 64),
        GpuBfsMethod::IndirectSparseQueuePush
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&star, 512),
        GpuBfsMethod::IndirectSparseQueuePush
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&star, 32),
        GpuBfsMethod::IndirectSparseQueuePush
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&star, 1),
        GpuBfsMethod::IndirectSparseQueuePush
    );

    // 3. Power-law R-MAT graph (hubs with max degree >= 128)
    let rmat = generate_graph500_rmat(10, 16, Some(42)).unwrap();
    assert!(
        rmat.max_degree() >= 128,
        "Expected R-MAT hub degree >= 128, got {}",
        rmat.max_degree()
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&rmat, 64),
        GpuBfsMethod::IndirectSparseQueuePush
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&rmat, 1024),
        GpuBfsMethod::IndirectSparseQueuePush
    );

    // 4. Edge cases
    let empty = webgpu_bfs_core::generators::empty_graph();
    assert_eq!(
        GpuBfsEngine::select_auto_method(&empty, 64),
        GpuBfsMethod::NaiveSequential
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&rmat, 0),
        GpuBfsMethod::NaiveSequential
    );
}

/// Test 2: Synthetic Graph500 R-MAT power-law graph across batch scaling S in {64, 256, 512, 1024}.
#[test]
fn test_synthetic_rmat_power_law_batch_scaling() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let graph = generate_graph500_rmat(10, 16, Some(42)).expect("Failed to generate R-MAT graph");
    let nv = graph.num_vertices;

    let isqp_engine = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create IndirectSparseQueuePush engine");
    let auto_engine = GpuBfsEngine::auto(ctx.clone()).expect("Failed to create Auto engine");

    for &s in &[64, 256, 512, 1024] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % nv as u32).collect();

        let cpu_dist = cpu
            .run_ms_bfs(&graph, &sources)
            .expect("CPU BFS run failed");

        // Test IndirectSparseQueuePush directly
        let isqp_dist = isqp_engine
            .run_ms_bfs(&graph, &sources)
            .unwrap_or_else(|e| panic!("ISQP BFS failed at S={s}: {e}"));
        assert_distance_parity(&isqp_dist, &cpu_dist, s, nv, &format!("RMAT_ISQP_S={s}"));

        // Test Auto router (routes to IndirectSparseQueuePush for R-MAT)
        let auto_dist = auto_engine
            .run_ms_bfs(&graph, &sources)
            .unwrap_or_else(|e| panic!("Auto BFS failed at S={s}: {e}"));
        assert_distance_parity(&auto_dist, &cpu_dist, s, nv, &format!("RMAT_Auto_S={s}"));
    }
}

/// Test 3: Synthetic star graph with an extreme central hub (degree 599) across batch sizes S in {64, 256, 512, 1024}.
#[test]
fn test_synthetic_star_extreme_hub_scaling() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let graph = star_graph(600);
    let nv = graph.num_vertices;

    let isqp_engine = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create IndirectSparseQueuePush engine");

    for &s in &[64, 256, 512, 1024] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % nv as u32).collect();

        let cpu_dist = cpu
            .run_ms_bfs(&graph, &sources)
            .expect("CPU BFS run failed");

        let gpu_dist = isqp_engine
            .run_ms_bfs(&graph, &sources)
            .unwrap_or_else(|e| panic!("Star ISQP BFS failed at S={s}: {e}"));

        assert_distance_parity(&gpu_dist, &cpu_dist, s, nv, &format!("Star_S={s}"));
    }
}

/// Test 4: 2D Grid graph non-regression across batch scaling S in {64, 256, 512, 1024} against analytical Manhattan oracle.
#[test]
fn test_grid2d_non_regression() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let grid = Grid2D::new(20, 20).expect("Failed to create Grid2D");
    let nv = grid.graph.num_vertices;

    let isqp_engine = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create IndirectSparseQueuePush engine");
    let auto_engine = GpuBfsEngine::auto(ctx.clone()).expect("Failed to create Auto engine");

    for &s in &[64, 256, 512, 1024] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 11) % nv as u32).collect();

        let oracle_dist = grid.ms_bfs_oracle(&sources);
        let cpu_dist = cpu
            .run_ms_bfs(&grid.graph, &sources)
            .expect("CPU BFS run failed");

        assert_distance_parity(
            &cpu_dist,
            &oracle_dist,
            s,
            nv,
            &format!("Grid2D_CPU_vs_Oracle_S={s}"),
        );

        // ISQP parity against oracle
        let isqp_dist = isqp_engine
            .run_ms_bfs(&grid.graph, &sources)
            .unwrap_or_else(|e| panic!("Grid2D ISQP BFS failed at S={s}: {e}"));
        assert_distance_parity(
            &isqp_dist,
            &oracle_dist,
            s,
            nv,
            &format!("Grid2D_ISQP_vs_Oracle_S={s}"),
        );

        // Auto router parity against oracle (Auto routes Grid2D to MultiWordBitParallel)
        let auto_dist = auto_engine
            .run_ms_bfs(&grid.graph, &sources)
            .unwrap_or_else(|e| panic!("Grid2D Auto BFS failed at S={s}: {e}"));
        assert_distance_parity(
            &auto_dist,
            &oracle_dist,
            s,
            nv,
            &format!("Grid2D_Auto_vs_Oracle_S={s}"),
        );
    }
}

/// Helper to locate `web-Stanford-undirected.mtx` across multiple relative directories.
fn find_web_stanford_undirected_path() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("data/web-Stanford/web-Stanford-undirected.mtx"),
        PathBuf::from("../../data/web-Stanford/web-Stanford-undirected.mtx"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/web-Stanford/web-Stanford-undirected.mtx"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data/web-Stanford/web-Stanford-undirected.mtx"),
    ];

    for path in &candidates {
        if path.exists() {
            return Some(path.clone());
        }
    }
    None
}

/// Test 5: Real-world `web-Stanford-undirected` large-batch parity (S in {64, 256, 512, 1024}) if dataset exists.
#[test]
fn test_real_web_stanford_undirected_large_batch() {
    let Some(path) = find_web_stanford_undirected_path() else {
        eprintln!("Skipping: web-Stanford-undirected.mtx dataset not found.");
        return;
    };

    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let file = File::open(&path).expect("Failed to open web-Stanford-undirected.mtx");
    let reader = BufReader::new(file);
    let graph =
        CsrGraph::from_mtx_reader(reader).expect("Failed to parse web-Stanford-undirected.mtx");
    let nv = graph.num_vertices;

    let isqp_engine = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create IndirectSparseQueuePush engine");
    let auto_engine = GpuBfsEngine::auto(ctx.clone()).expect("Failed to create Auto engine");

    for &s in &[64, 256, 512, 1024] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 997) % nv as u32).collect();

        let cpu_dist = cpu
            .run_ms_bfs(&graph, &sources)
            .expect("CPU BFS run failed");

        let isqp_dist = isqp_engine
            .run_ms_bfs(&graph, &sources)
            .unwrap_or_else(|e| panic!("web-Stanford-undirected ISQP BFS failed at S={s}: {e}"));
        assert_distance_parity(
            &isqp_dist,
            &cpu_dist,
            s,
            nv,
            &format!("Stanford_ISQP_S={s}"),
        );

        let auto_dist = auto_engine
            .run_ms_bfs(&graph, &sources)
            .unwrap_or_else(|e| panic!("web-Stanford-undirected Auto BFS failed at S={s}: {e}"));
        assert_distance_parity(
            &auto_dist,
            &cpu_dist,
            s,
            nv,
            &format!("Stanford_Auto_S={s}"),
        );
    }
}

/// Test 6: Reusable output buffer consistency across repeated invocations of `run_ms_bfs_into`.
#[test]
fn test_reusable_output_into_consistency() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let graph = generate_graph500_rmat(10, 16, Some(42)).expect("Failed to generate R-MAT graph");
    let nv = graph.num_vertices;
    let s = 512;

    let engine = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create IndirectSparseQueuePush engine");

    let mut output_buffer = vec![0u32; s * nv];

    // Iteration 1: sources set A
    let sources_a: Vec<u32> = (0..s as u32).map(|i| (i * 17) % nv as u32).collect();
    let cpu_dist_a = cpu
        .run_ms_bfs(&graph, &sources_a)
        .expect("CPU BFS run failed");

    engine
        .run_ms_bfs_into(&graph, &sources_a, &mut output_buffer)
        .expect("run_ms_bfs_into iteration 1 failed");
    assert_distance_parity(
        &output_buffer,
        &cpu_dist_a,
        s,
        nv,
        "Reusable_Output_Iteration_1",
    );

    // Corrupt buffer with sentinel noise to prove overwrite correctness
    output_buffer.fill(0xdeadbeef);

    // Iteration 2: sources set B
    let sources_b: Vec<u32> = (0..s as u32).map(|i| (i * 31) % nv as u32).collect();
    let cpu_dist_b = cpu
        .run_ms_bfs(&graph, &sources_b)
        .expect("CPU BFS run failed");

    engine
        .run_ms_bfs_into(&graph, &sources_b, &mut output_buffer)
        .expect("run_ms_bfs_into iteration 2 failed");
    assert_distance_parity(
        &output_buffer,
        &cpu_dist_b,
        s,
        nv,
        "Reusable_Output_Iteration_2",
    );
}
