#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M24 Challenger 2: Adversarial Multi-Chunk Pipelining & Parity Suite
//! ==============================================================================
//!
//! Empirically challenges and stress-tests:
//! 1. Direct invocation of `run_indirect_sparse_push_512_into` (which activates
//!    `run_indirect_sparse_push_512_pipelined` with double-buffered staging).
//! 2. Multi-chunk batch sweeps:
//!    - S = 512 (single-chunk fast path)
//!    - S = 700 (2 chunks: 512 + 188 fractional)
//!    - S = 1024 (2 full chunks: overlaps chunk 0 CPU unpack with chunk 1 GPU traversal)
//!    - S = 1025 (3 chunks: 512 + 512 + 1 unaligned edge case)
//!    - S = 2048 (4 full chunks: exercises staging buffer ping-pong buffer reuse)
//! 3. Parity verification against CPU reference:
//!    - D < 255 (u8 quantization path)
//!    - D >= 255 (u16 quantization fallback path)
//! 4. Race-condition / buffer-reuse stress: repeated back-to-back runs.
//! 5. Equivalence between `run_indirect_sparse_push_512_into` and `run_indirect_sparse_push_into`.

use std::sync::Arc;
use webgpu_bfs_core::{
    generators::{generate_grid2d, star_graph},
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{
    context::GpuContext,
    dispatch::indirect_sparse_push::{
        run_indirect_sparse_push_512_into, run_indirect_sparse_push_into,
        PersistentSparsePushBuffers, PersistentSparsePushBuffers512,
    },
    pipeline::PipelineCache,
};

static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn assert_exact_parity(
    actual: &[u32],
    expected: &[u32],
    sources: &[u32],
    nv: usize,
    tag: &str,
) {
    let ns = sources.len();
    assert_eq!(
        actual.len(),
        ns * nv,
        "[{tag}] Actual length mismatch: expected {}, got {}",
        ns * nv,
        actual.len()
    );
    assert_eq!(
        expected.len(),
        ns * nv,
        "[{tag}] Expected length mismatch: expected {}, got {}",
        ns * nv,
        expected.len()
    );

    let mut mismatches = 0usize;
    let mut first_mismatch = None;

    for s_idx in 0..ns {
        for v in 0..nv {
            let idx = s_idx * nv + v;
            if actual[idx] != expected[idx] {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((s_idx, sources[s_idx], v, actual[idx], expected[idx]));
                }
            }
        }
    }

    if let Some((s_idx, src_id, v, act, exp)) = first_mismatch {
        panic!(
            "[{tag}] PARITY FAILURE: {mismatches}/{} mismatches ({:.2}%). \
             First mismatch at source index {s_idx} (node {src_id}), vertex {v}: \
             actual={act}, expected={exp}",
            ns * nv,
            (mismatches as f64 / (ns * nv) as f64) * 100.0
        );
    }
}

/// Constructs an undirected path graph 0 - 1 - ... - (n-1) of diameter n - 1.
fn make_path_graph(n: usize) -> CsrGraph {
    let num_edges = 2 * (n - 1);
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(num_edges);

    row_offsets.push(0);
    for i in 0..n {
        if i > 0 {
            col_indices.push((i - 1) as u32);
        }
        if i + 1 < n {
            col_indices.push((i + 1) as u32);
        }
        row_offsets.push(col_indices.len() as u32);
    }

    CsrGraph {
        num_vertices: n,
        num_edges,
        row_offsets,
        col_indices,
    }
}

#[test]
fn test_m24_pipelined_direct_sweep_u8_quantization() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let pipelines = Arc::new(PipelineCache::new(ctx.device()).expect("Failed to create PipelineCache"));
    let cpu = CpuBfsEngine::new();

    // 2D Grid 25x25 = 625 vertices (D = 48 < 255 -> triggers u8 quantization)
    let grid = generate_grid2d(25, 25);
    let nv = grid.num_vertices;

    let sweep_batch_sizes = [512, 700, 1024, 1025, 2048];

    let mut pool_512: Option<PersistentSparsePushBuffers512> = None;

    for &s in &sweep_batch_sizes {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 11) % (nv as u32)).collect();
        let cpu_dist = cpu.run_ms_bfs(&grid, &sources).expect("CPU BFS ground truth");

        let mut pipelined_dist = vec![0u32; s * nv];
        run_indirect_sparse_push_512_into(
            &ctx,
            &pipelines,
            &grid,
            &sources,
            Some(&mut pool_512),
            &mut pipelined_dist,
        )
        .unwrap_or_else(|e| panic!("run_indirect_sparse_push_512_into failed at S={s}: {e}"));

        assert_exact_parity(
            &pipelined_dist,
            &cpu_dist,
            &sources,
            nv,
            &format!("Pipelined_Grid25x25_S={s}_u8"),
        );
    }
}

#[test]
fn test_m24_pipelined_direct_sweep_u16_quantization() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let pipelines = Arc::new(PipelineCache::new(ctx.device()).expect("Failed to create PipelineCache"));
    let cpu = CpuBfsEngine::new();

    // Path graph N = 300 (D = 299 >= 255 -> triggers u16 quantization fallback)
    let path = make_path_graph(300);
    let nv = path.num_vertices;

    let sweep_batch_sizes = [512, 700, 1024, 1025, 2048];

    let mut pool_512: Option<PersistentSparsePushBuffers512> = None;

    for &s in &sweep_batch_sizes {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 17) % (nv as u32)).collect();
        let cpu_dist = cpu.run_ms_bfs(&path, &sources).expect("CPU BFS ground truth");

        let mut pipelined_dist = vec![0u32; s * nv];
        run_indirect_sparse_push_512_into(
            &ctx,
            &pipelines,
            &path,
            &sources,
            Some(&mut pool_512),
            &mut pipelined_dist,
        )
        .unwrap_or_else(|e| panic!("run_indirect_sparse_push_512_into failed at S={s}: {e}"));

        assert_exact_parity(
            &pipelined_dist,
            &cpu_dist,
            &sources,
            nv,
            &format!("Pipelined_Path300_S={s}_u16"),
        );
    }
}

#[test]
fn test_m24_pipelined_repeated_stress_race_condition() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let pipelines = Arc::new(PipelineCache::new(ctx.device()).expect("Failed to create PipelineCache"));
    let cpu = CpuBfsEngine::new();

    // Star graph 400 vertices
    let star = star_graph(400);
    let nv = star.num_vertices;
    let s = 2048;

    let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % (nv as u32)).collect();
    let cpu_dist = cpu.run_ms_bfs(&star, &sources).expect("CPU BFS ground truth");

    let mut pool_512: Option<PersistentSparsePushBuffers512> = None;

    // Run 5 back-to-back iterations on the same persistent pool to stress buffer reuse & races
    for iter in 0..5 {
        let mut pipelined_dist = vec![0xDEADBEEFu32; s * nv];
        run_indirect_sparse_push_512_into(
            &ctx,
            &pipelines,
            &star,
            &sources,
            Some(&mut pool_512),
            &mut pipelined_dist,
        )
        .unwrap_or_else(|e| panic!("Stress iteration {iter} failed: {e}"));

        assert_exact_parity(
            &pipelined_dist,
            &cpu_dist,
            &sources,
            nv,
            &format!("Stress_Iter{iter}_S=2048"),
        );
    }
}

#[test]
fn test_m24_pipelined_vs_unpipelined_audit() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let pipelines = Arc::new(PipelineCache::new(ctx.device()).expect("Failed to create PipelineCache"));

    let grid = generate_grid2d(20, 20);
    let nv = grid.num_vertices;
    let s = 1024;
    let sources: Vec<u32> = (0..s as u32).map(|i| (i * 3) % (nv as u32)).collect();

    let mut pool_512: Option<PersistentSparsePushBuffers512> = None;
    let mut pool_generic: Option<PersistentSparsePushBuffers> = None;

    let mut dist_pipelined = vec![0u32; s * nv];
    let mut dist_unpipelined = vec![0u32; s * nv];

    run_indirect_sparse_push_512_into(
        &ctx,
        &pipelines,
        &grid,
        &sources,
        Some(&mut pool_512),
        &mut dist_pipelined,
    )
    .expect("Pipelined run failed");

    run_indirect_sparse_push_into(
        &ctx,
        &pipelines,
        &grid,
        &sources,
        Some(&mut pool_generic),
        &mut dist_unpipelined,
    )
    .expect("Unpipelined run failed");

    assert_eq!(
        dist_pipelined, dist_unpipelined,
        "Pipelined and unpipelined output arrays must be bitwise identical!"
    );
}

#[test]
fn test_m24_pipelined_vs_unpipelined_web_stanford_timing() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mtx_path = std::path::Path::new("../../data/web-Stanford/web-Stanford.mtx");
    if !mtx_path.exists() {
        println!("Skipping web-Stanford timing: file not found at {:?}", mtx_path);
        return;
    }

    let file = std::fs::File::open(mtx_path).expect("Failed to open web-Stanford.mtx");
    let mut reader = std::io::BufReader::new(file);
    let graph = CsrGraph::from_mtx_reader(&mut reader).expect("Failed to parse MTX");
    let nv = graph.num_vertices;

    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let pipelines = Arc::new(PipelineCache::new(ctx.device()).expect("Failed to create PipelineCache"));

    for &s in &[1024, 2048] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 73) % (nv as u32)).collect();

        // 1. Pipelined execution
        let mut pool_pipelined: Option<PersistentSparsePushBuffers512> = None;
        let mut dist_pipelined = vec![0u32; s * nv];
        let t0 = std::time::Instant::now();
        run_indirect_sparse_push_512_into(
            &ctx,
            &pipelines,
            &graph,
            &sources,
            Some(&mut pool_pipelined),
            &mut dist_pipelined,
        )
        .expect("Pipelined execution on web-Stanford failed");
        let pipelined_elapsed = t0.elapsed();

        // 2. Unpipelined execution
        let mut pool_unpipelined: Option<PersistentSparsePushBuffers> = None;
        let mut dist_unpipelined = vec![0u32; s * nv];
        let t1 = std::time::Instant::now();
        run_indirect_sparse_push_into(
            &ctx,
            &pipelines,
            &graph,
            &sources,
            Some(&mut pool_unpipelined),
            &mut dist_unpipelined,
        )
        .expect("Unpipelined execution on web-Stanford failed");
        let unpipelined_elapsed = t1.elapsed();

        println!(
            "[web-Stanford S={s}] Pipelined: {:?}, Unpipelined: {:?}, Difference: {:?}",
            pipelined_elapsed,
            unpipelined_elapsed,
            unpipelined_elapsed.saturating_sub(pipelined_elapsed)
        );

        assert_eq!(
            dist_pipelined, dist_unpipelined,
            "Pipelined vs Unpipelined output parity mismatch on web-Stanford at S={s}"
        );
    }
}

