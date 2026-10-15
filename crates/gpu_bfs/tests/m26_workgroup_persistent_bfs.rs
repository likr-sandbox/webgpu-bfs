#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M26: WorkgroupPersistentBfs Autonomous Traversal Verification Suite
//! ==============================================================================
//!
//! Tests:
//! 1. Autonomous 1 Workgroup/Source Single-Dispatch BFS correctness.
//! 2. 100% Bitwise Parity against CPU Reference (Rayon / FIFO).
//! 3. Compact Direct Reachability Bitmask Parity.
//! 4. Multi-chunk Pipelining at S=1024 with Spatial Source Ordering.
//! 5. High-diameter road-network-like paths and 2D grid meshes.

use std::sync::Arc;

use webgpu_bfs_core::{
    generators::{generate_grid2d, Grid2D},
    BfsEngine, CsrGraph, ReachabilityEngine,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuContext};

/// Global test mutex preventing driver command-queue races under WSL2 Direct3D 12.
static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn assert_parity(
    gpu_dist: &[u32],
    cpu_dist: &[u32],
    num_sources: usize,
    num_vertices: usize,
    tag: &str,
) {
    assert_eq!(
        gpu_dist.len(),
        num_sources * num_vertices,
        "[{tag}] GPU distance buffer size mismatch: got {}, expected {}",
        gpu_dist.len(),
        num_sources * num_vertices
    );
    assert_eq!(
        cpu_dist.len(),
        num_sources * num_vertices,
        "[{tag}] CPU distance buffer size mismatch: got {}, expected {}",
        cpu_dist.len(),
        num_sources * num_vertices
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

/// Constructs an undirected path graph 0 - 1 - 2 - ... - (n - 1) of diameter n - 1.
fn make_path_graph(n: usize) -> CsrGraph {
    assert!(n >= 2, "Path graph requires at least 2 vertices");
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
fn test_m26_workgroup_persistent_bfs_grid_and_path_parity() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let engine = GpuBfsEngine::workgroup_persistent_bfs(ctx.clone())
        .expect("Failed to create WorkgroupPersistentBfs engine");

    // 1. Grid 20x20 (nv = 400, max_degree = 4)
    let grid = generate_grid2d(20, 20);
    for &s in &[1, 16, 64, 128, 256, 512] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 400).collect();
        let cpu_dist = cpu.run_ms_bfs(&grid, &sources).expect("CPU BFS");
        let gpu_dist = engine
            .run_ms_bfs(&grid, &sources)
            .unwrap_or_else(|e| panic!("WorkgroupPersistentBfs failed on grid(20x20) S={s}: {e}"));

        assert_parity(&gpu_dist, &cpu_dist, s, grid.num_vertices, &format!("Grid(20x20)_S={s}"));
    }

    // 2. High-diameter Path (nv = 500, max_degree = 2, D = 499)
    let path = make_path_graph(500);
    for &s in &[1, 16, 64] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 37) % 500).collect();
        let cpu_dist = cpu.run_ms_bfs(&path, &sources).expect("CPU BFS");
        let gpu_dist = engine
            .run_ms_bfs(&path, &sources)
            .unwrap_or_else(|e| panic!("WorkgroupPersistentBfs failed on path(500) S={s}: {e}"));

        assert_parity(&gpu_dist, &cpu_dist, s, path.num_vertices, &format!("Path(500)_S={s}"));
    }
}

#[test]
fn test_m26_workgroup_persistent_reachability_parity() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let engine = GpuBfsEngine::workgroup_persistent_bfs(ctx.clone())
        .expect("Failed to create WorkgroupPersistentBfs engine");

    // Grid 15x15
    let grid = Grid2D::new(15, 15).unwrap().graph;
    let nv = grid.num_vertices;

    for &s in &[1, 16, 64, 128] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 11) % nv as u32).collect();
        let cpu_mask = cpu.run_reachability(&grid, &sources).expect("CPU reachability");
        let gpu_mask = engine.run_reachability(&grid, &sources).expect("GPU reachability");

        assert_eq!(
            gpu_mask.words.len(),
            cpu_mask.words.len(),
            "Reachability words length mismatch at S={s}"
        );
        assert_eq!(
            gpu_mask.words, cpu_mask.words,
            "Reachability bitmask contents mismatch at S={s}"
        );
    }
}

#[test]
fn test_m26_workgroup_persistent_multichunk_large_batch() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let engine = GpuBfsEngine::workgroup_persistent_bfs(ctx.clone())
        .expect("Failed to create WorkgroupPersistentBfs engine");

    // Narrow grid 5x60 = 300 vertices
    let grid = generate_grid2d(5, 60);
    let s = 1024;
    let sources: Vec<u32> = (0..s as u32).map(|i| (i * 3) % 300).collect();

    let cpu_dist = cpu.run_ms_bfs(&grid, &sources).expect("CPU BFS");
    let gpu_dist = engine
        .run_ms_bfs(&grid, &sources)
        .expect("WorkgroupPersistentBfs S=1024");

    assert_parity(&gpu_dist, &cpu_dist, s, grid.num_vertices, "NarrowGrid(5x60)_S=1024");

    // Reachability at S=1024
    let cpu_mask = cpu.run_reachability(&grid, &sources).expect("CPU reachability");
    let gpu_mask = engine.run_reachability(&grid, &sources).expect("GPU reachability");
    assert_eq!(gpu_mask.words, cpu_mask.words, "Reachability mismatch at S=1024");
}
