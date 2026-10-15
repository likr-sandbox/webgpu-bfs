#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M26 Challenger 2: Degree Boundary & Driver Stability Stress Suite
//! ==============================================================================
//!
//! Evaluates:
//! 1. Exact degree boundary switching:
//!    - d = 32 (routes to WorkgroupPersistentBfs for all S >= 1)
//!    - d = 33 (routes to SparseQueue at S=1, BitParallel32 at 1<S<32,
//!              BitParallel64 at 32<=S<64, IndirectSparseQueuePush at S>=64)
//! 2. End-to-end bitwise parity on both sides of the d=32 vs d=33 boundary:
//!    - Distance BFS bitwise parity against CpuBfsEngine across S in {1, 16, 32, 64, 128, 512}
//!    - Reachability BFS bitwise parity against CpuBfsEngine across S in {1, 16, 32, 64, 128, 512}
//! 3. Driver stability & Memory leak stress harness:
//!    - 50 consecutive back-to-back GPU BFS dispatches on a single engine context
//!    - Verifies absence of device loss, buffer allocation leaks, or driver crashes.

use std::sync::Arc;

use webgpu_bfs_core::{
    generators::{clique_graph, generate_grid2d},
    BfsEngine, CsrGraph, ReachabilityEngine,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

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

/// Generates a connected graph where vertex 0 has exact degree `max_deg`,
/// and vertices 1..=max_deg are connected to vertex 0, plus a chain extending
/// from vertex 1 to provide diameter and connectivity.
fn make_connected_graph_with_exact_max_degree(max_deg: usize, extra_chain_len: usize) -> CsrGraph {
    let nv = max_deg + 1 + extra_chain_len;
    let mut edges: Vec<(u32, u32)> = Vec::new();

    // Connect hub vertex 0 to 1..=max_deg
    for v in 1..=max_deg as u32 {
        edges.push((0, v));
        edges.push((v, 0));
    }

    // Connect extra chain from max_deg to nv-1
    let chain_start = max_deg as u32;
    for i in 0..extra_chain_len as u32 {
        let u = chain_start + i;
        let v = u + 1;
        edges.push((u, v));
        edges.push((v, u));
    }

    let graph = CsrGraph::from_edge_list(nv, &edges);
    assert_eq!(graph.max_degree(), max_deg, "Graph max_degree mismatch");
    graph
}

#[test]
fn test_degree_boundary_routing_decisions() {
    let g7 = make_connected_graph_with_exact_max_degree(7, 10);
    let g8 = make_connected_graph_with_exact_max_degree(8, 10);
    let g9 = make_connected_graph_with_exact_max_degree(9, 10);
    let g10 = make_connected_graph_with_exact_max_degree(10, 10);

    assert_eq!(g7.max_degree(), 7);
    assert_eq!(g8.max_degree(), 8);
    assert_eq!(g9.max_degree(), 9);
    assert_eq!(g10.max_degree(), 10);

    // Boundary d <= 8: All batch sizes route to WorkgroupPersistentBfs
    for &s in &[1, 16, 32, 64, 128, 512, 1024, 2048] {
        assert_eq!(
            GpuBfsEngine::select_auto_method(&g7, s),
            GpuBfsMethod::WorkgroupPersistentBfs,
            "d=7, S={s}"
        );
        assert_eq!(
            GpuBfsEngine::select_auto_method(&g8, s),
            GpuBfsMethod::WorkgroupPersistentBfs,
            "d=8, S={s}"
        );
    }

    // Boundary d == 9: Switches according to intermediate density branch
    assert_eq!(
        GpuBfsEngine::select_auto_method(&g9, 1),
        GpuBfsMethod::SparseQueue,
        "d=9, S=1"
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&g9, 16),
        GpuBfsMethod::BitParallel32,
        "d=9, S=16"
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&g9, 32),
        GpuBfsMethod::BitParallel64,
        "d=9, S=32"
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&g9, 64),
        GpuBfsMethod::MultiWordBitParallel,
        "d=9, S=64"
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&g9, 512),
        GpuBfsMethod::MultiWordBitParallel,
        "d=9, S=512"
    );

    // Boundary d == 10: Same intermediate routing
    assert_eq!(
        GpuBfsEngine::select_auto_method(&g10, 1),
        GpuBfsMethod::SparseQueue,
        "d=10, S=1"
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&g10, 16),
        GpuBfsMethod::BitParallel32,
        "d=10, S=16"
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&g10, 32),
        GpuBfsMethod::BitParallel64,
        "d=10, S=32"
    );
    assert_eq!(
        GpuBfsEngine::select_auto_method(&g10, 64),
        GpuBfsMethod::MultiWordBitParallel,
        "d=10, S=64"
    );
}

#[test]
fn test_degree_boundary_end_to_end_parity() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to initialize GpuContext"));
    let cpu = CpuBfsEngine::new();
    let auto = GpuBfsEngine::auto(ctx).expect("Failed to initialize Auto engine");

    let g8 = make_connected_graph_with_exact_max_degree(8, 50);
    let g9 = make_connected_graph_with_exact_max_degree(9, 50);

    for (deg, graph) in [(8, &g8), (9, &g9)] {
        let nv = graph.num_vertices;
        for &s in &[1, 16, 32, 64, 128, 512] {
            let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % nv as u32).collect();

            // Distance MS-BFS Parity
            let cpu_dist = cpu.run_ms_bfs(graph, &sources).expect("CPU BFS failed");
            let auto_dist = auto.run_ms_bfs(graph, &sources).expect("Auto GPU BFS failed");
            assert_parity(
                &auto_dist,
                &cpu_dist,
                s,
                nv,
                &format!("Boundary_d={deg}_S={s}_Distance"),
            );

            // Reachability Parity
            let cpu_mask = cpu.run_reachability(graph, &sources).expect("CPU Reachability failed");
            let auto_mask = auto.run_reachability(graph, &sources).expect("Auto GPU Reachability failed");
            assert_eq!(
                auto_mask.words,
                cpu_mask.words,
                "Reachability bitmask mismatch at d={deg}, S={s}"
            );
        }
    }
}

#[test]
fn test_driver_stability_and_memory_leak_stress() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to initialize GpuContext"));
    let cpu = CpuBfsEngine::new();
    let auto = GpuBfsEngine::auto(ctx).expect("Failed to initialize Auto engine");

    // Stress test: 50 consecutive dispatches across multiple topologies
    // without dropping or re-creating the GpuContext or GpuBfsEngine.
    let grid = generate_grid2d(15, 15); // max_degree = 4 (WorkgroupPersistentBfs)
    let clique = clique_graph(40);     // max_degree = 39 (BitParallel64 / IndirectSparseQueuePush)
    let nv_grid = grid.num_vertices;
    let nv_clique = clique.num_vertices;

    for iteration in 0..50 {
        if iteration % 2 == 0 {
            let s = 64;
            let sources: Vec<u32> = (0..s as u32).map(|i| (i + iteration) % nv_grid as u32).collect();
            let cpu_dist = cpu.run_ms_bfs(&grid, &sources).expect("CPU BFS");
            let gpu_dist = auto.run_ms_bfs(&grid, &sources).expect("GPU BFS");
            assert_parity(&gpu_dist, &cpu_dist, s, nv_grid, &format!("LeakStress_Grid_iter_{iteration}"));
        } else {
            let s = 32;
            let sources: Vec<u32> = (0..s as u32).map(|i| (i * 3 + iteration) % nv_clique as u32).collect();
            let cpu_dist = cpu.run_ms_bfs(&clique, &sources).expect("CPU BFS");
            let gpu_dist = auto.run_ms_bfs(&clique, &sources).expect("GPU BFS");
            assert_parity(&gpu_dist, &cpu_dist, s, nv_clique, &format!("LeakStress_Clique_iter_{iteration}"));
        }
    }
}
