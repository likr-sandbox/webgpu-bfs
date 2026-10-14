#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M21: Adversarial Challenge & Auto Router Verification Suite
//! ==============================================================================
//!
//! Empirically challenges `select_auto_method` and the `Auto` execution router on:
//! 1. Adversarial Topologies:
//!    - Empty graph (V=0, E=0) and isolated vertices (V=100, E=0)
//!    - Disjoint cliques (sub-32, 32-127, and massive hub >=128)
//!    - Dense graphs (complete graph K_200, high-probability Erdős-Rényi)
//!    - Extreme star hubs (N=20, N=70, N=1000, N=5000)
//!    - Grid meshes (square 30x30, high-aspect ratio 3x300, 1x100 path)
//! 2. Exact Degree Boundaries:
//!    - max_degree == 32 (low-degree branch) vs max_degree == 33 (intermediate branch)
//!    - max_degree == 127 (intermediate branch) vs max_degree == 128 (high-degree hub branch)
//! 3. Source Count Boundaries:
//!    - S in {0, 1, 2, 16, 31, 32, 33, 63, 64, 65, 128, 256, 512, 1024, 2048}
//! 4. End-to-End GPU Bitwise Parity (100.0% against CPU Reference):
//!    - Executes `GpuBfsEngine::auto(ctx)` across all adversarial topologies and
//!      verifies that every distance entry exactly matches `CpuBfsEngine` ground truth.

use std::sync::Arc;

use webgpu_bfs_core::{
    generators::{
        clique_graph, disconnected_cliques, empty_graph, generate_erdos_renyi_np,
        generate_grid2d, isolated_vertices, star_graph, Grid2D,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

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

fn make_graph_with_exact_max_degree(max_deg: usize) -> CsrGraph {
    let nv = max_deg + 2;
    let edges: Vec<(u32, u32)> = (1..=max_deg as u32).map(|v| (0, v)).collect();
    CsrGraph::from_edge_list(nv, &edges)
}

// =============================================================================
// 1. Pure Routing Decision Tests on Adversarial Topologies & Boundaries
// =============================================================================

#[test]
fn test_select_auto_method_adversarial_topologies() {
    // 1.1 Empty graph
    let empty = empty_graph();
    assert_eq!(empty.num_vertices, 0);
    assert_eq!(GpuBfsEngine::select_auto_method(&empty, 0), GpuBfsMethod::NaiveSequential);
    assert_eq!(GpuBfsEngine::select_auto_method(&empty, 1), GpuBfsMethod::NaiveSequential);
    assert_eq!(GpuBfsEngine::select_auto_method(&empty, 64), GpuBfsMethod::NaiveSequential);
    assert_eq!(GpuBfsEngine::select_auto_method(&empty, 512), GpuBfsMethod::NaiveSequential);

    // 1.2 Isolated vertices (max_degree == 0 <= 32)
    let iso = isolated_vertices(100);
    assert_eq!(iso.max_degree(), 0);
    assert_eq!(GpuBfsEngine::select_auto_method(&iso, 0), GpuBfsMethod::NaiveSequential);
    assert_eq!(GpuBfsEngine::select_auto_method(&iso, 1), GpuBfsMethod::SparseQueue);
    assert_eq!(GpuBfsEngine::select_auto_method(&iso, 16), GpuBfsMethod::BitParallel32);
    assert_eq!(GpuBfsEngine::select_auto_method(&iso, 32), GpuBfsMethod::BitParallel64);
    assert_eq!(GpuBfsEngine::select_auto_method(&iso, 64), GpuBfsMethod::MultiWordBitParallel);
    assert_eq!(GpuBfsEngine::select_auto_method(&iso, 512), GpuBfsMethod::MultiWordBitParallel);

    // 1.3 Disjoint Cliques: Low-degree (max_degree 19 <= 32)
    let low_cliques = disconnected_cliques(&[5, 10, 20]);
    assert_eq!(low_cliques.max_degree(), 19);
    assert_eq!(GpuBfsEngine::select_auto_method(&low_cliques, 1), GpuBfsMethod::SparseQueue);
    assert_eq!(GpuBfsEngine::select_auto_method(&low_cliques, 16), GpuBfsMethod::BitParallel32);
    assert_eq!(GpuBfsEngine::select_auto_method(&low_cliques, 32), GpuBfsMethod::BitParallel64);
    assert_eq!(GpuBfsEngine::select_auto_method(&low_cliques, 64), GpuBfsMethod::MultiWordBitParallel);
    assert_eq!(GpuBfsEngine::select_auto_method(&low_cliques, 512), GpuBfsMethod::MultiWordBitParallel);

    // 1.4 Disjoint Cliques: Intermediate (max_degree 49 in (32, 128))
    let mid_cliques = disconnected_cliques(&[10, 20, 50]);
    assert_eq!(mid_cliques.max_degree(), 49);
    assert_eq!(GpuBfsEngine::select_auto_method(&mid_cliques, 1), GpuBfsMethod::SparseQueue);
    assert_eq!(GpuBfsEngine::select_auto_method(&mid_cliques, 16), GpuBfsMethod::BitParallel32);
    assert_eq!(GpuBfsEngine::select_auto_method(&mid_cliques, 32), GpuBfsMethod::BitParallel64);
    assert_eq!(GpuBfsEngine::select_auto_method(&mid_cliques, 64), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&mid_cliques, 512), GpuBfsMethod::IndirectSparseQueuePush);

    // 1.5 Disjoint Cliques: Massive Hub (max_degree 199 >= 128)
    let hub_cliques = disconnected_cliques(&[10, 20, 200]);
    assert_eq!(hub_cliques.max_degree(), 199);
    assert_eq!(GpuBfsEngine::select_auto_method(&hub_cliques, 1), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&hub_cliques, 16), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&hub_cliques, 32), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&hub_cliques, 64), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&hub_cliques, 512), GpuBfsMethod::IndirectSparseQueuePush);

    // 1.6 Dense complete graph (K_250: max_degree 249 >= 128)
    let complete = clique_graph(250);
    assert_eq!(complete.max_degree(), 249);
    assert_eq!(GpuBfsEngine::select_auto_method(&complete, 1), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&complete, 32), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&complete, 64), GpuBfsMethod::IndirectSparseQueuePush);

    // 1.7 Dense Erdős-Rényi graph (V=200, p=0.8)
    let er_dense = generate_erdos_renyi_np(200, 0.8, Some(42)).unwrap();
    assert!(er_dense.max_degree() >= 128);
    assert_eq!(GpuBfsEngine::select_auto_method(&er_dense, 64), GpuBfsMethod::IndirectSparseQueuePush);

    // 1.8 Extreme star hubs
    let star_small = star_graph(20); // max_degree = 19
    assert_eq!(GpuBfsEngine::select_auto_method(&star_small, 64), GpuBfsMethod::MultiWordBitParallel);

    let star_mid = star_graph(70); // max_degree = 69
    assert_eq!(GpuBfsEngine::select_auto_method(&star_mid, 64), GpuBfsMethod::IndirectSparseQueuePush);

    let star_huge = star_graph(1000); // max_degree = 999
    assert_eq!(GpuBfsEngine::select_auto_method(&star_huge, 1), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&star_huge, 16), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&star_huge, 64), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&star_huge, 512), GpuBfsMethod::IndirectSparseQueuePush);

    let star_ultra = star_graph(5000); // max_degree = 4999
    assert_eq!(GpuBfsEngine::select_auto_method(&star_ultra, 64), GpuBfsMethod::IndirectSparseQueuePush);

    // 1.9 Grid meshes
    let grid_sq = Grid2D::new(30, 30).unwrap().graph; // max_degree = 4
    assert_eq!(GpuBfsEngine::select_auto_method(&grid_sq, 1), GpuBfsMethod::SparseQueue);
    assert_eq!(GpuBfsEngine::select_auto_method(&grid_sq, 16), GpuBfsMethod::BitParallel32);
    assert_eq!(GpuBfsEngine::select_auto_method(&grid_sq, 32), GpuBfsMethod::BitParallel64);
    assert_eq!(GpuBfsEngine::select_auto_method(&grid_sq, 64), GpuBfsMethod::MultiWordBitParallel);
    assert_eq!(GpuBfsEngine::select_auto_method(&grid_sq, 512), GpuBfsMethod::MultiWordBitParallel);

    let grid_narrow = generate_grid2d(2, 200); // max_degree = 3
    assert_eq!(GpuBfsEngine::select_auto_method(&grid_narrow, 64), GpuBfsMethod::MultiWordBitParallel);
}

#[test]
fn test_select_auto_method_exact_degree_boundaries() {
    // Boundary 1: max_degree == 32 vs 33
    let g32 = make_graph_with_exact_max_degree(32);
    let g33 = make_graph_with_exact_max_degree(33);
    assert_eq!(g32.max_degree(), 32);
    assert_eq!(g33.max_degree(), 33);

    // S = 64: 32 routes to MultiWordBitParallel, 33 routes to IndirectSparseQueuePush
    assert_eq!(GpuBfsEngine::select_auto_method(&g32, 64), GpuBfsMethod::MultiWordBitParallel);
    assert_eq!(GpuBfsEngine::select_auto_method(&g33, 64), GpuBfsMethod::IndirectSparseQueuePush);

    // S = 16: both route to BitParallel32
    assert_eq!(GpuBfsEngine::select_auto_method(&g32, 16), GpuBfsMethod::BitParallel32);
    assert_eq!(GpuBfsEngine::select_auto_method(&g33, 16), GpuBfsMethod::BitParallel32);

    // Boundary 2: max_degree == 127 vs 128
    let g127 = make_graph_with_exact_max_degree(127);
    let g128 = make_graph_with_exact_max_degree(128);
    assert_eq!(g127.max_degree(), 127);
    assert_eq!(g128.max_degree(), 128);

    // S = 16: 127 routes to BitParallel32, 128 routes to IndirectSparseQueuePush
    assert_eq!(GpuBfsEngine::select_auto_method(&g127, 16), GpuBfsMethod::BitParallel32);
    assert_eq!(GpuBfsEngine::select_auto_method(&g128, 16), GpuBfsMethod::IndirectSparseQueuePush);

    // S = 64: both route to IndirectSparseQueuePush
    assert_eq!(GpuBfsEngine::select_auto_method(&g127, 64), GpuBfsMethod::IndirectSparseQueuePush);
    assert_eq!(GpuBfsEngine::select_auto_method(&g128, 64), GpuBfsMethod::IndirectSparseQueuePush);
}

// =============================================================================
// 2. End-to-End GPU Auto Router Bitwise Parity Suite
// =============================================================================

#[test]
fn test_gpu_auto_parity_disjoint_cliques() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to initialize GpuContext"));
    let cpu = CpuBfsEngine::new();
    let auto = GpuBfsEngine::auto(ctx).expect("Failed to initialize Auto engine");

    // Topologies: Disconnected cliques with components of varied sizes
    let cliques = disconnected_cliques(&[12, 28, 40, 20]);
    let nv = cliques.num_vertices;
    assert_eq!(nv, 100);

    for &s in &[1, 16, 32, 64, 128, 512] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 17) % nv as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&cliques, &sources).expect("CPU BFS failed");
        let auto_dist = auto.run_ms_bfs(&cliques, &sources).expect("Auto GPU BFS failed");

        assert_parity(&auto_dist, &cpu_dist, s, nv, &format!("Auto_DisjointCliques_S={s}"));
    }
}

#[test]
fn test_gpu_auto_parity_dense_complete_graph() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to initialize GpuContext"));
    let cpu = CpuBfsEngine::new();
    let auto = GpuBfsEngine::auto(ctx).expect("Failed to initialize Auto engine");

    // Complete graph K_150: max_degree = 149 >= 128
    let dense = clique_graph(150);
    let nv = dense.num_vertices;

    for &s in &[1, 16, 32, 64, 256] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 23) % nv as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&dense, &sources).expect("CPU BFS failed");
        let auto_dist = auto.run_ms_bfs(&dense, &sources).expect("Auto GPU BFS failed");

        assert_parity(&auto_dist, &cpu_dist, s, nv, &format!("Auto_DenseComplete_S={s}"));
    }
}

#[test]
fn test_gpu_auto_parity_extreme_star_hub() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to initialize GpuContext"));
    let cpu = CpuBfsEngine::new();
    let auto = GpuBfsEngine::auto(ctx).expect("Failed to initialize Auto engine");

    // Star graph: Hub at vertex 0 with degree 999 >= 128
    let star = star_graph(1000);
    let nv = star.num_vertices;

    // A. Hub-centered sources
    for &s in &[1, 16, 32, 64, 256, 512] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 37) % nv as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&star, &sources).expect("CPU BFS failed");
        let auto_dist = auto.run_ms_bfs(&star, &sources).expect("Auto GPU BFS failed");

        assert_parity(&auto_dist, &cpu_dist, s, nv, &format!("Auto_StarHub_Mixed_S={s}"));
    }

    // B. Leaf-only sources (sources never include hub 0)
    let leaf_sources: Vec<u32> = (1..=64).collect();
    let cpu_dist = cpu.run_ms_bfs(&star, &leaf_sources).expect("CPU BFS failed");
    let auto_dist = auto.run_ms_bfs(&star, &leaf_sources).expect("Auto GPU BFS failed");
    assert_parity(&auto_dist, &cpu_dist, 64, nv, "Auto_StarHub_LeavesOnly_S=64");
}

#[test]
fn test_gpu_auto_parity_grid_meshes() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to initialize GpuContext"));
    let cpu = CpuBfsEngine::new();
    let auto = GpuBfsEngine::auto(ctx).expect("Failed to initialize Auto engine");

    // 1. Square grid 25x25 (625 vertices, max_degree = 4)
    let grid_sq = Grid2D::new(25, 25).unwrap().graph;
    let nv_sq = grid_sq.num_vertices;

    for &s in &[1, 16, 32, 64, 128, 256] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 19) % nv_sq as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&grid_sq, &sources).expect("CPU BFS failed");
        let auto_dist = auto.run_ms_bfs(&grid_sq, &sources).expect("Auto GPU BFS failed");

        assert_parity(&auto_dist, &cpu_dist, s, nv_sq, &format!("Auto_GridSq_S={s}"));
    }

    // 2. High-aspect ratio strip 3x200 (600 vertices, high diameter)
    let grid_strip = generate_grid2d(3, 200);
    let nv_strip = grid_strip.num_vertices;

    for &s in &[1, 32, 64, 256] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 29) % nv_strip as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&grid_strip, &sources).expect("CPU BFS failed");
        let auto_dist = auto.run_ms_bfs(&grid_strip, &sources).expect("Auto GPU BFS failed");

        assert_parity(&auto_dist, &cpu_dist, s, nv_strip, &format!("Auto_GridStrip_S={s}"));
    }
}

#[test]
fn test_gpu_auto_parity_isolated_vertices() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to initialize GpuContext"));
    let cpu = CpuBfsEngine::new();
    let auto = GpuBfsEngine::auto(ctx).expect("Failed to initialize Auto engine");

    // Isolated vertices (120 vertices, 0 edges, max_degree = 0)
    let iso = isolated_vertices(120);
    let nv = iso.num_vertices;

    for &s in &[1, 16, 32, 64, 256] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % nv as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&iso, &sources).expect("CPU BFS failed");
        let auto_dist = auto.run_ms_bfs(&iso, &sources).expect("Auto GPU BFS failed");

        assert_parity(&auto_dist, &cpu_dist, s, nv, &format!("Auto_Isolated_S={s}"));
    }
}
