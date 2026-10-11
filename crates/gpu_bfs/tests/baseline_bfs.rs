#![allow(clippy::identity_op, clippy::erasing_op)]

//! Integration tests verifying GPU BFS baselines (NaiveSequential and BatchedDenseSpMM)
//! against CPU reference BFS and analytical Manhattan distance ground-truth oracle.

use std::sync::Arc;
use webgpu_bfs_core::{
    generators::{generate_erdos_renyi, generate_grid2d, generate_random_tree},
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

/// Asserts 100% distance equivalence between GPU BFS and CPU reference BFS.
fn assert_gpu_cpu_equivalence(
    gpu_engine: &GpuBfsEngine,
    cpu_engine: &CpuBfsEngine,
    graph: &CsrGraph,
    sources: &[u32],
) {
    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .expect("CPU MS-BFS failed");
    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .expect("GPU MS-BFS failed");

    assert_eq!(
        gpu_dist.len(),
        cpu_dist.len(),
        "Output length mismatch for |V|={}, S={}",
        graph.num_vertices,
        sources.len()
    );

    for (s_idx, &src) in sources.iter().enumerate() {
        for v in 0..graph.num_vertices {
            let idx = s_idx * graph.num_vertices + v;
            let c = cpu_dist[idx];
            let g = gpu_dist[idx];
            if c != g {
                panic!(
                    "Distance mismatch! Method={:?}, Source idx {} (id {}), Vertex {}: CPU={}, GPU={}",
                    gpu_engine.method(),
                    s_idx,
                    src,
                    v,
                    c,
                    g
                );
            }
        }
    }
}

#[test]
fn test_gpu_grid2d_against_manhattan_oracle() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let width = 10;
    let height = 10;
    let graph = generate_grid2d(width, height);
    let sources = vec![0, 9, 90, 99, 45]; // corners + center

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);

        // Also explicitly check against analytical Manhattan distances
        let gpu_dist = gpu_engine.run_ms_bfs(&graph, &sources).unwrap();
        for (s_idx, &src) in sources.iter().enumerate() {
            let sx = (src % width as u32) as i32;
            let sy = (src / width as u32) as i32;

            for v in 0..graph.num_vertices {
                let vx = (v as u32 % width as u32) as i32;
                let vy = (v as u32 / width as u32) as i32;

                let manhattan = ((sx - vx).abs() + (sy - vy).abs()) as u32;
                let computed = gpu_dist[s_idx * graph.num_vertices + v];
                assert_eq!(
                    computed, manhattan,
                    "Manhattan oracle mismatch at method={:?}, src={}, v={}",
                    method, src, v
                );
            }
        }
    }
}

#[test]
fn test_gpu_random_tree_connectivity() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let n = 200;
    let graph = generate_random_tree(n, Some(42)).expect("Tree generation failed");
    let sources = vec![0, 10, 50, 100, 150];

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);
    }
}

#[test]
fn test_gpu_erdos_renyi_batch_sizes() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let n = 300;
    let graph = generate_erdos_renyi(n, 0.05, 12345);

    // Test powers of two and non-powers of two: S in {1, 3, 8, 32}
    for &s_count in &[1, 3, 8, 32] {
        let sources: Vec<u32> = (0..s_count).map(|s| (s * 7) % n as u32).collect();
        for method in [
            GpuBfsMethod::NaiveSequential,
            GpuBfsMethod::BatchedDenseSpMM,
        ] {
            let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
            assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);
        }
    }
}

#[test]
fn test_gpu_disconnected_graph_and_unreachable_sentinel() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    // Two isolated components: {0, 1, 2} and {3, 4}
    let row_offsets = vec![0, 2, 4, 6, 7, 8];
    let col_indices = vec![1, 2, 0, 2, 0, 1, 4, 3];
    let graph = CsrGraph {
        num_vertices: 5,
        num_edges: 8,
        row_offsets,
        col_indices,
    };
    let sources = vec![0, 3];

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);

        let gpu_dist = gpu_engine.run_ms_bfs(&graph, &sources).unwrap();
        // Component 0 -> {3, 4} must be u32::MAX
        assert_eq!(gpu_dist[0 * 5 + 3], u32::MAX);
        assert_eq!(gpu_dist[0 * 5 + 4], u32::MAX);
        // Component 1 -> {0, 1, 2} must be u32::MAX
        assert_eq!(gpu_dist[1 * 5 + 0], u32::MAX);
        assert_eq!(gpu_dist[1 * 5 + 1], u32::MAX);
        assert_eq!(gpu_dist[1 * 5 + 2], u32::MAX);
    }
}

#[test]
fn test_gpu_duplicate_sources() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let graph = generate_grid2d(6, 6);
    let sources = vec![0, 0, 15, 15];

    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);

        let dist = gpu_engine.run_ms_bfs(&graph, &sources).unwrap();
        let nv = graph.num_vertices;
        assert_eq!(&dist[0..nv], &dist[nv..2 * nv]);
        assert_eq!(&dist[2 * nv..3 * nv], &dist[3 * nv..4 * nv]);
    }
}

#[test]
fn test_gpu_boundary_corner_cases() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));

    // 1. Single vertex with zero edges
    let single_graph = CsrGraph {
        num_vertices: 1,
        num_edges: 0,
        row_offsets: vec![0, 0],
        col_indices: vec![],
    };
    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let dist = gpu.run_ms_bfs(&single_graph, &[0]).unwrap();
        assert_eq!(dist, vec![0]);
    }

    // 2. Empty sources
    let graph = generate_grid2d(5, 5);
    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let dist = gpu.run_ms_bfs(&graph, &[]).unwrap();
        assert_eq!(dist, Vec::<u32>::new());
    }

    // 3. Out of bounds error
    for method in [
        GpuBfsMethod::NaiveSequential,
        GpuBfsMethod::BatchedDenseSpMM,
    ] {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert!(gpu.run_ms_bfs(&graph, &[1000]).is_err());
    }
}
