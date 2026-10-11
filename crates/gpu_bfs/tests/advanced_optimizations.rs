#![allow(clippy::identity_op, clippy::erasing_op)]

//! Comprehensive integration tests for Milestone 3 Advanced GPU BFS Optimizations:
//! - BitParallel32 (32-source bit-parallel frontier expansion)
//! - BitParallel64 (64-source bit-parallel frontier expansion)
//! - BooleanReachability (3-stage pipelined reachability and distance update)
//! - PushPullAdaptive (direction-optimizing CSR Push / CSC Pull switching)
//!
//! Asserts 100% bitwise distance equivalence across ALL 6 GPU BFS methods and against
//! `CpuBfsEngine` and the analytical Manhattan distance oracle on `Grid2D`.

use std::sync::Arc;
use std::time::Instant;
use webgpu_bfs_core::{
    generators::{
        generate_erdos_renyi, generate_grid2d, generate_random_tree, generate_rmat, GRAPH500_A,
        GRAPH500_B, GRAPH500_C, GRAPH500_D,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

const ALL_METHODS: [GpuBfsMethod; 6] = [
    GpuBfsMethod::NaiveSequential,
    GpuBfsMethod::BatchedDenseSpMM,
    GpuBfsMethod::BitParallel32,
    GpuBfsMethod::BitParallel64,
    GpuBfsMethod::BooleanReachability,
    GpuBfsMethod::PushPullAdaptive,
];

const M3_ADVANCED_METHODS: [GpuBfsMethod; 4] = [
    GpuBfsMethod::BitParallel32,
    GpuBfsMethod::BitParallel64,
    GpuBfsMethod::BooleanReachability,
    GpuBfsMethod::PushPullAdaptive,
];

/// Asserts 100% bitwise distance equivalence between GPU BFS and CPU reference BFS.
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
        // Assert source self-distance invariant
        let self_dist = gpu_dist[s_idx * graph.num_vertices + src as usize];
        assert_eq!(
            self_dist,
            0,
            "Source self-distance must be 0! Method={:?}, src={}",
            gpu_engine.method(),
            src
        );

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

/// 1. Verifies 100% bitwise distance equivalence across ALL 6 GPU BFS methods,
///    `CpuBfsEngine`, and the analytical Manhattan distance oracle on `Grid2D`.
#[test]
fn test_all_six_methods_bitwise_equivalence_on_grid2d_oracle() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let width = 10;
    let height = 10;
    let graph = generate_grid2d(width, height);
    let sources = vec![0, 9, 90, 99, 45]; // corners + center

    let mut reference_distances: Option<Vec<u32>> = None;

    for method in ALL_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();

        // Check equivalence with CPU reference
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);

        let gpu_dist = gpu_engine.run_ms_bfs(&graph, &sources).unwrap();

        // Check 100% bitwise equivalence with previously run GPU methods
        if let Some(ref ref_dist) = reference_distances {
            assert_eq!(
                &gpu_dist, ref_dist,
                "Cross-method mismatch: {:?} produced different results",
                method
            );
        } else {
            reference_distances = Some(gpu_dist.clone());
        }

        // Verify against analytical Manhattan distance oracle
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

/// 2. Tests arbitrary batch sizes: S in {1, 7, 32, 64, 65, 128}.
#[test]
fn test_arbitrary_batch_sizes_sweep() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let n = 200;
    let graph = generate_erdos_renyi(n, 0.08, 98765);

    let batch_sizes = [1, 7, 32, 64, 65, 128];

    for &batch_size in &batch_sizes {
        let sources: Vec<u32> = (0..batch_size as u32).map(|i| (i * 3) % n as u32).collect();

        for method in M3_ADVANCED_METHODS {
            let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
            assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);
        }
    }
}

/// 3. Tests disconnected components and unreachable sentinel invariant (u32::MAX).
#[test]
fn test_disconnected_components_sentinel() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    // Graph with 3 disjoint triangles: (0,1,2), (3,4,5), (6,7,8)
    let edges = vec![
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 0),
        (0, 2),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 4),
        (5, 3),
        (3, 5),
        (6, 7),
        (7, 6),
        (7, 8),
        (8, 7),
        (8, 6),
        (6, 8),
    ];
    let graph = CsrGraph::from_edge_list(9, &edges);
    let sources = vec![0, 3, 6];

    for method in ALL_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);

        let dist = gpu_engine.run_ms_bfs(&graph, &sources).unwrap();

        // From source 0 (component 0), components 1 and 2 must be u32::MAX
        for v in 3..9 {
            assert_eq!(
                dist[0 * 9 + v],
                u32::MAX,
                "Vertex {} should be unreachable from source 0 for method {:?}",
                v,
                method
            );
        }
        // From source 3 (component 1), component 0 and 2 must be u32::MAX
        for v in [0, 1, 2, 6, 7, 8] {
            assert_eq!(
                dist[1 * 9 + v],
                u32::MAX,
                "Vertex {} should be unreachable from source 3 for method {:?}",
                v,
                method
            );
        }
    }
}

/// 4. Tests duplicate sources in input slice (idempotence).
#[test]
fn test_duplicate_sources_idempotence() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let graph = generate_random_tree(100, Some(777)).unwrap();
    let sources = vec![15, 15, 42, 42, 15];

    for method in M3_ADVANCED_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);

        let dist = gpu_engine.run_ms_bfs(&graph, &sources).unwrap();
        // Row 0 and Row 1 must be identical
        let row0 = &dist[0..100];
        let row1 = &dist[100..200];
        let row4 = &dist[400..500];
        assert_eq!(row0, row1, "Duplicate source rows 0 and 1 must match");
        assert_eq!(row0, row4, "Duplicate source rows 0 and 4 must match");
    }
}

/// 5. Tests Graph500 R-MAT scale-free power law graph (triggers Pull mode in PushPullAdaptive).
#[test]
fn test_scale_free_rmat_graph_push_pull_trigger() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let graph = generate_rmat(
        8,
        16,
        GRAPH500_A,
        GRAPH500_B,
        GRAPH500_C,
        GRAPH500_D,
        Some(42),
        false,
    )
    .unwrap();
    let sources = vec![0, 1, 2, 3, 50, 100];

    for method in M3_ADVANCED_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);
    }
}

/// 6. Tests high-diameter path graph (linear chain of 100 nodes).
#[test]
fn test_high_diameter_linear_chain() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let n = 100;
    let mut edges = Vec::new();
    for i in 0..n - 1 {
        edges.push((i as u32, (i + 1) as u32));
        edges.push(((i + 1) as u32, i as u32));
    }
    let graph = CsrGraph::from_edge_list(n, &edges);
    let sources = vec![0, (n - 1) as u32];

    for method in M3_ADVANCED_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);

        let dist = gpu_engine.run_ms_bfs(&graph, &sources).unwrap();
        // From 0 to i, distance must be exactly i
        for i in 0..n {
            assert_eq!(
                dist[0 * n + i],
                i as u32,
                "Linear path distance mismatch for method {:?}",
                method
            );
        }
    }
}

/// 7. Tests star graph with high-degree hub contention (1 hub + 200 leaves).
#[test]
fn test_star_graph_high_degree_hub_contention() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let n = 201;
    let mut edges = Vec::new();
    for leaf in 1..n {
        edges.push((0, leaf as u32));
        edges.push((leaf as u32, 0));
    }
    let graph = CsrGraph::from_edge_list(n, &edges);
    let sources = vec![0, 1, 50, 100, 200];

    for method in ALL_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);
    }
}

/// 8. Tests boundary cases: single vertex, zero edges, empty sources, out of bounds error.
#[test]
fn test_boundary_zero_edges_and_single_vertex() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));

    // Single vertex, 0 edges
    let g1 = CsrGraph::from_edge_list(1, &[]);
    for method in ALL_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let dist = gpu_engine.run_ms_bfs(&g1, &[0]).unwrap();
        assert_eq!(dist, vec![0]);
    }

    // 10 vertices, 0 edges
    let g10 = CsrGraph::from_edge_list(10, &[]);
    for method in ALL_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let dist = gpu_engine.run_ms_bfs(&g10, &[3]).unwrap();
        for (v, &d) in dist.iter().enumerate() {
            if v == 3 {
                assert_eq!(d, 0);
            } else {
                assert_eq!(d, u32::MAX);
            }
        }
    }

    // Empty sources
    for method in ALL_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let dist = gpu_engine.run_ms_bfs(&g10, &[]).unwrap();
        assert!(dist.is_empty());
    }

    // Out of bounds source
    for method in ALL_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        assert!(gpu_engine.run_ms_bfs(&g10, &[100]).is_err());
    }
}

/// 9. Performance comparison demonstrating speedup of Bit-Parallel and Boolean Reachability vs Naive.
#[test]
fn test_performance_speedup_demonstration() {
    let ctx = Arc::new(GpuContext::init_blocking().expect("GpuContext init failed"));
    let cpu_engine = CpuBfsEngine::new();

    let n = 300;
    let graph = generate_erdos_renyi(n, 0.06, 54321);
    let sources: Vec<u32> = (0..64).map(|i| (i * 4) % n as u32).collect();

    println!(
        "\n=== Performance Benchmark on Erdős-Rényi (|V|={}, S=64) ===",
        n
    );

    let mut timings = Vec::new();

    for &method in &ALL_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();

        // Warm up and verify correctness
        assert_gpu_cpu_equivalence(&gpu_engine, &cpu_engine, &graph, &sources);

        // Timed run
        let start = Instant::now();
        let _ = gpu_engine.run_ms_bfs(&graph, &sources).unwrap();
        let duration = start.elapsed();

        println!("Method {:<20}: {:?}", format!("{:?}", method), duration);
        timings.push((method, duration));
    }

    let naive_duration = timings[0].1;
    for (method, duration) in &timings[1..] {
        let speedup = naive_duration.as_secs_f64() / duration.as_secs_f64();
        println!(
            "Speedup of {:<20} vs NaiveSequential: {:.2}x",
            format!("{:?}", method),
            speedup
        );
    }
}
