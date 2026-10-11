#![allow(clippy::identity_op, clippy::erasing_op)]

//! Comprehensive Adversarial Stress & Topological Corner Case Verification Suite
//! for Milestone M9: `GpuBfsMethod::IndirectMacroBatch`.
//!
//! Evaluates strictly against `CpuBfsEngine` (Rayon) ground truth and analytical oracles:
//! 1. Chunk boundary and multi-chunk straddling ($N \in \{64, 65, 128, 129, 600\}$).
//! 2. Massive hub / star graph ($N = 2500, 1 \text{ hub} + 2499 \text{ leaves}$, leaf contention).
//! 3. Disconnected cliques ($N = 600$, 4 components) and isolated vertices ($N = 200, E = 0$).
//! 4. Arbitrary and multi-batch source counts ($S \in \{1, 2, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 256\}$).
//! 5. Heavy duplicate sources array and single-node boundary graph ($N = 1, E = 0$).
//! 6. Asymmetric directed cycle ($N = 100, 0 \to 1 \to \dots \to N-1 \to 0$).

use std::sync::Arc;
use std::time::Instant;

use webgpu_bfs_core::{
    generators::{
        disconnected_cliques, isolated_vertices, path_graph, single_vertex, star_graph, Grid2D,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuContext};

static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Helper asserting 100% bitwise parity against CPU Rayon baseline, analytical oracle,
/// source self-distance == 0, and metric triangle inequality on all graph edges.
fn verify_m9_adversarial_case<F>(
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
    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .expect("CPU BFS failed");
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

    for (s_idx, &src) in sources.iter().enumerate() {
        let base = s_idx * nv;

        // Invariant 1: Source self-distance == 0
        assert_eq!(
            gpu_dist[base + src as usize],
            0,
            "[{tag}] Source self-distance != 0 for src {} at s_idx {}",
            src,
            s_idx
        );

        // Invariant 2: 100% bitwise parity against CPU reference and analytical oracle
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

        // Invariant 3: Metric triangle inequality on all graph edges
        for u in 0..nv {
            let start = graph.row_offsets[u] as usize;
            let end = graph.row_offsets[u + 1] as usize;
            let d_u = gpu_dist[base + u];

            for edge in start..end {
                let v = graph.col_indices[edge] as usize;
                let d_v = gpu_dist[base + v];

                if d_u != u32::MAX && d_v != u32::MAX {
                    let diff = (d_u as i64 - d_v as i64).abs();
                    if is_sym {
                        assert!(
                            diff <= 1,
                            "[{tag}] Undirected triangle inequality violated! Edge ({}, {}), d(u)={}, d(v)={}",
                            u, v, d_u, d_v
                        );
                    } else {
                        // Directed edge u -> v: d(v) <= d(u) + 1
                        assert!(
                            d_v <= d_u + 1,
                            "[{tag}] Directed triangle inequality violated! Edge ({}, {}), d(u)={}, d(v)={}",
                            u, v, d_u, d_v
                        );
                    }
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
        "  -> {:<34} (S={:<4}) | CPU: {:>7.2} ms | GPU: {:>7.2} ms | Speedup: {:>5.2}x | Parity: 100% PASS",
        tag, s_count, cpu_ms, gpu_ms, speedup
    );
}

#[test]
fn test_m9_adversarial_chunk_boundaries_and_deep_chains() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [M9 Adversarial] Chunk Boundaries & High Diameter Chains ===");

    // Test exact chunk boundaries:
    // N=64 (D=63, exactly 1 chunk of 64 hops)
    // N=65 (D=64, straddles exactly into 2nd chunk)
    // N=128 (D=127, exactly 2 chunks of 64 hops)
    // N=129 (D=128, straddles exactly into 3rd chunk)
    // N=600 (D=599, spans 10 chunks)
    for &n in &[64usize, 65, 128, 129, 600] {
        let graph = path_graph(n);
        let oracle = |s: u32, v: u32| -> u32 { (s as i64 - v as i64).unsigned_abs() as u32 };

        // S=1 at boundary 0
        verify_m9_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &[0],
            &format!("Chain-N{}-S1-src0", n),
            Some(oracle),
        );

        // S=64
        let s64: Vec<u32> = (0..64).map(|i| (i * 7) % n as u32).collect();
        verify_m9_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &s64,
            &format!("Chain-N{}-S64", n),
            Some(oracle),
        );

        // S=128 (multi-batch test across chunks)
        let s128: Vec<u32> = (0..128).map(|i| (i * 11) % n as u32).collect();
        verify_m9_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &s128,
            &format!("Chain-N{}-S128", n),
            Some(oracle),
        );
    }
}

#[test]
fn test_m9_adversarial_massive_hub_star() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    const N: usize = 2500;
    let graph = star_graph(N);

    let oracle = |s: u32, v: u32| -> u32 {
        if s == v {
            0
        } else if s == 0 || v == 0 {
            1
        } else {
            2
        }
    };

    println!("\n=== [M9 Adversarial] Massive Hub Star (N=2500, 1 Hub + 2499 Leaves) ===");

    // 1. Hub source and single Leaf source
    verify_m9_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        "Star2500-S1-Hub",
        Some(oracle),
    );
    verify_m9_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[1],
        "Star2500-S1-Leaf",
        Some(oracle),
    );

    // 2. High Leaf Contention (64 leaves simultaneously pulling from hub)
    let leaf_sources_64: Vec<u32> = (1..=64).collect();
    verify_m9_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &leaf_sources_64,
        "Star2500-64LeafContention",
        Some(oracle),
    );

    // 3. Multi-batch Leaf Contention (128 leaves)
    let leaf_sources_128: Vec<u32> = (1..=128).collect();
    verify_m9_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &leaf_sources_128,
        "Star2500-128LeafContention",
        Some(oracle),
    );
}

#[test]
fn test_m9_adversarial_disconnected_and_isolated() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [M9 Adversarial] Disconnected Cliques & Isolated Vertices ===");

    // 1. Disconnected Cliques: 4 components of sizes 50, 100, 150, 300 (total N=600)
    let dis_graph = disconnected_cliques(&[50, 100, 150, 300]);
    assert_eq!(dis_graph.num_vertices, 600);

    for &s in &[1usize, 33, 64, 128] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 17) % 600).collect();
        verify_m9_adversarial_case(
            &ctx,
            &cpu,
            &dis_graph,
            &sources,
            &format!("DisCliques600-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }

    // 2. Completely isolated vertices (E = 0, N = 200)
    const N_ISO: usize = 200;
    let iso_graph = isolated_vertices(N_ISO);
    let iso_oracle = |s: u32, v: u32| -> u32 {
        if s == v {
            0
        } else {
            u32::MAX
        }
    };

    for &s in &[1usize, 32, 64, 100] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 23) % N_ISO as u32).collect();
        verify_m9_adversarial_case(
            &ctx,
            &cpu,
            &iso_graph,
            &sources,
            &format!("Isolated200-S{}", s),
            Some(iso_oracle),
        );
    }
}

#[test]
fn test_m9_adversarial_arbitrary_source_batches_and_duplicates() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    let grid = Grid2D::new(20, 20).expect("Grid20x20 generation failed");
    let graph = &grid.graph;
    let oracle = |s: u32, v: u32| -> u32 { grid.manhattan_distance(s, v) };

    println!("\n=== [M9 Adversarial] Arbitrary Source Batches & Duplicates (Grid 20x20) ===");

    // Test exact bit-parallel and batch boundaries:
    // S=1, 2, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 256
    for &s in &[1usize, 2, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 256] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % 400).collect();
        verify_m9_adversarial_case(
            &ctx,
            &cpu,
            graph,
            &sources,
            &format!("Grid400-S{}", s),
            Some(oracle),
        );
    }

    // Heavy duplicates: 64 identical sources all targeting node 42
    let dup_64 = vec![42u32; 64];
    verify_m9_adversarial_case(
        &ctx,
        &cpu,
        graph,
        &dup_64,
        "Grid400-Duplicates-64xNode42",
        Some(oracle),
    );

    // Multi-batch duplicates: 128 sources alternating [0, 42]
    let dup_128: Vec<u32> = (0..128).map(|i| if i % 2 == 0 { 0 } else { 42 }).collect();
    verify_m9_adversarial_case(
        &ctx,
        &cpu,
        graph,
        &dup_128,
        "Grid400-Duplicates-128xAlternating",
        Some(oracle),
    );
}

#[test]
fn test_m9_adversarial_single_vertex() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [M9 Adversarial] Single Vertex Boundary Graph (N=1, E=0) ===");

    let graph = single_vertex();
    let oracle = |_s: u32, _v: u32| -> u32 { 0 };

    // S=1
    verify_m9_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        "SingleVertex-S1",
        Some(oracle),
    );

    // S=64
    let s64 = vec![0u32; 64];
    verify_m9_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &s64,
        "SingleVertex-S64",
        Some(oracle),
    );

    // S=65 (multi-batch)
    let s65 = vec![0u32; 65];
    verify_m9_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &s65,
        "SingleVertex-S65",
        Some(oracle),
    );
}

#[test]
fn test_m9_adversarial_directed_asymmetric_cycle() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [M9 Adversarial] Directed Asymmetric Cycle (N=100, 0->1->...->99->0) ===");

    // Build directed cycle: edge i -> (i + 1) % N
    const N: usize = 100;
    let edges: Vec<(u32, u32)> = (0..N as u32).map(|i| (i, (i + 1) % N as u32)).collect();
    let graph = CsrGraph::from_edge_list(N, &edges);
    assert!(!graph.is_symmetric());

    let oracle = |s: u32, v: u32| -> u32 {
        (v + N as u32 - s) % (N as u32)
    };

    for &s in &[1usize, 32, 64, 100] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % N as u32).collect();
        verify_m9_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("DirectedCycle100-S{}", s),
            Some(oracle),
        );
    }
}

#[test]
fn test_m9_adversarial_custom_chunk_hops_and_empty() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();
    let pipelines = Arc::new(
        webgpu_bfs_gpu::pipeline::PipelineCache::new(ctx.device())
            .expect("Pipeline cache failed"),
    );

    println!("\n=== [M9 Adversarial] Custom Chunk Hops (H=1, H=7) & Empty Sources ===");

    let graph = path_graph(50);

    // Empty sources test
    let runner_default = webgpu_bfs_gpu::dispatch::indirect::IndirectMacroBatchBfs::new(
        ctx.clone(),
        pipelines.clone(),
        &graph,
        64,
    )
    .expect("Failed to create runner");
    let empty_res = runner_default.run(&[]).expect("Empty run failed");
    assert!(empty_res.is_empty(), "Empty sources should return empty vec");

    // chunk_hops = 1 (host inspects every hop)
    let runner_h1 = webgpu_bfs_gpu::dispatch::indirect::IndirectMacroBatchBfs::new(
        ctx.clone(),
        pipelines.clone(),
        &graph,
        1,
    )
    .expect("Failed to create runner with chunk_hops=1");
    let cpu_dist = cpu.run_ms_bfs(&graph, &[0, 25, 49]).expect("CPU run failed");
    let gpu_dist_h1 = runner_h1.run(&[0, 25, 49]).expect("GPU run failed");
    assert_eq!(cpu_dist, gpu_dist_h1, "chunk_hops=1 parity mismatch");

    // chunk_hops = 7 (odd chunk size, prime)
    let runner_h7 = webgpu_bfs_gpu::dispatch::indirect::IndirectMacroBatchBfs::new(
        ctx.clone(),
        pipelines.clone(),
        &graph,
        7,
    )
    .expect("Failed to create runner with chunk_hops=7");
    let gpu_dist_h7 = runner_h7.run(&[0, 25, 49]).expect("GPU run failed");
    assert_eq!(cpu_dist, gpu_dist_h7, "chunk_hops=7 parity mismatch");
}

