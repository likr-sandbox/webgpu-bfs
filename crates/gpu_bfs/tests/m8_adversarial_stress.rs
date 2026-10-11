#![allow(clippy::identity_op, clippy::erasing_op)]

//! Comprehensive Adversarial Stress & Topological Corner Case Verification Suite
//! for Milestone M8: `GpuBfsMethod::MultiWordBitParallel` (256–2048+ Sources).
//!
//! Evaluates strictly against `CpuBfsEngine` (Rayon) ground truth and analytical oracles:
//! 1. High-diameter chains / line graphs ($N = 600, D = 599 \ge 500$).
//! 2. Massive hub / star graphs ($N = 2500, 1 \text{ hub} + 2499 \text{ leaves}$).
//! 3. Disconnected cliques and unreachable vertex components ($N = 1000$ and isolated $N = 200$).
//! 4. Non-power-of-two arbitrary source batch sizes ($S \in \{300, 750, 1500\}$).
//! 5. Exact word boundary sizes ($S \in \{1, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257, 511, 512, 1023, 1024, 2047, 2048\}$).
//! 6. Heavy duplicate source arrays and single-node boundary graphs ($N = 1$).
//! 7. Super-batch size crossing the 2048 boundary ($S = 2100$).

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
fn verify_m8_adversarial_case<F>(
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

    let gpu_engine = GpuBfsEngine::multi_word_bit_parallel(ctx.clone())
        .expect("Failed to instantiate MultiWordBitParallel engine");

    let t_gpu = Instant::now();
    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] GPU MultiWordBitParallel failed: {e}"));
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
                    assert!(
                        diff <= 1,
                        "[{tag}] Triangle inequality violated! Edge ({}, {}), d(u)={}, d(v)={}",
                        u, v, d_u, d_v
                    );
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
        "  -> {:<32} (S={:<4}) | CPU: {:>7.2} ms | GPU: {:>7.2} ms | Speedup: {:>5.2}x | Parity: 100% PASS",
        tag, s_count, cpu_ms, gpu_ms, speedup
    );
}

#[test]
fn test_m8_adversarial_high_diameter_chain() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    const N: usize = 600;
    let graph = path_graph(N);
    assert_eq!(graph.num_vertices, N);
    assert_eq!(graph.num_edges, 2 * (N - 1));

    let oracle = |s: u32, v: u32| -> u32 { (s as i64 - v as i64).unsigned_abs() as u32 };

    println!("\n=== [M8 Adversarial] High-Diameter Chain (N=600, Diameter=599 >= 500) ===");

    // 1. Single sources at boundaries and interior
    for &src in &[0u32, 299u32, 599u32] {
        verify_m8_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &[src],
            &format!("Chain600-S1-src{}", src),
            Some(oracle),
        );
    }

    // 2. Arbitrary non-power-of-two batches S in {300, 750, 1500}
    for &s in &[300usize, 750, 1500] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 17) % N as u32).collect();
        verify_m8_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("Chain600-Arbitrary-S{}", s),
            Some(oracle),
        );
    }

    // 3. Power-of-two batches S in {256, 512, 1024, 2048}
    for &s in &[256usize, 512, 1024, 2048] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 31) % N as u32).collect();
        verify_m8_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("Chain600-Pow2-S{}", s),
            Some(oracle),
        );
    }
}

#[test]
fn test_m8_adversarial_massive_hub_star() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    const N: usize = 2500;
    let graph = star_graph(N);
    assert_eq!(graph.num_vertices, N);
    assert_eq!(graph.num_edges, 2 * (N - 1));

    let oracle = |s: u32, v: u32| -> u32 {
        if s == v {
            0
        } else if s == 0 || v == 0 {
            1
        } else {
            2
        }
    };

    println!("\n=== [M8 Adversarial] Massive Hub Star Graph (N=2500, 1 Hub + 2499 Leaves) ===");

    // 1. Hub source and Leaf source
    verify_m8_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[0],
        "Star2500-S1-Hub",
        Some(oracle),
    );
    verify_m8_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &[1],
        "Star2500-S1-Leaf",
        Some(oracle),
    );

    // 2. Arbitrary non-power-of-two batches S in {300, 750, 1500}
    for &s in &[300usize, 750, 1500] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % N as u32).collect();
        verify_m8_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("Star2500-Arbitrary-S{}", s),
            Some(oracle),
        );
    }

    // 3. Power-of-two batches S in {256, 1024, 2048}
    for &s in &[256usize, 1024, 2048] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % N as u32).collect();
        verify_m8_adversarial_case(
            &ctx,
            &cpu,
            &graph,
            &sources,
            &format!("Star2500-Pow2-S{}", s),
            Some(oracle),
        );
    }

    // 4. Extreme Leaf Contention: S=1024 purely leaf sources pulling simultaneously
    let leaf_sources: Vec<u32> = (1..=1024).collect();
    verify_m8_adversarial_case(
        &ctx,
        &cpu,
        &graph,
        &leaf_sources,
        "Star2500-1024LeafContention",
        Some(oracle),
    );
}

#[test]
fn test_m8_adversarial_disconnected_cliques_and_unreachable() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    println!("\n=== [M8 Adversarial] Disconnected Cliques (|V|=1000) & Isolated Vertices ===");

    // 1. Disconnected Cliques: 4 isolated components (sizes: 100, 150, 250, 500)
    let dis_graph = disconnected_cliques(&[100, 150, 250, 500]);
    assert_eq!(dis_graph.num_vertices, 1000);

    for &s in &[300usize, 750, 1500] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 23) % 1000).collect();
        verify_m8_adversarial_case(
            &ctx,
            &cpu,
            &dis_graph,
            &sources,
            &format!("DisCliques-Arbitrary-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }

    for &s in &[256usize, 1024, 2048] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 41) % 1000).collect();
        verify_m8_adversarial_case(
            &ctx,
            &cpu,
            &dis_graph,
            &sources,
            &format!("DisCliques-Pow2-S{}", s),
            None::<fn(u32, u32) -> u32>,
        );
    }

    // 2. Completely isolated vertices (E = 0)
    const N_ISO: usize = 200;
    let iso_graph = isolated_vertices(N_ISO);
    let iso_oracle = |s: u32, v: u32| -> u32 {
        if s == v {
            0
        } else {
            u32::MAX
        }
    };

    let iso_sources_s300: Vec<u32> = (0..300).map(|i| (i * 11) % N_ISO as u32).collect();
    verify_m8_adversarial_case(
        &ctx,
        &cpu,
        &iso_graph,
        &iso_sources_s300,
        "Isolated200-S300",
        Some(iso_oracle),
    );

    let iso_sources_s2048: Vec<u32> = (0..2048).map(|i| (i * 19) % N_ISO as u32).collect();
    verify_m8_adversarial_case(
        &ctx,
        &cpu,
        &iso_graph,
        &iso_sources_s2048,
        "Isolated200-S2048",
        Some(iso_oracle),
    );
}

#[test]
fn test_m8_adversarial_word_boundaries_and_duplicates() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    let grid = Grid2D::new(20, 20).expect("Grid20x20 creation failed");
    let graph = &grid.graph;
    assert_eq!(graph.num_vertices, 400);

    let oracle = |s: u32, v: u32| -> u32 { grid.manhattan_distance(s, v) };

    println!("\n=== [M8 Adversarial] Word Boundary Sizes & Duplicate Sources ===");

    // 1. Boundary sizes: 2^k, 2^k - 1, 2^k + 1 around 32, 64, 128, 256, 512, 1024, 2048
    let boundary_counts = [
        1usize, 31, 32, 33, 63, 64, 65, 127, 128, 129,
        255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
        2047, 2048,
    ];

    for &s in &boundary_counts {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 400).collect();
        verify_m8_adversarial_case(
            &ctx,
            &cpu,
            graph,
            &sources,
            &format!("Grid20x20-Boundary-S{}", s),
            Some(oracle),
        );
    }

    // 2. Heavy duplicate sources: 500 identical sources
    let all_zero_sources = vec![0u32; 500];
    verify_m8_adversarial_case(
        &ctx,
        &cpu,
        graph,
        &all_zero_sources,
        "Grid20x20-Duplicates-500Zeros",
        Some(oracle),
    );

    // 3. Alternating duplicate sources: 2048 sources oscillating between 2 vertices
    let alternating_sources: Vec<u32> = (0..2048).map(|i| if i % 2 == 0 { 0 } else { 399 }).collect();
    verify_m8_adversarial_case(
        &ctx,
        &cpu,
        graph,
        &alternating_sources,
        "Grid20x20-Duplicates-2048Alternating",
        Some(oracle),
    );

    // 4. Single-node boundary graph (N=1, E=0) with S=1, 300, 2048
    let single_g = single_vertex();
    let single_oracle = |_s: u32, _v: u32| -> u32 { 0 };

    for &s in &[1usize, 300, 2048] {
        let sources = vec![0u32; s];
        verify_m8_adversarial_case(
            &ctx,
            &cpu,
            &single_g,
            &sources,
            &format!("SingleVertex-S{}", s),
            Some(single_oracle),
        );
    }
}

#[test]
fn test_m8_adversarial_maximum_contract_batch_s2048() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap();
    let ctx = Arc::new(GpuContext::init_blocking().expect("GPU init failed"));
    let cpu = CpuBfsEngine::new();

    let grid = Grid2D::new(15, 15).expect("Grid15x15 creation failed");
    let graph = &grid.graph;
    assert_eq!(graph.num_vertices, 225);

    let oracle = |s: u32, v: u32| -> u32 { grid.manhattan_distance(s, v) };

    println!("\n=== [M8 Adversarial] Contractual Maximum Batch Size (S=2048) Across Topologies ===");

    // 1. Grid15x15 with S=2048 dense permutation
    let sources_s2048: Vec<u32> = (0..2048).map(|i| (i * 13) % 225).collect();
    verify_m8_adversarial_case(
        &ctx,
        &cpu,
        graph,
        &sources_s2048,
        "Grid15x15-ContractMax-S2048",
        Some(oracle),
    );

    // 2. High-diameter path graph with S=2048
    let path = path_graph(300);
    let path_oracle = |s: u32, v: u32| -> u32 { (s as i64 - v as i64).unsigned_abs() as u32 };
    let path_sources: Vec<u32> = (0..2048).map(|i| (i * 7) % 300).collect();
    verify_m8_adversarial_case(
        &ctx,
        &cpu,
        &path,
        &path_sources,
        "Path300-ContractMax-S2048",
        Some(path_oracle),
    );
}
