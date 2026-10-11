#![allow(clippy::identity_op)]

//! Adversarial Stress and Verification Suite for `MultiWordBitParallelPush` (Milestone M12).
//!
//! Empirically challenges:
//! 1. 2D Grid Graph (20x20 = 400 vertices) across multiple batch sizes S in {1, 31, 32, 64, 128, 256}.
//! 2. High-diameter Random Tree (N=500).
//! 3. Directed/Undirected Erdos-Renyi Random Graph (N=500, p=0.03).
//! 4. Disconnected graph components.
//! 5. Large batch sizes on small graphs (S=512, S=1024) to challenge word count scaling.

use std::sync::Arc;
use webgpu_bfs_core::{
    generators::{disconnected_cliques, generate_erdos_renyi, generate_random_tree, Grid2D},
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuContext};

static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn assert_parity(
    ctx: &Arc<GpuContext>,
    cpu_engine: &CpuBfsEngine,
    graph: &CsrGraph,
    sources: &[u32],
    tag: &str,
) {
    let nv = graph.num_vertices;
    let s_count = sources.len();

    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .expect("CPU BFS failed");
    assert_eq!(cpu_dist.len(), s_count * nv);

    let gpu_engine = GpuBfsEngine::multi_word_bit_parallel_push(ctx.clone())
        .expect("Failed to instantiate MultiWordBitParallelPush engine");

    let gpu_dist = gpu_engine
        .run_ms_bfs(graph, sources)
        .unwrap_or_else(|e| panic!("[{tag}] GPU MultiWordBitParallelPush failed: {e}"));
    assert_eq!(gpu_dist.len(), s_count * nv);

    let mut mismatch_count = 0;
    let mut first_mismatch = None;

    for (s_idx, &src_val) in sources.iter().enumerate() {
        for v in 0..nv {
            let idx = s_idx * nv + v;
            let d_cpu = cpu_dist[idx];
            let d_gpu = gpu_dist[idx];
            if d_cpu != d_gpu {
                mismatch_count += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((s_idx, src_val, v, d_cpu, d_gpu));
                }
            }
        }
    }

    if let Some((s_idx, src, v, d_cpu, d_gpu)) = first_mismatch {
        panic!(
            "[{tag}] PARITY MISMATCH! Total mismatches: {mismatch_count}/{} (first at s_idx={s_idx}, src={src}, v={v}: CPU={d_cpu}, GPU={d_gpu})",
            s_count * nv
        );
    }
}

#[test]
fn test_grid2d_batch_scaling() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let grid = Grid2D::new(20, 20).unwrap();
    let graph = &grid.graph;

    for &s in &[1, 31, 32, 64, 128, 256] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 400).collect();
        assert_parity(
            &ctx,
            &cpu,
            graph,
            &sources,
            &format!("Grid2D_20x20_S={s}"),
        );
    }
}

#[test]
fn test_random_tree() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let tree = generate_random_tree(500, Some(42)).unwrap();
    let sources: Vec<u32> = (0..128).map(|i| (i * 3) % 500).collect();
    assert_parity(&ctx, &cpu, &tree, &sources, "RandomTree_N500_S128");

    let sources256: Vec<u32> = (0..256).map(|i| (i * 3) % 500).collect();
    assert_parity(&ctx, &cpu, &tree, &sources256, "RandomTree_N500_S256");
}

#[test]
fn test_erdos_renyi() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let er = generate_erdos_renyi(500, 0.03, 1234);
    let sources: Vec<u32> = (0..256).map(|i| (i * 7) % 500).collect();
    assert_parity(&ctx, &cpu, &er, &sources, "ErdosRenyi_N500_S256");
}

#[test]
fn test_disconnected_cliques() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let cliques = disconnected_cliques(&[50, 50, 50, 50, 50]);
    let sources: Vec<u32> = (0..128).map(|i| i % 250).collect();
    assert_parity(&ctx, &cpu, &cliques, &sources, "DisconnectedCliques_S128");
}

#[test]
fn test_boundary_graphs() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    // 1. Single vertex (N=1)
    let single = webgpu_bfs_core::generators::single_vertex();
    assert_parity(&ctx, &cpu, &single, &[0], "SingleVertex_S1");
    let sources32: Vec<u32> = vec![0; 32];
    assert_parity(&ctx, &cpu, &single, &sources32, "SingleVertex_S32");

    // 2. Star graph (N=500, 1 hub + 499 leaves)
    let star = webgpu_bfs_core::generators::star_graph(500);
    let sources_star: Vec<u32> = (0..128).map(|i| i % 500).collect();
    assert_parity(&ctx, &cpu, &star, &sources_star, "StarGraph_S128");

    // 3. High diameter path graph (N=300, D=299)
    let path = webgpu_bfs_core::generators::path_graph(300);
    let sources_path: Vec<u32> = (0..64).map(|i| (i * 5) % 300).collect();
    assert_parity(&ctx, &cpu, &path, &sources_path, "PathGraph_S64");
}

#[test]
fn test_large_batch_512_passes() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let grid = Grid2D::new(10, 10).unwrap();
    let graph = &grid.graph;

    // S = 512: raw_k = 16, words_k = 16 (exactly W_k * 2 = 16 words covered)
    let sources512: Vec<u32> = (0..512).map(|i| (i * 3) % 100).collect();
    assert_parity(&ctx, &cpu, graph, &sources512, "Grid2D_10x10_S512");
}

#[test]
fn test_large_batch_1024_fails_due_to_wk_limit() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let grid = Grid2D::new(10, 10).unwrap();
    let graph = &grid.graph;

    // S = 1024: Verified defense-in-depth fix (dispatcher chunk capping <= 512 and shader stride loop).
    let sources1024: Vec<u32> = (0..1024).map(|i| (i * 3) % 100).collect();
    assert_parity(&ctx, &cpu, graph, &sources1024, "Grid2D_10x10_S1024");
}

#[test]
fn test_large_batch_2048_passes() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let grid = Grid2D::new(10, 10).unwrap();
    let graph = &grid.graph;

    // S = 2048: 4 chunks of 512 sources or single 64-word pass.
    let sources2048: Vec<u32> = (0..2048).map(|i| (i * 3) % 100).collect();
    assert_parity(&ctx, &cpu, graph, &sources2048, "Grid2D_10x10_S2048");
}


