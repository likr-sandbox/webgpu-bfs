use std::sync::Arc;
use webgpu_bfs_core::{BfsEngine, CsrGraph};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

use super::diagnostics::format_mismatch_diagnostic;

/// Complete array of all six GPU BFS methods implemented in webgpu-bfs.
pub const ALL_GPU_METHODS: [GpuBfsMethod; 6] = [
    GpuBfsMethod::NaiveSequential,
    GpuBfsMethod::BatchedDenseSpMM,
    GpuBfsMethod::BitParallel32,
    GpuBfsMethod::BitParallel64,
    GpuBfsMethod::BooleanReachability,
    GpuBfsMethod::PushPullAdaptive,
];

/// Validates structural invariants on any BFS output vector.
pub fn assert_distance_vector_invariants(graph: &CsrGraph, sources: &[u32], dist: &[u32]) {
    let nv = graph.num_vertices;
    let ns = sources.len();

    if nv == 0 || ns == 0 {
        assert_eq!(
            dist.len(),
            0,
            "Distance vector must be empty for 0 vertices or 0 sources"
        );
        return;
    }

    assert_eq!(
        dist.len(),
        ns * nv,
        "Distance vector length mismatch: expected {} * {} = {}, got {}",
        ns,
        nv,
        ns * nv,
        dist.len()
    );

    let is_undirected = graph.is_symmetric();

    for (s_idx, &src) in sources.iter().enumerate() {
        // Invariant 1: Source self-distance == 0
        let self_dist = dist[s_idx * nv + src as usize];
        assert_eq!(
            self_dist, 0,
            "Source self-distance must be 0! Source idx={}, id={}",
            s_idx, src
        );

        // Invariant 2: Triangle inequality for reachable edges
        for u in 0..nv {
            let d_u = dist[s_idx * nv + u];
            if d_u != u32::MAX {
                for &v_u32 in graph.neighbors(u) {
                    let v = v_u32 as usize;
                    let d_v = dist[s_idx * nv + v];
                    assert!(
                        d_v <= d_u.saturating_add(1),
                        "Directed triangle inequality violated: dist(s,{})={} > dist(s,{})={}+1 (edge {} -> {})",
                        v, d_v, u, d_u, u, v
                    );
                    if is_undirected && d_v != u32::MAX {
                        let diff = (d_u as i64 - d_v as i64).abs();
                        assert!(
                            diff <= 1,
                            "Undirected metric inequality violated: |dist(s,{}) - dist(s,{})| = |{} - {}| = {} > 1",
                            u, v, d_u, d_v, diff
                        );
                    }
                }
            }
        }
    }
}

/// Asserts 100% bitwise distance equivalence against CPU ground truth across all 6 GPU methods.
pub fn assert_cross_engine_parity(ctx: &Arc<GpuContext>, graph: &CsrGraph, sources: &[u32]) {
    let cpu_engine = CpuBfsEngine::new();
    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .expect("CPU BFS failed");
    assert_distance_vector_invariants(graph, sources, &cpu_dist);

    for &method in &ALL_GPU_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method)
            .unwrap_or_else(|e| panic!("Failed to create GPU engine for {:?}: {:?}", method, e));
        let gpu_dist = gpu_engine
            .run_ms_bfs(graph, sources)
            .unwrap_or_else(|e| panic!("GPU BFS failed for method {:?}: {:?}", method, e));

        assert_eq!(
            gpu_dist.len(),
            cpu_dist.len(),
            "Method {:?}: length mismatch",
            method
        );

        if gpu_dist != cpu_dist {
            let diag = format_mismatch_diagnostic(
                &format!("{:?}", method),
                graph,
                sources,
                &gpu_dist,
                &cpu_dist,
            );
            panic!("{}", diag);
        }
    }
}

/// Asserts 100% bitwise distance equivalence against an analytical ground truth oracle.
pub fn assert_oracle_parity(
    ctx: &Arc<GpuContext>,
    graph: &CsrGraph,
    sources: &[u32],
    oracle_dist: &[u32],
) {
    assert_distance_vector_invariants(graph, sources, oracle_dist);

    let cpu_engine = CpuBfsEngine::new();
    let cpu_dist = cpu_engine
        .run_ms_bfs(graph, sources)
        .expect("CPU BFS failed");
    if cpu_dist != oracle_dist {
        let diag =
            format_mismatch_diagnostic("CpuBfsEngine", graph, sources, &cpu_dist, oracle_dist);
        panic!(
            "CPU reference BFS does not match analytical oracle: {}",
            diag
        );
    }

    for &method in &ALL_GPU_METHODS {
        let gpu_engine = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let gpu_dist = gpu_engine.run_ms_bfs(graph, sources).unwrap();
        if gpu_dist != oracle_dist {
            let diag = format_mismatch_diagnostic(
                &format!("{:?}", method),
                graph,
                sources,
                &gpu_dist,
                oracle_dist,
            );
            panic!(
                "Method {:?} does not match analytical oracle: {}",
                method, diag
            );
        }
    }
}
