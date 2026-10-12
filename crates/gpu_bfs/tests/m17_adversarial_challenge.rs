#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M17: Adversarial Stress & Edge-Case Challenge Suite
//! ==============================================================================
//!
//! Empirically challenges M17 Multi-Word Sparse Push and Topology Router on:
//! 1. Edge-case topologies: empty graph, single-vertex graph, isolated vertices,
//!    disconnected cliques, high-diameter path graph.
//! 2. Arbitrary and non-power-of-2 source counts:
//!    S in {1, 7, 13, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513, 768, 1025}.
//! 3. Reusable buffer stress: repeated `run_ms_bfs_into` calls with mutating graphs,
//!    irregular source counts, and adversarial noise pre-fills.
//! 4. Source anomalies: duplicate sources, all-hub, all-leaf, alternating sources.

use std::sync::Arc;

use webgpu_bfs_core::{
    generators::{
        disconnected_cliques, empty_graph, generate_graph500_rmat,
        isolated_vertices, path_graph, single_vertex, star_graph,
    },
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuContext};

/// Global test synchronization mutex preventing driver race conditions during parallel execution.
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

/// Challenge 1: Edge topologies (Empty, Single-Vertex, Isolated Vertices, Disconnected Cliques, Mixed).
#[test]
fn test_challenge_edge_topologies() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");
    let auto = GpuBfsEngine::auto(ctx.clone()).expect("Failed to create Auto engine");

    // 1.1 Empty graph (V=0, E=0)
    let empty = empty_graph();
    assert_eq!(empty.num_vertices, 0);
    // S=0 on empty graph: should gracefully succeed and return empty
    let sources_0: Vec<u32> = vec![];
    let cpu_dist = cpu.run_ms_bfs(&empty, &sources_0).expect("CPU empty graph S=0");
    let isqp_dist = isqp.run_ms_bfs(&empty, &sources_0).expect("ISQP empty graph S=0");
    let auto_dist = auto.run_ms_bfs(&empty, &sources_0).expect("Auto empty graph S=0");
    assert_parity(&isqp_dist, &cpu_dist, 0, 0, "Empty_S=0");
    assert_parity(&auto_dist, &cpu_dist, 0, 0, "Empty_Auto_S=0");

    // S>0 on empty graph: non-existent source indices should be rejected
    let sources_invalid = vec![0u32];
    assert!(isqp.run_ms_bfs(&empty, &sources_invalid).is_err(), "ISQP must reject source on empty graph");
    assert!(auto.run_ms_bfs(&empty, &sources_invalid).is_err(), "Auto must reject source on empty graph");

    // 1.2 Single vertex graph (V=1, E=0)
    let single = single_vertex();
    assert_eq!(single.num_vertices, 1);
    for &s in &[1, 13, 64, 127, 257, 512, 768] {
        let sources: Vec<u32> = vec![0; s];
        let cpu_dist = cpu.run_ms_bfs(&single, &sources).expect("CPU single vertex");
        let isqp_dist = isqp.run_ms_bfs(&single, &sources).expect("ISQP single vertex");
        let auto_dist = auto.run_ms_bfs(&single, &sources).expect("Auto single vertex");
        assert_parity(&isqp_dist, &cpu_dist, s, 1, &format!("Single_ISQP_S={s}"));
        assert_parity(&auto_dist, &cpu_dist, s, 1, &format!("Single_Auto_S={s}"));
    }

    // 1.3 Isolated vertices (V=100, E=0)
    let iso = isolated_vertices(100);
    assert_eq!(iso.num_vertices, 100);
    for &s in &[1, 33, 65, 127, 255, 513] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 100).collect();
        let cpu_dist = cpu.run_ms_bfs(&iso, &sources).expect("CPU isolated");
        let isqp_dist = isqp.run_ms_bfs(&iso, &sources).expect("ISQP isolated");
        let auto_dist = auto.run_ms_bfs(&iso, &sources).expect("Auto isolated");
        assert_parity(&isqp_dist, &cpu_dist, s, 100, &format!("Iso_ISQP_S={s}"));
        assert_parity(&auto_dist, &cpu_dist, s, 100, &format!("Iso_Auto_S={s}"));
    }

    // 1.4 Disconnected cliques (K_15, K_25, K_30, K_10 => total V=80)
    let cliques = disconnected_cliques(&[15, 25, 30, 10]);
    assert_eq!(cliques.num_vertices, 80);
    for &s in &[13, 64, 127, 256, 512, 768] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 11) % 80).collect();
        let cpu_dist = cpu.run_ms_bfs(&cliques, &sources).expect("CPU disconnected cliques");
        let isqp_dist = isqp.run_ms_bfs(&cliques, &sources).expect("ISQP disconnected cliques");
        let auto_dist = auto.run_ms_bfs(&cliques, &sources).expect("Auto disconnected cliques");
        assert_parity(&isqp_dist, &cpu_dist, s, 80, &format!("Cliques_ISQP_S={s}"));
        assert_parity(&auto_dist, &cpu_dist, s, 80, &format!("Cliques_Auto_S={s}"));
    }

    // 1.5 Path graph with high diameter (V=150, diameter 149)
    let path = path_graph(150);
    for &s in &[1, 64, 256, 512] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 31) % 150).collect();
        let cpu_dist = cpu.run_ms_bfs(&path, &sources).expect("CPU path graph");
        let isqp_dist = isqp.run_ms_bfs(&path, &sources).expect("ISQP path graph");
        let auto_dist = auto.run_ms_bfs(&path, &sources).expect("Auto path graph");
        assert_parity(&isqp_dist, &cpu_dist, s, 150, &format!("Path_ISQP_S={s}"));
        assert_parity(&auto_dist, &cpu_dist, s, 150, &format!("Path_Auto_S={s}"));
    }
}

/// Challenge 2: Non-power-of-2 and arbitrary source counts across various graph families.
#[test]
fn test_challenge_non_power_of_two_sources() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");
    let auto = GpuBfsEngine::auto(ctx.clone()).expect("Failed to create Auto engine");

    let rmat = generate_graph500_rmat(9, 16, Some(1234)).expect("Generate R-MAT scale 9");
    let nv_rmat = rmat.num_vertices;

    let star = star_graph(500);
    let nv_star = star.num_vertices;

    // Test a wide gamut of non-power-of-2 source counts
    let irregular_source_counts = [1, 7, 13, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513, 768, 1023, 1025];

    for &s in &irregular_source_counts {
        // Test R-MAT
        let rmat_sources: Vec<u32> = (0..s as u32).map(|i| (i * 17) % nv_rmat as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&rmat, &rmat_sources).expect("CPU R-MAT BFS");
        let isqp_dist = isqp
            .run_ms_bfs(&rmat, &rmat_sources)
            .unwrap_or_else(|e| panic!("ISQP failed on R-MAT at S={s}: {e}"));
        assert_parity(&isqp_dist, &cpu_dist, s, nv_rmat, &format!("RMAT_ISQP_S={s}"));

        let auto_dist = auto
            .run_ms_bfs(&rmat, &rmat_sources)
            .unwrap_or_else(|e| panic!("Auto failed on R-MAT at S={s}: {e}"));
        assert_parity(&auto_dist, &cpu_dist, s, nv_rmat, &format!("RMAT_Auto_S={s}"));

        // Test Star graph on subset of counts to control test duration
        if s <= 513 {
            let star_sources: Vec<u32> = (0..s as u32).map(|i| (i * 23) % nv_star as u32).collect();
            let cpu_star = cpu.run_ms_bfs(&star, &star_sources).expect("CPU Star BFS");
            let isqp_star = isqp
                .run_ms_bfs(&star, &star_sources)
                .unwrap_or_else(|e| panic!("ISQP failed on Star at S={s}: {e}"));
            assert_parity(&isqp_star, &cpu_star, s, nv_star, &format!("Star_ISQP_S={s}"));
        }
    }
}

/// Challenge 3: Reusable buffer stress testing under mutating graphs, varying batch sizes, and memory pollution.
#[test]
fn test_challenge_reusable_buffer_stress() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    let rmat = generate_graph500_rmat(9, 16, Some(999)).expect("Generate R-MAT scale 9");
    let star = star_graph(500);
    let cliques = disconnected_cliques(&[20, 20, 20]);

    // Pre-allocate a large output buffer capable of holding the max test dimensions
    let max_s = 768;
    let max_v = rmat.num_vertices.max(star.num_vertices).max(cliques.num_vertices);
    let mut reusable_output = vec![0xccccccccu32; max_s * max_v];

    // Sequence of diverse queries executed into the exact same reusable buffer
    let schedule: Vec<(&CsrGraph, usize, u32)> = vec![
        (&rmat, 512, 0x11111111),
        (&star, 127, 0x22222222),
        (&cliques, 257, 0x33333333),
        (&rmat, 768, 0x44444444),
        (&star, 65, 0x55555555),
        (&rmat, 255, 0x66666666),
        (&cliques, 513, 0x77777777),
        (&star, 512, 0x88888888),
    ];

    for (step, (graph, s, poison_pattern)) in schedule.into_iter().enumerate() {
        let nv = graph.num_vertices;
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 37 + step as u32) % nv as u32).collect();

        // 1. Poison destination buffer with unique sentinel pattern
        reusable_output.resize(s * nv, poison_pattern);
        reusable_output.fill(poison_pattern);

        // 2. Compute CPU ground truth
        let cpu_dist = cpu.run_ms_bfs(graph, &sources).expect("CPU run failed");

        // 3. Execute GPU BFS into poisoned buffer
        isqp.run_ms_bfs_into(graph, &sources, &mut reusable_output)
            .unwrap_or_else(|e| panic!("Step {step} failed at S={s}: {e}"));

        // 4. Verify 100% parity (ensures every single word was overwritten correctly and no poison remains)
        assert_parity(
            &reusable_output,
            &cpu_dist,
            s,
            nv,
            &format!("ReusableBufferStress_Step_{step}_S={s}"),
        );
    }
}

/// Challenge 4: Extreme source distribution anomalies (all sources identical, all-zero, all-max).
#[test]
fn test_challenge_source_distribution_anomalies() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    let star = star_graph(300);
    let nv = star.num_vertices;

    // 4.1 All sources identical (vertex 0 = center hub)
    let s = 512;
    let all_center: Vec<u32> = vec![0; s];
    let cpu_center = cpu.run_ms_bfs(&star, &all_center).expect("CPU all center");
    let isqp_center = isqp.run_ms_bfs(&star, &all_center).expect("ISQP all center");
    assert_parity(&isqp_center, &cpu_center, s, nv, "AllCenterHub_S=512");

    // 4.2 All sources identical leaf (vertex 150 = single leaf)
    let all_leaf: Vec<u32> = vec![150; s];
    let cpu_leaf = cpu.run_ms_bfs(&star, &all_leaf).expect("CPU all leaf");
    let isqp_leaf = isqp.run_ms_bfs(&star, &all_leaf).expect("ISQP all leaf");
    assert_parity(&isqp_leaf, &cpu_leaf, s, nv, "AllLeaf_S=512");

    // 4.3 Alternating 2 vertices [0, 150, 0, 150, ...]
    let alternating: Vec<u32> = (0..s as u32).map(|i| if i % 2 == 0 { 0 } else { 150 }).collect();
    let cpu_alt = cpu.run_ms_bfs(&star, &alternating).expect("CPU alternating");
    let isqp_alt = isqp.run_ms_bfs(&star, &alternating).expect("ISQP alternating");
    assert_parity(&isqp_alt, &cpu_alt, s, nv, "Alternating_S=512");
}
