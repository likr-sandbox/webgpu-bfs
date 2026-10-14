#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M20: Adversarial Challenge & Extreme Hub Stress Test Suite
//! ==============================================================================
//!
//! Empirically challenges the M20 `indirect_sparse_push_512.wgsl` kernel and its
//! driver integration on:
//! 1. Extreme Hub Scaling & Boundary Thresholds:
//!    - Degree 1023 (< ULTRA_HUB_THRESHOLD = 1024)
//!    - Degree 1024 (== ULTRA_HUB_THRESHOLD: unrolling threshold boundary)
//!    - Degree 1025 (> ULTRA_HUB_THRESHOLD: non-multiple of stride 256)
//!    - Degree 2048 (exact multiple of 256)
//!    - Degree 2999, 4999 (degree up to |V|-1)
//! 2. Shared Memory Queue Spill & Quadrant Partitioning:
//!    - Massive discovery exceeding LDS capacity (WG_QUEUE_CAP = 2048, SUB_QUEUE_CAP = 512)
//!    - Concurrent spilling to VRAM `queue_sizes[out_slot]` alongside Phase 3 flush
//! 3. Synthetic Power-Law R-MAT Variation:
//!    - Scales 9, 10, 11 with edge factors 8, 16, 32 and distinct seeds
//!    - Batch sizes S in {1, 64, 256, 512, 1024, 2048}
//! 4. Disconnected Multi-Hub Topologies:
//!    - Disjoint star components with degree >= 1024 to verify unreachable sentinels (u32::MAX)
//! 5. Bank Boundaries and Irregular Batch Sizes:
//!    - S in {1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1024, 2048}
//! 6. Maximum Bit-Density Stress:
//!    - All S=512 sources identical on ultra-hub, activating all 16 words simultaneously.
//! 7. Super-Extreme Hub Scaling (|V|=40,000, Degree 39,999):
//!    - Surpassing web-Stanford's maximum degree (38,625), massive queue spilling (38k items)
//! 8. Multi-Hub Co-occurrence in Single Workgroup:
//!    - Multiple hubs (degree >= 1024) processed within the exact same workgroup
//! 9. Hub Placement Invariance:
//!    - Hub at middle index (N/2) and tail index (N-1) rather than index 0

use std::sync::Arc;

use webgpu_bfs_core::{
    generators::{generate_graph500_rmat, star_graph},
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuContext};

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

/// Helper to build a disconnected graph with two independent star components.
fn make_disconnected_double_star(n1: usize, n2: usize) -> CsrGraph {
    let total_v = n1 + n2;
    let total_e = 2 * (n1 - 1) + 2 * (n2 - 1);
    let mut row_offsets = Vec::with_capacity(total_v + 1);
    let mut col_indices = Vec::with_capacity(total_e);

    row_offsets.push(0);

    // Component 1: hub 0 connects to 1..n1
    for leaf in 1..n1 {
        col_indices.push(leaf as u32);
    }
    row_offsets.push(col_indices.len() as u32);

    // Leaves of component 1 connect to hub 0
    for _ in 1..n1 {
        col_indices.push(0);
        row_offsets.push(col_indices.len() as u32);
    }

    // Component 2: hub n1 connects to n1+1 .. total_v
    let hub2 = n1 as u32;
    for leaf in (n1 + 1)..total_v {
        col_indices.push(leaf as u32);
    }
    row_offsets.push(col_indices.len() as u32);

    // Leaves of component 2 connect to hub2
    for _ in (n1 + 1)..total_v {
        col_indices.push(hub2);
        row_offsets.push(col_indices.len() as u32);
    }

    CsrGraph {
        num_vertices: total_v,
        num_edges: total_e,
        row_offsets,
        col_indices,
    }
}

/// Helper to build a star graph where the hub is located at an arbitrary index `hub_idx`.
fn make_arbitrary_hub_star(n: usize, hub_idx: usize) -> CsrGraph {
    assert!(hub_idx < n);
    let num_edges = 2 * (n - 1);
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(num_edges);

    row_offsets.push(0);

    for v in 0..n {
        if v == hub_idx {
            // Hub connects to all other vertices
            for u in 0..n {
                if u != hub_idx {
                    col_indices.push(u as u32);
                }
            }
        } else {
            // Leaf connects to hub
            col_indices.push(hub_idx as u32);
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

/// Helper to build a graph with K interconnected hubs, each having its own dedicated leaves.
fn make_multi_hub_cluster(num_hubs: usize, leaves_per_hub: usize) -> CsrGraph {
    let total_v = num_hubs + num_hubs * leaves_per_hub;
    let mut row_offsets = Vec::with_capacity(total_v + 1);
    let mut col_indices = Vec::new();

    row_offsets.push(0);

    // First num_hubs vertices are hubs [0 .. num_hubs - 1]
    for h in 0..num_hubs {
        // Connect to all other hubs
        for other_h in 0..num_hubs {
            if h != other_h {
                col_indices.push(other_h as u32);
            }
        }
        // Connect to own dedicated leaves
        let leaf_start = num_hubs + h * leaves_per_hub;
        for l in 0..leaves_per_hub {
            col_indices.push((leaf_start + l) as u32);
        }
        row_offsets.push(col_indices.len() as u32);
    }

    // Now leaves: each connects back to its own hub
    for h in 0..num_hubs {
        for _ in 0..leaves_per_hub {
            col_indices.push(h as u32);
            row_offsets.push(col_indices.len() as u32);
        }
    }

    let num_edges = col_indices.len();
    CsrGraph {
        num_vertices: total_v,
        num_edges,
        row_offsets,
        col_indices,
    }
}

/// Challenge 1: Extreme Star Graph Hub Scaling around ULTRA_HUB_THRESHOLD (1024)
/// Degrees: 1023, 1024, 1025, 2048, 3000
#[test]
fn test_challenge_extreme_star_hub_thresholds() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");
    let auto = GpuBfsEngine::auto(ctx.clone()).expect("Failed to create Auto engine");

    let star_sizes = [1024, 1025, 1026, 2049, 3000];

    for &n in &star_sizes {
        let star = star_graph(n);
        let max_deg = star.max_degree();

        for &s in &[64, 256, 512, 1024] {
            let sources_mixed: Vec<u32> = (0..s as u32).map(|i| (i * 17) % n as u32).collect();
            let cpu_dist_a = cpu.run_ms_bfs(&star, &sources_mixed).expect("CPU BFS");
            let isqp_dist_a = isqp
                .run_ms_bfs(&star, &sources_mixed)
                .unwrap_or_else(|e| panic!("ISQP failed on star(N={n}, deg={max_deg}) at S={s}: {e}"));
            assert_parity(
                &isqp_dist_a,
                &cpu_dist_a,
                s,
                n,
                &format!("Star_N={n}_Deg={max_deg}_Mixed_ISQP_S={s}"),
            );

            let auto_dist_a = auto
                .run_ms_bfs(&star, &sources_mixed)
                .unwrap_or_else(|e| panic!("Auto failed on star(N={n}, deg={max_deg}) at S={s}: {e}"));
            assert_parity(
                &auto_dist_a,
                &cpu_dist_a,
                s,
                n,
                &format!("Star_N={n}_Deg={max_deg}_Mixed_Auto_S={s}"),
            );

            if s <= 512 {
                let sources_leaves: Vec<u32> = (1..=(s as u32)).map(|i| (i * 3) % (n as u32 - 1) + 1).collect();
                let cpu_dist_b = cpu.run_ms_bfs(&star, &sources_leaves).expect("CPU BFS");
                let isqp_dist_b = isqp
                    .run_ms_bfs(&star, &sources_leaves)
                    .unwrap_or_else(|e| panic!("ISQP failed on star leaves (N={n}) at S={s}: {e}"));
                assert_parity(
                    &isqp_dist_b,
                    &cpu_dist_b,
                    s,
                    n,
                    &format!("Star_N={n}_Deg={max_deg}_Leaves_ISQP_S={s}"),
                );
            }
        }
    }
}

/// Challenge 2: Massive Hub LDS Queue Overflow Spilling (|V|=6000, Degree 5999)
/// Tests workgroup queue exceeding WG_QUEUE_CAP (2048) and sub-quadrants (512 each).
#[test]
fn test_challenge_massive_queue_spill() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    let n = 6000;
    let star = star_graph(n);
    assert_eq!(star.max_degree(), 5999);

    for &s in &[1, 64, 256, 512, 1024] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 31) % n as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&star, &sources).expect("CPU BFS");
        let isqp_dist = isqp
            .run_ms_bfs(&star, &sources)
            .unwrap_or_else(|e| panic!("ISQP failed on massive star(N={n}) at S={s}: {e}"));

        assert_parity(
            &isqp_dist,
            &cpu_dist,
            s,
            n,
            &format!("MassiveQueueSpill_N={n}_S={s}"),
        );
    }
}

/// Challenge 3: Synthetic R-MAT Power-Law Graphs with Varied Scale, Edge Factor, and Seeds
#[test]
fn test_challenge_rmat_parameter_variations() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");
    let auto = GpuBfsEngine::auto(ctx.clone()).expect("Failed to create Auto engine");

    let configurations = [
        (9, 16, 42u64),    // V=512, moderate density
        (10, 8, 1234u64),   // V=1024, sparse power-law
        (10, 32, 7777u64),  // V=1024, dense power-law with massive hubs
        (11, 16, 9999u64),  // V=2048, high-volume power-law
    ];

    for &(scale, edge_factor, seed) in &configurations {
        let rmat = generate_graph500_rmat(scale, edge_factor, Some(seed))
            .unwrap_or_else(|e| panic!("Failed to generate R-MAT({scale}, {edge_factor}): {e}"));
        let nv = rmat.num_vertices;
        let max_deg = rmat.max_degree();

        for &s in &[64, 256, 512, 1024] {
            let sources: Vec<u32> = (0..s as u32).map(|i| (i * 47) % nv as u32).collect();
            let cpu_dist = cpu.run_ms_bfs(&rmat, &sources).expect("CPU BFS");

            let isqp_dist = isqp
                .run_ms_bfs(&rmat, &sources)
                .unwrap_or_else(|e| panic!("ISQP failed on R-MAT(scale={scale}, ef={edge_factor}) at S={s}: {e}"));
            assert_parity(
                &isqp_dist,
                &cpu_dist,
                s,
                nv,
                &format!("RMAT_sc{scale}_ef{edge_factor}_maxd{max_deg}_ISQP_S={s}"),
            );

            let auto_dist = auto
                .run_ms_bfs(&rmat, &sources)
                .unwrap_or_else(|e| panic!("Auto failed on R-MAT(scale={scale}, ef={edge_factor}) at S={s}: {e}"));
            assert_parity(
                &auto_dist,
                &cpu_dist,
                s,
                nv,
                &format!("RMAT_sc{scale}_ef{edge_factor}_maxd{max_deg}_Auto_S={s}"),
            );
        }
    }
}

/// Challenge 4: Disconnected Multi-Hub Topologies (Two independent star components with degree >= 1024)
#[test]
fn test_challenge_disconnected_ultra_hubs() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    let graph = make_disconnected_double_star(1200, 1200);
    let nv = graph.num_vertices;

    for &s in &[64, 256, 512, 1024] {
        let sources_c1: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 1200).collect();
        let cpu_dist_c1 = cpu.run_ms_bfs(&graph, &sources_c1).expect("CPU BFS");
        let isqp_dist_c1 = isqp
            .run_ms_bfs(&graph, &sources_c1)
            .unwrap_or_else(|e| panic!("ISQP failed on disconnected double star (C1) at S={s}: {e}"));
        assert_parity(&isqp_dist_c1, &cpu_dist_c1, s, nv, &format!("Disjoint_C1_S={s}"));

        let sources_mixed: Vec<u32> = (0..s as u32)
            .map(|i| if i % 2 == 0 { (i * 7) % 1200 } else { 1200 + ((i * 11) % 1200) })
            .collect();
        let cpu_dist_mixed = cpu.run_ms_bfs(&graph, &sources_mixed).expect("CPU BFS");
        let isqp_dist_mixed = isqp
            .run_ms_bfs(&graph, &sources_mixed)
            .unwrap_or_else(|e| panic!("ISQP failed on disconnected double star (Mixed) at S={s}: {e}"));
        assert_parity(&isqp_dist_mixed, &cpu_dist_mixed, s, nv, &format!("Disjoint_Mixed_S={s}"));
    }
}

/// Challenge 5: Bank Boundaries and Irregular Batch Sizes
#[test]
fn test_challenge_boundary_batch_sizes_and_bank_crossings() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    let rmat = generate_graph500_rmat(10, 16, Some(5555)).expect("Generate R-MAT scale 10");
    let nv = rmat.num_vertices;

    let boundary_counts = [
        1, 2, 63, 64, 65,
        127, 128, 129,
        255, 256, 257,
        511, 512, 513,
        1023, 1024, 2048
    ];

    for &s in &boundary_counts {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 29) % nv as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&rmat, &sources).expect("CPU BFS");
        let isqp_dist = isqp
            .run_ms_bfs(&rmat, &sources)
            .unwrap_or_else(|e| panic!("ISQP failed at boundary S={s}: {e}"));

        assert_parity(&isqp_dist, &cpu_dist, s, nv, &format!("Boundary_S={s}"));
    }
}

/// Challenge 6: Maximum Bit-Density Stress with All-Duplicate Ultra-Hub Sources
#[test]
fn test_challenge_all_duplicate_ultra_hub_sources() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    let n = 2049;
    let star = star_graph(n);
    assert_eq!(star.max_degree(), 2048);

    for &s in &[64, 256, 512, 1024] {
        let sources_all_hub = vec![0u32; s];
        let cpu_dist_hub = cpu.run_ms_bfs(&star, &sources_all_hub).expect("CPU BFS");
        let isqp_dist_hub = isqp
            .run_ms_bfs(&star, &sources_all_hub)
            .unwrap_or_else(|e| panic!("ISQP failed on all-hub sources at S={s}: {e}"));
        assert_parity(&isqp_dist_hub, &cpu_dist_hub, s, n, &format!("AllHub_S={s}"));

        let sources_all_leaf = vec![1u32; s];
        let cpu_dist_leaf = cpu.run_ms_bfs(&star, &sources_all_leaf).expect("CPU BFS");
        let isqp_dist_leaf = isqp
            .run_ms_bfs(&star, &sources_all_leaf)
            .unwrap_or_else(|e| panic!("ISQP failed on all-leaf sources at S={s}: {e}"));
        assert_parity(&isqp_dist_leaf, &cpu_dist_leaf, s, n, &format!("AllLeaf_S={s}"));
    }
}

/// Challenge 7: Super-Extreme Hub Scaling (|V|=40,000, Degree 39,999)
/// Tests scaling beyond web-Stanford's maximum degree (38,625) with massive queue overflow (38k items).
#[test]
fn test_challenge_super_extreme_hub_scaling() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    let n = 40_000;
    let star = star_graph(n);
    assert_eq!(star.max_degree(), 39_999);

    for &s in &[64, 512, 1024] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 97) % n as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&star, &sources).expect("CPU BFS");
        let isqp_dist = isqp
            .run_ms_bfs(&star, &sources)
            .unwrap_or_else(|e| panic!("ISQP failed on super-extreme star(N={n}) at S={s}: {e}"));

        assert_parity(
            &isqp_dist,
            &cpu_dist,
            s,
            n,
            &format!("SuperExtremeHub_N={n}_S={s}"),
        );
    }
}

/// Challenge 8: Multi-Hub Co-occurrence in Single Workgroup
/// 8 interconnected hubs each having degree 1200 (>= 1024). In hop 1, all 8 hubs are active
/// in the same initial frontier, forcing a single workgroup to collaboratively process 8 consecutive hubs in Phase 1.
#[test]
fn test_challenge_multi_hub_cooccurrence_single_workgroup() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    // 8 hubs, each with 1200 leaves => each hub has degree 7 + 1200 = 1207 (>= 1024)
    // Total vertices = 8 + 8 * 1200 = 9608
    let cluster = make_multi_hub_cluster(8, 1200);
    let nv = cluster.num_vertices;

    for &s in &[64, 256, 512] {
        // Sources placed on all 8 hubs concurrently
        let sources: Vec<u32> = (0..s as u32).map(|i| (i % 8) as u32).collect();
        let cpu_dist = cpu.run_ms_bfs(&cluster, &sources).expect("CPU BFS");
        let isqp_dist = isqp
            .run_ms_bfs(&cluster, &sources)
            .unwrap_or_else(|e| panic!("ISQP failed on multi-hub cluster at S={s}: {e}"));

        assert_parity(
            &isqp_dist,
            &cpu_dist,
            s,
            nv,
            &format!("MultiHubCluster_S={s}"),
        );
    }
}

/// Challenge 9: Hub Placement Invariance (Middle Index and Tail Index)
/// Confirms that ultra-hub unrolling and queueing does not depend on hub vertex being index 0.
#[test]
fn test_challenge_hub_placement_invariance() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();

    let n = 2500;
    // Hub at middle: N/2 = 1250 (degree 2499)
    let mid_star = make_arbitrary_hub_star(n, 1250);
    assert_eq!(mid_star.max_degree(), 2499);

    // Hub at tail: N-1 = 2499 (degree 2499)
    let tail_star = make_arbitrary_hub_star(n, 2499);
    assert_eq!(tail_star.max_degree(), 2499);

    // Use distinct engine instances so persistent CSR buffer pool does not alias across graphs with identical (V, E)
    let isqp_mid = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");
    let isqp_tail = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    for &s in &[64, 512] {
        let sources_mid: Vec<u32> = (0..s as u32).map(|i| (i * 23) % n as u32).collect();
        let cpu_dist_mid = cpu.run_ms_bfs(&mid_star, &sources_mid).expect("CPU BFS");
        let isqp_dist_mid = isqp_mid
            .run_ms_bfs(&mid_star, &sources_mid)
            .unwrap_or_else(|e| panic!("ISQP failed on mid-hub star at S={s}: {e}"));
        assert_parity(&isqp_dist_mid, &cpu_dist_mid, s, n, &format!("MidHub_S={s}"));

        let sources_tail: Vec<u32> = (0..s as u32).map(|i| (i * 41) % n as u32).collect();
        let cpu_dist_tail = cpu.run_ms_bfs(&tail_star, &sources_tail).expect("CPU BFS");
        let isqp_dist_tail = isqp_tail
            .run_ms_bfs(&tail_star, &sources_tail)
            .unwrap_or_else(|e| panic!("ISQP failed on tail-hub star at S={s}: {e}"));
        assert_parity(&isqp_dist_tail, &cpu_dist_tail, s, n, &format!("TailHub_S={s}"));
    }
}
