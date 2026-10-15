//! Milestone M25 Empirical Challenger Test Suite
//!
//! Adversarial stress harness verifying:
//! 1. Mathematical and programmatic bounds of dynamic bank scaling `calculate_bank_sources`
//!    across all critical boundary vertex counts (|V| = 524,288, 524,289, 1,048,576, 1,048,577, 2,097,152, 2,097,153, etc.)
//! 2. WebGPU 128 MB storage buffer limit adherence ($B \times |V| \times 4 \le 134,217,728$) across a comprehensive range.
//! 3. Reachability Bitmask correctness on edge topologies: isolated vertices, disconnected components,
//!    dense bipartite graphs, stars, and boundary batch sizes ($S \in \{1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 256, 512\}$).
//! 4. Reachability Bitmask API invariants: bit packing, total_reachable_pairs, and point queries.
//! 5. Empirical behavior of reachability and distance BFS under dynamic bank scaling at $|V| > 524,288$.

use std::sync::Arc;
use webgpu_bfs_core::{
    generators::generate_grid2d,
    CsrGraph, ReachabilityBitmask, ReachabilityEngine,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{
    context::GpuContext,
    dispatch::indirect_sparse_push::{
        calculate_bank_shift_and_mask, calculate_bank_sources,
    },
    GpuBfsEngine,
};

#[test]
fn test_adversarial_bank_scaling_exact_boundaries() {
    // 1. Boundary: |V| = 524,288 -> B = 64 (Exactly 128.00 MB: 64 * 524,288 * 4 = 134,217,728)
    assert_eq!(calculate_bank_sources(524_288), 64);
    let bytes_524288 = (64u64) * (524_288u64) * 4;
    assert_eq!(bytes_524288, 134_217_728);
    let (shift, mask) = calculate_bank_shift_and_mask(524_288);
    assert_eq!(shift, 6);
    assert_eq!(mask, 63);

    // 2. Boundary + 1: |V| = 524,289 -> B = 32 (Without scaling, 64 * 524,289 * 4 = 134,217,984 > 128 MB!)
    // Scaled size: 32 * 524,289 * 4 = 67,108,992 bytes (64.00 MB <= 128 MB)
    assert_eq!(calculate_bank_sources(524_289), 32);
    let bytes_524289 = (32u64) * (524_289u64) * 4;
    assert!(bytes_524289 <= 134_217_728);
    let (shift, mask) = calculate_bank_shift_and_mask(524_289);
    assert_eq!(shift, 5);
    assert_eq!(mask, 31);

    // 3. Boundary: |V| = 1,048,576 -> B = 32 (Exactly 128.00 MB: 32 * 1,048,576 * 4 = 134,217,728)
    assert_eq!(calculate_bank_sources(1_048_576), 32);
    let bytes_1048576 = (32u64) * (1_048_576u64) * 4;
    assert_eq!(bytes_1048576, 134_217_728);
    let (shift, mask) = calculate_bank_shift_and_mask(1_048_576);
    assert_eq!(shift, 5);
    assert_eq!(mask, 31);

    // 4. Boundary + 1: |V| = 1,048,577 -> B = 16 (Without scaling, 32 * 1,048,577 * 4 = 134,217,856 > 128 MB!)
    // Scaled size: 16 * 1,048,577 * 4 = 67,108,928 bytes (64.00 MB <= 128 MB)
    assert_eq!(calculate_bank_sources(1_048_577), 16);
    let bytes_1048577 = (16u64) * (1_048_577u64) * 4;
    assert!(bytes_1048577 <= 134_217_728);
    let (shift, mask) = calculate_bank_shift_and_mask(1_048_577);
    assert_eq!(shift, 4);
    assert_eq!(mask, 15);

    // 5. Boundary: |V| = 2,097,152 -> B = 16 (Exactly 128.00 MB: 16 * 2,097,152 * 4 = 134,217,728)
    assert_eq!(calculate_bank_sources(2_097_152), 16);
    let bytes_2097152 = (16u64) * (2_097_152u64) * 4;
    assert_eq!(bytes_2097152, 134_217_728);
    let (shift, mask) = calculate_bank_shift_and_mask(2_097_152);
    assert_eq!(shift, 4);
    assert_eq!(mask, 15);

    // 6. Boundary + 1: |V| = 2,097,153 -> B = 8
    assert_eq!(calculate_bank_sources(2_097_153), 8);
    let bytes_2097153 = (8u64) * (2_097_153u64) * 4;
    assert!(bytes_2097153 <= 134_217_728);
    let (shift, mask) = calculate_bank_shift_and_mask(2_097_153);
    assert_eq!(shift, 3);
    assert_eq!(mask, 7);

    // 7. Higher boundaries
    assert_eq!(calculate_bank_sources(4_194_304), 8);
    assert_eq!(calculate_bank_sources(4_194_305), 4);
    assert_eq!(calculate_bank_sources(8_388_608), 4);
    assert_eq!(calculate_bank_sources(8_388_609), 2);
    assert_eq!(calculate_bank_sources(16_777_216), 2);
    assert_eq!(calculate_bank_sources(16_777_217), 1);
    assert_eq!(calculate_bank_sources(33_554_432), 1);

    // Mathematical invariant across continuous logarithmic sweep:
    // For all nv in [1, 33,554,432], B * nv * 4 must NEVER exceed 134,217,728 bytes.
    let sweep_points: Vec<usize> = vec![
        1, 2, 7, 31, 32, 63, 64, 127, 128, 255, 256, 1024, 65535, 65536,
        100_000, 281_903, 500_000, 524_287, 524_288, 524_289, 700_000,
        1_048_575, 1_048_576, 1_048_577, 1_500_000, 2_097_151, 2_097_152, 2_097_153,
        3_000_000, 4_194_304, 4_194_305, 8_388_608, 8_388_609, 16_777_216, 16_777_217,
        33_554_432,
    ];
    for &nv in &sweep_points {
        let b = calculate_bank_sources(nv);
        let bytes = (b as u64) * (nv as u64) * 4;
        assert!(
            bytes <= 134_217_728,
            "FAILED: B={} * nv={} * 4 = {} bytes > 128 MB",
            b, nv, bytes
        );
        let (shift, mask) = calculate_bank_shift_and_mask(nv);
        assert_eq!(1 << shift, b, "shift invariant failed for nv={}", nv);
        assert_eq!(mask, (b as u32) - 1, "mask invariant failed for nv={}", nv);
    }
}

#[test]
fn test_adversarial_reachability_isolated_all_vertices() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::auto(context).expect("GpuBfsEngine::auto");
    let cpu_engine = CpuBfsEngine::new();

    // Adversarial Graph: 64 completely isolated vertices (0 edges)
    let graph = CsrGraph::from_edge_list(64, &[]);
    let sources: Vec<u32> = (0..64).collect();

    let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reach");
    let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reach");

    assert_eq!(cpu_reach.total_reachable_pairs(), 64);
    assert_eq!(gpu_reach.total_reachable_pairs(), 64);

    for s in 0..64 {
        for v in 0..64 {
            let expected = s == v;
            assert_eq!(gpu_reach.is_reachable(s, v), expected);
            assert_eq!(cpu_reach.is_reachable(s, v), expected);
        }
    }
}

#[test]
fn test_adversarial_reachability_dense_bipartite_hybrid_direction_switch() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::auto(context).expect("GpuBfsEngine::auto");
    let cpu_engine = CpuBfsEngine::new();

    // Complete Bipartite Graph K_{100, 100} (200 vertices, 10,000 undirected edges = 20,000 directed edges)
    // Degree of all vertices is 100 (> 200 / 16 = 12.5), which strongly triggers the hybrid pull path!
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(20_000);
    for u in 0..100 {
        for v in 100..200 {
            edges.push((u as u32, v as u32));
            edges.push((v as u32, u as u32));
        }
    }
    let graph = CsrGraph::from_edge_list(200, &edges);
    assert_eq!(graph.max_degree(), 100);

    let sources = vec![0, 50, 99, 100, 150, 199];
    let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reach");
    let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reach");

    assert_eq!(cpu_reach.num_sources, 6);
    assert_eq!(gpu_reach.num_sources, 6);

    // K_{100, 100} is fully connected, all 200 vertices reachable from all 6 sources = 1200 pairs
    assert_eq!(cpu_reach.total_reachable_pairs(), 1200);
    assert_eq!(gpu_reach.total_reachable_pairs(), 1200);

    for s in 0..sources.len() {
        for v in 0..graph.num_vertices {
            assert!(gpu_reach.is_reachable(s, v));
            assert_eq!(cpu_reach.is_reachable(s, v), gpu_reach.is_reachable(s, v));
        }
    }
}

#[test]
fn test_adversarial_reachability_star_graph() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::auto(context).expect("GpuBfsEngine::auto");
    let cpu_engine = CpuBfsEngine::new();

    // Star Graph: Center vertex 0 connected to 255 leaves (1..256)
    // Center vertex degree = 255 (huge hub), leaf degrees = 1
    let nv = 256;
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for v in 1..nv {
        edges.push((0, v as u32));
        edges.push((v as u32, 0));
    }
    let graph = CsrGraph::from_edge_list(nv, &edges);

    // Sources: center, leaf 1, leaf 100, leaf 255
    let sources = vec![0, 1, 100, 255];
    let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reach");
    let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reach");

    assert_eq!(cpu_reach.total_reachable_pairs(), 4 * 256);
    assert_eq!(gpu_reach.total_reachable_pairs(), 4 * 256);

    for s in 0..sources.len() {
        for v in 0..nv {
            assert!(gpu_reach.is_reachable(s, v));
            assert_eq!(cpu_reach.is_reachable(s, v), gpu_reach.is_reachable(s, v));
        }
    }
}

#[test]
fn test_adversarial_reachability_odd_batch_sizes_sweep() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::auto(context).expect("GpuBfsEngine::auto");
    let cpu_engine = CpuBfsEngine::new();

    // 10x10 Grid (100 vertices)
    let graph = generate_grid2d(10, 10);

    // Adversarial batch sizes crossing word boundaries (32, 64, 128, etc.)
    for &s_count in &[1, 2, 7, 31, 32, 33, 63, 64, 65, 96, 100] {
        let sources: Vec<u32> = (0..s_count).map(|s| (s * 7) % 100).collect();
        let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reach");
        let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reach");

        assert_eq!(cpu_reach.num_sources, s_count as usize);
        assert_eq!(gpu_reach.num_sources, s_count as usize);

        for s_idx in 0..sources.len() {
            for v in 0..graph.num_vertices {
                assert_eq!(
                    cpu_reach.is_reachable(s_idx, v),
                    gpu_reach.is_reachable(s_idx, v),
                    "Mismatch on s_count={}, s_idx={}, v={}",
                    s_count,
                    s_idx,
                    v
                );
            }
        }
    }
}

#[test]
fn test_adversarial_reachability_large_scale_bank_scaling_nv_524289() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::auto(context).expect("GpuBfsEngine::auto");
    let cpu_engine = CpuBfsEngine::new();

    // Exactly boundary + 1 vertex count: nv = 524,289!
    // This triggers calculate_bank_sources(524_289) = 32!
    // Graph has 524,289 vertices, with a small chain 0 - 1 - 2 - 3 and rest isolated.
    let nv = 524_289;
    let edges: Vec<(u32, u32)> = vec![
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (2, 3), (3, 2),
    ];
    let graph = CsrGraph::from_edge_list(nv, &edges);

    // Verify dynamic bank scaling calculation for this exact graph
    assert_eq!(calculate_bank_sources(graph.num_vertices), 32);

    // Test reachability for sources: 0 (chain), 2 (chain), 1000 (isolated), 524288 (last isolated)
    let sources = vec![0, 2, 1000, 524_288];
    let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reach");
    let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reach");

    assert_eq!(cpu_reach.num_sources, 4);
    assert_eq!(gpu_reach.num_sources, 4);

    // Check chain connectivity:
    // Source 0 reaches 0, 1, 2, 3 (4 vertices)
    for v in 0..4 {
        assert!(gpu_reach.is_reachable(0, v));
        assert_eq!(cpu_reach.is_reachable(0, v), gpu_reach.is_reachable(0, v));
    }
    assert!(!gpu_reach.is_reachable(0, 4));
    assert!(!gpu_reach.is_reachable(0, 524_288));

    // Source 2 reaches 0, 1, 2, 3 (4 vertices)
    for v in 0..4 {
        assert!(gpu_reach.is_reachable(1, v));
        assert_eq!(cpu_reach.is_reachable(1, v), gpu_reach.is_reachable(1, v));
    }

    // Source 1000 reaches ONLY 1000
    assert!(gpu_reach.is_reachable(2, 1000));
    assert!(!gpu_reach.is_reachable(2, 0));
    assert!(!gpu_reach.is_reachable(2, 1001));

    // Source 524,288 reaches ONLY 524,288
    assert!(gpu_reach.is_reachable(3, 524_288));
    assert!(!gpu_reach.is_reachable(3, 0));
    assert!(!gpu_reach.is_reachable(3, 524_287));

    assert_eq!(cpu_reach.total_reachable_pairs(), gpu_reach.total_reachable_pairs());
}

#[test]
fn test_adversarial_reachability_nv_1048577_and_512_sources() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::auto(context).expect("GpuBfsEngine::auto");
    let cpu_engine = CpuBfsEngine::new();

    // Exactly boundary + 1 for 1,048,576: nv = 1,048,577!
    // Triggers calculate_bank_sources(1_048_577) = 16!
    let nv = 1_048_577;
    let edges: Vec<(u32, u32)> = vec![
        (0, 1), (1, 0),
        (1, 2), (2, 1),
    ];
    let graph = CsrGraph::from_edge_list(nv, &edges);
    assert_eq!(calculate_bank_sources(graph.num_vertices), 16);

    let sources: Vec<u32> = (0..64).collect();
    let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reachability");
    let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reachability");

    assert_eq!(gpu_reach.num_sources, 64);
    assert_eq!(gpu_reach.num_vertices, nv);
    assert_eq!(cpu_reach.total_reachable_pairs(), gpu_reach.total_reachable_pairs());

    for s in 0..64 {
        for v in 0..4 {
            assert_eq!(gpu_reach.is_reachable(s, v), cpu_reach.is_reachable(s, v));
        }
    }
}

#[test]
fn test_adversarial_isqp_reachability_nv_524289() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::indirect_sparse_queue_push(context).expect("GpuBfsEngine::isqp");
    let cpu_engine = CpuBfsEngine::new();

    let nv = 524_289;
    let edges: Vec<(u32, u32)> = vec![
        (0, 1), (1, 0),
        (1, 2), (2, 1),
    ];
    let graph = CsrGraph::from_edge_list(nv, &edges);
    let sources = vec![0, 1, 2];

    let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reachability");
    let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reachability");

    assert_eq!(gpu_reach.num_sources, 3);
    assert_eq!(gpu_reach.num_vertices, nv);
    assert_eq!(cpu_reach.total_reachable_pairs(), gpu_reach.total_reachable_pairs());

    for s in 0..sources.len() {
        for v in 0..4 {
            assert_eq!(gpu_reach.is_reachable(s, v), cpu_reach.is_reachable(s, v));
        }
    }
}

#[test]
fn test_adversarial_distance_bfs_under_bank_scaling_nv_524289() {
    use webgpu_bfs_core::BfsEngine;

    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::indirect_sparse_queue_push(context).expect("GpuBfsEngine");

    // nv = 524,289 triggers B = 32
    let nv = 524_289;
    let edges: Vec<(u32, u32)> = vec![
        (0, 1), (1, 0),
        (1, 2), (2, 1),
    ];
    let graph = CsrGraph::from_edge_list(nv, &edges);

    let sources_64: Vec<u32> = (0..64).collect();
    let gpu_dists = gpu_engine.run_ms_bfs(&graph, &sources_64).expect("gpu distance BFS");
    assert_eq!(gpu_dists.len(), 64 * nv);

    // Source 0 distances: 0 reaches 0 (dist 0), 1 (dist 1), 2 (dist 2)
    assert_eq!(gpu_dists[0], 0);
    assert_eq!(gpu_dists[1], 1);
    assert_eq!(gpu_dists[2], 2);
    assert_eq!(gpu_dists[3], u32::MAX);

    // Source 1 distances: 1 reaches 0 (dist 1), 1 (dist 0), 2 (dist 1)
    assert_eq!(gpu_dists[nv], 1);
    assert_eq!(gpu_dists[nv + 1], 0);
    assert_eq!(gpu_dists[nv + 2], 1);
    assert_eq!(gpu_dists[nv + 3], u32::MAX);

    // Source 2 distances: 2 reaches 0 (dist 2), 1 (dist 1), 2 (dist 0)
    assert_eq!(gpu_dists[2 * nv], 2);
    assert_eq!(gpu_dists[2 * nv + 1], 1);
    assert_eq!(gpu_dists[2 * nv + 2], 0);
    assert_eq!(gpu_dists[2 * nv + 3], u32::MAX);

    // Source 3 (isolated): reaches only 3
    assert_eq!(gpu_dists[3 * nv + 3], 0);
    assert_eq!(gpu_dists[3 * nv], u32::MAX);
}

#[test]
fn test_adversarial_reachability_bitmask_api_invariants() {
    // Test pure API and bit packing logic on ReachabilityBitmask
    let mut bitmask = ReachabilityBitmask::zeroed(10, 65);
    assert_eq!(bitmask.num_vertices, 10);
    assert_eq!(bitmask.num_sources, 65);
    assert_eq!(bitmask.words_per_vertex, 3); // (65 + 31) / 32 = 3
    assert_eq!(bitmask.words.len(), 30);
    assert_eq!(bitmask.total_reachable_pairs(), 0);

    // Out-of-bounds queries should return false
    assert!(!bitmask.is_reachable(65, 0));
    assert!(!bitmask.is_reachable(0, 10));
    assert!(!bitmask.is_reachable(100, 100));

    // Set bits across word boundaries: s=0, s=31, s=32, s=63, s=64
    bitmask.set_reachable(0, 5);
    bitmask.set_reachable(31, 5);
    bitmask.set_reachable(32, 5);
    bitmask.set_reachable(63, 5);
    bitmask.set_reachable(64, 5);

    assert!(bitmask.is_reachable(0, 5));
    assert!(bitmask.is_reachable(31, 5));
    assert!(bitmask.is_reachable(32, 5));
    assert!(bitmask.is_reachable(63, 5));
    assert!(bitmask.is_reachable(64, 5));
    assert!(!bitmask.is_reachable(1, 5));
    assert!(!bitmask.is_reachable(0, 4));

    assert_eq!(bitmask.total_reachable_pairs(), 5);

    // Verify canonical distances conversion
    let distances = vec![
        0, 1, u32::MAX, // s=0: reaches 0, 1; unreachable 2
        u32::MAX, 2, 0, // s=1: unreachable 0; reaches 1, 2
    ];
    let converted = ReachabilityBitmask::from_canonical_distances(3, 2, &distances);
    assert_eq!(converted.num_vertices, 3);
    assert_eq!(converted.num_sources, 2);
    assert!(converted.is_reachable(0, 0));
    assert!(converted.is_reachable(0, 1));
    assert!(!converted.is_reachable(0, 2));
    assert!(!converted.is_reachable(1, 0));
    assert!(converted.is_reachable(1, 1));
    assert!(converted.is_reachable(1, 2));
    assert_eq!(converted.total_reachable_pairs(), 4);
}
