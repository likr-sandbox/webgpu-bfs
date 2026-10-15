//! Milestone M25 Adversarial & Reachability Bitmask Test Suite
//!
//! Verifies:
//! 1. `ReachabilityBitmask` GPU execution vs CPU ground truth across diverse topologies.
//! 2. Exact bitwise reachability parity across batch sizes (S=1, 16, 32, 64, 128).
//! 3. Dynamic bank width scaling calculation and memory bounds.
//! 4. Hybrid direction-optimizing pipeline stability and disconnected graph sentinels.

use std::sync::Arc;
use webgpu_bfs_core::{
    generators::{generate_erdos_renyi, generate_grid2d, generate_random_tree, generate_rmat},
    CsrGraph, ReachabilityEngine,
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
fn test_m25_dynamic_bank_scaling_invariants() {
    // For small and medium graphs (|V| <= 524,288), bank sources must be 64
    assert_eq!(calculate_bank_sources(100), 64);
    assert_eq!(calculate_bank_sources(100_000), 64);
    assert_eq!(calculate_bank_sources(524_288), 64);
    let (shift64, mask64) = calculate_bank_shift_and_mask(100);
    assert_eq!(shift64, 6);
    assert_eq!(mask64, 63);

    // For graphs with 524,288 < |V| <= 1,048,576, bank sources scale down to 32
    assert_eq!(calculate_bank_sources(600_000), 32);
    assert_eq!(calculate_bank_sources(1_048_576), 32);
    let (shift32, mask32) = calculate_bank_shift_and_mask(600_000);
    assert_eq!(shift32, 5);
    assert_eq!(mask32, 31);

    // For graphs with 1,048,576 < |V| <= 2,097,152, bank sources scale down to 16
    assert_eq!(calculate_bank_sources(1_500_000), 16);
    assert_eq!(calculate_bank_sources(2_097_152), 16);
    let (shift16, mask16) = calculate_bank_shift_and_mask(1_500_000);
    assert_eq!(shift16, 4);
    assert_eq!(mask16, 15);

    // In all cases, bank_bytes = B * |V| * 4 must NEVER exceed 128 MB (134,217,728 bytes)
    for nv in [100, 10_000, 524_288, 600_000, 1_048_576, 2_000_000] {
        let b = calculate_bank_sources(nv);
        let bytes = (b as u64) * (nv as u64) * 4;
        assert!(
            bytes <= 134_217_728,
            "Bank size {} bytes for nv={} exceeds 128 MB",
            bytes,
            nv
        );
    }
}

#[test]
fn test_m25_gpu_reachability_grid2d_parity() {
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
    let sources = vec![0, 49, 99];

    let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reachability");
    let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reachability");

    assert_eq!(cpu_reach.num_vertices, 100);
    assert_eq!(gpu_reach.num_vertices, 100);
    assert_eq!(cpu_reach.num_sources, 3);
    assert_eq!(gpu_reach.num_sources, 3);

    for s_idx in 0..sources.len() {
        for v in 0..graph.num_vertices {
            let cpu_bit = cpu_reach.is_reachable(s_idx, v);
            let gpu_bit = gpu_reach.is_reachable(s_idx, v);
            assert_eq!(
                cpu_bit, gpu_bit,
                "Mismatch at source {} vertex {}: CPU={} GPU={}",
                s_idx, v, cpu_bit, gpu_bit
            );
        }
    }
}

#[test]
fn test_m25_gpu_reachability_disconnected_and_isolated() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::auto(context).expect("GpuBfsEngine::auto");
    let cpu_engine = CpuBfsEngine::new();

    // Disconnected graph: Component 1 (0..5), Component 2 (5..10), Isolated (10..15)
    let mut edges = Vec::new();
    for i in 0..4 {
        edges.push((i, i + 1));
        edges.push((i + 1, i));
    }
    for i in 5..9 {
        edges.push((i, i + 1));
        edges.push((i + 1, i));
    }
    let graph = CsrGraph::from_edge_list(15, &edges);

    let sources = vec![0, 7, 12]; // In comp 1, comp 2, and isolated
    let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reachability");
    let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reachability");

    for s_idx in 0..sources.len() {
        for v in 0..15 {
            let cpu_bit = cpu_reach.is_reachable(s_idx, v);
            let gpu_bit = gpu_reach.is_reachable(s_idx, v);
            assert_eq!(
                cpu_bit, gpu_bit,
                "Disconnected mismatch at source {} vertex {}: CPU={} GPU={}",
                s_idx, v, cpu_bit, gpu_bit
            );
        }
    }

    // Verify isolation guarantees
    // Source 0 (in comp 1) should reach 0..5, but NOT 5..15
    for v in 0..5 {
        assert!(gpu_reach.is_reachable(0, v));
    }
    for v in 5..15 {
        assert!(!gpu_reach.is_reachable(0, v));
    }

    // Source 12 (isolated) should only reach 12
    for v in 0..15 {
        assert_eq!(gpu_reach.is_reachable(2, v), v == 12);
    }
}

#[test]
fn test_m25_gpu_reachability_batch_sweep_rmat() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::indirect_sparse_queue_push(context).expect("GpuBfsEngine::isqp");
    let cpu_engine = CpuBfsEngine::new();

    // R-MAT scale 7 (128 vertices, power-law degree skew)
    let graph = generate_rmat(7, 8, 0.57, 0.19, 0.19, 0.05, Some(42), false)
        .expect("rmat")
        .to_undirected();

    // Test batch sizes across word boundaries: S = 1, 16, 32, 33, 64, 65, 128
    for &s_count in &[1, 16, 32, 33, 64, 65, 128] {
        let sources: Vec<u32> = (0..s_count).map(|s| (s * 3) % graph.num_vertices as u32).collect();
        let cpu_reach = cpu_engine.run_reachability(&graph, &sources).expect("cpu reachability");
        let gpu_reach = gpu_engine.run_reachability(&graph, &sources).expect("gpu reachability");

        assert_eq!(cpu_reach.num_sources, s_count as usize);
        assert_eq!(gpu_reach.num_sources, s_count as usize);

        for s_idx in 0..sources.len() {
            for v in 0..graph.num_vertices {
                let cpu_bit = cpu_reach.is_reachable(s_idx, v);
                let gpu_bit = gpu_reach.is_reachable(s_idx, v);
                assert_eq!(
                    cpu_bit, gpu_bit,
                    "Batch S={} mismatch at source {} vertex {}: CPU={} GPU={}",
                    s_count, s_idx, v, cpu_bit, gpu_bit
                );
            }
        }
    }
}

#[test]
fn test_m25_gpu_reachability_random_tree_and_erdos_renyi() {
    let context = match GpuContext::init_blocking() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            return;
        }
    };
    let gpu_engine = GpuBfsEngine::auto(context).expect("GpuBfsEngine::auto");
    let cpu_engine = CpuBfsEngine::new();

    // Random Tree (100 vertices)
    let tree = generate_random_tree(100, Some(1234)).expect("random tree");
    let sources_tree = vec![0, 25, 50, 75];
    let cpu_tree = cpu_engine.run_reachability(&tree, &sources_tree).expect("cpu reach");
    let gpu_tree = gpu_engine.run_reachability(&tree, &sources_tree).expect("gpu reach");

    // Every connected tree vertex is reachable from any source
    for s_idx in 0..sources_tree.len() {
        for v in 0..tree.num_vertices {
            assert!(gpu_tree.is_reachable(s_idx, v));
            assert_eq!(cpu_tree.is_reachable(s_idx, v), gpu_tree.is_reachable(s_idx, v));
        }
    }

    // Erdős-Rényi G(80, 0.05)
    let er = generate_erdos_renyi(80, 0.05, 5678).to_undirected();
    let sources_er = vec![0, 10, 20, 30, 40, 50];
    let cpu_er = cpu_engine.run_reachability(&er, &sources_er).expect("cpu reach");
    let gpu_er = gpu_engine.run_reachability(&er, &sources_er).expect("gpu reach");

    for s_idx in 0..sources_er.len() {
        for v in 0..er.num_vertices {
            assert_eq!(
                cpu_er.is_reachable(s_idx, v),
                gpu_er.is_reachable(s_idx, v),
                "ER mismatch at s_idx={}, v={}",
                s_idx,
                v
            );
        }
    }
}
