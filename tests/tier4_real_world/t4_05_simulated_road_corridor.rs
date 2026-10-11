use webgpu_bfs_core::graph::CsrGraph;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod};

#[test]
fn test_t4_05_simulated_road_corridor_high_diameter() {
    let stages = 100usize;
    let lanes = 3usize;
    let corridor_nodes = stages * lanes; // 300
    let spur_nodes = 100usize;
    let total_v = corridor_nodes + spur_nodes; // 400

    let mut edges = Vec::new();

    // 1. Longitudinal highway lane segments (stage s -> stage s+1)
    for s in 0..(stages - 1) {
        for l in 0..lanes {
            let u = s * lanes + l;
            let v = (s + 1) * lanes + l;
            edges.push((u as u32, v as u32));
            edges.push((v as u32, u as u32));
        }
    }

    // 2. Lateral cross-overs between adjacent lanes every 5 stages
    for s in (0..stages).step_by(5) {
        for l in 0..(lanes - 1) {
            let u = s * lanes + l;
            let v = s * lanes + l + 1;
            edges.push((u as u32, v as u32));
            edges.push((v as u32, u as u32));
        }
    }

    // 3. Dead-end rural spurs attached to outer lane
    for s in 0..spur_nodes {
        let u = s * lanes; // outer lane 0
        let spur = corridor_nodes + s;
        edges.push((u as u32, spur as u32));
        edges.push((spur as u32, u as u32));
    }

    let graph = CsrGraph::from_edge_list(total_v, &edges);
    assert_eq!(graph.num_vertices, total_v);
    assert!(graph.is_symmetric());

    // 16 sources: northern terminus (stage 0), southern terminus (stage 99), and spurs
    let sources = [
        0u32,
        1,
        2,                             // North terminus
        ((stages - 1) * lanes) as u32, // South terminus
        corridor_nodes as u32,
        (corridor_nodes + 50) as u32,
        (corridor_nodes + 99) as u32,
        (stages / 2 * lanes) as u32,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ];

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).unwrap();

    let ctx = crate::common::shared_gpu_context();
    let engine = GpuBfsEngine::new(ctx, GpuBfsMethod::PushPullAdaptive).unwrap();
    let gpu_dist = engine.run_ms_bfs(&graph, &sources).unwrap();

    assert_eq!(gpu_dist, cpu_dist, "Scenario 5 Road Corridor parity failed");

    // High diameter verification: distance from North terminus (0) to South terminus ((stages - 1) * lanes)
    let s0_to_south = cpu_dist[(stages - 1) * lanes];
    assert!(
        s0_to_south >= 99,
        "Corridor distance must be at least 99 hops, got {}",
        s0_to_south
    );

    // Path continuity: for every reached node v with d > 0, there exists a neighbor u with d(u) == d - 1
    let row0 = &cpu_dist[0..total_v];
    for v in 0..total_v {
        let d_v = row0[v];
        if d_v > 0 && d_v != u32::MAX {
            let has_pred = graph
                .neighbors(v)
                .iter()
                .any(|&u| row0[u as usize] == d_v - 1);
            assert!(
                has_pred,
                "Node {} with distance {} must have a neighbor with distance {}",
                v,
                d_v,
                d_v - 1
            );
        }
    }
}
