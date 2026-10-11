use webgpu_bfs_core::generators::{generate_erdos_renyi_np, path_graph, star_graph};
use webgpu_bfs_core::graph::CsrGraph;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::GpuBfsEngine;

#[test]
fn test_t4_03_clustered_social_network_reachability_isolation() {
    let n0 = 120usize;
    let n1 = 150usize;
    let n2 = 100usize;
    let n3 = 80usize;
    let total_v = n0 + n1 + n2 + n3; // 450

    let c0 = generate_erdos_renyi_np(n0, 0.10, Some(101)).unwrap();
    let c1 = generate_erdos_renyi_np(n1, 0.04, Some(202)).unwrap();
    let c2 = star_graph(n2);
    let c3 = path_graph(n3);

    let offset0 = 0usize;
    let offset1 = n0;
    let offset2 = n0 + n1;
    let offset3 = n0 + n1 + n2;

    let mut combined_edges = Vec::new();
    let add_edges = |edges: &mut Vec<(u32, u32)>, g: &CsrGraph, offset: usize| {
        for u in 0..g.num_vertices {
            for &v in g.neighbors(u) {
                edges.push(((u + offset) as u32, (v as usize + offset) as u32));
            }
        }
    };

    add_edges(&mut combined_edges, &c0, offset0);
    add_edges(&mut combined_edges, &c1, offset1);
    add_edges(&mut combined_edges, &c2, offset2);
    add_edges(&mut combined_edges, &c3, offset3);

    let graph = CsrGraph::from_edge_list(total_v, &combined_edges);
    assert_eq!(graph.num_vertices, total_v);

    // 8 sources from each cluster = 32 sources total
    let mut sources = Vec::new();
    for i in 0..8 {
        sources.push((offset0 + i * 10) as u32);
        sources.push((offset1 + i * 15) as u32);
        sources.push((offset2 + i * 10) as u32);
        sources.push((offset3 + i * 9) as u32);
    }

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&graph, &sources).unwrap();

    let ctx = crate::common::shared_gpu_context();
    for &method in &crate::common::ALL_GPU_METHODS {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let gpu_dist = gpu.run_ms_bfs(&graph, &sources).unwrap();
        assert_eq!(
            gpu_dist, cpu_dist,
            "Method {:?} failed on Scenario 3 Clustered Network",
            method
        );
    }

    // Verify cross-cluster reachability isolation
    let cluster_of = |v: usize| -> usize {
        if v < offset1 {
            0
        } else if v < offset2 {
            1
        } else if v < offset3 {
            2
        } else {
            3
        }
    };

    for (s_idx, &src) in sources.iter().enumerate() {
        let src_cluster = cluster_of(src as usize);
        let row = &cpu_dist[s_idx * total_v..(s_idx + 1) * total_v];
        for (v, &d) in row.iter().enumerate().take(total_v) {
            let v_cluster = cluster_of(v);
            if src_cluster != v_cluster {
                assert_eq!(
                    d,
                    u32::MAX,
                    "Cross-cluster leakage: source {} (cluster {}) reached node {} (cluster {})",
                    src,
                    src_cluster,
                    v,
                    v_cluster
                );
            }
        }
    }
}
