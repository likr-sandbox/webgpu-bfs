use webgpu_bfs_core::generators::generate_random_tree;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::GpuBfsEngine;

#[test]
fn test_t4_02_deep_directory_tree_hierarchy() {
    let nv = 500usize;
    let tree = generate_random_tree(nv, Some(20261010)).expect("Failed to build directory tree");
    assert_eq!(tree.num_vertices, nv);
    assert_eq!(tree.num_edges, 2 * (nv - 1));

    // 32 sources: root 0 and 31 distributed nodes
    let sources: Vec<u32> = (0..32).map(|i| (i * 15) % (nv as u32)).collect();

    let cpu = CpuBfsEngine::new();
    let cpu_dist = cpu.run_ms_bfs(&tree, &sources).unwrap();

    let ctx = crate::common::shared_gpu_context();
    for &method in &crate::common::ALL_GPU_METHODS {
        let gpu = GpuBfsEngine::new(ctx.clone(), method).unwrap();
        let gpu_dist = gpu.run_ms_bfs(&tree, &sources).unwrap();

        assert_eq!(
            gpu_dist, cpu_dist,
            "Method {:?} failed on Scenario 2 Directory Tree",
            method
        );
    }

    // Assert tree properties: zero unreachable, and exact distance diff 1 on all edges
    for s_idx in 0..32 {
        let row = &cpu_dist[s_idx * nv..(s_idx + 1) * nv];
        for (v, &d) in row.iter().enumerate() {
            assert_ne!(
                d,
                u32::MAX,
                "Directory node {} must be reachable from source {}",
                v,
                sources[s_idx]
            );
        }
        for u in 0..nv {
            let d_u = row[u];
            for &v in tree.neighbors(u) {
                let d_v = row[v as usize];
                let diff = (d_u as i64 - d_v as i64).abs();
                assert_eq!(
                    diff, 1,
                    "Adjacent tree nodes must have distance diff 1: |{} - {}| = {}",
                    d_u, d_v, diff
                );
            }
        }
    }
}
