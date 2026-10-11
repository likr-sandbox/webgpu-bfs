use webgpu_bfs_core::generators::generate_random_tree;
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;

#[test]
fn test_t1_f6_01_random_tree_connectivity_and_edge_count() {
    let tree = generate_random_tree(250, Some(777)).expect("Random tree generation failed");
    assert_eq!(tree.num_vertices, 250);
    assert_eq!(tree.num_edges, 498); // 2 * (250 - 1)
    assert!(tree.validate().is_ok());
    assert!(tree.is_symmetric());
}

#[test]
fn test_t1_f6_02_random_tree_diameter_and_single_path_oracle() {
    let tree = generate_random_tree(100, Some(42)).unwrap();
    let cpu = CpuBfsEngine::new();
    let dist = cpu.run_ms_bfs(&tree, &[0]).unwrap();

    // Since trees are bipartite, adjacent vertices have |dist(u) - dist(v)| == 1 exactly
    for u in 0..100 {
        let d_u = dist[u];
        for &v in tree.neighbors(u) {
            let d_v = dist[v as usize];
            let diff = (d_u as i64 - d_v as i64).abs();
            assert_eq!(
                diff, 1,
                "Adjacent tree nodes must have distance difference exactly 1"
            );
        }
    }
}

#[test]
fn test_t1_f6_03_random_tree_deterministic_seeding() {
    let t1 = generate_random_tree(50, Some(111)).unwrap();
    let t2 = generate_random_tree(50, Some(111)).unwrap();
    let t3 = generate_random_tree(50, Some(222)).unwrap();

    assert_eq!(t1.row_offsets, t2.row_offsets);
    assert_eq!(t1.col_indices, t2.col_indices);
    assert_ne!(t1.col_indices, t3.col_indices);
}

#[test]
fn test_t1_f6_04_random_tree_zero_unreachable_vertices() {
    let tree = generate_random_tree(150, Some(555)).unwrap();
    let cpu = CpuBfsEngine::new();
    let dist = cpu.run_ms_bfs(&tree, &[0]).unwrap();

    for (v, &d) in dist.iter().enumerate() {
        assert_ne!(
            d,
            u32::MAX,
            "Vertex {} should be reachable in connected tree",
            v
        );
    }
}

#[test]
fn test_t1_f6_05_random_tree_cpu_gpu_bitwise_parity() {
    let tree = generate_random_tree(100, Some(333)).unwrap();
    let sources = [0, 10, 25, 50, 75, 99];

    let ctx = crate::common::shared_gpu_context();
    crate::common::assert_cross_engine_parity(&ctx, &tree, &sources);
}
