use webgpu_bfs_core::generators::{generate_erdos_renyi_np, generate_random_tree, Grid2D};
use webgpu_bfs_core::BfsEngine;
use webgpu_bfs_cpu::CpuBfsEngine;

#[test]
fn test_t1_f8_01_cpu_sequential_fifo_oracle_parity() {
    let grid = Grid2D::new(8, 8).unwrap();
    let sources = [0, 7, 56, 63, 27];
    let oracle = grid.ms_bfs_oracle(&sources);

    let fifo = CpuBfsEngine::fifo();
    let dist = fifo.run_ms_bfs(&grid.graph, &sources).unwrap();
    assert_eq!(dist, oracle);
}

#[test]
fn test_t1_f8_02_cpu_rayon_batched_parallel_parity() {
    let graph = generate_erdos_renyi_np(100, 0.08, Some(101)).unwrap();
    let sources: Vec<u32> = (0..32).collect();

    let fifo = CpuBfsEngine::fifo();
    let rayon = CpuBfsEngine::rayon();

    let dist_fifo = fifo.run_ms_bfs(&graph, &sources).unwrap();
    let dist_rayon = rayon.run_ms_bfs(&graph, &sources).unwrap();

    assert_eq!(dist_fifo, dist_rayon);
}

#[test]
fn test_t1_f8_03_cpu_bit_parallel_64_parity() {
    let tree = generate_random_tree(100, Some(202)).unwrap();
    let sources: Vec<u32> = (0..64).collect();

    let fifo = CpuBfsEngine::fifo();
    let bp64 = CpuBfsEngine::bit_parallel();

    let dist_fifo = fifo.run_ms_bfs(&tree, &sources).unwrap();
    let dist_bp64 = bp64.run_ms_bfs(&tree, &sources).unwrap();

    assert_eq!(dist_fifo, dist_bp64);
}

#[test]
fn test_t1_f8_04_cpu_source_order_invariance() {
    let graph = generate_erdos_renyi_np(50, 0.1, Some(303)).unwrap();
    let fifo = CpuBfsEngine::fifo();

    let dist_a = fifo.run_ms_bfs(&graph, &[2, 15, 40]).unwrap();
    let dist_b = fifo.run_ms_bfs(&graph, &[40, 2, 15]).unwrap();

    // Row for source 2 is at index 0 in dist_a, and index 1 in dist_b
    let row_a = &dist_a[0..50];
    let row_b = &dist_b[50..100];
    assert_eq!(row_a, row_b);
}

#[test]
fn test_t1_f8_05_cpu_engine_bounds_validation() {
    let graph = generate_erdos_renyi_np(20, 0.1, Some(404)).unwrap();
    let fifo = CpuBfsEngine::fifo();

    assert!(fifo.run_ms_bfs(&graph, &[20]).is_err());
    assert!(fifo.run_ms_bfs(&graph, &[0, 25]).is_err());
}
