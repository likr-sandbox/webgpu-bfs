#![allow(clippy::identity_op, clippy::erasing_op)]

//! ==============================================================================
//! Milestone M24: Adversarial Challenge Suite (Quantization & 2D Grid Edge Cases)
//! ==============================================================================
//!
//! Empirically challenges and stress-tests:
//! 1. `unpack_u8_words_to_u32` (scalar) and `unpack_u8_words_to_u32_parallel` across:
//!    - Boundary lengths: 0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 1023, 1024,
//!      32767, 32768, 32769 (parallel partition threshold), 65537, 100000.
//!    - Boundary distance values:
//!      * 0 (source distance)
//!      * 1..254 (reachable distances)
//!      * 255 (unreachable sentinel 0xFF)
//!      * u32::MAX (0xFFFFFFFF)
//!    - Dirty buffer pre-fill (0xDEADBEEF, 0xAAAAAAAA) to detect tail remainder leaks.
//! 2. 2D Grid Dispatch calculation & index reconstruction math under hypothetical massive queues:
//!    - 0, 1, 15, 16, 17, 1048560 (65535*16 boundary), 1048561 (1st 2D hop), 2000000, 16000000, 268431360.
//!    - Bijective 1-to-1 linear index mapping & sub-warp coordinate validity.
//!    - WGSL `select(num_groups.x, 256u, num_groups.x == 0u)` exact semantics under 1D and 2D grids.
//! 3. Diameter-based adaptive distance quantization (u8 vs u16 fallback) on GPU:
//!    - D < 255 (e.g. N=50, N=254 path graphs) -> triggers u8 quantization.
//!    - D >= 255 (e.g. N=256, N=300 path graphs) -> falls back to u16 quantization.
//!    - Disconnected topologies with unreachable sentinels (u32::MAX <-> 0xFF) for both D < 255 and D >= 255.
//!    - 100% Bitwise Parity guaranteed across all configurations against CpuBfsEngine.
//! 4. Double-buffered asynchronous chunk pipelining verification for large batches (S >= 1024).

use std::sync::Arc;

use webgpu_bfs_core::{
    generators::{generate_grid2d, star_graph},
    BfsEngine, CsrGraph,
};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{
    dispatch::indirect_sparse_push::{unpack_u8_words_to_u32, unpack_u8_words_to_u32_parallel},
    GpuBfsEngine, GpuContext,
};

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
            "[{tag}] PARITY MISMATCH: {mismatches}/{} entries differ. First mismatch at source {s_idx}, vertex {v}: GPU={d_gpu}, CPU={d_cpu}",
            num_sources * num_vertices
        );
    }
}

/// Constructs an undirected path graph 0 - 1 - 2 - ... - (n - 1) of diameter n - 1.
fn make_path_graph(n: usize) -> CsrGraph {
    assert!(n >= 2, "Path graph requires at least 2 vertices");
    let num_edges = 2 * (n - 1);
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(num_edges);

    row_offsets.push(0);
    for i in 0..n {
        if i > 0 {
            col_indices.push((i - 1) as u32);
        }
        if i + 1 < n {
            col_indices.push((i + 1) as u32);
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

/// Constructs a disconnected graph: component 1 is a path graph of `path_n` vertices,
/// component 2 consists of `isolated_n` isolated vertices.
fn make_path_plus_isolated(path_n: usize, isolated_n: usize) -> CsrGraph {
    let total_v = path_n + isolated_n;
    let num_edges = 2 * (path_n - 1);
    let mut row_offsets = Vec::with_capacity(total_v + 1);
    let mut col_indices = Vec::with_capacity(num_edges);

    row_offsets.push(0);
    // Path component
    for i in 0..path_n {
        if i > 0 {
            col_indices.push((i - 1) as u32);
        }
        if i + 1 < path_n {
            col_indices.push((i + 1) as u32);
        }
        row_offsets.push(col_indices.len() as u32);
    }
    // Isolated component
    for _ in 0..isolated_n {
        row_offsets.push(col_indices.len() as u32);
    }

    CsrGraph {
        num_vertices: total_v,
        num_edges,
        row_offsets,
        col_indices,
    }
}

/// Packs canonical u32 distances into packed u8 words (4 bytes per u32 word),
/// matching the exact behavior of `pack_distances_u32_to_u8.wgsl`.
fn pack_u32_to_u8_words(dist: &[u32]) -> Vec<u32> {
    let num_words = (dist.len() + 3) / 4;
    let mut packed = vec![0u32; num_words];
    for (w_idx, word) in packed.iter_mut().enumerate() {
        let in_idx = w_idx * 4;
        let mut val = 0u32;
        for k in 0..4 {
            let byte_val = if in_idx + k < dist.len() {
                let d = dist[in_idx + k];
                if d == u32::MAX {
                    0xFFu32
                } else {
                    d.min(0xFF)
                }
            } else {
                0xFFu32
            };
            val |= (byte_val & 0xFF) << (k * 8);
        }
        *word = val;
    }
    packed
}

// =============================================================================
// 1. Stress Test: unpack_u8_words_to_u32 & unpack_u8_words_to_u32_parallel
// =============================================================================

#[test]
fn test_adversarial_unpack_u8_boundary_lengths_and_values() {
    let boundary_lengths = [
        0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 1023, 1024,
        32767, 32768, 32769, 65537, 100000,
    ];

    for &len in &boundary_lengths {
        // Pattern A: All Zeros (source distances)
        {
            let expected = vec![0u32; len];
            let packed = pack_u32_to_u8_words(&expected);

            let mut actual_scalar = vec![0xDEADBEEFu32; len];
            unpack_u8_words_to_u32(&packed, &mut actual_scalar);
            assert_eq!(actual_scalar, expected, "Pattern A Scalar failed at len {len}");

            let mut actual_parallel = vec![0xDEADBEEFu32; len];
            unpack_u8_words_to_u32_parallel(&packed, &mut actual_parallel);
            assert_eq!(actual_parallel, expected, "Pattern A Parallel failed at len {len}");
        }

        // Pattern B: All Sentinel u32::MAX (unreachable nodes)
        {
            let expected = vec![u32::MAX; len];
            let packed = pack_u32_to_u8_words(&expected);

            let mut actual_scalar = vec![0xDEADBEEFu32; len];
            unpack_u8_words_to_u32(&packed, &mut actual_scalar);
            assert_eq!(actual_scalar, expected, "Pattern B Scalar failed at len {len}");

            let mut actual_parallel = vec![0xDEADBEEFu32; len];
            unpack_u8_words_to_u32_parallel(&packed, &mut actual_parallel);
            assert_eq!(actual_parallel, expected, "Pattern B Parallel failed at len {len}");
        }

        // Pattern C: Full Reachable Boundary Values (0, 1, 2, 127, 253, 254, u32::MAX cyclic)
        {
            let sample_values = [0u32, 1, 2, 127, 253, 254, u32::MAX];
            let expected: Vec<u32> = (0..len).map(|i| sample_values[i % sample_values.len()]).collect();
            let packed = pack_u32_to_u8_words(&expected);

            let mut actual_scalar = vec![0xDEADBEEFu32; len];
            unpack_u8_words_to_u32(&packed, &mut actual_scalar);
            assert_eq!(actual_scalar, expected, "Pattern C Scalar failed at len {len}");

            let mut actual_parallel = vec![0xDEADBEEFu32; len];
            unpack_u8_words_to_u32_parallel(&packed, &mut actual_parallel);
            assert_eq!(actual_parallel, expected, "Pattern C Parallel failed at len {len}");
        }

        // Pattern D: Continuous 0..254 sweep with interspersed sentinels
        {
            let expected: Vec<u32> = (0..len)
                .map(|i| {
                    if i % 11 == 0 {
                        u32::MAX
                    } else {
                        (i % 255) as u32
                    }
                })
                .collect();
            let packed = pack_u32_to_u8_words(&expected);

            let mut actual_scalar = vec![0xAAAAAAAAu32; len];
            unpack_u8_words_to_u32(&packed, &mut actual_scalar);
            assert_eq!(actual_scalar, expected, "Pattern D Scalar failed at len {len}");

            let mut actual_parallel = vec![0xAAAAAAAAu32; len];
            unpack_u8_words_to_u32_parallel(&packed, &mut actual_parallel);
            assert_eq!(actual_parallel, expected, "Pattern D Parallel failed at len {len}");
        }

        // Pattern E: Explicit 0xFF Raw Word unpack (should strictly map to u32::MAX)
        {
            let num_words = (len + 3) / 4;
            let packed = vec![0xFFFFFFFFu32; num_words];
            let expected = vec![u32::MAX; len];

            let mut actual_scalar = vec![0u32; len];
            unpack_u8_words_to_u32(&packed, &mut actual_scalar);
            assert_eq!(actual_scalar, expected, "Pattern E Scalar failed at len {len}");

            let mut actual_parallel = vec![0u32; len];
            unpack_u8_words_to_u32_parallel(&packed, &mut actual_parallel);
            assert_eq!(actual_parallel, expected, "Pattern E Parallel failed at len {len}");
        }
    }
}

#[test]
fn test_adversarial_unpack_u8_dirty_buffer_overwrite() {
    // Test that remainder elements (1, 2, 3 tail elements) strictly overwrite destination
    // memory and do not write beyond bounds or leave previous buffer contents.
    for remainder in 0..4 {
        for base in [0, 4, 16, 1024, 32768, 65536] {
            let len = base + remainder;
            let expected: Vec<u32> = (0..len).map(|i| (i % 200) as u32).collect();
            let packed = pack_u32_to_u8_words(&expected);

            let mut guard_buffer = vec![0x55555555u32; len + 8];
            unpack_u8_words_to_u32(&packed, &mut guard_buffer[..len]);

            assert_eq!(&guard_buffer[..len], &expected[..]);
            for &trailer in &guard_buffer[len..] {
                assert_eq!(trailer, 0x55555555, "Trailer overwritten beyond len {len}!");
            }

            let mut guard_buffer_par = vec![0x55555555u32; len + 8];
            unpack_u8_words_to_u32_parallel(&packed, &mut guard_buffer_par[..len]);

            assert_eq!(&guard_buffer_par[..len], &expected[..]);
            for &trailer in &guard_buffer_par[len..] {
                assert_eq!(trailer, 0x55555555, "Trailer overwritten beyond len {len} (parallel)!");
            }
        }
    }
}

// =============================================================================
// 2. 2D Grid Dispatch Calculation & Index Reconstruction Math
// =============================================================================

#[test]
fn test_adversarial_2d_grid_dispatch_massive_queues() {
    fn compute_dispatch(in_count: u32) -> (u32, u32, u32) {
        if in_count == 0 {
            return (0, 0, 0);
        }
        let total_wgs = (in_count + 15) / 16;
        if total_wgs > 65535 {
            (256, (total_wgs + 255) / 256, 1)
        } else {
            (total_wgs, 1, 1)
        }
    }

    // WGSL select(num_groups.x, 256u, num_groups.x == 0u) semantics
    fn wgsl_stride(num_groups_x: u32) -> u32 {
        if num_groups_x == 0 {
            256
        } else {
            num_groups_x
        }
    }

    let test_cases = [
        (0u32, (0, 0, 0)),
        (1u32, (1, 1, 1)),
        (15u32, (1, 1, 1)),
        (16u32, (1, 1, 1)),
        (17u32, (2, 1, 1)),
        (65535 * 16, (65535, 1, 1)),          // Exactly 1,048,560 vertices (1D limit boundary)
        (65535 * 16 + 1, (256, 256, 1)),      // 1,048,561 vertices (2D activated)
        (2_000_000, (256, 489, 1)),           // 2 million vertices
        (16_000_000, (256, 3907, 1)),         // 16 million vertices
        (268_431_360, (256, 65535, 1)),       // Maximum 2D capacity within WebGPU limits
    ];

    for &(in_count, expected_dispatch) in &test_cases {
        let dispatch = compute_dispatch(in_count);
        assert_eq!(
            dispatch, expected_dispatch,
            "Dispatch mismatch for in_count={in_count}"
        );

        if in_count == 0 {
            continue;
        }

        let (dim_x, dim_y, dim_z) = dispatch;
        assert!(dim_x <= 65535, "dim_x exceeds WebGPU limit: {dim_x}");
        assert!(dim_y <= 65535, "dim_y exceeds WebGPU limit: {dim_y}");
        assert_eq!(dim_z, 1);

        let stride = wgsl_stride(dim_x);
        let total_wgs = (in_count + 15) / 16;

        // Verify bounds: dim_x * dim_y >= total_wgs
        let total_workgroups = (dim_x as u64) * (dim_y as u64);
        assert!(
            total_workgroups >= total_wgs as u64,
            "Grid capacity {total_workgroups} < total_wgs {total_wgs} for in_count={in_count}"
        );

        // Verify index reconstruction on sample vertices
        let step = (in_count / 100).max(1);
        for q in (0..in_count).step_by(step as usize) {
            let target_wg = q / 16;
            let target_sub = q % 16;

            let gx = target_wg % dim_x;
            let gy = target_wg / dim_x;

            assert!(gx < dim_x);
            assert!(gy < dim_y);

            let reconstructed_wg = gx + gy * stride;
            assert_eq!(reconstructed_wg, target_wg);

            let in_queue_idx = reconstructed_wg * 16 + target_sub;
            assert_eq!(in_queue_idx, q);
            assert!(in_queue_idx < in_count);
        }

        // Verify that padding workgroups beyond total_wgs are safely skipped
        let excess_wg = total_wgs;
        let excess_in_queue = excess_wg * 16;
        assert!(
            excess_in_queue >= in_count,
            "Excess workgroup {excess_wg} loaded valid queue element {excess_in_queue} < {in_count}!"
        );
    }
}

// =============================================================================
// 3. GPU BFS Execution: Diameter Threshold (u8 vs u16) & Parity Verification
// =============================================================================

#[test]
fn test_adversarial_diameter_quantization_trigger_u8_vs_u16() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");
    let auto = GpuBfsEngine::auto(ctx.clone()).expect("Failed to create Auto engine");

    // Case 1: D = 49 < 255 (Path graph N = 50) -> Triggers u8 quantization
    {
        let graph = make_path_graph(50);
        let sources = [0u32, 25u32];
        let cpu_dist = cpu.run_ms_bfs(&graph, &sources).expect("CPU BFS");

        let isqp_dist = isqp
            .run_ms_bfs(&graph, &sources)
            .expect("ISQP on Path(50)");
        assert_parity(&isqp_dist, &cpu_dist, sources.len(), graph.num_vertices, "Path(50)_ISQP");

        let auto_dist = auto
            .run_ms_bfs(&graph, &sources)
            .expect("Auto on Path(50)");
        assert_parity(&auto_dist, &cpu_dist, sources.len(), graph.num_vertices, "Path(50)_Auto");
    }

    // Case 2: D = 253 < 255 (Path graph N = 254) -> Triggers u8 quantization at upper boundary
    {
        let graph = make_path_graph(254);
        let sources = [0u32, 100u32];
        let cpu_dist = cpu.run_ms_bfs(&graph, &sources).expect("CPU BFS");

        let isqp_dist = isqp
            .run_ms_bfs(&graph, &sources)
            .expect("ISQP on Path(254)");
        assert_parity(&isqp_dist, &cpu_dist, sources.len(), graph.num_vertices, "Path(254)_ISQP");

        let auto_dist = auto
            .run_ms_bfs(&graph, &sources)
            .expect("Auto on Path(254)");
        assert_parity(&auto_dist, &cpu_dist, sources.len(), graph.num_vertices, "Path(254)_Auto");
    }

    // Case 3: D = 255 >= 255 (Path graph N = 256) -> Falls back to u16 quantization
    {
        let graph = make_path_graph(256);
        let sources = [0u32, 128u32];
        let cpu_dist = cpu.run_ms_bfs(&graph, &sources).expect("CPU BFS");

        let isqp_dist = isqp
            .run_ms_bfs(&graph, &sources)
            .expect("ISQP on Path(256)");
        assert_parity(&isqp_dist, &cpu_dist, sources.len(), graph.num_vertices, "Path(256)_ISQP");

        let auto_dist = auto
            .run_ms_bfs(&graph, &sources)
            .expect("Auto on Path(256)");
        assert_parity(&auto_dist, &cpu_dist, sources.len(), graph.num_vertices, "Path(256)_Auto");
    }

    // Case 4: D = 299 >= 255 (Path graph N = 300) -> Falls back to u16 quantization
    {
        let graph = make_path_graph(300);
        let sources = [0u32, 150u32];
        let cpu_dist = cpu.run_ms_bfs(&graph, &sources).expect("CPU BFS");

        let isqp_dist = isqp
            .run_ms_bfs(&graph, &sources)
            .expect("ISQP on Path(300)");
        assert_parity(&isqp_dist, &cpu_dist, sources.len(), graph.num_vertices, "Path(300)_ISQP");

        let auto_dist = auto
            .run_ms_bfs(&graph, &sources)
            .expect("Auto on Path(300)");
        assert_parity(&auto_dist, &cpu_dist, sources.len(), graph.num_vertices, "Path(300)_Auto");
    }

    // Case 5: Disconnected graph with D = 49 < 255 and unreachable nodes (tests 0xFF sentinel mapping)
    {
        let graph = make_path_plus_isolated(50, 30);
        let sources = [0u32, 25u32];
        let cpu_dist = cpu.run_ms_bfs(&graph, &sources).expect("CPU BFS");

        let isqp_dist = isqp
            .run_ms_bfs(&graph, &sources)
            .expect("ISQP on Disconnected Path(50)+30");
        assert_parity(&isqp_dist, &cpu_dist, sources.len(), graph.num_vertices, "Disconnected_Path(50)+30_ISQP");
    }

    // Case 6: Disconnected graph with D = 299 >= 255 and unreachable nodes (tests u16 fallback with sentinels)
    {
        let graph = make_path_plus_isolated(300, 30);
        let sources = [0u32, 150u32];
        let cpu_dist = cpu.run_ms_bfs(&graph, &sources).expect("CPU BFS");

        let isqp_dist = isqp
            .run_ms_bfs(&graph, &sources)
            .expect("ISQP on Disconnected Path(300)+30");
        assert_parity(&isqp_dist, &cpu_dist, sources.len(), graph.num_vertices, "Disconnected_Path(300)+30_ISQP");
    }
}

// =============================================================================
// 4. Double-Buffered Pipelining & Large Batch Scaling (S >= 1024)
// =============================================================================

#[test]
fn test_adversarial_pipelined_multichunk_large_batch_parity() {
    let _guard = GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Arc::new(GpuContext::new().expect("Failed to create GpuContext"));
    let cpu = CpuBfsEngine::new();
    let isqp = GpuBfsEngine::indirect_sparse_queue_push(ctx.clone())
        .expect("Failed to create ISQP engine");

    // Star graph N = 500 (D = 2 < 255)
    let star = star_graph(500);

    // Test single-chunk (S <= 512) and multi-chunk (S = 1024) sweeps
    for &s in &[1, 16, 64, 128, 256, 512, 1024] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 13) % 500).collect();
        let cpu_dist = cpu.run_ms_bfs(&star, &sources).expect("CPU BFS");

        let isqp_dist = isqp
            .run_ms_bfs(&star, &sources)
            .unwrap_or_else(|e| panic!("ISQP failed on star(500) at S={s}: {e}"));

        assert_parity(&isqp_dist, &cpu_dist, s, star.num_vertices, &format!("Star(500)_S={s}"));
    }

    // 2D Grid graph 20x20 = 400 vertices (D = 38 < 255)
    let grid = generate_grid2d(20, 20);
    for &s in &[64, 512, 1024] {
        let sources: Vec<u32> = (0..s as u32).map(|i| (i * 7) % 400).collect();
        let cpu_dist = cpu.run_ms_bfs(&grid, &sources).expect("CPU BFS");

        let isqp_dist = isqp
            .run_ms_bfs(&grid, &sources)
            .unwrap_or_else(|e| panic!("ISQP failed on grid(20x20) at S={s}: {e}"));

        assert_parity(&isqp_dist, &cpu_dist, s, grid.num_vertices, &format!("Grid(20x20)_S={s}"));
    }
}
