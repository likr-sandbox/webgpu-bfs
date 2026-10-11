//! Automated headless WebGPU integration tests for `webgpu-bfs-wasm`.
//!
//! Verifies that `WasmGpuContext` initializes on native/headless (Mesa Lavapipe Vulkan)
//! and executes `run_gpu_bfs` and `verify_bfs` across all 6 GPU methods:
//! 1. `NaiveSequential`
//! 2. `BatchedDenseSpMM`
//! 3. `BitParallel32`
//! 4. `BitParallel64`
//! 5. `BooleanReachability`
//! 6. `PushPullAdaptive`
//!
//! Asserting 100% exact bitwise distance equivalence against CPU reference.

use webgpu_bfs_wasm::{run_cpu_bfs, run_gpu_bfs, verify_bfs, WasmGpuContext, WasmGraph};

const ALL_SIX_METHODS: [&str; 6] = [
    "NaiveSequential",
    "BatchedDenseSpMM",
    "BitParallel32",
    "BitParallel64",
    "BooleanReachability",
    "PushPullAdaptive",
];

#[test]
fn test_wasm_gpu_context_creation() {
    pollster::block_on(async {
        let ctx = WasmGpuContext::create()
            .await
            .expect("Failed to create WasmGpuContext on native headless");
        println!("Adapter Name: {}", ctx.adapter_name());
        println!("Adapter Type: {}", ctx.adapter_type());
        println!("Adapter Backend: {}", ctx.adapter_backend());
        assert!(!ctx.adapter_name().is_empty());
    });
}

#[test]
fn test_wasm_all_six_methods_single_source_grid() {
    pollster::block_on(async {
        let ctx = WasmGpuContext::create()
            .await
            .expect("Create WasmGpuContext");
        let graph = WasmGraph::generate_grid2d(5, 5).expect("Generate Grid2D 5x5");
        let sources = vec![0u32];

        let _cpu_res = run_cpu_bfs(&graph, &sources, Some("SequentialFifo".to_string()))
            .expect("CPU BFS failed");

        for method in ALL_SIX_METHODS {
            let gpu_res = run_gpu_bfs(&ctx, &graph, &sources, Some(method.to_string()))
                .await
                .unwrap_or_else(|e| panic!("GPU BFS method '{}' failed: {:?}", method, e));

            assert_eq!(
                gpu_res.num_vertices, 25,
                "Method '{}': wrong num_vertices",
                method
            );
            assert_eq!(
                gpu_res.num_sources, 1,
                "Method '{}': wrong num_sources",
                method
            );
            assert_eq!(
                gpu_res.distance_of(0, 0),
                0,
                "Method '{}': source distance must be 0",
                method
            );
            // Grid corner-to-corner (0,0) to (4,4) has Manhattan distance 8
            assert_eq!(
                gpu_res.distance_of(0, 24),
                8,
                "Method '{}': (0,0) to (4,4) distance must be 8",
                method
            );

            // Verify 100% bitwise parity with CPU reference
            let verify = verify_bfs(&ctx, &graph, &sources, Some(method.to_string()))
                .await
                .expect("verify_bfs failed");

            assert!(
                verify.is_equivalent,
                "Method '{}' parity failure: {} / {} mismatches. Sample mismatch: src={:?}, v={:?}, gpu={:?}, cpu={:?}",
                method,
                verify.mismatched_pairs,
                verify.total_pairs,
                verify.sample_mismatch_source,
                verify.sample_mismatch_vertex,
                verify.sample_gpu_dist,
                verify.sample_cpu_dist
            );
        }
    });
}

#[test]
fn test_wasm_all_six_methods_multi_source_grid() {
    pollster::block_on(async {
        let ctx = WasmGpuContext::create()
            .await
            .expect("Create WasmGpuContext");
        let graph = WasmGraph::generate_grid2d(6, 6).expect("Generate Grid2D 6x6");
        // 4 diverse sources on 36-vertex grid
        let sources = vec![0u32, 5, 30, 35];

        let _cpu_res = run_cpu_bfs(&graph, &sources, Some("SequentialFifo".to_string()))
            .expect("CPU BFS failed");

        for method in ALL_SIX_METHODS {
            let _gpu_res = run_gpu_bfs(&ctx, &graph, &sources, Some(method.to_string()))
                .await
                .unwrap_or_else(|e| panic!("GPU BFS method '{}' failed: {:?}", method, e));

            let verify = verify_bfs(&ctx, &graph, &sources, Some(method.to_string()))
                .await
                .expect("verify_bfs failed");

            assert!(
                verify.is_equivalent,
                "Method '{}' multi-source parity failure: {} mismatches",
                method, verify.mismatched_pairs
            );
        }
    });
}

#[test]
fn test_wasm_all_six_methods_batch_size_32_and_64() {
    pollster::block_on(async {
        let ctx = WasmGpuContext::create()
            .await
            .expect("Create WasmGpuContext");
        let graph = WasmGraph::generate_grid2d(8, 8).expect("Generate Grid2D 8x8 (64 vertices)");

        // Test exactly 32 sources (full u32 word)
        let sources_32: Vec<u32> = (0..32).collect();
        for method in ALL_SIX_METHODS {
            let verify = verify_bfs(&ctx, &graph, &sources_32, Some(method.to_string()))
                .await
                .expect("verify_bfs 32 sources failed");
            assert!(
                verify.is_equivalent,
                "Method '{}' with 32 sources failed: {} mismatches",
                method, verify.mismatched_pairs
            );
        }

        // Test exactly 64 sources (full vec2<u32> / 2x words)
        let sources_64: Vec<u32> = (0..64).collect();
        for method in ALL_SIX_METHODS {
            let verify = verify_bfs(&ctx, &graph, &sources_64, Some(method.to_string()))
                .await
                .expect("verify_bfs 64 sources failed");
            assert!(
                verify.is_equivalent,
                "Method '{}' with 64 sources failed: {} mismatches",
                method, verify.mismatched_pairs
            );
        }
    });
}

#[test]
fn test_wasm_disconnected_graph_unreachable_sentinels() {
    pollster::block_on(async {
        let ctx = WasmGpuContext::create()
            .await
            .expect("Create WasmGpuContext");
        // Two disjoint components: (0..2) triangle, (3..5) triangle
        let mtx = "%%MatrixMarket matrix coordinate pattern general\n6 6 6\n1 2\n2 3\n3 1\n4 5\n5 6\n6 4\n";
        let graph = WasmGraph::from_mtx(mtx).expect("Parse disconnected MTX");
        let sources = vec![0u32, 3u32];

        for method in ALL_SIX_METHODS {
            let verify = verify_bfs(&ctx, &graph, &sources, Some(method.to_string()))
                .await
                .unwrap_or_else(|e| panic!("Method '{}' failed: {:?}", method, e));

            assert!(
                verify.is_equivalent,
                "Method '{}' disconnected parity failed: {} mismatches",
                method, verify.mismatched_pairs
            );
        }
    });
}
