//! CPU reference BFS runner for WebAssembly.

use wasm_bindgen::prelude::*;
use webgpu_bfs_cpu::CpuBfsEngine;

use crate::{
    now_ms,
    types::{calc_mteps, count_traversed_edges, count_visited_vertices, WasmBfsResult},
    WasmGraph,
};

/// Runs CPU reference BFS synchronously.
#[wasm_bindgen]
pub fn run_cpu_bfs(
    graph: &WasmGraph,
    sources: &[u32],
    algorithm: Option<String>,
) -> Result<WasmBfsResult, JsValue> {
    let algo_str = algorithm.as_deref().unwrap_or("SequentialFifo");
    let engine = match algo_str {
        "BitParallel64" => CpuBfsEngine::bit_parallel(),
        "RayonBatched" | "SequentialFifo" | "Fifo" => CpuBfsEngine::fifo(),
        other => {
            return Err(JsValue::from_str(&format!(
                "Unknown CPU algorithm: {}",
                other
            )))
        }
    };

    let t0 = now_ms();
    let distances = webgpu_bfs_core::BfsEngine::run_ms_bfs(&engine, &graph.inner, sources)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let elapsed_ms = (now_ms() - t0).max(0.001);

    let traversed_edges = count_traversed_edges(&graph.inner, &distances, sources.len());
    let visited_vertices = count_visited_vertices(&distances);
    let mteps = calc_mteps(traversed_edges, elapsed_ms);

    Ok(WasmBfsResult::new(
        distances,
        graph.inner.num_vertices,
        sources.len(),
        elapsed_ms,
        mteps,
        traversed_edges,
        visited_vertices,
        algo_str.to_string(),
    ))
}
