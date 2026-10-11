//! Verification and benchmark suite runner.

use wasm_bindgen::prelude::*;

use crate::{
    cpu::run_cpu_bfs,
    gpu::{run_gpu_bfs, WasmGpuContext},
    now_ms,
    types::{WasmBfsResult, WasmVerifyResult},
    WasmGraph,
};

/// Verifies exact distance equivalence between GPU and CPU implementations.
#[wasm_bindgen]
pub async fn verify_bfs(
    ctx: &WasmGpuContext,
    graph: &WasmGraph,
    sources: &[u32],
    method: Option<String>,
) -> Result<WasmVerifyResult, JsValue> {
    let method_str = method.as_deref().unwrap_or("BitParallel64");

    let gpu_res = run_gpu_bfs(ctx, graph, sources, Some(method_str.to_string())).await?;
    let cpu_res = run_cpu_bfs(graph, sources, Some("SequentialFifo".to_string()))?;

    let gpu_dist = gpu_res.distances;
    let cpu_dist = cpu_res.distances;

    let total_pairs = gpu_dist.len();
    let mut mismatches = 0;
    let mut sample_src = None;
    let mut sample_v = None;
    let mut sample_g = None;
    let mut sample_c = None;

    let nv = graph.num_vertices();
    for (idx, (&g, &c)) in gpu_dist.iter().zip(cpu_dist.iter()).enumerate() {
        if g != c {
            if mismatches == 0 {
                let s = (idx / nv) as u32;
                let v = (idx % nv) as u32;
                sample_src = Some(s);
                sample_v = Some(v);
                sample_g = Some(g);
                sample_c = Some(c);
            }
            mismatches += 1;
        }
    }

    let is_eq = mismatches == 0;
    let msg = if is_eq {
        format!(
            "PASS: 100% exact match across all {} source-vertex pairs (Method: {})",
            total_pairs, method_str
        )
    } else {
        format!(
            "FAIL: {} / {} mismatches detected between GPU {} and CPU reference",
            mismatches, total_pairs, method_str
        )
    };

    Ok(WasmVerifyResult::new(
        is_eq,
        total_pairs,
        mismatches,
        sample_src,
        sample_v,
        sample_g,
        sample_c,
        msg,
    ))
}

/// Runs a multi-method benchmark comparing algorithms.
#[wasm_bindgen]
pub async fn run_benchmark(
    ctx: &WasmGpuContext,
    graph: &WasmGraph,
    sources: &[u32],
    methods: Option<js_sys::Array>,
    iterations: Option<u32>,
    warmup: Option<u32>,
) -> Result<js_sys::Array, JsValue> {
    let iters = iterations.unwrap_or(3).max(1);
    let warm = warmup.unwrap_or(1);

    let default_methods = vec![
        "PushPullAdaptive".to_string(),
        "BitParallel64".to_string(),
        "BitParallel32".to_string(),
        "BooleanReachability".to_string(),
        "BatchedDenseSpMM".to_string(),
        "NaiveSequential".to_string(),
        "CPU_FIFO".to_string(),
    ];

    let target_methods: Vec<String> = if let Some(m_arr) = methods {
        m_arr.iter().filter_map(|v| v.as_string()).collect()
    } else {
        default_methods
    };

    let results = js_sys::Array::new();

    for m in &target_methods {
        // 1. Warm-up
        for _ in 0..warm {
            if m.starts_with("CPU") {
                let _ = run_cpu_bfs(graph, sources, Some("SequentialFifo".to_string()))?;
            } else {
                let _ = run_gpu_bfs(ctx, graph, sources, Some(m.clone())).await?;
            }
        }

        // 2. Timed runs
        let mut times = Vec::with_capacity(iters as usize);
        let mut last_res: Option<WasmBfsResult> = None;

        for _ in 0..iters {
            let t0 = now_ms();
            let res = if m.starts_with("CPU") {
                run_cpu_bfs(graph, sources, Some("SequentialFifo".to_string()))?
            } else {
                run_gpu_bfs(ctx, graph, sources, Some(m.clone())).await?
            };
            let elapsed = (now_ms() - t0).max(0.001);
            times.push(elapsed);
            last_res = Some(res);
        }

        let min_t = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_t = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_t = times.iter().sum::<f64>() / (times.len() as f64);

        let lr = last_res.unwrap();
        let mteps = crate::types::calc_mteps(lr.traversed_edges, avg_t);

        let item = serde_json::json!({
            "method": m,
            "avg_time_ms": avg_t,
            "min_time_ms": min_t,
            "max_time_ms": max_t,
            "mteps": mteps,
            "traversed_edges": lr.traversed_edges,
            "visited_vertices": lr.visited_vertices,
        });

        let js_item = serde_wasm_bindgen::to_value(&item)?;
        results.push(&js_item);
    }

    Ok(results)
}
