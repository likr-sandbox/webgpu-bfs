//! Return structs, metrics, and ground-truth verification types.

use wasm_bindgen::prelude::*;
use webgpu_bfs_core::CsrGraph;

#[wasm_bindgen]
pub struct WasmBfsResult {
    pub(crate) distances: Vec<u32>,
    pub num_vertices: usize,
    pub num_sources: usize,
    pub elapsed_ms: f64,
    pub mteps: f64,
    pub traversed_edges: usize,
    pub visited_vertices: usize,
    method: String,
}

#[wasm_bindgen]
impl WasmBfsResult {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        distances: Vec<u32>,
        num_vertices: usize,
        num_sources: usize,
        elapsed_ms: f64,
        mteps: f64,
        traversed_edges: usize,
        visited_vertices: usize,
        method: String,
    ) -> Self {
        Self {
            distances,
            num_vertices,
            num_sources,
            elapsed_ms,
            mteps,
            traversed_edges,
            visited_vertices,
            method,
        }
    }

    /// Returns the distance array as a newly copied `Uint32Array`.
    pub fn distances(&self) -> js_sys::Uint32Array {
        let arr = js_sys::Uint32Array::new_with_length(self.distances.len() as u32);
        arr.copy_from(&self.distances);
        arr
    }

    pub fn method(&self) -> String {
        self.method.clone()
    }

    pub fn distance_of(&self, source_idx: usize, vertex: u32) -> u32 {
        if source_idx < self.num_sources && (vertex as usize) < self.num_vertices {
            self.distances[source_idx * self.num_vertices + vertex as usize]
        } else {
            u32::MAX
        }
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        let obj = serde_json::json!({
            "num_vertices": self.num_vertices,
            "num_sources": self.num_sources,
            "elapsed_ms": self.elapsed_ms,
            "mteps": self.mteps,
            "traversed_edges": self.traversed_edges,
            "visited_vertices": self.visited_vertices,
            "method": self.method,
        });
        serde_json::to_string(&obj).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[wasm_bindgen]
pub struct WasmVerifyResult {
    pub is_equivalent: bool,
    pub total_pairs: usize,
    pub mismatched_pairs: usize,
    pub sample_mismatch_source: Option<u32>,
    pub sample_mismatch_vertex: Option<u32>,
    pub sample_gpu_dist: Option<u32>,
    pub sample_cpu_dist: Option<u32>,
    message: String,
}

#[wasm_bindgen]
impl WasmVerifyResult {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        is_equivalent: bool,
        total_pairs: usize,
        mismatched_pairs: usize,
        sample_mismatch_source: Option<u32>,
        sample_mismatch_vertex: Option<u32>,
        sample_gpu_dist: Option<u32>,
        sample_cpu_dist: Option<u32>,
        message: String,
    ) -> Self {
        Self {
            is_equivalent,
            total_pairs,
            mismatched_pairs,
            sample_mismatch_source,
            sample_mismatch_vertex,
            sample_gpu_dist,
            sample_cpu_dist,
            message,
        }
    }

    pub fn message(&self) -> String {
        self.message.clone()
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        let obj = serde_json::json!({
            "is_equivalent": self.is_equivalent,
            "total_pairs": self.total_pairs,
            "mismatched_pairs": self.mismatched_pairs,
            "sample_mismatch_source": self.sample_mismatch_source,
            "sample_mismatch_vertex": self.sample_mismatch_vertex,
            "sample_gpu_dist": self.sample_gpu_dist,
            "sample_cpu_dist": self.sample_cpu_dist,
            "message": self.message,
        });
        serde_json::to_string(&obj).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

pub(crate) fn calc_mteps(traversed_edges: usize, elapsed_ms: f64) -> f64 {
    if elapsed_ms <= 0.0 {
        return 0.0;
    }
    (traversed_edges as f64) / (elapsed_ms * 1000.0)
}

pub(crate) fn count_traversed_edges(
    graph: &CsrGraph,
    distances: &[u32],
    num_sources: usize,
) -> usize {
    let nv = graph.num_vertices;
    let mut total = 0;
    for s in 0..num_sources {
        let base = s * nv;
        for v in 0..nv {
            if distances[base + v] != u32::MAX {
                total += graph.degree(v);
            }
        }
    }
    total
}

pub(crate) fn count_visited_vertices(distances: &[u32]) -> usize {
    distances.iter().filter(|&&d| d != u32::MAX).count()
}
