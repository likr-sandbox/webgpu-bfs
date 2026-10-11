//! WASM Graph handle and generators.

use std::io::Cursor;
use wasm_bindgen::prelude::*;
use webgpu_bfs_core::{
    generate_erdos_renyi_nm, generate_graph500_rmat, generate_grid2d, generate_random_tree,
    CsrGraph,
};

#[wasm_bindgen]
pub struct WasmGraph {
    pub(crate) inner: CsrGraph,
}

#[wasm_bindgen]
impl WasmGraph {
    /// Constructs a graph from raw CSR arrays.
    #[wasm_bindgen(constructor)]
    pub fn new(
        num_vertices: usize,
        row_offsets: &[u32],
        col_indices: &[u32],
    ) -> Result<WasmGraph, JsValue> {
        let graph = CsrGraph {
            num_vertices,
            num_edges: col_indices.len(),
            row_offsets: row_offsets.to_vec(),
            col_indices: col_indices.to_vec(),
        };
        Ok(Self { inner: graph })
    }

    /// Parses a Matrix Market (.mtx) text string into a CSR graph.
    pub fn from_mtx(mtx_content: &str) -> Result<WasmGraph, JsValue> {
        let cursor = Cursor::new(mtx_content.as_bytes());
        let graph = CsrGraph::from_mtx_reader(cursor)
            .map_err(|e| JsValue::from_str(&format!("MTX parse error: {:#}", e)))?;
        Ok(Self { inner: graph })
    }

    /// Generates a 2D Grid graph of dimension `width` x `height`.
    pub fn generate_grid2d(width: u32, height: u32) -> Result<WasmGraph, JsValue> {
        let graph = generate_grid2d(width as usize, height as usize);
        Ok(Self { inner: graph })
    }

    /// Generates an Erdős-Rényi random graph G(n, m).
    pub fn generate_erdos_renyi(
        num_vertices: usize,
        num_edges: usize,
        seed: Option<f64>,
    ) -> Result<WasmGraph, JsValue> {
        let s = seed.map(|v| v as u64);
        let graph = generate_erdos_renyi_nm(num_vertices, num_edges, s)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner: graph })
    }

    /// Generates a uniform random tree with `num_vertices` vertices.
    pub fn generate_random_tree(
        num_vertices: usize,
        seed: Option<f64>,
    ) -> Result<WasmGraph, JsValue> {
        let s = seed.map(|v| v as u64);
        let graph =
            generate_random_tree(num_vertices, s).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner: graph })
    }

    /// Generates an R-MAT power-law graph.
    pub fn generate_rmat(
        scale: u32,
        edge_factor: u32,
        seed: Option<f64>,
    ) -> Result<WasmGraph, JsValue> {
        let s = seed.map(|v| v as u64);
        let graph = generate_graph500_rmat(scale as usize, edge_factor as usize, s)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner: graph })
    }

    pub fn num_vertices(&self) -> usize {
        self.inner.num_vertices
    }

    pub fn num_edges(&self) -> usize {
        self.inner.num_edges
    }

    pub fn row_offsets(&self) -> js_sys::Uint32Array {
        let arr = js_sys::Uint32Array::new_with_length(self.inner.row_offsets.len() as u32);
        arr.copy_from(&self.inner.row_offsets);
        arr
    }

    pub fn col_indices(&self) -> js_sys::Uint32Array {
        let arr = js_sys::Uint32Array::new_with_length(self.inner.col_indices.len() as u32);
        arr.copy_from(&self.inner.col_indices);
        arr
    }

    pub fn out_degree(&self, v: u32) -> u32 {
        self.inner.degree(v as usize) as u32
    }

    pub fn is_symmetric(&self) -> bool {
        self.inner.is_symmetric()
    }

    /// Serializes graph topology (node list and edge list) as JSON for canvas rendering.
    /// Limits export up to `max_vertices` to protect canvas performance.
    pub fn get_topology_json(&self, max_vertices: Option<usize>) -> Result<String, JsValue> {
        let max_v = max_vertices.unwrap_or(200).min(self.inner.num_vertices);
        let mut edges = Vec::new();

        for u in 0..max_v {
            let neighbors = self.inner.neighbors(u);
            for &v in neighbors {
                if (v as usize) < max_v && u <= v as usize {
                    edges.push((u, v as usize));
                }
            }
        }

        let obj = serde_json::json!({
            "num_vertices": self.inner.num_vertices,
            "num_edges": self.inner.num_edges,
            "sampled_vertices": max_v,
            "edges": edges,
        });

        serde_json::to_string(&obj).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
