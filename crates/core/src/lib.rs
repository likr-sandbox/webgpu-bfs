//! High-performance graph representations, streaming Matrix Market parser,
//! and procedural graph generators for WebGPU Multi-Source BFS.

pub mod error;
pub mod generators;
pub mod graph;
pub mod mtx;

pub use error::{GeneratorError, GraphError, MtxError};
pub use generators::{
    clique_graph, cycle_graph, disconnected_cliques, empty_graph, generate_erdos_renyi,
    generate_erdos_renyi_nm, generate_erdos_renyi_np, generate_graph500_rmat, generate_grid2d,
    generate_random_tree, generate_rmat, isolated_vertices, path_graph, single_vertex, star_graph,
    Grid2D,
};
pub use graph::{CscGraph, CsrGraph, ReachabilityBitmask};
pub use mtx::{MtxField, MtxFormat, MtxHeader, MtxReader, MtxSymmetry};

/// Unified interface for multi-source breadth-first search execution.
#[cfg(not(target_arch = "wasm32"))]
pub trait BfsEngine: Send + Sync {
    /// Executes multi-source BFS for a slice of source vertex indices.
    ///
    /// # Arguments
    /// * `graph` - Compressed Sparse Row (CSR) graph reference.
    /// * `sources` - Slice of source vertex IDs (0-based indices).
    ///
    /// # Returns
    /// A flat `Vec<u32>` of length `sources.len() * graph.num_vertices`
    /// indexed by `dist[s * graph.num_vertices + v]`.
    ///
    /// # Invariants
    /// * Source vertices have distance 0: `dist[s * |V| + s] == 0`.
    /// * Unreachable vertices are set to `u32::MAX` (0xFFFF_FFFF).
    ///
    /// # Errors
    /// Returns `Err` if any source in `sources` is >= `graph.num_vertices`.
    fn run_ms_bfs(&self, graph: &CsrGraph, sources: &[u32]) -> anyhow::Result<Vec<u32>>;
}

/// Unified interface for multi-source breadth-first search execution.
#[cfg(target_arch = "wasm32")]
pub trait BfsEngine {
    /// Executes multi-source BFS for a slice of source vertex indices.
    ///
    /// # Arguments
    /// * `graph` - Compressed Sparse Row (CSR) graph reference.
    /// * `sources` - Slice of source vertex IDs (0-based indices).
    ///
    /// # Returns
    /// A flat `Vec<u32>` of length `sources.len() * graph.num_vertices`
    /// indexed by `dist[s * graph.num_vertices + v]`.
    ///
    /// # Invariants
    /// * Source vertices have distance 0: `dist[s * |V| + s] == 0`.
    /// * Unreachable vertices are set to `u32::MAX` (0xFFFF_FFFF).
    ///
    /// # Errors
    /// Returns `Err` if any source in `sources` is >= `graph.num_vertices`.
    fn run_ms_bfs(&self, graph: &CsrGraph, sources: &[u32]) -> anyhow::Result<Vec<u32>>;
}

/// Unified trait for multi-source reachability engines returning compact bitmasks.
#[cfg(not(target_arch = "wasm32"))]
pub trait ReachabilityEngine: Send + Sync {
    /// Executes multi-source reachability query for a slice of source vertex indices.
    ///
    /// # Arguments
    /// * `graph` - Compressed Sparse Row (CSR) graph reference.
    /// * `sources` - Slice of source vertex IDs (0-based indices).
    ///
    /// # Returns
    /// A `ReachabilityBitmask` storing reachability bits in vertex-major order:
    /// `words[v * words_per_vertex + s / 32] & (1 << (s % 32)) != 0`.
    fn run_reachability(&self, graph: &CsrGraph, sources: &[u32]) -> anyhow::Result<ReachabilityBitmask>;

    /// Executes multi-source reachability directly into a caller-supplied pre-allocated buffer.
    fn run_reachability_into(
        &self,
        graph: &CsrGraph,
        sources: &[u32],
        out: &mut ReachabilityBitmask,
    ) -> anyhow::Result<()> {
        let res = self.run_reachability(graph, sources)?;
        *out = res;
        Ok(())
    }
}

/// Unified trait for multi-source reachability engines returning compact bitmasks.
#[cfg(target_arch = "wasm32")]
pub trait ReachabilityEngine {
    /// Executes multi-source reachability query for a slice of source vertex indices.
    ///
    /// # Arguments
    /// * `graph` - Compressed Sparse Row (CSR) graph reference.
    /// * `sources` - Slice of source vertex IDs (0-based indices).
    ///
    /// # Returns
    /// A `ReachabilityBitmask` storing reachability bits in vertex-major order:
    /// `words[v * words_per_vertex + s / 32] & (1 << (s % 32)) != 0`.
    fn run_reachability(&self, graph: &CsrGraph, sources: &[u32]) -> anyhow::Result<ReachabilityBitmask>;

    /// Executes multi-source reachability directly into a caller-supplied pre-allocated buffer.
    fn run_reachability_into(
        &self,
        graph: &CsrGraph,
        sources: &[u32],
        out: &mut ReachabilityBitmask,
    ) -> anyhow::Result<()> {
        let res = self.run_reachability(graph, sources)?;
        *out = res;
        Ok(())
    }
}
