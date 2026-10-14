//! `GpuBfsEngine` implementation of `webgpu_bfs_core::BfsEngine`.

use anyhow::{bail, Result};
use std::sync::Arc;
use webgpu_bfs_core::{BfsEngine, CsrGraph};

use crate::{
    context::GpuContext,
    dispatch::{
        batched_spmm::run_batched_spmm, bit_parallel_32::run_bit_parallel_32,
        bit_parallel_64::run_bit_parallel_64, bit_parallel_64_pull::run_bit_parallel_64_pull,
        boolean_reachability::run_boolean_reachability, indirect::run_indirect_macro_batch,
        indirect_sparse_push::{
            run_indirect_sparse_push_into, run_indirect_sparse_push_pooled,
            PersistentSparsePushBuffers,
        },
        multi_word_bp::{
            run_multi_word_bp_into, run_multi_word_bp_pooled,
            PersistentPullBuffers512,
        },
        multi_word_push::run_multi_word_bp_push,
        naive_seq::run_naive_sequential, push_pull::run_push_pull_adaptive,
        push_pull::PushPullConfig, sparse_bp::run_sparse_bp_64, sparse_queue::run_sparse_queue,
    },
    pipeline::PipelineCache,
};

/// Method selector for GPU BFS execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuBfsMethod {
    #[default]
    NaiveSequential,
    BatchedDenseSpMM,
    BitParallel32,
    BitParallel64,
    BitParallel64Pull,
    BooleanReachability,
    PushPullAdaptive,
    SparseQueue,
    SparseBitParallel64,
    MultiWordBitParallel,
    IndirectMacroBatch,
    /// Milestone M12: Fused Indirect Sparse Queue Push for directed graphs (S=64).
    IndirectSparseQueuePush,
    /// Milestone M12: Multi-Word Bit-Parallel Push for massive batches (S >= 128..2048).
    MultiWordBitParallelPush,
    /// Milestone M17: Topology-aware automatic method selection.
    Auto,
}

/// GPU Multi-Source BFS engine implementing `webgpu_bfs_core::BfsEngine`.
pub struct GpuBfsEngine {
    context: Arc<GpuContext>,
    pipelines: Arc<PipelineCache>,
    method: GpuBfsMethod,
    persistent_isqp: std::sync::Mutex<Option<PersistentSparsePushBuffers>>,
    persistent_mwp: std::sync::Mutex<Option<PersistentPullBuffers512>>,
}

impl GpuBfsEngine {
    /// Creates a new `GpuBfsEngine` with the given context and method.
    pub fn new(context: Arc<GpuContext>, method: GpuBfsMethod) -> Result<Self> {
        let pipelines = Arc::new(PipelineCache::new(context.device())?);
        Ok(Self {
            context,
            pipelines,
            method,
            persistent_isqp: std::sync::Mutex::new(None),
            persistent_mwp: std::sync::Mutex::new(None),
        })
    }

    /// Initializes a default GPU context and creates an engine with the specified method.
    pub fn default_with_method(method: GpuBfsMethod) -> Result<Self> {
        let context = Arc::new(GpuContext::init_blocking()?);
        Self::new(context, method)
    }

    /// Creates an engine using the `NaiveSequential` method.
    pub fn naive(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::NaiveSequential)
    }

    /// Creates an engine using the `BatchedDenseSpMM` method.
    pub fn batched(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::BatchedDenseSpMM)
    }

    /// Creates an engine using the `BitParallel32` method.
    pub fn bit_parallel_32(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::BitParallel32)
    }

    /// Creates an engine using the `BitParallel64` method.
    pub fn bit_parallel_64(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::BitParallel64)
    }

    /// Creates an engine using the `BooleanReachability` method.
    pub fn boolean_reachability(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::BooleanReachability)
    }

    /// Creates an engine using the `PushPullAdaptive` method.
    pub fn push_pull_adaptive(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::PushPullAdaptive)
    }

    /// Creates an engine using the `SparseQueue` method.
    pub fn sparse_queue(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::SparseQueue)
    }

    /// Creates an engine using the `SparseBitParallel64` method.
    pub fn sparse_bit_parallel_64(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::SparseBitParallel64)
    }

    /// Creates an engine using the `MultiWordBitParallel` method.
    pub fn multi_word_bit_parallel(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::MultiWordBitParallel)
    }

    /// Creates an engine using the `IndirectMacroBatch` method.
    pub fn indirect_macro_batch(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::IndirectMacroBatch)
    }

    /// Creates an engine using the `IndirectSparseQueuePush` method (Milestone M12).
    pub fn indirect_sparse_queue_push(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::IndirectSparseQueuePush)
    }

    /// Creates an engine using the `MultiWordBitParallelPush` method (Milestone M12).
    pub fn multi_word_bit_parallel_push(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::MultiWordBitParallelPush)
    }

    /// Creates an engine with topology-aware automatic method selection (Milestone M17).
    pub fn auto(context: Arc<GpuContext>) -> Result<Self> {
        Self::new(context, GpuBfsMethod::Auto)
    }

    /// Selects the optimal BFS method based on graph degree distribution and batch size.
    pub fn select_auto_method(graph: &CsrGraph, num_sources: usize) -> GpuBfsMethod {
        if graph.is_empty() || num_sources == 0 {
            return GpuBfsMethod::NaiveSequential;
        }

        let max_degree = graph.max_degree();

        if max_degree <= 32 {
            // Branch 1: Degree-uniform road networks, planar meshes, and regular trees
            if num_sources >= 64 {
                GpuBfsMethod::MultiWordBitParallel
            } else if num_sources >= 32 {
                GpuBfsMethod::BitParallel64
            } else if num_sources > 1 {
                GpuBfsMethod::BitParallel32
            } else {
                GpuBfsMethod::SparseQueue
            }
        } else if max_degree >= 128 {
            // Branch 2: Scale-free power-law graphs with massive hubs
            // M22: Pooled IndirectSparseQueuePush with exact reset and M20 hub optimizations
            // is strictly superior across all batch sizes (S >= 1).
            GpuBfsMethod::IndirectSparseQueuePush
        } else {
            // Branch 3: Intermediate density graphs (32 < max_degree < 128)
            if num_sources >= 64 {
                GpuBfsMethod::IndirectSparseQueuePush
            } else if num_sources >= 32 {
                GpuBfsMethod::BitParallel64
            } else if num_sources > 1 {
                GpuBfsMethod::BitParallel32
            } else {
                GpuBfsMethod::SparseQueue
            }
        }
    }

    /// Resolves the concrete execution method, expanding `Auto` if configured.
    pub fn resolve_method(&self, graph: &CsrGraph, sources: &[u32]) -> GpuBfsMethod {
        match self.method {
            GpuBfsMethod::Auto => Self::select_auto_method(graph, sources.len()),
            other => other,
        }
    }

    /// Returns the active execution method.
    pub fn method(&self) -> GpuBfsMethod {
        self.method
    }

    /// Returns a reference to the underlying `GpuContext`.
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Executes MS-BFS directly into a caller-provided reusable destination buffer.
    pub fn run_ms_bfs_into(
        &self,
        graph: &CsrGraph,
        sources: &[u32],
        out: &mut Vec<u32>,
    ) -> Result<()> {
        let num_vertices = graph.num_vertices;
        let num_sources = sources.len();

        if num_sources == 0 {
            out.clear();
            return Ok(());
        }

        if num_vertices == 0 {
            bail!(
                "Source vertex {} at index 0 exceeds graph vertex count 0",
                sources[0]
            );
        }

        for (i, &s) in sources.iter().enumerate() {
            if s as usize >= num_vertices {
                bail!(
                    "Source vertex {} at index {} exceeds graph vertex count {}",
                    s,
                    i,
                    num_vertices
                );
            }
        }

        let total = num_sources * num_vertices;
        if out.len() != total {
            out.resize(total, u32::MAX);
        }

        let effective_method = self.resolve_method(graph, sources);
        match effective_method {
            GpuBfsMethod::IndirectSparseQueuePush => {
                let mut guard = self.persistent_isqp.lock().unwrap();
                run_indirect_sparse_push_into(
                    &self.context,
                    &self.pipelines,
                    graph,
                    sources,
                    Some(&mut *guard),
                    out,
                )
            }
            GpuBfsMethod::MultiWordBitParallel => {
                let mut guard = self.persistent_mwp.lock().unwrap();
                run_multi_word_bp_into(
                    &self.context,
                    &self.pipelines,
                    graph,
                    sources,
                    Some(&mut *guard),
                    out,
                )
            }
            other => {
                let res = self.run_ms_bfs_with_method(other, graph, sources)?;
                out.copy_from_slice(&res);
                Ok(())
            }
        }
    }

    /// Helper method to execute BFS using a resolved method.
    pub fn run_ms_bfs_with_method(
        &self,
        method: GpuBfsMethod,
        graph: &CsrGraph,
        sources: &[u32],
    ) -> Result<Vec<u32>> {
        let num_vertices = graph.num_vertices;
        let num_sources = sources.len();

        if num_sources == 0 {
            return Ok(Vec::new());
        }

        if num_vertices == 0 {
            bail!(
                "Source vertex {} at index 0 exceeds graph vertex count 0",
                sources[0]
            );
        }

        for (i, &s) in sources.iter().enumerate() {
            if s as usize >= num_vertices {
                bail!(
                    "Source vertex {} at index {} exceeds graph vertex count {}",
                    s,
                    i,
                    num_vertices
                );
            }
        }

        match method {
            GpuBfsMethod::Auto => {
                let resolved = Self::select_auto_method(graph, num_sources);
                self.run_ms_bfs_with_method(resolved, graph, sources)
            }
            GpuBfsMethod::NaiveSequential => {
                run_naive_sequential(&self.context, &self.pipelines, graph, sources)
            }
            GpuBfsMethod::BatchedDenseSpMM => {
                run_batched_spmm(&self.context, &self.pipelines, graph, sources)
            }
            GpuBfsMethod::BitParallel32 => {
                run_bit_parallel_32(&self.context, &self.pipelines, graph, sources)
            }
            GpuBfsMethod::BitParallel64 => {
                run_bit_parallel_64(&self.context, &self.pipelines, graph, sources)
            }
            GpuBfsMethod::BitParallel64Pull => {
                run_bit_parallel_64_pull(&self.context, &self.pipelines, graph, sources)
            }
            GpuBfsMethod::BooleanReachability => {
                run_boolean_reachability(&self.context, &self.pipelines, graph, sources)
            }
            GpuBfsMethod::PushPullAdaptive => run_push_pull_adaptive(
                &self.context,
                &self.pipelines,
                graph,
                sources,
                PushPullConfig::default(),
            ),
            GpuBfsMethod::SparseQueue => {
                run_sparse_queue(&self.context, &self.pipelines, graph, sources)
            }
            GpuBfsMethod::SparseBitParallel64 => {
                run_sparse_bp_64(&self.context, &self.pipelines, graph, sources)
            }
            GpuBfsMethod::MultiWordBitParallel => {
                let mut guard = self.persistent_mwp.lock().unwrap();
                run_multi_word_bp_pooled(
                    &self.context,
                    &self.pipelines,
                    graph,
                    sources,
                    Some(&mut *guard),
                )
            }
            GpuBfsMethod::IndirectMacroBatch => {
                run_indirect_macro_batch(&self.context, &self.pipelines, graph, sources)
            }
            GpuBfsMethod::IndirectSparseQueuePush => {
                let mut guard = self.persistent_isqp.lock().unwrap();
                run_indirect_sparse_push_pooled(
                    &self.context,
                    &self.pipelines,
                    graph,
                    sources,
                    Some(&mut *guard),
                )
            }
            GpuBfsMethod::MultiWordBitParallelPush => {
                run_multi_word_bp_push(&self.context, &self.pipelines, graph, sources)
            }
        }
    }
}

impl BfsEngine for GpuBfsEngine {
    fn run_ms_bfs(&self, graph: &CsrGraph, sources: &[u32]) -> Result<Vec<u32>> {
        let effective_method = self.resolve_method(graph, sources);
        self.run_ms_bfs_with_method(effective_method, graph, sources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph_with_max_degree(max_deg: usize) -> CsrGraph {
        let nv = max_deg + 2;
        let edges: Vec<(u32, u32)> = (1..=max_deg as u32).map(|v| (0, v)).collect();
        CsrGraph::from_edge_list(nv, &edges)
    }

    #[test]
    fn test_select_auto_method_empty_and_zero_sources() {
        let empty = CsrGraph::from_edge_list(0, &[]);
        assert_eq!(
            GpuBfsEngine::select_auto_method(&empty, 64),
            GpuBfsMethod::NaiveSequential
        );

        let g = make_graph_with_max_degree(5);
        assert_eq!(
            GpuBfsEngine::select_auto_method(&g, 0),
            GpuBfsMethod::NaiveSequential
        );
    }

    #[test]
    fn test_select_auto_method_low_degree_graphs() {
        // max_degree <= 32: road networks (e.g. luxembourg_osm has max_degree 7), planar grids, trees
        for &deg in &[0, 1, 4, 7, 16, 32] {
            let g = make_graph_with_max_degree(deg);
            assert_eq!(g.max_degree(), deg);

            // S = 1 -> SparseQueue
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 1),
                GpuBfsMethod::SparseQueue
            );

            // 1 < S < 32 -> BitParallel32
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 2),
                GpuBfsMethod::BitParallel32
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 16),
                GpuBfsMethod::BitParallel32
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 31),
                GpuBfsMethod::BitParallel32
            );

            // 32 <= S < 64 -> BitParallel64
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 32),
                GpuBfsMethod::BitParallel64
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 63),
                GpuBfsMethod::BitParallel64
            );

            // S >= 64 -> MultiWordBitParallel
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 64),
                GpuBfsMethod::MultiWordBitParallel
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 256),
                GpuBfsMethod::MultiWordBitParallel
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 512),
                GpuBfsMethod::MultiWordBitParallel
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 1024),
                GpuBfsMethod::MultiWordBitParallel
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 2048),
                GpuBfsMethod::MultiWordBitParallel
            );
        }
    }

    #[test]
    fn test_select_auto_method_intermediate_degree_graphs() {
        // 32 < max_degree < 128
        for &deg in &[33, 50, 64, 100, 127] {
            let g = make_graph_with_max_degree(deg);
            assert_eq!(g.max_degree(), deg);

            // S = 1 -> SparseQueue
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 1),
                GpuBfsMethod::SparseQueue
            );

            // 1 < S < 32 -> BitParallel32
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 16),
                GpuBfsMethod::BitParallel32
            );

            // 32 <= S < 64 -> BitParallel64
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 32),
                GpuBfsMethod::BitParallel64
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 63),
                GpuBfsMethod::BitParallel64
            );

            // S >= 64 -> IndirectSparseQueuePush
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 64),
                GpuBfsMethod::IndirectSparseQueuePush
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 512),
                GpuBfsMethod::IndirectSparseQueuePush
            );
            assert_eq!(
                GpuBfsEngine::select_auto_method(&g, 2048),
                GpuBfsMethod::IndirectSparseQueuePush
            );
        }
    }

    #[test]
    fn test_select_auto_method_high_degree_hub_graphs() {
        // max_degree >= 128: web graphs (web-Stanford), social graphs, star hubs
        for &deg in &[128, 256, 1024, 38625] {
            let g = make_graph_with_max_degree(deg);
            assert_eq!(g.max_degree(), deg);

            // M22: Scale-free power-law graphs with massive hubs route all S >= 1 to IndirectSparseQueuePush
            for &s in &[1, 2, 16, 32, 63, 64, 128, 256, 384, 512, 1024, 2048] {
                assert_eq!(
                    GpuBfsEngine::select_auto_method(&g, s),
                    GpuBfsMethod::IndirectSparseQueuePush,
                    "Failed for max_degree {} and S {}",
                    deg,
                    s
                );
            }
        }
    }
}

