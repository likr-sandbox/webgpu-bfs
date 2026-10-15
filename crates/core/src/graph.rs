use crate::error::{GraphError, MtxError};
use std::io::BufRead;

/// Compressed Sparse Row (CSR) representation of an unweighted graph.
/// Ideal for top-down (Push) BFS traversal: u -> v.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CsrGraph {
    /// Total number of vertices in the graph (|V|).
    pub num_vertices: usize,
    /// Total number of directed edges in the graph (|E|).
    pub num_edges: usize,
    /// Starting offset of outgoing edges for each vertex in `col_indices`.
    /// Length: num_vertices + 1.
    /// Invariant: row_offsets[0] == 0, row_offsets[num_vertices] == num_edges as u32.
    pub row_offsets: Vec<u32>,
    /// Destination vertex indices for all directed edges.
    /// Length: num_edges.
    /// Invariant: all elements < num_vertices as u32.
    pub col_indices: Vec<u32>,
}

/// Compressed Sparse Column (CSC) representation of an unweighted graph.
/// Ideal for bottom-up (Pull) BFS traversal: v <- u.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CscGraph {
    /// Total number of vertices in the graph (|V|).
    pub num_vertices: usize,
    /// Total number of directed edges in the graph (|E|).
    pub num_edges: usize,
    /// Starting offset of incoming edges for each vertex in `row_indices`.
    /// Length: num_vertices + 1.
    /// Invariant: col_offsets[0] == 0, col_offsets[num_vertices] == num_edges as u32.
    pub col_offsets: Vec<u32>,
    /// Source vertex indices for all incoming edges.
    /// Length: num_edges.
    /// Invariant: all elements < num_vertices as u32.
    pub row_indices: Vec<u32>,
}

impl CsrGraph {
    /// Builds a canonical `CsrGraph` from a slice of directed edges `(u, v)`.
    ///
    /// Automatically:
    /// - Discards self-loops (u == v)
    /// - Discards out-of-bounds vertex endpoints (u >= num_vertices or v >= num_vertices)
    /// - Sorts edges by (source, destination)
    /// - Eliminates duplicate edges
    /// - Constructs canonical `row_offsets` and `col_indices`
    pub fn from_edge_list(num_vertices: usize, edges: &[(u32, u32)]) -> Self {
        let mut filtered_edges: Vec<(u32, u32)> = edges
            .iter()
            .copied()
            .filter(|&(u, v)| u != v && (u as usize) < num_vertices && (v as usize) < num_vertices)
            .collect();

        filtered_edges.sort_unstable();
        filtered_edges.dedup();

        let num_edges = filtered_edges.len();

        let mut row_offsets = vec![0u32; num_vertices + 1];
        for &(u, _) in &filtered_edges {
            row_offsets[u as usize + 1] += 1;
        }

        for i in 0..num_vertices {
            row_offsets[i + 1] += row_offsets[i];
        }

        let col_indices: Vec<u32> = filtered_edges.into_iter().map(|(_, v)| v).collect();

        Self {
            num_vertices,
            num_edges,
            row_offsets,
            col_indices,
        }
    }

    /// Builds a `CsrGraph` by streaming from a Matrix Market formatted reader.
    pub fn from_mtx_reader<R: BufRead>(reader: R) -> Result<Self, MtxError> {
        crate::mtx::MtxReader::parse(reader)
    }

    /// Transposes the CSR graph into CSC format using an O(|V| + |E|) counting sort.
    ///
    /// Predecessor lists in the resulting `CscGraph` are guaranteed to be sorted.
    pub fn to_csc(&self) -> CscGraph {
        let v = self.num_vertices;
        let e = self.num_edges;

        // Step 1: Count in-degrees
        let mut in_degrees = vec![0u32; v];
        for &col in &self.col_indices {
            in_degrees[col as usize] += 1;
        }

        // Step 2: Compute prefix sums into col_offsets
        let mut col_offsets = vec![0u32; v + 1];
        for i in 0..v {
            col_offsets[i + 1] = col_offsets[i] + in_degrees[i];
        }

        // Step 3: Scatter into row_indices (preserves sorted row order)
        let mut curr_pos = col_offsets[..v].to_vec();
        let mut row_indices = vec![0u32; e];

        for u in 0..v {
            let start = self.row_offsets[u] as usize;
            let end = self.row_offsets[u + 1] as usize;
            for &col in &self.col_indices[start..end] {
                let write_idx = curr_pos[col as usize] as usize;
                row_indices[write_idx] = u as u32;
                curr_pos[col as usize] += 1;
            }
        }

        CscGraph {
            num_vertices: v,
            num_edges: e,
            col_offsets,
            row_indices,
        }
    }

    /// Checks whether the graph is undirected (symmetric) in O(|V| + |E|) time.
    pub fn is_symmetric(&self) -> bool {
        let csc = self.to_csc();
        self.row_offsets == csc.col_offsets && self.col_indices == csc.row_indices
    }

    /// Converts the graph to an undirected (symmetric) graph by adding reverse edges for all directed edges.
    /// If the graph is already symmetric, returns a clone.
    pub fn to_undirected(&self) -> Self {
        if self.is_symmetric() {
            return self.clone();
        }
        let mut edges = Vec::with_capacity(self.num_edges * 2);
        for u in 0..self.num_vertices {
            let start = self.row_offsets[u] as usize;
            let end = self.row_offsets[u + 1] as usize;
            for &v in &self.col_indices[start..end] {
                edges.push((u as u32, v));
                edges.push((v, u as u32));
            }
        }
        Self::from_edge_list(self.num_vertices, &edges)
    }

    /// Returns the number of vertices |V| in the graph.
    #[inline(always)]
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Returns the number of directed edges |E| in the graph.
    #[inline(always)]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Returns true if the graph has 0 vertices.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.num_vertices == 0
    }

    /// Returns the out-degree of vertex `u`.
    /// Panics if `u >= num_vertices`.
    #[inline]
    pub fn degree(&self, u: usize) -> usize {
        assert!(
            u < self.num_vertices,
            "Vertex {} out of bounds for graph with {} vertices",
            u,
            self.num_vertices
        );
        (self.row_offsets[u + 1] - self.row_offsets[u]) as usize
    }

    /// Returns the out-degree of vertex `u`, or None if out of bounds.
    #[inline]
    pub fn get_degree(&self, u: usize) -> Option<usize> {
        if u < self.num_vertices {
            Some((self.row_offsets[u + 1] - self.row_offsets[u]) as usize)
        } else {
            None
        }
    }

    /// Returns the slice of outgoing neighbor vertex IDs for vertex `u`.
    /// Panics if `u >= num_vertices`.
    #[inline]
    pub fn neighbors(&self, u: usize) -> &[u32] {
        assert!(
            u < self.num_vertices,
            "Vertex {} out of bounds for graph with {} vertices",
            u,
            self.num_vertices
        );
        let start = self.row_offsets[u] as usize;
        let end = self.row_offsets[u + 1] as usize;
        &self.col_indices[start..end]
    }

    /// Returns the maximum out-degree across all vertices in the graph.
    pub fn max_degree(&self) -> usize {
        (0..self.num_vertices)
            .map(|u| self.degree(u))
            .max()
            .unwrap_or(0)
    }

    /// Returns the average out-degree across all vertices in the graph.
    pub fn avg_degree(&self) -> f64 {
        if self.num_vertices == 0 {
            0.0
        } else {
            self.num_edges as f64 / self.num_vertices as f64
        }
    }

    /// Validates all structural invariants of the CSR graph.
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.row_offsets.len() != self.num_vertices + 1 {
            return Err(GraphError::InvalidRowOffsetsLength {
                expected: self.num_vertices + 1,
                actual: self.row_offsets.len(),
            });
        }
        if self.row_offsets[0] != 0 {
            return Err(GraphError::NonZeroStartOffset(self.row_offsets[0]));
        }
        for i in 0..self.num_vertices {
            let curr = self.row_offsets[i];
            let next = self.row_offsets[i + 1];
            if curr > next {
                return Err(GraphError::NonMonotonicOffsets {
                    index: i,
                    curr,
                    next_index: i + 1,
                    next,
                });
            }
        }
        let last_offset = self.row_offsets[self.num_vertices] as usize;
        if last_offset != self.col_indices.len() || last_offset != self.num_edges {
            return Err(GraphError::EdgeCountMismatch {
                last_offset: self.row_offsets[self.num_vertices],
                col_indices_len: self.col_indices.len(),
            });
        }
        for (edge_idx, &col) in self.col_indices.iter().enumerate() {
            if col as usize >= self.num_vertices {
                return Err(GraphError::ColumnIndexOutOfBounds {
                    col,
                    edge_idx,
                    num_vertices: self.num_vertices,
                });
            }
        }
        // Validate sorted and unique neighbor lists
        for u in 0..self.num_vertices {
            let neighbors = self.neighbors(u);
            for w in neighbors.windows(2) {
                if w[0] >= w[1] {
                    return Err(GraphError::UnsortedOrDuplicateNeighbors {
                        vertex: u,
                        first: w[0],
                        second: w[1],
                    });
                }
            }
        }
        Ok(())
    }
}

impl CscGraph {
    /// Transposes the CSC graph back into CSR format using an O(|V| + |E|) counting sort.
    pub fn to_csr(&self) -> CsrGraph {
        let v = self.num_vertices;
        let e = self.num_edges;

        let mut out_degrees = vec![0u32; v];
        for &row in &self.row_indices {
            out_degrees[row as usize] += 1;
        }

        let mut row_offsets = vec![0u32; v + 1];
        for i in 0..v {
            row_offsets[i + 1] = row_offsets[i] + out_degrees[i];
        }

        let mut curr_pos = row_offsets[..v].to_vec();
        let mut col_indices = vec![0u32; e];

        for col in 0..v {
            let start = self.col_offsets[col] as usize;
            let end = self.col_offsets[col + 1] as usize;
            for &row in &self.row_indices[start..end] {
                let write_idx = curr_pos[row as usize] as usize;
                col_indices[write_idx] = col as u32;
                curr_pos[row as usize] += 1;
            }
        }

        CsrGraph {
            num_vertices: v,
            num_edges: e,
            row_offsets,
            col_indices,
        }
    }

    /// Returns the number of vertices |V| in the graph.
    #[inline(always)]
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Returns the number of directed edges |E| in the graph.
    #[inline(always)]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Returns true if the graph has 0 vertices.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.num_vertices == 0
    }

    /// Returns the in-degree of vertex `v`.
    /// Panics if `v >= num_vertices`.
    #[inline]
    pub fn in_degree(&self, v: usize) -> usize {
        assert!(
            v < self.num_vertices,
            "Vertex {} out of bounds for graph with {} vertices",
            v,
            self.num_vertices
        );
        (self.col_offsets[v + 1] - self.col_offsets[v]) as usize
    }

    /// Returns the in-degree of vertex `v`, or None if out of bounds.
    #[inline]
    pub fn get_in_degree(&self, v: usize) -> Option<usize> {
        if v < self.num_vertices {
            Some((self.col_offsets[v + 1] - self.col_offsets[v]) as usize)
        } else {
            None
        }
    }

    /// Returns the slice of incoming predecessor vertex IDs for vertex `v`.
    /// Panics if `v >= num_vertices`.
    #[inline]
    pub fn predecessors(&self, v: usize) -> &[u32] {
        assert!(
            v < self.num_vertices,
            "Vertex {} out of bounds for graph with {} vertices",
            v,
            self.num_vertices
        );
        let start = self.col_offsets[v] as usize;
        let end = self.col_offsets[v + 1] as usize;
        &self.row_indices[start..end]
    }

    /// Returns the maximum in-degree across all vertices in the graph.
    pub fn max_in_degree(&self) -> usize {
        (0..self.num_vertices)
            .map(|v| self.in_degree(v))
            .max()
            .unwrap_or(0)
    }

    /// Validates all structural invariants of the CSC graph.
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.col_offsets.len() != self.num_vertices + 1 {
            return Err(GraphError::InvalidRowOffsetsLength {
                expected: self.num_vertices + 1,
                actual: self.col_offsets.len(),
            });
        }
        if self.col_offsets[0] != 0 {
            return Err(GraphError::NonZeroStartOffset(self.col_offsets[0]));
        }
        for i in 0..self.num_vertices {
            let curr = self.col_offsets[i];
            let next = self.col_offsets[i + 1];
            if curr > next {
                return Err(GraphError::NonMonotonicOffsets {
                    index: i,
                    curr,
                    next_index: i + 1,
                    next,
                });
            }
        }
        let last_offset = self.col_offsets[self.num_vertices] as usize;
        if last_offset != self.row_indices.len() || last_offset != self.num_edges {
            return Err(GraphError::EdgeCountMismatch {
                last_offset: self.col_offsets[self.num_vertices],
                col_indices_len: self.row_indices.len(),
            });
        }
        for (edge_idx, &row) in self.row_indices.iter().enumerate() {
            if row as usize >= self.num_vertices {
                return Err(GraphError::ColumnIndexOutOfBounds {
                    col: row,
                    edge_idx,
                    num_vertices: self.num_vertices,
                });
            }
        }
        for v in 0..self.num_vertices {
            let preds = self.predecessors(v);
            for w in preds.windows(2) {
                if w[0] >= w[1] {
                    return Err(GraphError::UnsortedOrDuplicateNeighbors {
                        vertex: v,
                        first: w[0],
                        second: w[1],
                    });
                }
            }
        }
        Ok(())
    }
}

/// Compact, high-throughput reachability bitmask representing reachability from
/// multiple sources to all graph vertices.
///
/// Organized in vertex-major order for zero-copy direct DMA compatibility with
/// WebGPU bit-parallel kernels:
/// `word = words[v * words_per_vertex + (s / 32)]`
/// `is_reachable = (word & (1 << (s % 32))) != 0`
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReachabilityBitmask {
    pub num_vertices: usize,
    pub num_sources: usize,
    pub words_per_vertex: usize,
    pub words: Vec<u32>,
}

impl ReachabilityBitmask {
    /// Creates a new `ReachabilityBitmask` from pre-allocated word buffer.
    pub fn new(num_vertices: usize, num_sources: usize, words: Vec<u32>) -> Self {
        let words_per_vertex = if num_sources == 0 { 0 } else { (num_sources + 31) / 32 };
        assert_eq!(
            words.len(),
            num_vertices * words_per_vertex,
            "Words buffer length must exactly equal num_vertices * words_per_vertex"
        );
        Self {
            num_vertices,
            num_sources,
            words_per_vertex,
            words,
        }
    }

    /// Allocates an empty (all unreachable) bitmask.
    pub fn zeroed(num_vertices: usize, num_sources: usize) -> Self {
        let words_per_vertex = if num_sources == 0 { 0 } else { (num_sources + 31) / 32 };
        Self {
            num_vertices,
            num_sources,
            words_per_vertex,
            words: vec![0u32; num_vertices * words_per_vertex],
        }
    }

    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    #[inline]
    pub fn num_sources(&self) -> usize {
        self.num_sources
    }

    #[inline]
    pub fn words_per_vertex(&self) -> usize {
        self.words_per_vertex
    }

    /// Returns whether vertex `v` is reachable from source index `s_idx` (0-based in `0..num_sources`).
    #[inline]
    pub fn is_reachable(&self, s_idx: usize, v: usize) -> bool {
        if s_idx >= self.num_sources || v >= self.num_vertices || self.words_per_vertex == 0 {
            return false;
        }
        let w = s_idx / 32;
        let bit = 1u32 << (s_idx % 32);
        (self.words[v * self.words_per_vertex + w] & bit) != 0
    }

    /// Sets the reachability bit for source `s_idx` to vertex `v`.
    #[inline]
    pub fn set_reachable(&mut self, s_idx: usize, v: usize) {
        if s_idx < self.num_sources && v < self.num_vertices && self.words_per_vertex > 0 {
            let w = s_idx / 32;
            let bit = 1u32 << (s_idx % 32);
            self.words[v * self.words_per_vertex + w] |= bit;
        }
    }

    /// Returns a slice of the 32-bit reachability words for vertex `v`.
    #[inline]
    pub fn vertex_words(&self, v: usize) -> &[u32] {
        if v >= self.num_vertices || self.words_per_vertex == 0 {
            return &[];
        }
        let start = v * self.words_per_vertex;
        &self.words[start..start + self.words_per_vertex]
    }

    /// Returns the number of sources that can reach vertex `v`.
    #[inline]
    pub fn reachable_source_count(&self, v: usize) -> usize {
        let words = self.vertex_words(v);
        let rem = self.num_sources % 32;
        if rem == 0 || words.is_empty() {
            words.iter().map(|&w| w.count_ones() as usize).sum()
        } else {
            let last_idx = words.len() - 1;
            let full_count: usize = words[..last_idx].iter().map(|&w| w.count_ones() as usize).sum();
            let mask = (1u32 << rem) - 1;
            full_count + (words[last_idx] & mask).count_ones() as usize
        }
    }

    /// Returns total reachable pairs across all sources and vertices.
    pub fn total_reachable_pairs(&self) -> u64 {
        (0..self.num_vertices)
            .map(|v| self.reachable_source_count(v) as u64)
            .sum()
    }

    /// Raw word slice (length: `num_vertices * words_per_vertex`).
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }

    /// Mutable raw word slice.
    pub fn as_words_mut(&mut self) -> &mut [u32] {
        &mut self.words
    }

    /// Direct byte slice for zero-copy PCIe / DMA mapping.
    pub fn as_bytes(&self) -> &[u8] {
        let byte_len = self.words.len() * std::mem::size_of::<u32>();
        unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, byte_len) }
    }

    /// Constructs the ground-truth ReachabilityBitmask from a canonical distance array.
    ///
    /// Canonical distance invariant: `dist[s_idx * num_vertices + v] != u32::MAX <==> reachable`.
    ///
    /// Cache-optimized: inverts the loop order to stream contiguous vertex chunks in L2 cache,
    /// writing full 32-bit words without read-modify-write stalls, parallelized with Rayon on native targets.
    pub fn from_canonical_distances(
        num_vertices: usize,
        num_sources: usize,
        distances: &[u32],
    ) -> Self {
        let mut bitmask = Self::zeroed(num_vertices, num_sources);
        if num_sources == 0 || num_vertices == 0 {
            return bitmask;
        }
        let wpv = bitmask.words_per_vertex;
        assert!(
            distances.len() >= num_sources * num_vertices,
            "Distances buffer length {} is smaller than required {}",
            distances.len(),
            num_sources * num_vertices
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            const CHUNK_SIZE: usize = 1024;
            bitmask
                .words
                .par_chunks_mut(CHUNK_SIZE * wpv)
                .enumerate()
                .for_each(|(chunk_idx, chunk_words)| {
                    let v_start = chunk_idx * CHUNK_SIZE;
                    let v_count = chunk_words.len() / wpv;

                    for w in 0..wpv {
                        let s_start = w * 32;
                        let s_count = (num_sources - s_start).min(32);
                        let mut s_offsets = [0usize; 32];
                        for b in 0..s_count {
                            s_offsets[b] = (s_start + b) * num_vertices;
                        }

                        for local_v in 0..v_count {
                            let v = v_start + local_v;
                            let mut word = 0u32;
                            for b in 0..s_count {
                                if distances[s_offsets[b] + v] != u32::MAX {
                                    word |= 1u32 << b;
                                }
                            }
                            chunk_words[local_v * wpv + w] = word;
                        }
                    }
                });
        }

        #[cfg(target_arch = "wasm32")]
        {
            for w in 0..wpv {
                let s_start = w * 32;
                let s_count = (num_sources - s_start).min(32);
                let mut s_offsets = [0usize; 32];
                for b in 0..s_count {
                    s_offsets[b] = (s_start + b) * num_vertices;
                }

                for v in 0..num_vertices {
                    let mut word = 0u32;
                    for b in 0..s_count {
                        if distances[s_offsets[b] + v] != u32::MAX {
                            word |= 1u32 << b;
                        }
                    }
                    bitmask.words[v * wpv + w] = word;
                }
            }
        }

        bitmask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_construction_and_queries() {
        // Directed graph: 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 0
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 0)];
        let csr = CsrGraph::from_edge_list(3, &edges);

        assert_eq!(csr.num_vertices(), 3);
        assert_eq!(csr.num_edges(), 4);
        assert_eq!(csr.row_offsets, vec![0, 2, 3, 4]);
        assert_eq!(csr.col_indices, vec![1, 2, 2, 0]);

        assert_eq!(csr.degree(0), 2);
        assert_eq!(csr.degree(1), 1);
        assert_eq!(csr.degree(2), 1);
        assert_eq!(csr.max_degree(), 2);
        assert_eq!(csr.neighbors(0), &[1, 2]);
        assert_eq!(csr.neighbors(1), &[2]);
        assert_eq!(csr.neighbors(2), &[0]);

        assert!(csr.validate().is_ok());
    }

    #[test]
    fn test_self_loops_and_duplicates_filtered() {
        let edges = vec![
            (0, 0), // self-loop (should be removed)
            (0, 1),
            (0, 1), // duplicate (should be removed)
            (1, 2),
            (2, 2), // self-loop (should be removed)
        ];
        let csr = CsrGraph::from_edge_list(3, &edges);

        assert_eq!(csr.num_vertices, 3);
        assert_eq!(csr.num_edges, 2);
        assert_eq!(csr.neighbors(0), &[1]);
        assert_eq!(csr.neighbors(1), &[2]);
        assert_eq!(csr.neighbors(2), &[]);
        assert!(csr.validate().is_ok());
    }

    #[test]
    fn test_counting_sort_transposition() {
        // Directed graph: 0 -> 1, 0 -> 2, 1 -> 2
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let csr = CsrGraph::from_edge_list(3, &edges);
        let csc = csr.to_csc();

        assert_eq!(csc.num_vertices(), 3);
        assert_eq!(csc.num_edges(), 3);
        assert_eq!(csc.col_offsets, vec![0, 0, 1, 3]);
        assert_eq!(csc.row_indices, vec![0, 0, 1]); // vertex 2 has incoming from 0 and 1, sorted!

        assert_eq!(csc.in_degree(0), 0);
        assert_eq!(csc.in_degree(1), 1);
        assert_eq!(csc.in_degree(2), 2);
        assert_eq!(csc.predecessors(2), &[0, 1]);
        assert!(csc.validate().is_ok());

        // Roundtrip identity
        let csr_roundtrip = csc.to_csr();
        assert_eq!(csr, csr_roundtrip);
    }

    #[test]
    fn test_symmetry_check() {
        // Asymmetric directed graph
        let dir_edges = vec![(0, 1), (1, 2)];
        let dir_graph = CsrGraph::from_edge_list(3, &dir_edges);
        assert!(!dir_graph.is_symmetric());

        // Symmetric undirected graph: 0 <-> 1, 1 <-> 2
        let sym_edges = vec![(0, 1), (1, 0), (1, 2), (2, 1)];
        let sym_graph = CsrGraph::from_edge_list(3, &sym_edges);
        assert!(sym_graph.is_symmetric());
    }

    #[test]
    fn test_to_undirected() {
        let dir_edges = vec![(0, 1), (1, 2)];
        let dir_graph = CsrGraph::from_edge_list(3, &dir_edges);
        assert!(!dir_graph.is_symmetric());

        let undir_graph = dir_graph.to_undirected();
        assert!(undir_graph.is_symmetric());
        assert_eq!(undir_graph.num_vertices(), 3);
        assert_eq!(undir_graph.num_edges(), 4);
        assert_eq!(undir_graph.neighbors(0), &[1]);
        assert_eq!(undir_graph.neighbors(1), &[0, 2]);
        assert_eq!(undir_graph.neighbors(2), &[1]);
    }

    #[test]
    fn test_empty_and_disconnected_graphs() {
        let empty = CsrGraph::from_edge_list(0, &[]);
        assert!(empty.is_empty());
        assert_eq!(empty.max_degree(), 0);
        assert!(empty.validate().is_ok());

        // 5 vertices, 0 edges
        let isolated = CsrGraph::from_edge_list(5, &[]);
        assert_eq!(isolated.num_vertices(), 5);
        assert_eq!(isolated.num_edges(), 0);
        assert_eq!(isolated.row_offsets, vec![0; 6]);
        assert_eq!(isolated.max_degree(), 0);
        assert!(isolated.validate().is_ok());
    }

    #[test]
    fn test_reachability_bitmask_operations() {
        let nv = 5;
        let ns = 70; // Spans 3 words (32 + 32 + 6)
        let mut mask = ReachabilityBitmask::zeroed(nv, ns);
        assert_eq!(mask.words_per_vertex(), 3);
        assert_eq!(mask.words.len(), 15);

        // Initially zero
        assert!(!mask.is_reachable(0, 0));
        assert_eq!(mask.total_reachable_pairs(), 0);

        // Set reachability
        mask.set_reachable(0, 0); // word 0 bit 0
        mask.set_reachable(33, 0); // word 1 bit 1
        mask.set_reachable(69, 4); // word 2 bit 5

        assert!(mask.is_reachable(0, 0));
        assert!(mask.is_reachable(33, 0));
        assert!(!mask.is_reachable(32, 0));
        assert!(mask.is_reachable(69, 4));
        assert!(!mask.is_reachable(69, 0));

        assert_eq!(mask.reachable_source_count(0), 2);
        assert_eq!(mask.reachable_source_count(4), 1);
        assert_eq!(mask.reachable_source_count(1), 0);
        assert_eq!(mask.total_reachable_pairs(), 3);

        // Canonical distance conversion test
        let mut dists = vec![u32::MAX; ns * nv];
        dists[0] = 0;
        dists[33 * nv] = 2;
        dists[69 * nv + 4] = 5;

        let from_dist = ReachabilityBitmask::from_canonical_distances(nv, ns, &dists);
        assert_eq!(mask, from_dist);
        assert_eq!(from_dist.as_bytes().len(), 15 * 4);
    }
}
