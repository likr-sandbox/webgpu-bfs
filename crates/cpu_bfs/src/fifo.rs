use anyhow::{bail, Result};
use webgpu_bfs_core::CsrGraph;

/// Executes single-source BFS on a CSR graph, populating the provided `dist` slice.
///
/// # Invariants
/// * `dist` must be of length `graph.num_vertices` and pre-initialized to `u32::MAX`.
/// * `queue` is cleared and reused to avoid repeated heap reallocations.
#[inline]
pub fn bfs_single_source_into(
    graph: &CsrGraph,
    source: u32,
    dist: &mut [u32],
    queue: &mut Vec<u32>,
) {
    let num_vertices = graph.num_vertices;
    if num_vertices == 0 {
        return;
    }

    let src = source as usize;
    dist[src] = 0;

    queue.clear();
    queue.push(source);
    let mut head = 0;

    while head < queue.len() {
        let u = queue[head] as usize;
        head += 1;

        let u_dist = dist[u];
        let next_dist = u_dist + 1;

        let start = graph.row_offsets[u] as usize;
        let end = graph.row_offsets[u + 1] as usize;

        for &v in &graph.col_indices[start..end] {
            let v_idx = v as usize;
            if dist[v_idx] == u32::MAX {
                dist[v_idx] = next_dist;
                queue.push(v);
            }
        }
    }
}

/// Executes single-source BFS, returning an allocated `Vec<u32>` of distances of length |V|.
pub fn bfs_single_source(graph: &CsrGraph, source: u32) -> Result<Vec<u32>> {
    if graph.num_vertices == 0 {
        return Ok(Vec::new());
    }
    if source as usize >= graph.num_vertices {
        bail!(
            "Source vertex {} is out of bounds for graph with {} vertices",
            source,
            graph.num_vertices
        );
    }

    let mut dist = vec![u32::MAX; graph.num_vertices];
    let mut queue = Vec::with_capacity(graph.num_vertices);
    bfs_single_source_into(graph, source, &mut dist, &mut queue);
    Ok(dist)
}

/// Executes Algorithm A: Multi-source sequential FIFO queue BFS.
/// Evaluates each source sequentially, writing results into a flat `Vec<u32>` of layout
/// `dist[s * |V| + v]`.
pub fn ms_bfs_fifo(graph: &CsrGraph, sources: &[u32]) -> Result<Vec<u32>> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    if num_sources == 0 || num_vertices == 0 {
        return Ok(Vec::new());
    }

    // Pre-validate all sources to fail fast
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

    let mut result = vec![u32::MAX; num_sources * num_vertices];
    let mut queue = Vec::with_capacity(num_vertices);

    for (s_idx, &src) in sources.iter().enumerate() {
        let slice_start = s_idx * num_vertices;
        let slice_end = slice_start + num_vertices;
        let dist_slice = &mut result[slice_start..slice_end];

        bfs_single_source_into(graph, src, dist_slice, &mut queue);
    }

    Ok(result)
}
