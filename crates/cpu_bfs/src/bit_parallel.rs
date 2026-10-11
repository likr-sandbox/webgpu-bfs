use anyhow::{bail, Result};
use webgpu_bfs_core::CsrGraph;

/// Executes a single 64-source bit-parallel BFS batch over `batch_sources`.
///
/// # Arguments
/// * `graph` - The CSR graph topology.
/// * `batch_sources` - Slice of at most 64 source vertex IDs.
/// * `batch_start` - Offset index of this batch within the global `sources` slice.
/// * `dist` - Global distance output buffer of length `S * num_vertices`.
pub fn run_bit_parallel_batch_64(
    graph: &CsrGraph,
    batch_sources: &[u32],
    batch_start: usize,
    dist: &mut [u32],
) {
    let num_vertices = graph.num_vertices;
    let batch_size = batch_sources.len();
    assert!(batch_size <= 64, "Batch size must be <= 64");
    if num_vertices == 0 || batch_size == 0 {
        return;
    }

    let mut seen = vec![0u64; num_vertices];
    let mut frontier = vec![0u64; num_vertices];
    let mut next_frontier = vec![0u64; num_vertices];

    let mut active_vertices = Vec::with_capacity(num_vertices);
    let mut next_active = Vec::with_capacity(num_vertices);

    // Initialize level 0
    for (k, &s) in batch_sources.iter().enumerate() {
        let s_idx = s as usize;
        let mask = 1u64 << k;
        let s_global = batch_start + k;

        dist[s_global * num_vertices + s_idx] = 0;

        if frontier[s_idx] == 0 {
            active_vertices.push(s_idx as u32);
        }
        frontier[s_idx] |= mask;
        seen[s_idx] |= mask;
    }

    let mut current_dist = 0u32;

    while !active_vertices.is_empty() {
        let next_dist = current_dist + 1;

        // Traverse active frontier and push bitmasks to neighbors
        for &u_u32 in &active_vertices {
            let u = u_u32 as usize;
            let f = frontier[u];
            frontier[u] = 0; // Clear frontier for next levels

            let start = graph.row_offsets[u] as usize;
            let end = graph.row_offsets[u + 1] as usize;

            for &v_u32 in &graph.col_indices[start..end] {
                let v = v_u32 as usize;
                if next_frontier[v] == 0 {
                    next_active.push(v_u32);
                }
                next_frontier[v] |= f;
            }
        }

        active_vertices.clear();

        // Evaluate newly visited vertices and record distances
        for &v_u32 in &next_active {
            let v = v_u32 as usize;
            let nf = next_frontier[v];
            next_frontier[v] = 0; // Clean up next_frontier with zero extra memset cost

            let new_visits = nf & !seen[v];
            if new_visits != 0 {
                seen[v] |= new_visits;
                frontier[v] = new_visits;
                active_vertices.push(v_u32);

                // Extract each set bit using trailing_zeros and Brian Kernighan's bit reset
                let mut bits = new_visits;
                while bits != 0 {
                    let bit_idx = bits.trailing_zeros() as usize;
                    let s_global = batch_start + bit_idx;
                    dist[s_global * num_vertices + v] = next_dist;
                    bits &= bits - 1;
                }
            }
        }

        next_active.clear();
        current_dist += 1;
    }
}

/// Executes Algorithm C: 64-bit bit-parallel MS-BFS across all sources.
/// Processes sources in chunks of up to 64. Handles arbitrary source counts (e.g., S=1, 15, 33, 65, 128).
pub fn ms_bfs_bit_parallel(graph: &CsrGraph, sources: &[u32]) -> Result<Vec<u32>> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    if num_sources == 0 || num_vertices == 0 {
        return Ok(Vec::new());
    }

    // Pre-validate all sources
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

    // Process sources in 64-source batches
    for (batch_idx, batch) in sources.chunks(64).enumerate() {
        let batch_start = batch_idx * 64;
        run_bit_parallel_batch_64(graph, batch, batch_start, &mut result);
    }

    Ok(result)
}
