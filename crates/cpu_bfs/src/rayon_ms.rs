use crate::fifo::bfs_single_source_into;
use anyhow::{bail, Result};
use webgpu_bfs_core::CsrGraph;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Executes Algorithm B: Multi-threaded Rayon batched MS-BFS.
/// Parallelizes source processing across the Rayon thread pool with lock-free,
/// zero-atomic disjoint chunk writes.
pub fn ms_bfs_rayon(graph: &CsrGraph, sources: &[u32]) -> Result<Vec<u32>> {
    let num_vertices = graph.num_vertices;
    let num_sources = sources.len();

    if num_sources == 0 || num_vertices == 0 {
        return Ok(Vec::new());
    }

    // Up-front validation prevents panics inside parallel tasks
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

    #[cfg(feature = "rayon")]
    {
        result
            .par_chunks_exact_mut(num_vertices)
            .zip(sources.par_iter())
            .for_each(|(dist_slice, &src)| {
                let mut local_queue = Vec::with_capacity(num_vertices);
                bfs_single_source_into(graph, src, dist_slice, &mut local_queue);
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        // Graceful single-threaded fallback if rayon feature is disabled
        let mut local_queue = Vec::with_capacity(num_vertices);
        for (dist_slice, &src) in result.chunks_exact_mut(num_vertices).zip(sources.iter()) {
            bfs_single_source_into(graph, src, dist_slice, &mut local_queue);
        }
    }

    Ok(result)
}
