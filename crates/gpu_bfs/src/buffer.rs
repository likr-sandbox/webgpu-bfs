//! GPU Buffer management abstractions for CSR graphs, frontiers, and distance matrices.

#[allow(unused_imports)]
use anyhow::{Context, Result};
use webgpu_bfs_core::CsrGraph;
use wgpu::{Buffer, BufferUsages, Device};

use crate::context::GpuContext;

/// Reads back `count` u32 words from an already-mapped or mappable staging buffer.
///
/// Blocking on native platforms via `device.poll(Maintain::Wait)`.
pub fn read_buffer_u32_blocking(
    device: &Device,
    staging_buffer: &Buffer,
    count: usize,
) -> Result<Vec<u32>> {
    let byte_size = (count * std::mem::size_of::<u32>()) as u64;
    let slice = staging_buffer.slice(..byte_size);

    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });

    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::Maintain::Wait);

    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(receiver)
        .context("Staging buffer map receiver dropped")?
        .context("Failed to map staging buffer for readback")?;

    #[cfg(target_arch = "wasm32")]
    {
        let _ = (device, receiver);
        anyhow::bail!("Synchronous readback not supported on wasm32");
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let view = slice.get_mapped_range();
        let data: &[u32] = bytemuck::cast_slice(&view);
        let result = data.to_vec();
        drop(view);
        staging_buffer.unmap();

        Ok(result)
    }
}

/// GPU storage buffers holding the static CSR graph topology.
#[derive(Debug)]
pub struct CsrGpuBuffers {
    pub row_offsets: Buffer,
    pub col_indices: Buffer,
    pub num_vertices: usize,
    pub num_edges: usize,
}

impl CsrGpuBuffers {
    /// Uploads a CPU `CsrGraph` into GPU storage buffers.
    pub fn from_graph(ctx: &GpuContext, graph: &CsrGraph) -> Self {
        let row_offsets = ctx.create_storage_buffer_init(
            "csr_row_offsets",
            &graph.row_offsets,
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );

        let dummy_col = [0u32];
        let col_slice: &[u32] = if graph.col_indices.is_empty() {
            &dummy_col
        } else {
            &graph.col_indices
        };

        let col_indices = ctx.create_storage_buffer_init(
            "csr_col_indices",
            col_slice,
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );

        Self {
            row_offsets,
            col_indices,
            num_vertices: graph.num_vertices,
            num_edges: graph.num_edges,
        }
    }
}

use std::sync::Arc;

/// GPU storage buffers holding both CSR (Push) and CSC (Pull) representations.
#[derive(Debug, Clone)]
pub struct PushPullGraphBuffers {
    pub csr_row_offsets: Arc<Buffer>,
    pub csr_col_indices: Arc<Buffer>,
    pub csc_col_offsets: Arc<Buffer>,
    pub csc_row_indices: Arc<Buffer>,
    pub num_vertices: usize,
    pub num_edges: usize,
    pub is_symmetric: bool,
}

impl PushPullGraphBuffers {
    /// Uploads a known-symmetric `CsrGraph` directly sharing CSR buffers as CSC without transpose.
    pub fn from_symmetric_graph(ctx: &GpuContext, graph: &CsrGraph) -> Self {
        let dummy_col = [0u32];

        let csr_row_offsets = Arc::new(ctx.create_storage_buffer_init(
            "csr_row_offsets",
            &graph.row_offsets,
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ));

        let col_slice: &[u32] = if graph.col_indices.is_empty() {
            &dummy_col
        } else {
            &graph.col_indices
        };
        let csr_col_indices = Arc::new(ctx.create_storage_buffer_init(
            "csr_col_indices",
            col_slice,
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ));

        let csc_col_offsets = Arc::clone(&csr_row_offsets);
        let csc_row_indices = Arc::clone(&csr_col_indices);

        Self {
            csr_row_offsets,
            csr_col_indices,
            csc_col_offsets,
            csc_row_indices,
            num_vertices: graph.num_vertices,
            num_edges: graph.num_edges,
            is_symmetric: true,
        }
    }

    /// Uploads a CPU `CsrGraph` and its transpose (`CscGraph`) into GPU storage buffers.
    ///
    /// Reuses CSR buffers for CSC if the graph is symmetric.
    pub fn from_graph(ctx: &GpuContext, graph: &CsrGraph) -> Self {
        let dummy_col = [0u32];

        let csr_row_offsets = Arc::new(ctx.create_storage_buffer_init(
            "csr_row_offsets",
            &graph.row_offsets,
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ));

        let col_slice: &[u32] = if graph.col_indices.is_empty() {
            &dummy_col
        } else {
            &graph.col_indices
        };
        let csr_col_indices = Arc::new(ctx.create_storage_buffer_init(
            "csr_col_indices",
            col_slice,
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ));

        // Fast odd-edge rejection: in simple undirected graphs, num_edges must be even
        let (csc_col_offsets, csc_row_indices, is_symmetric) = if graph.num_edges % 2 != 0 {
            let csc = graph.to_csc();
            let col_offsets = Arc::new(ctx.create_storage_buffer_init(
                "csc_col_offsets",
                &csc.col_offsets,
                BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            ));
            let row_slice: &[u32] = if csc.row_indices.is_empty() {
                &dummy_col
            } else {
                &csc.row_indices
            };
            let row_indices = Arc::new(ctx.create_storage_buffer_init(
                "csc_row_indices",
                row_slice,
                BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            ));
            (col_offsets, row_indices, false)
        } else {
            let csc = graph.to_csc();
            let is_sym = graph.row_offsets == csc.col_offsets && graph.col_indices == csc.row_indices;
            if is_sym {
                (Arc::clone(&csr_row_offsets), Arc::clone(&csr_col_indices), true)
            } else {
                let col_offsets = Arc::new(ctx.create_storage_buffer_init(
                    "csc_col_offsets",
                    &csc.col_offsets,
                    BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                ));
                let row_slice: &[u32] = if csc.row_indices.is_empty() {
                    &dummy_col
                } else {
                    &csc.row_indices
                };
                let row_indices = Arc::new(ctx.create_storage_buffer_init(
                    "csc_row_indices",
                    row_slice,
                    BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                ));
                (col_offsets, row_indices, false)
            }
        };

        Self {
            csr_row_offsets,
            csr_col_indices,
            csc_col_offsets,
            csc_row_indices,
            num_vertices: graph.num_vertices,
            num_edges: graph.num_edges,
            is_symmetric,
        }
    }
}

/// Ping-pong frontier buffers with atomic active counter.
#[derive(Debug)]
pub struct FrontierGpuBuffers {
    pub curr_frontier: Buffer,
    pub next_frontier: Buffer,
    pub counter: Buffer,
    pub capacity_words: usize,
}

impl FrontierGpuBuffers {
    /// Allocates ping-pong frontier buffers for a given number of words.
    pub fn new(ctx: &GpuContext, capacity_words: usize) -> Self {
        let size_bytes = (capacity_words * std::mem::size_of::<u32>()) as u64;

        let curr_frontier = ctx.create_storage_buffer(
            "frontier_curr",
            size_bytes,
            BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        let next_frontier = ctx.create_storage_buffer(
            "frontier_next",
            size_bytes,
            BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        let counter = ctx.create_storage_buffer_init(
            "frontier_counter",
            &[0u32],
            BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        Self {
            curr_frontier,
            next_frontier,
            counter,
            capacity_words,
        }
    }

    /// Resets the frontier buffers to all zeros.
    pub fn clear(&self, ctx: &GpuContext) {
        let zeros = vec![0u32; self.capacity_words];
        ctx.queue
            .write_buffer(&self.curr_frontier, 0, bytemuck::cast_slice(&zeros));
        ctx.queue
            .write_buffer(&self.next_frontier, 0, bytemuck::cast_slice(&zeros));
        ctx.queue
            .write_buffer(&self.counter, 0, bytemuck::cast_slice(&[0u32]));
    }

    /// Resets the counter to 0.
    pub fn reset_counter(&self, ctx: &GpuContext) {
        ctx.queue
            .write_buffer(&self.counter, 0, bytemuck::cast_slice(&[0u32]));
    }

    /// Reads back the atomic frontier counter to host.
    pub fn read_counter(&self, ctx: &GpuContext) -> Result<u32> {
        let count_vec = ctx.read_buffer_u32(&self.counter, 1)?;
        Ok(count_vec[0])
    }

    /// Swaps the `curr_frontier` and `next_frontier` buffer references.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.curr_frontier, &mut self.next_frontier);
    }
}

/// GPU storage buffer holding the multi-source distance matrix.
#[derive(Debug)]
pub struct DistanceGpuBuffer {
    pub buffer: Buffer,
    pub num_sources: usize,
    pub num_vertices: usize,
}

impl DistanceGpuBuffer {
    /// Creates and initializes the distance matrix buffer.
    ///
    /// Invariants:
    /// - Dimensions: `num_sources * num_vertices` u32 elements.
    /// - Flat layout: `dist[s * num_vertices + v]`.
    /// - Unreachable: `u32::MAX` (`0xFFFF_FFFF`).
    /// - Sources: `dist[s * num_vertices + s] = 0`.
    pub fn new(ctx: &GpuContext, num_vertices: usize, sources: &[u32]) -> Self {
        let num_sources = sources.len();
        let total_words = num_sources * num_vertices;
        let mut initial = vec![u32::MAX; total_words];

        for (s_idx, &src) in sources.iter().enumerate() {
            if (src as usize) < num_vertices {
                initial[s_idx * num_vertices + (src as usize)] = 0;
            }
        }

        let buffer = ctx.create_storage_buffer_init(
            "distance_matrix",
            &initial,
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );

        Self {
            buffer,
            num_sources,
            num_vertices,
        }
    }

    /// Reads back the complete distance matrix as a flat `Vec<u32>`.
    pub fn read_distances(&self, ctx: &GpuContext) -> Result<Vec<u32>> {
        let total_words = self.num_sources * self.num_vertices;
        ctx.read_buffer_u32(&self.buffer, total_words)
    }
}
