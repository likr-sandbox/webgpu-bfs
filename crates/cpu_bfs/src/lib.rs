//! High-performance CPU reference BFS engines and ground truth verification for webgpu-bfs.

pub mod bit_parallel;
pub mod engine;
pub mod fifo;
pub mod rayon_ms;

#[cfg(test)]
mod tests;

pub use bit_parallel::ms_bfs_bit_parallel;
pub use engine::{CpuBfsAlgorithm, CpuBfsEngine};
pub use fifo::{bfs_single_source, bfs_single_source_into, ms_bfs_fifo};
pub use rayon_ms::ms_bfs_rayon;
pub use webgpu_bfs_core::{BfsEngine, ReachabilityBitmask, ReachabilityEngine};
