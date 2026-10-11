//! WebGPU runtime, GPU buffer manager, and MS-BFS compute pipelines.

pub mod buffer;
pub mod context;
pub mod dispatch;
pub mod engine;
pub mod pipeline;

#[cfg(test)]
mod tests;

pub use buffer::{read_buffer_u32_blocking, CsrGpuBuffers, DistanceGpuBuffer, FrontierGpuBuffers};
pub use context::GpuContext;
pub use engine::{GpuBfsEngine, GpuBfsMethod};
pub use pipeline::PipelineCache;
