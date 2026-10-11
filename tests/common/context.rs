use std::sync::Arc;
use std::sync::OnceLock;
use webgpu_bfs_gpu::GpuContext;

/// Returns a shared, lazily initialized `GpuContext`.
/// This avoids the overhead of re-creating WebGPU instances, devices, and Lavapipe software drivers across hundreds of test cases.
pub fn shared_gpu_context() -> Arc<GpuContext> {
    static CTX: OnceLock<Arc<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| {
        Arc::new(
            GpuContext::init_blocking()
                .expect("Failed to initialize WebGPU context for integration testing"),
        )
    })
    .clone()
}
