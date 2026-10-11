//! WebGPU context initialization with 3-tier adapter selection.

use anyhow::{Context, Result};
#[allow(unused_imports)]
use wgpu::{
    Adapter, AdapterInfo, Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor,
    InstanceFlags, Limits, MemoryHints, PowerPreference, Queue, RequestAdapterOptions,
};

/// Encapsulates the WebGPU Instance, Adapter, Device, and Queue.
#[derive(Debug)]
pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub adapter_info: AdapterInfo,
}

impl GpuContext {
    /// Synchronously initializes the GPU context.
    ///
    /// Uses `pollster::block_on` to drive asynchronous adapter and device acquisition.
    pub fn new() -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(Self::new_async())
        }
        #[cfg(target_arch = "wasm32")]
        {
            anyhow::bail!("Synchronous GpuContext::new is not supported on wasm32; use GpuContext::new_async()")
        }
    }

    /// Alias for `GpuContext::new()`.
    pub fn init_blocking() -> Result<Self> {
        Self::new()
    }

    /// Asynchronously initializes the WebGPU context with a robust 3-tier adapter selection strategy.
    ///
    /// Selection tiers:
    /// 1. HighPerformance hardware GPU (discrete/integrated).
    /// 2. Fallback software adapter (Mesa Lavapipe / SwiftShader).
    /// 3. Adapter enumeration across all supported backends.
    pub async fn new_async() -> Result<Self> {
        let instance = Instance::new(
            &InstanceDescriptor {
                backends: Backends::all(),
                flags: InstanceFlags::default() | InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
                backend_options: Default::default(),
            }
            .with_env(),
        );

        log::info!("WebGPU Adapter Selection: Initiating 3-tier discovery...");

        // Tier 1: HighPerformance hardware GPU
        let mut adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        // Tier 2: Software fallback adapter (Lavapipe)
        if adapter.is_none() {
            log::warn!("Tier 1 (HighPerformance) failed. Retrying with Tier 2 (force_fallback_adapter: true)...");
            adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::None,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                })
                .await;
        }

        // Tier 3: Enumerate all adapters across all backends
        #[cfg(not(target_arch = "wasm32"))]
        if adapter.is_none() {
            log::warn!(
                "Tier 2 (Fallback adapter) failed. Retrying with Tier 3 (enumerate_adapters)..."
            );
            adapter = instance
                .enumerate_adapters(Backends::all())
                .into_iter()
                .next();
        }

        let adapter = adapter.context(
            "Failed to acquire any WebGPU adapter! Ensure Vulkan drivers or Mesa Lavapipe are installed.",
        )?;

        let adapter_info = adapter.get_info();
        log::info!(
            "WebGPU Adapter selected: '{}' (type: {:?}, backend: {:?}, driver: '{}')",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend,
            adapter_info.driver
        );

        let adapter_limits = adapter.limits();
        log::info!(
            "Adapter hardware limits: max_storage_buffer_binding_size = {} MB ({} bytes), max_buffer_size = {} MB ({} bytes)",
            adapter_limits.max_storage_buffer_binding_size / (1024 * 1024),
            adapter_limits.max_storage_buffer_binding_size,
            adapter_limits.max_buffer_size / (1024 * 1024),
            adapter_limits.max_buffer_size
        );

        // Target up to 2GB storage buffers (2^31 - 1 bytes for u32 binding size, 2^31 bytes for u64 buffer size)
        const TWO_GB_U32: u32 = (2u64 * 1024 * 1024 * 1024 - 1) as u32; // 2,147,483,647 bytes
        const TWO_GB_U64: u64 = 2u64 * 1024 * 1024 * 1024;             // 2,147,483,648 bytes

        let default_limits = Limits::default();
        let elevated_limits = Limits {
            max_storage_buffer_binding_size: adapter_limits
                .max_storage_buffer_binding_size
                .min(TWO_GB_U32)
                .max(default_limits.max_storage_buffer_binding_size),
            max_buffer_size: adapter_limits
                .max_buffer_size
                .min(TWO_GB_U64)
                .max(default_limits.max_buffer_size),
            max_compute_workgroup_storage_size: adapter_limits
                .max_compute_workgroup_storage_size
                .max(default_limits.max_compute_workgroup_storage_size),
            max_compute_invocations_per_workgroup: adapter_limits
                .max_compute_invocations_per_workgroup
                .max(default_limits.max_compute_invocations_per_workgroup),
            max_compute_workgroup_size_x: adapter_limits
                .max_compute_workgroup_size_x
                .max(default_limits.max_compute_workgroup_size_x),
            max_compute_workgroup_size_y: adapter_limits
                .max_compute_workgroup_size_y
                .max(default_limits.max_compute_workgroup_size_y),
            max_compute_workgroup_size_z: adapter_limits
                .max_compute_workgroup_size_z
                .max(default_limits.max_compute_workgroup_size_z),
            max_compute_workgroups_per_dimension: adapter_limits
                .max_compute_workgroups_per_dimension
                .max(default_limits.max_compute_workgroups_per_dimension),
            ..default_limits
        };

        let device_res = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("WebGPU-BFS Device (Elevated Limits)"),
                    required_features: Features::empty(),
                    required_limits: elevated_limits,
                    memory_hints: MemoryHints::default(),
                },
                None,
            )
            .await;

        let (device, queue) = match device_res {
            Ok((dev, q)) => {
                log::info!(
                    "Successfully acquired WebGPU device with elevated limits: max_storage_buffer_binding_size = {} MB, max_buffer_size = {} MB",
                    dev.limits().max_storage_buffer_binding_size / (1024 * 1024),
                    dev.limits().max_buffer_size / (1024 * 1024)
                );
                (dev, q)
            }
            Err(err) => {
                log::warn!(
                    "Elevated limits request failed ({:?}); falling back to Limits::default()...",
                    err
                );
                adapter
                    .request_device(
                        &DeviceDescriptor {
                            label: Some("WebGPU-BFS Device (Default Limits)"),
                            required_features: Features::empty(),
                            required_limits: Limits::default(),
                            memory_hints: MemoryHints::default(),
                        },
                        None,
                    )
                    .await
                    .context("Failed to create WebGPU logical device and queue with default limits")?
            }
        };

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_info,
        })
    }

    /// Returns a reference to the active `Device`.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the active `Queue`.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns a reference to the `AdapterInfo`.
    pub fn info(&self) -> &AdapterInfo {
        &self.adapter_info
    }

    /// Returns the active device limits negotiated with the GPU adapter.
    pub fn limits(&self) -> Limits {
        self.device.limits()
    }

    /// Returns the maximum allowed storage buffer binding size in bytes.
    pub fn max_storage_buffer_binding_size(&self) -> u64 {
        self.device.limits().max_storage_buffer_binding_size as u64
    }

    /// Returns the maximum allowed buffer allocation size in bytes.
    pub fn max_buffer_size(&self) -> u64 {
        self.device.limits().max_buffer_size
    }

    /// Checks if a buffer of `size_bytes` can be allocated and bound as a storage buffer.
    pub fn can_bind_storage_buffer(&self, size_bytes: u64) -> bool {
        size_bytes <= self.max_storage_buffer_binding_size() && size_bytes <= self.max_buffer_size()
    }

    /// Calculates the maximum number of BFS sources that can fit into a single distance matrix buffer
    /// without exceeding the device's storage buffer binding limit or maximum buffer size.
    ///
    /// The result is aligned down to a multiple of 32 sources (minimum 32).
    pub fn max_macro_chunk_sources(&self, num_vertices: usize) -> usize {
        let bytes_per_source = (num_vertices * std::mem::size_of::<u32>()) as u64;
        if bytes_per_source == 0 {
            return 2048;
        }
        let max_by_binding = self.max_storage_buffer_binding_size() / bytes_per_source;
        let max_by_buf = self.max_buffer_size() / bytes_per_source;
        let max_sources = max_by_binding.min(max_by_buf) as usize;
        let aligned = (max_sources / 32) * 32;
        aligned.max(32)
    }

    /// Creates a GPU storage buffer initialized with host data.
    pub fn create_storage_buffer_init<T: bytemuck::NoUninit>(
        &self,
        label: &str,
        data: &[T],
        additional_usages: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | additional_usages,
            })
    }

    /// Creates an uninitialized GPU storage buffer of specified byte size.
    pub fn create_storage_buffer(
        &self,
        label: &str,
        size_bytes: u64,
        additional_usages: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE | additional_usages,
            mapped_at_creation: false,
        })
    }

    /// Creates a host-mappable staging buffer for asynchronous readback.
    pub fn create_staging_buffer(&self, label: &str, size_bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Copies a storage buffer's content to a staging buffer and reads back as `Vec<u32>`.
    pub fn read_buffer_u32(&self, storage_buffer: &wgpu::Buffer, count: usize) -> Result<Vec<u32>> {
        let byte_size = (count * std::mem::size_of::<u32>()) as u64;
        let staging_buffer = self.create_staging_buffer("readback_staging", byte_size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(storage_buffer, 0, &staging_buffer, 0, byte_size);
        self.queue.submit(Some(encoder.finish()));

        crate::buffer::read_buffer_u32_blocking(&self.device, &staging_buffer, count)
    }
}
