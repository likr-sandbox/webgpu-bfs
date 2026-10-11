//! Unit tests verifying WebGPU context, 3-tier adapter selection,
//! and compute shader execution on Mesa Lavapipe Vulkan.

use super::*;
use webgpu_bfs_core::generate_grid2d;

#[test]
fn test_gpu_context_initialization_on_lavapipe() {
    let ctx = GpuContext::new().expect("GpuContext initialization must succeed");
    let info = ctx.info();

    println!(
        "Acquired adapter: name='{}', type={:?}, backend={:?}",
        info.name, info.device_type, info.backend
    );

    assert!(!info.name.is_empty(), "Adapter name should not be empty");
}

#[test]
fn test_trivial_compute_shader_execution_and_readback() {
    let ctx = GpuContext::new().expect("Failed to initialize GpuContext");

    let input: [u32; 4] = [10, 20, 30, 40];
    let storage_buf = ctx.create_storage_buffer_init(
        "test_input",
        &input,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );

    let shader_source = r#"
        @group(0) @binding(0)
        var<storage, read_write> data: array<u32>;

        @compute @workgroup_size(4)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            if (i < 4u) {
                data[i] = data[i] + 1u;
            }
        }
    "#;

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test_bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buf.as_entire_binding(),
        }],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("test_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));

    let output = ctx
        .read_buffer_u32(&storage_buf, 4)
        .expect("Buffer readback must succeed");

    assert_eq!(output, vec![11, 21, 31, 41]);
}

#[test]
fn test_csr_buffer_upload_and_readback_roundtrip() {
    let ctx = GpuContext::new().expect("Failed to initialize GpuContext");
    let grid = generate_grid2d(4, 4);

    let csr_gpu = CsrGpuBuffers::from_graph(&ctx, &grid);

    let row_offsets_read = ctx
        .read_buffer_u32(&csr_gpu.row_offsets, grid.row_offsets.len())
        .expect("Read row offsets");
    let col_indices_read = ctx
        .read_buffer_u32(&csr_gpu.col_indices, grid.col_indices.len())
        .expect("Read col indices");

    assert_eq!(row_offsets_read, grid.row_offsets);
    assert_eq!(col_indices_read, grid.col_indices);
}

#[test]
fn test_distance_buffer_initialization_and_readback() {
    let ctx = GpuContext::new().expect("Failed to initialize GpuContext");
    let num_vertices = 10;
    let sources = [0, 3, 7];

    let dist_gpu = DistanceGpuBuffer::new(&ctx, num_vertices, &sources);
    let distances = dist_gpu.read_distances(&ctx).expect("Read distances");

    assert_eq!(distances.len(), sources.len() * num_vertices);

    // Verify sources have distance 0 and others have u32::MAX
    for (s_idx, &src) in sources.iter().enumerate() {
        for v in 0..num_vertices {
            let val = distances[s_idx * num_vertices + v];
            if v == src as usize {
                assert_eq!(val, 0);
            } else {
                assert_eq!(val, u32::MAX);
            }
        }
    }
}
