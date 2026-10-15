//! crates/shaders/tests/shader_validation.rs
//! Automated Naga static analysis and architectural constraint validation for all embedded WGSL shaders.

use naga::{AddressSpace, ArraySize, Handle, Module, ShaderStage, Type, TypeInner};
use webgpu_bfs_shaders::*;

/// Calculates the size in bytes of a Naga type within workgroup memory.
fn compute_type_size(module: &Module, handle: Handle<Type>) -> u32 {
    match &module.types[handle].inner {
        TypeInner::Scalar(scalar) => scalar.width as u32,
        TypeInner::Vector { size, scalar } => (*size as u32) * (scalar.width as u32),
        TypeInner::Atomic(scalar) => scalar.width as u32,
        TypeInner::Array { base, size, stride } => {
            let elem_size = compute_type_size(module, *base);
            let effective_stride = (*stride).max(elem_size);
            match size {
                ArraySize::Constant(count) => count.get() * effective_stride,
                ArraySize::Dynamic => 0,
                ArraySize::Pending(_) => 0,
            }
        }
        TypeInner::Struct { span, .. } => *span,
        _ => 0,
    }
}

/// Computes the total workgroup shared memory (LDS) allocated across all `var<workgroup>` declarations.
fn compute_workgroup_storage_size(module: &Module) -> u32 {
    let mut total_bytes = 0u32;
    for (_, global) in module.global_variables.iter() {
        if global.space == AddressSpace::WorkGroup {
            total_bytes += compute_type_size(module, global.ty);
        }
    }
    total_bytes
}

/// Helper that parses and validates a WGSL source string, returning the verified module.
fn parse_and_validate(name: &str, source: &str) -> Module {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(m) => m,
        Err(e) => {
            panic!(
                "Shader '{}' failed WGSL syntax parsing:\n{}",
                name,
                e.emit_to_string(source)
            );
        }
    };

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );

    if let Err(e) = validator.validate(&module) {
        panic!("Shader '{}' failed Naga validation:\n{:?}", name, e);
    }

    module
}

#[test]
fn test_all_shaders_naga_validation() {
    for (name, source) in all_shaders() {
        let _ = parse_and_validate(name, source);
    }
}

#[test]
fn test_workgroup_storage_limits() {
    const WEBGPU_MAX_STORAGE_DEFAULT: u32 = 16384; // 16 KB default limit

    for (name, source) in all_shaders() {
        let module = parse_and_validate(name, source);
        let lds_size = compute_workgroup_storage_size(&module);

        assert!(
            lds_size <= WEBGPU_MAX_STORAGE_DEFAULT,
            "Shader '{}' workgroup storage ({} bytes) exceeds WebGPU default limit ({} bytes)!",
            name,
            lds_size,
            WEBGPU_MAX_STORAGE_DEFAULT
        );

        if *name == "indirect_sparse_push.wgsl" {
            // Must accommodate WG_QUEUE_CAP = 2048 (~8.6KB total)
            assert!(
                (4096..=12288).contains(&lds_size),
                "indirect_sparse_push.wgsl LDS size should be ~8.6KB, but got {} bytes",
                lds_size
            );
        }

        if *name == "indirect_sparse_push_256.wgsl" {
            // Must accommodate WG_QUEUE_CAP = 2048 (~8.9KB total)
            assert!(
                (4096..=12288).contains(&lds_size),
                "indirect_sparse_push_256.wgsl LDS size should be ~8.9KB, but got {} bytes",
                lds_size
            );
        }

        if *name == "indirect_sparse_push_512.wgsl" {
            assert!(
                (4096..=12288).contains(&lds_size),
                "indirect_sparse_push_512.wgsl LDS size should be ~9.5KB, but got {} bytes",
                lds_size
            );
        }

        if *name == "workgroup_persistent_bfs.wgsl" {
            assert_eq!(
                lds_size, 16384,
                "workgroup_persistent_bfs.wgsl LDS size must be exactly 16384 bytes, got {} bytes",
                lds_size
            );
        }
    }
}

#[test]
fn test_compute_workgroup_invocations_budget() {
    const WEBGPU_MAX_INVOCATIONS_DEFAULT: u32 = 256;

    for (name, source) in all_shaders() {
        let module = parse_and_validate(name, source);
        for ep in &module.entry_points {
            if ep.stage == ShaderStage::Compute {
                let invocations = ep.workgroup_size[0] * ep.workgroup_size[1] * ep.workgroup_size[2];
                assert!(
                    invocations <= WEBGPU_MAX_INVOCATIONS_DEFAULT,
                    "Shader '{}' compute entry point '{}' has {} invocations, exceeding WebGPU limit of {}!",
                    name,
                    ep.name,
                    invocations,
                    WEBGPU_MAX_INVOCATIONS_DEFAULT
                );
                assert!(ep.workgroup_size[0] <= 256);
                assert!(ep.workgroup_size[1] <= 256);
                assert!(ep.workgroup_size[2] <= 64);
            }
        }
    }
}

#[test]
fn test_resource_binding_uniqueness() {
    for (name, source) in all_shaders() {
        let module = parse_and_validate(name, source);
        let mut seen_bindings = std::collections::HashSet::new();

        for (_, global) in module.global_variables.iter() {
            if let Some(binding) = &global.binding {
                let key = (binding.group, binding.binding);
                assert!(
                    seen_bindings.insert(key),
                    "Shader '{}' contains duplicate binding @group({}) @binding({})!",
                    name,
                    binding.group,
                    binding.binding
                );
                // Standard single bind-group convention
                assert_eq!(binding.group, 0, "Shader '{}' uses unexpected group {}", name, binding.group);
            }
        }
    }
}
