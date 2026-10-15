//! crates/shaders/tests/adversarial_challenge.rs
//! Adversarial stress test suite for WGSL shaders in webgpu-bfs.
//! Validates Naga compilation, WebGPU compute resource limits, LDS bounds, and binding soundness.

use naga::{AddressSpace, ArraySize, Handle, Module, ShaderStage, StorageAccess, Type, TypeInner};
use std::collections::HashSet;
use webgpu_bfs_shaders::*;

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

fn compute_workgroup_storage_size(module: &Module) -> u32 {
    let mut total_bytes = 0u32;
    for (_, global) in module.global_variables.iter() {
        if global.space == AddressSpace::WorkGroup {
            total_bytes += compute_type_size(module, global.ty);
        }
    }
    total_bytes
}

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
fn test_adversarial_all_shaders_registered_and_valid() {
    let shaders = all_shaders();
    assert_eq!(
        shaders.len(),
        32,
        "Expected exactly 32 embedded shaders, but found {}",
        shaders.len()
    );

    let mut names = HashSet::new();
    for (name, source) in shaders {
        assert!(names.insert(*name), "Duplicate shader registered: {}", name);
        assert!(!source.is_empty(), "Shader '{}' source is empty!", name);
        let module = parse_and_validate(name, source);

        // Verify module has at least one entry point
        assert!(
            !module.entry_points.is_empty(),
            "Shader '{}' has no entry points!",
            name
        );
    }
}

#[test]
fn test_adversarial_compute_invocations_and_limits() {
    for (name, source) in all_shaders() {
        let module = parse_and_validate(name, source);

        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                ShaderStage::Compute,
                "Shader '{}' entry point '{}' is stage {:?}, expected Compute",
                name,
                ep.name,
                ep.stage
            );

            let wx = ep.workgroup_size[0];
            let wy = ep.workgroup_size[1];
            let wz = ep.workgroup_size[2];
            let invocations = wx * wy * wz;

            assert!(
                invocations <= 256,
                "Shader '{}' invocations {} exceed WebGPU max of 256",
                name,
                invocations
            );
            assert!(
                wx <= 256,
                "Shader '{}' workgroup_size.x {} exceeds WebGPU max 256",
                name,
                wx
            );
            assert!(
                wy <= 256,
                "Shader '{}' workgroup_size.y {} exceeds WebGPU max 256",
                name,
                wy
            );
            assert!(
                wz <= 64,
                "Shader '{}' workgroup_size.z {} exceeds WebGPU max 64",
                name,
                wz
            );
        }
    }
}

#[test]
fn test_adversarial_workgroup_memory_audit() {
    const WEBGPU_MAX_LDS: u32 = 16384; // 16 KB

    println!("\n=== WORKGROUP LDS MEMORY AUDIT ===");
    for (name, source) in all_shaders() {
        let module = parse_and_validate(name, source);
        let lds_bytes = compute_workgroup_storage_size(&module);

        if lds_bytes > 0 {
            println!(
                "Shader: {:<35} | LDS: {:>5} B ({:>5.2}% of 16 KB limit)",
                name,
                lds_bytes,
                (lds_bytes as f64 / WEBGPU_MAX_LDS as f64) * 100.0
            );
        }

        assert!(
            lds_bytes <= WEBGPU_MAX_LDS,
            "Shader '{}' requires {} bytes of workgroup memory, exceeding 16KB limit!",
            name,
            lds_bytes
        );
    }
}

#[test]
fn test_adversarial_bindings_and_access_modes() {
    println!("\n=== BINDING & ACCESS MODE AUDIT ===");
    for (name, source) in all_shaders() {
        let module = parse_and_validate(name, source);
        let mut bindings_seen = HashSet::new();
        let mut storage_count = 0;
        let mut uniform_count = 0;

        for (_, global) in module.global_variables.iter() {
            if let Some(b) = &global.binding {
                assert_eq!(
                    b.group, 0,
                    "Shader '{}' uses group {}, expected 0",
                    name, b.group
                );
                assert!(
                    bindings_seen.insert(b.binding),
                    "Shader '{}' duplicate binding index {}!",
                    name,
                    b.binding
                );

                match global.space {
                    AddressSpace::Storage { access } => {
                        storage_count += 1;
                        assert!(
                            access == StorageAccess::LOAD
                                || access == (StorageAccess::LOAD | StorageAccess::STORE),
                            "Shader '{}' storage binding {} has unexpected access {:?}",
                            name,
                            b.binding,
                            access
                        );
                    }
                    AddressSpace::Uniform => {
                        uniform_count += 1;
                    }
                    AddressSpace::WorkGroup => {
                        panic!(
                            "Shader '{}' has workgroup variable with @binding attribute!",
                            name
                        );
                    }
                    _ => {}
                }
            }
        }

        println!(
            "Shader: {:<35} | Uniforms: {:>2} | Storage Buffers: {:>2} | Total Bindings: {:>2}",
            name,
            uniform_count,
            storage_count,
            bindings_seen.len()
        );
    }
}

#[test]
fn test_adversarial_atomic_variable_placement() {
    for (name, source) in all_shaders() {
        let module = parse_and_validate(name, source);

        for (_, global) in module.global_variables.iter() {
            let is_atomic = match &module.types[global.ty].inner {
                TypeInner::Atomic(_) => true,
                TypeInner::Array { base, .. } => {
                    matches!(&module.types[*base].inner, TypeInner::Atomic(_))
                }
                _ => false,
            };

            if is_atomic {
                let valid_space = match global.space {
                    AddressSpace::Storage { access } => {
                        access.contains(StorageAccess::LOAD | StorageAccess::STORE)
                    }
                    AddressSpace::WorkGroup => true,
                    _ => false,
                };
                assert!(
                    valid_space,
                    "Shader '{}' declares atomic variable in invalid address space {:?}",
                    name, global.space
                );
            }
        }
    }
}
