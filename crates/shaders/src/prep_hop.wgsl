// crates/shaders/src/prep_hop.wgsl
// GPU Autonomous Indirect Dispatch & Iteration State Preparation Kernel
//
// Computes dynamic workgroup dimensions for WebGPU dispatch_workgroups_indirect,
// handles zero-workgroup early exit on frontier exhaustion, increments the BFS
// level counter on GPU, and resets the active count accumulator for the next hop.
//
// Bindings:
// @group(0) @binding(0) state: IterationState (storage, read_write)
// @group(0) @binding(1) indirect_args: IndirectArgs (storage, read_write)

struct IterationState {
    level: u32,
    active_count: atomic<u32>,
    terminated: u32,
    max_workgroups_x: u32,
};

struct IndirectArgs {
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
};

@group(0) @binding(0) var<storage, read_write> state: IterationState;
@group(0) @binding(1) var<storage, read_write> indirect_args: IndirectArgs;

@compute @workgroup_size(1)
fn main() {
    let count = atomicLoad(&state.active_count);

    // If the frontier is empty or previous hop terminated, shut down dispatch
    if (count == 0u || state.terminated != 0u) {
        state.terminated = 1u;
        indirect_args.workgroups_x = 0u;
        indirect_args.workgroups_y = 0u;
        indirect_args.workgroups_z = 0u;
    } else {
        indirect_args.workgroups_x = state.max_workgroups_x;
        indirect_args.workgroups_y = 1u;
        indirect_args.workgroups_z = 1u;

        // Advance level and reset active counter for next hop
        state.level = state.level + 1u;
        atomicStore(&state.active_count, 0u);
    }
}
