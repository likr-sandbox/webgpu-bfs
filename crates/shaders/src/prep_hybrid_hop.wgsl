// crates/shaders/src/prep_hybrid_hop.wgsl
// GPU-Autonomous Direction-Optimizing Hybrid Hop Preparation Kernel

struct PushUniforms {
    num_vertices: u32,
    chunk_start: u32,
    hub_threshold: u32,
    num_words: u32,
    density_threshold: u32,
    bank_shift: u32,
    bank_mask: u32,
    pad0: u32,
};

struct HybridIterationState {
    level: u32,
    terminated: u32,
    ping_pong_in: u32,
    ping_pong_out: u32,
    current_direction: u32,
    prev_direction: u32,
    pad0: u32,
    pad1: u32,
};

struct HybridIndirectArgs {
    push_x: u32,
    push_y: u32,
    push_z: u32,
    pad0: u32,
    pull_x: u32,
    pull_y: u32,
    pull_z: u32,
    pad1: u32,
    clear_x: u32,
    clear_y: u32,
    clear_z: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> uniforms: PushUniforms;
@group(0) @binding(1) var<storage, read_write> state: HybridIterationState;
@group(0) @binding(2) var<storage, read_write> indirect_args: HybridIndirectArgs;
@group(0) @binding(3) var<storage, read_write> queue_sizes: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    if (state.terminated != 0u) {
        indirect_args.push_x = 0u; indirect_args.push_y = 0u; indirect_args.push_z = 0u;
        indirect_args.pull_x = 0u; indirect_args.pull_y = 0u; indirect_args.pull_z = 0u;
        indirect_args.clear_x = 0u; indirect_args.clear_y = 0u; indirect_args.clear_z = 0u;
        return;
    }

    let in_slot = state.ping_pong_in;
    let out_slot = state.ping_pong_out;
    let in_count = atomicLoad(&queue_sizes[in_slot]);

    // Check frontier exhaustion
    if (in_count == 0u) {
        state.terminated = 1u;
        indirect_args.push_x = 0u; indirect_args.push_y = 0u; indirect_args.push_z = 0u;
        indirect_args.pull_x = 0u; indirect_args.pull_y = 0u; indirect_args.pull_z = 0u;
        indirect_args.clear_x = 0u; indirect_args.clear_y = 0u; indirect_args.clear_z = 0u;
        return;
    }

    let nv = uniforms.num_vertices;
    let density_thresh = select(nv / 16u, uniforms.density_threshold, uniforms.density_threshold > 0u);

    // Reset destination queue counter for the upcoming hop
    atomicStore(&queue_sizes[out_slot], 0u);

    // Advance hop distance directly on GPU
    state.level = state.level + 1u;
    state.prev_direction = state.current_direction;

    if (in_count > density_thresh) {
        // DENSE FRONTIER -> PULL MODE
        state.current_direction = 1u;

        // Push receives 0 workgroups
        indirect_args.push_x = 0u;
        indirect_args.push_y = 0u;
        indirect_args.push_z = 0u;

        // Pull workgroups: 8 vertices per workgroup (16 threads per vertex)
        let total_pull_wgs = (nv + 7u) / 8u;
        if (total_pull_wgs > 65535u) {
            indirect_args.pull_x = 256u;
            indirect_args.pull_y = (total_pull_wgs + 255u) / 256u;
            indirect_args.pull_z = 1u;
        } else {
            indirect_args.pull_x = total_pull_wgs;
            indirect_args.pull_y = 1u;
            indirect_args.pull_z = 1u;
        }

        // Clear pass: clears in_count active vertices from frontiers[in_slot]
        let clear_wgs = (in_count + 63u) / 64u;
        indirect_args.clear_x = clear_wgs;
        indirect_args.clear_y = 1u;
        indirect_args.clear_z = 1u;
    } else {
        // SPARSE FRONTIER -> PUSH MODE
        state.current_direction = 0u;

        // Pull receives 0 workgroups
        indirect_args.pull_x = 0u;
        indirect_args.pull_y = 0u;
        indirect_args.pull_z = 0u;

        // Clear receives 0 workgroups (Push Phase 0 clears in_slot on the fly)
        indirect_args.clear_x = 0u;
        indirect_args.clear_y = 0u;
        indirect_args.clear_z = 0u;

        // Push workgroups: 16 vertices per workgroup (64 threads / 4 sub-warp lanes)
        let total_push_wgs = (in_count + 15u) / 16u;
        if (total_push_wgs > 65535u) {
            indirect_args.push_x = 256u;
            indirect_args.push_y = (total_push_wgs + 255u) / 256u;
            indirect_args.push_z = 1u;
        } else {
            indirect_args.push_x = total_push_wgs;
            indirect_args.push_y = 1u;
            indirect_args.push_z = 1u;
        }
    }

    // Toggle ping-pong pointers for next hop
    state.ping_pong_in = out_slot;
    state.ping_pong_out = in_slot;
}
