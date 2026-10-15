// crates/shaders/src/prep_sparse_push_hop.wgsl
// GPU-Autonomous Hop Preparation & Indirect Dispatch Kernel for Sparse Push BFS

struct IterationState {
    level: u32,            // Current BFS hop distance (1, 2, 3...)
    terminated: u32,       // 0 = active, 1 = terminated
    ping_pong_in: u32,     // 0 = queue_a is input, 1 = queue_b is input
    ping_pong_out: u32,    // 0 = queue_a is output, 1 = queue_b is output
};

struct IndirectArgs {
    workgroups_x: u32,     // Ceil(in_count / 16)
    workgroups_y: u32,     // 1
    workgroups_z: u32,     // 1
};

@group(0) @binding(0) var<storage, read_write> state: IterationState;
@group(0) @binding(1) var<storage, read_write> indirect_args: IndirectArgs;
@group(0) @binding(2) var<storage, read_write> queue_sizes: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    // If search already terminated, enforce zero workgroups for subsequent speculative passes
    if (state.terminated != 0u) {
        indirect_args.workgroups_x = 0u;
        indirect_args.workgroups_y = 0u;
        indirect_args.workgroups_z = 0u;
        return;
    }

    let in_slot = state.ping_pong_in;
    let out_slot = state.ping_pong_out;
    let in_count = atomicLoad(&queue_sizes[in_slot]);

    // Check frontier exhaustion
    if (in_count == 0u) {
        state.terminated = 1u;
        indirect_args.workgroups_x = 0u;
        indirect_args.workgroups_y = 0u;
        indirect_args.workgroups_z = 0u;
    } else {
        // Workgroup processes 16 vertices (64 threads / 4 sub-warp lanes)
        let total_wgs = (in_count + 15u) / 16u;
        if (total_wgs > 65535u) {
            indirect_args.workgroups_x = 256u;
            indirect_args.workgroups_y = (total_wgs + 255u) / 256u;
            indirect_args.workgroups_z = 1u;
        } else {
            indirect_args.workgroups_x = total_wgs;
            indirect_args.workgroups_y = 1u;
            indirect_args.workgroups_z = 1u;
        }

        // Reset destination queue counter for the upcoming hop
        atomicStore(&queue_sizes[out_slot], 0u);

        // Advance hop distance directly on GPU
        state.level = state.level + 1u;

        // Toggle ping-pong pointers for next hop
        state.ping_pong_in = out_slot;
        state.ping_pong_out = in_slot;
    }
}
