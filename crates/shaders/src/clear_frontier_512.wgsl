// crates/shaders/src/clear_frontier_512.wgsl
// Conditional Reset of Stale Input Frontier Bits Following Pull Hop

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

@group(0) @binding(0) var<uniform> uniforms: PushUniforms;
@group(0) @binding(1) var<uniform> state: HybridIterationState;
@group(0) @binding(4) var<storage, read_write> queues: array<u32>;
@group(0) @binding(5) var<storage, read_write> frontiers: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> queue_sizes: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let nv = uniforms.num_vertices;
    let in_slot = state.ping_pong_out;
    let in_count = atomicLoad(&queue_sizes[in_slot]);

    let idx = global_id.x;
    if (idx < in_count) {
        let in_queue_base = in_slot * nv;
        let in_frontier_base = in_slot * (16u * nv);

        let u = queues[in_queue_base + idx];
        if (u < nv) {
            let u_base = in_frontier_base + u * 16u;
            for (var w = 0u; w < 16u; w = w + 1u) {
                atomicStore(&frontiers[u_base + w], 0u);
            }
        }
    }
}
