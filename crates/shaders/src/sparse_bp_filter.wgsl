// crates/shaders/src/sparse_bp_filter.wgsl
// Multi-Source Bit-Parallel Sparse Filter & Distance Update (Pass 2)
// Inspects touched candidate vertices in candidate_queue,
// updates visited and distance matrices, resets touched & next_candidate in-place,
// and enqueues newly-reached vertices into queue_out.

struct FilterParams {
    num_vertices: u32,
    current_level: u32,
    chunk_start: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: FilterParams;
@group(0) @binding(1) var<storage, read> candidate_queue: array<u32>; // size: num_vertices
@group(0) @binding(2) var<storage, read_write> next_candidate: array<u32>; // size: 2 * num_vertices
@group(0) @binding(3) var<storage, read_write> visited: array<u32>; // size: 2 * num_vertices
@group(0) @binding(4) var<storage, read_write> next_frontier: array<u32>; // size: 2 * num_vertices
@group(0) @binding(5) var<storage, read_write> distance: array<u32>; // size: 64 * num_vertices
@group(0) @binding(6) var<storage, read_write> touched: array<u32>; // size: num_vertices
@group(0) @binding(7) var<storage, read_write> queue_out: array<u32>; // size: num_vertices
@group(0) @binding(8) var<storage, read_write> queue_sizes: array<atomic<u32>>; // [in_size, out_size, cand_size, reserved]

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let cand_size = atomicLoad(&queue_sizes[2]);
    if (tid >= cand_size) {
        return;
    }

    let v = candidate_queue[tid];
    if (v >= params.num_vertices) {
        return;
    }

    let idx0 = v * 2u;
    let idx1 = v * 2u + 1u;

    let cand0 = next_candidate[idx0];
    let cand1 = next_candidate[idx1];
    let old_vis0 = visited[idx0];
    let old_vis1 = visited[idx1];

    let newly_reached_0 = cand0 & (~old_vis0);
    let newly_reached_1 = cand1 & (~old_vis1);

    var has_new = false;

    if (newly_reached_0 != 0u) {
        visited[idx0] = old_vis0 | newly_reached_0;
        next_frontier[idx0] = newly_reached_0;

        var bits = newly_reached_0;
        while (bits != 0u) {
            let bit_pos = countTrailingZeros(bits);
            let s = params.chunk_start + bit_pos;
            distance[s * params.num_vertices + v] = params.current_level + 1u;
            bits = bits & (bits - 1u);
        }
        has_new = true;
    } else {
        next_frontier[idx0] = 0u;
    }

    if (newly_reached_1 != 0u) {
        visited[idx1] = old_vis1 | newly_reached_1;
        next_frontier[idx1] = newly_reached_1;

        var bits = newly_reached_1;
        while (bits != 0u) {
            let bit_pos = countTrailingZeros(bits);
            let s = params.chunk_start + 32u + bit_pos;
            distance[s * params.num_vertices + v] = params.current_level + 1u;
            bits = bits & (bits - 1u);
        }
        has_new = true;
    } else {
        next_frontier[idx1] = 0u;
    }

    // Reset touched and next_candidate in-place for this candidate vertex
    next_candidate[idx0] = 0u;
    next_candidate[idx1] = 0u;
    touched[v] = 0u;

    // Enqueue v into queue_out for hop h+1 if newly reached
    if (has_new) {
        let slot = atomicAdd(&queue_sizes[1], 1u);
        queue_out[slot] = v;
    }
}
