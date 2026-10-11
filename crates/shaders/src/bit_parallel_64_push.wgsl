// Bit-Parallel Push Traversal (64 Sources packed into two u32 words per vertex)
struct Params {
    num_vertices: u32,
    current_level: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_ind: array<u32>;
@group(0) @binding(3) var<storage, read> curr_frontier: array<u32>; // size 2 * num_vertices
@group(0) @binding(4) var<storage, read_write> next_candidate: array<atomic<u32>>; // size 2 * num_vertices

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u = global_id.x;
    if (u >= params.num_vertices) {
        return;
    }

    let idx0 = u * 2u;
    let idx1 = u * 2u + 1u;

    let f0 = curr_frontier[idx0];
    let f1 = curr_frontier[idx1];

    // If neither word is active, skip all outgoing edges
    if (f0 == 0u && f1 == 0u) {
        return;
    }

    let start_edge = row_ptr[u];
    let end_edge = row_ptr[u + 1u];

    for (var edge = start_edge; edge < end_edge; edge = edge + 1u) {
        let v = col_ind[edge];
        let target_idx0 = v * 2u;
        let target_idx1 = v * 2u + 1u;

        if (f0 != 0u) {
            atomicOr(&next_candidate[target_idx0], f0);
        }
        if (f1 != 0u) {
            atomicOr(&next_candidate[target_idx1], f1);
        }
    }
}
