// Bit-Parallel Push Traversal (32 Sources packed into u32)
struct Params {
    num_vertices: u32,
    current_level: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_ind: array<u32>;
@group(0) @binding(3) var<storage, read> curr_frontier: array<u32>;
@group(0) @binding(4) var<storage, read_write> next_candidate: array<atomic<u32>>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u = global_id.x;
    if (u >= params.num_vertices) {
        return;
    }

    let f_u = curr_frontier[u];
    if (f_u == 0u) {
        return;
    }

    let start_edge = row_ptr[u];
    let end_edge = row_ptr[u + 1u];

    for (var edge = start_edge; edge < end_edge; edge = edge + 1u) {
        let v = col_ind[edge];
        atomicOr(&next_candidate[v], f_u);
    }
}
