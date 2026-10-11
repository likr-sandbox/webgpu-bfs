// crates/shaders/src/sparse_bp_push.wgsl
// Multi-Source Bit-Parallel Sparse Push Traversal (Pass 1)
// Expands active frontier vertices from queue_in into next_candidate,
// and deduplicates touched vertices into candidate_queue.

struct PushParams {
    num_vertices: u32,
    current_level: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: PushParams;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_ind: array<u32>;
@group(0) @binding(3) var<storage, read> queue_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> curr_frontier: array<u32>; // size: 2 * num_vertices
@group(0) @binding(5) var<storage, read_write> next_candidate: array<atomic<u32>>; // size: 2 * num_vertices
@group(0) @binding(6) var<storage, read_write> touched: array<atomic<u32>>; // size: num_vertices
@group(0) @binding(7) var<storage, read_write> candidate_queue: array<u32>; // size: num_vertices
@group(0) @binding(8) var<storage, read_write> queue_sizes: array<atomic<u32>>; // [in_size, out_size, cand_size, reserved]

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let in_size = atomicLoad(&queue_sizes[0]);
    if (tid >= in_size) {
        return;
    }

    let u = queue_in[tid];
    if (u >= params.num_vertices) {
        return;
    }

    let idx0 = u * 2u;
    let idx1 = u * 2u + 1u;
    let f0 = curr_frontier[idx0];
    let f1 = curr_frontier[idx1];

    // Clear curr_frontier in-place so buffer is clean when ping-ponged
    curr_frontier[idx0] = 0u;
    curr_frontier[idx1] = 0u;

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

        // Deduplicated insertion into candidate_queue
        if (atomicExchange(&touched[v], 1u) == 0u) {
            let slot = atomicAdd(&queue_sizes[2], 1u);
            candidate_queue[slot] = v;
        }
    }
}
