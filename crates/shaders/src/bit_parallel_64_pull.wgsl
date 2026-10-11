// Single-Pass 64-Source Bit-Parallel Pull Traversal
// Combines incoming neighbor aggregation, unvisited filtering, and distance matrix updates
// into a SINGLE compute pass with zero intermediate candidate buffer and zero graph atomics.

struct Params {
    num_vertices: u32,
    current_level: u32,
    chunk_start: u32,
    mask_low: u32,
    mask_high: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_ind: array<u32>;
@group(0) @binding(3) var<storage, read> curr_frontier: array<u32>; // 2 * num_vertices
@group(0) @binding(4) var<storage, read_write> visited: array<u32>; // 2 * num_vertices
@group(0) @binding(5) var<storage, read_write> next_frontier: array<u32>; // 2 * num_vertices
@group(0) @binding(6) var<storage, read_write> distance: array<u32>; // 64 * num_vertices
@group(0) @binding(7) var<storage, read_write> active_count: atomic<u32>;

var<workgroup> local_counts: array<u32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let v = global_id.x;
    var total_new: u32 = 0u;

    if (v < params.num_vertices) {
        let idx0 = v * 2u;
        let idx1 = v * 2u + 1u;
        let vis0 = visited[idx0];
        let vis1 = visited[idx1];
        let needed0 = (~vis0) & params.mask_low;
        let needed1 = (~vis1) & params.mask_high;

        if (needed0 == 0u && needed1 == 0u) {
            next_frontier[idx0] = 0u;
            next_frontier[idx1] = 0u;
        } else {
            let start = row_ptr[v];
            let end = row_ptr[v + 1u];
            var cand0: u32 = 0u;
            var cand1: u32 = 0u;

            for (var e = start; e < end; e = e + 1u) {
                let u = col_ind[e];
                let u_idx0 = u * 2u;
                let u_idx1 = u * 2u + 1u;

                cand0 = cand0 | curr_frontier[u_idx0];
                cand1 = cand1 | curr_frontier[u_idx1];

                if ((cand0 & needed0) == needed0 && (cand1 & needed1) == needed1) {
                    break;
                }
            }

            let newly0 = cand0 & needed0;
            let newly1 = cand1 & needed1;

            if (newly0 != 0u) {
                visited[idx0] = vis0 | newly0;
                next_frontier[idx0] = newly0;
                var bits = newly0;
                while (bits != 0u) {
                    let s = params.chunk_start + countTrailingZeros(bits);
                    distance[s * params.num_vertices + v] = params.current_level + 1u;
                    bits = bits & (bits - 1u);
                }
            } else {
                next_frontier[idx0] = 0u;
            }

            if (newly1 != 0u) {
                visited[idx1] = vis1 | newly1;
                next_frontier[idx1] = newly1;
                var bits = newly1;
                while (bits != 0u) {
                    let s = params.chunk_start + 32u + countTrailingZeros(bits);
                    distance[s * params.num_vertices + v] = params.current_level + 1u;
                    bits = bits & (bits - 1u);
                }
            } else {
                next_frontier[idx1] = 0u;
            }

            total_new = countOneBits(newly0) + countOneBits(newly1);
        }
    }

    local_counts[local_id.x] = total_new;
    workgroupBarrier();

    for (var stride: u32 = 64u; stride > 0u; stride = stride >> 1u) {
        if (local_id.x < stride) {
            local_counts[local_id.x] = local_counts[local_id.x] + local_counts[local_id.x + stride];
        }
        workgroupBarrier();
    }

    if (local_id.x == 0u && local_counts[0] > 0u) {
        atomicAdd(&active_count, local_counts[0]);
    }
}
