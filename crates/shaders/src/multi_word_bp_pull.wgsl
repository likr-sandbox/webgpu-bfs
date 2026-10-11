// crates/shaders/src/multi_word_bp_pull.wgsl
// Massive Multi-Word Bit-Parallel Pull Traversal (256 - 2048 Sources, 2D Warp-Tiled)
//
// Combines incoming neighbor aggregation, unvisited filtering, and distance matrix updates
// into a SINGLE compute pass with zero intermediate candidate buffer and zero graph atomics.
//
// 2D Workgroup Tiling:
// - W_k threads along the word dimension (X axis), where W_k = min(K, 32)
// - W_v vertices along the vertex dimension (Y axis)
// - Total workgroup invocations = WG_SIZE = W_k * W_v (e.g. 128 or 256)
//
// Memory Layout:
// - curr_frontier: flat vertex-major array<u32>, size = num_vertices * words_per_vertex
// - visited:       flat vertex-major array<u32>, size = num_vertices * words_per_vertex
// - next_frontier: flat vertex-major array<u32>, size = num_vertices * words_per_vertex
// - distance:      flat source-major array<u32>, size = total_sources * num_vertices

override W_k: u32 = 32u;
override W_v: u32 = 4u;
override WG_SIZE: u32 = 128u;

struct Params {
    num_vertices: u32,
    words_per_vertex: u32,
    current_level: u32,
    total_sources: u32,
    dispatch_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_ind: array<u32>;
@group(0) @binding(3) var<storage, read> curr_frontier: array<u32>;
@group(0) @binding(4) var<storage, read_write> visited: array<u32>;
@group(0) @binding(5) var<storage, read_write> next_frontier: array<u32>;
@group(0) @binding(6) var<storage, read_write> distance: array<u32>;
@group(0) @binding(7) var<storage, read_write> active_count: atomic<u32>;

var<workgroup> local_counts: array<u32, WG_SIZE>;

@compute @workgroup_size(W_k, W_v, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let wg_id = group_id.x + group_id.y * params.dispatch_x;
    let v = wg_id * W_v + local_id.y;
    var thread_new_count: u32 = 0u;

    if (v < params.num_vertices) {
        let v_base = v * params.words_per_vertex;
        let start_edge = row_ptr[v];
        let end_edge = row_ptr[v + 1u];

        // Process assigned word(s) for this vertex.
        // When W_k = min(K, 32), this loop executes:
        // - Exactly 1 iteration for K in {8, 16, 32}
        // - Exactly 2 iterations for K = 64 (k = local_id.x and k = local_id.x + 32)
        for (var k = local_id.x; k < params.words_per_vertex; k = k + W_k) {
            let state_idx = v_base + k;
            let vis = visited[state_idx];

            // Compute valid active source mask for this 32-source word
            var word_mask: u32 = 0u;
            let src_start = k * 32u;
            if (src_start < params.total_sources) {
                let rem = params.total_sources - src_start;
                if (rem >= 32u) {
                    word_mask = 0xFFFFFFFFu;
                } else {
                    word_mask = (1u << rem) - 1u;
                }
            }

            let needed = (~vis) & word_mask;

            // Early exit if all active sources for this word have already visited vertex v
            if (needed == 0u) {
                next_frontier[state_idx] = 0u;
            } else {
                var cand: u32 = 0u;

                // Pull traversal: aggregate frontier bits from all incoming neighbors u
                for (var edge = start_edge; edge < end_edge; edge = edge + 1u) {
                    let u = col_ind[edge];
                    let u_frontier = curr_frontier[u * params.words_per_vertex + k];
                    cand = cand | u_frontier;

                    // Early exit if all unvisited needed sources have been discovered
                    if ((cand & needed) == needed) {
                        break;
                    }
                }

                let newly = cand & needed;

                if (newly != 0u) {
                    visited[state_idx] = vis | newly;
                    next_frontier[state_idx] = newly;

                    // Brian Kernighan's bit extraction to update distances of newly reached sources
                    var bits = newly;
                    while (bits != 0u) {
                        let bit_pos = countTrailingZeros(bits);
                        let s = src_start + bit_pos;
                        if (s < params.total_sources) {
                            distance[s * params.num_vertices + v] = params.current_level + 1u;
                        }
                        bits = bits & (bits - 1u);
                    }

                    thread_new_count = thread_new_count + countOneBits(newly);
                } else {
                    next_frontier[state_idx] = 0u;
                }
            }
        }
    }

    // Workgroup parallel tree reduction to aggregate active counts across all threads
    local_counts[local_index] = thread_new_count;
    workgroupBarrier();

    for (var stride: u32 = WG_SIZE >> 1u; stride > 0u; stride = stride >> 1u) {
        if (local_index < stride) {
            local_counts[local_index] = local_counts[local_index] + local_counts[local_index + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 of the workgroup atomically increments global active counter
    if (local_index == 0u && local_counts[0] > 0u) {
        atomicAdd(&active_count, local_counts[0]);
    }
}
