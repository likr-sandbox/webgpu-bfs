// crates/shaders/src/multi_word_bp_filter.wgsl
// Multi-Word Bit-Parallel Filter & Distance Update (128 - 2048 Sources, 2D Warp-Tiled)
//
// Evaluates newly discovered nodes: newly = next_candidate & (~visited) & word_mask.
// Updates distances, updates visited, prepares next_frontier, resets next_candidate to 0 in-place,
// and reduces active_count across the workgroup.

override W_k: u32 = 8u;
override W_v: u32 = 16u;
override WG_SIZE: u32 = 128u;

struct FilterParams {
    num_vertices: u32,
    words_per_vertex: u32,
    current_level: u32,
    chunk_start: u32,
    total_sources: u32,
    dispatch_x: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: FilterParams;
@group(0) @binding(1) var<storage, read_write> next_candidate: array<u32>;
@group(0) @binding(2) var<storage, read_write> visited: array<u32>;
@group(0) @binding(3) var<storage, read_write> next_frontier: array<u32>;
@group(0) @binding(4) var<storage, read_write> distance: array<u32>;
@group(0) @binding(5) var<storage, read_write> active_count: atomic<u32>;

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

        for (var k = local_id.x; k < params.words_per_vertex; k = k + W_k) {
            let state_idx = v_base + k;
            let cand = next_candidate[state_idx];
            let vis = visited[state_idx];

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

            let newly = cand & (~vis) & word_mask;

            if (newly != 0u) {
                visited[state_idx] = vis | newly;
                next_frontier[state_idx] = newly;

                var bits = newly;
                while (bits != 0u) {
                    let bit_pos = countTrailingZeros(bits);
                    let s = params.chunk_start + src_start + bit_pos;
                    distance[s * params.num_vertices + v] = params.current_level + 1u;
                    bits = bits & (bits - 1u);
                }

                thread_new_count = thread_new_count + countOneBits(newly);
            } else {
                next_frontier[state_idx] = 0u;
            }

            // In-place reset of candidate word for the next hop
            next_candidate[state_idx] = 0u;
        }
    }

    local_counts[local_index] = thread_new_count;
    workgroupBarrier();

    for (var stride: u32 = WG_SIZE >> 1u; stride > 0u; stride = stride >> 1u) {
        if (local_index < stride) {
            local_counts[local_index] = local_counts[local_index] + local_counts[local_index + stride];
        }
        workgroupBarrier();
    }

    if (local_index == 0u && local_counts[0] > 0u) {
        atomicAdd(&active_count, local_counts[0]);
    }
}
