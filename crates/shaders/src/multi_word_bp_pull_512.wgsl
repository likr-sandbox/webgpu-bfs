// crates/shaders/src/multi_word_bp_pull_512.wgsl
// Multi-Word Bit-Parallel Pull Traversal Kernel for 512 sources (16 words per vertex)
// Milestone M19: 8 Multi-Bank Distance Storage Buffers & Zero-Atomic Direct Writing

const W_k: u32 = 16u;
const W_v: u32 = 8u;
const WG_SIZE: u32 = 128u;

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
@group(0) @binding(6) var<storage, read_write> active_count: atomic<u32>;

// 8 Distance Banks (each holding 64 sources, <= 128 MB for |V| <= 524,288)
@group(0) @binding(7)  var<storage, read_write> dist_bank0: array<u32>;
@group(0) @binding(8)  var<storage, read_write> dist_bank1: array<u32>;
@group(0) @binding(9)  var<storage, read_write> dist_bank2: array<u32>;
@group(0) @binding(10) var<storage, read_write> dist_bank3: array<u32>;
@group(0) @binding(11) var<storage, read_write> dist_bank4: array<u32>;
@group(0) @binding(12) var<storage, read_write> dist_bank5: array<u32>;
@group(0) @binding(13) var<storage, read_write> dist_bank6: array<u32>;
@group(0) @binding(14) var<storage, read_write> dist_bank7: array<u32>;

var<workgroup> local_counts: array<u32, WG_SIZE>;

fn write_distance_bank(bank: u32, idx: u32, level: u32) {
    switch (bank) {
        case 0u: { dist_bank0[idx] = level; }
        case 1u: { dist_bank1[idx] = level; }
        case 2u: { dist_bank2[idx] = level; }
        case 3u: { dist_bank3[idx] = level; }
        case 4u: { dist_bank4[idx] = level; }
        case 5u: { dist_bank5[idx] = level; }
        case 6u: { dist_bank6[idx] = level; }
        case 7u: { dist_bank7[idx] = level; }
        default: {}
    }
}

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

        for (var k = local_id.x; k < params.words_per_vertex; k = k + W_k) {
            let state_idx = v_base + k;
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

            let needed = (~vis) & word_mask;

            if (needed == 0u) {
                next_frontier[state_idx] = 0u;
            } else {
                var cand: u32 = 0u;

                for (var edge = start_edge; edge < end_edge; edge = edge + 1u) {
                    let u = col_ind[edge];
                    let u_frontier = curr_frontier[u * params.words_per_vertex + k];
                    cand = cand | u_frontier;

                    if ((cand & needed) == needed) {
                        break;
                    }
                }

                let newly = cand & needed;

                if (newly != 0u) {
                    visited[state_idx] = vis | newly;
                    next_frontier[state_idx] = newly;

                    var bits = newly;
                    let bank = k >> 1u;
                    let word_in_bank = k & 1u;
                    while (bits != 0u) {
                        let bit_pos = countTrailingZeros(bits);
                        let local_s = word_in_bank * 32u + bit_pos;
                        let dist_idx = local_s * params.num_vertices + v;
                        write_distance_bank(bank, dist_idx, params.current_level + 1u);
                        bits = bits & (bits - 1u);
                    }

                    thread_new_count = thread_new_count + countOneBits(newly);
                } else {
                    next_frontier[state_idx] = 0u;
                }
            }
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
