// crates/shaders/src/multi_word_bp_pull_512.wgsl
// Multi-Word Bit-Parallel Pull Traversal Kernel for 512 sources (16 words per vertex)
// Milestone M19: 8 Multi-Bank Distance Storage Buffers & Zero-Atomic Direct Writing
// Milestone M23 Optimizations:
// 1. Elimination of 8 tree-reduction workgroupBarrier() calls via direct atomicStore(&active_count, 1u)
// 2. Early continue for inactive sources (src_start >= params.total_sources)
// 3. Dirty-Check conditional zero-clearing on Ping-Pong next_frontier
// 4. Distance bank write hoisting (loop-external switch dispatch)

const W_k: u32 = 16u;
const W_v: u32 = 8u;

struct Params {
    num_vertices: u32,
    words_per_vertex: u32,
    current_level: u32,
    total_sources: u32,
    dispatch_x: u32,
    bank_shift: u32,
    bank_mask: u32,
    pad0: u32,
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

fn write_bank_bits(k: u32, mut_bits: u32, v: u32, nv: u32, cur_level: u32) {
    let s_base = k * 32u;
    let is_configured = params.pad0 == 1u || params.bank_shift > 0u;
    let b_shift = select(6u, params.bank_shift, is_configured);
    let b_mask = select(63u, params.bank_mask, is_configured);
    var bits = mut_bits;
    while (bits != 0u) {
        let bit_pos = countTrailingZeros(bits);
        bits = bits & (bits - 1u);
        let s_pass = s_base + bit_pos;
        let bank = s_pass >> b_shift;
        let s_in_bank = s_pass & b_mask;
        let idx = s_in_bank * nv + v;
        switch (bank) {
            case 0u: { dist_bank0[idx] = cur_level; }
            case 1u: { dist_bank1[idx] = cur_level; }
            case 2u: { dist_bank2[idx] = cur_level; }
            case 3u: { dist_bank3[idx] = cur_level; }
            case 4u: { dist_bank4[idx] = cur_level; }
            case 5u: { dist_bank5[idx] = cur_level; }
            case 6u: { dist_bank6[idx] = cur_level; }
            case 7u: { dist_bank7[idx] = cur_level; }
            default: {}
        }
    }
}

@compute @workgroup_size(W_k, W_v, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let wg_id = group_id.x + group_id.y * params.dispatch_x;
    let v = wg_id * W_v + local_id.y;
    var thread_new_count: u32 = 0u;

    if (v < params.num_vertices) {
        let v_base = v * params.words_per_vertex;
        let start_edge = row_ptr[v];
        let end_edge = row_ptr[v + 1u];

        for (var k = local_id.x; k < params.words_per_vertex; k = k + W_k) {
            let src_start = k * 32u;
            if (src_start >= params.total_sources) {
                continue;
            }

            let state_idx = v_base + k;
            let vis = visited[state_idx];

            var word_mask: u32 = 0xFFFFFFFFu;
            let rem = params.total_sources - src_start;
            if (rem < 32u) {
                word_mask = (1u << rem) - 1u;
            }

            let needed = (~vis) & word_mask;

            if (needed == 0u) {
                if (next_frontier[state_idx] != 0u) {
                    next_frontier[state_idx] = 0u;
                }
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

                    write_bank_bits(k, newly, v, params.num_vertices, params.current_level + 1u);

                    thread_new_count = thread_new_count + countOneBits(newly);
                } else {
                    if (next_frontier[state_idx] != 0u) {
                        next_frontier[state_idx] = 0u;
                    }
                }
            }
        }
    }

    if (thread_new_count > 0u) {
        atomicStore(&active_count, 1u);
    }
}
