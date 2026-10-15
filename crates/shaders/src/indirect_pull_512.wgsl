// crates/shaders/src/indirect_pull_512.wgsl
// Fused Multi-Word Pull Traversal with Workgroup Queue Aggregation for Hybrid MS-BFS

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
@group(0) @binding(2) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> col_ind: array<u32>;
@group(0) @binding(4) var<storage, read_write> queues: array<u32>;
@group(0) @binding(5) var<storage, read_write> frontiers: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> visited: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> enqueued: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> queue_sizes: array<atomic<u32>>;

@group(0) @binding(9)  var<storage, read_write> dist_bank0: array<u32>;
@group(0) @binding(10) var<storage, read_write> dist_bank1: array<u32>;
@group(0) @binding(11) var<storage, read_write> dist_bank2: array<u32>;
@group(0) @binding(12) var<storage, read_write> dist_bank3: array<u32>;
@group(0) @binding(13) var<storage, read_write> dist_bank4: array<u32>;
@group(0) @binding(14) var<storage, read_write> dist_bank5: array<u32>;
@group(0) @binding(15) var<storage, read_write> dist_bank6: array<u32>;
@group(0) @binding(16) var<storage, read_write> dist_bank7: array<u32>;

const W_k: u32 = 16u;
const W_v: u32 = 8u;

var<workgroup> s_v_active: array<atomic<u32>, 8>;

fn write_bank_bits(w: u32, mut_bits: u32, v: u32, nv: u32, cur_level: u32) {
    let s_base = w * 32u;
    let is_configured = uniforms.pad0 == 1u || uniforms.bank_shift > 0u;
    let b_shift = select(6u, uniforms.bank_shift, is_configured);
    let b_mask = select(63u, uniforms.bank_mask, is_configured);
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
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>
) {
    let nv = uniforms.num_vertices;
    let in_slot = state.ping_pong_out;
    let out_slot = state.ping_pong_in;
    let cur_level = state.level;

    let in_frontier_base = in_slot * (16u * nv);
    let out_frontier_base = out_slot * (16u * nv);
    let out_queue_base = out_slot * nv;

    let stride = select(num_groups.x, 256u, num_groups.x == 0u);
    let wg_id = group_id.x + group_id.y * stride;
    let v = wg_id * W_v + local_id.y;

    let k = local_id.x; // word index 0..15

    if (local_id.x < 8u && local_id.y == 0u) {
        atomicStore(&s_v_active[local_id.x], 0u);
    }
    workgroupBarrier();

    if (v < nv) {
        let state_idx = v * 16u + k;
        let start_edge = row_ptr[v];
        let end_edge = row_ptr[v + 1u];

        let vis = atomicLoad(&visited[state_idx]);
        let needed = ~vis;
        let out_idx = out_frontier_base + state_idx;

        if (needed == 0u) {
            // Vertex already visited for all sources in word k
            if (atomicLoad(&frontiers[out_idx]) != 0u) {
                atomicStore(&frontiers[out_idx], 0u);
            }
        } else {
            var cand: u32 = 0u;
            for (var edge = start_edge; edge < end_edge; edge = edge + 1u) {
                let u = col_ind[edge];
                let u_frontier = atomicLoad(&frontiers[in_frontier_base + u * 16u + k]);
                cand = cand | u_frontier;
                if ((cand & needed) == needed) {
                    break;
                }
            }

            let newly = cand & needed;
            if (newly != 0u) {
                atomicOr(&visited[state_idx], newly);
                atomicStore(&frontiers[out_idx], newly);
                write_bank_bits(k, newly, v, nv, cur_level);
                atomicStore(&s_v_active[local_id.y], 1u);
            } else {
                if (atomicLoad(&frontiers[out_idx]) != 0u) {
                    atomicStore(&frontiers[out_idx], 0u);
                }
            }
        }
    }
    workgroupBarrier();

    // Workgroup-aggregated queue enqueue
    if (local_id.x == 0u && local_id.y == 0u) {
        var active_mask: u32 = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            if (atomicLoad(&s_v_active[i]) != 0u) {
                active_mask = active_mask | (1u << i);
            }
        }
        let total_active = countOneBits(active_mask);
        if (total_active > 0u) {
            let base_slot = atomicAdd(&queue_sizes[out_slot], total_active);
            var slot_offset = 0u;
            var mask = active_mask;
            while (mask != 0u) {
                let bit = countTrailingZeros(mask);
                mask = mask & (mask - 1u);
                let vert = wg_id * W_v + bit;
                queues[out_queue_base + base_slot + slot_offset] = vert;
                atomicStore(&enqueued[vert], cur_level);
                slot_offset = slot_offset + 1u;
            }
        }
    }
}
