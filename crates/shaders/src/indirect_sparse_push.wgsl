// crates/shaders/src/indirect_sparse_push.wgsl
// Degree-Aware Collaborative Hub Multi-Source Bit-Parallel Sparse Queue Push Traversal Kernel

struct PushUniforms {
    num_vertices: u32,
    chunk_start: u32,      // Global source offset (e.g. 0, 64, 128...)
    hub_threshold: u32,    // Degree threshold for collaborative workgroup expansion (e.g. 128u or 256u)
    num_words: u32,        // Number of active words (e.g. 1 or 2)
};

struct IterationState {
    level: u32,            // Current BFS hop distance (1, 2, 3...)
    terminated: u32,       // 0 = active, 1 = terminated
    ping_pong_in: u32,     // Source buffer index (0 or 1)
    ping_pong_out: u32,    // Destination buffer index (1 or 0)
};

@group(0) @binding(0) var<uniform> uniforms: PushUniforms;
@group(0) @binding(1) var<uniform> state: IterationState;
@group(0) @binding(2) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> col_ind: array<u32>;
@group(0) @binding(4) var<storage, read_write> queues: array<u32>;
@group(0) @binding(5) var<storage, read_write> frontiers: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> visited: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> distance: array<u32>;
@group(0) @binding(8) var<storage, read_write> enqueued: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> queue_sizes: array<atomic<u32>>;

const SUB_WARP_SIZE: u32 = 4u;
const NUM_SUB_WARPS: u32 = 16u;
const WG_SIZE: u32 = 64u;
const WG_QUEUE_CAP: u32 = 2048u;
const SUB_QUEUE_CAP: u32 = 512u; // WG_QUEUE_CAP / 4u
const DEFAULT_HUB_THRESHOLD: u32 = 128u;
const ULTRA_HUB_THRESHOLD: u32 = 1024u;
const MAX_WORDS: u32 = 2u;

var<workgroup> s_f: array<u32, 32>; // NUM_SUB_WARPS * MAX_WORDS = 16 * 2 = 32
var<workgroup> s_active_mask: array<atomic<u32>, NUM_SUB_WARPS>;
var<workgroup> s_start_edge: array<u32, NUM_SUB_WARPS>;
var<workgroup> s_end_edge: array<u32, NUM_SUB_WARPS>;
var<workgroup> s_has_hub: atomic<u32>;

var<workgroup> wg_queue_items: array<u32, WG_QUEUE_CAP>;
var<workgroup> s_sub_queue_count: array<atomic<u32>, 4>;
var<workgroup> s_sub_offsets: array<u32, 4>;
var<workgroup> s_sub_counts: array<u32, 4>;

fn write_distance_bits(w_offset: u32, mut_bits: u32, v: u32, nv: u32, cur_level: u32) {
    var bits = mut_bits;
    while (bits != 0u) {
        let bit_pos = countTrailingZeros(bits);
        distance[(w_offset + bit_pos) * nv + v] = cur_level;
        bits = bits & (bits - 1u);
    }
}

fn process_neighbor(
    v: u32,
    sub_idx: u32,
    active_mask: u32,
    nv: u32,
    cur_level: u32,
    out_frontier_base: u32,
    out_queue_base: u32,
    out_slot: u32,
    qid: u32,
) {
    let v_vis_base = v * MAX_WORDS;
    var newly_discovered = false;
    var words_to_process = active_mask;

    while (words_to_process != 0u) {
        let w = countTrailingZeros(words_to_process);
        words_to_process = words_to_process & (words_to_process - 1u);

        let f = s_f[sub_idx * MAX_WORDS + w];
        let vis = atomicLoad(&visited[v_vis_base + w]);
        let needed = f & (~vis);

        if (needed == 0u) {
            continue;
        }

        let prev = atomicOr(&visited[v_vis_base + w], needed);
        let newly = needed & (~prev);

        if (newly == 0u) {
            continue;
        }

        newly_discovered = true;

        let w_offset = uniforms.chunk_start + w * 32u;
        write_distance_bits(w_offset, newly, v, nv, cur_level);

        let v_out_idx = out_frontier_base + v * MAX_WORDS + w;
        atomicOr(&frontiers[v_out_idx], newly);
    }

    if (newly_discovered) {
        if (atomicLoad(&enqueued[v]) != cur_level) {
            if (atomicExchange(&enqueued[v], cur_level) != cur_level) {
                let sub_slot = atomicAdd(&s_sub_queue_count[qid], 1u);
                if (sub_slot < SUB_QUEUE_CAP) {
                    wg_queue_items[qid * SUB_QUEUE_CAP + sub_slot] = v;
                } else {
                    let slot = atomicAdd(&queue_sizes[out_slot], 1u);
                    queues[out_queue_base + slot] = v;
                }
            }
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>
) {
    let nv = uniforms.num_vertices;
    let in_slot = state.ping_pong_out;
    let out_slot = state.ping_pong_in;

    let in_size = atomicLoad(&queue_sizes[in_slot]);
    let in_queue_base = in_slot * nv;
    let out_queue_base = out_slot * nv;
    let in_frontier_base = in_slot * (MAX_WORDS * nv);
    let out_frontier_base = out_slot * (MAX_WORDS * nv);

    let sub_id = local_id.x / SUB_WARP_SIZE;
    let sub_lane = local_id.x % SUB_WARP_SIZE;
    let qid = local_id.x & 3u;

    let hub_thresh = select(DEFAULT_HUB_THRESHOLD, uniforms.hub_threshold, uniforms.hub_threshold > 0u);

    if (local_id.x < NUM_SUB_WARPS) {
        atomicStore(&s_active_mask[local_id.x], 0u);
        s_start_edge[local_id.x] = 0u;
        s_end_edge[local_id.x] = 0u;
    }
    if (local_id.x < 4u) {
        atomicStore(&s_sub_queue_count[local_id.x], 0u);
    }
    if (local_id.x == 0u) {
        atomicStore(&s_has_hub, 0u);
    }
    workgroupBarrier();

    // Milestone M24: 2D Grid Dispatch support for queues > 65,535 workgroups (> 1,048,560 vertices)
    let stride = select(num_groups.x, 256u, num_groups.x == 0u);
    let wg_id = group_id.x + group_id.y * stride;

    // Phase 0: Sub-warps cooperatively load vertices from input queue into LDS
    let in_queue_idx = wg_id * NUM_SUB_WARPS + sub_id;
    if (in_queue_idx < in_size) {
        let u = queues[in_queue_base + in_queue_idx];
        if (u < nv) {
            let u_base = in_frontier_base + u * MAX_WORDS;
            var lane_mask = 0u;
            if (sub_lane < MAX_WORDS) {
                let w = sub_lane;
                if (w < uniforms.num_words) {
                    let f = atomicExchange(&frontiers[u_base + w], 0u);
                    s_f[sub_id * MAX_WORDS + w] = f;
                    if (f != 0u) {
                        lane_mask = lane_mask | (1u << w);
                    }
                } else {
                    s_f[sub_id * MAX_WORDS + w] = 0u;
                }
            }
            if (lane_mask != 0u) {
                atomicOr(&s_active_mask[sub_id], lane_mask);
            }
            if (sub_lane == 0u) {
                let start_e = row_ptr[u];
                let end_e = row_ptr[u + 1u];
                s_start_edge[sub_id] = start_e;
                s_end_edge[sub_id] = end_e;
                if ((end_e - start_e) >= hub_thresh) {
                    atomicStore(&s_has_hub, 1u);
                }
            }
        }
    }
    workgroupBarrier();

    let cur_level = state.level;

    // Phase 1: Collaborative Hub Processing (Tier 1)
    if (atomicLoad(&s_has_hub) != 0u) {
        for (var s = 0u; s < NUM_SUB_WARPS; s = s + 1u) {
            let h_start = s_start_edge[s];
            let h_end = s_end_edge[s];
            let h_deg = h_end - h_start;
            let h_mask = atomicLoad(&s_active_mask[s]);

            if (h_deg >= hub_thresh && h_mask != 0u) {
                if (h_deg >= ULTRA_HUB_THRESHOLD) {
                    // Ultra-hub: 4x unrolled stride 256 traversal
                    for (var edge_base = h_start; edge_base < h_end; edge_base = edge_base + 256u) {
                        for (var k = 0u; k < 4u; k = k + 1u) {
                            let edge = edge_base + k * WG_SIZE + local_id.x;
                            if (edge < h_end) {
                                process_neighbor(
                                    col_ind[edge],
                                    s,
                                    h_mask,
                                    nv,
                                    cur_level,
                                    out_frontier_base,
                                    out_queue_base,
                                    out_slot,
                                    qid,
                                );
                            }
                        }
                    }
                } else {
                    // Standard hub: stride 64 traversal
                    for (var edge = h_start + local_id.x; edge < h_end; edge = edge + WG_SIZE) {
                        process_neighbor(
                            col_ind[edge],
                            s,
                            h_mask,
                            nv,
                            cur_level,
                            out_frontier_base,
                            out_queue_base,
                            out_slot,
                            qid,
                        );
                    }
                }
            }
        }
    }
    workgroupBarrier();

    // Phase 2: Independent Sub-Warp Processing for Low-Degree Vertices (Tier 2)
    let my_start = s_start_edge[sub_id];
    let my_end = s_end_edge[sub_id];
    let my_deg = my_end - my_start;
    let my_mask = atomicLoad(&s_active_mask[sub_id]);

    if (my_deg < hub_thresh && my_mask != 0u) {
        for (var edge = my_start + sub_lane; edge < my_end; edge = edge + SUB_WARP_SIZE) {
            process_neighbor(
                col_ind[edge],
                sub_id,
                my_mask,
                nv,
                cur_level,
                out_frontier_base,
                out_queue_base,
                out_slot,
                qid,
            );
        }
    }
    workgroupBarrier();

    // Phase 3: Workgroup Shared Queue Aggregation and Cooperative Global Flush
    if (local_id.x == 0u) {
        let c0 = min(atomicLoad(&s_sub_queue_count[0]), SUB_QUEUE_CAP);
        let c1 = min(atomicLoad(&s_sub_queue_count[1]), SUB_QUEUE_CAP);
        let c2 = min(atomicLoad(&s_sub_queue_count[2]), SUB_QUEUE_CAP);
        let c3 = min(atomicLoad(&s_sub_queue_count[3]), SUB_QUEUE_CAP);
        s_sub_counts[0] = c0;
        s_sub_counts[1] = c1;
        s_sub_counts[2] = c2;
        s_sub_counts[3] = c3;
        let total_buffered = c0 + c1 + c2 + c3;
        var g_off = 0u;
        if (total_buffered > 0u) {
            g_off = atomicAdd(&queue_sizes[out_slot], total_buffered);
        }
        s_sub_offsets[0] = g_off;
        s_sub_offsets[1] = g_off + c0;
        s_sub_offsets[2] = g_off + c0 + c1;
        s_sub_offsets[3] = g_off + c0 + c1 + c2;
    }
    workgroupBarrier();

    for (var q = 0u; q < 4u; q = q + 1u) {
        let count = s_sub_counts[q];
        let g_base = s_sub_offsets[q];
        let q_base = q * SUB_QUEUE_CAP;
        for (var i = local_id.x; i < count; i = i + WG_SIZE) {
            queues[out_queue_base + g_base + i] = wg_queue_items[q_base + i];
        }
    }
}
