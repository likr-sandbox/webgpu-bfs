// crates/shaders/src/indirect_sparse_push_256.wgsl
// Degree-Aware Collaborative Hub Multi-Word (8 words = 256 sources) Fused Indirect Sparse Push Traversal Kernel

struct PushUniforms {
    num_vertices: u32,
    chunk_start: u32,      // Global source offset (e.g. 0, 256...)
    hub_threshold: u32,    // Degree threshold for collaborative workgroup expansion (e.g. 128u or 256u)
    num_words: u32,        // Number of active words (e.g. 8)
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
@group(0) @binding(7) var<storage, read_write> enqueued: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> queue_sizes: array<atomic<u32>>;

// 4 distance banks (each holding 64 sources = 2 words, <= 72.17 MB for web-Stanford)
@group(0) @binding(9)  var<storage, read_write> dist_bank0: array<u32>;
@group(0) @binding(10) var<storage, read_write> dist_bank1: array<u32>;
@group(0) @binding(11) var<storage, read_write> dist_bank2: array<u32>;
@group(0) @binding(12) var<storage, read_write> dist_bank3: array<u32>;

const SUB_WARP_SIZE: u32 = 4u;
const NUM_SUB_WARPS: u32 = 16u;
const WG_SIZE: u32 = 64u;
const WG_QUEUE_CAP: u32 = 1024u;
const DEFAULT_HUB_THRESHOLD: u32 = 128u;
const MAX_WORDS: u32 = 8u;

var<workgroup> s_f: array<u32, 128>; // NUM_SUB_WARPS * MAX_WORDS = 16 * 8 = 128
var<workgroup> s_active_mask: array<u32, NUM_SUB_WARPS>;
var<workgroup> s_start_edge: array<u32, NUM_SUB_WARPS>;
var<workgroup> s_end_edge: array<u32, NUM_SUB_WARPS>;
var<workgroup> s_has_hub: atomic<u32>;

var<workgroup> wg_queue_items: array<u32, WG_QUEUE_CAP>;
var<workgroup> wg_queue_count: atomic<u32>;
var<workgroup> wg_global_offset: u32;

fn write_distance_bank(bank: u32, idx: u32, cur_level: u32) {
    switch (bank) {
        case 0u: { dist_bank0[idx] = cur_level; }
        case 1u: { dist_bank1[idx] = cur_level; }
        case 2u: { dist_bank2[idx] = cur_level; }
        case 3u: { dist_bank3[idx] = cur_level; }
        default: {}
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
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

    let hub_thresh = select(DEFAULT_HUB_THRESHOLD, uniforms.hub_threshold, uniforms.hub_threshold > 0u);

    if (local_id.x == 0u) {
        atomicStore(&wg_queue_count, 0u);
        atomicStore(&s_has_hub, 0u);
    }

    // Phase 0: Sub-warps load vertices from input queue into LDS
    if (sub_lane == 0u) {
        s_active_mask[sub_id] = 0u;
        s_start_edge[sub_id] = 0u;
        s_end_edge[sub_id] = 0u;

        let in_queue_idx = group_id.x * NUM_SUB_WARPS + sub_id;
        if (in_queue_idx < in_size) {
            let u = queues[in_queue_base + in_queue_idx];
            if (u < nv) {
                let u_base = in_frontier_base + u * MAX_WORDS;
                var mask = 0u;
                for (var w = 0u; w < MAX_WORDS; w = w + 1u) {
                    let f = atomicExchange(&frontiers[u_base + w], 0u);
                    s_f[sub_id * MAX_WORDS + w] = f;
                    if (f != 0u) {
                        mask = mask | (1u << w);
                    }
                }
                s_active_mask[sub_id] = mask;
                if (mask != 0u) {
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
    }
    workgroupBarrier();

    let cur_level = state.level;

    // Phase 1: Collaborative Hub Processing (Tier 1)
    if (atomicLoad(&s_has_hub) != 0u) {
        for (var s = 0u; s < NUM_SUB_WARPS; s = s + 1u) {
            let h_start = s_start_edge[s];
            let h_end = s_end_edge[s];
            let h_deg = h_end - h_start;
            let h_mask = s_active_mask[s];

            if (h_deg >= hub_thresh && h_mask != 0u) {
                for (var edge = h_start + local_id.x; edge < h_end; edge = edge + WG_SIZE) {
                    let v = col_ind[edge];
                    let v_vis_base = v * MAX_WORDS;

                    var newly_discovered = false;
                    var words_to_process = h_mask;

                    while (words_to_process != 0u) {
                        let w = countTrailingZeros(words_to_process);
                        words_to_process = words_to_process & (words_to_process - 1u);

                        let f = s_f[s * MAX_WORDS + w];
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

                        var bits = newly;
                        while (bits != 0u) {
                            let bit_pos = countTrailingZeros(bits);
                            let local_s = (w & 1u) * 32u + bit_pos;
                            let dist_idx = local_s * nv + v;
                            let bank = w >> 1u;
                            write_distance_bank(bank, dist_idx, cur_level);
                            bits = bits & (bits - 1u);
                        }

                        let v_out_idx = out_frontier_base + v * MAX_WORDS + w;
                        atomicOr(&frontiers[v_out_idx], newly);
                    }

                    if (newly_discovered) {
                        if (atomicLoad(&enqueued[v]) != cur_level) {
                            if (atomicExchange(&enqueued[v], cur_level) != cur_level) {
                                let local_slot = atomicAdd(&wg_queue_count, 1u);
                                if (local_slot < WG_QUEUE_CAP) {
                                    wg_queue_items[local_slot] = v;
                                } else {
                                    let slot = atomicAdd(&queue_sizes[out_slot], 1u);
                                    queues[out_queue_base + slot] = v;
                                }
                            }
                        }
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
    let my_mask = s_active_mask[sub_id];

    if (my_deg < hub_thresh && my_mask != 0u) {
        for (var edge = my_start + sub_lane; edge < my_end; edge = edge + SUB_WARP_SIZE) {
            let v = col_ind[edge];
            let v_vis_base = v * MAX_WORDS;

            var newly_discovered = false;
            var words_to_process = my_mask;

            while (words_to_process != 0u) {
                let w = countTrailingZeros(words_to_process);
                words_to_process = words_to_process & (words_to_process - 1u);

                let f = s_f[sub_id * MAX_WORDS + w];
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

                var bits = newly;
                while (bits != 0u) {
                    let bit_pos = countTrailingZeros(bits);
                    let local_s = (w & 1u) * 32u + bit_pos;
                    let dist_idx = local_s * nv + v;
                    let bank = w >> 1u;
                    write_distance_bank(bank, dist_idx, cur_level);
                    bits = bits & (bits - 1u);
                }

                let v_out_idx = out_frontier_base + v * MAX_WORDS + w;
                atomicOr(&frontiers[v_out_idx], newly);
            }

            if (newly_discovered) {
                if (atomicLoad(&enqueued[v]) != cur_level) {
                    if (atomicExchange(&enqueued[v], cur_level) != cur_level) {
                        let local_slot = atomicAdd(&wg_queue_count, 1u);
                        if (local_slot < WG_QUEUE_CAP) {
                            wg_queue_items[local_slot] = v;
                        } else {
                            let slot = atomicAdd(&queue_sizes[out_slot], 1u);
                            queues[out_queue_base + slot] = v;
                        }
                    }
                }
            }
        }
    }
    workgroupBarrier();

    // Phase 3: Workgroup Shared Queue Aggregation and Cooperative Global Flush
    let total_wg_items = atomicLoad(&wg_queue_count);
    let buffered_items = min(total_wg_items, WG_QUEUE_CAP);

    if (local_id.x == 0u && buffered_items > 0u) {
        wg_global_offset = atomicAdd(&queue_sizes[out_slot], buffered_items);
    }
    workgroupBarrier();

    if (buffered_items > 0u) {
        let global_base = wg_global_offset;
        for (var i = local_id.x; i < buffered_items; i = i + WG_SIZE) {
            queues[out_queue_base + global_base + i] = wg_queue_items[i];
        }
    }
}
