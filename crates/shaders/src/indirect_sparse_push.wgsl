// crates/shaders/src/indirect_sparse_push.wgsl
// Optimized Fused Multi-Source Bit-Parallel Sparse Queue Push Traversal Kernel

struct PushUniforms {
    num_vertices: u32,
    chunk_start: u32,      // Global source offset (e.g. 0, 64, 128...)
    pad0: u32,
    pad1: u32,
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
const WG_QUEUE_CAP: u32 = 256u;

var<workgroup> s_f0: array<u32, NUM_SUB_WARPS>;
var<workgroup> s_f1: array<u32, NUM_SUB_WARPS>;
var<workgroup> s_start_edge: array<u32, NUM_SUB_WARPS>;
var<workgroup> s_end_edge: array<u32, NUM_SUB_WARPS>;

var<workgroup> wg_queue_items: array<u32, WG_QUEUE_CAP>;
var<workgroup> wg_queue_count: atomic<u32>;
var<workgroup> wg_global_offset: u32;

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
    let in_frontier_base = in_slot * (2u * nv);
    let out_frontier_base = out_slot * (2u * nv);

    let sub_id = local_id.x / SUB_WARP_SIZE;
    let sub_lane = local_id.x % SUB_WARP_SIZE;

    if (local_id.x == 0u) {
        atomicStore(&wg_queue_count, 0u);
    }

    if (sub_lane == 0u) {
        s_f0[sub_id] = 0u;
        s_f1[sub_id] = 0u;
        s_start_edge[sub_id] = 0u;
        s_end_edge[sub_id] = 0u;

        let in_queue_idx = group_id.x * NUM_SUB_WARPS + sub_id;
        if (in_queue_idx < in_size) {
            let u = queues[in_queue_base + in_queue_idx];
            if (u < nv) {
                let u_idx0 = in_frontier_base + u * 2u;
                let f0 = atomicExchange(&frontiers[u_idx0], 0u);
                let f1 = atomicExchange(&frontiers[u_idx0 + 1u], 0u);
                if (f0 != 0u || f1 != 0u) {
                    s_f0[sub_id] = f0;
                    s_f1[sub_id] = f1;
                    s_start_edge[sub_id] = row_ptr[u];
                    s_end_edge[sub_id] = row_ptr[u + 1u];
                }
            }
        }
    }
    workgroupBarrier();

    let f0 = s_f0[sub_id];
    let f1 = s_f1[sub_id];
    let start_edge = s_start_edge[sub_id];
    let end_edge = s_end_edge[sub_id];

    if (f0 != 0u || f1 != 0u) {
        let cur_level = state.level;
        for (var edge = start_edge + sub_lane; edge < end_edge; edge = edge + SUB_WARP_SIZE) {
            let v = col_ind[edge];
            let v_vis_idx0 = v * 2u;
            let v_vis_idx1 = v_vis_idx0 + 1u;

            var needed0: u32 = 0u;
            var needed1: u32 = 0u;

            if (f0 != 0u) {
                let vis0 = atomicLoad(&visited[v_vis_idx0]);
                needed0 = f0 & (~vis0);
            }
            if (f1 != 0u) {
                let vis1 = atomicLoad(&visited[v_vis_idx1]);
                needed1 = f1 & (~vis1);
            }

            if (needed0 == 0u && needed1 == 0u) {
                continue;
            }

            var newly0: u32 = 0u;
            var newly1: u32 = 0u;

            if (needed0 != 0u) {
                let prev0 = atomicOr(&visited[v_vis_idx0], needed0);
                newly0 = needed0 & (~prev0);
            }
            if (needed1 != 0u) {
                let prev1 = atomicOr(&visited[v_vis_idx1], needed1);
                newly1 = needed1 & (~prev1);
            }

            if (newly0 == 0u && newly1 == 0u) {
                continue;
            }

            let v_out_idx0 = out_frontier_base + v * 2u;
            let v_out_idx1 = v_out_idx0 + 1u;

            if (newly0 != 0u) {
                var bits = newly0;
                while (bits != 0u) {
                    let bit_pos = countTrailingZeros(bits);
                    let s = uniforms.chunk_start + bit_pos;
                    distance[s * nv + v] = cur_level;
                    bits = bits & (bits - 1u);
                }
                atomicOr(&frontiers[v_out_idx0], newly0);
            }

            if (newly1 != 0u) {
                var bits = newly1;
                while (bits != 0u) {
                    let bit_pos = countTrailingZeros(bits);
                    let s = uniforms.chunk_start + 32u + bit_pos;
                    distance[s * nv + v] = cur_level;
                    bits = bits & (bits - 1u);
                }
                atomicOr(&frontiers[v_out_idx1], newly1);
            }

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

    workgroupBarrier();

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
