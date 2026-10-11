// Dense Multi-Source SpMM Push Traversal (2D Warp-Broadcast Tiled)
// Workgroup size: (32 sources, 4 vertices) = 128 invocations
// Threads in warp x=0..31 broadcast-load identical CSR edges

struct Params {
    num_vertices: u32,
    num_sources: u32,
    current_level: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> curr_frontier: array<u32>;
@group(0) @binding(3) var<storage, read_write> next_frontier: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> distances: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> counter: array<atomic<u32>>;

const INF: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(32, 4, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = global_id.x;
    let v = global_id.y;

    if (s >= params.num_sources || v >= params.num_vertices) {
        return;
    }

    let frontier_idx = s * params.num_vertices + v;
    if (curr_frontier[frontier_idx] == 1u) {
        let start = row_offsets[v];
        let end = row_offsets[v + 1u];

        for (var i = start; i < end; i = i + 1u) {
            let neighbor = col_indices[i];
            let target_idx = s * params.num_vertices + neighbor;

            if (atomicLoad(&distances[target_idx]) == INF) {
                let prev = atomicExchange(&next_frontier[target_idx], 1u);
                if (prev == 0u) {
                    atomicStore(&distances[target_idx], params.current_level + 1u);
                    atomicAdd(&counter[0], 1u);
                }
            }
        }
    }
}
