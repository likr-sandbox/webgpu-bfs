// Single-Source Sparse Queue BFS Push Traversal Kernel
// Group 0 Bindings:
// 0: row_offsets   (storage, read)
// 1: col_indices   (storage, read)
// 2: queue_in      (storage, read)
// 3: queue_out     (storage, read_write)
// 4: distances     (storage, read_write, atomic)
// 5: queue_sizes   (storage, read_write, atomic: [in_size, out_size, cand_size, reserved])
// 6: params        (uniform)

struct Params {
    current_level: u32,
    num_vertices: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> queue_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> queue_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> distances: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> queue_sizes: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;

const INF: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let in_size = atomicLoad(&queue_sizes[0]);
    if (tid >= in_size) {
        return;
    }

    let u = queue_in[tid];
    if (u >= params.num_vertices) {
        return;
    }

    let start = row_offsets[u];
    let end = row_offsets[u + 1u];
    let next_dist = params.current_level + 1u;

    for (var edge = start; edge < end; edge = edge + 1u) {
        let v = col_indices[edge];

        // 1. Cheap non-modifying check
        if (atomicLoad(&distances[v]) == INF) {
            // 2. Atomic claim: exactly one thread wins exchange
            let prev = atomicMin(&distances[v], next_dist);
            if (prev == INF) {
                // Winning thread enqueues v into next frontier
                let slot = atomicAdd(&queue_sizes[1], 1u);
                queue_out[slot] = v;
            }
        }
    }
}
