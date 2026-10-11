// Naive Single-Source BFS Push Traversal Kernel
// Group 0 Bindings:
// 0: row_offsets   (storage, read)
// 1: col_indices   (storage, read)
// 2: curr_frontier (storage, read)
// 3: next_frontier (storage, read_write, atomic)
// 4: distances     (storage, read_write, atomic)
// 5: params        (uniform)
// 6: counter       (storage, read_write, atomic)

struct Params {
    num_vertices: u32,
    current_level: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> curr_frontier: array<u32>;
@group(0) @binding(3) var<storage, read_write> next_frontier: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> distances: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> counter: array<atomic<u32>>;

const INF: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let v = global_id.x;
    if (v >= params.num_vertices) {
        return;
    }

    if (curr_frontier[v] == 1u) {
        let start = row_offsets[v];
        let end = row_offsets[v + 1u];

        for (var i = start; i < end; i = i + 1u) {
            let neighbor = col_indices[i];

            if (atomicLoad(&distances[neighbor]) == INF) {
                let prev = atomicExchange(&next_frontier[neighbor], 1u);
                if (prev == 0u) {
                    atomicStore(&distances[neighbor], params.current_level + 1u);
                    atomicAdd(&counter[0], 1u);
                }
            }
        }
    }
}
