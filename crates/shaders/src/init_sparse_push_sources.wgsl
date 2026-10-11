// crates/shaders/src/init_sparse_push_sources.wgsl
// Fast GPU source vertex initialization for Sparse Push BFS

struct InitSourcesParams {
    num_vertices: u32,
    num_sources: u32,
    chunk_start: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: InitSourcesParams;
@group(0) @binding(1) var<storage, read> sources: array<u32>;
@group(0) @binding(2) var<storage, read_write> frontiers: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> visited: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> distance: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let s = local_id.x;
    if (s >= params.num_sources) {
        return;
    }
    let src = sources[s];
    let nv = params.num_vertices;
    if (src >= nv) {
        return;
    }

    // Source vertex distance is 0
    let dist_idx = s * nv + src;
    distance[dist_idx] = 0u;

    // Bit parallel masks in frontiers (slot 0) and visited
    if (s < 32u) {
        let mask = 1u << s;
        atomicOr(&frontiers[src * 2u], mask);
        atomicOr(&visited[src * 2u], mask);
    } else {
        let mask = 1u << (s - 32u);
        atomicOr(&frontiers[src * 2u + 1u], mask);
        atomicOr(&visited[src * 2u + 1u], mask);
    }
}
