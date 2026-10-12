// crates/shaders/src/init_sparse_push_sources_256.wgsl
// Multi-Word Source Initialization for 256 sources (8 words)

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

@group(0) @binding(4) var<storage, read_write> dist_bank0: array<u32>;
@group(0) @binding(5) var<storage, read_write> dist_bank1: array<u32>;
@group(0) @binding(6) var<storage, read_write> dist_bank2: array<u32>;
@group(0) @binding(7) var<storage, read_write> dist_bank3: array<u32>;

const MAX_WORDS: u32 = 8u;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_threads = num_workgroups.x * 256u;
    for (var s = group_id.x * 256u + local_id.x; s < params.num_sources; s = s + total_threads) {
        let src = sources[s];
        let nv = params.num_vertices;
        if (src >= nv) {
            continue;
        }

        let word_idx = s / 32u;
        let bit_idx = s % 32u;
        let mask = 1u << bit_idx;

        let v_base = src * MAX_WORDS;
        atomicOr(&frontiers[v_base + word_idx], mask);
        atomicOr(&visited[v_base + word_idx], mask);

        let bank = word_idx >> 1u;
        let local_s = (word_idx & 1u) * 32u + bit_idx;
        let dist_idx = local_s * nv + src;

        switch (bank) {
            case 0u: { dist_bank0[dist_idx] = 0u; }
            case 1u: { dist_bank1[dist_idx] = 0u; }
            case 2u: { dist_bank2[dist_idx] = 0u; }
            case 3u: { dist_bank3[dist_idx] = 0u; }
            default: {}
        }
    }
}
