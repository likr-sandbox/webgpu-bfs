// crates/shaders/src/pack_distances_u32_to_u16.wgsl
// Milestone M14: Ultra-fast GPU u32 -> u16 Distance Compaction Pass

struct PackParams {
    total_elements: u32,
    total_packed_words: u32,
    dispatch_x: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var<storage, read> dist_u32: array<u32>;
@group(0) @binding(2) var<storage, read_write> dist_u16_packed: array<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let wg_id = group_id.x + group_id.y * params.dispatch_x;
    let word_idx = wg_id * 256u + local_id.x;

    if (word_idx >= params.total_packed_words) {
        return;
    }

    let in_idx = word_idx * 2u;
    let d0 = dist_u32[in_idx];
    let p0 = min(d0, 0xFFFFu);

    var p1 = 0xFFFFu;
    if (in_idx + 1u < params.total_elements) {
        let d1 = dist_u32[in_idx + 1u];
        p1 = min(d1, 0xFFFFu);
    }

    dist_u16_packed[word_idx] = (p1 << 16u) | (p0 & 0xFFFFu);
}
