// crates/shaders/src/pack_distances_u32_to_u8.wgsl
// Milestone M24: GPU In-Place 8-bit (u8) Distance Quantization Pass
// Packs 4 vertex distances (u32) into a single packed u32 word.
// Unreachable sentinel: u32::MAX (0xFFFFFFFF) -> 0xFF.
// Applicable when max traversal level < 255.

struct PackParams {
    total_elements: u32,
    total_packed_words: u32,
    dispatch_x: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var<storage, read> dist_u32: array<u32>;
@group(0) @binding(2) var<storage, read_write> dist_u8_packed: array<u32>;

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

    let in_idx = word_idx * 4u;
    let d0 = dist_u32[in_idx];
    let p0 = min(d0, 0xFFu);

    var p1 = 0xFFu;
    if (in_idx + 1u < params.total_elements) {
        let d1 = dist_u32[in_idx + 1u];
        p1 = min(d1, 0xFFu);
    }

    var p2 = 0xFFu;
    if (in_idx + 2u < params.total_elements) {
        let d2 = dist_u32[in_idx + 2u];
        p2 = min(d2, 0xFFu);
    }

    var p3 = 0xFFu;
    if (in_idx + 3u < params.total_elements) {
        let d3 = dist_u32[in_idx + 3u];
        p3 = min(d3, 0xFFu);
    }

    dist_u8_packed[word_idx] = (p3 << 24u) | (p2 << 16u) | (p1 << 8u) | (p0 & 0xFFu);
}
