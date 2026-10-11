// crates/shaders/src/reset_buffer_u32_max.wgsl
// Fast GPU buffer reset to 0xFFFFFFFF (u32::MAX)

struct ResetParams {
    total_elements: u32,
    dispatch_x: u32,
};

@group(0) @binding(0) var<uniform> params: ResetParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let wg_id = group_id.x + group_id.y * params.dispatch_x;
    let idx = wg_id * 256u + local_id.x;
    if (idx < params.total_elements) {
        data[idx] = 0xFFFFFFFFu;
    }
}
