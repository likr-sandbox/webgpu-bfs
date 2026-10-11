// 64-Source Bit-Parallel Processing
struct Params {
    num_vertices: u32,
    current_level: u32,
    chunk_start: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> next_candidate: array<u32>;
@group(0) @binding(2) var<storage, read_write> visited: array<u32>;
@group(0) @binding(3) var<storage, read_write> next_frontier: array<u32>;
@group(0) @binding(4) var<storage, read_write> distance: array<u32>;
@group(0) @binding(5) var<storage, read_write> active_count: atomic<u32>;

var<workgroup> local_counts: array<u32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let v = global_id.x;
    var total_new: u32 = 0u;

    if (v < params.num_vertices) {
        let idx0 = v * 2u;
        let idx1 = v * 2u + 1u;

        let newly_reached_0 = next_candidate[idx0] & (~visited[idx0]);
        if (newly_reached_0 != 0u) {
            visited[idx0] = visited[idx0] | newly_reached_0;
            next_frontier[idx0] = newly_reached_0;

            var bits = newly_reached_0;
            while (bits != 0u) {
                let s = params.chunk_start + countTrailingZeros(bits);
                distance[s * params.num_vertices + v] = params.current_level + 1u;
                bits = bits & (bits - 1u);
            }
        } else {
            next_frontier[idx0] = 0u;
        }
        next_candidate[idx0] = 0u;

        let newly_reached_1 = next_candidate[idx1] & (~visited[idx1]);
        if (newly_reached_1 != 0u) {
            visited[idx1] = visited[idx1] | newly_reached_1;
            next_frontier[idx1] = newly_reached_1;

            var bits = newly_reached_1;
            while (bits != 0u) {
                let s = params.chunk_start + 32u + countTrailingZeros(bits);
                distance[s * params.num_vertices + v] = params.current_level + 1u;
                bits = bits & (bits - 1u);
            }
        } else {
            next_frontier[idx1] = 0u;
        }
        next_candidate[idx1] = 0u;

        total_new = countOneBits(newly_reached_0) + countOneBits(newly_reached_1);
    }

    local_counts[local_id.x] = total_new;
    workgroupBarrier();

    for (var stride: u32 = 64u; stride > 0u; stride = stride >> 1u) {
        if (local_id.x < stride) {
            local_counts[local_id.x] = local_counts[local_id.x] + local_counts[local_id.x + stride];
        }
        workgroupBarrier();
    }

    if (local_id.x == 0u && local_counts[0] > 0u) {
        atomicAdd(&active_count, local_counts[0]);
    }
}
