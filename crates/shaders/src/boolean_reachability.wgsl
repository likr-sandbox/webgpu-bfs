// Boolean Reachability Filtering and Write-Once Distance Update
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
    var count: u32 = 0u;

    if (v < params.num_vertices) {
        let cand = next_candidate[v];
        let old_vis = visited[v];
        let newly_reached = cand & (~old_vis);

        if (newly_reached != 0u) {
            visited[v] = old_vis | newly_reached;
            next_frontier[v] = newly_reached;

            var bits = newly_reached;
            while (bits != 0u) {
                let s_local = countTrailingZeros(bits);
                let s_global = params.chunk_start + s_local;
                distance[s_global * params.num_vertices + v] = params.current_level + 1u;
                bits = bits & (bits - 1u);
            }

            count = countOneBits(newly_reached);
        } else {
            next_frontier[v] = 0u;
        }

        next_candidate[v] = 0u;
    }

    local_counts[local_id.x] = count;
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
