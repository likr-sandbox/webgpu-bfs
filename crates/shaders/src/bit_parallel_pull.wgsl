// Bit-Parallel Pull Traversal (32 Sources, Early-Exit, Zero Atomics)
struct Params {
    num_vertices: u32,
    all_sources_mask: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> csc_col_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> csc_row_ind: array<u32>;
@group(0) @binding(3) var<storage, read> curr_frontier: array<u32>;
@group(0) @binding(4) var<storage, read> visited: array<u32>;
@group(0) @binding(5) var<storage, read_write> next_candidate: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let v = global_id.x;
    if (v >= params.num_vertices) {
        return;
    }

    let vis = visited[v];
    let needed = (~vis) & params.all_sources_mask;
    if (needed == 0u) {
        next_candidate[v] = 0u;
        return;
    }

    let start_edge = csc_col_ptr[v];
    let end_edge = csc_col_ptr[v + 1u];
    var accumulated_cand: u32 = 0u;

    for (var edge = start_edge; edge < end_edge; edge = edge + 1u) {
        let parent = csc_row_ind[edge];
        let parent_frontier = curr_frontier[parent];

        accumulated_cand = accumulated_cand | parent_frontier;

        if ((accumulated_cand & needed) == needed) {
            break;
        }
    }

    next_candidate[v] = accumulated_cand;
}
