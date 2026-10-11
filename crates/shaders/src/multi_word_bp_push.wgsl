// crates/shaders/src/multi_word_bp_push.wgsl
// Multi-Word Bit-Parallel Push Traversal (128 - 2048 Sources, 2D Warp-Tiled)
//
// Traverses CSR edges forward with bounded out-degrees (max 255 on web-Stanford).
// Threads along the X dimension (W_k) cooperatively push bit-parallel words for vertex u,
// loading neighbor v = col_indices[edge] once and issuing coalesced atomicOr operations.

override W_k: u32 = 8u;
override W_v: u32 = 16u;
override WG_SIZE: u32 = 128u;

struct PushParams {
    num_vertices: u32,
    words_per_vertex: u32,
    current_level: u32,
    dispatch_x: u32,
};

@group(0) @binding(0) var<uniform> params: PushParams;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_ind: array<u32>;
@group(0) @binding(3) var<storage, read> curr_frontier: array<u32>;
@group(0) @binding(4) var<storage, read_write> next_candidate: array<atomic<u32>>;

var<workgroup> row_active: array<atomic<u32>, W_v>;

@compute @workgroup_size(W_k, W_v, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let wg_id = group_id.x + group_id.y * params.dispatch_x;
    let u = wg_id * W_v + local_id.y;

    // Reset shared activity flag for each vertex row
    if (local_id.x == 0u) {
        atomicStore(&row_active[local_id.y], 0u);
    }
    workgroupBarrier();

    if (u >= params.num_vertices) {
        return;
    }

    let u_base = u * params.words_per_vertex;

    // Load active frontier words for this thread into local registers.
    // With W_k = 8 and maximum words_per_vertex = 64 (2048 sources),
    // each thread along dimension X handles up to 8 words with stride W_k.
    var f_words: array<u32, 8>;
    var thread_active: u32 = 0u;

    var w_load = 0u;
    for (var k = local_id.x; k < params.words_per_vertex && w_load < 8u; k = k + W_k) {
        let f = curr_frontier[u_base + k];
        f_words[w_load] = f;
        if (f != 0u) {
            thread_active = 1u;
        }
        w_load = w_load + 1u;
    }

    if (thread_active != 0u) {
        atomicOr(&row_active[local_id.y], 1u);
    }
    workgroupBarrier();

    // Early exit if vertex u has no active frontier bits across any word
    if (atomicLoad(&row_active[local_id.y]) == 0u) {
        return;
    }

    let start_edge = row_ptr[u];
    let end_edge = row_ptr[u + 1u];

    // Thread loads edge col_indices[edge] once and pushes active words from registers.
    // When W_k threads in the row execute in SIMD lockstep, col_indices[edge]
    // is broadcast by hardware L1 cache to all threads in row.
    for (var edge = start_edge; edge < end_edge; edge = edge + 1u) {
        let v = col_ind[edge];
        let v_base = v * params.words_per_vertex;

        if (thread_active != 0u) {
            var w = 0u;
            for (var k = local_id.x; k < params.words_per_vertex && w < 8u; k = k + W_k) {
                let f = f_words[w];
                if (f != 0u) {
                    atomicOr(&next_candidate[v_base + k], f);
                }
                w = w + 1u;
            }
        }
    }
}
