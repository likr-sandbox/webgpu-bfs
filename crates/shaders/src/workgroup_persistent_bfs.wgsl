// crates/shaders/src/workgroup_persistent_bfs.wgsl
// Milestone M26: 1 Workgroup = 1 Source Autonomous Persistent BFS Traversal Kernel
// Architecture:
// 1. Single compute dispatch for all hops (up to 1,311 hops on luxembourg_osm)
// 2. 100% on-chip SRAM storage within strict 16,384-byte WebGPU limit:
//    - s_visited: 3,584 atomic<u32> = 14,336 bytes (114,688 bits, covering all 114,599 vertices of luxembourg_osm)
//    - s_queue: 508 u32 = 2,032 bytes (single queue with register buffering)
//    - Control words: s_q_count (4B), s_next_count (4B), s_overflow (4B), s_level (4B) = 16 bytes
//    - Total shared memory: 14,336 + 2,032 + 16 = 16,384 bytes (100.0% compliant)
// 3. Bit-inverted direct distance encoding (packed_dist: 16-bit half-words in array<atomic<u32>>):
//    - Fast zero initialization via encoder.clear_buffer
//    - inv_d = 0xFFFFu - min(level, 0xFFFFu)
//    - atomicOr(&packed_dist[s_word_base + (v >> 1u)], inv_d << ((v & 1u) * 16u))
// 4. Reachability flush:
//    - When reachability_only != 0u, bypass packed_dist writes
//    - Flush s_visited to visited_out_buf upon traversal completion (14.3 KB per source)
// 5. VRAM-visited fallback for graphs with |V| > 114,688

struct PersistentBfsParams {
    num_vertices: u32,
    num_sources: u32,
    use_sram_visited: u32,   // 1u if nv <= 114,688 (use s_visited in SRAM), 0u if nv > 114,688 (use vram_visited)
    reachability_only: u32,  // 1u if reachability mode (flush s_visited to visited_out_buf), 0u for distance mode
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<uniform> params: PersistentBfsParams;
@group(0) @binding(1) var<storage, read> sources: array<u32>;
@group(0) @binding(2) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> col_ind: array<u32>;
@group(0) @binding(4) var<storage, read_write> packed_dist: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> vram_visited: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> visited_out_buf: array<u32>;
@group(0) @binding(7) var<storage, read_write> overflow_buf: array<atomic<u32>>;

const WG_SIZE: u32 = 128u;
const MAX_SRAM_WORDS: u32 = 3584u; // 3584 * 32 = 114,688 bits = 14,336 bytes
const QUEUE_CAP: u32 = 508u;       // 508 * 4 = 2,032 bytes

var<workgroup> s_visited: array<atomic<u32>, MAX_SRAM_WORDS>; // 14,336 B
var<workgroup> s_queue: array<u32, QUEUE_CAP>;               // 2,032 B
var<workgroup> s_q_count: atomic<u32>;                       // 4 B
var<workgroup> s_next_count: atomic<u32>;                    // 4 B
var<workgroup> s_overflow: atomic<u32>;                      // 4 B
var<workgroup> s_level: u32;                                 // 4 B

fn test_and_set_visited(v: u32, s_vram_base: u32) -> bool {
    let word = v >> 5u;
    let bit = 1u << (v & 31u);
    if (params.use_sram_visited != 0u) {
        let prev = atomicOr(&s_visited[word], bit);
        return (prev & bit) == 0u;
    } else {
        let prev = atomicOr(&vram_visited[s_vram_base + word], bit);
        return (prev & bit) == 0u;
    }
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let s_idx = group_id.x;
    if (s_idx >= params.num_sources) {
        return;
    }

    let nv = params.num_vertices;
    let wpv = (nv + 31u) >> 5u;
    let s_vram_base = s_idx * wpv;
    let words_per_source = (nv + 1u) >> 1u;
    let s_word_base = s_idx * words_per_source;

    // Phase 1: Initialize workgroup shared memory
    if (params.use_sram_visited != 0u) {
        for (var i = tid; i < MAX_SRAM_WORDS; i = i + WG_SIZE) {
            atomicStore(&s_visited[i], 0u);
        }
    }
    if (tid == 0u) {
        atomicStore(&s_q_count, 0u);
        atomicStore(&s_next_count, 0u);
        atomicStore(&s_overflow, 0u);
        s_level = 0u;
    }
    workgroupBarrier();

    // Phase 2: Setup root vertex for this workgroup's source
    let root = sources[s_idx];
    if (root < nv) {
        if (tid == 0u) {
            let newly = test_and_set_visited(root, s_vram_base);
            if (newly) {
                // Record distance 0 for root
                if (params.reachability_only == 0u) {
                    let inv_d = 0xFFFFu; // 0xFFFF - 0
                    let word_idx = s_word_base + (root >> 1u);
                    let shift = (root & 1u) * 16u;
                    atomicOr(&packed_dist[word_idx], inv_d << shift);
                }
                // Enqueue root
                s_queue[0] = root;
                atomicStore(&s_q_count, 1u);
            }
        }
    }
    workgroupBarrier();

    // Phase 3: Autonomous Hop Loop
    for (var level = 1u; level <= nv; level = level + 1u) {
        let curr_count = atomicLoad(&s_q_count);
        if (curr_count == 0u) {
            break; // Traversal complete!
        }

        // Pop from s_queue into local thread registers (up to 4 items per thread: 128 * 4 = 512 >= 508)
        var reg_u: array<u32, 4>;
        var num_items: u32 = 0u;
        for (var i = tid; i < curr_count; i = i + WG_SIZE) {
            reg_u[num_items] = s_queue[i];
            num_items = num_items + 1u;
        }

        // Barrier: all items safely loaded into private thread registers before s_queue is overwritten
        workgroupBarrier();

        // Expand neighbors from registers and push into s_queue
        let inv_d = 0xFFFFu - min(level, 0xFFFFu);

        for (var item = 0u; item < num_items; item = item + 1u) {
            let u = reg_u[item];
            let start_edge = row_ptr[u];
            let end_edge = row_ptr[u + 1u];

            for (var e = start_edge; e < end_edge; e = e + 1u) {
                let v = col_ind[e];

                if (test_and_set_visited(v, s_vram_base)) {
                    // Record distance directly in packed_dist
                    if (params.reachability_only == 0u) {
                        let word_idx = s_word_base + (v >> 1u);
                        let shift = (v & 1u) * 16u;
                        atomicOr(&packed_dist[word_idx], inv_d << shift);
                    }

                    // Enqueue newly discovered vertex
                    let slot = atomicAdd(&s_next_count, 1u);
                    if (slot < QUEUE_CAP) {
                        s_queue[slot] = v;
                    } else {
                        atomicStore(&s_overflow, 1u);
                    }
                }
            }
        }

        // Barrier: all neighbor pushes complete
        workgroupBarrier();

        // Thread 0 updates frontier count and resets next counter for the subsequent hop
        if (tid == 0u) {
            let n_count = min(atomicLoad(&s_next_count), QUEUE_CAP);
            atomicStore(&s_q_count, n_count);
            atomicStore(&s_next_count, 0u);
        }
        workgroupBarrier();
    }

    // Phase 4: Reachability Flush
    if (params.reachability_only != 0u) {
        let s_out_base = s_idx * wpv;
        if (params.use_sram_visited != 0u) {
            for (var i = tid; i < wpv; i = i + WG_SIZE) {
                visited_out_buf[s_out_base + i] = atomicLoad(&s_visited[i]);
            }
        } else {
            for (var i = tid; i < wpv; i = i + WG_SIZE) {
                visited_out_buf[s_out_base + i] = atomicLoad(&vram_visited[s_vram_base + i]);
            }
        }
    }

    // Phase 5: Overflow Status Reporting
    // If frontier exceeded QUEUE_CAP at any level during traversal,
    // record overflow for this source so host can transparently fall back.
    if (tid == 0u) {
        if (atomicLoad(&s_overflow) != 0u) {
            atomicStore(&overflow_buf[s_idx], 1u);
        }
    }
}
