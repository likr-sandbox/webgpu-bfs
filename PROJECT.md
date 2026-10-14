# Project: WebGPU Multi-Source BFS (MS-BFS)

## Architecture
A high-performance GPU-parallel Multi-Source Breadth-First Search (MS-BFS) library and benchmark environment in Rust using WebGPU (`wgpu`) and WGSL compute shaders, inspired by GraphBLAST techniques (algebraic SpMV/SpMM, Push/Pull direction optimization, frontier management, bit-parallel multi-source batching, tiled SpMM, boolean reachability updates).

### Crate Structure (Cargo Workspace)
```
/home/likr/webgpu-bfs/
├── Cargo.toml               # Workspace root
├── crates/
│   ├── core/                # Graph CSR/CSC, Matrix Market parser, procedural generators
│   ├── cpu_bfs/             # Multi-threaded CPU reference BFS (single-source, Rayon, bit-parallel)
│   ├── shaders/             # WGSL compute shaders embedded via include_str!
│   ├── gpu_bfs/             # wgpu compute pipelines, adapter fallback, buffers, MS-BFS dispatchers
│   ├── cli/                 # Native CLI runner, dataset downloader, benchmark harness
│   └── wasm/                # wasm-bindgen export, browser WebGPU demo bundle
├── web/                     # Web frontend (index.html, app.js, style.css)
├── tests/                   # Dual-track E2E test suites (Tiers 1-4)
├── scripts/                 # Dataset download and benchmark automation scripts
└── data/                    # Local storage for datasets (e.g., luxembourg_osm)
```

### Memory & Distance Layout
- **Distance Matrix**: Source-major flat layout `dist[s * |V| + v]` with `u32` elements.
- **Unreachable Sentinel**: `u32::MAX` (`0xFFFF_FFFF`). Source distance: `dist[s * |V| + s] = 0`.
- **Bit-Parallel Frontiers**: Packed bitmasks (`u32` for $S=32$, `vec2<u32>` for $S=64$) enabling full L2 cache residence.
- **3-Tier Adapter Selection**: `HighPerformance` → `force_fallback_adapter: true` → `enumerate_adapters()`, guaranteeing seamless execution on both hardware GPUs and Mesa Lavapipe Vulkan software driver.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Graph Core CSR/CSC | High-performance CSR and CSC data structures, degree queries, edge transposition | M1 | ORIGINAL_REQUEST §R1 |
| F2 | Matrix Market Parser | Streaming Matrix Market (.mtx) parser with comments handling and symmetric edge expansion | M1 | ORIGINAL_REQUEST §R4 |
| F3 | Synthetic Graph Generators | Procedural generators: Grid2D (exact shortest path oracle), Erdos-Renyi, Random Tree, R-MAT power-law | M1 | ORIGINAL_REQUEST §R4 |
| F4 | CPU Reference BFS Engines | Exact CPU BFS ground truth: single-source FIFO, multi-threaded Rayon batched, and 64-bit bit-parallel | M1 | ORIGINAL_REQUEST §R4 |
| F5 | WGSL Compute Shaders | WGSL kernels: Naive SpMV, Batched SpMM, Bit-Parallel Push, Bit-Parallel Pull, Boolean Reachability | M2 | ORIGINAL_REQUEST §R1, §R2 |
| F6 | WebGPU Runtime & Pipeline | Headless wgpu 24 runtime, 3-tier adapter fallback (Lavapipe Vulkan support), storage buffers, staging readback | M2 | ORIGINAL_REQUEST §R1, §R3 |
| F7 | Baseline GPU MS-BFS | Naive sequential GPU BFS and direct batched SpMM BFS implementations | M2 | ORIGINAL_REQUEST §R1, §R2 |
| F8 | Bit-Parallel MS-BFS (32/64) | 32-source (`u32`) and 64-source (`vec2<u32>`) bit-parallel frontier expansion with hardware bit-scan | M3 | ORIGINAL_REQUEST §R2 |
| F9 | Boolean Reachability Updates | 3-stage pipeline (Traversal → Filter → Distance Update) with race-free write-once distance recording | M3 | ORIGINAL_REQUEST §R2 |
| F10 | Push/Pull Direction Optimization | Adaptive switching based on frontier edge density ($m_f > \|E\| / \alpha$ with $\alpha=14, \beta=24$) | M3 | ORIGINAL_REQUEST §R1, §R2 |
| F11 | Dual Track E2E Test Suite | Comprehensive multi-tier test suite (Tiers 1-4: features, boundary/corner, pairwise, real-world) | M4 | ORIGINAL_REQUEST §Acceptance |
| F12 | Native CLI Runner | CLI binary (`clap` v4) with dataset loader, synthetic generation, multi-source batching, verification, and benchmark | M5 | ORIGINAL_REQUEST §R3 |
| F13 | WASM / Browser WebGPU Demo | `wasm-bindgen` export + modern HTML/JS/CSS WebGPU demo page with UI controls and benchmark execution | M5 | ORIGINAL_REQUEST §R3 |
| F14 | SuiteSparse Luxembourg Pipeline | Automated download, extraction, validation, and execution on `luxembourg_osm` dataset | M6 | ORIGINAL_REQUEST §R4 |
| F15 | Benchmark Suite & Reports | Graph500 Harmonic MTEPS measurement, 6-way algorithmic comparison (Markdown, CSV, JSON), documentation | M6 | ORIGINAL_REQUEST §R4, §Acceptance |
| F16 | Multi-Word Bit-Parallel Pull | 512-source 8-bank pull traversal with 0 barriers, 0 LDS bytes, dirty-check ping-pong, and single-counter termination | M19, M23 | ORIGINAL_REQUEST §R2 |
| F17 | Indirect Sparse Queue Push (ISQP) | Fused GPU-autonomous sparse push with workgroup hub expansion, LDS queue, u16 quantization, and adaptive macro-batching | M12, M16-M18, M20, M23 | ORIGINAL_REQUEST §R1, §R3 |
| F18 | Topology-Adaptive Auto Router | Degree-skew and batch-aware deterministic algorithm selector for optimal performance across all graph topologies | M17, M21, M22 | ORIGINAL_REQUEST §R1 |
| F19 | Fractional Batch Single-Pass Routing | Direct single-pass dispatch for fractional batches ($256 < S \le 512 \to \text{pool\_512}$, $64 < S \le 256 \to \text{pool\_256}$) with dynamic word masking | M22 | ORIGINAL_REQUEST §R2 |
| F20 | Exact Distance Buffer Reset & Word Guarding | Bounding memory reset to actual $S \times \|V\|$ elements, reducing $S=1$ initialization by $64\times$ | M22 | ORIGINAL_REQUEST §R1 |
| F21 | High-Diameter Scaling & Pull Barrier Elimination | Elimination of all 8 tree barriers/WG, dirty check skipping ~9.6 GB DRAM writes per pass, and $160 \to 512$ hop adaptive scaling | M23 | ORIGINAL_REQUEST §R3 |
| F22 | Symmetric Graph Buffer Deduplication | Zero-duplicate memory aliasing for CSC/CSR graph topology via `Arc<wgpu::Buffer>` and `from_symmetric_graph` | M23 | ORIGINAL_REQUEST §R1 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Graph Core & CPU Reference | Cargo workspace, `crates/core` (CSR/CSC, .mtx parser, generators), `crates/cpu_bfs` (CPU reference engines, unit tests) | none | DONE |
| M2 | WebGPU Runtime & Baseline GPU BFS | `crates/shaders` (WGSL templates), `crates/gpu_bfs` (wgpu context, buffer management, baseline GPU MS-BFS, CPU equivalence tests) | M1 | DONE |
| M3 | Advanced Optimizations | Bit-parallel (32/64-bit), Boolean reachability update engine, Push/Pull direction-optimizing controller | M2 | DONE |
| M4 | Dual-Track E2E Test Suite | Test runner, Tiers 1-4 automated test suite (50+ cases), topological edge cases, `TEST_READY.md` publication | M3 | DONE |
| M5 | Native CLI & WASM Web Demo | `crates/cli` binary with full flags, `crates/wasm` package, `web/` HTML/JS WebGPU demo frontend | M3 | DONE |
| M6 | SuiteSparse Luxembourg & Benchmarks | Automated `luxembourg_osm` loader, comprehensive benchmark harness, MTEPS reporting, README & performance documentation | M4, M5 | DONE |
| M7–M10 | Luxembourg OSM Reversal & Massive Multi-Source | 2D warp-tile pull, hardware target analysis (19k sources/s), sparse queue compaction, 256–2048 multi-source batching | M6 | DONE |
| M11–M14 | WSL2 Hardware Direct & web-Stanford CPU Reversal | WSL2 Direct3D 12 enablement, 64-bit ISQP, LDS active queue, GPU in-place u16 quantization, persistent buffer pooling | M10 | DONE |
| M15–M18 | Undirected Scale-Free Reversal ($S \le 2048$) | Degree-aware collaborative hub expansion, 512/256-bit multi-word push, multi-bank distance buffers (128 MB limit bypass), topology router `Auto` | M14 | DONE |
| M19 | High-Diameter Multi-Bank Pull Scaling | 8-bank 512-source pull shader, single-encoder DMA & u16 quantization, 4-pass $S=2048$ execution (2,686 ms) | M18 | DONE |
| M20 | 3rd-Gen Collaborative Hub Architecture | Phase 0 subwarp cooperative load, 2,048-entry LDS queue, 4-quadrant atomics, stride 256 ultra-hub unrolling (5,091 MTEPS) | M18 | DONE |
| M21 | Topology Router Verification & Complete Parity | Deterministic `Auto` router testing across degree/batch boundaries, workspace-wide bitwise parity verification | M19, M20 | DONE |
| M22 | Sparse Push Fractional Batch & Reset Optimization | Single-pass fractional routing ($S=128 < S=256$, $S=384 < S=512$), exact $S \times \|V\|$ reset, M20 horizontal expansion to 256/64 shaders, $S \ge 1$ `Auto` routing | M21 | DONE |
| M23 | High-Diameter Scaling & Pull Barrier Elimination | 0-barrier & 0-LDS pull kernel, dirty-check ping-pong zeroing, adaptive macro-batching ($160 \to 512$ hops), single final-counter check, `Arc<Buffer>` deduplication | M22 | DONE |

---

## Interface Contracts

### `crates/core` ↔ `crates/cpu_bfs` & `crates/gpu_bfs`
```rust
pub struct CsrGraph {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub row_offsets: Vec<u32>,
    pub col_indices: Vec<u32>,
}

pub struct CscGraph {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub col_offsets: Vec<u32>,
    pub row_indices: Vec<u32>,
}

impl CsrGraph {
    pub fn to_csc(&self) -> CscGraph;
    pub fn from_mtx_reader<R: std::io::BufRead>(reader: R) -> anyhow::Result<Self>;
}
```

### `crates/cpu_bfs` ↔ `crates/gpu_bfs` (Ground Truth Equivalence)
```rust
pub trait BfsEngine {
    /// Executes multi-source BFS for a slice of source vertex indices.
    /// Returns a flat vector of distances with layout: dist[s * num_vertices + v].
    /// Unreachable vertices must be u32::MAX (0xFFFF_FFFF).
    /// Source vertices must be 0: dist[s * num_vertices + s] == 0.
    fn run_ms_bfs(&self, graph: &CsrGraph, sources: &[u32]) -> anyhow::Result<Vec<u32>>;
}
```

### `crates/gpu_bfs` Method Selection Enum
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuBfsMethod {
    #[default]
    NaiveSequential,
    BatchedDenseSpMM,
    BitParallel32,
    BitParallel64,
    BitParallel64Pull,
    BooleanReachability,
    PushPullAdaptive,
    SparseQueue,
    SparseBitParallel64,
    MultiWordBitParallel,
    IndirectMacroBatch,
    IndirectSparseQueuePush,
    MultiWordBitParallelPush,
    Auto,
}
```

---

## Code Layout
- `Cargo.toml`: Root workspace file defining all members.
- `crates/core/src/`:
  - `lib.rs`: Module exports.
  - `graph.rs`: `CsrGraph`, `CscGraph`, degree calculations.
  - `mtx.rs`: Matrix Market format streaming reader & symmetric expansion.
  - `generators.rs`: Grid2D, Erdos-Renyi, Random Tree, R-MAT.
- `crates/cpu_bfs/src/`:
  - `lib.rs`: `CpuBfsEngine`, single-source queue BFS, Rayon parallel MS-BFS, 64-bit bit-parallel MS-BFS.
- `crates/shaders/src/`:
  - `lib.rs`: Static string constants exposing `.wgsl` files.
  - `naive_spmv.wgsl`
  - `batched_spmm.wgsl`
  - `bit_parallel_push.wgsl`
  - `bit_parallel_pull.wgsl`
  - `boolean_reachability.wgsl`
  - `bit_parallel_64.wgsl`
- `crates/gpu_bfs/src/`:
  - `context.rs`: WebGPU Instance, Adapter (3-tier fallback), Device, Queue.
  - `buffer.rs`: Storage buffers, staging buffers, mapped readback.
  - `pipeline.rs`: Compute pipeline cache.
  - `engine.rs`: Implementation of `BfsEngine` across all methods.
- `crates/cli/src/`:
  - `main.rs`: CLI entry point (`clap`), subcommands/flags, benchmark execution, CSV/JSON/MD formatting.
  - `dataset.rs`: SuiteSparse `luxembourg_osm` download & caching.
- `crates/wasm/src/`:
  - `lib.rs`: `wasm-bindgen` export exposing `run_wasm_bfs` returning JS objects.
- `web/`:
  - `index.html`: Responsive HTML5 interface.
  - `app.js`: WebGPU device initialization, graph generation/upload, WASM execution, canvas rendering.
  - `style.css`: Modern UI styling.
- `tests/`:
  - `e2e_tests.rs`: Dual-track automated tests covering Tiers 1-4.
