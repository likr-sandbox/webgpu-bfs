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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBfsMethod {
    NaiveSequential,
    BatchedDenseSpMM,
    BitParallel32,
    BitParallel64,
    BooleanReachability,
    PushPullAdaptive,
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
