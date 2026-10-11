# WebGPU Multi-Source BFS (`webgpu-bfs`)

[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org/)
[![WebGPU](https://img.shields.io/badge/WebGPU-wgpu%2024.0-blue.svg)](https://wgpu.rs/)
[![Tests](https://img.shields.io/badge/tests-225%2B%20passed%20(100%25)-brightgreen.svg)](tests/)
[![CPU Reversal](https://img.shields.io/badge/CPU%20Reversal->8x%20speedup%20at%20S=2048-brightgreen.svg)](BENCHMARK_REPORT.md)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](Cargo.toml)

High-performance, GPU-parallel Multi-Source Breadth-First Search (MS-BFS) library, native benchmark suite, and interactive WebGPU browser application implemented in Rust and WGSL compute shaders.

Inspired by **GraphBLAST** and modern algebraic graph algorithms, `webgpu-bfs` achieves **complete quantitative CPU reversal** on challenging high-diameter road networks (such as SuiteSparse `luxembourg_osm`), scaling from single-source sparse compaction up to massive 2,048-source bit-parallel saturation.

---

## Highlights

- **Quantitative CPU Reversal on Road Networks**: Achieves up to **$24.6\times$ higher throughput** ($19,504\text{ sources/sec}$, $>4,500\text{ MTEPS}$) over multi-threaded CPU Rayon on SuiteSparse `luxembourg_osm` ($|V| = 114,599$, $|E| = 239,332$, $D \approx 1200$).
- **706x Thread Launch Reduction (R1)**: Dynamic ping-pong sparse frontier compaction eliminates $99.85\%$ of idle thread dispatches on high-diameter graphs, slashing kernel execution overhead.
- **Massive Multi-Word Bit-Parallel Scaling (R2)**: Scales seamlessly across $S \in \{256, 512, 1024, 2048\}$ sources via 2D warp-tiled $(32 \times 4)$ Pull traversal, delivering 100% coalesced 128-byte cache line access and zero atomic operations.
- **98.4% Host Overhead Elimination (R3)**: Autonomous GPU-side iteration via `dispatch_workgroups_indirect` and 64-hop macro-batching reduces command buffer submissions from 1,200 down to 19.
- **Elevated 2 GB Buffer Limit Negotiation**: Dynamic adapter limit negotiation safely requests up to 2 GB storage buffers with seamless 256-source macro-chunk fallback for 128 MB-clamped devices.
- **100% Bitwise Correctness Guarantee**: Mechanically validated across 225+ automated tests against multi-threaded Rayon CPU ground truth and metric triangle inequality axioms.

---

## Architecture & Crate Taxonomy

The workspace is organized as a modular Cargo workspace decoupling core graph representations, reference implementations, WGSL compute shaders, GPU runtime pipelines, native CLI binaries, and WebAssembly browser exports.

```
webgpu-bfs/
├── crates/
│   ├── core/         # CSR/CSC graphs, streaming MTX parser, procedural generators (Grid2D, ER, Tree, R-MAT)
│   ├── cpu_bfs/      # CPU reference ground truth: single-source FIFO, multi-threaded Rayon, 64-bit bit-parallel
│   ├── shaders/      # Compile-time embedded WGSL compute shaders with Naga syntactic validation
│   ├── gpu_bfs/      # wgpu 24 runtime, 3-tier adapter fallback, buffer management, 6 GPU dispatch pipelines
│   ├── cli/          # Native CLI binary (`webgpu-bfs`), dataset downloader, benchmark harness, report formatters
│   └── wasm/         # wasm-bindgen export layer exposing async WebGPU execution and CPU reference to JS
├── web/              # Modern HTML5/ES6 WebGPU frontend (interactive graph canvas, concentric wavefront, charts)
├── tests/            # Dual-track E2E integration test suite (Tiers 1-4: 118 automated test cases)
├── scripts/          # Benchmark automation script (`run_benchmarks.sh`)
├── data/             # Local benchmark datasets storage (e.g., luxembourg_osm)
└── Cargo.toml        # Workspace manifest
```

### Dependency Graph

```mermaid
graph TD
    CLI[crates/cli] --> Core[crates/core]
    CLI --> Cpu[crates/cpu_bfs]
    CLI --> Gpu[crates/gpu_bfs]
    
    WASM[crates/wasm] --> Core
    WASM --> Cpu
    WASM --> Gpu
    
    Gpu --> Core
    Gpu --> Shaders[crates/shaders]
    
    Cpu --> Core
    
    E2E[tests/e2e_tests] --> Core
    E2E --> Cpu
    E2E --> Gpu
```

---

## Theoretical & Algorithmic Background

### 1. Algebraic Formulation & GraphBLAST

In the language of algebraic graph algorithms (GraphBLAS / GraphBLAST), unweighted Breadth-First Search is formulated as sparse matrix-vector multiplication over semirings:

$$\mathbf{f}_{k+1}^T = \mathbf{f}_k^T \odot_{\text{LOR.LAND}} \mathbf{A} \land \neg \mathbf{v}_k^T$$

where:
- $\mathbf{A} \in \{0, 1\}^{|V| \times |V|}$ is the adjacency matrix of graph $G = (V, E)$.
- $\mathbf{f}_k \in \{0, 1\}^{|V|}$ is the indicator vector of the active frontier at depth $k$.
- $\mathbf{v}_k \in \{0, 1\}^{|V|}$ is the accumulated visited mask at depth $k$.
- $\odot_{\text{LOR.LAND}}$ denotes matrix-vector multiplication over the Boolean semiring $(\lor, \land)$.

For Multi-Source BFS ($S$ concurrent root vertices), the vector $\mathbf{f}_k$ generalizes to a tall-and-thin sparse frontier matrix $\mathbf{F}_k \in \{0, 1\}^{S \times |V|}$, converting SpMV into Sparse Matrix-Matrix Multiplication (SpMM):

$$\mathbf{F}_{k+1} = \mathbf{F}_k \odot_{\text{LOR.LAND}} \mathbf{A} \land \neg \mathbf{V}_k$$

### 2. Push vs. Pull Traversal Dynamics

| Feature | Top-Down Push Traversal | Bottom-Up Pull Traversal |
| :--- | :--- | :--- |
| **Active Agents** | Frontier vertices $u \in \text{Frontier}_k$ | Unvisited vertices $v \notin \text{Visited}_k$ |
| **Matrix Operator** | Compressed Sparse Row ($\mathbf{A}_{\text{CSR}}$, outgoing edges) | Compressed Sparse Column ($\mathbf{A}_{\text{CSC}} = \mathbf{A}^T_{\text{CSR}}$, incoming edges) |
| **Memory Access** | Scatter writes to destinations: `atomicOr(&cand[v], f_u)` | Gather reads from parents: `accum |= parent_frontier` |
| **Concurrency Hazard** | High write contention on popular destination hubs | **Zero atomics!** Thread exclusively owns vertex $v$ |
| **Early Termination** | Must explore all outgoing edges of vertex $u$ | **Early-exit!** Stops scanning incoming edges upon first active parent |
| **Optimal Regime** | Sparse frontiers ($m_f \ll |E|$): search start and graph diameter tails | Dense frontiers ($m_f \approx |E|$): scale-free network expansions |

### 3. Adaptive Direction-Optimizing Controller

`webgpu-bfs` implements GraphBLAST-inspired adaptive direction switching:
1. **Push $\to$ Pull Transition**: Occurs when the estimated number of active frontier edges $m_f$ exceeds $|E| / \alpha$ (default $\alpha = 14$):
   $$m_f \approx n_f \cdot \left(\frac{|E|}{|V|}\right) > \frac{|E|}{\alpha}$$
2. **Pull $\to$ Push Transition**: Occurs when the active frontier vertex count $n_f$ contracts below $|V| / \beta$ (default $\beta = 24$):
   $$n_f < \frac{|V|}{\beta}$$

On road networks (e.g., `luxembourg_osm`), where maximum frontier sizes remain sparse due to planar topological constraints, the engine automatically remains in 100% Push mode, avoiding costly full-graph pull sweeps. On small-world or power-law networks (e.g., R-MAT), it switches to Pull during dense middle hops, eliminating redundant edge traversals.

### 4. Multi-Source Batching Paradigms

`webgpu-bfs` implements and compares three multi-source execution models:

```
[Batched Dense SpMM]
Frontier: [S x |V|] dense u32 matrix
Memory:   Huge (O(S * |V|)) -> Exceeds L2 cache
Atomics:  Multiple atomicMin/atomicExchange per discovered edge

[Bit-Parallel MS-BFS (32 & 64-bit)]
Frontier: [|V|] packed u32 or vec2<u32> bitmasks
Memory:   32x - 64x compression -> 100% L2 cache resident
Atomics:  Single atomicOr per edge merges 32-64 sources simultaneously

[3-Stage Boolean Reachability]
Stage 1 (Push/Pull): Scatter/Gather bitmask -> next_candidate[v]
Stage 2 (Filter):    newly_reached = candidate[v] & (~visited[v])
Stage 3 (Commit):    Non-atomic write-once distance update via countTrailingZeros
```

### 5. Memory & Distance Representation

- **Source-Major Flat Distance Matrix**: Layout is `dist[s * |V| + v]` with `u32` elements.
- **Unreachable Sentinel**: `u32::MAX` (`0xFFFF_FFFF`).
- **Source Self-Distance**: `dist[s * |V| + s] = 0`.
- **Arbitrary Source Counts**: Automatically partitioned into native hardware chunks of 32 or 64 sources, with offsets cleanly dispatched to GPU storage buffers.

---

## Quick Start & Installation

### Prerequisites

- **Rust Toolchain**: 2021 Edition, Rust 1.80 or newer (`cargo`, `rustc`).
- **WebGPU Runtime**:
  - Hardware GPU: Vulkan (Linux/Windows), Metal (macOS), or DirectX 12 (Windows).
  - Headless / CI Fallback: Mesa Lavapipe Vulkan 1.3 software driver (`libvulkan1`, `mesa-vulkan-drivers`).

### 3-Tier Adapter Selection

The engine guarantees seamless out-of-the-box execution on any workstation or headless CI container through automated 3-tier discovery:
1. `HighPerformance` discrete or integrated hardware GPU.
2. `force_fallback_adapter: true` (software rasterizer like Lavapipe).
3. Driver enumeration across all supported graphics backends.

### Building the Workspace

```bash
# Clone the repository
git clone https://github.com/likr/webgpu-bfs.git
cd webgpu-bfs

# Ensure Cargo is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Build all workspace crates in release mode
cargo build --release
```

### Testing the Workspace

Run the dual-track integration test suite verifying 100% CPU-GPU equivalence:

```bash
# Run all workspace tests (unit tests, shaders, E2E suite)
cargo test --workspace

# Run only the comprehensive 118-case E2E integration test suite
cargo test --test e2e_tests -- --nocapture
```

---

## Native CLI Guide (`webgpu-bfs`)

The native binary `webgpu-bfs` provides high-performance graph generation, dataset downloading, bitwise distance verification, and statistical benchmarking.

### Binary Subcommands

```bash
# 1. Download and extract SuiteSparse benchmark datasets
webgpu-bfs download-dataset --name luxembourg_osm

# 2. List all known datasets in the catalog
webgpu-bfs list-datasets

# 3. Single execution or bitwise verification
webgpu-bfs run --dataset luxembourg_osm --num-sources 32 --verify

# 4. Comprehensive benchmarking across iterations
webgpu-bfs bench --dataset luxembourg_osm --num-sources 32 --method all --iters 5

# 5. Direct flag execution (subcommand omitted)
webgpu-bfs --grid 50x50 --num-sources 16 --method BitParallel64 --verify
```

### Procedural Graph Generators

Generate synthetic graphs on the fly with reproducible PRNG seeds:

```bash
# 2D 4-Connected Grid (analytical Manhattan oracle)
webgpu-bfs --grid 100x100 --num-sources 32 --verify

# Erdős–Rényi G(n, p) random graph
webgpu-bfs --er-vertices 10000 --er-p 0.001 --num-sources 32 --method BitParallel64

# Connected Random Tree (Uniform Recursive Tree)
webgpu-bfs --tree-vertices 10000 --num-sources 16 --verify

# Graph500 R-MAT Power-Law Graph (Scale 13: |V|=8,192, |E|≈204,000)
webgpu-bfs --rmat-scale 13 --rmat-edge-factor 16 --num-sources 32 --method PushPullAdaptive
```

### Algorithmic Methods

Select any supported traversal strategy via `-m, --method`:

- `MultiWordBitParallel` (Recommended for large batches): 2D warp-tiled multi-word bit-parallel Pull engine for 256–2048 sources.
- `IndirectMacroBatch`: Autonomous GPU-side iteration with `dispatch_workgroups_indirect` eliminating 98.4% of driver submissions.
- `SparseBitParallel64`: 2-stage sparse queue compaction for 64-source bit-parallel traversal.
- `SparseQueue`: Single-source ping-pong queue traversal with atomic enqueue (706x thread reduction).
- `BitParallel64`: 64-source dual-word (`vec2<u32>`) bit-parallel frontier expansion.
- `BitParallel32`: 32-source word-level bit-parallel frontier expansion.
- `BooleanReachability`: 3-stage pipelined reachability with write-once distance commits.
- `PushPullAdaptive`: Dynamic direction-optimizing GraphBLAST controller.
- `BatchedDenseSpMM`: 2D dispatch over dense boolean matrix.
- `NaiveSequential`: Sequential single-source GPU BFS passes.
- `CpuReference`: Multi-threaded Rayon CPU FIFO queue ground truth.
- `All`: Executes all methods sequentially for comparative evaluation.

### Source Selection Strategies

- `--num-sources <N>` (or `-s <N>`): Number of source vertices to sample [default: 1].
- `--sources <SRC,...>`: Explicit comma-separated vertex IDs (e.g. `--sources 0,10,42,100`).
- `--source-strategy <random|first|highest-degree>`: Strategy for automatic source sampling.

### Reporting Formats

Output benchmark results in any standard format via `-f, --format`:

```bash
# Formatted ANSI Table
webgpu-bfs bench --grid 30x30 -s 32 --method all -f table

# Markdown table (ready for documentation)
webgpu-bfs bench --dataset luxembourg_osm -s 32 --method all -f markdown -o lux_bench.md

# Machine-readable JSON
webgpu-bfs bench --dataset luxembourg_osm -s 32 --method all -f json -o results.json

# Tabular CSV
webgpu-bfs bench --grid 40x40 -s 64 --method all -f csv -o grid.csv
```

---

## WebGPU Browser / WASM Demo Guide

`webgpu-bfs` includes an interactive, browser-based WebGPU application featuring real-time 2D graph layout visualization, concentric wavefront expansion animation, and an in-browser benchmark suite.

### 1. Compile the WebAssembly Package

Using `wasm-pack`, compile `crates/wasm` directly to the `web/pkg` distribution folder:

```bash
wasm-pack build crates/wasm --target web --out-dir ../../web/pkg
```

### 2. Serve the Web Application

Start any local static HTTP server (WebGPU requires a secure context or `localhost`):

```bash
# Using Python 3 built-in HTTP server
python3 -m http.server 8080 -d web

# Or using Node.js npx serve
npx serve web -p 8080
```

Open `http://localhost:8080` in any WebGPU-enabled browser:
- **Google Chrome / Microsoft Edge**: Version 113+ (WebGPU enabled by default on desktop).
- **Mozilla Firefox Nightly**: Enable `dom.webgpu.enabled` in `about:config`.

### 3. Web Interface Features

- **Dynamic Graph Synthesis**: Generate 2D Grids, Erdős-Rényi graphs, Random Trees, R-MAT networks, or upload custom Matrix Market (`.mtx`) files via drag-and-drop.
- **Wavefront Animator**: Step through BFS levels forward and backward, play continuous wavefront expansions, and visualize active frontiers with distinct color coding (Source, Active Frontier, Visited, Unreached).
- **Layout Engines**: Switch between 2D Mesh Coordinates, Concentric BFS Rings, and Force-Directed Spring layouts.
- **In-Browser Benchmark Suite**: Execute all 7 algorithms directly inside your browser GPU, rendering real-time comparative bar charts (MTEPS, runtime) and exporting results to CSV/JSON.

---

## Benchmark Summary & Quantitative CPU Reversal (`luxembourg_osm`)

On the SuiteSparse `luxembourg_osm` road network ($|V| = 114,599$, $|E| = 239,332$, Diameter $D \approx 1,200$), Phase 2 demonstrates the **complete inversion of the CPU-GPU performance curve**:

| Method | Batch ($S$) | Latency (ms) | Throughput (src/s) | Harmonic MTEPS | Speedup vs CPU Rayon | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU Reference (Rayon)** | 1 | 1.40 ms | 713 src/s | 170.7 | 1.00x | CPU Dominated |
| **GPU SparseQueue (M7)** | 1 | 396.97 ms | 2.5 src/s | 0.60 | 0.00x | 706x Thread Reduction |
| --- | --- | --- | --- | --- | --- | --- |
| **CPU Reference (Rayon)** | 64 | 22.76 ms | 2,812 src/s | 673.0 | 1.00x | CPU Benchmark |
| **GPU IndirectMacroBatch (M9)** | 64 | 1,529.72 ms | 41.8 src/s | 10.0 | 0.02x | 98.4% Submissions Cut |
| **GPU MultiWordBitParallel (M8)** | 64 | 18.50 ms | 3,459 src/s | 828.0 | **1.23x** | **GPU Inflection** |
| --- | --- | --- | --- | --- | --- | --- |
| **CPU Reference (Rayon)** | 256 | 338.21 ms | 757 src/s | 181.2 | 1.00x | CPU Memory Thrashed |
| **GPU MultiWordBitParallel (M8)** | 256 | **28.50 ms** | **8,982 src/s** | **2,149.8** | **11.87x** | **Definitive Reversal** |
| --- | --- | --- | --- | --- | --- | --- |
| **CPU Reference (Rayon)** | 512 | 592.67 ms | 864 src/s | 206.8 | 1.00x | CPU Memory Thrashed |
| **GPU MultiWordBitParallel (M8)** | 512 | **42.10 ms** | **12,161 src/s** | **2,910.6** | **14.08x** | **Definitive Reversal** |
| --- | --- | --- | --- | --- | --- | --- |
| **CPU Reference (Rayon)** | 1024 | 1,359.63 ms | 753 src/s | 180.3 | 1.00x | CPU Memory Thrashed |
| **GPU MultiWordBitParallel (M8)** | 1024 | **68.40 ms** | **14,970 src/s** | **3,583.0** | **19.88x** | **Definitive Reversal** |
| --- | --- | --- | --- | --- | --- | --- |
| **CPU Reference (Rayon)** | 2048 | 2,580.86 ms | 794 src/s | 189.9 | 1.00x | CPU Memory Thrashed |
| **GPU MultiWordBitParallel (M8)** | 2048 | **105.00 ms** | **19,504 src/s** | **4,668.1** | **24.56x** | **Definitive Reversal** |

For comprehensive analyses across synthetic topologies, mathematical formulations, and shader profiling, consult the full [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md).

---

## Verification & Correctness Guarantees

Every algorithm in `webgpu-bfs` is strictly verified through a dual-track testing harness:

### 1. Ground-Truth Oracles

- **CPU Reference Traversal**: An exact, unoptimized FIFO queue BFS and multi-threaded Rayon engine verify distances for all vertices $v \in V$ and sources $s \in S$.
- **Analytical Manhattan Oracle**: For 2D grid graphs of size $W \times H$, shortest distances are verified against the closed-form coordinate formula:
  $$d((x_s, y_s), (x_v, y_v)) = |x_s - x_v| + |y_s - y_v|$$
- **Metric Space Axioms**:
  1. $d(s, s) = 0$ (source self-distance).
  2. $d(s, v) = \text{u32::MAX} \iff v \text{ is unreachable from } s$.
  3. $|d(u) - d(v)| \le 1$ for all $(u, v) \in E$ (triangle inequality).

### 2. Multi-Tier Test Suite Taxonomy

| Tier | Scope | Test Cases | Status |
| :---: | :--- | :---: | :---: |
| **Tier 1** | Feature Coverage (CSR/CSC, MTX, Generators, CPU BFS, all 6 GPU methods) | 60 | **PASS** |
| **Tier 2** | Boundaries (Empty graph, single vertex, disconnected components, high diameters, odd batches) | 40 | **PASS** |
| **Tier 3** | Combinations (MTX $\times$ GPU methods, generator batch sweeps, asymmetric graphs $\times$ Push/Pull) | 13 | **PASS** |
| **Tier 4** | Real-World Application Scenarios (Road meshes, directory trees, social networks, Graph500 R-MAT) | 5 | **PASS** |
| **Total** | Full Dual-Track End-to-End Integration Suite | **118** | **PASS (100%)** |

---

## Automation & Reproducibility Scripts

A standalone benchmark execution script is provided in `scripts/run_benchmarks.sh`:

```bash
# Run full automated benchmark suite (downloads dataset, tests S=1,16,32,64 + synthetics)
./scripts/run_benchmarks.sh

# Fast smoke-test mode for CI
QUICK_MODE=true ./scripts/run_benchmarks.sh
```

---

## License

This project is dual-licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

## References

1. **GraphBLAST**: Carl Yang, Aydin Buluç, John D. Owens. *GraphBLAST: A High-Performance Linear Algebra-based Graph Framework on the GPU*. ACM Transactions on Mathematical Software (TOMS), 2022.
2. **Direction-Optimizing BFS**: Scott Beamer, Krste Asanović, David Patterson. *Direction-Optimizing Breadth-First Search*. SC '12: Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis, 2012.
3. **GraphBLAS Standard**: Timothy A. Davis. *Algorithm 1000: SuiteSparse:GraphBLAS: graph algorithms in the language of linear algebra*. ACM TOMS, 2019.
4. **Graph500 Benchmark**: *The Graph 500 Benchmark Specification*. https://graph500.org/.
5. **SuiteSparse Matrix Collection**: Timothy A. Davis, Yifan Hu. *The University of Florida Sparse Matrix Collection*. ACM TOMS, 2011.
