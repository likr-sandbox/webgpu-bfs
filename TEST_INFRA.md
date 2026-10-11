# WebGPU Multi-Source BFS (MS-BFS) Test Infrastructure

This document details the test infrastructure, runtime environment, ground-truth verification oracles, diagnostic harnesses, and execution protocols for the `webgpu-bfs` project.

---

## 1. Testing Philosophy & Guarantees

The test suite enforces a **Dual-Track Opaque-Box Verification** philosophy:

1. **Opaque-Box Boundary**: Integration tests interact exclusively with public crates, types, and traits (`webgpu_bfs_core::BfsEngine`, `CsrGraph`, `CscGraph`, `CpuBfsEngine`, `GpuBfsEngine`, `GpuBfsMethod`, `Grid2D`, `from_mtx_reader`, procedural generators). No internal crate structures, raw bind group handles, WGSL private layouts, or private module functions are accessed.
2. **Dual-Track Redundancy**: GPU compute results are never verified against heuristics or unchecked assumptions alone. Every test is verified against at least one of:
   - An exact CPU reference BFS implementation (FIFO, Rayon batched, or 64-bit bit-parallel).
   - A closed-form analytical mathematical distance oracle (e.g., 2D grid Manhattan distance: $d(s, v) = |x_s - x_v| + |y_s - y_v|$).
   - Topological metric space axioms (source self-distance 0, unreachable sentinel `0xFFFF_FFFF`, and edge-wise triangle inequality $|d(u) - d(v)| \le 1$).
3. **Bitwise Determinism**: For any given graph $G=(V, E)$ and source set $S \subset V$, all 6 GPU BFS implementations (`NaiveSequential`, `BatchedDenseSpMM`, `BitParallel32`, `BitParallel64`, `BooleanReachability`, `PushPullAdaptive`) must yield distance arrays with 100% bitwise parity.
4. **Progressive Testability & Isolation**: Every test is self-contained and isolated, initializing its own test graphs and verifying its own invariants without shared mutable state or execution-order dependence.

---

## 2. Hardware & Driver Execution Stack

The WebGPU compute infrastructure is built on `wgpu 24.0` with WGSL compute shaders.

### 2.1 3-Tier Adapter Fallback
To ensure test portability across CI runners, headless development servers, and hardware accelerators, the test harness initializes WebGPU via `GpuContext::init_blocking()`:
- **Tier 1 (Hardware GPU)**: Requests `wgpu::PowerPreference::HighPerformance`.
- **Tier 2 (Fallback Software Adapter)**: Requests `wgpu::RequestAdapterOptions { force_fallback_adapter: true, .. }`.
- **Tier 3 (Enumerated Fallback)**: Iterates over `instance.enumerate_adapters(Backends::all())` and selects the first available adapter supporting `compute`.

### 2.2 Headless Linux Mesa Lavapipe Support
On headless Linux servers lacking discrete GPUs, the harness automatically binds to the Mesa Lavapipe Vulkan 1.3 software rasterizer. All compute pipelines, atomic operations (`atomicOr`, `atomicAdd`), and workgroup storage barriers execute with full hardware feature equivalence.

---

## 3. Test Suite Taxonomy (Tiers 1–4)

The E2E suite (`tests/e2e_tests.rs`) is partitioned into 4 distinct testing tiers comprising 118 automated test cases:

| Tier | Module Path | Test Cases | Verification Focus |
|---|---|:---:|---|
| **Tier 1** | `tests/tier1_features/` | 60 | F1–F11 Feature coverage: CSR/CSC construction, transposition, MTX reader, procedural generators (Grid2D, ER, Tree, R-MAT), CPU BFS engines, and all 6 GPU MS-BFS methods. |
| **Tier 2** | `tests/tier2_boundaries/` | 40 | Pathological edge cases: empty graphs, single vertex, zero edges, multigraph dedup, disconnected components, odd/prime batch sizes ($S \in \{1, 3, 17, 31, 33, 63, 65, 129\}$), extreme diameters, and hub contention. |
| **Tier 3** | `tests/tier3_combinations/` | 13 | Pairwise cross-feature combinations: MTX reader $\times$ all 6 GPU methods; procedural generators $\times$ non-power-of-two batch sweeps; directed asymmetric graphs $\times$ Push/Pull adaptive controller; multi-chunk memory layout isolation. |
| **Tier 4** | `tests/tier4_real_world/` | 5 | Domain-specific application scenarios: 2D road navigation grids; deep directory hierarchies ($N=500$); disconnected social network communities ($N=450$); Graph500 R-MAT scale-free networks ($N=512$); simulated road network corridors ($N=400$, diameter $>100$). |
| **Total E2E** | `tests/e2e_tests.rs` | **118** | Complete dual-track integration suite |

---

## 4. Feature Inventory Coverage Matrix (F1–F11)

| Feature | Description | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Primary Oracles |
|---|---|:---:|:---:|:---:|:---:|---|
| **F1** | Graph Core CSR/CSC | `t1_01`, `t1_02` | `b2`, `b3`, `b4` | `t3_01`, `t3_03` | `t4_01`–`t4_05` | Invariant validator, counting-sort isomorphism |
| **F2** | Matrix Market Parser | `t1_03` | `b4_04` | `t3_01` | – | Spec format validator, CPU parity |
| **F3** | Procedural Generators | `t1_04`–`t1_07` | `b7`, `b8` | `t3_02` | `t4_01`, `t4_02`, `t4_04` | Analytical Manhattan, topological invariants |
| **F4** | CPU Reference BFS | `t1_08` | `b1`–`b8` | `t3_01`–`t3_04` | `t4_01`–`t4_05` | FIFO queue, Rayon thread pool, BitParallel64 |
| **F5** | WGSL Compute Shaders | `t1_09`–`t1_12` | `b1`–`b8` | `t3_01`–`t3_04` | `t4_01`–`t4_05` | Naga parser & compiler, compute execution |
| **F6** | WebGPU Runtime & Pipeline | `t1_09`–`t1_12` | `b1`–`b8` | `t3_01`–`t3_04` | `t4_01`–`t4_05` | Shared GpuContext, Lavapipe Vulkan |
| **F7** | Baseline GPU MS-BFS | `t1_09` | `b1`–`b8` | `t3_01`, `t3_02` | `t4_01`–`t4_03` | CPU ground truth, Manhattan oracle |
| **F8** | Bit-Parallel MS-BFS (32/64) | `t1_10` | `b6` | `t3_02`, `t3_04` | `t4_01`–`t4_04` | 32-bit & 64-bit parity, chunk isolation |
| **F9** | Boolean Reachability | `t1_11` | `b3`, `b5` | `t3_03` | `t4_01`–`t4_04` | Write-once distance update, CPU parity |
| **F10** | Push/Pull Direction Opt. | `t1_12` | `b7`, `b8` | `t3_03` | `t4_01`–`t4_05` | Adaptive $\alpha/\beta$ switching, CSC pull |
| **F11** | Dual Track E2E Suite | Yes | Yes | Yes | Yes | Multi-tier test suite runner |

---

## 5. Ground Truth Verification Oracles

### Oracle 1: CPU Reference BFS (`CpuBfsEngine`)
`CpuBfsEngine` serves as the primary algorithmic oracle. Implemented in `crates/cpu_bfs`, it provides three independent algorithmic paths:
- `SequentialFifo`: Single-source queue traversal with $O(|V| + |E|)$ time.
- `RayonBatched`: Multi-threaded parallel batch execution across CPU cores.
- `BitParallel64`: 64-bit bitmask frontier expansion using CPU SIMD/bit-scan.
All distance outputs follow the canonical flat memory layout:
$$\text{dist}[s \cdot |V| + v]$$

### Oracle 2: Closed-Form Manhattan Distance (`Grid2D`)
For an undirected grid graph of dimensions $W \times H$ where vertex $v$ maps to coordinate $(x_v, y_v) = (v \bmod W, \lfloor v / W \rfloor)$, the shortest path distance from source $s = (x_s, y_s)$ is analytically computed as:
$$d(s, v) = |x_s - x_v| + |y_s - y_v|$$
This oracle provides an analytical ground truth completely independent of any graph traversal algorithm.

### Oracle 3: Metric Space Graph Axioms & Triangle Inequality
For any valid graph and distance vector, the harness enforces the metric space properties:
1. **Source Self-Distance**: $d(s, s) = 0$.
2. **Identity of Indiscernibles**: $d(s, v) = 0 \iff v = s$ (for connected non-trivial graphs).
3. **Unreachable Sentinel**: $d(s, v) = \text{u32::MAX} \iff v \text{ is unreachable from } s$.
4. **Undirected Triangle Inequality**: For every edge $(u, v) \in E$, if $u$ and $v$ are reachable:
   $$|d(u) - d(v)| \le 1$$
5. **Directed Triangle Inequality**: For every directed edge $u \to v$, if $u$ is reachable:
   $$d(v) \le d(u) + 1$$
6. **Reachability Monotonicity**: For an undirected edge $(u, v)$, $u \text{ reachable} \iff v \text{ reachable}$.

### Oracle 4: 6-Way Cross-Method Bitwise Equivalence
The 6 GPU execution methods (`NaiveSequential`, `BatchedDenseSpMM`, `BitParallel32`, `BitParallel64`, `BooleanReachability`, `PushPullAdaptive`) must compute bitwise identical distance outputs for any graph and source set.

---

## 6. Diagnostic Error Reporting

Upon any distance mismatch or invariant violation, tests emit a structured diagnostic banner containing:
- Failing method, graph dimensions ($|V|, |E|$), and source set $S$.
- Aggregate failure metrics: total mismatches and mismatch percentage.
- First mismatch context: flat index, source index and ID, target vertex, expected distance, actual distance, target degree, and adjacent outgoing neighbor lists.

---

## 7. Execution Commands & Protocols

```bash
# Ensure Cargo is on PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Run full E2E test suite (118 tests)
cargo test --test e2e_tests

# Run full E2E test suite with real-time diagnostic output
cargo test --test e2e_tests -- --nocapture

# Run specific testing tiers
cargo test --test e2e_tests tier1
cargo test --test e2e_tests tier2
cargo test --test e2e_tests tier3
cargo test --test e2e_tests tier4

# Run workspace-wide test suite (all unit tests + E2E integration tests)
cargo test --workspace

# Lint and formatting validation
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
