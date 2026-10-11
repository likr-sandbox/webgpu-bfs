/**
 * WebGPU Multi-Source BFS Dashboard Main Application (app.js)
 * Coordinates UI events, WebGPU detection, graph generation, WASM execution,
 * canvas animation, and benchmark comparative analysis.
 *
 * Seamlessly integrates with `crates/wasm` exports (WasmGraph, WasmGpuContext,
 * run_gpu_bfs, run_cpu_bfs, run_benchmark_suite, verify_equivalence)
 * with robust client-side pure JS fallback for testing and preview.
 */

import { GraphRenderer } from './graph_renderer.js';
import { BenchmarkChart } from './benchmark_chart.js';
import { MtxParser } from './mtx_parser.js';

// Global application state
const state = {
  gpuSupported: false,
  gpuAdapter: null,
  gpuInfo: null,
  wasmModule: null,
  wasmGpuCtx: null,
  wasmGraph: null,
  graph: null, // Unified JS view { numVertices, numEdges, rowOffsets, colIndices, type }
  bfsResult: null,
  benchmarkResults: [],
  selectedSources: [0],
  activeAlgorithm: 'PushPullAdaptive',
  uploadedMtxText: null,
};

// DOM Element References
const elements = {
  gpuStatusBadge: document.getElementById('gpu-status-badge'),
  gpuStatusText: document.getElementById('gpu-status-text'),
  wasmStatusBadge: document.getElementById('wasm-status-badge'),
  wasmStatusText: document.getElementById('wasm-status-text'),
  webgpuBanner: document.getElementById('webgpu-banner'),
  bannerDismiss: document.getElementById('banner-dismiss'),

  graphTypeSelect: document.getElementById('graph-type-select'),
  btnGenerateGraph: document.getElementById('btn-generate-graph'),
  mtxFileInput: document.getElementById('mtx-file-input'),
  mtxDropzone: document.getElementById('mtx-dropzone'),
  mtxFileName: document.getElementById('mtx-file-name'),

  statVertexCount: document.getElementById('stat-vertex-count'),
  statEdgeCount: document.getElementById('stat-edge-count'),
  statAvgDegree: document.getElementById('stat-avg-degree'),
  statGraphType: document.getElementById('stat-graph-type'),

  algorithmSelect: document.getElementById('algorithm-select'),
  sourceModeButtons: document.querySelectorAll('.source-mode-selector .btn-toggle'),
  sourceIndicesInput: document.getElementById('source-indices-input'),
  sourceHint: document.getElementById('source-hint'),

  btnRunBfs: document.getElementById('btn-run-bfs'),
  btnRunBenchmark: document.getElementById('btn-run-benchmark'),

  graphCanvas: document.getElementById('graph-canvas'),
  nodeTooltip: document.getElementById('node-tooltip'),
  vizLayoutSelect: document.getElementById('viz-layout-select'),
  vizSourceSelect: document.getElementById('viz-source-select'),
  btnAnimPlay: document.getElementById('btn-anim-play'),
  btnAnimStepPrev: document.getElementById('btn-anim-step-prev'),
  btnAnimStepNext: document.getElementById('btn-anim-step-next'),
  btnAnimReset: document.getElementById('btn-anim-reset'),
  animSpeedSlider: document.getElementById('anim-speed-slider'),
  animStepLabel: document.getElementById('anim-step-label'),
  legendFrontierStat: document.getElementById('legend-frontier-stat'),
  btnZoomIn: document.getElementById('btn-zoom-in'),
  btnZoomOut: document.getElementById('btn-zoom-out'),
  btnZoomFit: document.getElementById('btn-zoom-fit'),

  metricTime: document.getElementById('metric-time'),
  metricMteps: document.getElementById('metric-mteps'),
  metricEdges: document.getElementById('metric-edges'),
  metricVertices: document.getElementById('metric-vertices'),

  benchmarkChart: document.getElementById('benchmark-chart'),
  benchmarkTableBody: document.getElementById('benchmark-table-body'),
  btnExportCsv: document.getElementById('btn-export-csv'),
  btnExportJson: document.getElementById('btn-export-json'),

  tabButtons: document.querySelectorAll('.tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  logWindow: document.getElementById('log-window'),
  logAutoscroll: document.getElementById('log-autoscroll'),
  btnClearLogs: document.getElementById('btn-clear-logs'),
  btnCopyLogs: document.getElementById('btn-copy-logs'),
};

// Sub-components
let graphRenderer = null;
let benchmarkChart = null;

// =============================================================================
// 1. Initialization & Feature Detection
// =============================================================================

async function initApp() {
  log('info', 'Initializing WebGPU MS-BFS web runtime...');

  // Initialize Canvas Sub-engines
  graphRenderer = new GraphRenderer(elements.graphCanvas, elements.nodeTooltip);
  benchmarkChart = new BenchmarkChart(elements.benchmarkChart);

  setupUIEventListeners();

  // 1. Check browser WebGPU capabilities
  await checkWebGpuSupport();

  // 2. Load Rust WASM module
  await loadWasmModule();

  // 3. Generate initial default graph (20x20 Grid2D)
  await generateCurrentGraph();
}

/**
 * Checks for navigator.gpu and requests adapter to verify compute capability.
 */
async function checkWebGpuSupport() {
  if (!navigator.gpu) {
    showGpuFallbackBanner('WebGPU API is not supported by this browser.');
    updateGpuStatusBadge(false, 'WebGPU Unsupported');
    return;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });

    if (!adapter) {
      showGpuFallbackBanner('WebGPU supported, but no compatible GPU adapter was found.');
      updateGpuStatusBadge(false, 'No Adapter');
      return;
    }

    state.gpuSupported = true;
    state.gpuAdapter = adapter;

    // Inspect adapter info
    let adapterInfo = null;
    if (adapter.requestAdapterInfo) {
      adapterInfo = await adapter.requestAdapterInfo();
    } else if (adapter.info) {
      adapterInfo = adapter.info;
    }

    const deviceName = adapterInfo?.device || adapterInfo?.vendor || 'Hardware GPU';
    state.gpuInfo = adapterInfo;

    updateGpuStatusBadge(true, `WebGPU Active (${deviceName})`);
    log('gpu', `WebGPU Adapter detected: ${deviceName} [Backend: ${adapterInfo?.architecture || 'Unknown'}]`);
  } catch (err) {
    showGpuFallbackBanner(`WebGPU initialization error: ${err.message}`);
    updateGpuStatusBadge(false, 'WebGPU Error');
  }
}

function updateGpuStatusBadge(active, text) {
  elements.gpuStatusBadge.className = active ? 'status-badge status-success' : 'status-badge status-danger';
  elements.gpuStatusText.textContent = text;
}

function showGpuFallbackBanner(msg) {
  elements.webgpuBanner.classList.remove('banner-hidden');
  const bannerMsg = document.getElementById('banner-message');
  if (bannerMsg) {
    bannerMsg.innerHTML = `${msg} GPU compute algorithms are disabled. Falling back gracefully to CPU Reference BFS.`;
  }
  // Disable GPU options in algorithm selector
  const gpuOptGroup = elements.algorithmSelect.querySelector('optgroup[label*="GPU"]');
  if (gpuOptGroup) {
    const options = gpuOptGroup.querySelectorAll('option');
    options.forEach(opt => opt.disabled = true);
  }
  elements.algorithmSelect.value = 'CpuReference';
  state.activeAlgorithm = 'CpuReference';
}

/**
 * Dynamically loads the wasm-pack output bundle if available,
 * otherwise provides transparent JS fallback.
 */
async function loadWasmModule() {
  try {
    // Attempt dynamic import of wasm package (from relative path ./pkg/webgpu_bfs_wasm.js)
    const wasmPkg = await import('./pkg/webgpu_bfs_wasm.js');
    if (wasmPkg.default) {
      await wasmPkg.default();
    }
    state.wasmModule = wasmPkg;

    // Initialize WASM logger and panic hook
    if (wasmPkg.init_wasm) {
      wasmPkg.init_wasm();
    }

    // Initialize WasmGpuContext if WebGPU is active
    if (state.gpuSupported && wasmPkg.WasmGpuContext) {
      try {
        state.wasmGpuCtx = await wasmPkg.WasmGpuContext.create();
        log('gpu', `WasmGpuContext initialized: ${state.wasmGpuCtx.adapter_name()}`);
      } catch (gpuInitErr) {
        log('warn', `WasmGpuContext creation failed: ${gpuInitErr}. Using CPU execution.`);
      }
    }

    elements.wasmStatusBadge.className = 'status-badge status-success';
    elements.wasmStatusText.textContent = 'WASM Ready';
    log('info', 'Loaded Rust WebAssembly bundle successfully.');
  } catch (e) {
    // Graceful fallback to pure JS execution mode
    log('warn', `WASM bundle './pkg/webgpu_bfs_wasm.js' not found. Using JS Reference Mode. (${e.message})`);
    elements.wasmStatusBadge.className = 'status-badge status-checking';
    elements.wasmStatusText.textContent = 'JS Reference Mode';
  }
}

// =============================================================================
// 2. Graph Generation Handlers
// =============================================================================

async function generateCurrentGraph() {
  const type = elements.graphTypeSelect.value;
  log('info', `Generating graph type: ${type}...`);

  try {
    let graph = null;
    let wasmGraph = null;

    if (state.wasmModule && state.wasmModule.WasmGraph) {
      const WasmGraph = state.wasmModule.WasmGraph;
      if (type === 'grid2d') {
        const w = parseInt(document.getElementById('grid-width').value, 10);
        const h = parseInt(document.getElementById('grid-height').value, 10);
        wasmGraph = WasmGraph.generate_grid2d(w, h);
        graph = {
          numVertices: wasmGraph.num_vertices(),
          numEdges: wasmGraph.num_edges(),
          rowOffsets: wasmGraph.row_offsets(),
          colIndices: wasmGraph.col_indices(),
          width: w,
          height: h,
          type: 'grid2d',
        };
      } else if (type === 'erdos_renyi') {
        const n = parseInt(document.getElementById('er-vertices').value, 10);
        const p = parseFloat(document.getElementById('er-prob').value);
        const seed = document.getElementById('er-seed').value ? parseFloat(document.getElementById('er-seed').value) : undefined;
        const targetEdges = Math.floor(n * (n - 1) * p / 2);
        wasmGraph = WasmGraph.generate_erdos_renyi(n, targetEdges, seed);
        graph = {
          numVertices: wasmGraph.num_vertices(),
          numEdges: wasmGraph.num_edges(),
          rowOffsets: wasmGraph.row_offsets(),
          colIndices: wasmGraph.col_indices(),
          type: 'erdos_renyi',
        };
      } else if (type === 'random_tree') {
        const n = parseInt(document.getElementById('tree-vertices').value, 10);
        const seed = document.getElementById('tree-seed').value ? parseFloat(document.getElementById('tree-seed').value) : undefined;
        wasmGraph = WasmGraph.generate_random_tree(n, seed);
        graph = {
          numVertices: wasmGraph.num_vertices(),
          numEdges: wasmGraph.num_edges(),
          rowOffsets: wasmGraph.row_offsets(),
          colIndices: wasmGraph.col_indices(),
          type: 'random_tree',
        };
      } else if (type === 'rmat') {
        const scale = parseInt(document.getElementById('rmat-scale').value, 10);
        const edgeFactor = parseInt(document.getElementById('rmat-edge-factor').value, 10);
        const seed = document.getElementById('rmat-seed').value ? parseFloat(document.getElementById('rmat-seed').value) : undefined;
        wasmGraph = WasmGraph.generate_rmat(scale, edgeFactor, seed);
        graph = {
          numVertices: wasmGraph.num_vertices(),
          numEdges: wasmGraph.num_edges(),
          rowOffsets: wasmGraph.row_offsets(),
          colIndices: wasmGraph.col_indices(),
          type: 'rmat',
        };
      } else if (type === 'mtx_upload') {
        if (!state.uploadedMtxText) {
          alert('Please select or drag & drop a .mtx file first.');
          return;
        }
        wasmGraph = WasmGraph.from_mtx(state.uploadedMtxText);
        graph = {
          numVertices: wasmGraph.num_vertices(),
          numEdges: wasmGraph.num_edges(),
          rowOffsets: wasmGraph.row_offsets(),
          colIndices: wasmGraph.col_indices(),
          type: 'mtx',
        };
      }
      state.wasmGraph = wasmGraph;
    } else {
      // Client-side pure JS Procedural Graph Fallback
      if (type === 'grid2d') {
        const w = parseInt(document.getElementById('grid-width').value, 10);
        const h = parseInt(document.getElementById('grid-height').value, 10);
        graph = generateGrid2d(w, h);
      } else if (type === 'erdos_renyi') {
        const n = parseInt(document.getElementById('er-vertices').value, 10);
        const p = parseFloat(document.getElementById('er-prob').value);
        graph = generateErdosRenyi(n, p);
      } else if (type === 'random_tree') {
        const n = parseInt(document.getElementById('tree-vertices').value, 10);
        graph = generateRandomTree(n);
      } else if (type === 'rmat') {
        const scale = parseInt(document.getElementById('rmat-scale').value, 10);
        const edgeFactor = parseInt(document.getElementById('rmat-edge-factor').value, 10);
        graph = generateRmat(scale, edgeFactor);
      } else if (type === 'mtx_upload') {
        if (!state.uploadedMtxText) {
          alert('Please select or drag & drop a .mtx file first.');
          return;
        }
        graph = MtxParser.parse(state.uploadedMtxText);
        graph.type = 'mtx';
      }
    }

    state.graph = graph;
    updateGraphStatsDisplay(graph);
    graphRenderer.setGraph(graph);

    // Update source selection
    updateSourceIndices();

    log('info', `Graph active: |V| = ${graph.numVertices.toLocaleString()}, |E| = ${graph.numEdges.toLocaleString()}, avg degree = ${(graph.numEdges / Math.max(1, graph.numVertices)).toFixed(2)}`);
  } catch (err) {
    log('error', `Graph generation failed: ${err.message}`);
    alert(`Failed to generate graph: ${err.message}`);
  }
}

function updateGraphStatsDisplay(graph) {
  elements.statVertexCount.textContent = graph.numVertices.toLocaleString();
  elements.statEdgeCount.textContent = graph.numEdges.toLocaleString();
  elements.statAvgDegree.textContent = (graph.numEdges / Math.max(1, graph.numVertices)).toFixed(2);
  elements.statGraphType.textContent = (graph.type || 'CUSTOM').toUpperCase();

  // Populate visualizer source selector
  elements.vizSourceSelect.innerHTML = '';
  for (let i = 0; i < state.selectedSources.length; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Source #${i} (Node ${state.selectedSources[i]})`;
    elements.vizSourceSelect.appendChild(opt);
  }
}

// Procedural JS Generators (Pure JS fallback)
function generateGrid2d(width, height) {
  const numVertices = width * height;
  const numEdges = 2 * ((width - 1) * height + width * (height - 1));
  const rowOffsets = new Uint32Array(numVertices + 1);
  const colIndices = new Uint32Array(numEdges);

  let offset = 0;
  rowOffsets[0] = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = y * width + x;
      if (y > 0) colIndices[offset++] = v - width; // North
      if (x > 0) colIndices[offset++] = v - 1;     // West
      if (x + 1 < width) colIndices[offset++] = v + 1; // East
      if (y + 1 < height) colIndices[offset++] = v + width; // South
      rowOffsets[v + 1] = offset;
    }
  }

  return {
    numVertices,
    numEdges,
    rowOffsets,
    colIndices,
    width,
    height,
    type: 'grid2d',
  };
}

function generateErdosRenyi(n, p) {
  const rawEdges = [];
  for (let u = 0; u < n - 1; u++) {
    for (let v = u + 1; v < n; v++) {
      if (Math.random() < p) {
        rawEdges.push([u, v]);
        rawEdges.push([v, u]);
      }
    }
  }
  return buildCsrFromPairs(n, rawEdges, 'erdos_renyi');
}

function generateRandomTree(n) {
  const rawEdges = [];
  const perm = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }

  for (let i = 1; i < n; i++) {
    const parentIdx = Math.floor(Math.random() * i);
    const u = perm[i];
    const v = perm[parentIdx];
    rawEdges.push([u, v]);
    rawEdges.push([v, u]);
  }

  return buildCsrFromPairs(n, rawEdges, 'random_tree');
}

function generateRmat(scale, edgeFactor) {
  const n = 1 << scale;
  const targetEdges = n * edgeFactor;
  const a = 0.57, b = 0.19, c = 0.19;
  const rawEdges = [];

  for (let i = 0; i < targetEdges; i++) {
    let u = 0, v = 0;
    for (let step = scale - 1; step >= 0; step--) {
      const bit = 1 << step;
      const r = Math.random();
      if (r < a) {
        // top-left (0, 0)
      } else if (r < a + b) {
        v |= bit; // top-right (0, 1)
      } else if (r < a + b + c) {
        u |= bit; // bottom-left (1, 0)
      } else {
        u |= bit; // bottom-right (1, 1)
        v |= bit;
      }
    }
    if (u !== v) {
      rawEdges.push([u, v]);
      rawEdges.push([v, u]);
    }
  }

  return buildCsrFromPairs(n, rawEdges, 'rmat');
}

function buildCsrFromPairs(numVertices, pairs, type) {
  const adj = Array.from({ length: numVertices }, () => []);
  for (const [u, v] of pairs) {
    if (u < numVertices && v < numVertices && u !== v) {
      adj[u].push(v);
    }
  }

  let totalEdges = 0;
  for (let u = 0; u < numVertices; u++) {
    adj[u].sort((a, b) => a - b);
    const dedup = [];
    for (let j = 0; j < adj[u].length; j++) {
      if (j === 0 || adj[u][j] !== adj[u][j - 1]) {
        dedup.push(adj[u][j]);
      }
    }
    adj[u] = dedup;
    totalEdges += dedup.length;
  }

  const rowOffsets = new Uint32Array(numVertices + 1);
  const colIndices = new Uint32Array(totalEdges);
  let offset = 0;
  rowOffsets[0] = 0;

  for (let u = 0; u < numVertices; u++) {
    for (let j = 0; j < adj[u].length; j++) {
      colIndices[offset + j] = adj[u][j];
    }
    offset += adj[u].length;
    rowOffsets[u + 1] = offset;
  }

  return {
    numVertices,
    numEdges: totalEdges,
    rowOffsets,
    colIndices,
    type,
  };
}

// =============================================================================
// 3. Multi-Source Configuration
// =============================================================================

function updateSourceIndices() {
  const inputStr = elements.sourceIndicesInput.value.trim();
  const tokens = inputStr.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));

  const validSources = [];
  const V = state.graph ? state.graph.numVertices : 1;

  for (const s of tokens) {
    if (s >= 0 && s < V && !validSources.includes(s)) {
      validSources.push(s);
    }
  }

  if (validSources.length === 0) {
    validSources.push(0);
  }

  state.selectedSources = validSources;
  elements.sourceHint.textContent = `Source count: ${validSources.length} (Valid IDs: [${validSources.slice(0, 4).join(', ')}${validSources.length > 4 ? '...' : ''}])`;

  if (state.graph) {
    updateGraphStatsDisplay(state.graph);
  }
}

// =============================================================================
// 4. Execution & BFS Traversal Engines
// =============================================================================

async function runBfs() {
  if (!state.graph) return;
  const method = elements.algorithmSelect.value;
  const sources = state.selectedSources;

  log('info', `Running BFS [Method: ${method}] for ${sources.length} sources...`);
  elements.btnRunBfs.disabled = true;

  try {
    let result = null;

    if (state.wasmModule && state.wasmGpuCtx && method !== 'CpuReference' && state.wasmModule.run_gpu_bfs) {
      // 1. WASM GPU Execution
      const sourcesU32 = new Uint32Array(sources);
      const wasmRes = await state.wasmModule.run_gpu_bfs(state.wasmGpuCtx, state.wasmGraph, sourcesU32, method);
      result = {
        distances: wasmRes.distances(),
        executionTimeMs: wasmRes.elapsed_ms,
        mteps: wasmRes.mteps,
        traversedEdges: wasmRes.traversed_edges,
        visitedVertices: wasmRes.visited_vertices,
      };
    } else if (state.wasmModule && method === 'CpuReference' && state.wasmModule.run_cpu_bfs) {
      // 2. WASM CPU Execution
      const sourcesU32 = new Uint32Array(sources);
      const wasmRes = state.wasmModule.run_cpu_bfs(state.wasmGraph, sourcesU32, "Fifo");
      result = {
        distances: wasmRes.distances(),
        executionTimeMs: wasmRes.elapsed_ms,
        mteps: wasmRes.mteps,
        traversedEdges: wasmRes.traversed_edges,
        visitedVertices: wasmRes.visited_vertices,
      };
    } else {
      // 3. Fallback Client-side pure JS BFS engine
      const startTime = performance.now();
      result = executeJsBfs(state.graph, sources);
      const elapsedMs = performance.now() - startTime;
      result.executionTimeMs = elapsedMs;
      result.mteps = result.traversedEdges / (Math.max(0.001, elapsedMs) * 1000);
    }

    state.bfsResult = result;

    // Update KPI Scorecards
    elements.metricTime.textContent = `${result.executionTimeMs.toFixed(2)} ms`;
    elements.metricMteps.textContent = `${result.mteps.toFixed(1)}`;
    elements.metricEdges.textContent = result.traversedEdges.toLocaleString();
    elements.metricVertices.textContent = `${result.visitedVertices.toLocaleString()} (${((result.visitedVertices / state.graph.numVertices) * 100).toFixed(1)}%)`;

    // Update visualizer
    graphRenderer.setBfsResults(result.distances, sources);

    // Update hop count badge
    elements.animStepLabel.textContent = `Hop: 0 / ${graphRenderer.maxDistance}`;
    elements.legendFrontierStat.textContent = `Max Hop: ${graphRenderer.maxDistance}`;

    log('gpu', `Completed BFS in ${result.executionTimeMs.toFixed(2)} ms (${result.mteps.toFixed(1)} MTEPS, ${result.visitedVertices} visited).`);
  } catch (err) {
    log('error', `BFS execution failed: ${err.message}`);
    alert(`Execution failed: ${err.message}`);
  } finally {
    elements.btnRunBfs.disabled = false;
  }
}

/**
 * Pure JavaScript CPU Reference Multi-Source BFS (Exact ground truth equivalence).
 */
function executeJsBfs(graph, sources) {
  const numVertices = graph.numVertices;
  const numSources = sources.length;
  const distances = new Uint32Array(numSources * numVertices);
  distances.fill(0xFFFFFFFF);

  let totalTraversedEdges = 0;
  let totalVisited = 0;

  for (let sIdx = 0; sIdx < numSources; sIdx++) {
    const s = sources[sIdx];
    const offset = sIdx * numVertices;

    const queue = [];
    distances[offset + s] = 0;
    queue.push(s);
    let visitedCount = 1;
    let traversedEdges = 0;

    let head = 0;
    while (head < queue.length) {
      const u = queue[head++];
      const currDist = distances[offset + u];
      const start = graph.rowOffsets[u];
      const end = graph.rowOffsets[u + 1];

      for (let j = start; j < end; j++) {
        const v = graph.colIndices[j];
        traversedEdges++;
        if (distances[offset + v] === 0xFFFFFFFF) {
          distances[offset + v] = currDist + 1;
          visitedCount++;
          queue.push(v);
        }
      }
    }

    totalTraversedEdges += traversedEdges;
    totalVisited += visitedCount;
  }

  return {
    distances,
    visitedVertices: Math.round(totalVisited / numSources),
    traversedEdges: totalTraversedEdges,
  };
}

// =============================================================================
// 5. Benchmark Suite Comparison
// =============================================================================

async function runBenchmarkComparison() {
  if (!state.graph) return;
  const sources = state.selectedSources;
  elements.btnRunBenchmark.disabled = true;

  const algorithms = [
    { id: 'PushPullAdaptive', name: 'Push/Pull (GraphBLAST)', isCpu: false },
    { id: 'BitParallel64', name: 'Bit-Parallel 64', isCpu: false },
    { id: 'BitParallel32', name: 'Bit-Parallel 32', isCpu: false },
    { id: 'BooleanReachability', name: 'Boolean Reachability', isCpu: false },
    { id: 'BatchedDenseSpMM', name: 'Batched SpMM', isCpu: false },
    { id: 'NaiveSequential', name: 'Naive Sequential', isCpu: false },
    { id: 'CpuReference', name: 'CPU Reference', isCpu: true },
  ];

  log('bench', `Starting Benchmark Comparison Suite across ${algorithms.length} methods...`);
  const results = [];

  // Compute CPU ground truth for verification
  const refRes = executeJsBfs(state.graph, sources);

  for (const algo of algorithms) {
    if (!state.gpuSupported && !algo.isCpu) {
      continue; // Skip GPU methods if WebGPU unavailable
    }

    log('bench', `Running benchmark for ${algo.name}...`);
    try {
      const iterations = 3;
      let totalTime = 0;
      let lastRes = null;

      for (let iter = 0; iter < iterations; iter++) {
        const t0 = performance.now();
        if (state.wasmModule && state.wasmGpuCtx && !algo.isCpu && state.wasmModule.run_gpu_bfs) {
          const wasmRes = await state.wasmModule.run_gpu_bfs(state.wasmGpuCtx, state.wasmGraph, new Uint32Array(sources), algo.id);
          totalTime += wasmRes.elapsed_ms;
          lastRes = {
            distances: wasmRes.distances(),
            traversedEdges: wasmRes.traversed_edges,
            visitedVertices: wasmRes.visited_vertices,
          };
        } else {
          lastRes = executeJsBfs(state.graph, sources);
          totalTime += (performance.now() - t0);
        }
      }

      const avgTimeMs = totalTime / iterations;
      const mteps = lastRes.traversedEdges / (Math.max(0.001, avgTimeMs) * 1000);

      // Verify exact equivalence against CPU ground truth
      let match = true;
      if (lastRes.distances) {
        for (let i = 0; i < lastRes.distances.length; i++) {
          if (lastRes.distances[i] !== refRes.distances[i]) {
            match = false;
            break;
          }
        }
      }

      results.push({
        id: algo.id,
        name: algo.name,
        timeMs: avgTimeMs,
        mteps,
        traversedEdges: lastRes.traversedEdges,
        visitedVertices: lastRes.visitedVertices,
        isCpu: algo.isCpu,
        correctness: match ? '100% Match' : 'Mismatch',
      });
    } catch (err) {
      log('error', `Benchmark ${algo.name} failed: ${err.message}`);
    }
  }

  state.benchmarkResults = results;
  benchmarkChart.setData(results);
  renderBenchmarkTable(results);

  elements.btnExportCsv.disabled = false;
  elements.btnExportJson.disabled = false;
  elements.btnRunBenchmark.disabled = false;

  log('bench', 'Benchmark suite execution complete!');
}

function renderBenchmarkTable(results) {
  const tbody = elements.benchmarkTableBody;
  tbody.innerHTML = '';

  if (results.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No results.</td></tr>';
    return;
  }

  const cpuResult = results.find(r => r.isCpu);
  const cpuTime = cpuResult ? cpuResult.timeMs : 1.0;

  for (const r of results) {
    const tr = document.createElement('tr');
    const speedup = (cpuTime / Math.max(0.001, r.timeMs)).toFixed(2);
    const isOk = r.correctness.includes('Match');

    tr.innerHTML = `
      <td><strong>${r.name}</strong></td>
      <td>${r.timeMs.toFixed(2)} ms</td>
      <td>${r.mteps.toFixed(1)}</td>
      <td><span class="${parseFloat(speedup) >= 1.0 ? 'text-success' : 'text-dim'}">${speedup}x</span></td>
      <td><span class="${isOk ? 'text-success' : 'text-danger'}">${isOk ? '&#10004;' : '&#10008;'} ${r.correctness}</span></td>
    `;
    tbody.appendChild(tr);
  }
}

// =============================================================================
// 6. UI Event Listeners & Tab Controls
// =============================================================================

function setupUIEventListeners() {
  // Graph type select visibility toggle
  elements.graphTypeSelect.addEventListener('change', () => {
    const val = elements.graphTypeSelect.value;
    document.querySelectorAll('.param-group').forEach(el => el.classList.add('param-group-hidden'));
    const targetGroup = document.getElementById(`params-${val}`);
    if (targetGroup) targetGroup.classList.remove('param-group-hidden');
  });

  // Generate Graph Button
  elements.btnGenerateGraph.addEventListener('click', generateCurrentGraph);

  // File Dropzone & Picker
  elements.mtxFileInput.addEventListener('change', handleMtxFileSelect);
  elements.mtxDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.mtxDropzone.classList.add('dragover');
  });
  elements.mtxDropzone.addEventListener('dragleave', () => {
    elements.mtxDropzone.classList.remove('dragover');
  });
  elements.mtxDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.mtxDropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleMtxFile(e.dataTransfer.files[0]);
    }
  });

  // Source Mode Buttons
  elements.sourceModeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.sourceModeButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.mode;
      const V = state.graph ? state.graph.numVertices : 100;

      if (mode === 'single') {
        elements.sourceIndicesInput.value = '0';
      } else if (mode === 'batch32') {
        const count = Math.min(32, V);
        elements.sourceIndicesInput.value = Array.from({ length: count }, (_, i) => i).join(', ');
      } else if (mode === 'batch64') {
        const count = Math.min(64, V);
        elements.sourceIndicesInput.value = Array.from({ length: count }, (_, i) => i).join(', ');
      }
      updateSourceIndices();
    });
  });

  elements.sourceIndicesInput.addEventListener('input', updateSourceIndices);

  // Run BFS & Benchmark Buttons
  elements.btnRunBfs.addEventListener('click', runBfs);
  elements.btnRunBenchmark.addEventListener('click', runBenchmarkComparison);

  // Canvas Toolbar
  elements.vizLayoutSelect.addEventListener('change', (e) => {
    graphRenderer.layoutMode = e.target.value;
    graphRenderer.computeLayout();
    graphRenderer.fitToView();
    graphRenderer.render();
  });

  elements.vizSourceSelect.addEventListener('change', (e) => {
    graphRenderer.setDisplaySource(parseInt(e.target.value, 10));
  });

  elements.btnAnimPlay.addEventListener('click', () => {
    if (graphRenderer.isPlaying) {
      graphRenderer.pause();
      elements.btnAnimPlay.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
    } else {
      graphRenderer.play();
      elements.btnAnimPlay.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
    }
  });

  elements.btnAnimStepPrev.addEventListener('click', () => {
    if (graphRenderer.currentHop > 0) {
      graphRenderer.currentHop--;
      elements.animStepLabel.textContent = `Hop: ${graphRenderer.currentHop} / ${graphRenderer.maxDistance}`;
      graphRenderer.render();
    }
  });

  elements.btnAnimStepNext.addEventListener('click', () => {
    if (graphRenderer.currentHop < graphRenderer.maxDistance) {
      graphRenderer.currentHop++;
      elements.animStepLabel.textContent = `Hop: ${graphRenderer.currentHop} / ${graphRenderer.maxDistance}`;
      graphRenderer.render();
    }
  });

  elements.btnAnimReset.addEventListener('click', () => {
    graphRenderer.currentHop = 0;
    elements.animStepLabel.textContent = `Hop: 0 / ${graphRenderer.maxDistance}`;
    graphRenderer.render();
  });

  elements.animSpeedSlider.addEventListener('input', (e) => {
    graphRenderer.animSpeed = parseInt(e.target.value, 10);
  });

  elements.btnZoomIn.addEventListener('click', () => {
    graphRenderer.zoom *= 1.25;
    graphRenderer.render();
  });
  elements.btnZoomOut.addEventListener('click', () => {
    graphRenderer.zoom *= 0.8;
    graphRenderer.render();
  });
  elements.btnZoomFit.addEventListener('click', () => {
    graphRenderer.fitToView();
    graphRenderer.render();
  });

  // Tab Navigation
  elements.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.tabButtons.forEach(b => b.classList.remove('active'));
      elements.tabPanes.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const tabId = btn.dataset.tab;
      document.getElementById(tabId).classList.add('active');

      if (tabId === 'tab-benchmark') {
        benchmarkChart.resize();
        benchmarkChart.render();
      }
    });
  });

  // Log Panel Actions
  elements.btnClearLogs.addEventListener('click', () => {
    elements.logWindow.innerHTML = '';
  });
  elements.btnCopyLogs.addEventListener('click', () => {
    navigator.clipboard.writeText(elements.logWindow.innerText);
    log('info', 'Logs copied to clipboard.');
  });

  // Export CSV & JSON
  elements.btnExportCsv.addEventListener('click', exportBenchmarkCsv);
  elements.btnExportJson.addEventListener('click', exportBenchmarkJson);

  // Banner dismiss
  elements.bannerDismiss.addEventListener('click', () => {
    elements.webgpuBanner.classList.add('banner-hidden');
  });
}

function handleMtxFileSelect(e) {
  if (e.target.files.length) {
    handleMtxFile(e.target.files[0]);
  }
}

function handleMtxFile(file) {
  elements.mtxFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  log('info', `Reading Matrix Market file: ${file.name}...`);

  const reader = new FileReader();
  reader.onload = async (e) => {
    state.uploadedMtxText = e.target.result;
    await generateCurrentGraph();
  };
  reader.readAsText(file);
}

// =============================================================================
// 7. Logging & Export Utilities
// =============================================================================

function log(type, message) {
  const ts = new Date().toTimeString().split(' ')[0];
  const line = document.createElement('div');
  line.className = `log-line log-${type}`;
  line.innerHTML = `<span class="log-ts">[${ts}]</span> [${type.toUpperCase()}] ${message}`;

  elements.logWindow.appendChild(line);
  if (elements.logAutoscroll.checked) {
    elements.logWindow.scrollTop = elements.logWindow.scrollHeight;
  }
}

function exportBenchmarkCsv() {
  if (!state.benchmarkResults.length) return;
  const header = 'Algorithm,Time_ms,MTEPS,Traversed_Edges,Visited_Vertices,Correctness\n';
  const rows = state.benchmarkResults.map(r =>
    `"${r.name}",${r.timeMs.toFixed(3)},${r.mteps.toFixed(2)},${r.traversedEdges},${r.visitedVertices},"${r.correctness}"`
  ).join('\n');

  downloadFile('bfs_benchmark_results.csv', 'text/csv', header + rows);
}

function exportBenchmarkJson() {
  if (!state.benchmarkResults.length) return;
  const json = JSON.stringify(state.benchmarkResults, null, 2);
  downloadFile('bfs_benchmark_results.json', 'application/json', json);
}

function downloadFile(filename, type, content) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Start Application on DOM Ready
window.addEventListener('DOMContentLoaded', initApp);
