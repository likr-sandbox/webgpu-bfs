/**
 * Interactive Graph Topology & BFS Wave Animation Renderer
 * Vanilla Canvas 2D engine with pan/zoom, multiple layout algorithms,
 * and multi-source BFS frontier wave propagation visualizer.
 */

export class GraphRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {HTMLElement} tooltipEl
   */
  constructor(canvas, tooltipEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.tooltipEl = tooltipEl;

    // Graph state
    this.graph = null; // { numVertices, numEdges, rowOffsets, colIndices, width, height, type }
    this.nodePositions = []; // [{ x, y }]

    // BFS Distance state
    this.distances = null; // Uint32Array (flat: s * V + v)
    this.sources = [0];
    this.displaySourceIdx = 0; // index into sources array
    this.maxDistance = 0;

    // Animation & Wave state
    this.currentHop = 0;
    this.isPlaying = false;
    this.animSpeed = 150; // ms per hop
    this.lastFrameTime = 0;
    this.animFrameId = null;

    // Viewport transform (Pan & Zoom)
    this.panX = 0;
    this.panY = 0;
    this.zoom = 1.0;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;

    // Hover state
    this.hoveredNode = -1;

    // Current layout mode: 'auto', 'grid2d', 'radial_bfs', 'force'
    this.layoutMode = 'auto';

    this.setupEventListeners();
    this.resize();
  }

  setupEventListeners() {
    window.addEventListener('resize', () => {
      this.resize();
      this.render();
    });

    const wrapper = this.canvas.parentElement;

    wrapper.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.dragStartX = e.clientX - this.panX;
      this.dragStartY = e.clientY - this.panY;
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        this.panX = e.clientX - this.dragStartX;
        this.panY = e.clientY - this.dragStartY;
        this.render();
      } else {
        this.handleMouseMove(e);
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    wrapper.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      const newZoom = Math.max(0.05, Math.min(20.0, this.zoom * zoomFactor));

      // Zoom towards mouse position
      this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
      this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
      this.zoom = newZoom;

      this.render();
    }, { passive: false });
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = Math.floor(this.width * dpr);
    this.canvas.height = Math.floor(this.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  setGraph(graph) {
    this.graph = graph;
    this.distances = null;
    this.currentHop = 0;
    this.isPlaying = false;
    this.computeLayout();
    this.fitToView();
    this.render();
  }

  setBfsResults(distances, sources) {
    this.distances = distances;
    this.sources = sources && sources.length > 0 ? sources : [0];
    this.displaySourceIdx = 0;

    // Compute max reachable distance for display source
    this.updateMaxDistance();
    this.currentHop = 0;

    if (this.layoutMode === 'radial_bfs') {
      this.computeLayout();
    }

    this.render();
  }

  setDisplaySource(sourceIndex) {
    if (sourceIndex >= 0 && sourceIndex < this.sources.length) {
      this.displaySourceIdx = sourceIndex;
      this.updateMaxDistance();
      if (this.layoutMode === 'radial_bfs') {
        this.computeLayout();
      }
      this.render();
    }
  }

  updateMaxDistance() {
    if (!this.graph || !this.distances) {
      this.maxDistance = 0;
      return;
    }
    const V = this.graph.numVertices;
    const offset = this.displaySourceIdx * V;
    let maxD = 0;
    for (let v = 0; v < V; v++) {
      const d = this.distances[offset + v];
      if (d !== 0xFFFFFFFF && d > maxD) {
        maxD = d;
      }
    }
    this.maxDistance = maxD;
  }

  computeLayout() {
    if (!this.graph) return;
    const V = this.graph.numVertices;
    this.nodePositions = new Array(V);

    const mode = this.layoutMode === 'auto'
      ? (this.graph.type === 'grid2d' ? 'grid2d' : 'radial_bfs')
      : this.layoutMode;

    if (mode === 'grid2d' && this.graph.width && this.graph.height) {
      const W = this.graph.width;
      const H = this.graph.height;
      const spacing = 36;
      for (let v = 0; v < V; v++) {
        const x = (v % W) * spacing;
        const y = Math.floor(v / W) * spacing;
        this.nodePositions[v] = { x, y };
      }
    } else if (mode === 'radial_bfs' && this.distances) {
      // Concentric rings arranged by BFS hop distance from active source
      const V = this.graph.numVertices;
      const offset = this.displaySourceIdx * V;
      const buckets = Array.from({ length: this.maxDistance + 2 }, () => []);

      for (let v = 0; v < V; v++) {
        const d = this.distances[offset + v];
        const bucketIdx = (d === 0xFFFFFFFF) ? this.maxDistance + 1 : d;
        buckets[bucketIdx].push(v);
      }

      const ringRadiusStep = 55;
      for (let hop = 0; hop <= this.maxDistance + 1; hop++) {
        const nodesInRing = buckets[hop];
        const r = (hop === 0) ? 0 : hop * ringRadiusStep;
        const count = nodesInRing.length;
        for (let i = 0; i < count; i++) {
          const v = nodesInRing[i];
          const theta = (2 * Math.PI * i) / count;
          this.nodePositions[v] = {
            x: r * Math.cos(theta),
            y: r * Math.sin(theta),
          };
        }
      }
    } else {
      // Circle / Spring layout fallback
      const radius = Math.min(this.width, this.height) * 0.42;
      for (let v = 0; v < V; v++) {
        const theta = (2 * Math.PI * v) / V;
        this.nodePositions[v] = {
          x: radius * Math.cos(theta),
          y: radius * Math.sin(theta),
        };
      }
    }
  }

  fitToView() {
    if (!this.nodePositions.length) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const pos of this.nodePositions) {
      if (pos.x < minX) minX = pos.x;
      if (pos.x > maxX) maxX = pos.x;
      if (pos.y < minY) minY = pos.y;
      if (pos.y > maxY) maxY = pos.y;
    }

    const graphW = Math.max(1, maxX - minX);
    const graphH = Math.max(1, maxY - minY);
    const margin = 60;

    const availableW = this.width - margin * 2;
    const availableH = this.height - margin * 2;

    const zoomX = availableW / graphW;
    const zoomY = availableH / graphH;
    this.zoom = Math.min(zoomX, zoomY, 2.5);

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    this.panX = this.width / 2 - centerX * this.zoom;
    this.panY = this.height / 2 - centerY * this.zoom;
  }

  handleMouseMove(e) {
    if (!this.graph || !this.nodePositions.length) return;
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const worldX = (mouseX - this.panX) / this.zoom;
    const worldY = (mouseY - this.panY) / this.zoom;

    const threshold = Math.max(10, 14 / this.zoom);
    let nearestNode = -1;
    let nearestDistSq = threshold * threshold;

    for (let v = 0; v < this.graph.numVertices; v++) {
      const pos = this.nodePositions[v];
      const dx = pos.x - worldX;
      const dy = pos.y - worldY;
      const dSq = dx * dx + dy * dy;
      if (dSq < nearestDistSq) {
        nearestDistSq = dSq;
        nearestNode = v;
      }
    }

    if (nearestNode !== this.hoveredNode) {
      this.hoveredNode = nearestNode;
      if (nearestNode >= 0) {
        this.showTooltip(nearestNode, mouseX, mouseY);
      } else {
        this.hideTooltip();
      }
      this.render();
    }
  }

  showTooltip(nodeId, screenX, screenY) {
    if (!this.tooltipEl || !this.graph) return;
    const degree = this.graph.rowOffsets[nodeId + 1] - this.graph.rowOffsets[nodeId];
    let distStr = 'Unvisited';
    if (this.distances) {
      const d = this.distances[this.displaySourceIdx * this.graph.numVertices + nodeId];
      distStr = (d === 0xFFFFFFFF) ? 'Unreachable' : `${d} hops`;
    }

    this.tooltipEl.innerHTML = `
      <div><strong>Vertex #${nodeId}</strong></div>
      <div>Degree: ${degree}</div>
      <div>Distance: ${distStr}</div>
    `;
    this.tooltipEl.style.left = `${screenX + 12}px`;
    this.tooltipEl.style.top = `${screenY + 12}px`;
    this.tooltipEl.classList.remove('hidden');
  }

  hideTooltip() {
    if (this.tooltipEl) {
      this.tooltipEl.classList.add('hidden');
    }
  }

  play() {
    this.isPlaying = true;
    this.lastFrameTime = performance.now();
    this.loop();
  }

  pause() {
    this.isPlaying = false;
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
  }

  loop(timestamp = performance.now()) {
    if (!this.isPlaying) return;

    if (timestamp - this.lastFrameTime >= this.animSpeed) {
      this.lastFrameTime = timestamp;
      if (this.currentHop < this.maxDistance) {
        this.currentHop++;
      } else {
        this.pause(); // Reached end of traversal
      }
      this.render();
    }

    this.animFrameId = requestAnimationFrame((ts) => this.loop(ts));
  }

  render() {
    const { ctx, width, height } = this;
    ctx.clearRect(0, 0, width, height);

    if (!this.graph || !this.nodePositions.length) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Select graph parameters and click "Generate Graph"', width / 2, height / 2);
      return;
    }

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    const V = this.graph.numVertices;
    const activeSource = this.sources[this.displaySourceIdx] || 0;
    const distOffset = this.distances ? this.displaySourceIdx * V : 0;

    // 1. Draw Edges
    const shouldDrawEdges = V <= 1000 || this.zoom > 0.8;
    if (shouldDrawEdges) {
      for (let u = 0; u < V; u++) {
        const start = this.graph.rowOffsets[u];
        const end = this.graph.rowOffsets[u + 1];
        const pU = this.nodePositions[u];

        const dU = this.distances ? this.distances[distOffset + u] : 0xFFFFFFFF;

        for (let j = start; j < end; j++) {
          const v = this.graph.colIndices[j];
          if (v < u) continue; // Draw undirected edge once
          const pV = this.nodePositions[v];

          const dV = this.distances ? this.distances[distOffset + v] : 0xFFFFFFFF;

          // Determine edge highlight
          let strokeColor = 'rgba(45, 60, 85, 0.4)';
          let lineWidth = 1.0;

          if (this.distances) {
            const isFrontierEdge = (dU === this.currentHop && dV === this.currentHop + 1) ||
                                  (dV === this.currentHop && dU === this.currentHop + 1);
            const isTraversedEdge = dU <= this.currentHop && dV <= this.currentHop;

            if (isFrontierEdge) {
              strokeColor = '#06b6d4'; // Bright cyan wave
              lineWidth = 2.0;
            } else if (isTraversedEdge) {
              strokeColor = 'rgba(59, 130, 246, 0.5)'; // Visited blue
              lineWidth = 1.2;
            }
          }

          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = lineWidth;
          ctx.beginPath();
          ctx.moveTo(pU.x, pU.y);
          ctx.lineTo(pV.x, pV.y);
          ctx.stroke();
        }
      }
    }

    // 2. Draw Vertices
    const nodeRadius = Math.max(2, Math.min(8, 20 / Math.sqrt(V)));

    for (let u = 0; u < V; u++) {
      const pos = this.nodePositions[u];
      const dist = this.distances ? this.distances[distOffset + u] : 0xFFFFFFFF;

      let fillColor = '#334155'; // Unreached dark slate
      let isSource = (u === activeSource);
      let isFrontier = (this.distances && dist === this.currentHop);
      let isVisited = (this.distances && dist !== 0xFFFFFFFF && dist <= this.currentHop);

      if (isSource) {
        fillColor = '#f59e0b'; // Amber source
      } else if (isFrontier) {
        fillColor = '#06b6d4'; // Cyan frontier
      } else if (isVisited) {
        // Gradient from blue to purple based on distance
        const t = Math.min(1.0, dist / Math.max(1, this.maxDistance));
        fillColor = this.interpolateColor('#3b82f6', '#8b5cf6', t);
      }

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isSource || isFrontier ? nodeRadius * 1.4 : nodeRadius, 0, Math.PI * 2);
      ctx.fillStyle = fillColor;
      ctx.fill();

      // Frontier Glow Ring
      if (isFrontier) {
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.7)';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeRadius * 2.0, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Hover highlight
      if (u === this.hoveredNode) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeRadius + 3, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  interpolateColor(color1, color2, factor) {
    const c1 = parseInt(color1.slice(1), 16);
    const c2 = parseInt(color2.slice(1), 16);

    const r1 = (c1 >> 16) & 255, g1 = (c1 >> 8) & 255, b1 = c1 & 255;
    const r2 = (c2 >> 16) & 255, g2 = (c2 >> 8) & 255, b2 = c2 & 255;

    const r = Math.round(r1 + factor * (r2 - r1));
    const g = Math.round(g1 + factor * (g2 - g1));
    const b = Math.round(b1 + factor * (b2 - b1));

    return `rgb(${r}, ${g}, ${b})`;
  }
}
