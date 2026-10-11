/**
 * Benchmark Comparison Chart Renderer
 * Vanilla Canvas 2D bar chart engine for MTEPS and execution runtime comparisons.
 * Zero external library dependencies.
 */

export class BenchmarkChart {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = []; // Array of { name, timeMs, mteps, isCpu, isFastest }
    this.mode = 'mteps'; // 'mteps' or 'time'
    this.resize();

    window.addEventListener('resize', () => {
      this.resize();
      this.render();
    });
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

  setData(results) {
    if (!results || results.length === 0) {
      this.data = [];
      this.render();
      return;
    }

    // Determine fastest GPU method
    let maxMteps = 0;
    let fastestMethod = '';
    for (const r of results) {
      if (!r.isCpu && r.mteps > maxMteps) {
        maxMteps = r.mteps;
        fastestMethod = r.name;
      }
    }

    this.data = results.map(r => ({
      ...r,
      isFastest: r.name === fastestMethod,
    }));

    this.render();
  }

  render() {
    const { ctx, width, height } = this;
    ctx.clearRect(0, 0, width, height);

    if (this.data.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No benchmark results yet. Run Benchmark Suite.', width / 2, height / 2);
      return;
    }

    const padding = { top: 20, right: 60, bottom: 25, left: 140 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const values = this.data.map(d => this.mode === 'mteps' ? d.mteps : d.timeMs);
    const maxVal = Math.max(...values, 0.001) * 1.15; // 15% headroom

    const barHeight = Math.min(24, Math.floor(chartH / this.data.length) - 6);
    const gap = Math.floor((chartH - (barHeight * this.data.length)) / (this.data.length + 1));

    // Draw grid lines
    ctx.strokeStyle = '#22314a';
    ctx.lineWidth = 1;
    const numGrid = 4;
    for (let i = 0; i <= numGrid; i++) {
      const x = padding.left + (chartW / numGrid) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartH);
      ctx.stroke();

      // X-axis label
      const val = (maxVal / numGrid) * i;
      ctx.fillStyle = '#64748b';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const label = this.mode === 'mteps' ? `${Math.round(val)} M` : `${val.toFixed(1)}ms`;
      ctx.fillText(label, x, padding.top + chartH + 6);
    }

    // Draw bars
    for (let i = 0; i < this.data.length; i++) {
      const item = this.data[i];
      const val = this.mode === 'mteps' ? item.mteps : item.timeMs;
      const barW = Math.max(2, (val / maxVal) * chartW);
      const y = padding.top + gap + i * (barHeight + gap);

      // Label (Algorithm name)
      ctx.fillStyle = item.isFastest ? '#10b981' : item.isCpu ? '#f59e0b' : '#f1f5f9';
      ctx.font = item.isFastest ? 'bold 11px -apple-system, sans-serif' : '11px -apple-system, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(item.name, padding.left - 10, y + barHeight / 2);

      // Bar fill
      let barGrad;
      if (item.isFastest) {
        barGrad = ctx.createLinearGradient(padding.left, 0, padding.left + barW, 0);
        barGrad.addColorStop(0, '#059669');
        barGrad.addColorStop(1, '#10b981');
      } else if (item.isCpu) {
        barGrad = ctx.createLinearGradient(padding.left, 0, padding.left + barW, 0);
        barGrad.addColorStop(0, '#d97706');
        barGrad.addColorStop(1, '#f59e0b');
      } else {
        barGrad = ctx.createLinearGradient(padding.left, 0, padding.left + barW, 0);
        barGrad.addColorStop(0, '#1d4ed8');
        barGrad.addColorStop(1, '#3b82f6');
      }

      ctx.fillStyle = barGrad;
      ctx.beginPath();
      ctx.roundRect(padding.left, y, barW, barHeight, [0, 4, 4, 0]);
      ctx.fill();

      // Bar Value Text
      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const valText = this.mode === 'mteps' ? `${item.mteps.toFixed(1)} MTEPS` : `${item.timeMs.toFixed(2)} ms`;
      ctx.fillText(valText, padding.left + barW + 8, y + barHeight / 2);
    }
  }
}
