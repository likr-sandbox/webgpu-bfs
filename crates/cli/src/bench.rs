use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use webgpu_bfs_core::{BfsEngine, CsrGraph};
use webgpu_bfs_cpu::CpuBfsEngine;
use webgpu_bfs_gpu::{GpuBfsEngine, GpuBfsMethod, GpuContext};

use crate::args::MethodArg;
use crate::verify::{verify_engine_output, VerificationResult};

enum BenchEngine {
    Cpu(Box<dyn BfsEngine>),
    Gpu(GpuBfsEngine),
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkResult {
    pub method: String,
    pub warmup_iters: usize,
    pub timed_iters: usize,
    pub durations_ms: Vec<f64>,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub stddev_ms: f64,
    pub traversed_edges: usize,
    pub harmonic_mteps: f64,
    pub arithmetic_mteps: f64,
    pub throughput_sources_per_sec: f64,
    pub speedup_vs_baseline: f64,
    pub verification: Option<VerificationResult>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkSuiteResult {
    pub graph_name: String,
    pub num_vertices: usize,
    pub num_edges: usize,
    pub is_symmetric: bool,
    pub num_sources: usize,
    pub adapter_name: String,
    pub adapter_backend: String,
    pub adapter_device_type: String,
    pub results: Vec<BenchmarkResult>,
}

pub struct BenchmarkHarness {
    context: Arc<GpuContext>,
    warmup: usize,
    iters: usize,
    verify: bool,
    max_diffs: usize,
}

impl BenchmarkHarness {
    pub fn new(warmup: usize, iters: usize, verify: bool, max_diffs: usize) -> Result<Self> {
        let context = Arc::new(GpuContext::init_blocking()?);
        Ok(Self {
            context,
            warmup,
            iters,
            verify,
            max_diffs,
        })
    }

    #[allow(dead_code)]
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    pub fn run_suite(
        &self,
        graph_name: &str,
        graph: &CsrGraph,
        sources: &[u32],
        method: MethodArg,
    ) -> Result<BenchmarkSuiteResult> {
        let adapter_info = self.context.info();
        let methods_to_run = match method {
            MethodArg::All => vec![
                MethodArg::CpuReference,
                MethodArg::CpuSingleThread,
                MethodArg::NaiveSequential,
                MethodArg::BatchedDenseSpMM,
                MethodArg::BitParallel32,
                MethodArg::BitParallel64,
                MethodArg::BitParallel64Pull,
                MethodArg::BooleanReachability,
                MethodArg::PushPullAdaptive,
                MethodArg::SparseQueue,
                MethodArg::SparseBitParallel64,
                MethodArg::MultiWordBitParallel,
                MethodArg::IndirectMacroBatch,
                MethodArg::IndirectSparseQueuePush,
                MethodArg::MultiWordBitParallelPush,
            ],
            single => vec![single],
        };

        // 1. Calculate ground-truth traversed edges for the batch
        let cpu_engine = CpuBfsEngine::new();
        let ground_truth_dist = cpu_engine.run_ms_bfs(graph, sources)?;
        let traversed_edges = calculate_traversed_edges(graph, sources, &ground_truth_dist);

        let mut results = Vec::new();
        let mut baseline_mean_ms = None;

        let total_methods = methods_to_run.len();
        for (idx, &m) in methods_to_run.iter().enumerate() {
            eprintln!("[{}/{}] Running {:?}...", idx + 1, total_methods, m);
            let res = self.benchmark_single_method(
                graph,
                sources,
                m,
                traversed_edges,
                &ground_truth_dist,
            )?;
            eprintln!("    Completed in {:.2} ms (mean)", res.mean_ms);

            // Set baseline for speedup calculation (first method)
            if baseline_mean_ms.is_none() {
                baseline_mean_ms = Some(res.mean_ms);
            }

            results.push(res);
        }

        // Calculate speedup relative to baseline
        let base = baseline_mean_ms.unwrap_or(1.0);
        for r in &mut results {
            r.speedup_vs_baseline = if r.mean_ms > 0.0 {
                base / r.mean_ms
            } else {
                1.0
            };
        }

        Ok(BenchmarkSuiteResult {
            graph_name: graph_name.to_string(),
            num_vertices: graph.num_vertices,
            num_edges: graph.num_edges,
            is_symmetric: graph.is_symmetric(),
            num_sources: sources.len(),
            adapter_name: adapter_info.name.clone(),
            adapter_backend: format!("{:?}", adapter_info.backend),
            adapter_device_type: format!("{:?}", adapter_info.device_type),
            results,
        })
    }

    pub fn benchmark_single_method(
        &self,
        graph: &CsrGraph,
        sources: &[u32],
        method: MethodArg,
        traversed_edges: usize,
        ground_truth_dist: &[u32],
    ) -> Result<BenchmarkResult> {
        let (engine, method_name) = match method {
            MethodArg::CpuReference => (
                BenchEngine::Cpu(Box::new(CpuBfsEngine::rayon())),
                "CPU Reference (Rayon)".to_string(),
            ),
            MethodArg::CpuSingleThread => (
                BenchEngine::Cpu(Box::new(CpuBfsEngine::fifo())),
                "CPU Single-Thread (FIFO)".to_string(),
            ),
            MethodArg::NaiveSequential => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::NaiveSequential,
                )?),
                "GPU NaiveSequential".to_string(),
            ),
            MethodArg::BatchedDenseSpMM => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::BatchedDenseSpMM,
                )?),
                "GPU BatchedDenseSpMM".to_string(),
            ),
            MethodArg::BitParallel32 => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::BitParallel32,
                )?),
                "GPU BitParallel32".to_string(),
            ),
            MethodArg::BitParallel64 => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::BitParallel64,
                )?),
                "GPU BitParallel64".to_string(),
            ),
            MethodArg::BitParallel64Pull => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::BitParallel64Pull,
                )?),
                "GPU BitParallel64Pull".to_string(),
            ),
            MethodArg::BooleanReachability => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::BooleanReachability,
                )?),
                "GPU BooleanReachability".to_string(),
            ),
            MethodArg::PushPullAdaptive => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::PushPullAdaptive,
                )?),
                "GPU PushPullAdaptive".to_string(),
            ),
            MethodArg::SparseQueue => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::SparseQueue,
                )?),
                "GPU SparseQueue".to_string(),
            ),
            MethodArg::SparseBitParallel64 => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::SparseBitParallel64,
                )?),
                "GPU SparseBitParallel64".to_string(),
            ),
            MethodArg::MultiWordBitParallel => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::MultiWordBitParallel,
                )?),
                "GPU MultiWordBitParallel".to_string(),
            ),
            MethodArg::IndirectMacroBatch => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::IndirectMacroBatch,
                )?),
                "GPU IndirectMacroBatch".to_string(),
            ),
            MethodArg::IndirectSparseQueuePush => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::IndirectSparseQueuePush,
                )?),
                "GPU IndirectSparseQueuePush".to_string(),
            ),
            MethodArg::MultiWordBitParallelPush => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::MultiWordBitParallelPush,
                )?),
                "GPU MultiWordBitParallelPush".to_string(),
            ),
            MethodArg::Auto => (
                BenchEngine::Gpu(GpuBfsEngine::new(
                    self.context.clone(),
                    GpuBfsMethod::Auto,
                )?),
                "GPU Auto (Topology-Aware)".to_string(),
            ),
            MethodArg::All => unreachable!(),
        };

        let total_elements = sources.len() * graph.num_vertices;
        let mut reusable_output = vec![0u32; total_elements];

        // A. Warm-up Phase
        for _ in 0..self.warmup {
            match &engine {
                BenchEngine::Cpu(cpu) => {
                    let _ = cpu.run_ms_bfs(graph, sources)?;
                }
                BenchEngine::Gpu(gpu) => {
                    gpu.run_ms_bfs_into(graph, sources, &mut reusable_output)?;
                }
            }
        }

        // B. Timed Measurement Phase
        let mut durations = Vec::with_capacity(self.iters);
        let mut last_output = Vec::new();

        for _ in 0..self.iters {
            let start = Instant::now();
            match &engine {
                BenchEngine::Cpu(cpu) => {
                    let dist = cpu.run_ms_bfs(graph, sources)?;
                    let elapsed = start.elapsed();
                    durations.push(elapsed);
                    last_output = dist;
                }
                BenchEngine::Gpu(gpu) => {
                    gpu.run_ms_bfs_into(graph, sources, &mut reusable_output)?;
                    let elapsed = start.elapsed();
                    durations.push(elapsed);
                }
            }
        }

        if matches!(engine, BenchEngine::Gpu(_)) && self.iters > 0 {
            last_output = reusable_output;
        }

        // C. Statistical Aggregation
        let durations_ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();

        let sum_ms: f64 = durations_ms.iter().sum();
        let count = durations_ms.len() as f64;
        let mean_ms = if count > 0.0 { sum_ms / count } else { 0.0 };

        let mut sorted_ms = durations_ms.clone();
        sorted_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_ms = if sorted_ms.is_empty() {
            0.0
        } else if sorted_ms.len().is_multiple_of(2) {
            let mid = sorted_ms.len() / 2;
            (sorted_ms[mid - 1] + sorted_ms[mid]) / 2.0
        } else {
            sorted_ms[sorted_ms.len() / 2]
        };

        let min_ms = *sorted_ms.first().unwrap_or(&0.0);
        let max_ms = *sorted_ms.last().unwrap_or(&0.0);

        let variance = if count > 1.0 {
            durations_ms
                .iter()
                .map(|v| (v - mean_ms).powi(2))
                .sum::<f64>()
                / (count - 1.0)
        } else {
            0.0
        };
        let stddev_ms = variance.sqrt();

        // D. Graph500 Harmonic MTEPS & Throughput
        let mean_sec = mean_ms / 1000.0;
        let total_sec: f64 = durations.iter().map(|d| d.as_secs_f64()).sum();

        let harmonic_mteps = if total_sec > 0.0 {
            ((count * traversed_edges as f64) / total_sec) * 1e-6
        } else {
            0.0
        };

        let arithmetic_mteps = if mean_sec > 0.0 {
            (traversed_edges as f64 / mean_sec) * 1e-6
        } else {
            0.0
        };

        let throughput_sources_per_sec = if mean_sec > 0.0 {
            sources.len() as f64 / mean_sec
        } else {
            0.0
        };

        // E. Verification Check
        let verification = if self.verify {
            Some(verify_engine_output(
                &method_name,
                graph,
                sources,
                &last_output,
                ground_truth_dist,
                self.max_diffs,
            ))
        } else {
            None
        };

        Ok(BenchmarkResult {
            method: method_name,
            warmup_iters: self.warmup,
            timed_iters: self.iters,
            durations_ms,
            mean_ms,
            median_ms,
            min_ms,
            max_ms,
            stddev_ms,
            traversed_edges,
            harmonic_mteps,
            arithmetic_mteps,
            throughput_sources_per_sec,
            speedup_vs_baseline: 1.0,
            verification,
        })
    }
}

pub fn calculate_traversed_edges(graph: &CsrGraph, sources: &[u32], dist: &[u32]) -> usize {
    let nv = graph.num_vertices;
    let mut total_edges = 0usize;

    for (s_idx, _) in sources.iter().enumerate() {
        for u in 0..nv {
            let d = dist[s_idx * nv + u];
            if d != u32::MAX {
                total_edges += graph.degree(u);
            }
        }
    }
    total_edges
}
