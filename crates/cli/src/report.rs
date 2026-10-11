use anyhow::{Context, Result};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::args::OutputFormat;
use crate::bench::BenchmarkSuiteResult;

pub fn render_and_save_report(
    suite: &BenchmarkSuiteResult,
    format: OutputFormat,
    output_path: Option<&Path>,
) -> Result<()> {
    let rendered = match format {
        OutputFormat::Table => format_table(suite),
        OutputFormat::Markdown => format_markdown(suite),
        OutputFormat::Csv => format_csv(suite),
        OutputFormat::Json => {
            serde_json::to_string_pretty(suite).context("Failed to serialize suite to JSON")?
        }
    };

    if let Some(path) = output_path {
        let mut file = File::create(path)
            .with_context(|| format!("Failed to create output file {:?}", path))?;
        file.write_all(rendered.as_bytes())
            .with_context(|| format!("Failed to write report to {:?}", path))?;
        println!("Report saved to: {}", path.display());
    } else {
        println!("{}", rendered);
    }

    Ok(())
}

pub fn format_table(suite: &BenchmarkSuiteResult) -> String {
    let mut out = String::new();
    out.push_str("\n========================================================================================\n");
    out.push_str("                       WebGPU Multi-Source BFS Benchmark Report                         \n");
    out.push_str("========================================================================================\n");
    out.push_str(&format!(
        " Graph:         {} (|V| = {}, |E| = {}, symmetric = {})\n",
        suite.graph_name, suite.num_vertices, suite.num_edges, suite.is_symmetric
    ));
    out.push_str(&format!(" Sources:       {} sources\n", suite.num_sources));
    out.push_str(&format!(
        " Device:        {} ({}, {})\n",
        suite.adapter_name, suite.adapter_backend, suite.adapter_device_type
    ));
    out.push_str("----------------------------------------------------------------------------------------\n");
    out.push_str(
        " Method                 Mean (ms)  Median(ms)   Min (ms)   Max (ms)  Harmonic MTEPS  Sources/s  Speedup  Parity\n",
    );
    out.push_str("----------------------------------------------------------------------------------------\n");

    for r in &suite.results {
        let parity_str = match &r.verification {
            Some(v) if v.is_valid => "✅ 100.0%",
            Some(_) => "❌ Diff",
            None => "  -  ",
        };

        out.push_str(&format!(
            " {:<22} {:9.2}  {:9.2}  {:8.2}  {:8.2}  {:13.2}  {:9.0}  {:6.2}x  {}\n",
            r.method,
            r.mean_ms,
            r.median_ms,
            r.min_ms,
            r.max_ms,
            r.harmonic_mteps,
            r.throughput_sources_per_sec,
            r.speedup_vs_baseline,
            parity_str
        ));
    }
    out.push_str("========================================================================================\n");
    out
}

pub fn format_markdown(suite: &BenchmarkSuiteResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "### WebGPU Multi-Source BFS Benchmark: `{}`\n\n",
        suite.graph_name
    ));
    out.push_str(&format!(
        "- **Graph**: `|V| = {}`, `|E| = {}`, Symmetric: `{}`\n",
        suite.num_vertices, suite.num_edges, suite.is_symmetric
    ));
    out.push_str(&format!("- **Batch**: `{}` sources\n", suite.num_sources));
    out.push_str(&format!(
        "- **Hardware**: `{}` (`{}`, `{}`)\n\n",
        suite.adapter_name, suite.adapter_backend, suite.adapter_device_type
    ));

    out.push_str("| Method | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Harmonic MTEPS | Sources/s | Speedup | Parity |\n");
    out.push_str("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n");

    for r in &suite.results {
        let parity_str = match &r.verification {
            Some(v) if v.is_valid => "✅ 100.0%",
            Some(_) => "❌ Failed",
            None => "N/A",
        };

        out.push_str(&format!(
            "| **{}** | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} | {:.0} | {:.2}x | {} |\n",
            r.method,
            r.mean_ms,
            r.median_ms,
            r.min_ms,
            r.max_ms,
            r.harmonic_mteps,
            r.throughput_sources_per_sec,
            r.speedup_vs_baseline,
            parity_str
        ));
    }
    out
}

pub fn format_csv(suite: &BenchmarkSuiteResult) -> String {
    let mut out = String::new();
    out.push_str("graph,vertices,edges,sources,method,mean_ms,median_ms,min_ms,max_ms,stddev_ms,harmonic_mteps,throughput_src_per_sec,speedup,verified,mismatches\n");

    for r in &suite.results {
        let (verified, mismatches) = match &r.verification {
            Some(v) => (v.is_valid, v.mismatched_elements),
            None => (false, 0),
        };

        out.push_str(&format!(
            "{},{},{},{},\"{}\",{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.4},{},{}\n",
            suite.graph_name,
            suite.num_vertices,
            suite.num_edges,
            suite.num_sources,
            r.method,
            r.mean_ms,
            r.median_ms,
            r.min_ms,
            r.max_ms,
            r.stddev_ms,
            r.harmonic_mteps,
            r.throughput_sources_per_sec,
            r.speedup_vs_baseline,
            verified,
            mismatches
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::BenchmarkResult;

    #[test]
    fn test_format_table_and_markdown_and_csv() {
        let suite = BenchmarkSuiteResult {
            graph_name: "TestGraph".to_string(),
            num_vertices: 100,
            num_edges: 400,
            is_symmetric: true,
            num_sources: 4,
            adapter_name: "MockAdapter".to_string(),
            adapter_backend: "Vulkan".to_string(),
            adapter_device_type: "Cpu".to_string(),
            results: vec![BenchmarkResult {
                method: "BitParallel64".to_string(),
                warmup_iters: 1,
                timed_iters: 2,
                durations_ms: vec![10.0, 10.0],
                mean_ms: 10.0,
                median_ms: 10.0,
                min_ms: 10.0,
                max_ms: 10.0,
                stddev_ms: 0.0,
                traversed_edges: 1600,
                harmonic_mteps: 160.0,
                arithmetic_mteps: 160.0,
                throughput_sources_per_sec: 400.0,
                speedup_vs_baseline: 1.0,
                verification: None,
            }],
        };

        let table = format_table(&suite);
        assert!(table.contains("TestGraph"));
        assert!(table.contains("BitParallel64"));

        let md = format_markdown(&suite);
        assert!(md.contains("### WebGPU Multi-Source BFS Benchmark"));
        assert!(md.contains("| **BitParallel64** |"));

        let csv = format_csv(&suite);
        assert!(csv.contains("graph,vertices,edges,sources,method"));
        assert!(csv.contains("TestGraph,100,400,4,\"BitParallel64\""));
    }
}
