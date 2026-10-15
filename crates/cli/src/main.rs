mod args;
mod bench;
mod dataset;
mod graph_loader;
mod report;
mod verify;

use anyhow::Result;
use clap::Parser;
use std::process::ExitCode;

use args::{Cli, Commands, RunArgs};
use bench::BenchmarkHarness;
use dataset::{list_datasets, run_download_dataset};
use graph_loader::load_graph_and_sources;
use report::render_and_save_report;

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match run_app(cli) {
        Ok(exit_code) => exit_code,
        Err(err) => {
            eprintln!("\n❌ Error: {:?}", err);
            ExitCode::FAILURE
        }
    }
}

fn run_app(cli: Cli) -> Result<ExitCode> {
    if let Some(cmd) = cli.command {
        match cmd {
            Commands::ListDatasets => {
                list_datasets();
                return Ok(ExitCode::SUCCESS);
            }

            Commands::DownloadDataset(args) => {
                run_download_dataset(&args)?;
                return Ok(ExitCode::SUCCESS);
            }

            Commands::Verify(args) => {
                let (graph_name, graph, sources) =
                    load_graph_and_sources(&args.graph, &args.sources)?;

                println!(
                    "Verifying graph '{}' (|V|={}, |E|={}) with {} sources...",
                    graph_name,
                    graph.num_vertices,
                    graph.num_edges,
                    sources.len()
                );

                let harness = BenchmarkHarness::new(1, 1, true, args.max_diffs)?
                    .with_reachability(args.reachability);
                let suite = harness.run_suite(&graph_name, &graph, &sources, args.method)?;

                render_and_save_report(&suite, args.format, args.output.as_deref())?;

                let all_passed = suite
                    .results
                    .iter()
                    .all(|r| r.verification.as_ref().map(|v| v.is_valid).unwrap_or(true));

                return Ok(if all_passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                });
            }

            Commands::Bench(args) => {
                let (graph_name, graph, sources) =
                    load_graph_and_sources(&args.graph, &args.sources)?;

                let harness =
                    BenchmarkHarness::new(args.warmup, args.iters, args.verify, args.max_diffs)?
                        .with_reachability(args.reachability);
                let suite = harness.run_suite(&graph_name, &graph, &sources, args.method)?;

                render_and_save_report(&suite, args.format, args.output.as_deref())?;

                let all_passed = suite
                    .results
                    .iter()
                    .all(|r| r.verification.as_ref().map(|v| v.is_valid).unwrap_or(true));

                return Ok(if all_passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                });
            }

            Commands::Run(run_args) => {
                return execute_run_command(run_args);
            }
        }
    }

    // Default: execute top-level arguments
    execute_run_command(cli.run_args)
}

fn execute_run_command(args: RunArgs) -> Result<ExitCode> {
    let (graph_name, graph, sources) = load_graph_and_sources(&args.graph, &args.sources)?;

    let iters = if args.bench { args.iters } else { 1 };
    let warmup = if args.bench { args.warmup } else { 0 };

    let harness = BenchmarkHarness::new(warmup, iters, args.verify, args.max_diffs)?
        .with_reachability(args.reachability);
    let suite = harness.run_suite(&graph_name, &graph, &sources, args.method)?;

    render_and_save_report(&suite, args.format, args.output.as_deref())?;

    let all_passed = suite
        .results
        .iter()
        .all(|r| r.verification.as_ref().map(|v| v.is_valid).unwrap_or(true));

    Ok(if all_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_grid_format() {
        let (w, h) = graph_loader::parse_grid_dimensions("30x40").unwrap();
        assert_eq!(w, 30);
        assert_eq!(h, 40);
    }

    #[test]
    fn test_cli_help_parses() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
