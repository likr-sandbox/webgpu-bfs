use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "webgpu-bfs",
    version,
    about = "High-performance WebGPU Multi-Source BFS Benchmark & Verification Suite",
    long_about = "A high-performance GPU-parallel Multi-Source Breadth-First Search (MS-BFS) \
                  library and benchmark environment in Rust using WebGPU (wgpu) and WGSL compute shaders. \
                  Supports algebraic SpMV/SpMM, Push/Pull direction optimization, 32/64-bit bit-parallel \
                  batching, boolean reachability updates, and SuiteSparse Matrix Collection evaluation."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    #[command(flatten)]
    pub run_args: RunArgs,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Execute MS-BFS with single execution or verification
    Run(RunArgs),

    /// Benchmark multi-source BFS across warm-up and timed iterations
    Bench(BenchCommandArgs),

    /// Verify 100% bitwise distance equivalence against CPU reference ground truth
    Verify(VerifyCommandArgs),

    /// Download and extract SuiteSparse Matrix Collection datasets
    DownloadDataset(DownloadDatasetArgs),

    /// List known SuiteSparse benchmark datasets in the local catalog
    ListDatasets,
}

#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    #[command(flatten)]
    pub graph: GraphInputArgs,

    #[command(flatten)]
    pub sources: SourceArgs,

    /// Algorithmic strategy to execute
    #[arg(long, short = 'm', value_enum, default_value_t = MethodArg::BitParallel64)]
    pub method: MethodArg,

    /// Verify distance array against CPU reference implementation (100% parity check)
    #[arg(long)]
    pub verify: bool,

    /// Max number of discrepancy details to print on verification failure
    #[arg(long, default_value_t = 10)]
    pub max_diffs: usize,

    /// Run benchmark measurement mode with warm-up and timed iterations
    #[arg(long)]
    pub bench: bool,

    /// Number of warm-up iterations (untimed)
    #[arg(long, default_value_t = 3)]
    pub warmup: usize,

    /// Number of timed benchmark iterations
    #[arg(long, default_value_t = 10, alias = "iterations")]
    pub iters: usize,

    /// Output reporting format
    #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Table)]
    pub format: OutputFormat,

    /// Optional file path to write results (defaults to standard output)
    #[arg(long, short = 'o', value_name = "PATH")]
    pub output: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub struct BenchCommandArgs {
    #[command(flatten)]
    pub graph: GraphInputArgs,

    #[command(flatten)]
    pub sources: SourceArgs,

    /// Algorithmic strategy to benchmark (or 'all' for complete 7-way comparison)
    #[arg(long, short = 'm', value_enum, default_value_t = MethodArg::All)]
    pub method: MethodArg,

    /// Number of warm-up iterations (untimed)
    #[arg(long, default_value_t = 3)]
    pub warmup: usize,

    /// Number of timed iterations
    #[arg(long, default_value_t = 10)]
    pub iters: usize,

    /// Also verify bitwise correctness during benchmarking
    #[arg(long)]
    pub verify: bool,

    /// Max number of discrepancy details to print on verification failure
    #[arg(long, default_value_t = 10)]
    pub max_diffs: usize,

    /// Output reporting format
    #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Table)]
    pub format: OutputFormat,

    /// Optional file path to write results
    #[arg(long, short = 'o', value_name = "PATH")]
    pub output: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub struct VerifyCommandArgs {
    #[command(flatten)]
    pub graph: GraphInputArgs,

    #[command(flatten)]
    pub sources: SourceArgs,

    /// Algorithmic strategy to verify (or 'all' to verify all GPU implementations)
    #[arg(long, short = 'm', value_enum, default_value_t = MethodArg::All)]
    pub method: MethodArg,

    /// Max number of discrepancy details to print on verification failure
    #[arg(long, default_value_t = 10)]
    pub max_diffs: usize,

    /// Output reporting format
    #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Table)]
    pub format: OutputFormat,

    /// Optional file path to write results
    #[arg(long, short = 'o', value_name = "PATH")]
    pub output: Option<PathBuf>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct GraphInputArgs {
    /// Path to Matrix Market (.mtx) graph file
    #[arg(long, value_name = "PATH", conflicts_with_all = ["dataset", "grid", "grid_width", "er_vertices", "tree_vertices", "rmat_scale"])]
    pub mtx: Option<PathBuf>,

    /// Preset SuiteSparse dataset name (e.g., luxembourg_osm)
    #[arg(long, value_name = "NAME", conflicts_with_all = ["mtx", "grid", "grid_width", "er_vertices", "tree_vertices", "rmat_scale"])]
    pub dataset: Option<String>,

    /// 2D Grid graph dimensions in WxH format (e.g., "100x100")
    #[arg(long, value_name = "WxH", conflicts_with_all = ["mtx", "dataset", "grid_width", "er_vertices", "tree_vertices", "rmat_scale"])]
    pub grid: Option<String>,

    /// 2D Grid width (used in conjunction with --grid-height)
    #[arg(
        long,
        value_name = "WIDTH",
        requires = "grid_height",
        conflicts_with = "grid"
    )]
    pub grid_width: Option<usize>,

    /// 2D Grid height (used in conjunction with --grid-width)
    #[arg(
        long,
        value_name = "HEIGHT",
        requires = "grid_width",
        conflicts_with = "grid"
    )]
    pub grid_height: Option<usize>,

    /// Erdős–Rényi random graph vertex count |V|
    #[arg(long, value_name = "N", conflicts_with_all = ["mtx", "dataset", "grid", "tree_vertices", "rmat_scale"])]
    pub er_vertices: Option<usize>,

    /// Erdős–Rényi edge probability p in [0.0, 1.0] for G(n, p)
    #[arg(
        long,
        value_name = "P",
        requires = "er_vertices",
        conflicts_with = "er_m"
    )]
    pub er_p: Option<f64>,

    /// Erdős–Rényi exact edge count M for G(n, m)
    #[arg(
        long,
        value_name = "M",
        requires = "er_vertices",
        conflicts_with = "er_p"
    )]
    pub er_m: Option<usize>,

    /// Random connected tree vertex count |V| (Uniform Recursive Tree)
    #[arg(long, value_name = "N", conflicts_with_all = ["mtx", "dataset", "grid", "er_vertices", "rmat_scale"])]
    pub tree_vertices: Option<usize>,

    /// R-MAT power-law graph scale (vertex count |V| = 2^scale)
    #[arg(long, value_name = "SCALE", conflicts_with_all = ["mtx", "dataset", "grid", "er_vertices", "tree_vertices"])]
    pub rmat_scale: Option<usize>,

    /// R-MAT average degree edge factor (|E| = edge_factor * 2^scale)
    #[arg(
        long,
        value_name = "FACTOR",
        default_value_t = 16,
        requires = "rmat_scale"
    )]
    pub rmat_edge_factor: usize,

    /// R-MAT partition A probability
    #[arg(
        long,
        value_name = "FLOAT",
        default_value_t = 0.57,
        requires = "rmat_scale"
    )]
    pub rmat_a: f64,

    /// R-MAT partition B probability
    #[arg(
        long,
        value_name = "FLOAT",
        default_value_t = 0.19,
        requires = "rmat_scale"
    )]
    pub rmat_b: f64,

    /// R-MAT partition C probability
    #[arg(
        long,
        value_name = "FLOAT",
        default_value_t = 0.19,
        requires = "rmat_scale"
    )]
    pub rmat_c: f64,

    /// R-MAT partition D probability
    #[arg(
        long,
        value_name = "FLOAT",
        default_value_t = 0.05,
        requires = "rmat_scale"
    )]
    pub rmat_d: f64,

    /// Generate directed R-MAT graph (default: false, edges symmetrized)
    #[arg(long, requires = "rmat_scale")]
    pub rmat_directed: bool,

    /// PRNG seed for deterministic procedural generation and source sampling
    #[arg(long, value_name = "SEED")]
    pub seed: Option<u64>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct SourceArgs {
    /// Explicit comma-separated list of source vertex IDs (e.g., "0,10,42,100")
    #[arg(
        long,
        value_name = "SRC,...",
        value_delimiter = ',',
        conflicts_with = "num_sources"
    )]
    pub sources: Option<Vec<u32>>,

    /// Number of source vertices to sample randomly (batch size) [default: 1]
    #[arg(long, short = 's', value_name = "N", conflicts_with = "sources", alias = "batch-size")]
    pub num_sources: Option<usize>,

    /// Source vertex selection strategy
    #[arg(long, value_enum, default_value_t = SourceStrategy::Random)]
    pub source_strategy: SourceStrategy,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SourceStrategy {
    #[default]
    #[value(name = "random")]
    Random,
    #[value(name = "first")]
    First,
    #[value(name = "highest-degree", alias = "degree")]
    HighestDegree,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MethodArg {
    #[value(
        name = "NaiveSequential",
        alias = "naive",
        alias = "seq",
        alias = "naive-sequential"
    )]
    NaiveSequential,

    #[value(
        name = "BatchedDenseSpMM",
        alias = "batched",
        alias = "spmm",
        alias = "batched-dense-spmm"
    )]
    BatchedDenseSpMM,

    #[value(
        name = "BitParallel32",
        alias = "bit32",
        alias = "bp32",
        alias = "bit-parallel-32"
    )]
    BitParallel32,

    #[default]
    #[value(
        name = "BitParallel64",
        alias = "bit64",
        alias = "bp64",
        alias = "bit-parallel-64"
    )]
    BitParallel64,

    #[value(
        name = "BitParallel64Pull",
        alias = "pull64",
        alias = "bp64pull",
        alias = "bit-parallel-64-pull"
    )]
    BitParallel64Pull,

    #[value(
        name = "BooleanReachability",
        alias = "boolean",
        alias = "bool",
        alias = "reachability",
        alias = "boolean-reachability"
    )]
    BooleanReachability,

    #[value(
        name = "PushPullAdaptive",
        alias = "push-pull",
        alias = "adaptive",
        alias = "graphblast",
        alias = "pushpull"
    )]
    PushPullAdaptive,

    #[value(
        name = "SparseQueue",
        alias = "sparse",
        alias = "queue",
        alias = "sparse-queue"
    )]
    SparseQueue,

    #[value(
        name = "SparseBitParallel64",
        alias = "sparse-bp64",
        alias = "sparse-bp",
        alias = "sparse-bit-parallel-64"
    )]
    SparseBitParallel64,

    #[value(
        name = "MultiWordBitParallel",
        alias = "multi-word",
        alias = "multi-word-bp",
        alias = "multiword",
        alias = "mw-bp"
    )]
    MultiWordBitParallel,

    #[value(
        name = "IndirectMacroBatch",
        alias = "indirect",
        alias = "macro-batch",
        alias = "indirect-macro-batch",
        alias = "indirect-batch"
    )]
    IndirectMacroBatch,

    #[value(
        name = "CpuReference",
        alias = "cpu",
        alias = "cpu-ref",
        alias = "cpu-reference",
        alias = "rayon"
    )]
    CpuReference,

    #[value(
        name = "CpuSingleThread",
        alias = "cpu-single",
        alias = "cpu-seq",
        alias = "single-thread",
        alias = "fifo"
    )]
    CpuSingleThread,

    #[value(name = "All", alias = "all")]
    All,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
pub enum OutputFormat {
    #[default]
    #[value(name = "table", alias = "text")]
    Table,
    #[value(name = "json")]
    Json,
    #[value(name = "csv")]
    Csv,
    #[value(name = "markdown", alias = "md")]
    Markdown,
}

#[derive(Args, Debug, Clone)]
pub struct DownloadDatasetArgs {
    /// Canonical dataset name (e.g., luxembourg_osm, ca-GrQc, delaunay_n16)
    #[arg(long, short = 'n', value_name = "NAME")]
    pub name: String,

    /// SuiteSparse matrix group name (e.g., DIMACS10, SNAP) [optional if in catalog]
    #[arg(long, short = 'g', value_name = "GROUP")]
    pub group: Option<String>,

    /// Direct archive download URL [optional, overrides name/group]
    #[arg(long, value_name = "URL")]
    pub url: Option<String>,

    /// Destination root directory [default: data]
    #[arg(long, short = 'd', value_name = "DIR", default_value = "data")]
    pub dir: PathBuf,

    /// Force re-download even if dataset file already exists
    #[arg(long, short = 'f')]
    pub force: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_parse_flags_mode() {
        let args = vec![
            "webgpu-bfs",
            "--grid",
            "20x20",
            "--num-sources",
            "16",
            "--method",
            "all",
            "--verify",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.run_args.graph.grid.as_deref(), Some("20x20"));
        assert_eq!(cli.run_args.sources.num_sources, Some(16));
        assert_eq!(cli.run_args.method, MethodArg::All);
        assert!(cli.run_args.verify);
    }

    #[test]
    fn test_parse_bench_subcommand() {
        let args = vec![
            "webgpu-bfs",
            "bench",
            "--er-vertices",
            "500",
            "--er-p",
            "0.02",
            "--num-sources",
            "32",
            "--method",
            "BitParallel64",
            "--format",
            "table",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        match cli.command {
            Some(Commands::Bench(bench_args)) => {
                assert_eq!(bench_args.graph.er_vertices, Some(500));
                assert_eq!(bench_args.graph.er_p, Some(0.02));
                assert_eq!(bench_args.sources.num_sources, Some(32));
                assert_eq!(bench_args.method, MethodArg::BitParallel64);
                assert_eq!(bench_args.format, OutputFormat::Table);
            }
            other => panic!("Expected Bench command, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_download_subcommand() {
        let args = vec![
            "webgpu-bfs",
            "download-dataset",
            "--name",
            "luxembourg_osm",
            "--dir",
            "data",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        match cli.command {
            Some(Commands::DownloadDataset(d_args)) => {
                assert_eq!(d_args.name, "luxembourg_osm");
                assert_eq!(d_args.dir, PathBuf::from("data"));
            }
            other => panic!("Expected DownloadDataset command, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_indirect_macro_batch_method() {
        let args = vec![
            "webgpu-bfs",
            "--grid",
            "20x20",
            "--num-sources",
            "64",
            "--method",
            "indirect-macro-batch",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.run_args.method, MethodArg::IndirectMacroBatch);
    }
}
