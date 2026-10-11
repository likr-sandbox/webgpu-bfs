use crate::{bit_parallel::ms_bfs_bit_parallel, fifo::ms_bfs_fifo, rayon_ms::ms_bfs_rayon};
use webgpu_bfs_core::{BfsEngine, CsrGraph};

/// Algorithmic strategies available for CPU BFS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CpuBfsAlgorithm {
    /// Multi-threaded Rayon batched MS-BFS (scales across all CPU cores).
    #[default]
    RayonBatched,

    /// Cache-friendly single-source sequential FIFO queue BFS with flat Vec cursor.
    SequentialFifo,

    /// 64-bit bit-parallel MS-BFS using u64 bitmasks and hardware bit-scan intrinsics.
    BitParallel64,
}

/// CPU Breadth-First Search reference engine.
#[derive(Debug, Clone)]
pub struct CpuBfsEngine {
    algorithm: CpuBfsAlgorithm,
}

impl CpuBfsEngine {
    /// Creates a new `CpuBfsEngine` with the default strategy (RayonBatched).
    pub fn new() -> Self {
        Self {
            algorithm: CpuBfsAlgorithm::default(),
        }
    }

    /// Creates a `CpuBfsEngine` with a specific algorithmic strategy.
    pub fn with_algorithm(algorithm: CpuBfsAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Convenience constructor for Sequential FIFO BFS.
    pub fn fifo() -> Self {
        Self::with_algorithm(CpuBfsAlgorithm::SequentialFifo)
    }

    /// Convenience constructor for Rayon Batched MS-BFS.
    pub fn rayon() -> Self {
        Self::with_algorithm(CpuBfsAlgorithm::RayonBatched)
    }

    /// Convenience constructor for 64-bit Bit-Parallel MS-BFS.
    pub fn bit_parallel() -> Self {
        Self::with_algorithm(CpuBfsAlgorithm::BitParallel64)
    }

    /// Returns the currently active algorithm.
    pub fn algorithm(&self) -> CpuBfsAlgorithm {
        self.algorithm
    }
}

impl Default for CpuBfsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BfsEngine for CpuBfsEngine {
    fn run_ms_bfs(&self, graph: &CsrGraph, sources: &[u32]) -> anyhow::Result<Vec<u32>> {
        match self.algorithm {
            CpuBfsAlgorithm::SequentialFifo => ms_bfs_fifo(graph, sources),
            CpuBfsAlgorithm::RayonBatched => ms_bfs_rayon(graph, sources),
            CpuBfsAlgorithm::BitParallel64 => ms_bfs_bit_parallel(graph, sources),
        }
    }
}
