use thiserror::Error;

/// Errors arising from invalid graph topology or structure.
#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum GraphError {
    #[error("Invalid row_offsets length: expected {expected} (|V| + 1), found {actual}")]
    InvalidRowOffsetsLength { expected: usize, actual: usize },

    #[error("row_offsets must start with 0, but found {0}")]
    NonZeroStartOffset(u32),

    #[error("row_offsets must be monotonically non-decreasing: offset at {index} ({curr}) > offset at {next_index} ({next})")]
    NonMonotonicOffsets {
        index: usize,
        curr: u32,
        next_index: usize,
        next: u32,
    },

    #[error(
        "Last row offset ({last_offset}) does not match col_indices length ({col_indices_len})"
    )]
    EdgeCountMismatch {
        last_offset: u32,
        col_indices_len: usize,
    },

    #[error("Column index {col} at edge position {edge_idx} is out of bounds for graph with {num_vertices} vertices")]
    ColumnIndexOutOfBounds {
        col: u32,
        edge_idx: usize,
        num_vertices: usize,
    },

    #[error("Neighbor list for vertex {vertex} is unsorted or contains duplicate entries ({first} >= {second})")]
    UnsortedOrDuplicateNeighbors {
        vertex: usize,
        first: u32,
        second: u32,
    },
}

/// Errors encountered while reading and parsing Matrix Market files.
#[derive(Error, Debug)]
pub enum MtxError {
    #[error("I/O error during Matrix Market parsing: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid Matrix Market header at line {line_number}: {reason}")]
    InvalidHeader { line_number: usize, reason: String },

    #[error(
        "Unsupported Matrix Market format '{0}': only sparse 'coordinate' format is supported"
    )]
    UnsupportedFormat(String),

    #[error("Invalid dimensions at line {line_number}: {reason}")]
    InvalidDimensions { line_number: usize, reason: String },

    #[error("Invalid coordinate entry at line {line_number}: '{line}'")]
    InvalidCoordinateLine { line_number: usize, line: String },

    #[error("Matrix Market coordinates are 1-based, but found 0 at line {line_number}: row={row}, col={col}")]
    ZeroIndex {
        line_number: usize,
        row: usize,
        col: usize,
    },

    #[error("Coordinate out of bounds at line {line_number}: entry ({row}, {col}) exceeds matrix dimensions ({max_rows}, {max_cols})")]
    IndexOutOfBounds {
        line_number: usize,
        row: usize,
        col: usize,
        max_rows: usize,
        max_cols: usize,
    },

    #[error("Premature end of file: expected {expected_nnz} coordinate lines, but read {actual}")]
    UnexpectedEof { expected_nnz: usize, actual: usize },
}

/// Errors occurring during procedural graph generation.
#[derive(Error, Debug, PartialEq, Clone)]
pub enum GeneratorError {
    #[error("Vertex count must be at least 1, got {0}")]
    ZeroVertices(usize),

    #[error("Grid dimensions must be at least 1x1, got width={width}, height={height}")]
    InvalidGridDimensions { width: usize, height: usize },

    #[error("Edge probability p must be in the closed interval [0.0, 1.0], got {0}")]
    InvalidProbability(String),

    #[error("Requested edge count M={requested} exceeds maximum possible edges {max_possible} for N={vertices} vertices")]
    ExcessiveEdgeCount {
        requested: usize,
        max_possible: usize,
        vertices: usize,
    },

    #[error("R-MAT partition probabilities must sum to 1.0 within tolerance, got a={a}, b={b}, c={c}, d={d} (sum={sum})")]
    InvalidRmatProbabilities {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        sum: f64,
    },

    #[error("Scale {0} exceeds maximum supported scale of 30 (would exceed 2^30 vertices)")]
    ScaleTooLarge(usize),
}
