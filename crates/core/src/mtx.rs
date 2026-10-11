use crate::error::MtxError;
use crate::graph::CsrGraph;
use std::io::BufRead;

/// Format specification in Matrix Market header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtxFormat {
    Coordinate,
    Array,
}

/// Field / data type in Matrix Market header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtxField {
    Pattern,
    Real,
    Integer,
    Complex,
}

/// Symmetry structure in Matrix Market header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtxSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

impl MtxSymmetry {
    /// Returns true if this symmetry structure implies that non-diagonal entries are symmetric.
    pub fn is_symmetric(&self) -> bool {
        matches!(
            self,
            MtxSymmetry::Symmetric | MtxSymmetry::SkewSymmetric | MtxSymmetry::Hermitian
        )
    }
}

/// Parsed Matrix Market header banner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtxHeader {
    pub format: MtxFormat,
    pub field: MtxField,
    pub symmetry: MtxSymmetry,
}

impl MtxHeader {
    /// Parses the header banner line: `%%MatrixMarket matrix <format> <field> <symmetry>`.
    pub fn parse_with_line(line: &str, line_number: usize) -> Result<Self, MtxError> {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() != 5 {
            return Err(MtxError::InvalidHeader {
                line_number,
                reason: format!(
                    "Expected 5 tokens in header line, found {}: '{}'",
                    tokens.len(),
                    line
                ),
            });
        }
        if !tokens[0].eq_ignore_ascii_case("%%MatrixMarket") {
            return Err(MtxError::InvalidHeader {
                line_number,
                reason: format!("Missing '%%MatrixMarket' banner: '{}'", tokens[0]),
            });
        }
        if !tokens[1].eq_ignore_ascii_case("matrix") {
            return Err(MtxError::InvalidHeader {
                line_number,
                reason: format!(
                    "Unsupported object '{}', only 'matrix' is supported",
                    tokens[1]
                ),
            });
        }
        let format = match tokens[2].to_ascii_lowercase().as_str() {
            "coordinate" => MtxFormat::Coordinate,
            "array" => return Err(MtxError::UnsupportedFormat("array".to_string())),
            other => {
                return Err(MtxError::InvalidHeader {
                    line_number,
                    reason: format!("Unknown format '{}'", other),
                })
            }
        };
        let field = match tokens[3].to_ascii_lowercase().as_str() {
            "pattern" => MtxField::Pattern,
            "real" => MtxField::Real,
            "integer" => MtxField::Integer,
            "complex" => MtxField::Complex,
            other => {
                return Err(MtxError::InvalidHeader {
                    line_number,
                    reason: format!("Unknown field '{}'", other),
                })
            }
        };
        let symmetry = match tokens[4].to_ascii_lowercase().as_str() {
            "general" => MtxSymmetry::General,
            "symmetric" => MtxSymmetry::Symmetric,
            "skew-symmetric" => MtxSymmetry::SkewSymmetric,
            "hermitian" => MtxSymmetry::Hermitian,
            other => {
                return Err(MtxError::InvalidHeader {
                    line_number,
                    reason: format!("Unknown symmetry '{}'", other),
                })
            }
        };
        Ok(Self {
            format,
            field,
            symmetry,
        })
    }
}

/// Streaming reader for Matrix Market format.
pub struct MtxReader;

impl MtxReader {
    /// Streams and parses an unweighted graph in Matrix Market (.mtx) format from any BufRead.
    pub fn parse<R: BufRead>(reader: R) -> Result<CsrGraph, MtxError> {
        let mut lines = reader.lines();
        let mut line_number = 0usize;

        // 1. Read header line
        let header = loop {
            line_number += 1;
            let line = match lines.next() {
                Some(res) => res?,
                None => {
                    return Err(MtxError::InvalidHeader {
                        line_number,
                        reason: "Empty input: no MatrixMarket header found".to_string(),
                    })
                }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            break MtxHeader::parse_with_line(trimmed, line_number)?;
        };

        if header.format != MtxFormat::Coordinate {
            return Err(MtxError::UnsupportedFormat("array".to_string()));
        }

        // 2. Read dimensions line (skipping comments)
        let (rows, cols, nnz) = loop {
            line_number += 1;
            let line = match lines.next() {
                Some(res) => res?,
                None => {
                    return Err(MtxError::InvalidDimensions {
                        line_number,
                        reason: "Unexpected end of file while searching for dimension line"
                            .to_string(),
                    })
                }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('%') {
                continue;
            }

            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.len() != 3 {
                return Err(MtxError::InvalidDimensions {
                    line_number,
                    reason: format!(
                        "Expected 3 integer tokens (rows cols nnz), got {}: '{}'",
                        tokens.len(),
                        trimmed
                    ),
                });
            }

            let r: usize = tokens[0].parse().map_err(|e| MtxError::InvalidDimensions {
                line_number,
                reason: format!("Failed to parse rows count '{}': {}", tokens[0], e),
            })?;
            let c: usize = tokens[1].parse().map_err(|e| MtxError::InvalidDimensions {
                line_number,
                reason: format!("Failed to parse cols count '{}': {}", tokens[1], e),
            })?;
            let nz: usize = tokens[2].parse().map_err(|e| MtxError::InvalidDimensions {
                line_number,
                reason: format!("Failed to parse nnz count '{}': {}", tokens[2], e),
            })?;

            break (r, c, nz);
        };

        if header.symmetry.is_symmetric() && rows != cols {
            return Err(MtxError::InvalidDimensions {
                line_number,
                reason: format!(
                    "Symmetric matrix must be square, but got {} rows and {} cols",
                    rows, cols
                ),
            });
        }

        let num_vertices = rows.max(cols);
        let is_symmetric = header.symmetry.is_symmetric();

        // 3. Stream coordinate lines
        let estimated_capacity = if is_symmetric {
            nnz.saturating_mul(2)
        } else {
            nnz
        };
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(estimated_capacity);

        for line_res in lines {
            line_number += 1;
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('%') {
                continue;
            }

            let mut parts = trimmed.split_whitespace();
            let row_str = match parts.next() {
                Some(s) => s,
                None => continue,
            };
            let col_str = match parts.next() {
                Some(s) => s,
                None => {
                    return Err(MtxError::InvalidCoordinateLine {
                        line_number,
                        line: trimmed.to_string(),
                    })
                }
            };

            let row: usize = row_str
                .parse()
                .map_err(|_| MtxError::InvalidCoordinateLine {
                    line_number,
                    line: trimmed.to_string(),
                })?;
            let col: usize = col_str
                .parse()
                .map_err(|_| MtxError::InvalidCoordinateLine {
                    line_number,
                    line: trimmed.to_string(),
                })?;

            if row == 0 || col == 0 {
                return Err(MtxError::ZeroIndex {
                    line_number,
                    row,
                    col,
                });
            }
            if row > rows || col > cols {
                return Err(MtxError::IndexOutOfBounds {
                    line_number,
                    row,
                    col,
                    max_rows: rows,
                    max_cols: cols,
                });
            }

            let u = (row - 1) as u32;
            let v = (col - 1) as u32;

            // Eliminate self-loops
            if u == v {
                continue;
            }

            edges.push((u, v));
            if is_symmetric {
                edges.push((v, u));
            }
        }

        // 4. Canonical CSR construction
        Ok(CsrGraph::from_edge_list(num_vertices, &edges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_mtx_header_parsing_variants() {
        // Standard coordinate pattern symmetric
        let h1 =
            MtxHeader::parse_with_line("%%MatrixMarket matrix coordinate pattern symmetric", 1)
                .unwrap();
        assert_eq!(h1.format, MtxFormat::Coordinate);
        assert_eq!(h1.field, MtxField::Pattern);
        assert_eq!(h1.symmetry, MtxSymmetry::Symmetric);
        assert!(h1.symmetry.is_symmetric());

        // Real general case insensitive
        let h2 =
            MtxHeader::parse_with_line("%%matrixmarket MATRIX COORDINATE real general", 1).unwrap();
        assert_eq!(h2.field, MtxField::Real);
        assert_eq!(h2.symmetry, MtxSymmetry::General);
        assert!(!h2.symmetry.is_symmetric());

        // Reject array format
        let err =
            MtxHeader::parse_with_line("%%MatrixMarket matrix array real general", 1).unwrap_err();
        assert!(matches!(err, MtxError::UnsupportedFormat(_)));
    }

    #[test]
    fn test_mtx_streaming_parser_symmetric_pattern() {
        let mtx_data = "\
%%MatrixMarket matrix coordinate pattern symmetric
% Test comments
% Another comment line
4 4 3
1 2
2 3
3 4
";
        let reader = Cursor::new(mtx_data);
        let graph = MtxReader::parse(reader).unwrap();

        assert_eq!(graph.num_vertices, 4);
        // 3 symmetric edges -> 6 directed edges
        assert_eq!(graph.num_edges, 6);
        assert_eq!(graph.neighbors(0), &[1]);
        assert_eq!(graph.neighbors(1), &[0, 2]);
        assert_eq!(graph.neighbors(2), &[1, 3]);
        assert_eq!(graph.neighbors(3), &[2]);
        assert!(graph.is_symmetric());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_mtx_self_loops_and_deduplication() {
        let mtx_data = "\
%%MatrixMarket matrix coordinate real general
3 3 5
1 1 1.0
1 2 2.5
1 2 3.0
2 3 1.0
3 3 0.5
";
        let reader = Cursor::new(mtx_data);
        let graph = MtxReader::parse(reader).unwrap();

        assert_eq!(graph.num_vertices, 3);
        // Self loops (1,1) and (3,3) eliminated; duplicate (1,2) deduped
        assert_eq!(graph.num_edges, 2);
        assert_eq!(graph.neighbors(0), &[1]);
        assert_eq!(graph.neighbors(1), &[2]);
        assert_eq!(graph.neighbors(2), &[]);
    }

    #[test]
    fn test_mtx_error_zero_index() {
        let mtx_data = "\
%%MatrixMarket matrix coordinate pattern general
3 3 1
0 1
";
        let reader = Cursor::new(mtx_data);
        let err = MtxReader::parse(reader).unwrap_err();
        assert!(matches!(
            err,
            MtxError::ZeroIndex {
                line_number: 3,
                row: 0,
                col: 1
            }
        ));
    }

    #[test]
    fn test_mtx_error_out_of_bounds() {
        let mtx_data = "\
%%MatrixMarket matrix coordinate pattern general
3 3 1
4 1
";
        let reader = Cursor::new(mtx_data);
        let err = MtxReader::parse(reader).unwrap_err();
        assert!(matches!(
            err,
            MtxError::IndexOutOfBounds {
                line_number: 3,
                row: 4,
                col: 1,
                max_rows: 3,
                max_cols: 3
            }
        ));
    }

    #[test]
    fn test_mtx_luxembourg_osm_mock() {
        let mtx_data = "\
%%MatrixMarket matrix coordinate pattern symmetric
% Luxembourg road segment mock
6 6 5
1 2
2 3
3 4
4 5
5 6
";
        let reader = Cursor::new(mtx_data);
        let graph = MtxReader::parse(reader).unwrap();

        assert_eq!(graph.num_vertices, 6);
        assert_eq!(graph.num_edges, 10);
        assert_eq!(graph.max_degree(), 2);
        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(1), 2);
        assert!(graph.is_symmetric());
    }
}
