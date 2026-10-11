use crate::error::GeneratorError;
use crate::graph::CsrGraph;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

/// Helper function to instantiate a deterministic or entropy-seeded PRNG.
pub(crate) fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// 2D 4-connected grid graph with analytical Manhattan distance ground-truth oracle.
#[derive(Debug, Clone)]
pub struct Grid2D {
    pub width: usize,
    pub height: usize,
    pub graph: CsrGraph,
}

impl Grid2D {
    /// Generates a 4-connected 2D grid graph and builds its `CsrGraph` directly.
    /// Neighbors are emitted in order `[North, West, East, South]`, guaranteeing
    /// strictly ascending neighbor indices without intermediate sorting.
    pub fn new(width: usize, height: usize) -> Result<Self, GeneratorError> {
        if width == 0 || height == 0 {
            return Err(GeneratorError::InvalidGridDimensions { width, height });
        }

        let num_vertices = width * height;
        // Total directed edges: 2 * ((W-1)*H + W*(H-1))
        let num_edges =
            2 * ((width.saturating_sub(1) * height) + (width * height.saturating_sub(1)));

        let mut row_offsets = Vec::with_capacity(num_vertices + 1);
        let mut col_indices = Vec::with_capacity(num_edges);

        row_offsets.push(0);

        for y in 0..height {
            for x in 0..width {
                let v = y * width + x;

                // 1. North neighbor (y - 1)
                if y > 0 {
                    col_indices.push((v - width) as u32);
                }
                // 2. West neighbor (x - 1)
                if x > 0 {
                    col_indices.push((v - 1) as u32);
                }
                // 3. East neighbor (x + 1)
                if x + 1 < width {
                    col_indices.push((v + 1) as u32);
                }
                // 4. South neighbor (y + 1)
                if y + 1 < height {
                    col_indices.push((v + width) as u32);
                }

                row_offsets.push(col_indices.len() as u32);
            }
        }

        let graph = CsrGraph {
            num_vertices,
            num_edges,
            row_offsets,
            col_indices,
        };

        Ok(Self {
            width,
            height,
            graph,
        })
    }

    /// Converts (x, y) coordinates to linear 1D vertex ID.
    #[inline(always)]
    pub fn vertex_id(&self, x: usize, y: usize) -> u32 {
        (y * self.width + x) as u32
    }

    /// Converts linear 1D vertex ID to (x, y) coordinates.
    #[inline(always)]
    pub fn coordinates(&self, vertex: u32) -> (usize, usize) {
        let v = vertex as usize;
        (v % self.width, v / self.width)
    }

    /// Analytical Manhattan distance ground-truth oracle for any pair of vertices:
    /// `|x1 - x2| + |y1 - y2|`.
    #[inline(always)]
    pub fn manhattan_distance(&self, v1: u32, v2: u32) -> u32 {
        let (x1, y1) = self.coordinates(v1);
        let (x2, y2) = self.coordinates(v2);
        let dx = (x1 as isize - x2 as isize).unsigned_abs() as u32;
        let dy = (y1 as isize - y2 as isize).unsigned_abs() as u32;
        dx + dy
    }

    /// Generates exact ground-truth distance vector of length |V| for single source `s`.
    pub fn single_source_oracle(&self, source: u32) -> Vec<u32> {
        let num_vertices = self.width * self.height;
        let mut dist = Vec::with_capacity(num_vertices);
        for v in 0..num_vertices {
            dist.push(self.manhattan_distance(source, v as u32));
        }
        dist
    }

    /// Generates exact ground-truth distance matrix for multiple sources `sources`.
    /// Memory layout matches the engine contract: `dist[s_idx * |V| + v]`.
    pub fn ms_bfs_oracle(&self, sources: &[u32]) -> Vec<u32> {
        let num_vertices = self.width * self.height;
        let mut dist = Vec::with_capacity(sources.len() * num_vertices);
        for &s in sources {
            for v in 0..num_vertices {
                dist.push(self.manhattan_distance(s, v as u32));
            }
        }
        dist
    }
}

/// Convenience function to generate a 2D grid `CsrGraph`.
pub fn generate_grid2d(width: usize, height: usize) -> CsrGraph {
    Grid2D::new(width, height)
        .expect("Valid grid dimensions required")
        .graph
}

/// Generates G(n, p) using the Batagelj-Brandes geometric skipping algorithm
/// for fast, scalable generation in O(N + M) time.
pub fn generate_erdos_renyi_np(
    num_vertices: usize,
    p: f64,
    seed: Option<u64>,
) -> Result<CsrGraph, GeneratorError> {
    if num_vertices == 0 {
        return Err(GeneratorError::ZeroVertices(0));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(GeneratorError::InvalidProbability(p.to_string()));
    }
    if num_vertices == 1 || p == 0.0 {
        return Ok(isolated_vertices(num_vertices));
    }

    let mut rng = make_rng(seed);
    let mut raw_edges = Vec::new();

    if p >= 0.5 {
        for u in 0..(num_vertices - 1) {
            for v in (u + 1)..num_vertices {
                if rng.gen_bool(p) {
                    raw_edges.push((u as u32, v as u32));
                }
            }
        }
    } else {
        // Batagelj-Brandes geometric skipping
        let ln_1_minus_p = (1.0 - p).ln();
        let mut u = 0usize;
        let mut v = 1usize;

        while u < num_vertices - 1 {
            let rand_val: f64 = rng.gen();
            // Skip distance k
            let k = (rand_val.ln() / ln_1_minus_p).floor() as usize;
            v += k;

            while v >= num_vertices && u < num_vertices - 1 {
                v -= num_vertices - 1 - u;
                u += 1;
                v += u;
            }

            if u < num_vertices - 1 {
                raw_edges.push((u as u32, v as u32));
                v += 1;
            }
        }
    }

    Ok(csr_from_undirected_pairs(num_vertices, &raw_edges))
}

/// Generates G(n, m) with exactly M undirected edges.
pub fn generate_erdos_renyi_nm(
    num_vertices: usize,
    m: usize,
    seed: Option<u64>,
) -> Result<CsrGraph, GeneratorError> {
    if num_vertices == 0 {
        return Err(GeneratorError::ZeroVertices(0));
    }
    let max_possible = num_vertices * (num_vertices - 1) / 2;
    if m > max_possible {
        return Err(GeneratorError::ExcessiveEdgeCount {
            requested: m,
            max_possible,
            vertices: num_vertices,
        });
    }
    if m == 0 {
        return Ok(isolated_vertices(num_vertices));
    }

    let mut rng = make_rng(seed);
    let mut edges = HashSet::with_capacity(m);

    while edges.len() < m {
        let u = rng.gen_range(0..num_vertices as u32);
        let v = rng.gen_range(0..num_vertices as u32);
        if u != v {
            let (min, max) = if u < v { (u, v) } else { (v, u) };
            edges.insert((min, max));
        }
    }

    let raw_edges: Vec<(u32, u32)> = edges.into_iter().collect();
    Ok(csr_from_undirected_pairs(num_vertices, &raw_edges))
}

/// Convenience function to generate an Erdős-Rényi random graph.
pub fn generate_erdos_renyi(num_vertices: usize, p: f64, seed: u64) -> CsrGraph {
    generate_erdos_renyi_np(num_vertices, p, Some(seed))
        .expect("Valid parameters for erdos-renyi graph")
}

/// Generates a connected random tree on N vertices with exactly N - 1 undirected edges
/// (2N - 2 directed CSR edges for N >= 2) using a Uniform Recursive Tree (URT) algorithm
/// with Fisher-Yates vertex permutation.
pub fn generate_random_tree(
    num_vertices: usize,
    seed: Option<u64>,
) -> Result<CsrGraph, GeneratorError> {
    if num_vertices == 0 {
        return Err(GeneratorError::ZeroVertices(0));
    }
    if num_vertices == 1 {
        return Ok(single_vertex());
    }

    let mut rng = make_rng(seed);

    let mut perm: Vec<u32> = (0..num_vertices as u32).collect();
    perm.shuffle(&mut rng);

    let mut raw_edges = Vec::with_capacity(num_vertices - 1);
    for i in 1..num_vertices {
        let parent_idx = rng.gen_range(0..i);
        let u = perm[i];
        let v = perm[parent_idx];
        raw_edges.push((u, v));
    }

    Ok(csr_from_undirected_pairs(num_vertices, &raw_edges))
}

/// Standard Graph500 R-MAT configuration parameters.
pub const GRAPH500_A: f64 = 0.57;
pub const GRAPH500_B: f64 = 0.19;
pub const GRAPH500_C: f64 = 0.19;
pub const GRAPH500_D: f64 = 0.05;

/// Generates an R-MAT power-law scale-free graph.
/// - `scale`: defines vertex count `N = 2^scale`.
/// - `edge_factor`: average degree multiplier; requested edges `M = edge_factor * N`.
/// - `directed`: if false, edges are symmetrized.
#[allow(clippy::too_many_arguments)]
pub fn generate_rmat(
    scale: usize,
    edge_factor: usize,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    seed: Option<u64>,
    directed: bool,
) -> Result<CsrGraph, GeneratorError> {
    if scale == 0 {
        return Err(GeneratorError::ZeroVertices(0));
    }
    if scale > 30 {
        return Err(GeneratorError::ScaleTooLarge(scale));
    }
    let sum = a + b + c + d;
    if (sum - 1.0).abs() > 1e-6 {
        return Err(GeneratorError::InvalidRmatProbabilities { a, b, c, d, sum });
    }

    let num_vertices = 1usize << scale;
    let target_edges = edge_factor * num_vertices;

    let mut rng = make_rng(seed);
    let mut raw_edges = Vec::with_capacity(target_edges);

    let t_a = a;
    let t_b = a + b;
    let t_c = a + b + c;

    for _ in 0..target_edges {
        let mut u = 0u32;
        let mut v = 0u32;

        for step in (0..scale).rev() {
            let bit = 1u32 << step;
            let r: f64 = rng.gen();

            if r < t_a {
                // Top-Left: (0, 0)
            } else if r < t_b {
                // Top-Right: (0, 1)
                v |= bit;
            } else if r < t_c {
                // Bottom-Left: (1, 0)
                u |= bit;
            } else {
                // Bottom-Right: (1, 1)
                u |= bit;
                v |= bit;
            }
        }

        // Exclude self-loops
        if u != v {
            raw_edges.push((u, v));
        }
    }

    if directed {
        Ok(csr_from_directed_pairs(num_vertices, &raw_edges))
    } else {
        Ok(csr_from_undirected_pairs(num_vertices, &raw_edges))
    }
}

/// Convenience function using standard Graph500 constants (a=0.57, b=0.19, c=0.19, d=0.05).
pub fn generate_graph500_rmat(
    scale: usize,
    edge_factor: usize,
    seed: Option<u64>,
) -> Result<CsrGraph, GeneratorError> {
    generate_rmat(
        scale,
        edge_factor,
        GRAPH500_A,
        GRAPH500_B,
        GRAPH500_C,
        GRAPH500_D,
        seed,
        false, // undirected for standard BFS
    )
}

// ==========================================
// Edge Case Topologies Factory
// ==========================================

/// Empty graph with 0 vertices and 0 edges.
pub fn empty_graph() -> CsrGraph {
    CsrGraph {
        num_vertices: 0,
        num_edges: 0,
        row_offsets: vec![0],
        col_indices: Vec::new(),
    }
}

/// Graph with N isolated vertices and 0 edges.
pub fn isolated_vertices(n: usize) -> CsrGraph {
    CsrGraph {
        num_vertices: n,
        num_edges: 0,
        row_offsets: vec![0; n + 1],
        col_indices: Vec::new(),
    }
}

/// Single vertex graph with 0 edges.
pub fn single_vertex() -> CsrGraph {
    isolated_vertices(1)
}

/// Linear chain graph 0 - 1 - 2 - ... - (N - 1) with diameter N - 1.
pub fn path_graph(n: usize) -> CsrGraph {
    if n <= 1 {
        return isolated_vertices(n);
    }

    let num_edges = 2 * (n - 1);
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(num_edges);

    row_offsets.push(0);
    for u in 0..n {
        if u > 0 {
            col_indices.push((u - 1) as u32);
        }
        if u + 1 < n {
            col_indices.push((u + 1) as u32);
        }
        row_offsets.push(col_indices.len() as u32);
    }

    CsrGraph {
        num_vertices: n,
        num_edges,
        row_offsets,
        col_indices,
    }
}

/// Star graph with central hub 0 connected to leaves 1 .. N - 1.
pub fn star_graph(n: usize) -> CsrGraph {
    if n <= 1 {
        return isolated_vertices(n);
    }

    let num_edges = 2 * (n - 1);
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(num_edges);

    row_offsets.push(0);
    // Hub 0 connects to 1 .. n - 1
    for leaf in 1..n {
        col_indices.push(leaf as u32);
    }
    row_offsets.push(col_indices.len() as u32);

    // Leaves 1 .. n - 1 connect back to hub 0
    for _ in 1..n {
        col_indices.push(0);
        row_offsets.push(col_indices.len() as u32);
    }

    CsrGraph {
        num_vertices: n,
        num_edges,
        row_offsets,
        col_indices,
    }
}

/// Simple cycle graph 0 - 1 - 2 - ... - (N - 1) - 0.
pub fn cycle_graph(n: usize) -> CsrGraph {
    if n <= 2 {
        return path_graph(n);
    }

    let num_edges = 2 * n;
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(num_edges);

    row_offsets.push(0);
    for u in 0..n {
        let prev = if u == 0 { n - 1 } else { u - 1 };
        let next = if u + 1 == n { 0 } else { u + 1 };
        // Ascending sort: min(prev, next), max(prev, next)
        let (first, second) = if prev < next {
            (prev, next)
        } else {
            (next, prev)
        };
        col_indices.push(first as u32);
        col_indices.push(second as u32);
        row_offsets.push(col_indices.len() as u32);
    }

    CsrGraph {
        num_vertices: n,
        num_edges,
        row_offsets,
        col_indices,
    }
}

/// Complete graph K_n where every pair of vertices is connected.
pub fn clique_graph(n: usize) -> CsrGraph {
    if n <= 1 {
        return isolated_vertices(n);
    }

    let num_edges = n * (n - 1);
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(num_edges);

    row_offsets.push(0);
    for u in 0..n {
        for v in 0..n {
            if u != v {
                col_indices.push(v as u32);
            }
        }
        row_offsets.push(col_indices.len() as u32);
    }

    CsrGraph {
        num_vertices: n,
        num_edges,
        row_offsets,
        col_indices,
    }
}

/// Disconnected graph containing multiple isolated cliques of specified sizes.
pub fn disconnected_cliques(sizes: &[usize]) -> CsrGraph {
    let total_vertices: usize = sizes.iter().sum();
    if total_vertices == 0 {
        return empty_graph();
    }

    let mut row_offsets = Vec::with_capacity(total_vertices + 1);
    let mut col_indices = Vec::new();

    row_offsets.push(0);
    let mut offset = 0usize;

    for &size in sizes {
        for u in 0..size {
            for v in 0..size {
                if u != v {
                    col_indices.push((offset + v) as u32);
                }
            }
            row_offsets.push(col_indices.len() as u32);
        }
        offset += size;
    }

    let num_edges = col_indices.len();
    CsrGraph {
        num_vertices: total_vertices,
        num_edges,
        row_offsets,
        col_indices,
    }
}

/// Builds a clean, deduplicated, sorted symmetric CsrGraph from undirected pairs (u, v).
pub fn csr_from_undirected_pairs(num_vertices: usize, pairs: &[(u32, u32)]) -> CsrGraph {
    let mut directed_pairs = Vec::with_capacity(pairs.len() * 2);
    for &(u, v) in pairs {
        if u != v && (u as usize) < num_vertices && (v as usize) < num_vertices {
            directed_pairs.push((u, v));
            directed_pairs.push((v, u));
        }
    }
    csr_from_directed_pairs(num_vertices, &directed_pairs)
}

/// Builds a clean, deduplicated, sorted CsrGraph from directed pairs (u, v).
pub fn csr_from_directed_pairs(num_vertices: usize, pairs: &[(u32, u32)]) -> CsrGraph {
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); num_vertices];

    for &(u, v) in pairs {
        let u_idx = u as usize;
        let v_idx = v as usize;
        if u_idx < num_vertices && v_idx < num_vertices && u_idx != v_idx {
            adjacency[u_idx].push(v);
        }
    }

    let mut row_offsets = Vec::with_capacity(num_vertices + 1);
    let mut col_indices = Vec::new();
    row_offsets.push(0);

    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
        col_indices.extend_from_slice(neighbors);
        row_offsets.push(col_indices.len() as u32);
    }

    let num_edges = col_indices.len();
    CsrGraph {
        num_vertices,
        num_edges,
        row_offsets,
        col_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid2d_dimensions_and_edge_counts() {
        let grid = Grid2D::new(4, 3).expect("4x3 grid should succeed");
        assert_eq!(grid.width, 4);
        assert_eq!(grid.height, 3);
        assert_eq!(grid.graph.num_vertices, 12);

        // Undirected edges: (4-1)*3 + 4*(3-1) = 9 + 8 = 17
        // Directed CSR edges: 17 * 2 = 34
        assert_eq!(grid.graph.num_edges, 34);
        assert_eq!(grid.graph.row_offsets.len(), 13);
        assert_eq!(grid.graph.col_indices.len(), 34);
        assert_eq!(grid.graph.row_offsets[12], 34);
        assert!(grid.graph.validate().is_ok());
    }

    #[test]
    fn test_grid2d_neighbors_sorted_and_unique() {
        let grid = Grid2D::new(5, 5).unwrap();
        for u in 0..grid.graph.num_vertices {
            let start = grid.graph.row_offsets[u] as usize;
            let end = grid.graph.row_offsets[u + 1] as usize;
            let neighbors = &grid.graph.col_indices[start..end];

            // Verify strict ascending order
            for window in neighbors.windows(2) {
                assert!(window[0] < window[1], "Neighbors must be strictly sorted");
            }
        }
    }

    #[test]
    fn test_grid2d_manhattan_distance_oracle() {
        let grid = Grid2D::new(10, 10).unwrap();

        // Corner to corner
        let v_0_0 = grid.vertex_id(0, 0);
        let v_9_9 = grid.vertex_id(9, 9);
        assert_eq!(grid.manhattan_distance(v_0_0, v_9_9), 18);

        // Center to edge
        let v_center = grid.vertex_id(5, 5);
        let v_edge = grid.vertex_id(5, 0);
        assert_eq!(grid.manhattan_distance(v_center, v_edge), 5);

        // Distance vector oracle check
        let dist = grid.single_source_oracle(v_0_0);
        assert_eq!(dist.len(), 100);
        assert_eq!(dist[v_0_0 as usize], 0);
        assert_eq!(dist[v_9_9 as usize], 18);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_grid2d_multi_source_oracle_layout() {
        let grid = Grid2D::new(3, 3).unwrap();
        let sources = vec![0, 4, 8];
        let ms_dist = grid.ms_bfs_oracle(&sources);

        assert_eq!(ms_dist.len(), 3 * 9);
        let s0 = 0usize;
        let s1 = 1usize;
        let s2 = 2usize;
        // Source 0 (top-left) to node 8 (bottom-right)
        assert_eq!(ms_dist[s0 * 9 + 8], 4);
        // Source 4 (center) to node 0 (top-left)
        assert_eq!(ms_dist[s1 * 9 + 0], 2);
        // Source 8 (bottom-right) to node 8 (itself)
        assert_eq!(ms_dist[s2 * 9 + 8], 0);
    }

    #[test]
    fn test_erdos_renyi_np_properties() {
        let n = 50;
        let p = 0.2;
        let seed = Some(12345);
        let graph = generate_erdos_renyi_np(n, p, seed).unwrap();

        assert_eq!(graph.num_vertices, n);
        assert_eq!(graph.num_edges % 2, 0);
        assert!(graph.is_symmetric());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_erdos_renyi_nm_exact_edges() {
        let n = 30;
        let m = 45;
        let graph = generate_erdos_renyi_nm(n, m, Some(999)).unwrap();

        assert_eq!(graph.num_vertices, n);
        assert_eq!(graph.num_edges, 2 * m);
        assert!(graph.is_symmetric());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_random_tree_invariants() {
        let n = 100;
        let tree = generate_random_tree(n, Some(42)).unwrap();

        assert_eq!(tree.num_vertices, n);
        assert_eq!(tree.num_edges, 2 * (n - 1));
        assert!(tree.is_symmetric());
        assert!(tree.validate().is_ok());

        // Connectivity check via DFS
        let mut visited = vec![false; n];
        let mut stack = vec![0usize];
        visited[0] = true;
        let mut visited_count = 1;

        while let Some(u) = stack.pop() {
            let start = tree.row_offsets[u] as usize;
            let end = tree.row_offsets[u + 1] as usize;
            for &v in &tree.col_indices[start..end] {
                let v_idx = v as usize;
                if !visited[v_idx] {
                    visited[v_idx] = true;
                    visited_count += 1;
                    stack.push(v_idx);
                }
            }
        }

        assert_eq!(visited_count, n, "Random tree must be 100% connected");
    }

    #[test]
    fn test_rmat_scale_free_degree_distribution() {
        let scale = 6; // N = 64
        let edge_factor = 16;
        let graph = generate_graph500_rmat(scale, edge_factor, Some(777)).unwrap();

        assert_eq!(graph.num_vertices, 64);
        assert!(graph.num_edges > 0);
        assert!(graph.validate().is_ok());

        let hub_degree = graph.degree(0);
        let tail_degree = graph.degree(63);

        assert!(
            hub_degree > tail_degree,
            "R-MAT must exhibit power-law hub effect (hub deg {} > tail deg {})",
            hub_degree,
            tail_degree
        );
    }

    #[test]
    fn test_edge_case_topologies() {
        // 1. Single vertex
        let single = single_vertex();
        assert_eq!(single.num_vertices, 1);
        assert_eq!(single.num_edges, 0);
        assert_eq!(single.row_offsets, vec![0, 0]);
        assert!(single.validate().is_ok());

        // 2. Isolated vertices
        let iso = isolated_vertices(10);
        assert_eq!(iso.num_vertices, 10);
        assert_eq!(iso.num_edges, 0);
        assert!(iso.validate().is_ok());

        // 3. Path graph (Chain)
        let path = path_graph(5);
        assert_eq!(path.num_vertices, 5);
        assert_eq!(path.num_edges, 8); // 2 * 4
        assert!(path.validate().is_ok());

        // 4. Star graph
        let star = star_graph(6);
        assert_eq!(star.num_vertices, 6);
        assert_eq!(star.num_edges, 10); // 2 * 5
        assert_eq!(star.degree(0), 5);
        assert!(star.validate().is_ok());

        // 5. Clique graph
        let clique = clique_graph(4);
        assert_eq!(clique.num_vertices, 4);
        assert_eq!(clique.num_edges, 12); // 4 * 3
        assert!(clique.validate().is_ok());

        // 6. Disconnected cliques
        let dis = disconnected_cliques(&[3, 3]);
        assert_eq!(dis.num_vertices, 6);
        assert_eq!(dis.num_edges, 12);
        assert!(!dis.neighbors(0).contains(&3));
        assert!(dis.validate().is_ok());
    }

    #[test]
    fn test_deduplication_and_self_loop_filtering() {
        let raw_pairs = vec![
            (0, 1),
            (0, 1), // duplicate
            (1, 0), // reciprocal
            (0, 0), // self-loop
            (2, 2), // self-loop
            (2, 1),
            (2, 1), // duplicate
        ];

        let graph = csr_from_directed_pairs(3, &raw_pairs);
        assert_eq!(graph.num_vertices, 3);
        assert_eq!(graph.num_edges, 3);
        assert_eq!(graph.neighbors(0), &[1]);
        assert_eq!(graph.neighbors(1), &[0]);
        assert_eq!(graph.neighbors(2), &[1]);
        assert!(graph.validate().is_ok());
    }
}
