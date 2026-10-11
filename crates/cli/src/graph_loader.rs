use anyhow::{bail, Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use webgpu_bfs_core::{
    generate_erdos_renyi_nm, generate_erdos_renyi_np, generate_grid2d, generate_random_tree,
    generate_rmat, CsrGraph,
};

use crate::args::{GraphInputArgs, SourceArgs, SourceStrategy};
use crate::dataset;

pub fn load_graph_and_sources(
    graph_args: &GraphInputArgs,
    source_args: &SourceArgs,
) -> Result<(String, CsrGraph, Vec<u32>)> {
    let (name, graph) = load_graph(graph_args)?;
    let sources = sample_sources(&graph, source_args, graph_args.seed)?;
    Ok((name, graph, sources))
}

fn load_graph(args: &GraphInputArgs) -> Result<(String, CsrGraph)> {
    if let Some(ref mtx_path) = args.mtx {
        return load_mtx_file(mtx_path);
    }

    if let Some(ref dataset_name) = args.dataset {
        let path = dataset::ensure_dataset_downloaded(dataset_name, Path::new("data"), false)?;
        return load_mtx_file(&path);
    }

    if let Some(ref grid_spec) = args.grid {
        let (w, h) = parse_grid_dimensions(grid_spec)?;
        let graph = generate_grid2d(w, h);
        return Ok((format!("Grid2D_{}x{}", w, h), graph));
    }

    if let (Some(w), Some(h)) = (args.grid_width, args.grid_height) {
        let graph = generate_grid2d(w, h);
        return Ok((format!("Grid2D_{}x{}", w, h), graph));
    }

    if let Some(nv) = args.er_vertices {
        let seed = args.seed;
        let graph = if let Some(p) = args.er_p {
            generate_erdos_renyi_np(nv, p, seed)?
        } else if let Some(m) = args.er_m {
            generate_erdos_renyi_nm(nv, m, seed)?
        } else {
            // Default probability: p = (ln(n) + 4) / n
            let p = ((nv as f64).ln() + 4.0) / (nv as f64);
            let p_clamped = p.clamp(0.0, 1.0);
            generate_erdos_renyi_np(nv, p_clamped, seed)?
        };
        return Ok((format!("ErdosRenyi_N{}", nv), graph));
    }

    if let Some(nv) = args.tree_vertices {
        let graph = generate_random_tree(nv, args.seed)?;
        return Ok((format!("RandomTree_N{}", nv), graph));
    }

    if let Some(scale) = args.rmat_scale {
        let graph = generate_rmat(
            scale,
            args.rmat_edge_factor,
            args.rmat_a,
            args.rmat_b,
            args.rmat_c,
            args.rmat_d,
            args.seed,
            args.rmat_directed,
        )?;
        return Ok((
            format!("RMAT_Scale{}_Factor{}", scale, args.rmat_edge_factor),
            graph,
        ));
    }

    // Default graph: 64x64 Grid2D (|V|=4096, |E|=16128)
    log::info!("No graph specified; defaulting to 64x64 Grid2D (|V|=4096, |E|=16128)");
    let graph = generate_grid2d(64, 64);
    Ok(("Grid2D_64x64_Default".to_string(), graph))
}

fn load_mtx_file(path: &Path) -> Result<(String, CsrGraph)> {
    let file = File::open(path).with_context(|| format!("Cannot open MTX file {:?}", path))?;
    let reader = BufReader::new(file);
    let graph = CsrGraph::from_mtx_reader(reader)
        .with_context(|| format!("Failed to parse MTX file {:?}", path))?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("matrix")
        .to_string();
    Ok((stem, graph))
}

pub fn parse_grid_dimensions(spec: &str) -> Result<(usize, usize)> {
    let parts: Vec<&str> = spec.split(&['x', 'X', ','][..]).collect();
    if parts.len() != 2 {
        bail!(
            "Invalid grid dimensions format '{}'; expected WxH (e.g. 50x50)",
            spec
        );
    }
    let w: usize = parts[0]
        .trim()
        .parse()
        .context("Invalid grid width integer")?;
    let h: usize = parts[1]
        .trim()
        .parse()
        .context("Invalid grid height integer")?;
    if w == 0 || h == 0 {
        bail!("Grid dimensions must be positive integers");
    }
    Ok((w, h))
}

pub fn sample_sources(graph: &CsrGraph, args: &SourceArgs, seed: Option<u64>) -> Result<Vec<u32>> {
    let nv = graph.num_vertices;
    if nv == 0 {
        bail!("Cannot sample sources from an empty graph with 0 vertices");
    }

    if let Some(ref explicit) = args.sources {
        if explicit.is_empty() {
            bail!("Explicit sources list cannot be empty");
        }
        for &s in explicit {
            if s as usize >= nv {
                bail!(
                    "Explicit source vertex {} is out of bounds (graph has {} vertices)",
                    s,
                    nv
                );
            }
        }
        return Ok(explicit.clone());
    }

    let count = args.num_sources.unwrap_or(1).min(nv);
    if count == 0 {
        bail!("Number of sources must be >= 1");
    }

    match args.source_strategy {
        SourceStrategy::First => Ok((0..count as u32).collect()),
        SourceStrategy::HighestDegree => {
            let mut degrees: Vec<(u32, usize)> = (0..nv as u32)
                .map(|v| (v, graph.degree(v as usize)))
                .collect();
            degrees.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            Ok(degrees.into_iter().take(count).map(|(v, _)| v).collect())
        }
        SourceStrategy::Random => {
            let mut all_v: Vec<u32> = (0..nv as u32).collect();
            let mut rng = match seed {
                Some(s) => ChaCha8Rng::seed_from_u64(s),
                None => ChaCha8Rng::from_entropy(),
            };
            all_v.shuffle(&mut rng);
            let mut sampled: Vec<u32> = all_v.into_iter().take(count).collect();
            sampled.sort_unstable();
            Ok(sampled)
        }
    }
}
