use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tar::Archive;
use webgpu_bfs_core::MtxHeader;

use crate::args::DownloadDatasetArgs;

#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub name: &'static str,
    pub group: &'static str,
    pub vertices: usize,
    pub edges: usize,
    pub description: &'static str,
    pub url: &'static str,
}

pub const KNOWN_DATASETS: &[DatasetMetadata] = &[
    DatasetMetadata {
        name: "luxembourg_osm",
        group: "DIMACS10",
        vertices: 114_599,
        edges: 239_332,
        description: "Luxembourg road network (DIMACS10 Challenge)",
        url: "https://sparse.tamu.edu/MM/DIMACS10/luxembourg_osm.tar.gz",
    },
    DatasetMetadata {
        name: "ca-GrQc",
        group: "SNAP",
        vertices: 5_242,
        edges: 28_980,
        description: "General Relativity collaboration network",
        url: "https://sparse.tamu.edu/MM/SNAP/ca-GrQc.tar.gz",
    },
    DatasetMetadata {
        name: "delaunay_n16",
        group: "DIMACS10",
        vertices: 65_536,
        edges: 393_150,
        description: "Delaunay triangulation with 2^16 vertices",
        url: "https://sparse.tamu.edu/MM/DIMACS10/delaunay_n16.tar.gz",
    },
    DatasetMetadata {
        name: "minnesota",
        group: "Gleich",
        vertices: 2_642,
        edges: 6_606,
        description: "Minnesota road network",
        url: "https://sparse.tamu.edu/MM/Gleich/minnesota.tar.gz",
    },
    DatasetMetadata {
        name: "web-Stanford",
        group: "SNAP",
        vertices: 281_903,
        edges: 2_312_497,
        description: "Stanford web page graph from 2002 (SNAP)",
        url: "https://sparse.tamu.edu/MM/SNAP/web-Stanford.tar.gz",
    },
];

pub fn find_in_catalog(name: &str) -> Option<&'static DatasetMetadata> {
    KNOWN_DATASETS
        .iter()
        .find(|d| d.name.eq_ignore_ascii_case(name))
}

pub fn list_datasets() {
    println!("\nAvailable SuiteSparse Benchmark Datasets in Local Catalog:\n");
    println!(
        " {:<16} {:<10} {:>10} {:>10}  Description",
        "Dataset", "Group", "|V|", "|E| (approx)"
    );
    println!("--------------------------------------------------------------------------------");
    for meta in KNOWN_DATASETS {
        println!(
            " {:<16} {:<10} {:>10} {:>10}  {}",
            meta.name, meta.group, meta.vertices, meta.edges, meta.description
        );
    }
    println!("--------------------------------------------------------------------------------");
    println!("Download any dataset using: webgpu-bfs download-dataset --name <NAME>\n");
}

pub fn ensure_dataset_downloaded(name: &str, target_dir: &Path, force: bool) -> Result<PathBuf> {
    let mtx_path = target_dir.join(name).join(format!("{}.mtx", name));
    if mtx_path.exists() && !force {
        if validate_mtx_file(&mtx_path).is_ok() {
            log::info!("Dataset '{}' already exists at {:?}", name, mtx_path);
            return Ok(mtx_path);
        }
        log::warn!("Existing file {:?} is invalid; re-downloading...", mtx_path);
    }

    let args = DownloadDatasetArgs {
        name: name.to_string(),
        group: None,
        url: None,
        dir: target_dir.to_path_buf(),
        force,
    };
    run_download_dataset(&args)
}

pub fn run_download_dataset(args: &DownloadDatasetArgs) -> Result<PathBuf> {
    let name = &args.name;
    let url = if let Some(ref u) = args.url {
        u.clone()
    } else if let Some(meta) = find_in_catalog(name) {
        meta.url.to_string()
    } else if let Some(ref grp) = args.group {
        format!("https://sparse.tamu.edu/MM/{}/{}.tar.gz", grp, name)
    } else {
        bail!(
            "Dataset '{}' not recognized in catalog and no --group or --url provided. \
             Use 'webgpu-bfs list-datasets' to see available presets.",
            name
        );
    };

    let dataset_dir = args.dir.join(name);
    fs::create_dir_all(&dataset_dir)
        .with_context(|| format!("Failed to create dataset directory {:?}", dataset_dir))?;

    let target_mtx = dataset_dir.join(format!("{}.mtx", name));

    if target_mtx.exists() && !args.force {
        if let Ok((v, e)) = validate_mtx_file(&target_mtx) {
            println!(
                "Dataset '{}' already exists and is valid (|V|={}, |E|={}) at: {}",
                name,
                v,
                e,
                target_mtx.display()
            );
            return Ok(target_mtx);
        }
    }

    println!("Downloading dataset '{}' from: {}", name, url);
    let agent = ureq::builder()
        .redirects(5)
        .timeout(std::time::Duration::from_secs(60))
        .build();

    let response = agent
        .get(&url)
        .call()
        .with_context(|| format!("HTTP GET request failed for URL: {}", url))?;

    if response.status() != 200 {
        bail!(
            "HTTP request returned non-200 status {}: {}",
            response.status(),
            response.status_text()
        );
    }

    println!("Decompressing tar.gz archive stream...");
    let gz_decoder = GzDecoder::new(response.into_reader());
    let mut archive = Archive::new(gz_decoder);

    let mut extracted_files: Vec<PathBuf> = Vec::new();

    for entry in archive.entries().context("Failed to read tar entries")? {
        let mut entry = entry.context("Failed to unpack tar entry")?;
        let path = entry.path().context("Invalid tar entry path")?.into_owned();

        if let Some(ext) = path.extension() {
            if ext.eq_ignore_ascii_case("mtx") {
                let filename = match path.file_name() {
                    Some(f) if !f.is_empty() => f,
                    _ => continue,
                };
                let dest = dataset_dir.join(filename);
                entry
                    .unpack(&dest)
                    .with_context(|| format!("Failed to unpack entry to {:?}", dest))?;
                extracted_files.push(dest);
            }
        }
    }

    if extracted_files.is_empty() {
        bail!(
            "Archive did not contain any .mtx file for dataset '{}'",
            name
        );
    }

    let final_path = select_primary_mtx_file(&dataset_dir, name, &extracted_files)?;
    let (v, e) = validate_mtx_file(&final_path)
        .with_context(|| format!("Extracted file {:?} failed MTX validation", final_path))?;

    println!(
        "Successfully downloaded and verified '{}': |V| = {}, |E| = {}\nLocation: {}",
        name,
        v,
        e,
        final_path.display()
    );

    Ok(final_path)
}

/// Identifies and prepares the primary graph MTX file from all extracted MTX files in the archive.
///
/// Priority:
/// 1. Direct match: `<dataset_name>.mtx` that validates as coordinate-format Matrix Market.
/// 2. Case-insensitive match: e.g. `<DATASET_NAME>.mtx`, copied to `target_mtx`.
/// 3. Fallback: Any extracted MTX file that is NOT an auxiliary file (excluding `*_coord.mtx`,
///    `*_nodename.mtx`, `*_b.mtx`) and validates as coordinate format, copied to `target_mtx`.
/// 4. Secondary fallback: Any file validating as coordinate format.
pub fn select_primary_mtx_file(
    dataset_dir: &Path,
    dataset_name: &str,
    extracted_files: &[PathBuf],
) -> Result<PathBuf> {
    let target_mtx = dataset_dir.join(format!("{}.mtx", dataset_name));
    let expected_filename = format!("{}.mtx", dataset_name);

    // 1. Direct exact filename match
    if extracted_files.contains(&target_mtx) && validate_mtx_file(&target_mtx).is_ok() {
        return Ok(target_mtx);
    }

    // 2. Case-insensitive match
    if let Some(case_match) = extracted_files.iter().find(|p| {
        p.file_name()
            .and_then(|f| f.to_str())
            .map(|f| f.eq_ignore_ascii_case(&expected_filename))
            .unwrap_or(false)
            && validate_mtx_file(p).is_ok()
    }) {
        if case_match != &target_mtx {
            fs::copy(case_match, &target_mtx)
                .with_context(|| format!("Failed to copy {:?} to {:?}", case_match, target_mtx))?;
        }
        return Ok(target_mtx);
    }

    // Helper: identify auxiliary metadata/coordinate files that use array format or are not adjacency matrices
    let is_auxiliary = |p: &Path| -> bool {
        let name_lower = p
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        name_lower.ends_with("_coord.mtx")
            || name_lower.ends_with("_nodename.mtx")
            || name_lower.ends_with("_b.mtx")
    };

    // 3. Fallback: filter for valid coordinate-format MTX files
    let mut valid_candidates: Vec<&PathBuf> = Vec::new();
    for file in extracted_files {
        if validate_mtx_file(file).is_ok() {
            valid_candidates.push(file);
        }
    }

    let chosen = valid_candidates
        .iter()
        .find(|p| !is_auxiliary(p))
        .or_else(|| valid_candidates.first())
        .copied()
        .context(format!(
            "No valid coordinate-format Matrix Market graph found in archive for dataset '{}'",
            dataset_name
        ))?;

    if chosen != &target_mtx {
        fs::copy(chosen, &target_mtx)
            .with_context(|| format!("Failed to copy {:?} to {:?}", chosen, target_mtx))?;
    }

    Ok(target_mtx)
}

pub fn validate_mtx_file(path: &Path) -> Result<(usize, usize)> {
    let file = File::open(path).with_context(|| format!("Cannot open MTX file {:?}", path))?;
    let mut reader = BufReader::new(file);

    let mut first_line = String::new();
    reader.read_line(&mut first_line)?;
    let header = MtxHeader::parse_with_line(first_line.trim(), 1)
        .context("Invalid MatrixMarket banner line")?;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }

        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() == 3 {
            let rows: usize = tokens[0].parse()?;
            let cols: usize = tokens[1].parse()?;
            let nnz: usize = tokens[2].parse()?;
            let vertices = rows.max(cols);
            let edges = if header.symmetry.is_symmetric() {
                nnz * 2
            } else {
                nnz
            };
            return Ok((vertices, edges));
        }
        bail!("Invalid dimensions line {}: '{}'", line_no + 2, trimmed);
    }

    bail!("No dimensions line found in MTX file")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_finds_luxembourg() {
        let meta = find_in_catalog("luxembourg_osm").expect("Should find luxembourg_osm");
        assert_eq!(meta.group, "DIMACS10");
        assert_eq!(meta.vertices, 114_599);
    }

    #[test]
    fn test_catalog_case_insensitive() {
        assert!(find_in_catalog("LUXEMBOURG_OSM").is_some());
        assert!(find_in_catalog("ca-grqc").is_some());
    }

    #[test]
    fn test_unknown_dataset_returns_none() {
        assert!(find_in_catalog("nonexistent_graph").is_none());
    }

    #[test]
    fn test_catalog_minnesota_metadata() {
        let meta = find_in_catalog("minnesota").expect("Should find minnesota");
        assert_eq!(meta.group, "Gleich");
        assert_eq!(
            meta.url,
            "https://sparse.tamu.edu/MM/Gleich/minnesota.tar.gz"
        );
        assert_eq!(meta.vertices, 2_642);
        assert_eq!(meta.edges, 6_606);
    }

    #[test]
    fn test_select_primary_mtx_preserves_main_over_coord() {
        let temp_dir = std::env::temp_dir().join("test_select_primary_mtx_preserves_main");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let main_mtx = temp_dir.join("luxembourg_osm.mtx");
        let coord_mtx = temp_dir.join("luxembourg_osm_coord.mtx");

        // Write coordinate graph to main_mtx
        fs::write(
            &main_mtx,
            "%%MatrixMarket matrix coordinate pattern symmetric\n3 3 2\n1 2\n2 3\n",
        )
        .unwrap();

        // Write array data to coord_mtx
        fs::write(
            &coord_mtx,
            "%%MatrixMarket matrix array real general\n3 2\n1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n",
        )
        .unwrap();

        let extracted = vec![coord_mtx.clone(), main_mtx.clone()];
        let selected = select_primary_mtx_file(&temp_dir, "luxembourg_osm", &extracted).unwrap();

        assert_eq!(selected, main_mtx);
        // Verify main_mtx content was not overwritten by coord_mtx
        let content = fs::read_to_string(&main_mtx).unwrap();
        assert!(content.contains("coordinate"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_select_primary_mtx_fallback_different_name() {
        let temp_dir = std::env::temp_dir().join("test_select_primary_mtx_fallback");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let other_mtx = temp_dir.join("some_graph.mtx");
        let other_coord = temp_dir.join("some_graph_coord.mtx");

        fs::write(
            &other_mtx,
            "%%MatrixMarket matrix coordinate pattern symmetric\n2 2 1\n1 2\n",
        )
        .unwrap();
        fs::write(
            &other_coord,
            "%%MatrixMarket matrix array real general\n2 2\n1.0\n2.0\n3.0\n4.0\n",
        )
        .unwrap();

        let extracted = vec![other_coord, other_mtx];
        let selected = select_primary_mtx_file(&temp_dir, "custom_dataset", &extracted).unwrap();

        let target_mtx = temp_dir.join("custom_dataset.mtx");
        assert_eq!(selected, target_mtx);
        let content = fs::read_to_string(&target_mtx).unwrap();
        assert!(content.contains("coordinate"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_select_primary_mtx_rejects_only_array_files() {
        let temp_dir = std::env::temp_dir().join("test_select_primary_mtx_rejects_array");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let coord_mtx = temp_dir.join("only_coord.mtx");
        fs::write(
            &coord_mtx,
            "%%MatrixMarket matrix array real general\n2 2\n1.0\n2.0\n3.0\n4.0\n",
        )
        .unwrap();

        let extracted = vec![coord_mtx];
        let result = select_primary_mtx_file(&temp_dir, "test_array", &extracted);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
