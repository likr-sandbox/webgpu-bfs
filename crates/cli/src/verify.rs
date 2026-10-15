use serde::Serialize;
use webgpu_bfs_core::CsrGraph;

#[derive(Debug, Clone, Serialize)]
pub struct DiscrepancyDetail {
    pub flat_index: usize,
    pub source_index: usize,
    pub source_vertex: u32,
    pub target_vertex: u32,
    pub target_degree: usize,
    pub expected_dist: u32,
    pub actual_dist: u32,
    pub neighbor_sample: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerificationResult {
    pub method_name: String,
    pub is_valid: bool,
    pub total_elements: usize,
    pub mismatched_elements: usize,
    pub self_distance_passed: bool,
    pub triangle_inequality_passed: bool,
    pub message: String,
    pub diff_samples: Vec<DiscrepancyDetail>,
}

pub fn verify_engine_output(
    method_name: &str,
    graph: &CsrGraph,
    sources: &[u32],
    gpu_dist: &[u32],
    cpu_dist: &[u32],
    max_diffs: usize,
) -> VerificationResult {
    let nv = graph.num_vertices;
    let num_sources = sources.len();
    let total_elements = num_sources * nv;

    if gpu_dist.len() != total_elements {
        return VerificationResult {
            method_name: method_name.to_string(),
            is_valid: false,
            total_elements,
            mismatched_elements: total_elements,
            self_distance_passed: false,
            triangle_inequality_passed: false,
            message: format!(
                "Output buffer length mismatch: expected {} (S={} * |V|={}), got {}",
                total_elements,
                num_sources,
                nv,
                gpu_dist.len()
            ),
            diff_samples: Vec::new(),
        };
    }

    let mut mismatched_elements = 0;
    let mut diff_samples = Vec::new();

    // 1. Bitwise Parity Check against CPU ground truth
    for (s_idx, &s_id) in sources.iter().enumerate() {
        let offset = s_idx * nv;

        for v in 0..nv {
            let idx = offset + v;
            let actual = gpu_dist[idx];
            let expected = cpu_dist[idx];

            if actual != expected {
                mismatched_elements += 1;
                if diff_samples.len() < max_diffs {
                    let neighbors = graph.neighbors(v);
                    let sample: Vec<u32> = neighbors.iter().take(8).copied().collect();
                    diff_samples.push(DiscrepancyDetail {
                        flat_index: idx,
                        source_index: s_idx,
                        source_vertex: s_id,
                        target_vertex: v as u32,
                        target_degree: neighbors.len(),
                        expected_dist: expected,
                        actual_dist: actual,
                        neighbor_sample: sample,
                    });
                }
            }
        }
    }

    // 2. Self-Distance Invariant: dist[s * |V| + s] == 0
    let mut self_distance_passed = true;
    for (s_idx, &s_id) in sources.iter().enumerate() {
        if (s_id as usize) < nv {
            let self_dist = gpu_dist[s_idx * nv + (s_id as usize)];
            if self_dist != 0 {
                self_distance_passed = false;
            }
        }
    }

    // 3. Triangle Inequality: for each edge (u, v), dist(s, v) <= dist(s, u) + 1
    let mut triangle_inequality_passed = true;
    for s_idx in 0..num_sources {
        let offset = s_idx * nv;
        for u in 0..nv {
            let d_u = gpu_dist[offset + u];
            if d_u != u32::MAX {
                for &v_nbr in graph.neighbors(u) {
                    let d_v = gpu_dist[offset + v_nbr as usize];
                    if d_v == u32::MAX || d_v > d_u + 1 {
                        triangle_inequality_passed = false;
                        break;
                    }
                }
            }
            if !triangle_inequality_passed {
                break;
            }
        }
        if !triangle_inequality_passed {
            break;
        }
    }

    let is_valid = mismatched_elements == 0 && self_distance_passed && triangle_inequality_passed;
    let message = if is_valid {
        format!(
            "PASS: 100% bitwise parity and structural invariants verified ({} elements)",
            total_elements
        )
    } else {
        format!(
            "FAIL: {} / {} discrepancies detected in method '{}'",
            mismatched_elements, total_elements, method_name
        )
    };

    VerificationResult {
        method_name: method_name.to_string(),
        is_valid,
        total_elements,
        mismatched_elements,
        self_distance_passed,
        triangle_inequality_passed,
        message,
        diff_samples,
    }
}

pub fn verify_reachability_output(
    method_name: &str,
    graph: &CsrGraph,
    sources: &[u32],
    gpu_mask: &webgpu_bfs_core::ReachabilityBitmask,
    cpu_mask: &webgpu_bfs_core::ReachabilityBitmask,
    max_diffs: usize,
) -> VerificationResult {
    let nv = graph.num_vertices;
    let num_sources = sources.len();
    let total_elements = num_sources * nv;

    let mut mismatched_elements = 0;
    let mut diff_samples = Vec::new();
    let mut self_reach_passed = true;

    for (s_idx, &s_id) in sources.iter().enumerate() {
        if (s_id as usize) < nv && !gpu_mask.is_reachable(s_idx, s_id as usize) {
            self_reach_passed = false;
        }

        for v in 0..nv {
            let actual = gpu_mask.is_reachable(s_idx, v);
            let expected = cpu_mask.is_reachable(s_idx, v);

            if actual != expected {
                mismatched_elements += 1;
                if diff_samples.len() < max_diffs {
                    let neighbors = graph.neighbors(v);
                    let sample: Vec<u32> = neighbors.iter().take(8).copied().collect();
                    diff_samples.push(DiscrepancyDetail {
                        flat_index: s_idx * nv + v,
                        source_index: s_idx,
                        source_vertex: s_id,
                        target_vertex: v as u32,
                        target_degree: neighbors.len(),
                        expected_dist: expected as u32,
                        actual_dist: actual as u32,
                        neighbor_sample: sample,
                    });
                }
            }
        }
    }

    let is_valid = mismatched_elements == 0 && self_reach_passed;
    let message = if is_valid {
        format!(
            "PASS: 100% bitwise reachability parity verified ({} source-vertex pairs)",
            total_elements
        )
    } else {
        format!(
            "FAIL: {} / {} reachability discrepancies detected in method '{}'",
            mismatched_elements, total_elements, method_name
        )
    };

    VerificationResult {
        method_name: method_name.to_string(),
        is_valid,
        total_elements,
        mismatched_elements,
        self_distance_passed: self_reach_passed,
        triangle_inequality_passed: true,
        message,
        diff_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use webgpu_bfs_core::generate_grid2d;

    #[test]
    fn test_verification_pass_on_identical_distances() {
        let graph = generate_grid2d(4, 4);
        let sources = vec![0, 5];
        let dist = vec![
            0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 2, 1, 2, 3, 1, 0, 1, 2, 2, 1, 2, 3, 3,
            2, 3, 4,
        ];
        let res = verify_engine_output("TestEngine", &graph, &sources, &dist, &dist, 10);
        assert!(res.is_valid);
        assert_eq!(res.mismatched_elements, 0);
        assert!(res.self_distance_passed);
        assert!(res.triangle_inequality_passed);
    }

    #[test]
    fn test_verification_detects_mismatch_and_diffs() {
        let graph = generate_grid2d(4, 4);
        let sources = vec![0];
        let cpu_dist = vec![0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6];
        let mut gpu_dist = cpu_dist.clone();
        gpu_dist[3] = 99; // Introduce discrepancy at vertex 3

        let res = verify_engine_output("TestEngine", &graph, &sources, &gpu_dist, &cpu_dist, 5);
        assert!(!res.is_valid);
        assert_eq!(res.mismatched_elements, 1);
        assert_eq!(res.diff_samples.len(), 1);
        assert_eq!(res.diff_samples[0].target_vertex, 3);
        assert_eq!(res.diff_samples[0].expected_dist, 3);
        assert_eq!(res.diff_samples[0].actual_dist, 99);
    }
}
