use webgpu_bfs_core::CsrGraph;

/// Emits a rich diagnostic failure message when distance array parity or invariant assertion fails.
pub fn format_mismatch_diagnostic(
    method_name: &str,
    graph: &CsrGraph,
    sources: &[u32],
    actual: &[u32],
    expected: &[u32],
) -> String {
    let nv = graph.num_vertices;
    let ns = sources.len();
    let total_len = actual.len();

    let mut mismatches = 0usize;
    let mut first_mismatch = None;

    for i in 0..total_len.min(expected.len()) {
        if actual[i] != expected[i] {
            mismatches += 1;
            if first_mismatch.is_none() {
                let s_idx = i.checked_div(nv).unwrap_or(0);
                let v = i.checked_rem(nv).unwrap_or(0);
                let src_id = sources.get(s_idx).copied().unwrap_or(u32::MAX);
                first_mismatch = Some((i, s_idx, src_id, v, actual[i], expected[i]));
            }
        }
    }

    let mut msg = format!(
        "\n=================== BFS VERIFICATION FAILURE ===================\n\
         Method: {}\n\
         Graph: |V| = {}, |E| = {}\n\
         Sources: {} (len = {})\n\
         Total elements: {}, Total mismatches: {} ({:.2}%)\n",
        method_name,
        nv,
        graph.num_edges,
        if ns <= 10 {
            format!("{:?}", sources)
        } else {
            format!("[{:?}, ... +{} more]", &sources[..5], ns - 5)
        },
        ns,
        total_len,
        mismatches,
        if total_len > 0 {
            (mismatches as f64 / total_len as f64) * 100.0
        } else {
            0.0
        }
    );

    if let Some((idx, s_idx, src_id, v, act, exp)) = first_mismatch {
        let neighbors = if v < nv { graph.neighbors(v) } else { &[] };

        msg.push_str(&format!(
            "--- First Discrepancy Context ---\n\
             Flat Index: {}\n\
             Source Index: {} (Vertex ID: {})\n\
             Target Vertex: {}\n\
             Expected Distance: {}\n\
             Actual Distance:   {}\n\
             Target Degree:     {}\n\
             Outgoing Neighbors: {:?}\n",
            idx,
            s_idx,
            src_id,
            v,
            if exp == u32::MAX {
                "u32::MAX (unreachable)".to_string()
            } else {
                exp.to_string()
            },
            if act == u32::MAX {
                "u32::MAX (unreachable)".to_string()
            } else {
                act.to_string()
            },
            neighbors.len(),
            if neighbors.len() <= 16 {
                format!("{:?}", neighbors)
            } else {
                format!("[{:?}, ... +{} more]", &neighbors[..8], neighbors.len() - 8)
            }
        ));
    }

    msg.push_str("=================================================================\n");
    msg
}
