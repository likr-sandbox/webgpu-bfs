/**
 * Matrix Market (.mtx) Coordinate Format Parser
 * Handles directed, pattern, real/integer, and symmetric graph matrices.
 */

export class MtxParser {
  /**
   * Parses text content of a Matrix Market file into CSR representation.
   * @param {string} text - Raw content of the .mtx file.
   * @returns {{ numVertices: number, numEdges: number, rowOffsets: Uint32Array, colIndices: Uint32Array, isSymmetric: boolean }}
   */
  static parse(text) {
    const lines = text.split(/\r?\n/);
    let bannerFound = false;
    let isSymmetric = false;
    let headerParsed = false;
    let numRows = 0;
    let numCols = 0;
    let numEntries = 0;

    const rawEdges = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      // Banner check
      if (line.startsWith('%%MatrixMarket')) {
        bannerFound = true;
        const tokens = line.toLowerCase().split(/\s+/);
        if (tokens.includes('symmetric') || tokens.includes('skew-symmetric')) {
          isSymmetric = true;
        }
        continue;
      }

      // Comment check
      if (line.startsWith('%')) {
        continue;
      }

      // Matrix dimensions line
      if (!headerParsed) {
        const dims = line.split(/\s+/).map(Number);
        if (dims.length < 3) {
          throw new Error(`Invalid Matrix Market dimension line: "${line}"`);
        }
        numRows = dims[0];
        numCols = dims[1];
        numEntries = dims[2];
        headerParsed = true;
        continue;
      }

      // Coordinate line: row col [val] (1-based indices)
      const parts = line.split(/\s+/);
      if (parts.length >= 2) {
        const u = parseInt(parts[0], 10) - 1; // Convert 1-based to 0-based
        const v = parseInt(parts[1], 10) - 1;

        if (u >= 0 && u < numRows && v >= 0 && v < numCols) {
          if (u !== v) { // Ignore self-loops for BFS
            rawEdges.push([u, v]);
            if (isSymmetric) {
              rawEdges.push([v, u]);
            }
          }
        }
      }
    }

    if (!headerParsed) {
      throw new Error('Matrix Market header not found in provided text.');
    }

    const numVertices = Math.max(numRows, numCols);

    // Group outgoing edges by source node u
    const adj = Array.from({ length: numVertices }, () => []);
    for (const [u, v] of rawEdges) {
      adj[u].push(v);
    }

    // Sort neighbors and deduplicate (strictly sorted CSR)
    let totalEdges = 0;
    for (let u = 0; u < numVertices; u++) {
      adj[u].sort((a, b) => a - b);
      // Dedup
      const unique = [];
      for (let j = 0; j < adj[u].length; j++) {
        if (j === 0 || adj[u][j] !== adj[u][j - 1]) {
          unique.push(adj[u][j]);
        }
      }
      adj[u] = unique;
      totalEdges += unique.length;
    }

    // Build CSR rowOffsets and colIndices
    const rowOffsets = new Uint32Array(numVertices + 1);
    const colIndices = new Uint32Array(totalEdges);

    let offset = 0;
    rowOffsets[0] = 0;
    for (let u = 0; u < numVertices; u++) {
      const neighbors = adj[u];
      for (let j = 0; j < neighbors.length; j++) {
        colIndices[offset + j] = neighbors[j];
      }
      offset += neighbors.length;
      rowOffsets[u + 1] = offset;
    }

    return {
      numVertices,
      numEdges: totalEdges,
      rowOffsets,
      colIndices,
      isSymmetric,
    };
  }
}
