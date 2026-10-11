//! WebAssembly and browser WebGPU bindings for `webgpu-bfs`.

use wasm_bindgen::prelude::*;

pub mod benchmark;
pub mod cpu;
pub mod gpu;
pub mod graph;
pub mod types;

pub use benchmark::*;
pub use cpu::*;
pub use gpu::*;
pub use graph::*;
pub use types::*;

/// Initializes panic hooks and console logging in the browser.
#[wasm_bindgen]
pub fn init_wasm() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    log::info!("webgpu-bfs-wasm initialized successfully");
}

/// Checks if WebGPU (`navigator.gpu`) is supported in the current browser window.
#[wasm_bindgen]
pub fn is_webgpu_supported() -> bool {
    web_sys::window()
        .and_then(|w| {
            let nav = w.navigator();
            let gpu = js_sys::Reflect::get(&nav, &JsValue::from_str("gpu")).ok()?;
            Some(!gpu.is_undefined() && !gpu.is_null())
        })
        .unwrap_or(false)
}

/// Returns current high-precision timestamp in milliseconds.
pub(crate) fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or_else(|| js_sys::Date::now())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_graph_grid_and_cpu_bfs() {
        let graph = WasmGraph::generate_grid2d(4, 4).unwrap();
        assert_eq!(graph.num_vertices(), 16);
        assert_eq!(graph.num_edges(), 48);

        let sources = vec![0u32];
        let result = run_cpu_bfs(&graph, &sources, Some("SequentialFifo".to_string())).unwrap();
        assert_eq!(result.num_vertices, 16);
        assert_eq!(result.num_sources, 1);
        assert_eq!(result.distance_of(0, 0), 0);
        assert_eq!(result.distance_of(0, 15), 6); // Manhattan distance (3-0)+(3-0)=6
    }

    #[test]
    fn test_wasm_graph_from_mtx() {
        let mtx_data = "%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n2 3\n";
        let graph = WasmGraph::from_mtx(mtx_data).unwrap();
        assert_eq!(graph.num_vertices(), 3);
        assert_eq!(graph.num_edges(), 2);
    }
}
