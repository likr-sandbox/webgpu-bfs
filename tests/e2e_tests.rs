//! Multi-tier Dual-Track End-to-End (E2E) Test Suite for webgpu-bfs.
//!
//! Tiers covered:
//! - Tier 1: Feature Coverage (Features F1 through F11, 60 test cases)
//! - Tier 2: Boundary & Corner Cases (Pathological structures, 40 test cases)
//! - Tier 3: Cross-Feature Combinations (Pairwise matrix sweeps, 14 test cases)
//! - Tier 4: Real-World Application Scenarios (5 domain scenarios)

pub mod common;
pub mod tier1_features;
pub mod tier2_boundaries;
pub mod tier3_combinations;
pub mod tier4_real_world;
