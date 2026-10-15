//! Dispatcher implementations for GPU BFS methods.

pub mod batched_spmm;
pub mod bit_parallel_32;
pub mod bit_parallel_64;
pub mod bit_parallel_64_pull;
pub mod boolean_reachability;
pub mod indirect;
pub mod multi_word_bp;
pub mod naive_seq;
pub mod push_pull;
pub mod sparse_bp;
pub mod sparse_queue;
pub mod indirect_sparse_push;
pub mod multi_word_push;
pub mod workgroup_persistent;

