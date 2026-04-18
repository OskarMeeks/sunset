// cascade_core/mod.rs
//
// Shared library code used by both cascade_trainer and scan_predict.
//
//  cascade_core::config   — Config struct (serde, defaults)
//  cascade_core::features — extract(), resample(), NF constant
//  cascade_core::net      — Layer, Net (inference), save/load_all_weights

pub mod config;
pub mod features;
pub mod net;

// Convenience re-exports so callers can write `cascade_core::Config` etc.
pub use config::Config;
pub use features::{extract, resample, NF};
pub use net::{Layer, Net, load_all_weights, save_all_weights};
