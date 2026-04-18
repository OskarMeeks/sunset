// cascade_core/config.rs
//
// Shared Config struct used by both cascade_trainer and scan_predict.
// Serde defaults ensure old .weights files (missing newer fields) still load.

use serde::{Deserialize, Serialize};

fn default_l2()          -> f64  { 1e-4 }
fn default_patience()    -> usize { 15  }
fn default_bar_mins()    -> usize { 5   }
fn default_sniper_only() -> bool  { true }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub lookback:    usize,
    pub hidden:      usize,
    pub layers:      usize,
    pub batch_size:  usize,
    pub lr_decay:    f64,
    pub epochs1:     usize,
    pub epochs2:     usize,
    pub epochs3:     usize,
    pub lr1:         f64,
    pub lr2:         f64,
    pub lr3:         f64,
    #[serde(default = "default_bar_mins")]
    pub bar_mins:    usize,   // candle size in minutes
    pub dir_weight:  f64,     // MSE magnitude weight vs BCE direction term
    #[serde(default = "default_l2")]
    pub l2_lambda:   f64,     // L2 weight decay
    #[serde(default = "default_patience")]
    pub early_stop_patience: usize,
    pub out_prefix:  String,
    /// Always true — cascade_trainer now trains Sniper directly without Scout/Spotter.
    /// Retained in the file format for backwards compatibility with old .weights files.
    #[serde(default = "default_sniper_only")]
    pub sniper_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lookback: 60, hidden: 128, layers: 2, batch_size: 256, lr_decay: 0.997,
            epochs1: 300, epochs2: 300, epochs3: 300,
            lr1: 3e-4, lr2: 3e-4, lr3: 3e-4,
            bar_mins: 5,
            // dir_weight: 1.5 keeps magnitude refinement meaningful without drowning BCE.
            dir_weight: 1.5,
            l2_lambda: 1e-4,
            early_stop_patience: 15,
            out_prefix: "default".into(),
            sniper_only: true,
        }
    }
}
