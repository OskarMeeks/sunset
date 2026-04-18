// cascade_core/features.rs
//
// Per-bar feature extraction and bar resampling shared between
// cascade_trainer and scan_predict.

use crate::generator::StockData;

/// Number of raw features extracted per bar by `extract()`.
pub const NF: usize = 7;

/// Extract 7 normalised price/volume features for one bar relative to the
/// previous bar.  All values are clamped so no single bar can dominate inputs.
#[inline(always)]
pub fn extract(cur: &StockData, prev: &StockData) -> [f64; NF] {
    let eps = 1e-8;
    let pc = prev.close.max(eps);
    let cc = cur.close.max(eps);
    let pv = (prev.volume as f64).max(1.0);
    [
        ((cur.close - pc)              / pc).clamp(-0.05, 0.05),  // close-to-close return
        ((cur.close - cur.open)        / cc).clamp(-0.05, 0.05),  // body / close
        ((cur.high  - cc)              / cc).clamp( 0.0,  0.05),  // upper wick
        ((cc        - cur.low)         / cc).clamp( 0.0,  0.05),  // lower wick
        ((cur.volume as f64 - pv)      / pv).clamp(-1.0,  1.0),   // volume change
        ((cur.high  - cur.low)         / cc).clamp( 0.0,  0.05),  // range / close
        ((cur.open  - pc)              / pc).clamp(-0.05, 0.05),  // open gap
    ]
}

/// Collapse 1-min bars into `interval_mins`-min bars.
/// Returns `data` unchanged when `interval_mins <= 1`.
pub fn resample(data: Vec<StockData>, interval_mins: usize) -> Vec<StockData> {
    if interval_mins <= 1 { return data; }
    let mut out = Vec::with_capacity(data.len() / interval_mins + 1);
    let mut i = 0;
    while i < data.len() {
        let anchor_ts = data[i].ts;
        let mut j = i;
        while j < data.len() {
            let mins = (data[j].ts - anchor_ts).num_minutes();
            if mins < 0 || mins >= interval_mins as i64 { break; }
            j += 1;
        }
        let slice = &data[i..j];
        if slice.is_empty() { i += 1; continue; }
        out.push(StockData {
            ts:     slice[0].ts,
            open:   slice[0].open,
            high:   slice.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max),
            low:    slice.iter().map(|b| b.low) .fold(f64::INFINITY,     f64::min),
            close:  slice.last().unwrap().close,
            volume: slice.iter().map(|b| b.volume).sum(),
        });
        i = j;
    }
    out
}
