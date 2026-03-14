// scan_predict.rs
//
// Loads pre-trained cascade weights and predicts the next 10 minutes.
//
// Confidence is based on two signals that measure whether the model actually
// understands this specific situation:
//
//   1. Prediction stability — run the same input through the network N times
//      with small noise added. If the direction stays consistent across all
//      runs, the model is decisive. If it flips, we're on a decision boundary.
//
//   2. Training familiarity — find the K nearest feature vectors in the
//      training data. If the current market looks like situations the model
//      has seen many times, confidence is higher. If it looks unusual,
//      confidence drops regardless of how loud the raw output is.
//
//   confidence = stability × familiarity
//
//   ≥ 70%  →  HIGH
//   ≥ 40%  →  MEDIUM
//   < 40%  →  LOW
//
// ─── Workflow ────────────────────────────────────────────────────────────────
//
//   # Train once (overnight):
//   cargo run --release --bin cascade_trainer -- AAPL_data.csv \
//       --epochs1 300 --epochs2 300 --epochs3 300              \
//       --save-weights AAPL.weights
//
//   # Live prediction right now:
//   cargo run --release --bin scan_predict -- \
//       AAPL.weights --symbol AAPL --api-key YOUR_KEY
//
//   # Test on a specific past moment:
//   cargo run --release --bin scan_predict -- \
//       AAPL.weights --symbol AAPL --api-key YOUR_KEY \
//       --at "2025-06-13 14:30:00"
//
//   # Use local CSV (no API call):
//   cargo run --release --bin scan_predict -- \
//       AAPL.weights --csv AAPL_data.csv \
//       --at "2025-06-13 14:30:00"
//
// ─── Cargo.toml ──────────────────────────────────────────────────────────────
//   [[bin]]
//   name = "scan_predict"
//   path = "src/scan_predict.rs"
// ─────────────────────────────────────────────────────────────────────────────

use chrono::{DateTime, Duration as ChronoDuration, NaiveDate, NaiveDateTime, TimeZone, Utc};
use std::collections::HashMap;
use std::env;
use std::thread;
use std::time::Duration;

// ─────────────────────────────────────────────
//  Config
// ─────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Config {
    lookback:   usize,
    hidden:     usize,
    layers:     usize,
    batch_size: usize,
    lr_decay:   f64,
    epochs1:    usize,
    epochs2:    usize,
    epochs3:    usize,
    lr1:        f64,
    lr2:        f64,
    lr3:        f64,
    dir_weight: f64,
    // New fields — serde defaults so old .weights files still load cleanly
    #[serde(default = "default_l2")]
    l2_lambda:  f64,
    #[serde(default = "default_patience")]
    early_stop_patience: usize,
    out_prefix: String,
}

fn default_l2()       -> f64   { 1e-4 }
fn default_patience() -> usize { 15   }

impl Default for Config {
    fn default() -> Self {
        Self {
            lookback: 60, hidden: 128, layers: 2, batch_size: 256, lr_decay: 0.997,
            epochs1: 300, epochs2: 300, epochs3: 300,
            lr1: 0.001, lr2: 0.001, lr3: 0.001,
            dir_weight: 0.3, l2_lambda: 1e-4, early_stop_patience: 15,
            out_prefix: "model".into(),
        }
    }
}


// ─────────────────────────────────────────────
//  Data fetching — delegated to generator module
// ─────────────────────────────────────────────

mod generator;
use generator::{ensure_data_for_date, fetch_or_load, parse_csv, StockData};

// ─────────────────────────────────────────────
//  Technical indicators  (18 features)
// ─────────────────────────────────────────────

const INDICATOR_NF: usize = 18;

fn ema(closes: &[f64], p: usize) -> f64 {
    if closes.is_empty() { return 0.0; }
    let k = 2.0 / (p as f64 + 1.0);
    closes.iter().skip(1).fold(closes[0], |e, &c| c * k + e * (1.0 - k))
}

fn rsi(closes: &[f64], p: usize) -> f64 {
    if closes.len() < p + 1 { return 0.5; }
    let w = &closes[closes.len() - p - 1..];
    let (mut ag, mut al) = (0.0_f64, 0.0_f64);
    for s in w.windows(2) { let d = s[1]-s[0]; if d>0.0 {ag+=d;} else {al+=d.abs();} }
    ag /= p as f64; al /= p as f64;
    if al < 1e-10 { return 1.0; }
    let rs = ag / al; (rs / (1.0 + rs)).clamp(0.0, 1.0)
}

fn macd(closes: &[f64]) -> (f64, f64) {
    if closes.len() < 26 { return (0.0, 0.0); }
    let line = ema(closes, 12) - ema(closes, 26);
    let n = closes.len();
    let series: Vec<f64> = (9..=n).map(|e| ema(&closes[..e], 12) - ema(&closes[..e], 26)).collect();
    let sig = ema(&series, 9);
    let hist = (line - sig) / closes.last().unwrap_or(&1.0).abs().max(1e-8);
    (hist.clamp(-0.01, 0.01) / 0.01, sig.signum())
}

fn bollinger(closes: &[f64], p: usize) -> (f64, f64) {
    if closes.len() < p { return (0.0, 0.0); }
    let w = &closes[closes.len()-p..];
    let mean = w.iter().sum::<f64>() / p as f64;
    let std  = (w.iter().map(|c| (c-mean).powi(2)).sum::<f64>() / p as f64).sqrt().max(1e-8);
    let c = *closes.last().unwrap();
    let pb = ((c-(mean-2.0*std))/(4.0*std).max(1e-8)).clamp(0.0,1.0)*2.0-1.0;
    let bw = (4.0*std/mean.abs().max(1e-8)).clamp(0.0,0.1)/0.1;
    (pb, bw)
}

fn atr_ratio(data: &[StockData], p: usize) -> f64 {
    if data.len() < p+1 { return 0.0; }
    let trs: Vec<f64> = data.windows(2).map(|w| {
        (w[1].high-w[1].low).max((w[1].high-w[0].close).abs()).max((w[1].low-w[0].close).abs())
        / w[1].close.max(1e-8)
    }).collect();
    let avg = trs[trs.len().saturating_sub(p)..].iter().sum::<f64>() / p.min(trs.len()) as f64;
    ((*trs.last().unwrap_or(&0.0)/avg.max(1e-8)).clamp(0.0,5.0)/5.0)*2.0-1.0
}

fn pivot_sr(data: &[StockData]) -> (f64, f64, f64) {
    if data.len() < 5 { return (0.0,0.0,0.0); }
    let close = data.last().unwrap().close;
    let (mut sup, mut res): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
    for i in 2..data.len()-2 {
        let (h,l) = (data[i].high, data[i].low);
        if h>data[i-1].high&&h>data[i-2].high&&h>data[i+1].high&&h>data[i+2].high { res.push(h); }
        if l<data[i-1].low &&l<data[i-2].low &&l<data[i+1].low &&l<data[i+2].low  { sup.push(l); }
    }
    let sd = sup.iter().filter(|&&s|s<close).map(|&s|(close-s)/close).fold(f64::MAX,f64::min);
    let rd = res.iter().filter(|&&r|r>close).map(|&r|(r-close)/close).fold(f64::MAX,f64::min);
    let sn = if sd==f64::MAX{-1.0}else{(1.0-sd.clamp(0.0,0.05)/0.05)*2.0-1.0};
    let rn = if rd==f64::MAX{ 1.0}else{(1.0-rd.clamp(0.0,0.05)/0.05)*2.0-1.0};
    let all: Vec<f64> = sup.iter().chain(res.iter()).copied().collect();
    let cl = if all.is_empty(){0.0}else{
        (all.iter().filter(|&&p|((p-close)/close).abs()<0.005).count() as f64/all.len() as f64).clamp(0.0,1.0)
    };
    (sn, rn, cl)
}

fn vwap(data: &[StockData]) -> (f64, f64) {
    if data.is_empty() { return (0.0,0.0); }
    let (mut cpv, mut cv) = (0.0, 0.0);
    let vwaps: Vec<f64> = data.iter().map(|b| {
        cpv += (b.high+b.low+b.close)/3.0*(b.volume as f64).max(1.0);
        cv  += (b.volume as f64).max(1.0); cpv/cv
    }).collect();
    let vw = *vwaps.last().unwrap();
    let c  = data.last().unwrap().close;
    let dist  = ((c-vw)/c.max(1e-8)).clamp(-0.05,0.05)/0.05;
    let n     = vwaps.len();
    let slope = if n>=5{((vwaps[n-1]-vwaps[n-5])/vwaps[n-5].abs().max(1e-8)).clamp(-0.02,0.02)/0.02}else{0.0};
    (dist, slope)
}

fn obv(data: &[StockData]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let (mut v, mut scale) = (0.0_f64, 0.0_f64);
    let mut obvs = vec![0.0_f64; data.len()];
    for i in 1..data.len() {
        let d = data[i].close - data[i-1].close;
        v += if d>0.0{data[i].volume as f64}else if d<0.0{-(data[i].volume as f64)}else{0.0};
        obvs[i] = v; scale = scale.max(v.abs());
    }
    let n = obvs.len(); let lb = 10.min(n-1);
    ((obvs[n-1]-obvs[n-1-lb])/scale.max(1.0)).clamp(-1.0,1.0)
}

fn vol_ratio(data: &[StockData]) -> f64 {
    if data.is_empty() { return 0.0; }
    let w   = &data[data.len().saturating_sub(20)..];
    let avg = w.iter().map(|b| b.volume as f64).sum::<f64>() / w.len().max(1) as f64;
    (data.last().unwrap().volume as f64/avg.max(1.0)).clamp(0.0,5.0)/5.0*2.0-1.0
}

fn compute_indicators(data: &[StockData]) -> [f64; INDICATOR_NF] {
    let closes: Vec<f64> = data.iter().map(|b| b.close).collect();
    let c   = *closes.last().unwrap_or(&1.0);
    let e9  = ema(&closes, 9);
    let e21 = ema(&closes, 21);
    let e9d  = ((c-e9) /c.max(1e-8)).clamp(-0.05,0.05)/0.05;
    let e21d = ((c-e21)/c.max(1e-8)).clamp(-0.05,0.05)/0.05;
    let e9sl = if closes.len()>=2 {
        let now=ema(&closes,9); let prev=ema(&closes[..closes.len()-1],9);
        ((now-prev)/prev.abs().max(1e-8)).clamp(-0.02,0.02)/0.02
    } else { 0.0 };
    let cross    = if e9>e21{1.0}else{-1.0};
    let rsi14    = rsi(&closes,14)*2.0-1.0;
    let (mh,ms)  = macd(&closes);
    let (pb,bw)  = bollinger(&closes,20);
    let atr_r    = atr_ratio(data,14);
    let (sup,res,sr) = pivot_sr(data);
    let (vd,vs)  = vwap(data);
    let o        = obv(data);
    let vol      = vol_ratio(data);
    let bar      = data.last().unwrap();
    let body     = ((bar.close-bar.open)/(bar.high-bar.low).max(1e-8)).clamp(-1.0,1.0);
    [e9d,e21d,e9sl,cross,rsi14,mh,ms,pb,bw,atr_r,sup,res,sr,vd,vs,o,vol,body]
}

// ─────────────────────────────────────────────
//  Indicator confluence helper
//
//  Returns the fraction of the 18 indicators whose sign agrees with the
//  predicted direction (`bullish` = true means prediction is positive).
//
//  Each indicator has a natural polarity (positive value = bullish signal).
//  We count how many align, then normalise to 0.0–1.0.
//
//  Indicator index map (same order as compute_indicators):
//    [0]  e9d      — price above EMA9  → bullish
//    [1]  e21d     — price above EMA21 → bullish
//    [2]  e9sl     — EMA9 sloping up   → bullish
//    [3]  cross    — EMA9 > EMA21      → bullish
//    [4]  rsi14    — RSI > 0.5 (>50)   → bullish
//    [5]  macd_h   — MACD histogram +  → bullish
//    [6]  macd_sig — MACD signal +     → bullish
//    [7]  bollinger %B — above midband → bullish
//    [8]  boll_bw  — bandwidth neutral  (excluded from directional vote)
//    [9]  atr_r    — ATR neutral        (excluded from directional vote)
//   [10]  pivot_sup — support near     → bullish (higher = closer support)
//   [11]  pivot_res — resistance near  → bearish (higher = closer resistance → inverted)
//   [12]  sr_cluster — neutral          (excluded)
//   [13]  vwap_dist — above VWAP       → bullish
//   [14]  vwap_slope — VWAP rising     → bullish
//   [15]  obv       — OBV momentum +   → bullish
//   [16]  vol_ratio — high volume      → confirms direction (neutral, excluded)
//   [17]  body      — bullish candle   → bullish
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
//  Confidence: stability + familiarity
// ─────────────────────────────────────────────
//
//  stability   — how consistently the Sniper predicts the same direction
//                when the input is perturbed with small noise (30 passes).
//                A model that's decisive about this situation won't flip.
//
//  familiarity — how close the current feature vector is to the K nearest
//                neighbours in the training set. Far from training data = low trust.
//
//  confidence  = stability × familiarity   (both in [0,1])

// ─────────────────────────────────────────────
//  k-NN index over training feature vectors
// ─────────────────────────────────────────────

pub struct KnnIndex {
    // Each entry is the flattened indicator vector for one training sample.
    // We only store the INDICATOR_NF features (not the full bar sequence) since
    // those are the normalised signals that best describe market regime.
    vecs: Vec<[f64; INDICATOR_NF]>,
    // 90th-percentile distance across random sample pairs — used to normalise
    // familiarity so "average" training distance maps to ~0.5 familiarity.
    p90_dist: f64,
}

impl KnnIndex {
    /// Build from the full CSV data. Uses the same indicator computation as training.
    pub fn build(all_data: &[StockData], lookback: usize) -> Self {
        let n = all_data.len();
        if n < lookback {
            return KnnIndex { vecs: vec![], p90_dist: 1.0 };
        }
        let vecs: Vec<[f64; INDICATOR_NF]> = (lookback..n)
            .map(|i| compute_indicators(&all_data[i + 1 - lookback..=i]))
            .collect();

        // Estimate p90 distance from a random sample of 500 pairs
        let sample_n = vecs.len().min(500);
        let mut dists: Vec<f64> = Vec::with_capacity(sample_n);
        // Deterministic pseudo-random walk through the index
        let mut rng = 0xdeadbeef_u64;
        for _ in 0..sample_n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let i = (rng >> 33) as usize % vecs.len();
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % vecs.len();
            dists.push(l2_dist(&vecs[i], &vecs[j]));
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90_dist = dists[(dists.len() as f64 * 0.9) as usize].max(1e-8);

        KnnIndex { vecs, p90_dist }
    }

    /// Returns a familiarity score in [0, 1].
    /// 1.0 = query is right in the middle of training data.
    /// 0.0 = query is far from anything the model has seen.
    pub fn familiarity(&self, query: &[f64; INDICATOR_NF], k: usize) -> f64 {
        if self.vecs.is_empty() { return 0.5; }
        // Find k nearest distances
        let mut heap: Vec<f64> = Vec::with_capacity(k + 1);
        for v in &self.vecs {
            let d = l2_dist(query, v);
            if heap.len() < k {
                heap.push(d);
                if heap.len() == k {
                    // turn into max-heap by sorting descending
                    heap.sort_by(|a,b| b.partial_cmp(a).unwrap());
                }
            } else if d < heap[0] {
                heap[0] = d;
                heap.sort_by(|a,b| b.partial_cmp(a).unwrap());
            }
        }
        let avg_knn_dist = heap.iter().sum::<f64>() / heap.len().max(1) as f64;
        // Map distance to [0,1]: distance=0 → 1.0, distance=p90 → 0.5, distance=2*p90 → ~0.0
        let ratio = avg_knn_dist / self.p90_dist;
        (1.0 - ratio * 0.5).clamp(0.0, 1.0)
    }
}

fn l2_dist(a: &[f64; INDICATOR_NF], b: &[f64; INDICATOR_NF]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// ─────────────────────────────────────────────
//  Stability: jitter test on the Sniper
// ─────────────────────────────────────────────
//
//  Adds small Gaussian noise to the input vector, runs N forward passes,
//  counts how many agree with the original direction.
//  Returns fraction of agreeing passes → 1.0 = rock solid, 0.5 = coin flip.

// ─────────────────────────────────────────────
//  Combined confidence score
// ─────────────────────────────────────────────

fn confidence(stability: f64, familiarity: f64) -> f64 {
    (stability * familiarity).clamp(0.0, 1.0)
}

fn confidence_label(c: f64) -> &'static str {
    if c >= 0.70 { "HIGH  " }
    else if c >= 0.40 { "MEDIUM" }
    else { "LOW   " }
}

fn confidence_bar(c: f64) -> String {
    let filled = (c * 10.0).round() as usize;
    (0..10).map(|i| if i < filled { '█' } else { '░' }).collect()
}

// ─────────────────────────────────────────────
//  Per-bar feature extraction  (7 features)
// ─────────────────────────────────────────────

const NF: usize = 7;


// ─────────────────────────────────────────────
//  Resampling — collapse 1-min bars into N-min bars
// ─────────────────────────────────────────────

fn resample(data: Vec<StockData>, interval_mins: usize) -> Vec<StockData> {
    if interval_mins <= 1 { return data; }
    let mut out = Vec::with_capacity(data.len() / interval_mins + 1);
    let mut i = 0;
    while i < data.len() {
        let anchor_ts = data[i].ts;
        // Collect all bars within this interval window
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
            low:    slice.iter().map(|b| b.low).fold(f64::INFINITY, f64::min),
            close:  slice.last().unwrap().close,
            volume: slice.iter().map(|b| b.volume).sum(),
        });
        i = j;
    }
    out
}
#[inline(always)]
fn extract(cur: &StockData, prev: &StockData) -> [f64; NF] {
    let eps=1e-8;
    let pc=prev.close.max(eps); let cc=cur.close.max(eps); let pv=(prev.volume as f64).max(1.0);
    [
        ((cur.close-pc)/pc).clamp(-0.05,0.05),
        ((cur.close-cur.open)/cc).clamp(-0.05,0.05),
        ((cur.high-cc)/cc).clamp(0.0,0.05),
        ((cc-cur.low)/cc).clamp(0.0,0.05),
        ((cur.volume as f64-pv)/pv).clamp(-1.0,1.0),
        ((cur.high-cur.low)/cc).clamp(0.0,0.05),
        ((cur.open-pc)/pc).clamp(-0.05,0.05),
    ]
}

// ─────────────────────────────────────────────
//  Neural layer  (forward-only)
// ─────────────────────────────────────────────

#[derive(Clone)]
struct Layer {
    in_size: usize, out_size: usize,
    w: Vec<f64>, w_t: Vec<f64>, b: Vec<f64>,
    is_output: bool,
}

impl Layer {
    fn blank(in_size: usize, out_size: usize, is_output: bool) -> Self {
        Layer { in_size, out_size,
            w:   vec![0.0; in_size*out_size],
            w_t: vec![0.0; in_size*out_size],
            b:   vec![0.0; out_size],
            is_output,
        }
    }
    fn sync_transpose(&mut self) {
        for o in 0..self.out_size {
            for i in 0..self.in_size { self.w_t[i*self.out_size+o] = self.w[o*self.in_size+i]; }
        }
    }
    fn forward(&self, inp: &[f64], out: &mut [f64]) {
        for o in 0..self.out_size {
            let off = o*self.in_size;
            let z = self.b[o]+self.w[off..off+self.in_size].iter().zip(inp).map(|(&w,&x)|w*x).sum::<f64>();
            out[o] = if self.is_output { z } else { z.tanh() };
        }
    }
}

// ─────────────────────────────────────────────
//  Net — inference only
// ─────────────────────────────────────────────

struct Net {
    layers: Vec<Layer>,
    target_offsets: Vec<usize>,
    lookback: usize,
}

impl Net {
    fn blank(target_offsets: Vec<usize>, n_extra: usize, cfg: &Config) -> Self {
        let input_size = (cfg.lookback-1)*NF + INDICATOR_NF + n_extra;
        let mut layers = Vec::new();
        let mut prev = input_size;
        for _ in 0..cfg.layers { layers.push(Layer::blank(prev, cfg.hidden, false)); prev = cfg.hidden; }
        layers.push(Layer::blank(prev, target_offsets.len(), true));
        Net { layers, target_offsets, lookback: cfg.lookback }
    }

    fn deserialize_weights(&mut self, blob: &[u8]) {
        let mut pos = 0usize;
        let read_u64 = |p: &mut usize| -> u64 { let v=u64::from_le_bytes(blob[*p..*p+8].try_into().unwrap()); *p+=8; v };
        let read_f64 = |p: &mut usize| -> f64 { let v=f64::from_le_bytes(blob[*p..*p+8].try_into().unwrap()); *p+=8; v };
        let n = read_u64(&mut pos) as usize;
        assert_eq!(n, self.layers.len(), "Layer count mismatch");
        for l in &mut self.layers {
            let is=read_u64(&mut pos) as usize; let os=read_u64(&mut pos) as usize; let io=read_u64(&mut pos)!=0;
            assert_eq!(is,l.in_size); assert_eq!(os,l.out_size); assert_eq!(io,l.is_output);
            for w in &mut l.w { *w=read_f64(&mut pos); }
            for b in &mut l.b { *b=read_f64(&mut pos); }
            l.sync_transpose();
        }
    }

    fn predict(&self, data: &[StockData], ind: &[f64; INDICATOR_NF], cascade: &[f64], anchor: f64) -> HashMap<usize,(f64,f64)> {
        let mut inp = Vec::with_capacity(self.layers[0].in_size);
        for i in 1..self.lookback { inp.extend_from_slice(&extract(&data[i], &data[i-1])); }
        inp.extend_from_slice(ind);
        inp.extend_from_slice(cascade);
        let mut acts: Vec<Vec<f64>> = vec![inp];
        for layer in &self.layers {
            let mut out = vec![0.0; layer.out_size];
            layer.forward(acts.last().unwrap(), &mut out);
            acts.push(out);
        }
        let preds = acts.last().unwrap();
        self.target_offsets.iter().enumerate()
            .map(|(i,&off)| { let pct=preds[i]; (off+1,(pct, anchor*(1.0+pct))) })
            .collect()
    }
}

impl Net {
    /// Build the input vector for this window (same as predict, without running the net).
    fn build_input(&self, data: &[StockData], ind: &[f64; INDICATOR_NF], cascade: &[f64]) -> Vec<f64> {
        let mut inp = Vec::with_capacity(self.layers[0].in_size);
        for i in 1..self.lookback { inp.extend_from_slice(&extract(&data[i], &data[i-1])); }
        inp.extend_from_slice(ind);
        inp.extend_from_slice(cascade);
        inp
    }

    /// Forward pass from a pre-built input vector. Returns raw output for target index 0.
    fn forward_raw(&self, inp: &[f64]) -> Vec<f64> {
        let mut acts = inp.to_vec();
        let mut buf  = vec![0.0f64; self.layers.iter().map(|l| l.out_size).max().unwrap_or(1)];
        for layer in &self.layers {
            buf[..layer.out_size].iter_mut().for_each(|x| *x = 0.0);
            layer.forward(&acts, &mut buf[..layer.out_size]);
            acts.resize(layer.out_size, 0.0);
            acts.copy_from_slice(&buf[..layer.out_size]);
        }
        acts
    }

    /// Jitter stability for a given minute offset (0-indexed).
    /// Returns (stability_fraction, per_pass_predictions).
    /// stability = fraction of N noisy passes that agree with the clean prediction direction.
    pub fn stability(
        &self,
        inp:       &[f64],
        target_i:  usize,    // index into target_offsets
        n_passes:  usize,
        noise_std: f64,
    ) -> (f64, Vec<f64>) {
        let clean = self.forward_raw(inp);
        let direction = clean[target_i] >= 0.0;
        let mut agree = 0usize;
        let mut pass_pcts: Vec<f64> = Vec::with_capacity(n_passes);
        let mut rng = 0xc0ffee_u64;
        for _ in 0..n_passes {
            let noisy: Vec<f64> = inp.iter().map(|&x| {
                // Uniform noise in [-noise_std, +noise_std]
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (rng >> 1) as f64 / i64::MAX as f64 - 1.0; // [-1, 1]
                x + u * noise_std
            }).collect();
            let out = self.forward_raw(&noisy)[target_i];
            pass_pcts.push(out);
            if (out >= 0.0) == direction {
                agree += 1;
            }
        }
        (agree as f64 / n_passes as f64, pass_pcts)
    }
}

// ─────────────────────────────────────────────
//  Weight file loader
// ─────────────────────────────────────────────

fn load_weights(path: &str) -> (Config, Net, Net, Net) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Cannot read '{}': {}", path, e));
    assert!(bytes.len() >= 8, "File too small");
    assert_eq!(&bytes[..8], b"CASC_W01", "Not a valid .weights file — use --save-weights when training");
    let mut pos = 8usize;
    let read_u64 = |p: &mut usize| -> u64 { let v=u64::from_le_bytes(bytes[*p..*p+8].try_into().unwrap()); *p+=8; v };
    let cfg_len = read_u64(&mut pos) as usize;
    let cfg: Config = serde_json::from_slice(&bytes[pos..pos+cfg_len]).expect("Bad config in weights");
    pos += cfg_len;
    let mut scout   = Net::blank(vec![9],           0, &cfg);
    let mut spotter = Net::blank(vec![0,4,9],       1, &cfg);
    let mut sniper  = Net::blank((0..10).collect(), 4, &cfg);
    for net in [&mut scout, &mut spotter, &mut sniper] {
        let len = read_u64(&mut pos) as usize;
        net.deserialize_weights(&bytes[pos..pos+len]);
        pos += len;
    }
    (cfg, scout, spotter, sniper)
}

// ─────────────────────────────────────────────
//  Data sources
// ─────────────────────────────────────────────


// ─────────────────────────────────────────────
//  Prediction + display
// ─────────────────────────────────────────────

fn run_prediction(
    window:        &[StockData],
    scout:         &Net,
    spotter:       &Net,
    sniper:        &Net,
    future:        &[StockData],
    mode_label:    &str,
    knn:           &KnnIndex,
    jitter_passes: usize,
    knn_k:         usize,
    noise_std:     f64,
) -> (usize, usize) { // (correct, total) directional calls
    if window.is_empty() { eprintln!("Window is empty — cannot predict."); return (0, 0); }

    let anchor_bar = window.last().unwrap();
    let anchor     = anchor_bar.close;

    println!();
    println!("  Mode       : {}", mode_label);
    println!("  Anchor bar : {}  |  close = ${:.4}",
        anchor_bar.ts.format("%Y-%m-%d %H:%M:%S UTC"), anchor);
    println!("  Window     : {} bars  ({:.1}h of 1-min data)",
        window.len(), window.len() as f64 / 60.0);
    println!();

    let live_ind = compute_indicators(window);

    // ── Cascade inference ─────────────────────────────────────────────────────
    let scout_map = scout.predict(window, &live_ind, &[], anchor);
    let s_val     = scout_map.get(&10).map(|&(p,_)| p).unwrap_or(0.0);

    let sp_map = spotter.predict(window, &live_ind, &[s_val], anchor);
    let sp_v   = [
        sp_map.get(&1) .map(|&(p,_)| p).unwrap_or(0.0),
        sp_map.get(&5) .map(|&(p,_)| p).unwrap_or(0.0),
        sp_map.get(&10).map(|&(p,_)| p).unwrap_or(0.0),
    ];

    let sn_map = sniper.predict(window, &live_ind, &[s_val, sp_v[0], sp_v[1], sp_v[2]], anchor);

    let mut sniper_pct = [0.0f64; 10];
    for m in 1..=10 {
        sniper_pct[m-1] = sn_map.get(&m).map(|&(p,_)| p.clamp(-0.05, 0.05)).unwrap_or(0.0);
    }

    // ── Familiarity (k-NN) — computed once, same for all minutes ─────────────
    let familiarity = knn.familiarity(&live_ind, knn_k);

    // ── Per-minute stability + confidence ────────────────────────────────────
    // Build the sniper input once, then run jitter for each minute's target index.
    let cascade = vec![s_val, sp_v[0], sp_v[1], sp_v[2]];
    let sniper_inp = sniper.build_input(window, &live_ind, &cascade);

    let mut confidences = [0.0f64; 10];
    let mut stabilities = [0.0f64; 10];

    // Scale noise relative to output magnitude so jitter is meaningful regardless
    // of how large or small the predictions are. Use 20% of the mean absolute output
    // as the noise level, floored at noise_std so the flag still has effect.
    let mean_abs_output = sniper_pct.iter().map(|p| p.abs()).sum::<f64>() / 10.0;
    let effective_noise = (mean_abs_output * 0.20).max(noise_std);

    // jitter_passes_data[m][pass] = raw prediction for minute m on noisy pass
    let mut jitter_passes_data: Vec<Vec<f64>> = Vec::with_capacity(10);
    for m in 1..=10 {
        let (stab, pass_pcts) = sniper.stability(&sniper_inp, m - 1, jitter_passes, effective_noise);
        stabilities[m-1] = stab;
        confidences[m-1] = confidence(stab, familiarity);
        jitter_passes_data.push(pass_pcts);
    }

    // ── Results table ─────────────────────────────────────────────────────────
    println!("━━━ Prediction from {} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        anchor_bar.ts.format("%Y-%m-%d %H:%M:%S"));

    let fmt_pred = |map: &HashMap<usize,(f64,f64)>, m: usize| -> String {
        map.get(&m).map(|(p,d)| format!("{:>+6.3}% / ${:<8.4}", p*100.0, d))
           .unwrap_or_else(|| "       --         ".into())
    };

    let has_actual = !future.is_empty();

    // Show up to 10 jitter passes as columns
    let show_passes = jitter_passes.min(10);
    let clean_dir   = |pct: f64| if pct >= 0.0 { "▲" } else { "▼" };

    // Build header — Actual always before jitter columns when available
    let pass_header: String = (0..show_passes).map(|i| format!(" J{:<2}", i+1)).collect();
    if has_actual {
        println!("  Min | Sniper prediction   | Confidence             | Dir | Actual               |{}", pass_header);
        println!("  ----|---------------------|------------------------|-----|----------------------|{}",
            "-".repeat(show_passes * 4));
    } else {
        println!("  Min | Sniper prediction   | Confidence             | Dir |{}", pass_header);
        println!("  ----|---------------------|------------------------|-----|{}",
            "-".repeat(show_passes * 4));
    }

    for m in 1..=10 {
        let conf  = confidences[m-1];
        let label = confidence_label(conf);
        let bar   = confidence_bar(conf);
        let dir   = clean_dir(sniper_pct[m-1]);

        // Jitter pass columns: filled arrow = agrees with clean, hollow = disagrees
        let clean_bull = sniper_pct[m-1] >= 0.0;
        let pass_cols: String = jitter_passes_data[m-1][..show_passes].iter().map(|&p| {
            let agrees = (p >= 0.0) == clean_bull;
            format!(" {} ", if agrees { clean_dir(p) } else { if p >= 0.0 { "△" } else { "▽" } })
        }).collect::<Vec<_>>().join("|");

        if has_actual {
            let actual = if m <= future.len() {
                let pct = (future[m-1].close - anchor) / anchor;
                let correct = (sniper_pct[m-1] >= 0.0) == (pct >= 0.0);
                format!("{:>+6.3}% / ${:<8.4} {}", pct*100.0, future[m-1].close,
                    if correct {"✓"} else {"✗"})
            } else { "  (no data)        ".into() };
            println!("  {:>2}  | {:<19} | {} {} {:.0}% | {} | {:<20} |{}",
                m, fmt_pred(&sn_map, m), bar, label, conf*100.0, dir, actual, pass_cols);
        } else {
            println!("  {:>2}  | {:<19} | {} {} {:.0}% | {} |{}",
                m, fmt_pred(&sn_map, m), bar, label, conf*100.0, dir, pass_cols);
        }
    }

    // ── Accuracy score (--at mode only) ──────────────────────────────────────
    if has_actual && future.len() >= 10 {
        let correct: usize = (1..=10).filter(|&m| {
            let pct = (future[m-1].close - anchor) / anchor;
            (sniper_pct[m-1] >= 0.0) == (pct >= 0.0)
        }).count();
        println!();
        println!("  Directional accuracy (Sniper): {}/10  ({:.0}%)", correct, correct as f64 * 10.0);

        let high_conf_correct = (1..=10).filter(|&m| {
            confidences[m-1] >= 0.40 && {
                let pct = (future[m-1].close - anchor) / anchor;
                (sniper_pct[m-1] >= 0.0) == (pct >= 0.0)
            }
        }).count();
        let high_conf_total = (1..=10).filter(|&m| confidences[m-1] >= 0.40).count();
        if high_conf_total > 0 {
            println!("  MEDIUM/HIGH confidence calls: {}/{}  ({:.0}%)",
                high_conf_correct, high_conf_total,
                high_conf_correct as f64 / high_conf_total as f64 * 100.0);
        }
    }

    // ── Confidence breakdown ──────────────────────────────────────────────────
    let avg_stab = stabilities.iter().sum::<f64>() / 10.0;
    println!();
    println!("  Confidence factors (minute 1):");
    println!("    Stability   : {:.1}%  ({}/{} noisy passes agreed on direction, noise={:.5})",
        stabilities[0] * 100.0, (stabilities[0] * jitter_passes as f64).round() as usize, jitter_passes, effective_noise);
    println!("    Familiarity : {:.1}%  (k-NN distance to training data)",
        familiarity * 100.0);
    println!("    Combined    : {:.1}%  (stability × familiarity)",
        confidences[0] * 100.0);

    // ── Consensus ─────────────────────────────────────────────────────────────
    let bullish   = sniper_pct.iter().filter(|&&p| p > 0.0).count();
    let avg_pct   = sniper_pct.iter().sum::<f64>() / 10.0;
    let avg_conf  = confidences.iter().sum::<f64>() / 10.0;
    let p10_price = sn_map.get(&10).map(|&(_,d)| d).unwrap_or(anchor);
    let direction = if avg_pct >= 0.0 { "▲ BULLISH" } else { "▼ BEARISH" };

    println!();
    println!("  ─────────────────────────────────────────────────");
    println!("  Consensus   : {}  ({}/10 bars agree)", direction, bullish.max(10-bullish));
    println!("  Avg Δ       : {:>+.4}%  over next 10 minutes", avg_pct*100.0);
    println!("  Avg conf    : {:.0}%  {}  {}  (stab {:.0}% × fam {:.0}%)",
        avg_conf*100.0, confidence_label(avg_conf), confidence_bar(avg_conf),
        avg_stab*100.0, familiarity*100.0);
    println!("  +10m target : ${:.4}  ({:>+.4}%)",
        p10_price, (p10_price-anchor)/anchor*100.0);

    let correct = if has_actual && future.len() >= 10 {
        (1..=10).filter(|&m| {
            let pct = (future[m-1].close - anchor) / anchor;
            (sniper_pct[m-1] >= 0.0) == (pct >= 0.0)
        }).count()
    } else { 0 };
    let total = if has_actual { future.len().min(10) } else { 0 };
    (correct, total)
}

// ─────────────────────────────────────────────
//  Date-range backtesting helpers
// ─────────────────────────────────────────────

/// One day's worth of results from the date-range backtest.
#[derive(Clone)]
struct DayResult {
    date:          NaiveDate,
    anchor_price:  f64,
    correct:       usize,          // directional wins out of `total`
    total:         usize,          // minutes with actual data (≤10)
    avg_conf:      f64,            // average sniper confidence
    consensus_dir: bool,           // true = bullish consensus
    avg_pct:       f64,            // predicted avg % change over 10 min
    actual_pct_10: Option<f64>,    // actual % change at minute 10
}

/// Run prediction silently for one anchor bar, return a DayResult.
/// Mirrors run_prediction but suppresses all terminal output.
fn run_day_quiet(
    window:        &[StockData],
    scout:         &Net,
    spotter:       &Net,
    sniper:        &Net,
    future:        &[StockData],
    knn:           &KnnIndex,
    jitter_passes: usize,
    knn_k:         usize,
    noise_std:     f64,
    date:          NaiveDate,
) -> Option<DayResult> {
    if window.len() < 2 { return None; }

    let anchor     = window.last().unwrap().close;
    let live_ind   = compute_indicators(window);

    let scout_map  = scout.predict(window, &live_ind, &[], anchor);
    let s_val      = scout_map.get(&10).map(|&(p,_)| p).unwrap_or(0.0);

    let sp_map     = spotter.predict(window, &live_ind, &[s_val], anchor);
    let sp_v       = [
        sp_map.get(&1) .map(|&(p,_)| p).unwrap_or(0.0),
        sp_map.get(&5) .map(|&(p,_)| p).unwrap_or(0.0),
        sp_map.get(&10).map(|&(p,_)| p).unwrap_or(0.0),
    ];

    let sn_map     = sniper.predict(window, &live_ind, &[s_val, sp_v[0], sp_v[1], sp_v[2]], anchor);

    let mut sniper_pct = [0.0f64; 10];
    for m in 1..=10 {
        sniper_pct[m-1] = sn_map.get(&m).map(|&(p,_)| p.clamp(-0.05, 0.05)).unwrap_or(0.0);
    }

    let familiarity  = knn.familiarity(&live_ind, knn_k);
    let cascade      = vec![s_val, sp_v[0], sp_v[1], sp_v[2]];
    let sniper_inp   = sniper.build_input(window, &live_ind, &cascade);
    let mean_abs     = sniper_pct.iter().map(|p| p.abs()).sum::<f64>() / 10.0;
    let eff_noise    = (mean_abs * 0.20).max(noise_std);

    let mut confidences = [0.0f64; 10];
    for m in 1..=10 {
        let (stab, _) = sniper.stability(&sniper_inp, m-1, jitter_passes, eff_noise);
        confidences[m-1] = confidence(stab, familiarity);
    }

    let has_actual  = !future.is_empty();
    let correct     = if has_actual {
        (1..=10.min(future.len())).filter(|&m| {
            let pct = (future[m-1].close - anchor) / anchor;
            (sniper_pct[m-1] >= 0.0) == (pct >= 0.0)
        }).count()
    } else { 0 };
    let total       = if has_actual { future.len().min(10) } else { 0 };
    let avg_conf    = confidences.iter().sum::<f64>() / 10.0;
    let avg_pct     = sniper_pct.iter().sum::<f64>() / 10.0;
    let consensus   = avg_pct >= 0.0;
    let actual_10   = if future.len() >= 10 {
        Some((future[9].close - anchor) / anchor)
    } else { None };

    Some(DayResult {
        date, anchor_price: anchor,
        correct, total, avg_conf, consensus_dir: consensus,
        avg_pct, actual_pct_10: actual_10,
    })
}

/// Run full verbose prediction for a single day, printing the per-minute table
/// with jitter columns and real values. Used for best/worst day drill-down.
fn run_day_verbose(
    all_bars:      &[StockData],
    scout:         &Net,
    spotter:       &Net,
    sniper:        &Net,
    knn:           &KnnIndex,
    cfg:           &Config,
    date:          NaiveDate,
    anchor_time_h: u32,
    anchor_time_m: u32,
    jitter_passes: usize,
    knn_k:         usize,
    noise_std:     f64,
    interval_mins: usize,
    label:         &str,
) {
    let anchor_ndt = date.and_hms_opt(anchor_time_h, anchor_time_m, 0)
        .unwrap_or_else(|| date.and_hms_opt(17, 0, 0).unwrap());
    let anchor_utc: DateTime<Utc> = Utc.from_utc_datetime(&anchor_ndt);

    let anchor_idx = match all_bars.iter().rposition(|b| b.ts <= anchor_utc) {
        Some(i) => i,
        None    => { println!("  No bar found for {} at {:02}:{:02} UTC", date, anchor_time_h, anchor_time_m); return; }
    };
    if all_bars[anchor_idx].ts.date_naive() != date {
        println!("  No trading data for {} — skipping verbose output.", date);
        return;
    }

    let win_start = anchor_idx.saturating_sub(cfg.lookback - 1);
    let window: Vec<StockData> = all_bars[win_start..=anchor_idx].to_vec();
    let future: Vec<StockData> = all_bars[anchor_idx+1..]
        .iter()
        .take_while(|b| b.ts.date_naive() == date)
        .take(10)
        .cloned()
        .collect();

    if window.len() < 2 { println!("  Window too small for {}", date); return; }

    let anchor_bar = window.last().unwrap();
    let anchor     = anchor_bar.close;
    let live_ind   = compute_indicators(&window);

    let scout_map  = scout.predict(&window, &live_ind, &[], anchor);
    let s_val      = scout_map.get(&10).map(|&(p,_)| p).unwrap_or(0.0);
    let sp_map     = spotter.predict(&window, &live_ind, &[s_val], anchor);
    let sp_v = [
        sp_map.get(&1) .map(|&(p,_)| p).unwrap_or(0.0),
        sp_map.get(&5) .map(|&(p,_)| p).unwrap_or(0.0),
        sp_map.get(&10).map(|&(p,_)| p).unwrap_or(0.0),
    ];
    let sn_map = sniper.predict(&window, &live_ind, &[s_val, sp_v[0], sp_v[1], sp_v[2]], anchor);

    let mut sniper_pct = [0.0f64; 10];
    for m in 1..=10 {
        sniper_pct[m-1] = sn_map.get(&m).map(|&(p,_)| p.clamp(-0.05, 0.05)).unwrap_or(0.0);
    }

    let familiarity  = knn.familiarity(&live_ind, knn_k);
    let cascade      = vec![s_val, sp_v[0], sp_v[1], sp_v[2]];
    let sniper_inp   = sniper.build_input(&window, &live_ind, &cascade);
    let mean_abs     = sniper_pct.iter().map(|p| p.abs()).sum::<f64>() / 10.0;
    let eff_noise    = (mean_abs * 0.20).max(noise_std);

    let show_passes  = jitter_passes.min(10);
    let mut confidences      = [0.0f64; 10];
    let mut stabilities      = [0.0f64; 10];
    let mut jitter_data: Vec<Vec<f64>> = Vec::with_capacity(10);

    for m in 1..=10 {
        let (stab, pass_pcts) = sniper.stability(&sniper_inp, m-1, jitter_passes, eff_noise);
        stabilities[m-1]  = stab;
        confidences[m-1]  = confidence(stab, familiarity);
        jitter_data.push(pass_pcts);
    }

    let clean_dir = |pct: f64| if pct >= 0.0 { "▲" } else { "▼" };
    let has_actual = !future.is_empty();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  {}  ║", label);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!("  Date       : {}  |  anchor close = ${:.4}", date, anchor);
    println!("  Anchor bar : {}  UTC", anchor_bar.ts.format("%Y-%m-%d %H:%M:%S"));
    println!("  Familiarity: {:.1}%   Noise used: {:.5}", familiarity * 100.0, eff_noise);
    println!();

    // Build header
    let pass_header: String = (0..show_passes).map(|i| format!(" J{:<2}", i+1)).collect();
    println!("  Min | Sniper prediction    | Conf           | Dir | Actual               |{}", pass_header);
    println!("  ----|---------------------|----------------|-----|----------------------|{}", "-".repeat(show_passes * 4));

    let mut total_correct = 0usize;
    for m in 1..=10 {
        let conf      = confidences[m-1];
        let label_c   = confidence_label(conf);
        let dir       = clean_dir(sniper_pct[m-1]);
        let pred_str  = sn_map.get(&m)
            .map(|(p,d)| format!("{:>+6.3}% / ${:<8.4}", p*100.0, d))
            .unwrap_or_else(|| "       --         ".into());
        let clean_bull = sniper_pct[m-1] >= 0.0;
        let pass_cols: String = jitter_data[m-1][..show_passes].iter().map(|&p| {
            let agrees = (p >= 0.0) == clean_bull;
            format!(" {} ", if agrees { clean_dir(p) } else { if p >= 0.0 { "△" } else { "▽" } })
        }).collect::<Vec<_>>().join("|");

        if has_actual {
            let actual_str = if m <= future.len() {
                let pct     = (future[m-1].close - anchor) / anchor;
                let correct = (sniper_pct[m-1] >= 0.0) == (pct >= 0.0);
                if correct { total_correct += 1; }
                format!("{:>+6.3}% / ${:<8.4} {}", pct*100.0, future[m-1].close,
                    if correct { "✓" } else { "✗" })
            } else { "  (no data)        ".into() };
            println!("  {:>2}  | {:<19} | {} {:.0}% | {} | {:<20} |{}",
                m, pred_str, label_c, conf*100.0, dir, actual_str, pass_cols);
        } else {
            println!("  {:>2}  | {:<19} | {} {:.0}% | {} |{}",
                m, pred_str, label_c, conf*100.0, dir, pass_cols);
        }
    }

    if has_actual {
        println!();
        println!("  Directional accuracy: {}/{}  ({:.0}%)",
            total_correct, future.len().min(10),
            total_correct as f64 / future.len().min(10).max(1) as f64 * 100.0);
    }

    let avg_conf = confidences.iter().sum::<f64>() / 10.0;
    let avg_stab = stabilities.iter().sum::<f64>() / 10.0;
    let avg_pct  = sniper_pct.iter().sum::<f64>() / 10.0;
    let bullish  = sniper_pct.iter().filter(|&&p| p > 0.0).count();
    println!("  Consensus : {}  ({}/10 minutes agree)  |  avg conf {:.0}%  (stab {:.0}% × fam {:.0}%)",
        if avg_pct >= 0.0 { "▲ BULLISH" } else { "▼ BEARISH" },
        bullish.max(10 - bullish),
        avg_conf * 100.0, avg_stab * 100.0, familiarity * 100.0);
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a=="--help" || a=="-h") {
        println!("scan_predict — predict next 10 minutes with confidence scores");
        println!();
        println!("Default behaviour (no extra flags):");
        println!("  Anchors to the LAST BAR in the local <prefix>_data.csv database.");
        println!("  The prefix comes from --out-prefix used at training time.");
        println!("  No API key or internet connection needed.");
        println!();
        println!("Usage:");
        println!("  # Single symbol — last bar:");
        println!("  scan_predict AAPL.weights");
        println!();
        println!("  # Single symbol — specific past moment:");
        println!("  scan_predict AAPL.weights --at \"2025-06-13 14:30:00\"");
        println!();
        println!("  # Date-range backtest — one prediction per trading day:");
        println!("  scan_predict AAPL.weights --from 2026-01-01 --to 2026-03-10");
        println!("  scan_predict AAPL.weights --from 2026-01-01 --to 2026-03-10 --daily-time 14:30");
        println!();
        println!("  # Multi-symbol scan — each symbol loads its own {{SYM}}.weights:");
        println!("  scan_predict AAPL.weights --symbols \"AAPL,MSFT,NVDA,TSLA,GOOGL\" \\");
        println!("      --at \"2026-03-10 17:00:00\" --jitter-passes 10 --knn-k 10");
        println!("  (AAPL.weights is the fallback if {{SYM}}.weights is missing)");
        println!();
        println!("  # Live fetch from Twelve Data:");
        println!("  scan_predict AAPL.weights --symbol AAPL --api-key YOUR_KEY");
        println!();
        println!("Flags:");
        println!("  --symbol SYM       Ticker — defaults to the out-prefix in the weights file");
        println!("  --symbols S1,S2,.. Multi-symbol scan; loads {{SYM}}.weights per symbol");
        println!("  --api-key KEY      Twelve Data API key — only needed for live fetch");
        println!("  --csv PATH         Explicit CSV path — overrides auto-detected database");
        println!("  --at DATETIME      Anchor to this past timestamp instead of the last bar");
        println!("                     Format: \"YYYY-MM-DD HH:MM:SS\"  or  \"YYYY-MM-DD\"");
        println!("  --from DATE        Start of date-range backtest  (YYYY-MM-DD)");
        println!("  --to   DATE        End   of date-range backtest  (YYYY-MM-DD)");
        println!("  --daily-time HH:MM Anchor time for each day in range (default: 17:00)");
        println!("                     Use market-close (17:00 UTC / 13:00 ET) for EOD test");
        println!("                     or e.g. 14:30 for the 30-min-after-open snapshot");
        println!("  --jitter-passes N  Noisy forward passes for stability score (default: 30)");
        println!("  --knn-k N          Nearest neighbours for familiarity scoring (default: 10)");
        println!("  --noise F          Jitter noise magnitude (default: 0.001)");
        println!("  --interval N       Resample 1-min bars to N-min bars (default: 1)");
        std::process::exit(0);
    }

    let weights_path         = args[1].clone();
    let mut symbol           = String::new();
    let mut api_key          = String::new();
    let mut csv_path         = String::new();
    let mut at_str           = String::new();
    let mut jitter_passes: usize = 30;
    let mut knn_k:         usize = 10;
    let mut interval_mins: usize = 1;
    let mut noise_std:     f64   = 0.001;
    let mut symbols_str:   String = String::new();
    let mut from_str:      String = String::new(); // --from YYYY-MM-DD
    let mut to_str:        String = String::new(); // --to   YYYY-MM-DD
    let mut daily_time:    String = "17:00".into(); // --daily-time HH:MM

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--symbol"        => { symbol        = args[i+1].clone();               i += 2; }
            "--api-key"       => { api_key        = args[i+1].clone();               i += 2; }
            "--csv"           => { csv_path       = args[i+1].clone();               i += 2; }
            "--at"            => { at_str         = args[i+1].clone();               i += 2; }
            "--from"          => { from_str       = args[i+1].clone();               i += 2; }
            "--to"            => { to_str         = args[i+1].clone();               i += 2; }
            "--daily-time"    => { daily_time      = args[i+1].clone();              i += 2; }
            "--jitter-passes" => { jitter_passes  = args[i+1].parse().unwrap_or(30); i += 2; }
            "--knn-k"         => { knn_k          = args[i+1].parse().unwrap_or(10); i += 2; }
            "--interval"      => { interval_mins  = args[i+1].parse().unwrap_or(1);  i += 2; }
            "--noise"         => { noise_std      = args[i+1].parse().unwrap_or(0.001); i += 2; }
            "--symbols"       => { symbols_str     = args[i+1].clone();              i += 2; }
            _                 => { i += 1; }
        }
    }

    println!("Loading weights from '{}'...", weights_path);
    let (cfg, scout, spotter, sniper) = load_weights(&weights_path);
    println!("  lookback={} hidden={} layers={} prefix={}",
        cfg.lookback, cfg.hidden, cfg.layers, cfg.out_prefix);

    if symbol.is_empty() { symbol = cfg.out_prefix.clone(); }

    // ── Date-range backtest mode  --from YYYY-MM-DD --to YYYY-MM-DD ──────────
    //
    //  Walks every calendar day in [from, to].  For each day it:
    //    1. Finds the last bar at or before YYYY-MM-DD HH:MM:SS (daily-time UTC).
    //    2. Runs a silent prediction and records correct/total.
    //    3. After the loop, prints a per-day table + aggregate stats.
    //
    //  Trading days with no data (weekends, holidays) are silently skipped.
    //  If you need to download data first, run with --api-key as well.
    if !from_str.is_empty() && !to_str.is_empty() {
        let from_date = NaiveDate::parse_from_str(&from_str, "%Y-%m-%d")
            .unwrap_or_else(|_| panic!("--from must be YYYY-MM-DD, got '{}'", from_str));
        let to_date = NaiveDate::parse_from_str(&to_str, "%Y-%m-%d")
            .unwrap_or_else(|_| panic!("--to must be YYYY-MM-DD, got '{}'", to_str));
        if from_date > to_date {
            eprintln!("--from {} is after --to {} — nothing to do.", from_str, to_str);
            return;
        }

        // Parse daily_time as HH:MM
        let (hh, mm): (u32, u32) = {
            let parts: Vec<&str> = daily_time.splitn(2, ':').collect();
            let h = parts.get(0).and_then(|s| s.parse().ok()).unwrap_or(17u32);
            let m = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0u32);
            (h, m)
        };

        // Load the CSV once
        let src = if !csv_path.is_empty() { csv_path.clone() }
                  else { format!("{}_data.csv", symbol.to_uppercase()) };
        if !std::path::Path::new(&src).exists() {
            eprintln!("No data file '{}' — cannot run date-range backtest.", src);
            return;
        }
        println!("\nLoading data from '{}' for date-range backtest...", src);
        let all_bars = { let r = parse_csv(&src); if interval_mins > 1 { resample(r, interval_mins) } else { r } };
        if all_bars.is_empty() { eprintln!("No bars in '{}'.", src); return; }

        // Build kNN index once
        let knn = {
            let idx = KnnIndex::build(&all_bars, cfg.lookback);
            println!("  {} training vectors indexed for familiarity.\n", idx.vecs.len());
            idx
        };

        println!("━━━ Date-range backtest: {} → {}  (daily anchor {:02}:{:02} UTC) ━━━",
            from_date, to_date, hh, mm);
        println!("  {:<12} | {:>9} | {:>6} | {:>8} | {:>8} | {:>8} | {:<9}",
            "Date", "Close", "Acc", "AvgConf", "Pred%", "Act10m%", "Consensus");
        println!("  {}",  "-".repeat(75));

        let mut results: Vec<DayResult> = Vec::new();
        let mut cur = from_date;

        while cur <= to_date {
            let anchor_ndt = cur.and_hms_opt(hh, mm, 0)
                .unwrap_or_else(|| cur.and_hms_opt(17, 0, 0).unwrap());
            let anchor_utc: DateTime<Utc> = Utc.from_utc_datetime(&anchor_ndt);

            // Find anchor bar
            if let Some(anchor_idx) = all_bars.iter().rposition(|b| b.ts <= anchor_utc) {
                // Only use bars from the same calendar date so we don't bleed across days
                let bar_date = all_bars[anchor_idx].ts.date_naive();
                if bar_date == cur {
                    let win_start = anchor_idx.saturating_sub(cfg.lookback - 1);
                    let window    = all_bars[win_start..=anchor_idx].to_vec();
                    let future: Vec<_> = all_bars[anchor_idx+1..]
                        .iter()
                        .take_while(|b| b.ts.date_naive() == cur)
                        .take(10)
                        .cloned()
                        .collect();

                    if window.len() >= 2 {
                        if let Some(dr) = run_day_quiet(
                            &window, &scout, &spotter, &sniper,
                            &future, &knn,
                            jitter_passes, knn_k, noise_std, cur,
                        ) {
                            let acc_str = if dr.total > 0 {
                                format!("{}/{}", dr.correct, dr.total)
                            } else { "--".into() };
                            let act_str = dr.actual_pct_10
                                .map(|p| format!("{:>+6.3}%", p*100.0))
                                .unwrap_or("   --   ".into());
                            let cons = if dr.consensus_dir { "▲ BULL" } else { "▼ BEAR" };
                            println!("  {:<12} | {:>9.4} | {:>6} | {:>7.1}% | {:>+7.3}% | {} | {}",
                                cur.to_string(),
                                dr.anchor_price,
                                acc_str,
                                dr.avg_conf * 100.0,
                                dr.avg_pct * 100.0,
                                act_str,
                                cons);
                            results.push(dr);
                        }
                    }
                }
            }
            // Advance one calendar day
            cur = cur + ChronoDuration::days(1);
        }

        // ── Aggregate summary ─────────────────────────────────────────────────
        if results.is_empty() {
            println!("\n  No trading days found in range {} → {}.", from_date, to_date);
            return;
        }

        let days_total     = results.len();
        let days_with_data = results.iter().filter(|r| r.total > 0).count();
        let total_correct: usize = results.iter().map(|r| r.correct).sum();
        let total_calls:   usize = results.iter().map(|r| r.total).sum();
        let avg_conf_all:  f64   = results.iter().map(|r| r.avg_conf).sum::<f64>() / days_total as f64;

        // Direction accuracy: was the daily consensus correct about +10m?
        let (dir_correct, dir_total) = results.iter().fold((0usize, 0usize), |(c, t), r| {
            if let Some(act) = r.actual_pct_10 {
                let predicted_bull = r.consensus_dir;
                let actual_bull    = act >= 0.0;
                (c + if predicted_bull == actual_bull { 1 } else { 0 }, t + 1)
            } else { (c, t) }
        });

        // Streak analysis: longest winning / losing streak
        let (mut best_streak, mut worst_streak, mut cur_streak) = (0i32, 0i32, 0i32);
        for r in &results {
            if let Some(act) = r.actual_pct_10 {
                let win = r.consensus_dir == (act >= 0.0);
                cur_streak = if win { (cur_streak + 1).max(1) } else { (cur_streak - 1).min(-1) };
                best_streak  = best_streak.max(cur_streak);
                worst_streak = worst_streak.min(cur_streak);
            }
        }

        // High-conf days only
        let hc_thresh = 0.70;
        let hc_days: Vec<&DayResult> = results.iter().filter(|r| r.avg_conf >= hc_thresh).collect();
        let (hc_correct, hc_total) = hc_days.iter().fold((0usize, 0usize), |(c, t), r| {
            if let Some(act) = r.actual_pct_10 {
                (c + if r.consensus_dir == (act >= 0.0) { 1 } else { 0 }, t + 1)
            } else { (c, t) }
        });

        // ── Per-confidence-band calibration ───────────────────────────────────
        // Answers "when the model said HIGH/MEDIUM/LOW, was it actually right?"
        // A well-trained model should show HIGH > MEDIUM > LOW accuracy.
        // If HIGH confidence calls are below 55% accurate, the confidence metric
        // is miscalibrated (stability × familiarity does not reflect correctness)
        // and the model weights should be retrained before live use.
        struct Band { label: &'static str, lo: f64, hi: f64, correct: usize, total: usize }
        let mut bands = [
            Band { label: "HIGH  (≥70%)", lo: 0.70, hi: 1.01, correct: 0, total: 0 },
            Band { label: "MEDIUM(40-70%)", lo: 0.40, hi: 0.70, correct: 0, total: 0 },
            Band { label: "LOW   (<40%)", lo: 0.00, hi: 0.40, correct: 0, total: 0 },
        ];
        for r in &results {
            if let Some(act) = r.actual_pct_10 {
                let win = r.consensus_dir == (act >= 0.0);
                for b in &mut bands {
                    if r.avg_conf >= b.lo && r.avg_conf < b.hi {
                        b.total += 1;
                        if win { b.correct += 1; }
                    }
                }
            }
        }

        println!();
        println!("━━━ Aggregate Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Period          : {} → {}", from_date, to_date);
        println!("  Trading days    : {}  ({} with full future data)", days_total, days_with_data);
        println!("  Minute accuracy : {}/{} ({:.1}%)  — all 10 minute slots",
            total_correct, total_calls,
            if total_calls > 0 { total_correct as f64 / total_calls as f64 * 100.0 } else { 0.0 });
        println!("  Daily direction : {}/{} ({:.1}%)  — was consensus right about +10m close?",
            dir_correct, dir_total,
            if dir_total > 0 { dir_correct as f64 / dir_total as f64 * 100.0 } else { 0.0 });
        if hc_total > 0 {
        println!("  HIGH conf days  : {}/{} ({:.1}%)  — days with avg conf ≥ {:.0}%",
            hc_correct, hc_total,
            hc_correct as f64 / hc_total as f64 * 100.0,
            hc_thresh * 100.0);
        }
        println!("  Avg confidence  : {:.1}%", avg_conf_all * 100.0);
        println!("  Best streak     : {} consecutive correct daily calls", best_streak.max(0));
        println!("  Worst streak    : {} consecutive wrong  daily calls", worst_streak.abs());

        // ── Confidence calibration report ─────────────────────────────────────
        println!();
        println!("  Confidence calibration (directional acc per band):");
        let mut any_miscal = false;
        for b in &bands {
            if b.total > 0 {
                let acc = b.correct as f64 / b.total as f64 * 100.0;
                let flag = if b.lo >= 0.70 && acc < 55.0 {
                    any_miscal = true;
                    " ⚠  HIGH conf calls are below 55% — model needs retraining"
                } else { "" };
                println!("    {:15} : {}/{} ({:.1}%){}",
                    b.label, b.correct, b.total, acc, flag);
            }
        }
        if any_miscal {
            println!();
            println!("  ⚠  CALIBRATION WARNING: The confidence score (stability × familiarity)");
            println!("     does not correlate with directional accuracy on this backtest window.");
            println!("     This typically means the model overfit to magnitude patterns rather");
            println!("     than direction — retrain with BCE loss (cascade_trainer default).");
        } else if bands[0].total > 0 {
            let high_acc = bands[0].correct as f64 / bands[0].total as f64 * 100.0;
            let low_acc  = if bands[2].total > 0 { bands[2].correct as f64 / bands[2].total as f64 * 100.0 } else { 0.0 };
            if high_acc > low_acc + 5.0 {
                println!("    ✓ Confidence is well-calibrated: HIGH > LOW accuracy.");
            }
        }
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // ── Best & worst day drill-down ───────────────────────────────────────
        // Only consider days where we have full actual data (total == 10).
        let scoreable: Vec<&DayResult> = results.iter().filter(|r| r.total == 10).collect();
        if scoreable.len() >= 2 {
            let best  = scoreable.iter().max_by_key(|r| r.correct).unwrap();
            let worst = scoreable.iter().min_by_key(|r| r.correct).unwrap();

            println!();
            println!("  Best day  : {}  ({}/{} correct)", best.date,  best.correct,  best.total);
            println!("  Worst day : {}  ({}/{} correct)", worst.date, worst.correct, worst.total);
            println!();
            println!("Running full per-minute breakdown for best and worst days...");

            for (day_result, tag) in &[(*best, "BEST DAY"), (*worst, "WORST DAY")] {
                let label = format!(
                    "{:<10}  {}  —  {}/{} correct  ({:.0}% acc)                     ",
                    tag, day_result.date, day_result.correct, day_result.total,
                    day_result.correct as f64 / day_result.total as f64 * 100.0
                );
                run_day_verbose(
                    &all_bars,
                    &scout, &spotter, &sniper,
                    &knn, &cfg,
                    day_result.date,
                    hh, mm,
                    jitter_passes,
                    knn_k,
                    noise_std,
                    interval_mins,
                    &label,
                );
            }
        } else {
            println!("\n  (Need at least 2 days with full 10-minute future data for best/worst drill-down)");
        }

        return;
    }
    // ── End date-range backtest ───────────────────────────────────────────────
    if !symbols_str.is_empty() {
        let sym_list: Vec<&str> = symbols_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();

        let parse_at = NaiveDateTime::parse_from_str(&at_str, "%Y-%m-%d %H:%M:%S")
            .unwrap_or_else(|_| {
                let d = chrono::NaiveDate::parse_from_str(&at_str, "%Y-%m-%d")
                    .expect("--at must be YYYY-MM-DD HH:MM:SS or YYYY-MM-DD");
                d.and_hms_opt(23,59,59).unwrap()
            });
        let at_utc: DateTime<Utc> = Utc.from_utc_datetime(&parse_at);

        let mut total_correct = 0usize;
        let mut total_calls   = 0usize;

        for sym in &sym_list {
            let sym_upper = sym.to_uppercase();
            let src = format!("{}_data.csv", sym_upper);
            println!("\n══════════════════════════════════════════════════════════");
            println!("  Symbol: {}  |  --at {}", sym_upper, at_str);
            println!("══════════════════════════════════════════════════════════");

            // ── Per-symbol weight loading ─────────────────────────────────────
            // Try {SYM}.weights first so each symbol uses its own trained model.
            // Without this, all symbols get AAPL weights and produce identical % predictions.
            let sym_weights = format!("{}.weights", sym_upper);
            let (sym_cfg, sym_scout, sym_spotter, sym_sniper) =
                if std::path::Path::new(&sym_weights).exists() {
                    println!("  Weights : '{}'", sym_weights);
                    load_weights(&sym_weights)
                } else {
                    println!("  ⚠  No {}.weights found — falling back to '{}'", sym_upper, weights_path);
                    println!("     Train per-symbol weights for accurate predictions.");
                    load_weights(&weights_path)
                };

            // ── kNN index built from THIS symbol's own training data ──────────
            // Familiarity should measure "does this look like what the model learned",
            // which means it must reference the same symbol's historical data.
            let training_csv = format!("{}_data.csv", sym_cfg.out_prefix.to_uppercase());
            let sym_knn = if std::path::Path::new(&training_csv).exists() {
                println!("  kNN idx : '{}'", training_csv);
                let all = { let r = parse_csv(&training_csv); if interval_mins > 1 { resample(r, interval_mins) } else { r } };
                let idx = KnnIndex::build(&all, sym_cfg.lookback);
                println!("  {} training vectors indexed.", idx.vecs.len());
                idx
            } else {
                println!("  Warning : '{}' not found — familiarity will default to 0.5", training_csv);
                KnnIndex { vecs: vec![], p90_dist: 1.0 }
            };

            let existed_before = std::path::Path::new(&src).exists();
            ensure_data_for_date(sym, &src, parse_at.date(), &api_key);
            if !existed_before && !api_key.is_empty() {
                println!("  Waiting 10s before next symbol (rate limit)...");
                thread::sleep(Duration::from_secs(10));
            }

            if !std::path::Path::new(&src).exists() {
                eprintln!("  Skipping {} — no data file found at '{}'", sym_upper, src);
                continue;
            }

            println!("Loading historical data from '{}'...", src);
            let all = { let r = parse_csv(&src); if interval_mins > 1 { resample(r, interval_mins) } else { r } };
            if all.is_empty() { eprintln!("  Skipping {} — no bars.", sym_upper); continue; }

            let anchor_idx = match all.iter().rposition(|b| b.ts <= at_utc) {
                Some(i) => i,
                None    => { eprintln!("  Skipping {} — no bar at or before {}", sym_upper, at_str); continue; }
            };

            let at_date   = all[anchor_idx].ts.date_naive();
            let win_start = anchor_idx.saturating_sub(sym_cfg.lookback - 1);
            let window    = all[win_start..=anchor_idx].to_vec();
            let future: Vec<_> = all[anchor_idx+1..]
                .iter()
                .take_while(|b| b.ts.date_naive() == at_date)
                .take(10)
                .cloned()
                .collect();

            let label = format!("HISTORICAL  --at \"{}\"  ({} future bars)", at_str, future.len());

            if window.len() >= 2 {
                let (correct, total) = run_prediction(&window, &sym_scout, &sym_spotter, &sym_sniper,
                    &future, &label, &sym_knn, jitter_passes, knn_k, noise_std);
                total_correct += correct;
                total_calls   += total;
            }
        }

        println!("\n══════════════════════════════════════════════════════════");
        println!("  Multi-symbol test complete: {} symbols at {}", sym_list.len(), at_str);
        if total_calls > 0 {
            println!("  Aggregate directional accuracy: {}/{} ({:.0}%)",
                total_correct, total_calls,
                total_correct as f64 / total_calls as f64 * 100.0);
        }
        println!("══════════════════════════════════════════════════════════\n");
        return;
    }

    let (window, future, mode_label): (Vec<StockData>, Vec<StockData>, String) = if !at_str.is_empty() {

        // ── Historical override ───────────────────────────────────────────────
        let parse_at = NaiveDateTime::parse_from_str(&at_str, "%Y-%m-%d %H:%M:%S")
            .unwrap_or_else(|_| {
                let d = chrono::NaiveDate::parse_from_str(&at_str, "%Y-%m-%d")
                    .expect("--at must be YYYY-MM-DD HH:MM:SS or YYYY-MM-DD");
                d.and_hms_opt(23,59,59).unwrap()
            });
        let at_utc: DateTime<Utc> = Utc.from_utc_datetime(&parse_at);

        let src = if !csv_path.is_empty() { csv_path.clone() }
                  else { format!("{}_data.csv", symbol.to_uppercase()) };

        // Auto-download any missing data for the requested date before loading
        ensure_data_for_date(&symbol, &src, parse_at.date(), &api_key);

        println!("Loading historical data from '{}'...", src);
        let all = { let r = parse_csv(&src); if interval_mins > 1 { resample(r, interval_mins) } else { r } };
        if all.is_empty() { panic!("No bars in '{}'.", src); }

        let anchor_idx = all.iter().rposition(|b| b.ts <= at_utc)
            .unwrap_or_else(|| panic!("No bar at or before {} in '{}'. The date may be a weekend or holiday.", at_str, src));

        let at_date   = all[anchor_idx].ts.date_naive();
        let win_start = anchor_idx.saturating_sub(cfg.lookback - 1);
        let window    = all[win_start..=anchor_idx].to_vec();
        let future: Vec<_> = all[anchor_idx+1..]
            .iter()
            .take_while(|b| b.ts.date_naive() == at_date)
            .take(10)
            .cloned()
            .collect();

        let label = format!("HISTORICAL  --at \"{}\"  ({} future bars for accuracy check)",
            at_str, future.len());
        (window, future, label)

    } else {

        // ── No --at flag: use last available bar ──────────────────────────────
        //
        // Priority:
        //   1. --csv <path>            explicit file, last bar
        //   2. <SYMBOL>_data.csv       auto-detected local database, last bar
        //   3. Twelve Data live fetch  only if --api-key given AND no local file found
        //
        // This means running with no extra flags always works as long as the
        // local <SYMBOL>_data.csv exists (which it does after any training run).

        // Resolve which CSV to try first
        let auto_csv = format!("{}_data.csv", symbol.to_uppercase());
        let resolved_csv = if !csv_path.is_empty() {
            Some(csv_path.clone())
        } else if std::path::Path::new(&auto_csv).exists() {
            println!("  No --csv given — using local database '{}'", auto_csv);
            Some(auto_csv.clone())
        } else {
            None
        };

        if let Some(src) = resolved_csv {
            // ── Read local CSV ────────────────────────────────────────────────
            println!("Loading data from '{}' ...", src);
            let all = { let r = parse_csv(&src); if interval_mins > 1 { resample(r, interval_mins) } else { r } };
            if all.is_empty() { panic!("No bars in '{}'.", src); }

            // Find the last trading day in the file.
            let last_date = all.last().unwrap().ts.date_naive();
            let day_bars: usize = all.iter().filter(|b| b.ts.date_naive() == last_date).count();

            // Live path: anchor to the very last bar so predictions are as current as possible.
            let anchor_idx = all.len() - 1;

            let win_start = anchor_idx.saturating_sub(cfg.lookback - 1);
            let window    = all[win_start..=anchor_idx].to_vec();
            let anchor_ts = window.last().unwrap().ts;

            // No future bars expected in live mode (we're at the tip of the data).
            let future: Vec<_> = vec![];

            let has_future = !future.is_empty();
            println!("  Last trading day   : {}  ({} bars total)", last_date, day_bars);
            println!("  Anchor bar         : {}  (use --at to pick a different time)",
                anchor_ts.format("%Y-%m-%d %H:%M:%S UTC"));
            if has_future {
                println!("  Actuals available  : {} bars after anchor -> accuracy will be shown",
                    future.len());
            }

            let label = format!(
                "LAST TRADING DAY  {}  anchor {}{}",
                last_date,
                anchor_ts.format("%H:%M UTC"),
                if has_future { format!("  [{} actual bars]", future.len()) } else { String::new() },
            );
            (window, future, label)


        } else {
            // ── No local CSV found — try API (fetch_or_load handles the logic) ─
            let fetch_n = (cfg.lookback + 30).min(5000);
            let bars    = fetch_or_load(&symbol, fetch_n, &api_key);
            if bars.is_empty() { panic!("No bars available. Supply --api-key for a live fetch, or use --csv."); }
            let start  = bars.len().saturating_sub(cfg.lookback);
            let window = bars[start..].to_vec();
            let is_live = bars.last()
                .map(|b| (Utc::now() - b.ts).num_seconds() <= 120)
                .unwrap_or(false);
            let label  = format!("{} (anchor {})",
                if is_live { "LIVE" } else { "CACHED" },
                window.last().map(|b| b.ts.format("%Y-%m-%d %H:%M:%S UTC").to_string()).unwrap_or_default());
            (window, vec![], label)
        }
    };

    if window.len() < 2 {
        eprintln!("Window has fewer than 2 bars — cannot predict."); std::process::exit(1);
    }
    if window.len() < cfg.lookback {
        eprintln!("⚠  Window has {} bars but lookback={}. Accuracy may be reduced.",
            window.len(), cfg.lookback);
    }

    // Build k-NN index from full training CSV for familiarity scoring
    let knn_csv = if !csv_path.is_empty() { csv_path.clone() }
                  else { format!("{}_data.csv", symbol.to_uppercase()) };
    let knn = if std::path::Path::new(&knn_csv).exists() {
        println!("Building confidence index from '{}'...", knn_csv);
        let all = { let r = parse_csv(&knn_csv); if interval_mins > 1 { resample(r, interval_mins) } else { r } };
        let idx = KnnIndex::build(&all, cfg.lookback);
        println!("  {} training vectors indexed.", idx.vecs.len());
        idx
    } else {
        println!("Warning: no training CSV found for familiarity scoring — familiarity will be 0.5");
        KnnIndex { vecs: vec![], p90_dist: 1.0 }
    };

    let _ = run_prediction(&window, &scout, &spotter, &sniper, &future, &mode_label, &knn, jitter_passes, knn_k, noise_std);
}
