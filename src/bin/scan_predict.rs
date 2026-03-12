// scan_predict.rs
//
// Loads pre-trained cascade weights and predicts the next 10 minutes,
// with a certainty score for every individual prediction.
//
// Certainty is derived from four real signals the model actually produces:
//
//   1. Signal strength   — how far the raw output is from zero.
//                          A prediction of +0.04% is timid; +0.8% is decisive.
//   2. Cascade agreement — Scout, Spotter, and Sniper all pointing the same
//                          direction adds confidence; disagreement reduces it.
//   3. Indicator confluence — how many of the 18 technical indicators agree
//                          with the predicted direction (trend, momentum,
//                          volatility, VWAP, volume, candle structure).
//   4. Time decay        — uncertainty compounds: minute-1 is always more
//                          certain than minute-10.
//
// Each factor is normalised 0-1, then combined into a final score that is
// displayed as a labelled percentage:
//   ≥ 80%  →  HIGH
//   ≥ 55%  →  MEDIUM
//   < 55%  →  LOW
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

use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, BufRead};

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
    out_prefix: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lookback: 60, hidden: 128, layers: 2, batch_size: 256, lr_decay: 0.997,
            epochs1: 300, epochs2: 300, epochs3: 300,
            lr1: 0.001, lr2: 0.001, lr3: 0.001,
            dir_weight: 0.3, out_prefix: "model".into(),
        }
    }
}

// ─────────────────────────────────────────────
//  StockData
// ─────────────────────────────────────────────

#[derive(Clone)]
struct StockData {
    ts:     DateTime<Utc>,
    open:   f64,
    high:   f64,
    low:    f64,
    close:  f64,
    volume: u64,
}

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

fn indicator_confluence(ind: &[f64; INDICATOR_NF], bullish: bool) -> f64 {
    // (index, invert_polarity)  — invert=true means positive value is bearish
    let votes: &[(usize, bool)] = &[
        (0,  false),  // e9d
        (1,  false),  // e21d
        (2,  false),  // e9sl
        (3,  false),  // ema cross
        (4,  false),  // rsi14
        (5,  false),  // macd hist
        (6,  false),  // macd sig
        (7,  false),  // bollinger %B
        (10, false),  // pivot support proximity (higher = more bullish)
        (11, true),   // pivot resistance proximity (higher = closer ceiling = bearish)
        (13, false),  // vwap distance
        (14, false),  // vwap slope
        (15, false),  // obv
        (17, false),  // candle body
    ];

    let total = votes.len() as f64;
    let agreeing = votes.iter().filter(|&&(idx, invert)| {
        let v = if invert { -ind[idx] } else { ind[idx] };
        // A value > 0 means bullish signal
        if bullish { v > 0.0 } else { v < 0.0 }
    }).count() as f64;

    agreeing / total
}

// ─────────────────────────────────────────────
//  Certainty calculation
//
//  Returns a score in [0.0, 1.0] for minute `m` (1-indexed).
//
//  Component weights:
//    signal_strength   35%  — magnitude of the Sniper's raw output
//    cascade_agreement 30%  — Scout / Spotter / Sniper directional consensus
//    ind_confluence    25%  — how many of the 14 directional indicators agree
//    time_decay        10%  — penalty that grows linearly with the forecast horizon
// ─────────────────────────────────────────────

struct CascadeSignals {
    // raw percentage prediction for every minute from each model
    scout_pct:   f64,          // scout only predicts +10; same value for all m
    spotter_pct: [f64; 3],     // [+1, +5, +10]
    sniper_pct:  [f64; 10],    // [+1 .. +10]
}

fn certainty(
    m:        usize,           // minute offset 1-10
    signals:  &CascadeSignals,
    ind:      &[f64; INDICATOR_NF],
) -> f64 {
    let sn_pct = signals.sniper_pct[m - 1];
    let bullish = sn_pct >= 0.0;

    // ── Factor 1: Signal strength (35%) ──────────────────────────────────────
    // Sniper outputs are clamped to ±5% during training.
    // Map |prediction| from [0, 0.05] → [0, 1].
    // Very small predictions (< 0.02%) are near-zero confidence.
    let strength = (sn_pct.abs() / 0.0005).clamp(0.0, 1.0);

    // ── Factor 2: Cascade agreement (30%) ────────────────────────────────────
    // Each model that agrees with the Sniper's direction on this minute adds 1/3.
    // Scout only has a +10m output; for minutes < 10 it still provides directional bias.
    let scout_agrees   = (signals.scout_pct   >= 0.0) == bullish;
    // Spotter: use +1m for m≤2, +5m for m≤6, +10m for m>6
    let spotter_pct = if m <= 2 { signals.spotter_pct[0] }
                      else if m <= 6 { signals.spotter_pct[1] }
                      else            { signals.spotter_pct[2] };
    let spotter_agrees = (spotter_pct >= 0.0) == bullish;
    // Also check the adjacent Sniper minute if available
    let sniper_neighbor = if m > 1 { (signals.sniper_pct[m-2] >= 0.0) == bullish }
                          else if m < 10 { (signals.sniper_pct[m] >= 0.0) == bullish }
                          else { true };

    let agreement_votes = [scout_agrees, spotter_agrees, sniper_neighbor]
        .iter().filter(|&&x| x).count() as f64 / 3.0;

    // ── Factor 3: Indicator confluence (25%) ─────────────────────────────────
    let confluence = indicator_confluence(ind, bullish);

    // ── Factor 4: Time decay (10%) ────────────────────────────────────────────
    // minute 1 → 1.0,  minute 10 → ~0.35
    // Uses a mild exponential decay so near-term stays high
    // but the 10th minute is clearly less certain than the 1st.
    let decay = (-0.11 * (m as f64 - 1.0)).exp();  // e^0=1.0, e^{-0.99}≈0.37

    // ── Weighted combination ───────────────────────────────────────────────────
    let raw = 0.35 * strength
            + 0.30 * agreement_votes
            + 0.25 * confluence
            + 0.10 * decay;

    raw.clamp(0.0, 1.0)
}

fn certainty_label(c: f64) -> &'static str {
    if c >= 0.80 { "HIGH  " }
    else if c >= 0.55 { "MEDIUM" }
    else { "LOW   " }
}

fn certainty_bar(c: f64) -> String {
    let filled = (c * 10.0).round() as usize;
    let bar: String = (0..10).map(|i| if i < filled { '█' } else { '░' }).collect();
    bar
}

// ─────────────────────────────────────────────
//  Per-bar feature extraction  (7 features)
// ─────────────────────────────────────────────

const NF: usize = 7;

#[inline(always)]
fn extract(cur: &StockData, prev: &StockData) -> [f64; NF] {
    let eps=1e-8;
    let pc=prev.close.max(eps); let cc=cur.close.max(eps); let pv=(prev.volume as f64).max(1.0);
    [
        ((cur.close-pc)/pc).clamp(-0.05,0.05),
        ((cur.close-cur.open)/cc).clamp(-0.05,0.05),
        ((cur.high-cc)/cc).clamp(0.0,0.05),
        ((cc-cur.low)/cc).clamp(0.0,0.05),
        ((cur.volume as f64-pv)/pv).clamp(-3.0,3.0),
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

fn parse_csv(path: &str) -> Vec<StockData> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open '{}': {}", path, e));
    let mut rows = Vec::new();
    for (i, line) in io::BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; }
        let l = line.unwrap();
        let c: Vec<&str> = l.split(',').collect();
        if c.len() < 7 { continue; }
        let ts = chrono::DateTime::parse_from_str(&format!("{} +0000", c[0]), "%Y-%m-%d %H:%M:%S %z")
            .map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now());
        rows.push(StockData {
            ts,
            open:   c[1].trim().parse().unwrap_or(0.0),
            high:   c[3].trim().parse().unwrap_or(0.0),
            low:    c[4].trim().parse().unwrap_or(0.0),
            close:  c[5].trim().parse().unwrap_or(0.0),
            volume: c[6].trim().parse().unwrap_or(0),
        });
    }
    rows
}

// ─────────────────────────────────────────────
//  ensure_data_for_date
//
//  Called by the --at path before loading the CSV.
//  Checks whether the requested date is already present in the local CSV.
//  If not, fetches every missing week between the last saved bar (or the
//  requested date itself) and the requested date, saves them, and returns.
//
//  Requires --api-key; if none is provided and data is missing it exits with
//  a clear message.
// ─────────────────────────────────────────────

fn ensure_data_for_date(
    symbol:   &str,
    csv_path: &str,
    need_date: chrono::NaiveDate,
    api_key:  &str,
) {
    use chrono::{Datelike, Duration as CDur};
    use std::collections::HashSet;
    use std::io::Write;

    // ── Load existing timestamps ──────────────────────────────────────────────
    let mut existing_ts: HashSet<String> = HashSet::new();
    let mut last_date: Option<NaiveDate> = None;

    if std::path::Path::new(csv_path).exists() {
        let file = std::fs::File::open(csv_path).expect("Cannot open CSV");
        for (i, line) in io::BufReader::new(file).lines().enumerate() {
            if i == 0 { continue; }
            if let Ok(l) = line {
                if let Some(ts) = l.split(',').next() {
                    let ts = ts.trim().to_string();
                    // Track the latest date we have
                    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&ts, "%Y-%m-%d %H:%M:%S") {
                        let d = dt.date();
                        last_date = Some(last_date.map_or(d, |prev: NaiveDate| prev.max(d)));
                    }
                    existing_ts.insert(ts);
                }
            }
        }
    }

    // ── Check if we already have data for the needed date ────────────────────
    // A trading day is "present" if we have at least one bar on that date.
    let already_have = existing_ts.iter().any(|ts| ts.starts_with(&need_date.to_string()));
    if already_have {
        return; // nothing to do
    }

    // ── We need to download ───────────────────────────────────────────────────
    if api_key.is_empty() {
        eprintln!("Error: no data for {} in '{}' and no --api-key provided.", need_date, csv_path);
        eprintln!("  Pass --api-key so scan_predict can auto-download the missing data.");
        std::process::exit(1);
    }

    // Determine the range of weeks to fetch:
    // from the Monday of the week after our last saved date (or the needed week
    // if we have nothing at all), up to the Monday of the needed week.
    let need_monday = {
        let days_from_mon = need_date.weekday().num_days_from_monday();
        need_date - CDur::days(days_from_mon as i64)
    };

    let start_monday = if let Some(ld) = last_date {
        // If our last bar is on or after the needed date, nothing to fetch
        if ld >= need_date {
            return;
        }
        // Start from the Monday of the week that contains the day after our last bar
        let next_day  = ld + CDur::days(1);
        let days_back = next_day.weekday().num_days_from_monday() as i64;
        next_day - CDur::days(days_back)
    } else {
        need_monday
    };

    // Collect all Mondays in range
    let mut mondays: Vec<NaiveDate> = Vec::new();
    let mut cur = start_monday;
    while cur <= need_monday {
        mondays.push(cur);
        cur += CDur::weeks(1);
    }

    if mondays.is_empty() {
        return;
    }

    println!("  Auto-downloading {} missing week(s) for {} ...", mondays.len(), symbol);

    const RATE_LIMIT_SECS: u64 = 8;
    const POINTS_PER_CALL: usize = 1950;
    const TWELVE_DATA_URL: &str = "https://api.twelvedata.com/time_series";

    for (i, &monday) in mondays.iter().enumerate() {
        let friday    = monday + CDur::days(4);
        let start_str = format!("{} 00:00:00", monday).replace(' ', "%20");
        let end_str   = format!("{} 23:59:59", friday).replace(' ', "%20");

        let url = format!(
            "{}?symbol={}&interval=1min&start_date={}&end_date={}&outputsize={}&apikey={}&format=JSON&timezone=UTC",
            TWELVE_DATA_URL, symbol, start_str, end_str, POINTS_PER_CALL, api_key,
        );

        println!("  Fetching week of {} ...", monday);

        let resp = match reqwest::blocking::get(&url) {
            Ok(r)  => r,
            Err(e) => { eprintln!("  Request failed: {}", e); continue; }
        };
        let json: serde_json::Value = match resp.json() {
            Ok(j)  => j,
            Err(e) => { eprintln!("  JSON parse failed: {}", e); continue; }
        };
        if json["status"] == "error" {
            eprintln!("  API error: {}", json["message"].as_str().unwrap_or("unknown"));
            continue;
        }
        let values = match json["values"].as_array() {
            Some(v) => v,
            None    => { println!("  No data (holiday week?)"); continue; }
        };

        // Parse new rows
        let mut new_rows: Vec<(String, String, String, String, String, String)> = values.iter()
            .filter_map(|v| {
                let ts = v["datetime"].as_str()?.to_string();
                if existing_ts.contains(&ts) { return None; }
                Some((
                    ts,
                    v["open"]  .as_str().unwrap_or("0").to_string(),
                    v["high"]  .as_str().unwrap_or("0").to_string(),
                    v["low"]   .as_str().unwrap_or("0").to_string(),
                    v["close"] .as_str().unwrap_or("0").to_string(),
                    v["volume"].as_str().unwrap_or("0").to_string(),
                ))
            })
            .collect();

        if new_rows.is_empty() {
            println!("  All rows already present.");
            continue;
        }

        new_rows.sort_by(|a, b| a.0.cmp(&b.0));

        // Append to CSV (re-sort whole file)
        let mut all_rows: Vec<Vec<String>> = Vec::new();
        if std::path::Path::new(csv_path).exists() {
            let f = std::fs::File::open(csv_path).expect("Cannot open CSV");
            for (idx, line) in io::BufReader::new(f).lines().enumerate() {
                if idx == 0 { continue; }
                if let Ok(l) = line {
                    let cols: Vec<String> = l.split(',').map(|s| s.trim().to_string()).collect();
                    if cols.len() >= 7 { all_rows.push(cols); }
                }
            }
        }
        for (ts, o, h, l, c, v) in &new_rows {
            existing_ts.insert(ts.clone());
            all_rows.push(vec![ts.clone(), symbol.to_uppercase(), o.clone(), h.clone(), l.clone(), c.clone(), v.clone()]);
        }
        all_rows.sort_by(|a, b| a[0].cmp(&b[0]));

        let mut f = std::fs::File::create(csv_path).expect("Cannot write CSV");
        writeln!(f, "Timestamp,Symbol,Open,High,Low,Close,Volume").unwrap();
        for row in &all_rows { writeln!(f, "{}", row.join(",")).unwrap(); }

        println!("  Saved {} new bars ({} total)", new_rows.len(), all_rows.len());

        // Rate limit between calls
        if i + 1 < mondays.len() {
            println!("  Waiting {}s (rate limit)...", RATE_LIMIT_SECS);
            std::thread::sleep(std::time::Duration::from_secs(RATE_LIMIT_SECS));
        }
    }
}

// ─────────────────────────────────────────────
//  fetch_or_load
//
//  Smart data loader used by the live (no --at) path when no local CSV exists
//  or when the local CSV is stale (last bar is older than 2 minutes ago).
//
//  Strategy:
//    1. If <SYMBOL>_data.csv exists, load it and check how fresh the last bar is.
//       - Fresh enough (≤ 2 min old): return the in-memory bars, no API call.
//       - Stale: call the API to get recent bars, merge with what we have,
//         re-sort, write the updated file back, and return all bars.
//    2. If no file exists at all: full API fetch, save to file, return bars.
//
//  "Need at least `n` bars" is still respected — if the merged result is
//  shorter than n we log a warning but proceed with what we have.
// ─────────────────────────────────────────────

fn fetch_or_load(symbol: &str, n: usize, api_key: &str) -> Vec<StockData> {
    let csv_path = format!("{}_data.csv", symbol.to_uppercase());

    // ── Step 1: try to load existing CSV ─────────────────────────────────────
    let mut existing: Vec<StockData> = if std::path::Path::new(&csv_path).exists() {
        println!("  Found local database '{}'", csv_path);
        let bars = parse_csv(&csv_path);
        println!("  Loaded {} existing bars", bars.len());
        bars
    } else {
        println!("  No local database found for {} — will download from API", symbol);
        vec![]
    };

    // ── Step 2: check freshness ───────────────────────────────────────────────
    let now_utc = Utc::now();
    let stale_threshold_secs = 120; // 2 minutes

    let needs_fetch = if let Some(last) = existing.last() {
        let age = (now_utc - last.ts).num_seconds();
        if age <= stale_threshold_secs {
            println!("  Data is fresh (last bar {} — {}s ago). Skipping API call.",
                last.ts.format("%Y-%m-%d %H:%M:%S UTC"), age);
            false
        } else {
            println!("  Last bar: {}  ({}s ago — stale, fetching updates)",
                last.ts.format("%Y-%m-%d %H:%M:%S UTC"), age);
            true
        }
    } else {
        true // no bars at all
    };

    // ── Step 3: fetch from API if needed ─────────────────────────────────────
    if needs_fetch {
        if api_key.is_empty() {
            if existing.is_empty() {
                panic!("No local data and no --api-key provided. Cannot fetch live data.");
            }
            println!("  No --api-key provided; using stale local data as-is.");
        } else {
            let new_bars = fetch_live_raw(symbol, n, api_key);

            if !new_bars.is_empty() {
                // Merge: collect timestamps already present
                let existing_ts: std::collections::HashSet<String> = existing
                    .iter()
                    .map(|b| b.ts.format("%Y-%m-%d %H:%M:%S").to_string())
                    .collect();

                let mut added = 0usize;
                for bar in new_bars {
                    let ts_key = bar.ts.format("%Y-%m-%d %H:%M:%S").to_string();
                    if !existing_ts.contains(&ts_key) {
                        existing.push(bar);
                        added += 1;
                    }
                }

                existing.sort_by_key(|b| b.ts);
                println!("  Merged {} new bars into local database ({} total)", added, existing.len());

                // ── Step 4: write updated CSV back ───────────────────────────
                save_bars_to_csv(&existing, &csv_path, symbol);
            }
        }
    }

    if existing.len() < n {
        eprintln!("  ⚠  Only {} bars available (need {}). Accuracy may be reduced.",
            existing.len(), n);
    }

    existing
}

/// Appends + rewrites the CSV with the full sorted bar list.
fn save_bars_to_csv(bars: &[StockData], path: &str, symbol: &str) {
    use std::io::Write;
    let mut file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Cannot write '{}': {}", path, e));
    writeln!(file, "Timestamp,Symbol,Open,High,Low,Close,Volume").unwrap();
    for b in bars {
        writeln!(file, "{},{},{:.4},{:.4},{:.4},{:.4},{}",
            b.ts.format("%Y-%m-%d %H:%M:%S"),
            symbol.to_uppercase(),
            b.open, b.high, b.low, b.close, b.volume,
        ).unwrap();
    }
    println!("  Saved {} bars to '{}'", bars.len(), path);
}

/// Raw API fetch — returns bars newest-to-oldest from Twelve Data, sorted oldest-first.
/// Does not touch the filesystem.
fn fetch_live_raw(symbol: &str, n: usize, api_key: &str) -> Vec<StockData> {
    println!("  Fetching {} bars for {} from Twelve Data...", n, symbol);
    let url = format!(
        "https://api.twelvedata.com/time_series\
         ?symbol={}&interval=1min&outputsize={}&apikey={}&format=JSON&timezone=UTC",
        symbol, n, api_key
    );
    let resp = reqwest::blocking::get(&url)
        .unwrap_or_else(|e| panic!("API request failed: {}", e));
    let json: serde_json::Value = resp.json()
        .unwrap_or_else(|e| panic!("Failed to parse API response: {}", e));
    if json["status"] == "error" {
        panic!("Twelve Data API error: {}", json["message"].as_str().unwrap_or("unknown"));
    }
    let values = json["values"].as_array()
        .unwrap_or_else(|| panic!("No 'values' in API response — market may be closed"));
    let mut bars: Vec<StockData> = values.iter().map(|v| {
        let ts_str = v["datetime"].as_str().unwrap_or("");
        let ts = chrono::DateTime::parse_from_str(&format!("{} +0000", ts_str), "%Y-%m-%d %H:%M:%S %z")
            .map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now());
        StockData {
            ts,
            open:   v["open"]  .as_str().unwrap_or("0").parse().unwrap_or(0.0),
            high:   v["high"]  .as_str().unwrap_or("0").parse().unwrap_or(0.0),
            low:    v["low"]   .as_str().unwrap_or("0").parse().unwrap_or(0.0),
            close:  v["close"] .as_str().unwrap_or("0").parse().unwrap_or(0.0),
            volume: v["volume"].as_str().unwrap_or("0").parse().unwrap_or(0),
        }
    }).collect();
    bars.sort_by_key(|b| b.ts);
    println!("  Got {} bars  ({} → {})",
        bars.len(),
        bars.first().map(|b| b.ts.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or_default(),
        bars.last() .map(|b| b.ts.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or_default(),
    );
    bars
}

// ─────────────────────────────────────────────
//  Prediction + display
// ─────────────────────────────────────────────

fn run_prediction(
    window:     &[StockData],
    scout:      &Net,
    spotter:    &Net,
    sniper:     &Net,
    future:     &[StockData],
    mode_label: &str,
) {
    if window.is_empty() { eprintln!("Window is empty — cannot predict."); return; }

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

    // ── Build signals struct for certainty calculation ────────────────────────
    let mut sniper_pct = [0.0f64; 10];
    for m in 1..=10 {
        sniper_pct[m-1] = sn_map.get(&m).map(|&(p,_)| p).unwrap_or(0.0);
    }
    let signals = CascadeSignals {
        scout_pct:   s_val,
        spotter_pct: sp_v,
        sniper_pct,
    };

    // ── Results table ─────────────────────────────────────────────────────────
    println!("━━━ Prediction from {} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        anchor_bar.ts.format("%Y-%m-%d %H:%M:%S"));

    let fmt_pred = |map: &HashMap<usize,(f64,f64)>, m: usize| -> String {
        map.get(&m).map(|(p,d)| format!("{:>+6.3}% / ${:<8.4}", p*100.0, d))
           .unwrap_or_else(|| "       --         ".into())
    };

    let has_actual = !future.is_empty();

    // Header
    if has_actual {
        println!("  Min | Sniper prediction   | Certainty              | Dir | Actual");
        println!("  ----|---------------------|------------------------|-----|------------------");
    } else {
        println!("  Min | Sniper prediction   | Certainty              | Dir");
        println!("  ----|---------------------|------------------------|-----");
    }

    let mut certainties = [0.0f64; 10];
    for m in 1..=10 {
        let cert  = certainty(m, &signals, &live_ind);
        certainties[m-1] = cert;
        let label = certainty_label(cert);
        let bar   = certainty_bar(cert);
        let dir   = if sniper_pct[m-1] >= 0.0 { "▲" } else { "▼" };

        if has_actual {
            let actual = if m <= future.len() {
                let pct = (future[m-1].close - anchor) / anchor;
                let correct = (sniper_pct[m-1] >= 0.0) == (pct >= 0.0);
                format!("{:>+6.3}% / ${:<8.4} {}", pct*100.0, future[m-1].close,
                    if correct {"✓"} else {"✗"})
            } else { "  (no data)".into() };

            println!("  {:>2}  | {:<19} | {} {} {:.0}% | {} | {}",
                m,
                fmt_pred(&sn_map, m),
                bar, label, cert*100.0,
                dir, actual,
            );
        } else {
            println!("  {:>2}  | {:<19} | {} {} {:.0}% | {}",
                m,
                fmt_pred(&sn_map, m),
                bar, label, cert*100.0,
                dir,
            );
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

        // Check whether high-certainty calls were more accurate
        let high_cert_correct = (1..=10).filter(|&m| {
            certainties[m-1] >= 0.55 && {
                let pct = (future[m-1].close - anchor) / anchor;
                (sniper_pct[m-1] >= 0.0) == (pct >= 0.0)
            }
        }).count();
        let high_cert_total = (1..=10).filter(|&m| certainties[m-1] >= 0.55).count();
        if high_cert_total > 0 {
            println!("  MEDIUM/HIGH certainty calls:  {}/{}  ({:.0}%)",
                high_cert_correct, high_cert_total,
                high_cert_correct as f64 / high_cert_total as f64 * 100.0);
        }
    }

    // ── Certainty breakdown legend ────────────────────────────────────────────
    println!();
    println!("  Certainty factors (Sniper, minute 1):");
    let m1_bull = sniper_pct[0] >= 0.0;
    let strength_pct  = (sniper_pct[0].abs() / 0.0005).clamp(0.0,1.0) * 100.0;
    let confluence_pct = indicator_confluence(&live_ind, m1_bull) * 100.0;
    let cascade_agrees = [
        (s_val >= 0.0) == m1_bull,
        (sp_v[0] >= 0.0) == m1_bull,
    ].iter().filter(|&&x| x).count();
    println!("    Signal strength    : {:.1}%  (raw output magnitude)", strength_pct);
    println!("    Cascade agreement  : {}/3 models agree on direction", cascade_agrees + 1);
    println!("    Indicator confluence: {:.1}%  of 14 directional indicators align", confluence_pct);
    println!("    Time decay         : 100% → ~37% from min 1 to min 10");

    // ── Consensus ─────────────────────────────────────────────────────────────
    let bullish   = sniper_pct.iter().filter(|&&p| p > 0.0).count();
    let avg_pct   = sniper_pct.iter().sum::<f64>() / 10.0;
    let avg_cert  = certainties.iter().sum::<f64>() / 10.0;
    let p10_price = sn_map.get(&10).map(|&(_,d)| d).unwrap_or(anchor);
    let direction = if avg_pct >= 0.0 { "▲ BULLISH" } else { "▼ BEARISH" };

    println!();
    println!("  ─────────────────────────────────────────────────");
    println!("  Consensus  : {}  ({}/10 bars agree)", direction, bullish.max(10-bullish));
    println!("  Avg Δ      : {:>+.4}%  over next 10 minutes", avg_pct*100.0);
    println!("  Avg cert   : {:.0}%  {}  {}",
        avg_cert*100.0, certainty_label(avg_cert), certainty_bar(avg_cert));
    println!("  +10m target: ${:.4}  ({:>+.4}%)",
        p10_price, (p10_price-anchor)/anchor*100.0);
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a=="--help" || a=="-h") {
        println!("scan_predict — predict next 10 minutes with certainty scores");
        println!();
        println!("Default behaviour (no extra flags):");
        println!("  Anchors to the LAST BAR in the local <prefix>_data.csv database.");
        println!("  The prefix comes from --out-prefix used at training time.");
        println!("  No API key or internet connection needed.");
        println!();
        println!("Usage:");
        println!("  # Simplest — last bar in local database:");
        println!("  scan_predict AAPL.weights");
        println!();
        println!("  # Explicit CSV:");
        println!("  scan_predict AAPL.weights --csv AAPL_data.csv");
        println!();
        println!("  # Test a specific past moment (shows Actual column + accuracy):");
        println!("  scan_predict AAPL.weights --at \"2025-06-13 14:30:00\"");
        println!("  scan_predict AAPL.weights --csv AAPL_data.csv --at \"2025-06-13 14:30:00\"");
        println!();
        println!("  # True live fetch from Twelve Data right now:");
        println!("  scan_predict AAPL.weights --symbol AAPL --api-key YOUR_KEY");
        println!();
        println!("Flags:");
        println!("  --symbol SYM     Ticker — defaults to the out-prefix in the weights file");
        println!("  --api-key KEY    Twelve Data API key — only needed for live fetch");
        println!("                   (when no local <SYMBOL>_data.csv is found)");
        println!("  --csv PATH       Explicit CSV path — overrides auto-detected database");
        println!("  --at DATETIME    Anchor to this past timestamp instead of the last bar");
        println!("                   Format: \"YYYY-MM-DD HH:MM:SS\"  or  \"YYYY-MM-DD\"");
        std::process::exit(0);
    }

    let weights_path = args[1].clone();
    let mut symbol   = String::new();
    let mut api_key  = String::new();
    let mut csv_path = String::new();
    let mut at_str   = String::new();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--symbol"  => { symbol   = args[i+1].clone(); i += 2; }
            "--api-key" => { api_key  = args[i+1].clone(); i += 2; }
            "--csv"     => { csv_path = args[i+1].clone(); i += 2; }
            "--at"      => { at_str   = args[i+1].clone(); i += 2; }
            _           => { i += 1; }
        }
    }

    println!("Loading weights from '{}'...", weights_path);
    let (cfg, scout, spotter, sniper) = load_weights(&weights_path);
    println!("  lookback={} hidden={} layers={} prefix={}",
        cfg.lookback, cfg.hidden, cfg.layers, cfg.out_prefix);

    if symbol.is_empty() { symbol = cfg.out_prefix.clone(); }

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
        let all = parse_csv(&src);
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
            let all = parse_csv(&src);
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

    run_prediction(&window, &scout, &spotter, &sniper, &future, &mode_label);
}
