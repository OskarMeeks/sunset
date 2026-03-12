
use chrono::{DateTime, NaiveDate, Utc};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::Instant;

// ─────────────────────────────────────────────
//  Data Structures
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StockData {
    pub timestamp: DateTime<Utc>,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub volume: u64,
}

pub struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    input_cache: Vec<f64>,
    output_cache: Vec<f64>,
}

pub struct StockTrainer {
    layers: Vec<Layer>,
    lookback: usize,
    horizon: usize,
    learning_rate: f64,
}

// ─────────────────────────────────────────────
//  Technical indicators (18 features)
// ─────────────────────────────────────────────
//
//  Pre-computed signals appended to each sample's raw bar features.
//  Groups: Trend (EMA), Momentum (RSI, MACD), Volatility (Bollinger, ATR),
//          Structure (pivot S/R), VWAP, Volume (OBV, vol ratio), Candle body.

const INDICATOR_NF: usize = 18;

fn td_ema(closes: &[f64], period: usize) -> f64 {
    if closes.is_empty() { return 0.0; }
    let k = 2.0 / (period as f64 + 1.0);
    closes.iter().skip(1).fold(closes[0], |e, &c| c * k + e * (1.0 - k))
}

fn td_rsi(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 { return 0.5; }
    let w = &closes[closes.len() - period - 1..];
    let (mut ag, mut al) = (0.0_f64, 0.0_f64);
    for s in w.windows(2) { let d = s[1] - s[0]; if d > 0.0 { ag += d; } else { al += d.abs(); } }
    ag /= period as f64; al /= period as f64;
    if al < 1e-10 { return 1.0; }
    let rs = ag / al; (rs / (1.0 + rs)).clamp(0.0, 1.0)
}

fn td_macd(closes: &[f64]) -> (f64, f64) {
    if closes.len() < 26 { return (0.0, 0.0); }
    let line   = td_ema(closes, 12) - td_ema(closes, 26);
    let n      = closes.len();
    let series: Vec<f64> = (9..=n).map(|e| td_ema(&closes[..e], 12) - td_ema(&closes[..e], 26)).collect();
    let signal = td_ema(&series, 9);
    let hist   = (line - signal) / closes.last().unwrap_or(&1.0).abs().max(1e-8);
    (hist.clamp(-0.01, 0.01) / 0.01, signal.signum())
}

fn td_bollinger(closes: &[f64], period: usize) -> (f64, f64) {
    if closes.len() < period { return (0.0, 0.0); }
    let w    = &closes[closes.len() - period..];
    let mean = w.iter().sum::<f64>() / period as f64;
    let std  = (w.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / period as f64).sqrt().max(1e-8);
    let c    = *closes.last().unwrap();
    let pb   = ((c - (mean - 2.0 * std)) / (4.0 * std).max(1e-8)).clamp(0.0, 1.0) * 2.0 - 1.0;
    let bw   = (4.0 * std / mean.abs().max(1e-8)).clamp(0.0, 0.1) / 0.1;
    (pb, bw)
}

fn td_atr_ratio(data: &[StockData], period: usize) -> f64 {
    if data.len() < period + 1 { return 0.0; }
    let trs: Vec<f64> = data.windows(2).map(|w| {
        let tr = (w[1].high - w[1].low)
            .max((w[1].high - w[0].close).abs())
            .max((w[1].low  - w[0].close).abs());
        tr / w[1].close.max(1e-8)
    }).collect();
    let avg  = trs[trs.len().saturating_sub(period)..].iter().sum::<f64>() / period.min(trs.len()) as f64;
    ((trs.last().unwrap_or(&0.0) / avg.max(1e-8)).clamp(0.0, 5.0) / 5.0) * 2.0 - 1.0
}

fn td_pivot_sr(data: &[StockData]) -> (f64, f64, f64) {
    if data.len() < 5 { return (0.0, 0.0, 0.0); }
    let close = data.last().unwrap().close;
    let (mut sup, mut res): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for i in 2..data.len() - 2 {
        let (h, l) = (data[i].high, data[i].low);
        if h > data[i-1].high && h > data[i-2].high && h > data[i+1].high && h > data[i+2].high { res.push(h); }
        if l < data[i-1].low  && l < data[i-2].low  && l < data[i+1].low  && l < data[i+2].low  { sup.push(l); }
    }
    let sd = sup.iter().filter(|&&s| s < close).map(|&s| (close - s) / close).fold(f64::MAX, f64::min);
    let rd = res.iter().filter(|&&r| r > close).map(|&r| (r - close) / close).fold(f64::MAX, f64::min);
    let sn = if sd == f64::MAX { -1.0 } else { (1.0 - sd.clamp(0.0, 0.05) / 0.05) * 2.0 - 1.0 };
    let rn = if rd == f64::MAX {  1.0 } else { (1.0 - rd.clamp(0.0, 0.05) / 0.05) * 2.0 - 1.0 };
    let all: Vec<f64> = sup.iter().chain(res.iter()).copied().collect();
    let cl = if all.is_empty() { 0.0 } else {
        (all.iter().filter(|&&p| ((p - close) / close).abs() < 0.005).count() as f64 / all.len() as f64).clamp(0.0, 1.0)
    };
    (sn, rn, cl)
}

fn td_vwap(data: &[StockData]) -> (f64, f64) {
    if data.is_empty() { return (0.0, 0.0); }
    let (mut cpv, mut cv) = (0.0, 0.0);
    let vwaps: Vec<f64> = data.iter().map(|b| {
        cpv += (b.high + b.low + b.close) / 3.0 * (b.volume as f64).max(1.0);
        cv  += (b.volume as f64).max(1.0); cpv / cv
    }).collect();
    let vw = *vwaps.last().unwrap();
    let c  = data.last().unwrap().close;
    let dist  = ((c - vw) / c.max(1e-8)).clamp(-0.05, 0.05) / 0.05;
    let n     = vwaps.len();
    let slope = if n >= 5 { ((vwaps[n-1] - vwaps[n-5]) / vwaps[n-5].abs().max(1e-8)).clamp(-0.02, 0.02) / 0.02 } else { 0.0 };
    (dist, slope)
}

fn td_obv(data: &[StockData]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let (mut v, mut scale) = (0.0_f64, 0.0_f64);
    let mut obvs = vec![0.0_f64; data.len()];
    for i in 1..data.len() {
        let d = data[i].close - data[i-1].close;
        v += if d > 0.0 { data[i].volume as f64 } else if d < 0.0 { -(data[i].volume as f64) } else { 0.0 };
        obvs[i] = v; scale = scale.max(v.abs());
    }
    let n = obvs.len(); let lb = 10.min(n - 1);
    ((obvs[n-1] - obvs[n-1-lb]) / scale.max(1.0)).clamp(-1.0, 1.0)
}

fn td_compute(data: &[StockData]) -> [f64; INDICATOR_NF] {
    let closes: Vec<f64> = data.iter().map(|b| b.close).collect();
    let c   = *closes.last().unwrap_or(&1.0);
    let e9  = td_ema(&closes, 9);
    let e21 = td_ema(&closes, 21);
    let e9d  = ((c - e9)  / c.max(1e-8)).clamp(-0.05, 0.05) / 0.05;
    let e21d = ((c - e21) / c.max(1e-8)).clamp(-0.05, 0.05) / 0.05;
    let e9sl = if closes.len() >= 2 {
        let now = td_ema(&closes, 9);
        let prev= td_ema(&closes[..closes.len()-1], 9);
        ((now - prev) / prev.abs().max(1e-8)).clamp(-0.02, 0.02) / 0.02
    } else { 0.0 };
    let cross      = if e9 > e21 { 1.0 } else { -1.0 };
    let rsi14      = td_rsi(&closes, 14) * 2.0 - 1.0;
    let (mh, ms)   = td_macd(&closes);
    let (pb, bw)   = td_bollinger(&closes, 20);
    let atr_r      = td_atr_ratio(data, 14);
    let (sup,res,sr) = td_pivot_sr(data);
    let (vd, vs)   = td_vwap(data);
    let obv        = td_obv(data);
    let w          = &data[data.len().saturating_sub(20)..];
    let avg_vol    = w.iter().map(|b| b.volume as f64).sum::<f64>() / w.len().max(1) as f64;
    let vol_r      = (data.last().unwrap().volume as f64 / avg_vol.max(1.0)).clamp(0.0, 5.0) / 5.0 * 2.0 - 1.0;
    let bar        = data.last().unwrap();
    let body       = ((bar.close - bar.open) / (bar.high - bar.low).max(1e-8)).clamp(-1.0, 1.0);
    // Note: train_multi_day StockData doesn't have open; approximate body as close momentum
    let _ = body; // suppress unused if open missing — use close momentum instead
    let body_approx = if data.len() >= 2 { ((data.last().unwrap().close - data[data.len()-2].close) / data[data.len()-2].close.max(1e-8)).clamp(-0.05, 0.05) / 0.05 } else { 0.0 };
    [e9d, e21d, e9sl, cross, rsi14, mh, ms, pb, bw, atr_r, sup, res, sr, vd, vs, obv, vol_r, body_approx]
}

// ─────────────────────────────────────────────
//  The Neural Brain Implementation
// ─────────────────────────────────────────────

impl StockTrainer {
    pub fn new(num_hidden: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let (lookback, horizon) = (60, 10);
        let input_size = lookback * 3 + INDICATOR_NF;  // bar features + 18 indicator signals

        let mut layers = Vec::new();
        let mut dims = vec![input_size];
        for _ in 0..num_hidden { dims.push(hidden_size); }
        dims.push(horizon);

        for i in 0..dims.len() - 1 {
            let (in_d, out_d) = (dims[i], dims[i + 1]);
            let limit = (6.0 / (in_d + out_d) as f64).sqrt();
            let weights = (0..in_d)
                .map(|_| (0..out_d).map(|_| rng.gen_range(-limit..limit)).collect())
                .collect();
            layers.push(Layer {
                weights,
                biases: vec![0.0; out_d],
                input_cache: vec![0.0; in_d],
                output_cache: vec![0.0; out_d],
            });
        }
        StockTrainer { layers, lookback, horizon, learning_rate: 0.0003 }
    }

    /// Update the learning rate — called each generation from the warmup schedule.
    pub fn set_lr(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn activate(x: f64) -> f64 { x.tanh() }
    fn activate_deriv(x: f64) -> f64 { 1.0 - x.powi(2) }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        let n_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.input_cache = current.clone();
            let mut next = vec![0.0; layer.biases.len()];
            for j in 0..layer.biases.len() {
                let sum: f64 = layer.biases[j] + layer.weights.iter()
                    .zip(&layer.input_cache).map(|(w, x)| w[j] * x).sum::<f64>();
                next[j] = if i == n_layers - 1 { sum } else { Self::activate(sum) };
            }
            layer.output_cache = next.clone();
            current = next;
        }
        current
    }

    pub fn train_one_pass(&mut self, samples: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        let n_layers = self.layers.len();
        let mut total_err = 0.0;
        for (input, targets) in samples {
            let preds = self.forward(input);
            let mut deltas: Vec<f64> = preds.iter().zip(targets).map(|(p, t)| p - t).collect();
            total_err += deltas.iter().map(|d| d * d).sum::<f64>() / targets.len() as f64;

            for i in (0..n_layers).rev() {
                let mut next_deltas = vec![0.0; self.layers[i].weights.len()];
                if i > 0 {
                    for j in 0..self.layers[i].weights.len() {
                        let sum: f64 = (0..deltas.len()).map(|k| deltas[k] * self.layers[i].weights[j][k]).sum();
                        next_deltas[j] = sum * Self::activate_deriv(self.layers[i].input_cache[j]);
                    }
                }
                let layer = &mut self.layers[i];
                for j in 0..layer.weights.len() {
                    for k in 0..layer.weights[0].len() {
                        layer.weights[j][k] -= (self.learning_rate * deltas[k] * layer.input_cache[j]).clamp(-0.01, 0.01);
                    }
                }
                for k in 0..layer.biases.len() {
                    layer.biases[k] -= (self.learning_rate * deltas[k]).clamp(-0.01, 0.01);
                }
                deltas = next_deltas;
            }
        }
        total_err / samples.len() as f64
    }

    /// Returns: (MSE, Directional Accuracy %, Average Difference)
    pub fn test(&mut self, test_days: &[(NaiveDate, Vec<StockData>)]) -> (f64, f64, f64) {
        let (mut ok, mut tot) = (0, 0);
        let (mut mse, mut total_diff) = (0.0, 0.0);
        for (_, d) in test_days {
            if let Some(s) = self.build_samples(d) {
                for (inp, tar) in s {
                    let p_all = self.forward(&inp);
                    let p = p_all[0];
                    mse += (p - tar[0]).powi(2);
                    total_diff += (p - tar[0]).abs();
                    if (p > 0.0 && tar[0] > 0.0) || (p < 0.0 && tar[0] < 0.0) { ok += 1; }
                    tot += 1;
                }
            }
        }
        (
            mse / tot.max(1) as f64,
            (ok as f64 / tot.max(1) as f64) * 100.0,
            total_diff / tot.max(1) as f64
        )
    }

    pub fn build_samples_with_indicators(
        &self,
        data: &[StockData],
        indicators: &[[f64; INDICATOR_NF]],
    ) -> Option<Vec<(Vec<f64>, Vec<f64>)>> {
        if data.len() < self.lookback + self.horizon + 1 { return None; }
        let mut samples = Vec::new();
        for i in 1..(data.len() - self.lookback - self.horizon) {
            let mut input = Vec::new();
            for j in 0..self.lookback {
                let (c, p) = (&data[i+j], &data[i+j-1]);
                input.push((c.close - p.close) / p.close.max(1e-6));
                input.push((c.high - c.low) / c.close.max(1e-6));
                input.push((c.volume as f64 - p.volume as f64) / (p.volume as f64).max(1.0));
            }
            // Use precomputed indicator snapshot — index is end of this lookback window
            input.extend_from_slice(&indicators[i + self.lookback - 1]);
            let cur_p = data[i+self.lookback-1].close;
            let tar: Vec<f64> = (0..self.horizon).map(|s| (data[i+self.lookback+s].close - cur_p) / cur_p.max(1e-6)).collect();
            samples.push((input, tar));
        }
        Some(samples)
    }

    pub fn build_samples(&self, data: &[StockData]) -> Option<Vec<(Vec<f64>, Vec<f64>)>> {
        if data.len() < self.lookback + self.horizon + 1 { return None; }
        let mut samples = Vec::new();
        for i in 1..(data.len() - self.lookback - self.horizon) {
            let mut input = Vec::new();
            for j in 0..self.lookback {
                let (c, p) = (&data[i+j], &data[i+j-1]);
                input.push((c.close - p.close) / p.close.max(1e-6));
                input.push((c.high - c.low) / c.close.max(1e-6));
                input.push((c.volume as f64 - p.volume as f64) / (p.volume as f64).max(1.0));
            }
            let indicator_window = &data[i..i + self.lookback];
            input.extend_from_slice(&td_compute(indicator_window));
            let cur_p = data[i+self.lookback-1].close;
            let tar: Vec<f64> = (0..self.horizon).map(|s| (data[i+self.lookback+s].close - cur_p) / cur_p.max(1e-6)).collect();
            samples.push((input, tar));
        }
        Some(samples)
    }
}

// ─────────────────────────────────────────────
//  Main Logic
// ─────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 {
        println!("Usage: cargo run -- <SYMBOL> <START_DATE> <END_DATE> <GENS> <LAYERS>");
        return;
    }

    let symbol = &args[1];
    let gens   = args[4].parse::<usize>().unwrap_or(10);
    let layers = args[5].parse::<usize>().unwrap_or(2);
    let csv_path = format!("{}_data.csv", symbol);

    let file = File::open(&csv_path).expect("CSV file not found");
    let mut raw_data = Vec::new();
    for (i, line) in io::BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; }
        let l = line.unwrap();
        let c: Vec<&str> = l.split(',').collect();
        if c.len() < 7 { continue; }
        let ts = DateTime::parse_from_str(&format!("{} +0000", c[0]), "%Y-%m-%d %H:%M:%S %z")
            .map(|dt| dt.with_timezone(&Utc)).unwrap_or(Utc::now());
        raw_data.push(StockData {
            timestamp: ts, close: c[5].parse().unwrap_or(0.0),
            high: c[3].parse().unwrap_or(0.0), low: c[4].parse().unwrap_or(0.0),
            volume: c[6].parse().unwrap_or(0),
        });
    }

    let mut map: HashMap<NaiveDate, Vec<StockData>> = HashMap::new();
    for row in raw_data { map.entry(row.timestamp.date_naive()).or_default().push(row); }
    let mut days: Vec<_> = map.into_iter().collect();
    days.sort_by_key(|(d, _)| *d);

    let test_days = days.split_off(days.len().saturating_sub(10));
    let mut trainer = StockTrainer::new(layers, 128);

    // ── Precompute indicator snapshots for every day, once before training ───
    // For each day's bar slice, we build a Vec<[f64; INDICATOR_NF]> where
    // entry [i] corresponds to the window ending at bar i of that day.
    // This means build_samples_with_indicators just does an index lookup
    // instead of recomputing EMA/RSI/MACD/etc. every generation.
    println!("[info] Precomputing technical indicators for all days...");
    let t0 = Instant::now();
    let lookback = trainer.lookback;

    let train_indicators: Vec<Vec<[f64; INDICATOR_NF]>> = days.iter()
        .map(|(_, d)| {
            (0..d.len()).map(|i| {
                let start = i.saturating_sub(lookback - 1);
                td_compute(&d[start..=i])
            }).collect()
        })
        .collect();

    let test_indicators: Vec<Vec<[f64; INDICATOR_NF]>> = test_days.iter()
        .map(|(_, d)| {
            (0..d.len()).map(|i| {
                let start = i.saturating_sub(lookback - 1);
                td_compute(&d[start..=i])
            }).collect()
        })
        .collect();

    println!("[info] Indicators ready in {:.2?}\n", t0.elapsed());
    // ─────────────────────────────────────────────────────────────────────────

    // Grab the absolute last window of data for the final report
    let last_day_data = &test_days.last().expect("Not enough data for test set").1;
    let last_day_ind  = test_indicators.last().unwrap();
    let final_sample  = trainer.build_samples_with_indicators(last_day_data, last_day_ind)
        .and_then(|s| s.last().cloned());

    let total_start = Instant::now();

    let base_lr     = 0.0003_f64;
    let lr_decay    = 0.997_f64;
    let warmup_gens = (gens / 10).max(3).min(15); // warmup over first 10% of generations
    println!("[info] LR warmup over first {} generations\n", warmup_gens);

    for g in 1..=gens {
        // ── Warmup schedule ─────────────────────────────────────────────────
        // Ramp from base_lr/10 up to base_lr over warmup_gens, then decay.
        let current_lr = if g <= warmup_gens {
            base_lr * g as f64 / warmup_gens as f64
        } else {
            base_lr * lr_decay.powi((g - warmup_gens) as i32)
        };
        trainer.set_lr(current_lr);
        // ────────────────────────────────────────────────────────────────────

        let gen_start = Instant::now();
        let mut rng = rand::thread_rng();
        let mut train_mse = 0.0;
        let mut count = 0;

        // Shuffle day indices so the order is randomised each generation
        let mut day_indices: Vec<usize> = (0..days.len()).collect();
        day_indices.shuffle(&mut rng);

        for idx in &day_indices {
            let (_, d) = &days[*idx];
            let ind    = &train_indicators[*idx];
            if let Some(s) = trainer.build_samples_with_indicators(d, ind) {
                train_mse += trainer.train_one_pass(&s);
                count += 1;
            }
        }

        // Test using precomputed indicators for test days
        let (test_mse, acc, avg_diff) = {
            let (mut ok, mut tot) = (0, 0);
            let (mut mse, mut total_diff) = (0.0, 0.0);
            for (ti, (_, d)) in test_days.iter().enumerate() {
                if let Some(s) = trainer.build_samples_with_indicators(d, &test_indicators[ti]) {
                    for (inp, tar) in s {
                        let p_all = trainer.forward(&inp);
                        let p = p_all[0];
                        mse += (p - tar[0]).powi(2);
                        total_diff += (p - tar[0]).abs();
                        if (p > 0.0 && tar[0] > 0.0) || (p < 0.0 && tar[0] < 0.0) { ok += 1; }
                        tot += 1;
                    }
                }
            }
            (mse / tot.max(1) as f64, (ok as f64 / tot.max(1) as f64) * 100.0, total_diff / tot.max(1) as f64)
        };
        let elapsed = gen_start.elapsed();
        let eta = elapsed * (gens - g) as u32;
        let phase = if g <= warmup_gens { "warmup" } else { "train " };

        println!("\n━━━ Generation {}/{} [{}] ━━━━━━━━━━━━━━━━━━━━━━━━", g, gens, phase);
        println!("  Metrics      ┃ Train Loss: {:.8} | Test MSE: {:.8}", train_mse / count.max(1) as f64, test_mse);
        println!("  Performance  ┃ Accuracy:   {:>5.2}%     | Avg Diff: {:.8}", acc, avg_diff);
        println!("  Timing       ┃ Gen Time:   {:.2?}      | ETA:      {:.2?} | LR: {:.2e}", elapsed, eta, current_lr);

        if let Some((input, target)) = &final_sample {
            let pred = trainer.forward(input);
            println!("\n  Minute | Predicted % | Actual %  | Error (Abs)");
            println!("  -------|-------------|-----------|------------");
            for i in 0..10 {
                let diff = (pred[i] - target[i]).abs();
                println!("    {:>2}   |   {:>7.4}%  |  {:>7.4}% | {:.6}",
                         i + 1, pred[i] * 100.0, target[i] * 100.0, diff);
            }
        }
    }
    println!("\n[info] Finished in {:.2?}.", total_start.elapsed());
}
