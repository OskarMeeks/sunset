// scan_predict.rs
//
// Loads pre-trained cascade weights and predicts the next 10 minutes.
//
// Confidence is based on two signals:
//   1. Prediction stability — run the same input N times with small noise.
//      If the direction stays consistent, the model is decisive.
//   2. Training familiarity — k-NN distance to training feature vectors.
//      Far from training data = lower confidence.
//
//   confidence = stability × familiarity
//   ≥ 70% → HIGH   ≥ 40% → MEDIUM   < 40% → LOW

use chrono::{DateTime, Duration as ChronoDuration, NaiveDate, NaiveDateTime, TimeZone, Utc};
use std::env;
use std::thread;
use std::time::Duration;

use stock_tracker::generator::{ensure_data_for_date, fetch_or_load, parse_csv, StockData};
use stock_tracker::indicators::{bars_from_cascade, compute_indicators, INDICATOR_NF};

use stock_tracker::cascade_core::{resample, Config};
use stock_tracker::cascade_core::net::{Net, load_all_weights};

// ─────────────────────────────────────────────
//  Confidence helpers
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
//  k-NN index over training feature vectors
// ─────────────────────────────────────────────

pub struct KnnIndex {
    vecs:     Vec<[f64; INDICATOR_NF]>,
    p90_dist: f64,
}

impl KnnIndex {
    pub fn build(all_data: &[StockData], lookback: usize) -> Self {
        let n = all_data.len();
        if n < lookback { return KnnIndex { vecs: vec![], p90_dist: 1.0 }; }
        let vecs: Vec<[f64; INDICATOR_NF]> = (lookback..n)
            .map(|i| compute_indicators(&bars_from_cascade(&all_data[i + 1 - lookback..=i])))
            .collect();

        let sample_n = vecs.len().min(500);
        let mut dists: Vec<f64> = Vec::with_capacity(sample_n);
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

    pub fn familiarity(&self, query: &[f64; INDICATOR_NF], k: usize) -> f64 {
        if self.vecs.is_empty() { return 0.5; }
        let mut heap: Vec<f64> = Vec::with_capacity(k + 1);
        for v in &self.vecs {
            let d = l2_dist(query, v);
            if heap.len() < k {
                heap.push(d);
                if heap.len() == k { heap.sort_by(|a, b| b.partial_cmp(a).unwrap()); }
            } else if d < heap[0] {
                heap[0] = d;
                heap.sort_by(|a, b| b.partial_cmp(a).unwrap());
            }
        }
        let avg_knn_dist = heap.iter().sum::<f64>() / heap.len().max(1) as f64;
        let ratio = avg_knn_dist / self.p90_dist;
        (1.0 - ratio * 0.5).clamp(0.0, 1.0)
    }
}

fn l2_dist(a: &[f64; INDICATOR_NF], b: &[f64; INDICATOR_NF]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// ─────────────────────────────────────────────
//  Date-range backtest helpers
// ─────────────────────────────────────────────

#[derive(Clone)]
struct DayResult {
    date:          NaiveDate,
    anchor_price:  f64,
    correct:       usize,
    total:         usize,
    avg_conf:      f64,
    consensus_dir: bool,
    avg_pct:       f64,
    actual_pct_10: Option<f64>,
}

fn run_day_quiet(
    window:        &[StockData],
    sniper:        &Net,
    future:        &[StockData],
    knn:           &KnnIndex,
    jitter_passes: usize,
    knn_k:         usize,
    noise_std:     f64,
    date:          NaiveDate,
    _cfg:          &Config,
) -> Option<DayResult> {
    if window.len() < 2 { return None; }

    let anchor   = window.last().unwrap().close;
    let live_ind = compute_indicators(&bars_from_cascade(window));

    let sn_map = sniper.predict(window, &live_ind, &[], anchor);

    let mut sniper_pct = [0.0f64; 10];
    for m in 1..=10 { sniper_pct[m-1] = sn_map.get(&m).map(|&(p,_)| p).unwrap_or(0.0); }

    let familiarity = knn.familiarity(&live_ind, knn_k);
    let sniper_inp  = sniper.build_input(window, &live_ind, &[]);

    let mut confidences = [0.0f64; 10];
    for m in 1..=10 {
        let (stab, _) = sniper.stability(&sniper_inp, m-1, jitter_passes, noise_std);
        confidences[m-1] = confidence(stab, familiarity);
    }

    let has_actual = !future.is_empty();
    let correct    = if has_actual {
        (1..=10.min(future.len())).filter(|&m| {
            let pct = (future[m-1].close - anchor) / anchor;
            (sniper_pct[m-1] >= 0.0) == (pct >= 0.0)
        }).count()
    } else { 0 };
    let total      = if has_actual { future.len().min(10) } else { 0 };
    let avg_conf   = confidences.iter().sum::<f64>() / 10.0;
    let avg_pct    = sniper_pct.iter().sum::<f64>() / 10.0;
    let actual_10  = if future.len() >= 10 { Some((future[9].close - anchor) / anchor) } else { None };

    Some(DayResult {
        date, anchor_price: anchor,
        correct, total, avg_conf, consensus_dir: avg_pct >= 0.0,
        avg_pct, actual_pct_10: actual_10,
    })
}

fn run_day_verbose(
    all_bars:      &[StockData],
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

    let anchor_idx: usize = match all_bars.iter().rposition(|b| b.ts <= anchor_utc) {
        None      => { println!("  No trading data for {} — skipping verbose output.", date); return; }
        Some(idx) => idx,
    };

    let win_start = anchor_idx.saturating_sub(cfg.lookback - 1);
    let window: Vec<StockData> = all_bars[win_start..=anchor_idx].to_vec();
    let future: Vec<StockData> = all_bars[anchor_idx+1..]
        .iter().take_while(|b| b.ts.date_naive() == date).take(10).cloned().collect();

    if window.len() < 2 { println!("  Window too small for {}", date); return; }

    let anchor_bar = window.last().unwrap();
    let anchor     = anchor_bar.close;
    let live_ind   = compute_indicators(&bars_from_cascade(&window));

    let sn_map = sniper.predict(&window, &live_ind, &[], anchor);

    let mut sniper_pct = [0.0f64; 10];
    for m in 1..=10 { sniper_pct[m-1] = sn_map.get(&m).map(|&(p,_)| p).unwrap_or(0.0); }

    let familiarity = knn.familiarity(&live_ind, knn_k);
    let sniper_inp  = sniper.build_input(&window, &live_ind, &[]);
    let show_passes = jitter_passes.min(10);

    let mut confidences  = [0.0f64; 10];
    let mut stabilities  = [0.0f64; 10];
    let mut jitter_data: Vec<Vec<f64>> = Vec::with_capacity(10);
    for m in 1..=10 {
        let (stab, pass_pcts) = sniper.stability(&sniper_inp, m-1, jitter_passes, noise_std);
        stabilities[m-1]  = stab;
        confidences[m-1]  = confidence(stab, familiarity);
        jitter_data.push(pass_pcts);
    }

    let clean_dir  = |pct: f64| if pct >= 0.0 { "▲" } else { "▼" };
    let has_actual = !future.is_empty();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  {}  ║", label);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!("  Date       : {}  |  anchor close = ${:.4}", date, anchor);
    println!("  Anchor bar : {}  UTC", anchor_bar.ts.format("%Y-%m-%d %H:%M:%S"));
    println!("  Familiarity: {:.1}%   Noise used: {:.5}", familiarity * 100.0, noise_std);
    println!();

    let pass_header: String = (0..show_passes).map(|i| format!(" J{:<2}", i+1)).collect();
    println!("  Min | AI pred%   | Conf           | Dir | Actual (candle%)     |{}", pass_header);
    println!("  ----|-----------|----------------|-----|----------------------|{}", "-".repeat(show_passes * 4));

    let mut total_correct = 0usize;
    for m in 1..=10 {
        let actual_min = m * interval_mins;
        let conf      = confidences[m-1];
        let label_c   = confidence_label(conf);
        let dir       = clean_dir(sniper_pct[m-1]);
        // sniper_pct[m-1] is the deviation from 0.5 — treat it as the predicted % move signal
        let pred_pct  = sniper_pct[m-1] * 100.0;
        let pred_str  = format!("{:>+6.3}%", pred_pct);
        let clean_bull = sniper_pct[m-1] >= 0.0;
        let pass_cols: String = jitter_data[m-1][..show_passes].iter().map(|&p| {
            let agrees = (p >= 0.5) == clean_bull;
            format!(" {} ", if agrees { if p >= 0.5 { "▲" } else { "▼" } }
                            else      { if p >= 0.5 { "△" } else { "▽" } })
        }).collect::<Vec<_>>().join("|");

        if has_actual {
            let actual_str = if m <= future.len() {
                // per-candle % change from previous bar
                let prev_close = if m == 1 { anchor } else { future[m-2].close };
                let pct     = (future[m-1].close - prev_close) / prev_close;
                let correct = (sniper_pct[m-1] >= 0.0) == (pct >= 0.0);
                if correct { total_correct += 1; }
                format!("{:>+6.3}% / ${:<8.4} {}", pct*100.0, future[m-1].close,
                    if correct { "✓" } else { "✗" })
            } else { "  (no data)        ".into() };
            println!("  {:>3} | {:<9} | {} {:.0}% | {} | {:<20} |{}",
                actual_min, pred_str, label_c, conf*100.0, dir, actual_str, pass_cols);
        } else {
            println!("  {:>3} | {:<9} | {} {:.0}% | {} |{}",
                actual_min, pred_str, label_c, conf*100.0, dir, pass_cols);
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
//  run_prediction  (full verbose single-anchor)
// ─────────────────────────────────────────────

fn run_prediction(
    window:        &[StockData],
    sniper:        &Net,
    future:        &[StockData],
    mode_label:    &str,
    knn:           &KnnIndex,
    jitter_passes: usize,
    knn_k:         usize,
    noise_std:     f64,
    interval_mins: usize,
    _cfg:          &Config,
) -> (usize, usize) {
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

    let live_ind = compute_indicators(&bars_from_cascade(window));

    let ind_mean = live_ind.iter().sum::<f64>() / live_ind.len() as f64;
    let ind_std  = (live_ind.iter().map(|&x|(x-ind_mean).powi(2)).sum::<f64>()/live_ind.len() as f64).sqrt();
    println!("  [DIAG] Indicators  μ={:+.4}  σ={:.4}  vals: {}",
        ind_mean, ind_std,
        live_ind.iter().map(|&x| format!("{:+.3}",x)).collect::<Vec<_>>().join(" "));

    {
        let inp      = sniper.build_input(window, &live_ind, &[]);
        let inp_mean = inp.iter().sum::<f64>() / inp.len() as f64;
        let inp_std  = (inp.iter().map(|&x|(x-inp_mean).powi(2)).sum::<f64>()/inp.len() as f64).sqrt();
        println!("  [DIAG] Sniper input μ={:+.5}  σ={:.5}  len={}", inp_mean, inp_std, inp.len());
        let mut acts = inp;
        for (li, layer) in sniper.layers.iter().enumerate() {
            let mut out = vec![0.0f64; layer.out_size];
            layer.forward(&acts, &mut out);
            let amean = out.iter().sum::<f64>() / out.len() as f64;
            let astd  = (out.iter().map(|&x|(x-amean).powi(2)).sum::<f64>()/out.len() as f64).sqrt();
            let amin  = out.iter().cloned().fold(f64::INFINITY, f64::min);
            let amax  = out.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let tag   = if layer.is_output { "output".to_string() } else { format!("hidden{}", li+1) };
            println!("  [DIAG] Sniper layer {:8}  n={}  μ={:+.5}  σ={:.5}  [{:.3},{:.3}]",
                tag, out.len(), amean, astd, amin, amax);
            acts = out;
        }
    }

    let sn_map = sniper.predict(window, &live_ind, &[], anchor);

    let mut sniper_pct = [0.0f64; 10];
    for m in 1..=10 { sniper_pct[m-1] = sn_map.get(&m).map(|&(p,_)| p).unwrap_or(0.0); }

    let familiarity = knn.familiarity(&live_ind, knn_k);
    let sniper_inp  = sniper.build_input(window, &live_ind, &[]);
    let show_passes = jitter_passes.min(10);

    let mut confidences      = [0.0f64; 10];
    let mut stabilities      = [0.0f64; 10];
    let mut jitter_passes_data: Vec<Vec<f64>> = Vec::with_capacity(10);
    for m in 1..=10 {
        let (stab, pass_pcts) = sniper.stability(&sniper_inp, m - 1, jitter_passes, noise_std);
        stabilities[m-1]       = stab;
        confidences[m-1]       = confidence(stab, familiarity);
        jitter_passes_data.push(pass_pcts);
    }

    println!("━━━ Prediction from {} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        anchor_bar.ts.format("%Y-%m-%d %H:%M:%S"));

    let fmt_pred = |map: &std::collections::HashMap<usize,(f64,f64)>, m: usize| -> String {
        map.get(&m).map(|&(p,d)| format!("p={:.3} ({:>+.3})", d, p))
           .unwrap_or_else(|| "       --         ".into())
    };

    let has_actual  = !future.is_empty();
    let clean_dir   = |pct: f64| if pct >= 0.0 { "▲" } else { "▼" };
    let pass_header: String = (0..show_passes).map(|i| format!(" J{:<2}", i+1)).collect();

    if has_actual {
        println!("  Min | AI pred%   | Confidence             | Dir | Actual (candle%)     |{}", pass_header);
        println!("  ----|-----------|------------------------|-----|----------------------|{}",
            "-".repeat(show_passes * 4));
    } else {
        println!("  Min | AI pred%   | Confidence             | Dir |{}", pass_header);
        println!("  ----|-----------|------------------------|-----|{}",
            "-".repeat(show_passes * 4));
    }

    for m in 1..=10 {
        let actual_min = m * interval_mins;
        let conf  = confidences[m-1];
        let label = confidence_label(conf);
        let bar   = confidence_bar(conf);
        let dir   = clean_dir(sniper_pct[m-1]);

        // sniper_pct[m-1] is deviation from 0.5 — use as predicted % move signal
        let pred_pct = sniper_pct[m-1] * 100.0;
        let pred_str = format!("{:>+6.3}%", pred_pct);

        let clean_bull = sniper_pct[m-1] >= 0.0;
        let pass_cols: String = jitter_passes_data[m-1][..show_passes].iter().map(|&p| {
            let agrees = (p >= 0.5) == clean_bull;
            format!(" {} ", if agrees { if p >= 0.5 { "▲" } else { "▼" } }
                            else      { if p >= 0.5 { "△" } else { "▽" } })
        }).collect::<Vec<_>>().join("|");

        if has_actual {
            let actual = if m <= future.len() {
                // per-candle % change from previous bar
                let prev_close = if m == 1 { anchor } else { future[m-2].close };
                let pct     = (future[m-1].close - prev_close) / prev_close;
                let correct = (sniper_pct[m-1] >= 0.0) == (pct >= 0.0);
                format!("{:>+6.3}% / ${:<8.4} {}", pct*100.0, future[m-1].close,
                    if correct { "✓" } else { "✗" })
            } else { "  (no data)        ".into() };
            println!("  {:>3} | {:<9} | {} {} {:.0}% | {} | {:<20} |{}",
                actual_min, pred_str, bar, label, conf*100.0, dir, actual, pass_cols);
        } else {
            println!("  {:>3} | {:<9} | {} {} {:.0}% | {} |{}",
                actual_min, pred_str, bar, label, conf*100.0, dir, pass_cols);
        }
    }

    if has_actual && future.len() >= 10 {
        let correct: usize = (1..=10).filter(|&m| {
            let prev_close = if m == 1 { anchor } else { future[m-2].close };
            let pct = (future[m-1].close - prev_close) / prev_close;
            (sniper_pct[m-1] >= 0.0) == (pct >= 0.0)
        }).count();
        println!();
        println!("  Directional accuracy (Sniper): {}/10  ({:.0}%)", correct, correct as f64 * 10.0);

        let hc_correct = (1..=10).filter(|&m| {
            confidences[m-1] >= 0.40 && {
                let prev_close = if m == 1 { anchor } else { future[m-2].close };
                let pct = (future[m-1].close - prev_close) / prev_close;
                (sniper_pct[m-1] >= 0.0) == (pct >= 0.0)
            }
        }).count();
        let hc_total = (1..=10).filter(|&m| confidences[m-1] >= 0.40).count();
        if hc_total > 0 {
            println!("  MEDIUM/HIGH confidence calls: {}/{}  ({:.0}%)",
                hc_correct, hc_total,
                hc_correct as f64 / hc_total as f64 * 100.0);
        }
    }

    let avg_stab = stabilities.iter().sum::<f64>() / 10.0;
    println!();
    println!("  Confidence factors (minute 1):");
    println!("    Stability   : {:.1}%  ({}/{} noisy passes agreed on direction, noise={:.5})",
        stabilities[0] * 100.0, (stabilities[0] * jitter_passes as f64).round() as usize, jitter_passes, noise_std);
    println!("    Familiarity : {:.1}%  (k-NN distance to training data)", familiarity * 100.0);
    println!("    Combined    : {:.1}%  (stability × familiarity)", confidences[0] * 100.0);

    let bullish   = sniper_pct.iter().filter(|&&p| p > 0.0).count();
    let avg_pct   = sniper_pct.iter().sum::<f64>() / 10.0;
    let avg_conf  = confidences.iter().sum::<f64>() / 10.0;
    let p10_price = sn_map.get(&10).map(|&(_,d)| d).unwrap_or(anchor);
    let direction = if avg_pct >= 0.0 { "▲ BULLISH" } else { "▼ BEARISH" };

    println!();
    println!("  ─────────────────────────────────────────────────");
    println!("  Consensus   : {}  ({}/10 bars agree)", direction, bullish.max(10-bullish));
    println!("  Avg signal  : {:>+.4}  over next 10 minutes (0=neutral, ±0.5=max)", avg_pct);
    println!("  Avg conf    : {:.0}%  {}  {}  (stab {:.0}% × fam {:.0}%)",
        avg_conf*100.0, confidence_label(avg_conf), confidence_bar(avg_conf),
        avg_stab*100.0, familiarity*100.0);
    println!("  +10m prob   : {:.3}  ({} {:.1}% confidence)",
        p10_price, if p10_price >= 0.5 { "▲ BULLISH" } else { "▼ BEARISH" },
        (p10_price - 0.5).abs() * 200.0);

    let correct = if has_actual && future.len() >= 10 {
        (1..=10).filter(|&m| {
            let prev_close = if m == 1 { anchor } else { future[m-2].close };
            let pct = (future[m-1].close - prev_close) / prev_close;
            (sniper_pct[m-1] >= 0.0) == (pct >= 0.0)
        }).count()
    } else { 0 };
    let total = if has_actual { future.len().min(10) } else { 0 };
    (correct, total)
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a=="--help" || a=="-h") {
        println!("scan_predict — predict next 10 minutes with confidence scores");
        println!();
        println!("Usage:");
        println!("  scan_predict AAPL.weights");
        println!("  scan_predict AAPL.weights --at \"2025-06-13 14:30:00\"");
        println!("  scan_predict AAPL.weights --from 2026-01-01 --to 2026-03-10");
        println!("  scan_predict AAPL.weights --symbols \"AAPL,MSFT,NVDA\" --at \"2026-03-10 17:00:00\"");
        println!();
        println!("Flags:");
        println!("  --symbol SYM       Ticker (defaults to out-prefix in weights file)");
        println!("  --symbols S1,S2,.. Multi-symbol scan; loads {{SYM}}.weights per symbol");
        println!("  --api-key KEY      Twelve Data API key — only needed for live fetch");
        println!("  --csv PATH         Explicit CSV path");
        println!("  --at DATETIME      Anchor to past timestamp  (YYYY-MM-DD HH:MM:SS)");
        println!("  --from DATE        Start of date-range backtest  (YYYY-MM-DD)");
        println!("  --to   DATE        End   of date-range backtest  (YYYY-MM-DD)");
        println!("  --daily-time HH:MM Anchor time for each day in range (default: 17:00)");
        println!("  --jitter-passes N  Noisy forward passes for stability score (default: 30)");
        println!("  --knn-k N          Nearest neighbours for familiarity scoring (default: 10)");
        println!("  --noise F          Jitter noise magnitude (default: 0.001)");
        println!("  --interval N       Resample 1-min bars to N-min bars (default: 1)");
        std::process::exit(0);
    }

    let weights_path          = args[1].clone();
    let mut symbol            = String::new();
    let mut api_key           = String::new();
    let mut csv_path          = String::new();
    let mut at_str            = String::new();
    let mut jitter_passes: usize = 30;
    let mut knn_k:         usize = 10;
    let mut interval_mins: usize = 1;
    let mut noise_std:     f64   = 0.001;
    let mut symbols_str:   String = String::new();
    let mut from_str:      String = String::new();
    let mut to_str:        String = String::new();
    let mut daily_time:    String = "17:00".into();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--symbol"        => { symbol          = args[i+1].clone();                i += 2; }
            "--api-key"       => { api_key          = args[i+1].clone();               i += 2; }
            "--csv"           => { csv_path         = args[i+1].clone();               i += 2; }
            "--at"            => { at_str           = args[i+1].clone();               i += 2; }
            "--from"          => { from_str         = args[i+1].clone();               i += 2; }
            "--to"            => { to_str           = args[i+1].clone();               i += 2; }
            "--daily-time"    => { daily_time       = args[i+1].clone();               i += 2; }
            "--jitter-passes" => { jitter_passes    = args[i+1].parse().unwrap_or(30); i += 2; }
            "--knn-k"         => { knn_k            = args[i+1].parse().unwrap_or(10); i += 2; }
            "--interval"      => { interval_mins    = args[i+1].parse().unwrap_or(1);  i += 2; }
            "--noise"         => { noise_std        = args[i+1].parse().unwrap_or(0.001); i += 2; }
            "--symbols"       => { symbols_str      = args[i+1].clone();               i += 2; }
            _                 => { i += 1; }
        }
    }

    println!("Loading weights from '{}'...", weights_path);
    let (mut cfg, _scout, _spotter, sniper) = load_all_weights(&weights_path);
    cfg.sniper_only = true;
    println!("  lookback={} hidden={} layers={} prefix={}",
        cfg.lookback, cfg.hidden, cfg.layers, cfg.out_prefix);

    if symbol.is_empty() { symbol = cfg.out_prefix.clone(); }

    // ── Date-range backtest  --from / --to ───────────────────────────────────
    if !from_str.is_empty() && !to_str.is_empty() {
        let from_date = NaiveDate::parse_from_str(&from_str, "%Y-%m-%d")
            .unwrap_or_else(|_| panic!("--from must be YYYY-MM-DD, got '{}'", from_str));
        let to_date = NaiveDate::parse_from_str(&to_str, "%Y-%m-%d")
            .unwrap_or_else(|_| panic!("--to must be YYYY-MM-DD, got '{}'", to_str));
        if from_date > to_date {
            eprintln!("--from {} is after --to {} — nothing to do.", from_str, to_str);
            return;
        }

        let (hh, mm): (u32, u32) = {
            let parts: Vec<&str> = daily_time.splitn(2, ':').collect();
            let h = parts.get(0).and_then(|s| s.parse().ok()).unwrap_or(17u32);
            let m = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0u32);
            (h, m)
        };

        let src = if !csv_path.is_empty() { csv_path.clone() }
                  else { format!("{}_data.csv", symbol.to_uppercase()) };

        // Auto-download data if needed — mirrors --at and --symbols behaviour.
        ensure_data_for_date(&symbol, &src, to_date, &api_key);

        if !std::path::Path::new(&src).exists() {
            eprintln!("No data file '{}' — cannot run date-range backtest.", src);
            return;
        }
        println!("\nLoading data from '{}' for date-range backtest...", src);
        let all_bars = { let r = parse_csv(&src); if interval_mins > 1 { resample(r, interval_mins) } else { r } };
        if all_bars.is_empty() { eprintln!("No bars in '{}'.", src); return; }

        // Build KNN index only from bars strictly before the backtest window so
        // familiarity reflects the training distribution, not future data.
        let from_utc: DateTime<Utc> = Utc.from_utc_datetime(
            &from_date.and_hms_opt(0, 0, 0).unwrap()
        );
        let training_bars: Vec<_> = all_bars.iter()
            .filter(|b| b.ts < from_utc)
            .cloned()
            .collect();
        println!("  {} bars before {} used for familiarity index ({} total in CSV).",
            training_bars.len(), from_date, all_bars.len());
        let knn = {
            let idx = KnnIndex::build(&training_bars, cfg.lookback);
            println!("  {} training vectors indexed for familiarity.\n", idx.vecs.len());
            idx
        };

        println!("━━━ Date-range backtest: {} → {}  (daily anchor {:02}:{:02} UTC) ━━━",
            from_date, to_date, hh, mm);
        let act_label = format!("Act{}m%", 10 * interval_mins);
        println!("  {:<12} | {:>9} | {:>6} | {:>8} | {:>8} | {:>8} | {:<9}",
            "Date", "Close", "Acc", "AvgConf", "Pred%", act_label, "Consensus");
        println!("  {}", "-".repeat(75));

        let mut results: Vec<DayResult> = Vec::new();
        let mut cur = from_date;
        while cur <= to_date {
            let anchor_ndt = cur.and_hms_opt(hh, mm, 0).unwrap_or_else(|| cur.and_hms_opt(17, 0, 0).unwrap());
            let anchor_utc: DateTime<Utc> = Utc.from_utc_datetime(&anchor_ndt);

            if let Some(anchor_idx) = all_bars.iter().rposition(|b| b.ts <= anchor_utc) {
                let bar_date = all_bars[anchor_idx].ts.date_naive();
                if bar_date == cur {
                    let win_start = anchor_idx.saturating_sub(cfg.lookback - 1);
                    let window    = all_bars[win_start..=anchor_idx].to_vec();
                    let future: Vec<_> = all_bars[anchor_idx+1..]
                        .iter().take_while(|b| b.ts.date_naive() == cur).take(10).cloned().collect();

                    if window.len() >= 2 {
                        if let Some(dr) = run_day_quiet(&window, &sniper, &future, &knn,
                            jitter_passes, knn_k, noise_std, cur, &cfg)
                        {
                            let acc_str = if dr.total > 0 { format!("{}/{}", dr.correct, dr.total) } else { "--".into() };
                            let act_str = dr.actual_pct_10.map(|p| format!("{:>+6.3}%", p*100.0)).unwrap_or("   --   ".into());
                            let cons    = if dr.consensus_dir { "▲ BULL" } else { "▼ BEAR" };
                            println!("  {:<12} | {:>9.4} | {:>6} | {:>7.1}% | {:>+7.3}% | {} | {}",
                                cur.to_string(), dr.anchor_price, acc_str, dr.avg_conf * 100.0,
                                dr.avg_pct * 100.0, act_str, cons);
                            results.push(dr);
                        }
                    }
                }
            }
            cur = cur + ChronoDuration::days(1);
        }

        if results.is_empty() { println!("\n  No trading days found in range {} → {}.", from_date, to_date); return; }

        let days_total     = results.len();
        let days_with_data = results.iter().filter(|r| r.total > 0).count();
        let total_correct: usize = results.iter().map(|r| r.correct).sum();
        let total_calls:   usize = results.iter().map(|r| r.total).sum();
        let avg_conf_all:  f64   = results.iter().map(|r| r.avg_conf).sum::<f64>() / days_total as f64;

        let (dir_correct, dir_total) = results.iter().fold((0usize, 0usize), |(c, t), r| {
            if let Some(act) = r.actual_pct_10 {
                (c + if r.consensus_dir == (act >= 0.0) { 1 } else { 0 }, t + 1)
            } else { (c, t) }
        });

        let (mut best_streak, mut worst_streak, mut cur_streak) = (0i32, 0i32, 0i32);
        for r in &results {
            if let Some(act) = r.actual_pct_10 {
                let win = r.consensus_dir == (act >= 0.0);
                cur_streak   = if win { (cur_streak + 1).max(1) } else { (cur_streak - 1).min(-1) };
                best_streak  = best_streak.max(cur_streak);
                worst_streak = worst_streak.min(cur_streak);
            }
        }

        let hc_thresh = 0.70;
        let hc_days: Vec<&DayResult> = results.iter().filter(|r| r.avg_conf >= hc_thresh).collect();
        let (hc_correct, hc_total) = hc_days.iter().fold((0usize, 0usize), |(c, t), r| {
            if let Some(act) = r.actual_pct_10 { (c + if r.consensus_dir == (act >= 0.0) { 1 } else { 0 }, t + 1) }
            else { (c, t) }
        });

        struct Band { label: &'static str, lo: f64, hi: f64, correct: usize, total: usize }
        let mut bands = [
            Band { label: "HIGH  (≥70%)",   lo: 0.70, hi: 1.01, correct: 0, total: 0 },
            Band { label: "MEDIUM(40-70%)", lo: 0.40, hi: 0.70, correct: 0, total: 0 },
            Band { label: "LOW   (<40%)",   lo: 0.00, hi: 0.40, correct: 0, total: 0 },
        ];
        for r in &results {
            if let Some(act) = r.actual_pct_10 {
                let win = r.consensus_dir == (act >= 0.0);
                for b in &mut bands {
                    if r.avg_conf >= b.lo && r.avg_conf < b.hi { b.total += 1; if win { b.correct += 1; } }
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
                hc_correct, hc_total, hc_correct as f64 / hc_total as f64 * 100.0, hc_thresh * 100.0);
        }
        println!("  Avg confidence  : {:.1}%", avg_conf_all * 100.0);
        println!("  Best streak     : {} consecutive correct daily calls", best_streak.max(0));
        println!("  Worst streak    : {} consecutive wrong  daily calls", worst_streak.abs());

        println!();
        println!("  Confidence calibration (directional acc per band):");
        let mut any_miscal = false;
        for b in &bands {
            if b.total > 0 {
                let acc  = b.correct as f64 / b.total as f64 * 100.0;
                let flag = if b.lo >= 0.70 && acc < 55.0 {
                    any_miscal = true; " ⚠  HIGH conf calls are below 55% — model needs retraining"
                } else { "" };
                println!("    {:15} : {}/{} ({:.1}%){}", b.label, b.correct, b.total, acc, flag);
            }
        }
        if any_miscal {
            println!();
            println!("  ⚠  CALIBRATION WARNING: confidence score does not correlate with directional accuracy.");
            println!("     Retrain with BCE loss (cascade_trainer default).");
        } else if bands[0].total > 0 {
            let high_acc = bands[0].correct as f64 / bands[0].total as f64 * 100.0;
            let low_acc  = if bands[2].total > 0 { bands[2].correct as f64 / bands[2].total as f64 * 100.0 } else { 0.0 };
            if high_acc > low_acc + 5.0 { println!("    ✓ Confidence is well-calibrated: HIGH > LOW accuracy."); }
        }
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

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
                let lbl = format!("{:<10}  {}  —  {}/{} correct  ({:.0}% acc)                     ",
                    tag, day_result.date, day_result.correct, day_result.total,
                    day_result.correct as f64 / day_result.total as f64 * 100.0);
                run_day_verbose(&all_bars, &sniper, &knn, &cfg,
                    day_result.date, hh, mm, jitter_passes, knn_k, noise_std, interval_mins, &lbl);
            }
        } else {
            println!("\n  (Need at least 2 days with full 10-minute future data for best/worst drill-down)");
        }
        return;
    }
    // ── End date-range backtest ───────────────────────────────────────────────

    // ── Multi-symbol scan  --symbols ─────────────────────────────────────────
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

            let sym_weights = format!("{}.weights", sym_upper);
            let (sym_cfg, _sym_scout, _sym_spotter, sym_sniper) =
                if std::path::Path::new(&sym_weights).exists() {
                    println!("  Weights : '{}'", sym_weights);
                    load_all_weights(&sym_weights)
                } else {
                    println!("  ⚠  No {}.weights — falling back to '{}'", sym_upper, weights_path);
                    load_all_weights(&weights_path)
                };

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

            let anchor_idx: usize = match all.iter().rposition(|b| b.ts <= at_utc) {
                Some(i) => i,
                None    => { eprintln!("  Skipping {} — no bar at or before {}", sym_upper, at_str); continue; }
            };

            let at_date   = all[anchor_idx].ts.date_naive();
            let win_start = anchor_idx.saturating_sub(sym_cfg.lookback - 1);
            let window    = all[win_start..=anchor_idx].to_vec();
            let future: Vec<_> = all[anchor_idx+1..]
                .iter().take_while(|b| b.ts.date_naive() == at_date).take(10).cloned().collect();

            let label = format!("HISTORICAL  --at \"{}\"  ({} future bars)", at_str, future.len());

            if window.len() >= 2 {
                let (correct, total) = run_prediction(&window, &sym_sniper,
                    &future, &label, &sym_knn, jitter_passes, knn_k, noise_std, interval_mins, &sym_cfg);
                total_correct += correct;
                total_calls   += total;
            }
        }

        println!("\n══════════════════════════════════════════════════════════");
        println!("  Multi-symbol test complete: {} symbols at {}", sym_list.len(), at_str);
        if total_calls > 0 {
            println!("  Aggregate directional accuracy: {}/{} ({:.0}%)",
                total_correct, total_calls, total_correct as f64 / total_calls as f64 * 100.0);
        }
        println!("══════════════════════════════════════════════════════════\n");
        return;
    }
    // ── End multi-symbol scan ─────────────────────────────────────────────────

    // ── Single-symbol prediction ──────────────────────────────────────────────
    let (window, future, mode_label): (Vec<StockData>, Vec<StockData>, String) = if !at_str.is_empty() {

        let parse_at = NaiveDateTime::parse_from_str(&at_str, "%Y-%m-%d %H:%M:%S")
            .unwrap_or_else(|_| {
                let d = chrono::NaiveDate::parse_from_str(&at_str, "%Y-%m-%d")
                    .expect("--at must be YYYY-MM-DD HH:MM:SS or YYYY-MM-DD");
                d.and_hms_opt(23,59,59).unwrap()
            });
        let at_utc: DateTime<Utc> = Utc.from_utc_datetime(&parse_at);

        let src = if !csv_path.is_empty() { csv_path.clone() }
                  else { format!("{}_data.csv", symbol.to_uppercase()) };
        ensure_data_for_date(&symbol, &src, parse_at.date(), &api_key);

        println!("Loading historical data from '{}'...", src);
        let all = { let r = parse_csv(&src); if interval_mins > 1 { resample(r, interval_mins) } else { r } };
        if all.is_empty() { panic!("No bars in '{}'.", src); }

        let anchor_idx = all.iter().rposition(|b| b.ts <= at_utc)
            .unwrap_or_else(|| panic!("No bar at or before {} in '{}'.", at_str, src));

        let at_date   = all[anchor_idx].ts.date_naive();
        let win_start = anchor_idx.saturating_sub(cfg.lookback - 1);
        let window    = all[win_start..=anchor_idx].to_vec();
        let future: Vec<_> = all[anchor_idx+1..]
            .iter().take_while(|b| b.ts.date_naive() == at_date).take(10).cloned().collect();

        let label = format!("HISTORICAL  --at \"{}\"  ({} future bars for accuracy check)", at_str, future.len());
        (window, future, label)

    } else {

        let auto_csv     = format!("{}_data.csv", symbol.to_uppercase());
        let resolved_csv = if !csv_path.is_empty() {
            Some(csv_path.clone())
        } else if std::path::Path::new(&auto_csv).exists() {
            println!("  No --csv given — using local database '{}'", auto_csv);
            Some(auto_csv.clone())
        } else {
            None
        };

        if let Some(src) = resolved_csv {
            println!("Loading data from '{}' ...", src);
            let all = { let r = parse_csv(&src); if interval_mins > 1 { resample(r, interval_mins) } else { r } };
            if all.is_empty() { panic!("No bars in '{}'.", src); }

            let last_date  = all.last().unwrap().ts.date_naive();
            let day_bars   = all.iter().filter(|b| b.ts.date_naive() == last_date).count();
            let anchor_idx = all.len() - 1;
            let win_start  = anchor_idx.saturating_sub(cfg.lookback - 1);
            let window     = all[win_start..=anchor_idx].to_vec();
            let anchor_ts  = window.last().unwrap().ts;
            let future: Vec<_> = vec![];

            println!("  Last trading day   : {}  ({} bars total)", last_date, day_bars);
            println!("  Anchor bar         : {}  (use --at to pick a different time)",
                anchor_ts.format("%Y-%m-%d %H:%M:%S UTC"));

            let label = format!("LAST TRADING DAY  {}  anchor {}", last_date, anchor_ts.format("%H:%M UTC"));
            (window, future, label)

        } else {
            let fetch_n = (cfg.lookback + 30).min(5000);
            let bars    = fetch_or_load(&symbol, fetch_n, &api_key);
            if bars.is_empty() { panic!("No bars available. Supply --api-key or --csv."); }
            let start  = bars.len().saturating_sub(cfg.lookback);
            let window = bars[start..].to_vec();
            let is_live = bars.last()
                .map(|b| Utc::now().signed_duration_since(b.ts).num_seconds() <= 120)
                .unwrap_or(false);
            let label = format!("{} (anchor {})",
                if is_live { "LIVE" } else { "CACHED" },
                window.last().map(|b| b.ts.format("%Y-%m-%d %H:%M:%S UTC").to_string()).unwrap_or_default());
            (window, vec![], label)
        }
    };

    if window.len() < 2 {
        eprintln!("Window has fewer than 2 bars — cannot predict."); std::process::exit(1);
    }
    if window.len() < cfg.lookback {
        eprintln!("⚠  Window has {} bars but lookback={}. Accuracy may be reduced.", window.len(), cfg.lookback);
    }

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

    let _ = run_prediction(&window, &sniper, &future, &mode_label, &knn, jitter_passes, knn_k, noise_std, interval_mins, &cfg);
}
