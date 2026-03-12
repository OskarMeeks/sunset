
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead};
use std::path::Path;
use std::thread;
use std::time::Instant;

// ─────────────────────────────────────────────
//  Configuration & Persistence
// ─────────────────────────────────────────────

use rust_xlsxwriter::{Chart, ChartLegendPosition, ChartLine, ChartType, Color, Format, FormatBorder, Workbook, XlsxError};

// ─────────────────────────────────────────────
//  Per-epoch metrics captured during training
// ─────────────────────────────────────────────

#[derive(Clone)]
pub struct EpochMetrics {
    pub loss:           f64,
    pub accuracy_pct:   f64,
    pub epoch_secs:     f64,
}

// ─────────────────────────────────────────────
//  One complete run record
// ─────────────────────────────────────────────

#[derive(Clone)]
pub struct RunRecord {
    pub run_label: String,   // e.g. "run-3 lr=0.001 decay=0.997"
    pub cfg:       Config,
    pub epochs:    Vec<EpochMetrics>,
}

// ─────────────────────────────────────────────
//  Palette — distinct colours for up to 12 runs
// ─────────────────────────────────────────────

const PALETTE: [u32; 12] = [
    0x1F77B4, 0xFF7F0E, 0x2CA02C, 0xD62728,
    0x9467BD, 0x8C564B, 0xE377C2, 0x7F7F7F,
    0xBCBD22, 0x17BECF, 0xAEC7E8, 0xFFBB78,
];

// ─────────────────────────────────────────────
//  TuningTracker
//
//  Layout per phase sheet (e.g. "Scout"):
//
//  Row 0        │ Config summary header
//  Rows 1-13    │ Config key/value pairs for the latest run
//  Row 15       │ Run labels header  (col 0 = "Epoch", col 1+ = run labels)
//  Rows 16+     │ Epoch data: col 0 = epoch#, then groups of 3 cols per run:
//               │   [loss | accuracy% | epoch_secs]
//  Row 0 col 5+ │ Charts stacked to the right
// ─────────────────────────────────────────────

pub struct TuningTracker {
    file_path:    String,
    scout_runs:   Vec<RunRecord>,
    spotter_runs: Vec<RunRecord>,
    sniper_runs:  Vec<RunRecord>,
}

impl TuningTracker {
    /// Load existing run history from JSON sidecar, or start fresh.
    pub fn new(file_path: &str) -> Self {
        let sidecar = format!("{}.runs.json", file_path);
        let (scout_runs, spotter_runs, sniper_runs) =
            if let Ok(json) = fs::read_to_string(&sidecar) {
                serde_json::from_str::<(Vec<RunRecord>, Vec<RunRecord>, Vec<RunRecord>)>(&json)
                    .unwrap_or_default()
            } else {
                (vec![], vec![], vec![])
            };
        TuningTracker { file_path: file_path.to_string(), scout_runs, spotter_runs, sniper_runs }
    }

    /// Add a run for a given phase ("Scout", "Spotter", "Sniper").
    pub fn add_run(&mut self, phase: &str, record: RunRecord) {
        let vec = match phase {
            "Scout"   => &mut self.scout_runs,
            "Spotter" => &mut self.spotter_runs,
            _         => &mut self.sniper_runs,
        };
        vec.push(record);
    }

    /// Persist history + write Excel workbook.
    pub fn save(self) -> Result<(), XlsxError> {
        // Save JSON sidecar so history accumulates across runs
        let sidecar = format!("{}.runs.json", self.file_path);
        let payload = (&self.scout_runs, &self.spotter_runs, &self.sniper_runs);
        fs::write(&sidecar, serde_json::to_string(&payload).unwrap_or_default()).ok();

        let mut wb = Workbook::new();

        // One sheet per phase
        Self::write_phase_sheet(&mut wb, "Scout",   &self.scout_runs)?;
        Self::write_phase_sheet(&mut wb, "Spotter", &self.spotter_runs)?;
        Self::write_phase_sheet(&mut wb, "Sniper",  &self.sniper_runs)?;

        wb.save(&self.file_path)
    }

    // ── Sheet layout ──────────────────────────────────────────────────────

    fn write_phase_sheet(wb: &mut Workbook, phase: &str, runs: &[RunRecord]) -> Result<(), XlsxError> {
        if runs.is_empty() { return Ok(()); }

        let ws = wb.add_worksheet().set_name(phase)?;

        // ── Header style ──
        let hdr_fmt = Format::new()
            .set_bold()
            .set_background_color(Color::RGB(0x2C3E50))
            .set_font_color(Color::White)
            .set_border(FormatBorder::Thin);
        let label_fmt = Format::new().set_bold();

        // ── Config summary (latest run) — cols A-B, rows 1-14 ──
        let cfg = &runs.last().unwrap().cfg;
        ws.write_with_format(0, 0, "Parameter", &hdr_fmt)?;
        ws.write_with_format(0, 1, "Value (latest run)", &hdr_fmt)?;
        let params: &[(&str, String)] = &[
            ("LR 1",        format!("{}", cfg.lr1)),
            ("LR 2",        format!("{}", cfg.lr2)),
            ("LR 3",        format!("{}", cfg.lr3)),
            ("LR Decay",    format!("{}", cfg.lr_decay)),
            ("Dir Weight",  format!("{}", cfg.dir_weight)),
            ("Layers",      format!("{}", cfg.layers)),
            ("Hidden",      format!("{}", cfg.hidden)),
            ("Lookback",    format!("{}", cfg.lookback)),
            ("Batch Size",  format!("{}", cfg.batch_size)),
            ("Epochs 1",    format!("{}", cfg.epochs1)),
            ("Epochs 2",    format!("{}", cfg.epochs2)),
            ("Epochs 3",    format!("{}", cfg.epochs3)),
            ("Out Prefix",  cfg.out_prefix.clone()),
            ("Total Runs",  format!("{}", runs.len())),
        ];
        for (i, (k, v)) in params.iter().enumerate() {
            ws.write(1 + i as u32, 0, *k)?;
            ws.write(1 + i as u32, 1, v.as_str())?;
        }

        // ── Data table header row ──
        // Epoch | run0_loss | run0_acc | run0_secs | run1_loss | ...
        let data_header_row: u32 = 15;
        let data_start_row:  u32 = 16;
        ws.write_with_format(data_header_row, 0, "Epoch", &hdr_fmt)?;

        for (ri, run) in runs.iter().enumerate() {
            let base_col = 1 + ri as u16 * 3;
            let short = Self::short_label(&run.run_label);
            ws.write_with_format(data_header_row, base_col,     format!("{} Loss",    short).as_str(), &hdr_fmt)?;
            ws.write_with_format(data_header_row, base_col + 1, format!("{} Acc%",    short).as_str(), &hdr_fmt)?;
            ws.write_with_format(data_header_row, base_col + 2, format!("{} Sec/Ep",  short).as_str(), &hdr_fmt)?;
        }

        // ── Data rows ──
        let max_epochs = runs.iter().map(|r| r.epochs.len()).max().unwrap_or(0);
        for ep in 0..max_epochs {
            let row = data_start_row + ep as u32;
            ws.write(row, 0, ep as u32 + 1)?;
            for (ri, run) in runs.iter().enumerate() {
                if let Some(m) = run.epochs.get(ep) {
                    let base_col = 1 + ri as u16 * 3;
                    ws.write(row, base_col,     m.loss)?;
                    ws.write(row, base_col + 1, m.accuracy_pct)?;
                    ws.write(row, base_col + 2, m.epoch_secs)?;
                }
            }
        }

        // ── Column widths ──
        ws.set_column_width(0, 8)?;
        for ri in 0..runs.len() {
            let b = (1 + ri * 3) as u16;
            ws.set_column_width(b,     13)?;
            ws.set_column_width(b + 1, 10)?;
            ws.set_column_width(b + 2, 10)?;
        }

        let last_data_row = data_start_row + max_epochs as u32 - 1;
        let chart_anchor_col = (1 + runs.len() * 3 + 1) as u16;

        // ── Chart 1: Loss curves (all runs) ──
        let mut loss_chart = Chart::new(ChartType::Line);
        for (ri, run) in runs.iter().enumerate() {
            let base_col = (1 + ri * 3) as u16;
            let hex      = format!("#{:06X}", PALETTE[ri % PALETTE.len()]);
            let label    = Self::short_label(&run.run_label);
            loss_chart.add_series()
                .set_values((phase, data_start_row, base_col, last_data_row, base_col))
                .set_categories((phase, data_start_row, 0, last_data_row, 0))
                .set_name(label.as_str())
                .set_format(ChartLine::new().set_color(hex.as_str()));
        }
        loss_chart.title().set_name(format!("{} — Training Loss", phase).as_str());
        loss_chart.x_axis().set_name("Epoch");
        loss_chart.y_axis().set_name("MSE Loss");
        loss_chart.legend().set_position(ChartLegendPosition::Bottom);
        ws.insert_chart(0, chart_anchor_col, &loss_chart)?;

        // ── Chart 2: Accuracy curves (all runs) ──
        let mut acc_chart = Chart::new(ChartType::Line);
        for (ri, run) in runs.iter().enumerate() {
            let base_col = (1 + ri * 3) as u16;
            let acc_col  = base_col + 1;
            let hex      = format!("#{:06X}", PALETTE[ri % PALETTE.len()]);
            let label    = Self::short_label(&run.run_label);
            acc_chart.add_series()
                .set_values((phase, data_start_row, acc_col, last_data_row, acc_col))
                .set_categories((phase, data_start_row, 0, last_data_row, 0))
                .set_name(label.as_str())
                .set_format(ChartLine::new().set_color(hex.as_str()));
        }
        acc_chart.title().set_name(format!("{} — Directional Accuracy %", phase).as_str());
        acc_chart.x_axis().set_name("Epoch");
        acc_chart.y_axis().set_name("Accuracy %");
        acc_chart.legend().set_position(ChartLegendPosition::Bottom);
        ws.insert_chart(16, chart_anchor_col, &acc_chart)?;

        // ── Chart 3: Epoch duration (all runs) ──
        let mut time_chart = Chart::new(ChartType::Line);
        for (ri, run) in runs.iter().enumerate() {
            let base_col  = (1 + ri * 3) as u16;
            let secs_col  = base_col + 2;
            let hex       = format!("#{:06X}", PALETTE[ri % PALETTE.len()]);
            let label     = Self::short_label(&run.run_label);
            time_chart.add_series()
                .set_values((phase, data_start_row, secs_col, last_data_row, secs_col))
                .set_categories((phase, data_start_row, 0, last_data_row, 0))
                .set_name(label.as_str())
                .set_format(ChartLine::new().set_color(hex.as_str()));
        }
        time_chart.title().set_name(format!("{} — Seconds per Epoch", phase).as_str());
        time_chart.x_axis().set_name("Epoch");
        time_chart.y_axis().set_name("Seconds");
        time_chart.legend().set_position(ChartLegendPosition::Bottom);
        ws.insert_chart(32, chart_anchor_col, &time_chart)?;

        // ── Run label legend block — color-coded cells matching chart series ──
        ws.write_with_format(14, 0, "Run Legend", &label_fmt)?;
        for (ri, run) in runs.iter().enumerate() {
            let rgb       = PALETTE[ri % PALETTE.len()];
            let cell_fmt  = Format::new()
                .set_background_color(Color::RGB(rgb))
                .set_font_color(Color::White)
                .set_bold();
            ws.write_with_format(14, 1 + ri as u16, run.run_label.as_str(), &cell_fmt)?;
        }

        Ok(())
    }

    fn short_label(label: &str) -> String {
        // Shorten to fit chart legend: keep last 20 chars
        if label.len() > 20 { format!("…{}", &label[label.len()-19..]) } else { label.to_string() }
    }
}

// ── Serde for RunRecord & EpochMetrics so they survive between sessions ──

impl serde::Serialize for EpochMetrics {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("EpochMetrics", 3)?;
        st.serialize_field("loss",         &self.loss)?;
        st.serialize_field("accuracy_pct", &self.accuracy_pct)?;
        st.serialize_field("epoch_secs",   &self.epoch_secs)?;
        st.end()
    }
}
impl<'de> serde::Deserialize<'de> for EpochMetrics {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)] struct H { loss: f64, accuracy_pct: f64, epoch_secs: f64 }
        let h = H::deserialize(d)?;
        Ok(EpochMetrics { loss: h.loss, accuracy_pct: h.accuracy_pct, epoch_secs: h.epoch_secs })
    }
}
impl serde::Serialize for RunRecord {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("RunRecord", 3)?;
        st.serialize_field("run_label", &self.run_label)?;
        st.serialize_field("cfg",       &self.cfg)?;
        st.serialize_field("epochs",    &self.epochs)?;
        st.end()
    }
}
impl<'de> serde::Deserialize<'de> for RunRecord {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)] struct H { run_label: String, cfg: Config, epochs: Vec<EpochMetrics> }
        let h = H::deserialize(d)?;
        Ok(RunRecord { run_label: h.run_label, cfg: h.cfg, epochs: h.epochs })
    }
}


#[derive(Serialize, Deserialize, Clone, Debug)]
struct Config {
    lookback: usize,
    hidden: usize,
    layers: usize,
    batch_size: usize,
    lr_decay: f64,
    epochs1: usize, epochs2: usize, epochs3: usize,
    lr1: f64, lr2: f64, lr3: f64,
    dir_weight: f64,   // directional penalty weight (0.0 = pure MSE, 0.3 = default)
    out_prefix: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lookback: 60, hidden: 128, layers: 2, batch_size: 256, lr_decay: 0.997,
            epochs1: 300, epochs2: 300, epochs3: 300,
            lr1: 0.001, lr2: 0.001, lr3: 0.001,
            dir_weight: 0.3,
            out_prefix: "default".into(),
        }
    }
}

// ─────────────────────────────────────────────
//  Neural Structures (Optimized)
// ─────────────────────────────────────────────

#[derive(Clone)]
struct StockData {
    _ts: DateTime<Utc>,
    open: f64, high: f64, low: f64, close: f64, volume: u64,
}

// ─────────────────────────────────────────────
//  Technical indicators — 18 pre-computed market signals
// ─────────────────────────────────────────────
//
//  These are appended ONCE per sample at the end of the flat input vector
//  (after the bar-by-bar rolling features), so the network receives both
//  raw price action AND pre-derived signals it would otherwise have to
//  discover from scratch:
//
//   Trend      [0-3]  EMA(9) dist, EMA(21) dist, EMA slope, EMA crossover
//   Momentum   [4-6]  RSI(14), MACD histogram, MACD signal direction
//   Volatility [7-9]  Bollinger %B, bandwidth, ATR ratio
//   Structure  [10-12] Pivot support dist, resistance dist, SR cluster
//   VWAP       [13-14] VWAP distance, VWAP slope
//   Volume     [15-16] OBV momentum, volume ratio vs 20-bar avg
//   Candle     [17]   Body ratio (bullish/bearish bar strength)

const INDICATOR_NF: usize = 18;

fn ema_calc(closes: &[f64], period: usize) -> f64 {
    if closes.is_empty() { return 0.0; }
    let k = 2.0 / (period as f64 + 1.0);
    closes.iter().skip(1).fold(closes[0], |e, &c| c * k + e * (1.0 - k))
}

fn rsi_calc(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 { return 0.5; }
    let recent = &closes[closes.len() - period - 1..];
    let (mut ag, mut al) = (0.0_f64, 0.0_f64);
    for w in recent.windows(2) {
        let d = w[1] - w[0];
        if d > 0.0 { ag += d; } else { al += d.abs(); }
    }
    ag /= period as f64; al /= period as f64;
    if al < 1e-10 { return 1.0; }
    let rs = ag / al;
    (rs / (1.0 + rs)).clamp(0.0, 1.0)
}

fn macd_calc(closes: &[f64]) -> (f64, f64) {
    if closes.len() < 26 { return (0.0, 0.0); }
    let fast = ema_calc(closes, 12);
    let slow = ema_calc(closes, 26);
    let line = fast - slow;
    let n    = closes.len();
    let series: Vec<f64> = (9..=n).map(|e| ema_calc(&closes[..e], 12) - ema_calc(&closes[..e], 26)).collect();
    let signal = ema_calc(&series, 9);
    let hist   = (line - signal) / closes.last().unwrap_or(&1.0).abs().max(1e-8);
    (hist.clamp(-0.01, 0.01) / 0.01, signal.signum())
}

fn bollinger_calc(closes: &[f64], period: usize) -> (f64, f64) {
    if closes.len() < period { return (0.0, 0.0); }
    let w    = &closes[closes.len() - period..];
    let mean = w.iter().sum::<f64>() / period as f64;
    let std  = (w.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / period as f64).sqrt().max(1e-8);
    let c    = *closes.last().unwrap();
    let pct_b = ((c - (mean - 2.0 * std)) / (4.0 * std).max(1e-8)).clamp(0.0, 1.0) * 2.0 - 1.0;
    let bw    = (4.0 * std / mean.abs().max(1e-8)).clamp(0.0, 0.1) / 0.1;
    (pct_b, bw)
}

fn atr_ratio_calc(data: &[StockData], period: usize) -> f64 {
    if data.len() < period + 1 { return 0.0; }
    let trs: Vec<f64> = data.windows(2).map(|w| {
        let tr = (w[1].high - w[1].low)
            .max((w[1].high - w[0].close).abs())
            .max((w[1].low  - w[0].close).abs());
        tr / w[1].close.max(1e-8)
    }).collect();
    let avg  = trs[trs.len().saturating_sub(period)..].iter().sum::<f64>() / period.min(trs.len()) as f64;
    let last = *trs.last().unwrap_or(&0.0);
    ((last / avg.max(1e-8)).clamp(0.0, 5.0) / 5.0) * 2.0 - 1.0
}

fn pivot_sr_calc(data: &[StockData]) -> (f64, f64, f64) {
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
        let near = all.iter().filter(|&&p| ((p - close) / close).abs() < 0.005).count();
        (near as f64 / all.len() as f64).clamp(0.0, 1.0)
    };
    (sn, rn, cl)
}

fn vwap_calc(data: &[StockData]) -> (f64, f64) {
    if data.is_empty() { return (0.0, 0.0); }
    let (mut cpv, mut cv) = (0.0, 0.0);
    let vwaps: Vec<f64> = data.iter().map(|b| {
        cpv += (b.high + b.low + b.close) / 3.0 * (b.volume as f64).max(1.0);
        cv  += (b.volume as f64).max(1.0);
        cpv / cv
    }).collect();
    let vw = *vwaps.last().unwrap();
    let c  = data.last().unwrap().close;
    let dist  = ((c - vw) / c.max(1e-8)).clamp(-0.05, 0.05) / 0.05;
    let n     = vwaps.len();
    let slope = if n >= 5 { ((vwaps[n-1] - vwaps[n-5]) / vwaps[n-5].abs().max(1e-8)).clamp(-0.02, 0.02) / 0.02 } else { 0.0 };
    (dist, slope)
}

fn obv_calc(data: &[StockData]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let (mut v, mut scale) = (0.0_f64, 0.0_f64);
    let mut obvs = vec![0.0_f64; data.len()];
    for i in 1..data.len() {
        let d = data[i].close - data[i-1].close;
        v += if d > 0.0 { data[i].volume as f64 } else if d < 0.0 { -(data[i].volume as f64) } else { 0.0 };
        obvs[i] = v;
        scale = scale.max(v.abs());
    }
    let n  = obvs.len();
    let lb = 10.min(n - 1);
    ((obvs[n-1] - obvs[n-1-lb]) / scale.max(1.0)).clamp(-1.0, 1.0)
}

fn volume_ratio_calc(data: &[StockData]) -> f64 {
    if data.is_empty() { return 0.0; }
    let w   = &data[data.len().saturating_sub(20)..];
    let avg = w.iter().map(|b| b.volume as f64).sum::<f64>() / w.len().max(1) as f64;
    (data.last().unwrap().volume as f64 / avg.max(1.0)).clamp(0.0, 5.0) / 5.0 * 2.0 - 1.0
}

/// Build the full 18-feature indicator vector for a lookback window.
fn compute_indicators(data: &[StockData]) -> [f64; INDICATOR_NF] {
    let closes: Vec<f64> = data.iter().map(|b| b.close).collect();
    let c  = *closes.last().unwrap_or(&1.0);
    let e9 = ema_calc(&closes, 9);
    let e21= ema_calc(&closes, 21);
    let e9_d  = ((c - e9)  / c.max(1e-8)).clamp(-0.05, 0.05) / 0.05;
    let e21_d = ((c - e21) / c.max(1e-8)).clamp(-0.05, 0.05) / 0.05;
    let e9_sl = if closes.len() >= 2 {
        let now  = ema_calc(&closes, 9);
        let prev = ema_calc(&closes[..closes.len()-1], 9);
        ((now - prev) / prev.abs().max(1e-8)).clamp(-0.02, 0.02) / 0.02
    } else { 0.0 };
    let cross   = if e9 > e21 { 1.0 } else { -1.0 };
    let rsi14   = rsi_calc(&closes, 14) * 2.0 - 1.0;
    let (mh, ms)= macd_calc(&closes);
    let (pb, bw)= bollinger_calc(&closes, 20);
    let atr_r   = atr_ratio_calc(data, 14);
    let (sup,res,sr) = pivot_sr_calc(data);
    let (vd, vs)= vwap_calc(data);
    let obv     = obv_calc(data);
    let vol     = volume_ratio_calc(data);
    let bar     = data.last().unwrap();
    let body    = ((bar.close - bar.open) / (bar.high - bar.low).max(1e-8)).clamp(-1.0, 1.0);
    [e9_d, e21_d, e9_sl, cross, rsi14, mh, ms, pb, bw, atr_r, sup, res, sr, vd, vs, obv, vol, body]
}

const NF: usize = 7;

#[inline(always)]
fn extract(cur: &StockData, prev: &StockData) -> [f64; NF] {
    let eps = 1e-8;
    let pc = prev.close.max(eps);
    let cc = cur.close.max(eps);
    let pv = (prev.volume as f64).max(1.0);
    [
        ((cur.close - pc) / pc).clamp(-0.05, 0.05),
        ((cur.close - cur.open) / cc).clamp(-0.05, 0.05),
        ((cur.high - cc) / cc).clamp(0.0, 0.05),
        ((cc - cur.low) / cc).clamp(0.0, 0.05),
        ((cur.volume as f64 - pv) / pv).clamp(-3.0, 3.0),
        ((cur.high - cur.low) / cc).clamp(0.0, 0.05),
        ((cur.open - pc) / pc).clamp(-0.05, 0.05),
    ]
}

/// Rolling average true range — kept for potential future use.
#[allow(dead_code)]
fn atr(data: &[StockData], end: usize, n: usize) -> f64 {
    let start = end.saturating_sub(n);
    let count = (end - start).max(1);
    let sum: f64 = (start..end).map(|i| {
        let hi = data[i].high; let lo = data[i].low; let cc = data[i].close.max(1e-8);
        (hi - lo) / cc
    }).sum();
    (sum / count as f64).max(1e-6)
}

#[derive(Clone)]
struct Layer {
    in_size: usize, out_size: usize,
    w: Vec<f64>, w_t: Vec<f64>, b: Vec<f64>,
    is_output: bool,
}

impl Layer {
    fn new(in_size: usize, out_size: usize, is_output: bool, rng: &mut u64) -> Self {
        let limit = (6.0_f64 / (in_size + out_size) as f64).sqrt();
        let w: Vec<f64> = (0..in_size * out_size).map(|_| lcg(rng) * 2.0 * limit - limit).collect();
        let mut s = Layer { in_size, out_size, w_t: vec![0.0; in_size * out_size], w, b: vec![0.0; out_size], is_output };
        s.sync_transpose();
        s
    }
    fn sync_transpose(&mut self) {
        for o in 0..self.out_size {
            for i in 0..self.in_size { self.w_t[i * self.out_size + o] = self.w[o * self.in_size + i]; }
        }
    }
    fn forward(&self, inp: &[f64], out: &mut [f64]) {
        for o in 0..self.out_size {
            let off = o * self.in_size;
            let mut z = self.b[o] + self.w[off..off + self.in_size].iter().zip(inp).map(|(&w, &x)| w * x).sum::<f64>();
            out[o] = if self.is_output { z } else { z.tanh() };
        }
    }
}

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f64) / (u32::MAX as f64)
}

#[derive(Clone)]
struct Workspace {
    acts: Vec<Vec<f64>>, deltas: Vec<Vec<f64>>, errors: Vec<f64>,
    w_grad: Vec<Vec<f64>>, b_grad: Vec<Vec<f64>>,
}

impl Workspace {
    fn new(layers: &[Layer], input_size: usize) -> Self {
        let mut acts = vec![vec![0.0; input_size]];
        for l in layers { acts.push(vec![0.0; l.out_size]); }
        let deltas = layers.iter().map(|l| vec![0.0; l.out_size]).collect();
        let errors = vec![0.0; layers.last().map(|l| l.out_size).unwrap_or(1)];
        let w_grad = layers.iter().map(|l| vec![0.0; l.out_size * l.in_size]).collect();
        let b_grad = layers.iter().map(|l| vec![0.0; l.out_size]).collect();
        Workspace { acts, deltas, errors, w_grad, b_grad }
    }
}

struct Net {
    layers: Vec<Layer>,
    target_offsets: Vec<usize>,
    lookback: usize,
}

impl Net {
    fn new(target_offsets: Vec<usize>, n_extra: usize, cfg: &Config) -> Self {
        // input = (lookback-1) bar features + INDICATOR_NF technical signals + cascade extras
        let input_size = (cfg.lookback - 1) * NF + INDICATOR_NF + n_extra;
        let mut rng = 0xdeadbeef_cafebabe_u64;
        let mut layers = Vec::new();
        let mut prev = input_size;
        for _ in 0..cfg.layers {
            layers.push(Layer::new(prev, cfg.hidden, false, &mut rng));
            prev = cfg.hidden;
        }
        layers.push(Layer::new(prev, target_offsets.len(), true, &mut rng));
        Net { layers, target_offsets, lookback: cfg.lookback }
    }

    fn train<F>(&mut self, data: &[StockData], indicators: &[[f64; INDICATOR_NF]], name: &str, epochs: usize, lr_init: f64, cfg: &Config, cascade_fn: F) -> Vec<EpochMetrics>
    where F: Fn(usize) -> Vec<f64> + Sync {
        let max_off = *self.target_offsets.iter().max().unwrap_or(&0);
        let n_samples = data.len().saturating_sub(self.lookback + max_off);
        if n_samples == 0 { return vec![]; }

        let input_dim = self.layers[0].in_size;
        let target_dim = self.target_offsets.len();
        let mut flat_inputs = vec![0.0; n_samples * input_dim];
        let mut flat_targets = vec![0.0; n_samples * target_dim];

        for i in 0..n_samples {
            let anchor     = data[i + self.lookback - 1].close.max(1e-8);
            let mut inp = Vec::with_capacity(input_dim);
            for j in i + 1..i + self.lookback { inp.extend_from_slice(&extract(&data[j], &data[j - 1])); }
            inp.extend_from_slice(&indicators[i + self.lookback - 1]);
            inp.extend(cascade_fn(i));
            flat_inputs[i*input_dim..(i+1)*input_dim].copy_from_slice(&inp);
            for (ti, &off) in self.target_offsets.iter().enumerate() {
                let raw = (data[i+self.lookback+off].close - anchor) / anchor;
                // Raw percentage change, clamped to ±5% — natural scale for 1-min bars,
                // no ATR division which was blowing targets to ±8-9 and causing the
                // network to collapse to predicting zero for everything.
                flat_targets[i*target_dim + ti] = raw.clamp(-0.05, 0.05);
            }
        }

        let ws_proto = Workspace::new(&self.layers, input_dim);
        let mut lr = lr_init;
        let start_time = Instant::now();
        let mut history: Vec<EpochMetrics> = Vec::with_capacity(epochs);

        let warmup_epochs = (epochs / 10).max(5).min(20); // 10% of run, capped 5-20
        println!("━━━ Training {} ━━━  (warmup: {} epochs)", name, warmup_epochs);

        for epoch in 0..epochs {
            // ── Learning rate schedule ───────────────────────────────────────
            // Warmup phase: ramp from lr/10 up to lr_init over warmup_epochs.
            // This lets the network find a stable gradient direction before
            // taking full-size steps, preventing the "crash then flatline" pattern.
            // After warmup, decay normally each epoch.
            lr = if epoch < warmup_epochs {
                lr_init * (epoch + 1) as f64 / warmup_epochs as f64
            } else {
                lr_init * cfg.lr_decay.powi((epoch - warmup_epochs) as i32)
            };
            // ────────────────────────────────────────────────────────────────
            let epoch_start = Instant::now();
            let results = (0..n_samples).into_par_iter().step_by(cfg.batch_size)
                .fold(|| (ws_proto.clone(), 0.0, 0), |(mut ws, mut mse, mut correct), b_start| {
                    let b_end = (b_start + cfg.batch_size).min(n_samples);
                    for i in b_start..b_end {
                        ws.acts[0].copy_from_slice(&flat_inputs[i*input_dim..(i+1)*input_dim]);
                        for (l, layer) in self.layers.iter().enumerate() {
                            let (left, right) = ws.acts.split_at_mut(l + 1);
                            layer.forward(&left[l], &mut right[0]);
                        }
                        let targets = &flat_targets[i*target_dim..(i+1)*target_dim];
                        let last_idx = self.layers.len();
                        for (k, (&p, &t)) in ws.acts[last_idx].iter().zip(targets).enumerate() {
                            let e = p - t;
                            // Directional penalty: scaled by target magnitude so it stays
                            // proportional to the MSE error regardless of target scale.
                            // When signs disagree, push prediction toward the correct side
                            // by dir_weight × |target| — this way a 0.001% target produces
                            // a proportionally sized penalty, not a fixed 1.5 that drowns the signal.
                            let dir_penalty = if p * t < 0.0 {
                                cfg.dir_weight * t.abs() * t.signum() * -1.0
                            } else {
                                0.0
                            };
                            ws.errors[k] = e + dir_penalty;
                            mse += e * e;
                            if (p > 0.0 && t > 0.0) || (p < 0.0 && t < 0.0) { correct += 1; }
                        }
                        let last_layer = last_idx - 1;
                        for o in 0..self.layers[last_layer].out_size { ws.deltas[last_layer][o] = ws.errors[o]; }
                        for l in (0..last_layer).rev() {
                            let next = l + 1;
                            for o in 0..self.layers[l].out_size {
                                let off = o * self.layers[next].out_size;
                                let sum: f64 = self.layers[next].w_t[off..off + self.layers[next].out_size].iter().zip(&ws.deltas[next]).map(|(&w, &d)| w * d).sum();
                                ws.deltas[l][o] = sum * (1.0 - ws.acts[l+1][o].powi(2));
                            }
                        }
                        for l in 0..=last_layer {
                            let in_s = self.layers[l].in_size;
                            for o in 0..self.layers[l].out_size {
                                let d = ws.deltas[l][o]; let off = o * in_s;
                                for idx in 0..in_s { ws.w_grad[l][off + idx] += d * ws.acts[l][idx]; }
                                ws.b_grad[l][o] += d;
                            }
                        }
                    }
                    (ws, mse, correct)
                })
                .reduce(|| (ws_proto.clone(), 0.0, 0), |mut a, b| {
                    for l in 0..a.0.w_grad.len() {
                        for (wa, &wb) in a.0.w_grad[l].iter_mut().zip(&b.0.w_grad[l]) { *wa += wb; }
                        for (ba, &bb) in a.0.b_grad[l].iter_mut().zip(&b.0.b_grad[l]) { *ba += bb; }
                    }
                    (a.0, a.1 + b.1, a.2 + b.2)
                });

            // Apply Gradients
            for l in 0..self.layers.len() {
                let in_s = self.layers[l].in_size;
                for o in 0..self.layers[l].out_size {
                    let off = o * in_s;
                    for idx in 0..in_s { self.layers[l].w[off+idx] -= lr * results.0.w_grad[l][off+idx] / n_samples as f64; }
                    self.layers[l].b[o] -= lr * results.0.b_grad[l][o] / n_samples as f64;
                }
                self.layers[l].sync_transpose();
            }

            let epoch_loss = results.1 / (n_samples * target_dim) as f64;
            let accuracy   = (results.2 as f64 / (n_samples * target_dim) as f64) * 100.0;
            let epoch_secs = epoch_start.elapsed().as_secs_f64();

            if (epoch + 1) % (epochs / 10).max(1) == 0 {
                let phase = if epoch < warmup_epochs { "warmup" } else { "train " };
                println!("  Epoch {:>3}/{:<3} | Loss: {:.8} | Acc: {:>5.2}% | lr: {:.2e} | {:.2}s/ep [{}]",
                         epoch + 1, epochs, epoch_loss, accuracy, lr,
                         epoch_secs, phase);
            }
            history.push(EpochMetrics { loss: epoch_loss, accuracy_pct: accuracy, epoch_secs });
        }
        history
    }

    fn predict(&self, data: &[StockData], ind: &[f64; INDICATOR_NF], cascade: &[f64], anchor: f64) -> HashMap<usize, (f64, f64)> {
        let mut inp = Vec::with_capacity(self.layers[0].in_size);
        for i in 1..self.lookback { inp.extend_from_slice(&extract(&data[i], &data[i - 1])); }
        inp.extend_from_slice(ind);
        inp.extend_from_slice(cascade);
        let mut ws = Workspace::new(&self.layers, self.layers[0].in_size);
        ws.acts[0].copy_from_slice(&inp);
        for (l, layer) in self.layers.iter().enumerate() {
            let (left, right) = ws.acts.split_at_mut(l + 1);
            layer.forward(&left[l], &mut right[0]);
        }
        let preds = &ws.acts[self.layers.len()];
        // Predictions are raw percentage changes — no ATR conversion needed
        self.target_offsets.iter().enumerate().map(|(i, &off)| {
            let pct = preds[i];
            (off + 1, (pct, anchor * (1.0 + pct)))
        }).collect()
    }

    /// Serialize all layer weights + biases to a flat binary blob.
    ///
    /// Format (little-endian):
    ///   [n_layers: u64]
    ///   for each layer:
    ///     [in_size: u64] [out_size: u64] [is_output: u64]
    ///     [weights: in_size * out_size × f64]
    ///     [biases:  out_size × f64]
    pub fn serialize_weights(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        macro_rules! push {
            ($v:expr, u64) => { buf.extend_from_slice(&($v as u64).to_le_bytes()); };
            ($v:expr, f64) => { buf.extend_from_slice(&($v as f64).to_le_bytes()); };
        }
        push!(self.layers.len(), u64);
        for l in &self.layers {
            push!(l.in_size,     u64);
            push!(l.out_size,    u64);
            push!(l.is_output,   u64);
            for &w in &l.w { push!(w, f64); }
            for &b in &l.b { push!(b, f64); }
        }
        buf
    }

    /// Deserialize weights produced by `serialize_weights` back into `self`.
    /// Panics if the blob doesn't match the current layer topology.
    pub fn deserialize_weights(&mut self, blob: &[u8]) {
        let mut pos = 0usize;
        macro_rules! read_u64 {
            () => {{ let v = u64::from_le_bytes(blob[pos..pos+8].try_into().unwrap()); pos += 8; v }};
        }
        macro_rules! read_f64 {
            () => {{ let v = f64::from_le_bytes(blob[pos..pos+8].try_into().unwrap()); pos += 8; v }};
        }
        let n_layers = read_u64!() as usize;
        assert_eq!(n_layers, self.layers.len(), "Layer count mismatch in weight file");
        for l in &mut self.layers {
            let in_sz  = read_u64!() as usize;
            let out_sz = read_u64!() as usize;
            let is_out = read_u64!() != 0;
            assert_eq!(in_sz,  l.in_size,   "Layer in_size mismatch");
            assert_eq!(out_sz, l.out_size,  "Layer out_size mismatch");
            assert_eq!(is_out, l.is_output, "Layer is_output mismatch");
            for w in &mut l.w { *w = read_f64!(); }
            for b in &mut l.b { *b = read_f64!(); }
            l.sync_transpose();
        }
    }
}


// ─────────────────────────────────────────────
//  Weight persistence — save / load all three nets + Config
// ─────────────────────────────────────────────
//
//  Binary layout of the .weights file:
//    [magic: 8 bytes "CASC_W01"]
//    [config_json_len: u64]  [config_json: UTF-8 bytes]
//    [scout_blob_len:  u64]  [scout  weights blob]
//    [spotter_blob_len: u64] [spotter weights blob]
//    [sniper_blob_len:  u64] [sniper  weights blob]

fn save_all_weights(path: &str, cfg: &Config, scout: &Net, spotter: &Net, sniper: &Net) {
    use std::io::Write;
    let mut f = fs::File::create(path).expect("Cannot create weights file");
    f.write_all(b"CASC_W01").unwrap();
    let cfg_json  = serde_json::to_string(cfg).unwrap();
    let cfg_bytes = cfg_json.as_bytes();
    f.write_all(&(cfg_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write_all(cfg_bytes).unwrap();
    for net in &[scout, spotter, sniper] {
        let blob = net.serialize_weights();
        f.write_all(&(blob.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&blob).unwrap();
    }
    println!("✅ Weights saved to: {}", path);
}

fn parse_csv(path: &str) -> Vec<StockData> {
    let file = File::open(path).expect("Cannot open CSV");
    let mut rows = Vec::new();
    for (i, line) in io::BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; }
        let l = line.unwrap();
        let c: Vec<&str> = l.split(',').collect();
        if c.len() < 7 { continue; }
        let ts = chrono::DateTime::parse_from_str(&format!("{} +0000", c[0]), "%Y-%m-%d %H:%M:%S %z")
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        rows.push(StockData {
            _ts: ts,
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
//  Main (CPU Limit Added Here)
// ─────────────────────────────────────────────


// ─────────────────────────────────────────────
//  Auto-download: Twelve Data API
// ─────────────────────────────────────────────

const TWELVE_DATA_URL: &str = "https://api.twelvedata.com/time_series";
const RATE_LIMIT_SECS: u64  = 8;
const POINTS_PER_CALL: usize = 1950;

fn data_file_for(symbol: &str) -> String { format!("{}_data.csv", symbol) }

fn load_existing_timestamps(filepath: &str) -> HashSet<String> {
    let mut ts = HashSet::new();
    if !Path::new(filepath).exists() { return ts; }
    let file = match std::fs::File::open(filepath) { Ok(f) => f, Err(_) => return ts };
    for (i, line) in io::BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; }
        if let Ok(l) = line {
            if let Some(t) = l.split(',').next() { ts.insert(t.trim().to_string()); }
        }
    }
    println!("  Found {} existing rows in {}", ts.len(), filepath);
    ts
}

fn week_already_downloaded(monday: NaiveDate, existing: &HashSet<String>) -> bool {
    for minute in 0..10_u32 {
        let ts = format!("{} 13:{:02}:00", monday, 30 + minute);
        if existing.contains(&ts) { return true; }
    }
    false
}

fn get_mondays(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut mondays = Vec::new();
    let mut cur = start;
    while cur.weekday().num_days_from_monday() != 0 { cur += Duration::days(1); }
    while cur <= end { mondays.push(cur); cur += Duration::weeks(1); }
    mondays
}

fn fetch_week(symbol: &str, monday: NaiveDate, api_key: &str) -> Vec<[String; 7]> {
    let friday    = monday + Duration::days(4);
    let start_str = format!("{} 00:00:00", monday).replace(' ', "%20");
    let end_str   = format!("{} 23:59:59", friday).replace(' ', "%20");
    let url = format!(
        "{url}?symbol={sym}&interval=1min&start_date={s}&end_date={e}&outputsize={n}&apikey={k}&format=JSON&timezone=UTC",
        url = TWELVE_DATA_URL, sym = symbol, s = start_str, e = end_str,
        n = POINTS_PER_CALL, k = api_key
    );
    let resp = match reqwest::blocking::get(&url) {
        Ok(r) => r, Err(e) => { eprintln!("  Request failed: {}", e); return vec![]; }
    };
    let json: serde_json::Value = match resp.json() {
        Ok(j) => j, Err(e) => { eprintln!("  JSON parse error: {}", e); return vec![]; }
    };
    if json["status"] == "error" {
        eprintln!("  API error: {}", json["message"].as_str().unwrap_or("?")); return vec![];
    }
    let values = match json["values"].as_array() {
        Some(v) => v, None => { println!("  No data (holiday week?)"); return vec![]; }
    };
    let mut bars: Vec<[String; 7]> = values.iter().map(|v| {
        let s = |k: &str| v[k].as_str().unwrap_or("0").to_string();
        [s("datetime"), symbol.to_string(), s("open"), s("high"), s("low"), s("close"), s("volume")]
    }).collect();
    bars.sort_by(|a, b| a[0].cmp(&b[0]));
    bars
}

fn save_bars(new_bars: &[[String; 7]], filepath: &str, existing: &HashSet<String>) -> usize {
    let filtered: Vec<&[String; 7]> = new_bars.iter().filter(|b| !existing.contains(&b[0])).collect();
    if filtered.is_empty() { println!("  All rows already present."); return 0; }
    let mut all: Vec<Vec<String>> = Vec::new();
    if Path::new(filepath).exists() {
        let f = std::fs::File::open(filepath).expect("open");
        for (i, line) in io::BufReader::new(f).lines().enumerate() {
            if i == 0 { continue; }
            if let Ok(l) = line {
                let cols: Vec<String> = l.split(',').map(|s| s.trim().to_string()).collect();
                if cols.len() >= 7 { all.push(cols); }
            }
        }
    }
    for b in &filtered { all.push(b.to_vec()); }
    all.sort_by(|a, b| a[0].cmp(&b[0]));
    let mut f = std::fs::File::create(filepath).expect("create");
    use std::io::Write;
    writeln!(f, "Timestamp,Symbol,Open,High,Low,Close,Volume").unwrap();
    for row in &all { writeln!(f, "{}", row.join(",")).unwrap(); }
    println!("  Saved {} new rows  ({} total)", filtered.len(), all.len());
    filtered.len()
}

/// Download data if --start-date / --end-date / --api-key are provided.
/// Returns the resolved CSV path to load (may be auto-generated).
fn maybe_download(symbol: &str, start: &str, end: &str, api_key: &str) -> String {
    let filepath = data_file_for(symbol);
    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .expect("--start-date must be YYYY-MM-DD");
    let end_date = NaiveDate::parse_from_str(end, "%Y-%m-%d")
        .expect("--end-date must be YYYY-MM-DD");
    let today    = chrono::Local::now().date_naive();
    let mut existing = load_existing_timestamps(&filepath);
    let mondays  = get_mondays(start_date, end_date);

    println!("Auto-download: {} weeks to check for {}", mondays.len(), symbol);
    let mut total_new = 0usize;
    let mut calls     = 0usize;

    for (i, &monday) in mondays.iter().enumerate() {
        print!("[Week {:>3}/{}]  {}  ", i + 1, mondays.len(), monday);
        if week_already_downloaded(monday, &existing) { println!("already downloaded."); continue; }
        if monday > today { println!("future, skipping."); continue; }
        println!("fetching...");
        let bars = fetch_week(symbol, monday, api_key);
        calls += 1;
        if !bars.is_empty() {
            let added = save_bars(&bars, &filepath, &existing);
            total_new += added;
            for b in &bars { existing.insert(b[0].clone()); }
        }
        let remaining = mondays[i + 1..].iter()
            .filter(|&&m| !week_already_downloaded(m, &existing) && m <= today)
            .count();
        if remaining > 0 {
            println!("  Waiting {}s (rate limit)...", RATE_LIMIT_SECS);
            thread::sleep(std::time::Duration::from_secs(RATE_LIMIT_SECS));
        }
    }
    println!("Download complete: {} API calls, {} new rows, file: {}\n", calls, total_new, filepath);
    filepath
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

fn main() -> io::Result<()> {
    // Limit CPU to 75%
    let total_cores = num_cpus::get();
    let cap = (total_cores as f64 * 0.75).floor() as usize;
    rayon::ThreadPoolBuilder::new().num_threads(cap.max(1)).build_global().unwrap();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cascade_trainer <csv_or_symbol> [flags]");
        println!();
        println!("  Training flags:");
        println!("    --lookback N      Bars of history per model       [60]");
        println!("    --hidden N        Neurons per hidden layer         [128]");
        println!("    --layers N        Hidden layers per model          [2]");
        println!("    --epochs1/2/3 N   Epochs per cascade stage        [300]");
        println!("    --lr1/2/3 F       Learning rate per stage         [0.001]");
        println!("    --lr-decay F      LR multiplier per epoch         [0.997]");
        println!("    --batch-size N    Samples per gradient update      [256]");
        println!("    --dir-weight F    Directional penalty weight       [0.3]");
        println!("    --out-prefix S    Label for this run               [model]");
        println!("    --save-weights F  Save trained weights to this file");
        println!();
        println!("  Auto-download flags (all three required together):");
        println!("    --start-date YYYY-MM-DD   History start date");
        println!("    --end-date   YYYY-MM-DD   History end date");
        println!("    --api-key    KEY           Twelve Data API key");
        println!("    (first arg must be the ticker symbol, e.g. AAPL)");
        println!();
        println!("  Example (train only):");
        println!("    cascade_trainer AAPL_data.csv --lookback 120 --hidden 256 --layers 2 \\");
        println!("        --epochs1 50 --epochs2 50 --epochs3 50 --lr1 0.001 --dir-weight 1.5 \\");
        println!("        --out-prefix AAPL");
        println!();
        println!("  Example (download + train):");
        println!("    cascade_trainer AAPL --start-date 2024-01-01 --end-date 2026-03-09 \\");
        println!("        --api-key YOUR_KEY --lookback 120 --out-prefix AAPL");
        return Ok(());
    }

    let mut cfg = if let Ok(c) = fs::read_to_string("trainer_config.json") {
        serde_json::from_str(&c).unwrap_or_default()
    } else {
        Config::default()
    };

    // Parse all flags
    let mut start_date        = String::new();
    let mut end_date          = String::new();
    let mut api_key           = String::new();
    let mut save_weights_path = String::new();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--lookback"       => { cfg.lookback   = args[i+1].parse().unwrap(); i += 2; }
            "--hidden"         => { cfg.hidden     = args[i+1].parse().unwrap(); i += 2; }
            "--layers"         => { cfg.layers     = args[i+1].parse().unwrap(); i += 2; }
            "--epochs1"        => { cfg.epochs1    = args[i+1].parse().unwrap(); i += 2; }
            "--epochs2"        => { cfg.epochs2    = args[i+1].parse().unwrap(); i += 2; }
            "--epochs3"        => { cfg.epochs3    = args[i+1].parse().unwrap(); i += 2; }
            "--lr1"            => { cfg.lr1        = args[i+1].parse().unwrap(); i += 2; }
            "--lr2"            => { cfg.lr2        = args[i+1].parse().unwrap(); i += 2; }
            "--lr3"            => { cfg.lr3        = args[i+1].parse().unwrap(); i += 2; }
            "--lr-decay"       => { cfg.lr_decay   = args[i+1].parse().unwrap(); i += 2; }
            "--batch-size"     => { cfg.batch_size = args[i+1].parse().unwrap(); i += 2; }
            "--dir-weight"     => { cfg.dir_weight = args[i+1].parse().unwrap(); i += 2; }
            "--out-prefix"     => { cfg.out_prefix = args[i+1].clone();          i += 2; }
            "--start-date"     => { start_date     = args[i+1].clone();          i += 2; }
            "--end-date"       => { end_date        = args[i+1].clone();          i += 2; }
            "--api-key"        => { api_key         = args[i+1].clone();          i += 2; }
            "--save-weights"   => { save_weights_path = args[i+1].clone();       i += 2; }
            _                  => { i += 1; }
        }
    }
    fs::write("trainer_config.json", serde_json::to_string_pretty(&cfg).unwrap()).ok();

    // Resolve CSV path — auto-download if date range + key are provided
    let csv_path = if !start_date.is_empty() && !end_date.is_empty() && !api_key.is_empty() {
        // First arg is a ticker symbol, not a file
        let symbol = args[1].to_uppercase();
        maybe_download(&symbol, &start_date, &end_date, &api_key)
    } else {
        // First arg is a CSV path directly
        args[1].clone()
    };

    println!("CPU capped at 75% ({}/{} threads)", cap, total_cores);
    println!("Loading {} ...", csv_path);

    let data = parse_csv(&csv_path);
    if data.len() < cfg.lookback + 10 {
        println!("Not enough data ({} rows, need >= {}).", data.len(), cfg.lookback + 10);
        return Ok(());
    }
    println!("Loaded {} rows.\n", data.len());

    // Precompute all indicator snapshots once
    println!("Precomputing technical indicators...");
    let t_ind = Instant::now();
    let indicators: Vec<[f64; INDICATOR_NF]> = (0..data.len())
        .map(|i| {
            let start = i.saturating_sub(cfg.lookback - 1);
            compute_indicators(&data[start..=i])
        })
        .collect();
    println!("  Done in {:.2?} ({} snapshots)\n", t_ind.elapsed(), indicators.len());

    let r_end    = data.len() - 10;
    let train_d  = &data[..r_end];
    let live_w   = &data[r_end - cfg.lookback..r_end];
    let anchor   = live_w.last().unwrap().close;
    let live_ind = &indicators[r_end - 1];

    let mut scout = Net::new(vec![9], 0, &cfg);
    let scout_history = scout.train(train_d, &indicators, "SCOUT (+10m)", cfg.epochs1, cfg.lr1, &cfg, |_| vec![]);
    let s_val = scout.predict(live_w, live_ind, &[], anchor).get(&10).unwrap().0;

    let mut spotter = Net::new(vec![0, 4, 9], 1, &cfg);
    let spotter_history = spotter.train(train_d, &indicators, "SPOTTER (+1,+5,+10m)", cfg.epochs2, cfg.lr2, &cfg, |_| vec![s_val]);
    let sp_live = spotter.predict(live_w, live_ind, &[s_val], anchor);
    let sp_v = [
        sp_live.get(&1).unwrap().0,
        sp_live.get(&5).unwrap().0,
        sp_live.get(&10).unwrap().0,
    ];

    let mut sniper = Net::new((0..10).collect(), 4, &cfg);
    let sniper_history = sniper.train(train_d, &indicators, "SNIPER (+1..10m)", cfg.epochs3, cfg.lr3, &cfg, |_| vec![s_val, sp_v[0], sp_v[1], sp_v[2]]);
    let sn_live = sniper.predict(live_w, live_ind, &[s_val, sp_v[0], sp_v[1], sp_v[2]], anchor);

    // Save weights if --save-weights was specified
    if !save_weights_path.is_empty() {
        save_all_weights(&save_weights_path, &cfg, &scout, &spotter, &sniper);
    }

    println!("━━━ [{}] Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", cfg.out_prefix);
    println!("  Min | Scout (+10)      | Spotter (1,5,10) | Sniper (1-10)    | Actual");
    println!("  ----|------------------|------------------|------------------|--------");
    for m in 1..=10 {
        let get_fmt = |map: &HashMap<usize, (f64, f64)>, min: usize| {
            map.get(&min)
               .map(|(p, d)| format!("{:>+6.3}% /${:<7.2}", p * 100.0, d))
               .unwrap_or("      --        ".into())
        };
        let actual_pct = (data[r_end + m - 1].close - anchor) / anchor;
        println!("  {:>2}  | {:<16} | {:<16} | {:<16} | {:>+6.3}%",
            m,
            get_fmt(&scout.predict(live_w, live_ind, &[], anchor), m),
            get_fmt(&sp_live, m),
            get_fmt(&sn_live, m),
            actual_pct * 100.0);
    }

    let mut tracker = TuningTracker::new("AI_Tuning_Log.xlsx");
    let run_label = format!("lr={} decay={} dw={} l={} h={} lb={}",
        cfg.lr1, cfg.lr_decay, cfg.dir_weight, cfg.layers, cfg.hidden, cfg.lookback);
    tracker.add_run("Scout",   RunRecord { run_label: run_label.clone(), cfg: cfg.clone(), epochs: scout_history });
    tracker.add_run("Spotter", RunRecord { run_label: run_label.clone(), cfg: cfg.clone(), epochs: spotter_history });
    tracker.add_run("Sniper",  RunRecord { run_label: run_label.clone(), cfg: cfg.clone(), epochs: sniper_history });
    match tracker.save() {
        Ok(_)  => println!("\n📊 Excel charts saved to AI_Tuning_Log.xlsx"),
        Err(e) => eprintln!("\n⚠  Could not save Excel: {}", e),
    }

    Ok(())
}
