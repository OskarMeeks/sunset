
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
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
            ("Bar Mins",    format!("{}", cfg.bar_mins)),
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
    bar_mins: usize,   // candle size in minutes (1 = 1-min bars, 5 = 5-min bars)
    dir_weight: f64,   // directional penalty weight (0.0 = pure MSE, 0.3 = default)
    l2_lambda: f64,    // L2 weight decay — penalises large weights to reduce overfitting
    early_stop_patience: usize, // stop training if directional acc doesn't improve for N epochs
    out_prefix: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lookback: 60, hidden: 128, layers: 2, batch_size: 256, lr_decay: 0.997,
            epochs1: 300, epochs2: 300, epochs3: 300,
            lr1: 3e-4, lr2: 3e-4, lr3: 3e-4,
            // dir_weight scales the MSE magnitude term relative to the BCE direction term.
            // 1.5 keeps magnitude refinement meaningful without drowning the BCE signal.
            // Old value of 0.3 let MSE dominate once hinge cut off, causing the 49% plateau.
            bar_mins: 5,
            dir_weight: 1.5,
            l2_lambda: 1e-4,
            early_stop_patience: 15,
            out_prefix: "default".into(),
        }
    }
}

// ─────────────────────────────────────────────
//  Neural Structures (Optimized)
// ─────────────────────────────────────────────



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
    let eps = 1e-8;
    let pc = prev.close.max(eps);
    let cc = cur.close.max(eps);
    let pv = (prev.volume as f64).max(1.0);
    [
        ((cur.close - pc) / pc).clamp(-0.05, 0.05),
        ((cur.close - cur.open) / cc).clamp(-0.05, 0.05),
        ((cur.high - cc) / cc).clamp(0.0, 0.05),
        ((cc - cur.low) / cc).clamp(0.0, 0.05),
        ((cur.volume as f64 - pv) / pv).clamp(-1.0, 1.0),
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
        // Output layer uses tiny init so predictions start near zero (targets are ±0.05).
        // Hidden layers use Xavier uniform for stable gradient flow.
        let limit = if is_output {
            0.01
        } else {
            (6.0_f64 / (in_size + out_size) as f64).sqrt()
        };
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
            let z = self.b[o] + self.w[off..off + self.in_size].iter().zip(inp).map(|(&w, &x)| w * x).sum::<f64>();
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

        // ── Class balance weights ─────────────────────────────────────────────
        // Scan flat_targets to count up/down moves per target dimension.
        // Upweight the minority class so the model doesn't learn to always predict
        // the majority direction (bullish bias when training on trending data).
        println!("  Class balance scan for '{}':", name);
        let class_weight_pairs: Vec<(f64, f64)> = (0..target_dim).map(|ti| {
            let pos = (0..n_samples).filter(|&i| flat_targets[i*target_dim + ti] > 0.0).count();
            let neg = (0..n_samples).filter(|&i| flat_targets[i*target_dim + ti] < 0.0).count();
            let total = (pos + neg).max(1);
            // Inverse frequency weighting: minority class gets weight > 1, majority < 1.
            // Clamped to [0.5, 2.0] so a heavily skewed dataset can't destabilize training.
            let pos_w = (neg as f64 / pos.max(1) as f64).clamp(0.5, 2.0);
            let neg_w = (pos as f64 / neg.max(1) as f64).clamp(0.5, 2.0);
            let _ = total;
            println!("    +{}m  {}↑  {}↓  (up_w={:.3}  dn_w={:.3})",
                (self.target_offsets[ti] + 1) * cfg.bar_mins, pos, neg, pos_w, neg_w);
            (pos_w, neg_w)
        }).collect();

        let ws_proto = Workspace::new(&self.layers, input_dim);
        let mut lr;
        let mut history: Vec<EpochMetrics> = Vec::with_capacity(epochs);

        let warmup_epochs = (epochs / 10).max(5).min(20); // 10% of run, capped 5-20
        println!("━━━ Training {} ━━━  (warmup: {} epochs)", name, warmup_epochs);

        // ── Early stopping state ──────────────────────────────────────────────
        // Track best LOSS for weight restoration and early stop.
        // Accuracy on tiny ±0.1% returns stays at ~49% regardless of training
        // quality — MSE drives outputs toward 0 which has random directional
        // accuracy. Tracking accuracy restores weights from epoch 11 (large
        // chaotic outputs, ±26% predictions). Loss always improves meaningfully
        // so we restore at lowest-MSE weights, giving small-magnitude predictions.
        let mut best_loss: f64        = f64::MAX;
        let mut epochs_no_improve     = 0usize;
        let mut best_weights: Vec<(Vec<f64>, Vec<f64>)> =
            self.layers.iter().map(|l| (l.w.clone(), l.b.clone())).collect();

        // ── Shuffled index array — permuted each epoch ────────────────────────
        let mut indices: Vec<usize> = (0..n_samples).collect();

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

            // ── Fisher-Yates shuffle (deterministic but different each epoch) ─
            {
                let mut rng = epoch as u64 ^ 0xdeadbeef_cafe_u64;
                for i in (1..n_samples).rev() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let j = (rng >> 33) as usize % (i + 1);
                    indices.swap(i, j);
                }
            }

            let epoch_start = Instant::now();
            let results = (0..n_samples).into_par_iter().step_by(cfg.batch_size)
                .fold(|| (ws_proto.clone(), 0.0, 0), |(mut ws, mut mse, mut correct), b_start| {
                    let b_end = (b_start + cfg.batch_size).min(n_samples);
                    for pos in b_start..b_end {
                        let i = indices[pos];
                        ws.acts[0].copy_from_slice(&flat_inputs[i*input_dim..(i+1)*input_dim]);
                        for (l, layer) in self.layers.iter().enumerate() {
                            let (left, right) = ws.acts.split_at_mut(l + 1);
                            layer.forward(&left[l], &mut right[0]);
                        }
                        let targets = &flat_targets[i*target_dim..(i+1)*target_dim];
                        let last_idx = self.layers.len();
                        for (k, (&p, &t)) in ws.acts[last_idx].iter().zip(targets).enumerate() {
                            let e = p - t;








                            // MSE gradient + directional penalty (proportional to target magnitude).
                            // dir_penalty pushes prediction toward the correct sign when wrong,
                            // scaled by |t| so small targets produce small corrections and large
                            // targets produce proportionally larger ones — no saturation, no explosion.
                            let dir_penalty = if p * t < 0.0 {
                                cfg.dir_weight * t.abs() * t.signum() * -1.0
                            } else {
                                0.0
                            };
                            // Class balance: upweight minority direction
                            let (pos_w, neg_w) = class_weight_pairs[k];
                            let class_w = if t > 0.0 { pos_w } else { neg_w };

                            ws.errors[k] = (e + dir_penalty) * class_w;
                            mse += e * e;
                            if (p > 0.0 && t > 0.0) || (p < 0.0 && t < 0.0) { correct += 1; }
                        }
                        let last_layer = last_idx - 1;
                        for o in 0..self.layers[last_layer].out_size { ws.deltas[last_layer][o] = ws.errors[o]; }
                        for l in (0..last_layer).rev() {
                            let next = l + 1;
                            let next_in  = self.layers[next].in_size;   // = layers[l].out_size
                            let next_out = self.layers[next].out_size;
                            for o in 0..self.layers[l].out_size {
                                // Correct backprop: sum over every output neuron j of next
                                // layer, using w[next][j * next_in + o]  (column o of W_next).
                                // The old code used w_t[o * next_out .. o * next_out + next_out]
                                // which was a *row* of w_t — i.e. a row of W — not a column.
                                let sum: f64 = (0..next_out)
                                    .map(|j| self.layers[next].w[j * next_in + o] * ws.deltas[next][j])
                                    .sum();
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

            // ── Apply gradients with L2 weight decay ─────────────────────────
            // L2 shrinks weights toward zero each step, preventing memorisation
            // of training-sequence-specific patterns.
            for l in 0..self.layers.len() {
                let in_s = self.layers[l].in_size;
                let l2_factor = 1.0 - lr * cfg.l2_lambda;
                for o in 0..self.layers[l].out_size {
                    let off = o * in_s;
                    for idx in 0..in_s {
                        self.layers[l].w[off+idx] = self.layers[l].w[off+idx] * l2_factor
                            - lr * results.0.w_grad[l][off+idx] / n_samples as f64;
                    }
                    self.layers[l].b[o] -= lr * results.0.b_grad[l][o] / n_samples as f64;
                }
                self.layers[l].sync_transpose();
            }

            let epoch_loss = results.1 / (n_samples * target_dim) as f64;
            let accuracy   = (results.2 as f64 / (n_samples * target_dim) as f64) * 100.0;
            let epoch_secs = epoch_start.elapsed().as_secs_f64();

            // ── Early stopping — track best loss ─────────────────────────────
            if epoch >= warmup_epochs {
                if epoch_loss < best_loss {
                    best_loss = epoch_loss;
                    epochs_no_improve = 0;
                    best_weights = self.layers.iter().map(|l| (l.w.clone(), l.b.clone())).collect();
                } else {
                    epochs_no_improve += 1;
                }
                if cfg.early_stop_patience > 0 && epochs_no_improve >= cfg.early_stop_patience {
                    let phase = "stopped";
                    println!("  Epoch {:>3}/{:<3} | Loss: {:.8} | Acc: {:>5.2}% | lr: {:.2e} | {:.2}s/ep [{}]",
                             epoch + 1, epochs, epoch_loss, accuracy, lr, epoch_secs, phase);
                    println!("  ⏹  Early stop — no loss improvement for {} epochs (best loss {:.8})",
                        cfg.early_stop_patience, best_loss);
                    history.push(EpochMetrics { loss: epoch_loss, accuracy_pct: accuracy, epoch_secs });
                    break;
                }
            }

            if (epoch + 1) % (epochs / 10).max(1) == 0 {
                let phase = if epoch < warmup_epochs { "warmup" } else { "train " };
                println!("  Epoch {:>3}/{:<3} | Loss: {:.8} | Acc: {:>5.2}% | lr: {:.2e} | {:.2}s/ep [{}]",
                         epoch + 1, epochs, epoch_loss, accuracy, lr,
                         epoch_secs, phase);
            }
            history.push(EpochMetrics { loss: epoch_loss, accuracy_pct: accuracy, epoch_secs });
        }

        // ── Restore best weights found during training ────────────────────────
        for (l, (w, b)) in self.layers.iter_mut().zip(best_weights) {
            l.w = w;
            l.b = b;
            l.sync_transpose();
        }
        println!("  ✓ Restored best weights (loss {:.8})", best_loss);

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



// ─────────────────────────────────────────────
//  Main (CPU Limit Added Here)

// ─────────────────────────────────────────────
//  Data fetching — delegated to generator module
// ─────────────────────────────────────────────

mod generator;
use generator::{maybe_download, parse_csv, StockData};

fn main() -> io::Result<()> {
    // Limit CPU to 75%
    let total_cores = num_cpus::get();
    let cap = (total_cores as f64 * 0.75).floor() as usize;
    rayon::ThreadPoolBuilder::new().num_threads(cap.max(1)).build_global().unwrap();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cascade_trainer <csv_or_symbol> [flags]");
        println!("       cascade_trainer --symbols \"AAPL,MSFT,NVDA\" [flags]  (batch mode)");
        println!();
        println!("  Training flags:");
        println!("    --lookback N      Bars of history per model       [60]");
        println!("    --hidden N        Neurons per hidden layer         [128]");
        println!("    --layers N        Hidden layers per model          [2]");
        println!("    --epochs1/2/3 N   Epochs per cascade stage        [300]");
        println!("    --lr1/2/3 F       Learning rate per stage         [0.001]");
        println!("    --lr-decay F      LR multiplier per epoch         [0.997]");
        println!("    --batch-size N    Samples per gradient update      [256]");
        println!("    --dir-weight F    MSE magnitude weight vs BCE direction  [1.5]");
        println!("    --l2-lambda F     L2 weight decay (0=off)          [1e-4]");
        println!("    --early-stop N    Stop if acc flat for N epochs    [15]");
        println!("    --out-prefix S    Label for this run               [model]");
        println!("    --save-weights F  Save trained weights to this file");
        println!();
        println!("  Batch mode (trains multiple symbols sequentially):");
        println!("    --symbols SYM1,SYM2,...  Comma-separated tickers");
        println!("    Each symbol trains from {{SYM}}_data.csv → saves {{SYM}}.weights");
        println!();
        println!("  Auto-download flags (all three required together):");
        println!("    --start-date YYYY-MM-DD   History start date");
        println!("    --end-date   YYYY-MM-DD   History end date");
        println!("    --api-key    KEY           Twelve Data API key");
        println!("    (first arg must be the ticker symbol when not using --symbols)");
        println!();
        println!("  Single symbol examples:");
        println!("    cascade_trainer AAPL_data.csv --lookback 120 --hidden 128 --layers 3 \\");
        println!("        --epochs1 75 --epochs2 75 --epochs3 75 --lr1 3e-4 \\");
        println!("        --l2-lambda 1e-4 --early-stop 15 --out-prefix AAPL --save-weights AAPL.weights");
        println!();
        println!("    cascade_trainer AAPL --start-date 2024-01-01 --end-date 2026-03-09 \\");
        println!("        --api-key YOUR_KEY --lookback 120 --out-prefix AAPL --save-weights AAPL.weights");
        println!();
        println!("  Batch mode examples:");
        println!("    cascade_trainer . --symbols \"AAPL,MSFT,NVDA,TSLA,GOOGL\" \\");
        println!("        --lookback 120 --hidden 128 --layers 3 \\");
        println!("        --epochs1 75 --epochs2 75 --epochs3 75 --lr1 3e-4 \\");
        println!("        --l2-lambda 1e-4 --early-stop 15");
        println!();
        println!("    cascade_trainer . --symbols \"AAPL,MSFT,NVDA,TSLA,GOOGL\" \\");
        println!("        --start-date 2024-01-01 --end-date 2026-03-09 --api-key YOUR_KEY \\");
        println!("        --lookback 120 --l2-lambda 1e-4 --early-stop 15");
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
    let mut interval_mins: usize = 5;
    let mut symbols_str:  String = String::new(); // --symbols "AAPL,MSFT,NVDA" for batch training

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--lookback"          => { cfg.lookback             = args[i+1].parse().unwrap(); i += 2; }
            "--hidden"            => { cfg.hidden               = args[i+1].parse().unwrap(); i += 2; }
            "--layers"            => { cfg.layers               = args[i+1].parse().unwrap(); i += 2; }
            "--epochs1"           => { cfg.epochs1              = args[i+1].parse().unwrap(); i += 2; }
            "--epochs2"           => { cfg.epochs2              = args[i+1].parse().unwrap(); i += 2; }
            "--epochs3"           => { cfg.epochs3              = args[i+1].parse().unwrap(); i += 2; }
            "--lr1"               => { cfg.lr1                  = args[i+1].parse().unwrap(); i += 2; }
            "--lr2"               => { cfg.lr2                  = args[i+1].parse().unwrap(); i += 2; }
            "--lr3"               => { cfg.lr3                  = args[i+1].parse().unwrap(); i += 2; }
            "--lr-decay"          => { cfg.lr_decay             = args[i+1].parse().unwrap(); i += 2; }
            "--batch-size"        => { cfg.batch_size           = args[i+1].parse().unwrap(); i += 2; }
            "--dir-weight"        => { cfg.dir_weight           = args[i+1].parse().unwrap(); i += 2; }
            "--l2-lambda"         => { cfg.l2_lambda            = args[i+1].parse().unwrap(); i += 2; }
            "--early-stop"        => { cfg.early_stop_patience  = args[i+1].parse().unwrap(); i += 2; }
            "--out-prefix"        => { cfg.out_prefix           = args[i+1].clone();          i += 2; }
            "--start-date"        => { start_date               = args[i+1].clone();          i += 2; }
            "--end-date"          => { end_date                 = args[i+1].clone();          i += 2; }
            "--api-key"           => { api_key                  = args[i+1].clone();          i += 2; }
            "--save-weights"      => { save_weights_path        = args[i+1].clone();          i += 2; }
            "--interval"          => { interval_mins = args[i+1].parse().unwrap_or(5); cfg.bar_mins = interval_mins; i += 2; }
            "--symbols"           => { symbols_str              = args[i+1].clone();          i += 2; }
            _                     => { i += 1; }
        }
    }
    fs::write("trainer_config.json", serde_json::to_string_pretty(&cfg).unwrap()).ok();

    // ── Multi-symbol batch training mode ─────────────────────────────────────
    // Usage: cascade_trainer --symbols "AAPL,MSFT,NVDA,TSLA,GOOGL" [shared flags]
    //        --start-date 2024-01-01 --end-date 2026-03-09 --api-key KEY
    //
    // For each symbol, this will:
    //   1. Download (or reuse) {SYM}_data.csv
    //   2. Train Scout → Spotter → Sniper with the shared config
    //   3. Save weights to {SYM}.weights
    //
    // The first positional arg is ignored in this mode — supply a placeholder
    // like "." or just omit it by using the flag form shown above.
    cfg.bar_mins = interval_mins;
    if !symbols_str.is_empty() {
        let sym_list: Vec<&str> = symbols_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        println!("━━━ Batch training {} symbols: {} ━━━", sym_list.len(), sym_list.join(", "));

        for sym in &sym_list {
            let sym_upper = sym.to_uppercase();
            println!("\n{}", "═".repeat(60));
            println!("  Symbol: {}  ({}/{})", sym_upper, sym_list.iter().position(|s| s == sym).unwrap_or(0) + 1, sym_list.len());
            println!("{}", "═".repeat(60));

            // ── Resolve CSV — download if date range + key provided ───────────
            let csv_path = if !start_date.is_empty() && !end_date.is_empty() && !api_key.is_empty() {
                maybe_download(&sym_upper, &start_date, &end_date, &api_key)
            } else {
                let auto = format!("{}_data.csv", sym_upper);
                if std::path::Path::new(&auto).exists() {
                    auto
                } else {
                    eprintln!("  ⚠  Skipping {} — no data file and no --api-key/--start-date/--end-date provided", sym_upper);
                    continue;
                }
            };

            println!("  Loading {} ...", csv_path);
            let data = {
                let raw = parse_csv(&csv_path);
                if interval_mins > 1 {
                    println!("  Resampling to {}-min bars...", interval_mins);
                    let r = resample(raw, interval_mins);
                    println!("  {} bars after resampling", r.len());
                    r
                } else { raw }
            };
            if data.len() < cfg.lookback + 10 {
                eprintln!("  ⚠  Skipping {} — only {} rows (need >= {})", sym_upper, data.len(), cfg.lookback + 10);
                continue;
            }
            println!("  Loaded {} rows.", data.len());

            // Per-symbol config: override prefix so weights file is named {SYM}.weights
            let mut sym_cfg = cfg.clone();
            sym_cfg.out_prefix = sym_upper.clone();

            println!("  Precomputing technical indicators...");
            let t_ind = Instant::now();
            let indicators: Vec<[f64; INDICATOR_NF]> = (0..data.len())
                .map(|i| { let start = i.saturating_sub(sym_cfg.lookback - 1); compute_indicators(&data[start..=i]) })
                .collect();
            println!("  Done in {:.2?} ({} snapshots)", t_ind.elapsed(), indicators.len());

            let r_end   = data.len() - 10;
            let train_d = &data[..r_end];
            let live_w  = &data[r_end - sym_cfg.lookback..r_end];
            let anchor  = live_w.last().unwrap().close;
            let live_ind = &indicators[r_end - 1];

            let mut scout = Net::new(vec![9], 0, &sym_cfg);
            sym_cfg.bar_mins = interval_mins;
            let scout_history = scout.train(train_d, &indicators, &format!("SCOUT (+{}m) [{}]", 10 * sym_cfg.bar_mins, sym_upper),
                sym_cfg.epochs1, sym_cfg.lr1, &sym_cfg, |_| vec![]);
            let s_val = scout.predict(live_w, live_ind, &[], anchor).get(&10).unwrap().0;

            // Pre-compute per-sample cascade predictions (see single-symbol path for rationale).
            let n_cascade = train_d.len().saturating_sub(sym_cfg.lookback + 9);
            let scout_train_preds: Vec<f64> = (0..n_cascade).map(|i| {
                let w   = &train_d[i..i + sym_cfg.lookback];
                let ind = &indicators[i + sym_cfg.lookback - 1];
                scout.predict(w, ind, &[], train_d[i + sym_cfg.lookback - 1].close)
                     .get(&10).unwrap().0
            }).collect();

            let mut spotter = Net::new(vec![0, 4, 9], 1, &sym_cfg);
            let spotter_history = spotter.train(train_d, &indicators, &format!("SPOTTER (+{},+{},+{}m) [{}]", sym_cfg.bar_mins, 5*sym_cfg.bar_mins, 10*sym_cfg.bar_mins, sym_upper),
                sym_cfg.epochs2, sym_cfg.lr2, &sym_cfg,
                |i| vec![scout_train_preds.get(i).copied().unwrap_or(0.0)]);
            let sp_live = spotter.predict(live_w, live_ind, &[s_val], anchor);
            let sp_v = [sp_live.get(&1).unwrap().0, sp_live.get(&5).unwrap().0, sp_live.get(&10).unwrap().0];

            let spotter_train_preds: Vec<[f64; 3]> = (0..n_cascade).map(|i| {
                let w   = &train_d[i..i + sym_cfg.lookback];
                let ind = &indicators[i + sym_cfg.lookback - 1];
                let sv  = scout_train_preds.get(i).copied().unwrap_or(0.0);
                let sp  = spotter.predict(w, ind, &[sv], train_d[i + sym_cfg.lookback - 1].close);
                [sp.get(&1).unwrap().0, sp.get(&5).unwrap().0, sp.get(&10).unwrap().0]
            }).collect();

            let mut sniper = Net::new((0..10).collect(), 4, &sym_cfg);
            let sniper_history = sniper.train(train_d, &indicators, &format!("SNIPER (+{}..{}m) [{}]", sym_cfg.bar_mins, 10*sym_cfg.bar_mins, sym_upper),
                sym_cfg.epochs3, sym_cfg.lr3, &sym_cfg, |i| {
                    let sv = scout_train_preds.get(i).copied().unwrap_or(0.0);
                    let sp = spotter_train_preds.get(i).copied().unwrap_or([0.0; 3]);
                    vec![sv, sp[0], sp[1], sp[2]]
                });
            let sn_live = sniper.predict(live_w, live_ind, &[s_val, sp_v[0], sp_v[1], sp_v[2]], anchor);

            // Always save to {SYM}.weights in batch mode
            let weights_out = format!("{}.weights", sym_upper);
            save_all_weights(&weights_out, &sym_cfg, &scout, &spotter, &sniper);

            // Quick results preview
            println!("━━━ [{}] Results (last 10 bars) ━━━━━━━━━━━━━━━━━━━━", sym_upper);
            println!("  {:>4} | Sniper ({}-{})  | Actual", "Min", sym_cfg.bar_mins, 10*sym_cfg.bar_mins);
            println!("  ----|------------------|--------");
            for m in 1..=10 {
                let actual_pct = (data[r_end + m - 1].close - anchor) / anchor;
                let pred = sn_live.get(&m).map(|(p, d)| format!("{:>+6.3}% /${:<7.2}", p * 100.0, d))
                    .unwrap_or("      --        ".into());
                println!("  {:>4} | {:<16} | {:>+6.3}%", m * sym_cfg.bar_mins, pred, actual_pct * 100.0);
            }

            // Append to tuning log
            let mut tracker = TuningTracker::new(&format!("AI_Tuning_Log_{}.xlsx", sym_upper));
            let run_label = format!("lr={} l2={} es={} l={} h={} lb={}",
                sym_cfg.lr1, sym_cfg.l2_lambda, sym_cfg.early_stop_patience,
                sym_cfg.layers, sym_cfg.hidden, sym_cfg.lookback);
            tracker.add_run("Scout",   RunRecord { run_label: run_label.clone(), cfg: sym_cfg.clone(), epochs: scout_history });
            tracker.add_run("Spotter", RunRecord { run_label: run_label.clone(), cfg: sym_cfg.clone(), epochs: spotter_history });
            tracker.add_run("Sniper",  RunRecord { run_label: run_label.clone(), cfg: sym_cfg.clone(), epochs: sniper_history });
            match tracker.save() {
                Ok(_)  => println!("📊 Excel charts saved to AI_Tuning_Log_{}.xlsx", sym_upper),
                Err(e) => eprintln!("⚠  Could not save Excel for {}: {}", sym_upper, e),
            }
        }

        println!("\n✅ Batch training complete. Weights saved:");
        for sym in &sym_list { println!("   {}.weights", sym.to_uppercase()); }
        return Ok(());
    }
    // ── End multi-symbol batch mode ───────────────────────────────────────────

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

    let data = {
        let raw = parse_csv(&csv_path);
        if interval_mins > 1 {
            println!("Resampling from 1-min to {}-min bars...", interval_mins);
            let resampled = resample(raw, interval_mins);
            println!("  {} bars after resampling\n", resampled.len());
            resampled
        } else {
            raw
        }
    };
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
    let scout_history = scout.train(train_d, &indicators, &format!("SCOUT (+{}m)", 10 * cfg.bar_mins), cfg.epochs1, cfg.lr1, &cfg, |_| vec![]);
    let s_val = scout.predict(live_w, live_ind, &[], anchor).get(&10).unwrap().0;

    // Pre-compute Scout's prediction for every training sample so Spotter and Sniper
    // receive a realistic, varying cascade signal during training rather than the single
    // live value (which was a constant for all ~195k samples — defeating the cascade).
    let n_cascade = train_d.len().saturating_sub(cfg.lookback + 9);
    let scout_train_preds: Vec<f64> = (0..n_cascade).map(|i| {
        let w   = &train_d[i..i + cfg.lookback];
        let ind = &indicators[i + cfg.lookback - 1];
        scout.predict(w, ind, &[], train_d[i + cfg.lookback - 1].close)
             .get(&10).unwrap().0
    }).collect();

    let mut spotter = Net::new(vec![0, 4, 9], 1, &cfg);
    let spotter_history = spotter.train(train_d, &indicators, &format!("SPOTTER (+{},+{},+{}m)", cfg.bar_mins, 5*cfg.bar_mins, 10*cfg.bar_mins), cfg.epochs2, cfg.lr2, &cfg,
        |i| vec![scout_train_preds.get(i).copied().unwrap_or(0.0)]);
    let sp_live = spotter.predict(live_w, live_ind, &[s_val], anchor);
    let sp_v = [
        sp_live.get(&1).unwrap().0,
        sp_live.get(&5).unwrap().0,
        sp_live.get(&10).unwrap().0,
    ];

    // Pre-compute Spotter's per-sample predictions for Sniper's cascade inputs.
    let spotter_train_preds: Vec<[f64; 3]> = (0..n_cascade).map(|i| {
        let w   = &train_d[i..i + cfg.lookback];
        let ind = &indicators[i + cfg.lookback - 1];
        let sv  = scout_train_preds.get(i).copied().unwrap_or(0.0);
        let sp  = spotter.predict(w, ind, &[sv], train_d[i + cfg.lookback - 1].close);
        [sp.get(&1).unwrap().0, sp.get(&5).unwrap().0, sp.get(&10).unwrap().0]
    }).collect();

    let mut sniper = Net::new((0..10).collect(), 4, &cfg);
    let sniper_history = sniper.train(train_d, &indicators, &format!("SNIPER (+{}..{}m)", cfg.bar_mins, 10*cfg.bar_mins), cfg.epochs3, cfg.lr3, &cfg, |i| {
        let sv = scout_train_preds.get(i).copied().unwrap_or(0.0);
        let sp = spotter_train_preds.get(i).copied().unwrap_or([0.0; 3]);
        vec![sv, sp[0], sp[1], sp[2]]
    });
    let sn_live = sniper.predict(live_w, live_ind, &[s_val, sp_v[0], sp_v[1], sp_v[2]], anchor);

    // Save weights if --save-weights was specified
    if !save_weights_path.is_empty() {
        save_all_weights(&save_weights_path, &cfg, &scout, &spotter, &sniper);
    }

    println!("━━━ [{}] Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", cfg.out_prefix);
    println!("  {:>4} | Scout (+{:<2})     | Spotter ({},{},{}) | Sniper ({}-{})  | Actual",
        "Min", 10*cfg.bar_mins, cfg.bar_mins, 5*cfg.bar_mins, 10*cfg.bar_mins, cfg.bar_mins, 10*cfg.bar_mins);
    println!("  ----|------------------|------------------|------------------|--------");
    for m in 1..=10 {
        let get_fmt = |map: &HashMap<usize, (f64, f64)>, min: usize| {
            map.get(&min)
               .map(|(p, d)| format!("{:>+6.3}% /${:<7.2}", p * 100.0, d))
               .unwrap_or("      --        ".into())
        };
        let actual_pct = (data[r_end + m - 1].close - anchor) / anchor;
        println!("  {:>4} | {:<16} | {:<16} | {:<16} | {:>+6.3}%",
            m * cfg.bar_mins,
            get_fmt(&scout.predict(live_w, live_ind, &[], anchor), m),
            get_fmt(&sp_live, m),
            get_fmt(&sn_live, m),
            actual_pct * 100.0);
    }

    let mut tracker = TuningTracker::new("AI_Tuning_Log.xlsx");
    let run_label = format!("lr={} l2={} es={} l={} h={} lb={}",
        cfg.lr1, cfg.l2_lambda, cfg.early_stop_patience, cfg.layers, cfg.hidden, cfg.lookback);
    tracker.add_run("Scout",   RunRecord { run_label: run_label.clone(), cfg: cfg.clone(), epochs: scout_history });
    tracker.add_run("Spotter", RunRecord { run_label: run_label.clone(), cfg: cfg.clone(), epochs: spotter_history });
    tracker.add_run("Sniper",  RunRecord { run_label: run_label.clone(), cfg: cfg.clone(), epochs: sniper_history });
    match tracker.save() {
        Ok(_)  => println!("\n📊 Excel charts saved to AI_Tuning_Log.xlsx"),
        Err(e) => eprintln!("\n⚠  Could not save Excel: {}", e),
    }

    Ok(())
}
