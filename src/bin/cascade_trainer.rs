use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::time::Instant;

use rust_xlsxwriter::{Chart, ChartLegendPosition, ChartLine, ChartType, Color, Format, FormatBorder, Workbook, XlsxError};

use stock_tracker::generator::{maybe_download, parse_csv, StockData};
use stock_tracker::indicators::{self, compute_indicators, INDICATOR_NF};

use stock_tracker::cascade_core::{resample, extract, NF, Config};
use stock_tracker::cascade_core::net::save_all_weights;

// ─────────────────────────────────────────────
//  Per-epoch metrics captured during training
// ─────────────────────────────────────────────

#[derive(Clone)]
pub struct EpochMetrics {
    pub loss:         f64,
    pub accuracy_pct: f64,
    pub epoch_secs:   f64,
}

// ── Serde for EpochMetrics ────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────
//  One complete run record
// ─────────────────────────────────────────────

#[derive(Clone)]
pub struct RunRecord {
    pub run_label: String,
    pub cfg:       Config,
    pub epochs:    Vec<EpochMetrics>,
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
// ─────────────────────────────────────────────

pub struct TuningTracker {
    file_path:   String,
    sniper_runs: Vec<RunRecord>,
}

impl TuningTracker {
    pub fn new(file_path: &str) -> Self {
        let sidecar = format!("{}.runs.json", file_path);
        let sniper_runs =
            if let Ok(json) = fs::read_to_string(&sidecar) {
                serde_json::from_str::<Vec<RunRecord>>(&json).unwrap_or_default()
            } else {
                vec![]
            };
        TuningTracker { file_path: file_path.to_string(), sniper_runs }
    }

    pub fn add_run(&mut self, record: RunRecord) {
        self.sniper_runs.push(record);
    }

    pub fn save(self) -> Result<(), XlsxError> {
        let sidecar = format!("{}.runs.json", self.file_path);
        fs::write(&sidecar, serde_json::to_string(&self.sniper_runs).unwrap_or_default()).ok();

        let mut wb = Workbook::new();
        Self::write_phase_sheet(&mut wb, "Sniper", &self.sniper_runs)?;
        wb.save(&self.file_path)
    }

    fn write_phase_sheet(wb: &mut Workbook, phase: &str, runs: &[RunRecord]) -> Result<(), XlsxError> {
        if runs.is_empty() { return Ok(()); }

        let ws = wb.add_worksheet().set_name(phase)?;

        let hdr_fmt = Format::new()
            .set_bold()
            .set_background_color(Color::RGB(0x2C3E50))
            .set_font_color(Color::White)
            .set_border(FormatBorder::Thin);
        let label_fmt = Format::new().set_bold();

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

        let data_header_row: u32 = 15;
        let data_start_row:  u32 = 16;
        ws.write_with_format(data_header_row, 0, "Epoch", &hdr_fmt)?;

        for (ri, run) in runs.iter().enumerate() {
            let base_col = 1 + ri as u16 * 3;
            let short = Self::short_label(&run.run_label);
            ws.write_with_format(data_header_row, base_col,     format!("{} Loss",   short).as_str(), &hdr_fmt)?;
            ws.write_with_format(data_header_row, base_col + 1, format!("{} Acc%",   short).as_str(), &hdr_fmt)?;
            ws.write_with_format(data_header_row, base_col + 2, format!("{} Sec/Ep", short).as_str(), &hdr_fmt)?;
        }

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

        ws.set_column_width(0, 8)?;
        for ri in 0..runs.len() {
            let b = (1 + ri * 3) as u16;
            ws.set_column_width(b,     13)?;
            ws.set_column_width(b + 1, 10)?;
            ws.set_column_width(b + 2, 10)?;
        }

        let last_data_row    = data_start_row + max_epochs as u32 - 1;
        let chart_anchor_col = (1 + runs.len() * 3 + 1) as u16;

        // ── Chart 1: Loss ──
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

        // ── Chart 2: Accuracy ──
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

        // ── Chart 3: Epoch duration ──
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

        // ── Run legend ──
        ws.write_with_format(14, 0, "Run Legend", &label_fmt)?;
        for (ri, run) in runs.iter().enumerate() {
            let rgb      = PALETTE[ri % PALETTE.len()];
            let cell_fmt = Format::new()
                .set_background_color(Color::RGB(rgb))
                .set_font_color(Color::White)
                .set_bold();
            ws.write_with_format(14, 1 + ri as u16, run.run_label.as_str(), &cell_fmt)?;
        }

        Ok(())
    }

    fn short_label(label: &str) -> String {
        if label.len() > 20 { format!("…{}", &label[label.len()-19..]) } else { label.to_string() }
    }
}

// ─────────────────────────────────────────────
//  Trainer-only Layer
//
//  Extends the shared cascade_core::net::Layer with Adam moment accumulators.
//  The base Layer fields (w, w_t, b, in_size, out_size, is_output) are embedded
//  directly; we alias sync_transpose and forward through it.
// ─────────────────────────────────────────────

#[derive(Clone)]
struct Layer {
    in_size:   usize,
    out_size:  usize,
    w:         Vec<f64>,
    w_t:       Vec<f64>,
    b:         Vec<f64>,
    is_output: bool,
    // Adam moment accumulators — not needed for inference, only for training
    m_w: Vec<f64>, v_w: Vec<f64>,
    m_b: Vec<f64>, v_b: Vec<f64>,
}

impl Layer {
    /// He-initialised layer with zeroed Adam state.
    fn new(in_size: usize, out_size: usize, is_output: bool, rng: &mut u64) -> Self {
        // Output layer: tiny init → predictions start near sigmoid midpoint (0.5).
        // Hidden layers: He init (sqrt(2/fan_in)) — correct for leaky ReLU.
        let limit = if is_output { 0.01 } else { (2.0_f64 / in_size as f64).sqrt() };
        let w: Vec<f64> = (0..in_size * out_size).map(|_| lcg(rng) * 2.0 * limit - limit).collect();
        let n_w = in_size * out_size;
        let mut s = Layer {
            in_size, out_size, is_output,
            w_t: vec![0.0; n_w], w, b: vec![0.0; out_size],
            m_w: vec![0.0; n_w], v_w: vec![0.0; n_w],
            m_b: vec![0.0; out_size], v_b: vec![0.0; out_size],
        };
        s.sync_transpose();
        s
    }

    fn sync_transpose(&mut self) {
        for o in 0..self.out_size {
            for i in 0..self.in_size {
                self.w_t[i * self.out_size + o] = self.w[o * self.in_size + i];
            }
        }
    }

    fn forward(&self, inp: &[f64], out: &mut [f64]) {
        for o in 0..self.out_size {
            let off = o * self.in_size;
            let z = self.b[o] + self.w[off..off + self.in_size].iter().zip(inp).map(|(&w, &x)| w * x).sum::<f64>();
            out[o] = if self.is_output {
                if o % 2 == 0 { 1.0 / (1.0 + (-z).exp()) }  // direction: sigmoid
                else           { z.clamp(-0.1, 0.1) }         // magnitude: linear, clamped to ±10%
            } else if z > 0.0 { z } else { 0.01 * z };
        }
    }

    /// Produce a cascade_core::net::Layer for weight serialisation.
    fn to_core_layer(&self) -> stock_tracker::cascade_core::net::Layer {
        let mut cl = stock_tracker::cascade_core::net::Layer::blank(self.in_size, self.out_size, self.is_output);
        cl.w.copy_from_slice(&self.w);
        cl.b.copy_from_slice(&self.b);
        cl.sync_transpose();
        cl
    }
}


fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f64) / (u32::MAX as f64)
}

// ─────────────────────────────────────────────
//  Workspace  (gradient accumulation buffers)
// ─────────────────────────────────────────────

#[derive(Clone)]
struct Workspace {
    acts:   Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
    errors: Vec<f64>,
    w_grad: Vec<Vec<f64>>,
    b_grad: Vec<Vec<f64>>,
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

// ─────────────────────────────────────────────
//  Net  (training)
// ─────────────────────────────────────────────

struct Net {
    layers:         Vec<Layer>,
    target_offsets: Vec<usize>,
    lookback:       usize,
}

impl Net {
    fn new(target_offsets: Vec<usize>, n_extra: usize, cfg: &Config) -> Self {
        let input_size = cfg.lookback * (NF + INDICATOR_NF) + n_extra;
        let mut rng = 0xdeadbeef_cafebabe_u64;
        let mut layers = Vec::new();
        let mut prev = input_size;
        for _ in 0..cfg.layers {
            layers.push(Layer::new(prev, cfg.hidden, false, &mut rng));
            prev = cfg.hidden;
        }
        layers.push(Layer::new(prev, target_offsets.len() * 2, true, &mut rng));
        Net { layers, target_offsets, lookback: cfg.lookback }
    }

    /// Convert to a cascade_core::net::Net for weight serialisation.
    /// The core Net is blanked with target_offsets so its output layer has
    /// target_offsets.len() nodes; we then replace every layer wholesale with
    /// to_core_layer() which copies the actual weights — including the output
    /// layer which has target_offsets.len()*2 nodes (direction + magnitude).
    fn to_core_net(&self) -> stock_tracker::cascade_core::net::Net {
        let n_extra = self.layers[0].in_size - self.lookback * (NF + INDICATOR_NF);
        let mut cn = stock_tracker::cascade_core::net::Net::blank(
            self.target_offsets.clone(),
            n_extra,
            &Config {
                lookback: self.lookback,
                hidden:   self.layers.first().map(|l| l.out_size).unwrap_or(128),
                layers:   self.layers.len() - 1,
                ..Config::default()
            },
        );
        // Replace all layers (including output) with the actual trained weights.
        // The output layer in cn was blanked with target_offsets.len() outputs but
        // the real output has target_offsets.len()*2; rebuild it correctly.
        cn.layers = self.layers.iter().map(|l| l.to_core_layer()).collect();
        cn
    }

    fn train<F>(
        &mut self,
        data:       &[StockData],
        indicators: &[[f64; INDICATOR_NF]],
        name:       &str,
        epochs:     usize,
        lr_init:    f64,
        cfg:        &Config,
        cascade_fn: F,
    ) -> Vec<EpochMetrics>
    where F: Fn(usize) -> Vec<f64> + Sync {
        let max_off   = *self.target_offsets.iter().max().unwrap_or(&0);
        let n_samples = data.len().saturating_sub(self.lookback + max_off);
        if n_samples == 0 { return vec![]; }

        let input_dim  = self.layers[0].in_size;
        let target_dim  = self.target_offsets.len();
        let output_dim  = target_dim * 2;  // interleaved: [dir_0, mag_0, dir_1, mag_1, ...]
        let mut flat_inputs  = vec![0.0; n_samples * input_dim];
        let mut flat_targets = vec![0.0; n_samples * output_dim];

        for i in 0..n_samples {
            let anchor = data[i + self.lookback - 1].close.max(1e-8);
            let mut inp = Vec::with_capacity(input_dim);
            inp.extend_from_slice(&extract(&data[i], &data[i]));
            inp.extend_from_slice(&indicators[i]);
            for j in i + 1..i + self.lookback {
                inp.extend_from_slice(&extract(&data[j], &data[j - 1]));
                inp.extend_from_slice(&indicators[j]);
            }
            inp.extend(cascade_fn(i));
            flat_inputs[i*input_dim..(i+1)*input_dim].copy_from_slice(&inp);
            for (ti, &off) in self.target_offsets.iter().enumerate() {
                let raw = (data[i+self.lookback+off].close - anchor) / anchor;
                let mag = raw.clamp(-0.05, 0.05);  // cap at ±5% so outliers don't dominate
                flat_targets[i*output_dim + ti*2    ] = if raw > 0.0 { 1.0 } else { 0.0 };  // direction
                flat_targets[i*output_dim + ti*2 + 1] = mag;                                  // magnitude
            }
        }

        // ── Class balance weights ─────────────────────────────────────────────
        // class weights only apply to direction outputs (even indices)
        let class_weight_pairs: Vec<(f64, f64)> = (0..target_dim).map(|ti| {
            let pos   = (0..n_samples).filter(|&i| flat_targets[i*output_dim + ti*2] > 0.5).count();
            let neg   = n_samples - pos;
            let pos_w = (neg as f64 / pos.max(1) as f64).clamp(0.5, 5.0);
            let neg_w = (pos as f64 / neg.max(1) as f64).clamp(0.5, 5.0);
            (pos_w, neg_w)
        }).collect();
        {
            let (p0, n0) = { let p = (0..n_samples).filter(|&i| flat_targets[i*output_dim] > 0.5).count(); (p, n_samples - p) };
            let balance_str = if target_dim == 1 {
                format!("{}↑ {}↓  ({:.1}% bull)", p0, n0, p0 as f64 / n_samples as f64 * 100.0)
            } else {
                let (pl, nl) = { let p = (0..n_samples).filter(|&i| flat_targets[i*output_dim + (target_dim-1)*2] > 0.5).count(); (p, n_samples - p) };
                format!("+{}m {}↑{}↓({:.1}%bull) … +{}m {}↑{}↓({:.1}%bull)",
                    (self.target_offsets[0] + 1) * cfg.bar_mins, p0, n0, p0 as f64 / n_samples as f64 * 100.0,
                    (self.target_offsets[target_dim-1] + 1) * cfg.bar_mins, pl, nl, pl as f64 / n_samples as f64 * 100.0)
            };
            println!("  Balance: {}  |  {} samples  |  {} outputs ({}dir + {}mag)  |  class weight clamp ±5×", balance_str, n_samples, output_dim, target_dim, target_dim);
        }

        let ws_proto      = Workspace::new(&self.layers, input_dim);
        let mut lr;
        let mut history: Vec<EpochMetrics> = Vec::with_capacity(epochs);
        let warmup_epochs = (epochs / 10).max(5).min(20);

        println!("━━━ {} ━━━", name);
        println!("  {:>6} │ {:>10} │ {:>7} │ {:>8} │ {:>5} │ {:>9} │ {:>9}",
            "Epoch", "Loss", "Acc%", "LR", "s/ep", "h1 grad", "out grad");
        println!("  {}┼{}┼{}┼{}┼{}┼{}┼{}",
            "─".repeat(7), "─".repeat(12), "─".repeat(9), "─".repeat(10),
            "─".repeat(7), "─".repeat(11), "─".repeat(10));

        let mut best_acc: f64     = 0.0;
        let mut epochs_no_improve = 0usize;
        let mut best_weights: Vec<(Vec<f64>, Vec<f64>)> =
            self.layers.iter().map(|l| (l.w.clone(), l.b.clone())).collect();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Global Adam step counter — bias correction must use the total number of
        // updates applied, not the epoch number.  One update fires per mini-batch,
        // so this increments (n_samples / batch_size) times per epoch.
        let mut adam_t = 0u32;
        let adam_beta1 = 0.9_f64;
        let adam_beta2 = 0.999_f64;
        let adam_eps   = 1e-8_f64;

        for epoch in 0..epochs {
            lr = if epoch < warmup_epochs {
                lr_init * (epoch + 1) as f64 / warmup_epochs as f64
            } else {
                lr_init * cfg.lr_decay.powi((epoch - warmup_epochs) as i32)
            };

            // Fisher-Yates shuffle
            {
                let mut rng = epoch as u64 ^ 0xdeadbeef_cafe_u64;
                for i in (1..n_samples).rev() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let j = (rng >> 33) as usize % (i + 1);
                    indices.swap(i, j);
                }
            }

            let epoch_start   = Instant::now();
            let mut epoch_loss    = 0.0_f64;
            let mut epoch_correct = 0usize;
            let l2_factor = 1.0 - lr * cfg.l2_lambda;
            // Accumulate squared gradient norms across all batches for accurate
            // epoch-level logging (replaces the old last-batch-only approach).
            let mut epoch_grad_sq: Vec<Vec<f64>> = self.layers.iter()
                .map(|l| vec![0.0; l.out_size * l.in_size]).collect();
            let mut epoch_batch_count = 0usize;

            // ── Mini-batch loop ───────────────────────────────────────────────
            // Each iteration accumulates gradients for one batch in parallel,
            // then immediately applies an Adam update before moving to the next
            // batch.  This replaces the previous full-dataset fold that only
            // updated weights once per epoch.
            for b_start in (0..n_samples).step_by(cfg.batch_size) {
                let b_end   = (b_start + cfg.batch_size).min(n_samples);
                let b_count = (b_end - b_start) as f64;

                let (ws_result, batch_mse, batch_correct) =
                    (b_start..b_end).into_par_iter()
                    .fold(|| (ws_proto.clone(), 0.0_f64, 0usize),
                          |(mut ws, mut mse, mut correct), pos| {
                        let i = indices[pos];
                        ws.acts[0].copy_from_slice(
                            &flat_inputs[i*input_dim..(i+1)*input_dim]);
                        for (l, layer) in self.layers.iter().enumerate() {
                            let (left, right) = ws.acts.split_at_mut(l + 1);
                            layer.forward(&left[l], &mut right[0]);
                        }
                        let targets  = &flat_targets[i*output_dim..(i+1)*output_dim];
                        let last_idx = self.layers.len();
                        for k in 0..self.layers[last_idx - 1].out_size {
                            let p = ws.acts[last_idx][k];
                            let t = targets[k];
                            if k % 2 == 0 {
                                // direction output: BCE loss + class weighting
                                let ti = k / 2;
                                let eps_bce = 1e-7_f64;
                                let bce = -(t * (p + eps_bce).ln()
                                          + (1.0 - t) * (1.0 - p + eps_bce).ln());
                                let (pos_w, neg_w) = class_weight_pairs[ti];
                                let class_w = if t > 0.5 { pos_w } else { neg_w };
                                ws.errors[k] = (p - t) * class_w;
                                mse += bce;
                                if (p > 0.5 && t > 0.5) || (p < 0.5 && t < 0.5) { correct += 1; }
                            } else {
                                // magnitude output: MSE loss scaled by dir_weight
                                // derivative of (1/2 * mse * (p-t)^2) = mse * (p-t)
                                // linear activation so no extra derivative term
                                ws.errors[k] = cfg.dir_weight * (p - t);
                                mse += cfg.dir_weight * 0.5 * (p - t).powi(2);
                            }
                        }
                        let last_layer = last_idx - 1;
                        for o in 0..self.layers[last_layer].out_size {
                            ws.deltas[last_layer][o] = ws.errors[o];
                        }
                        for l in (0..last_layer).rev() {
                            let next     = l + 1;
                            let next_in  = self.layers[next].in_size;
                            let next_out = self.layers[next].out_size;
                            for o in 0..self.layers[l].out_size {
                                let sum: f64 = (0..next_out)
                                    .map(|j| self.layers[next].w[j * next_in + o]
                                             * ws.deltas[next][j])
                                    .sum();
                                ws.deltas[l][o] = sum
                                    * if ws.acts[l+1][o] > 0.0 { 1.0 } else { 0.01 };
                            }
                        }
                        for l in 0..=last_layer {
                            let in_s = self.layers[l].in_size;
                            for o in 0..self.layers[l].out_size {
                                let d = ws.deltas[l][o]; let off = o * in_s;
                                for idx in 0..in_s {
                                    ws.w_grad[l][off + idx] += d * ws.acts[l][idx];
                                }
                                ws.b_grad[l][o] += d;
                            }
                        }
                        (ws, mse, correct)
                    })
                    .reduce(|| (ws_proto.clone(), 0.0, 0), |mut a, b| {
                        for l in 0..a.0.w_grad.len() {
                            for (wa, &wb) in a.0.w_grad[l].iter_mut().zip(&b.0.w_grad[l]) {
                                *wa += wb;
                            }
                            for (ba, &bb) in a.0.b_grad[l].iter_mut().zip(&b.0.b_grad[l]) {
                                *ba += bb;
                            }
                        }
                        (a.0, a.1 + b.1, a.2 + b.2)
                    });

                epoch_loss    += batch_mse;
                epoch_correct += batch_correct;

                // ── Adam + L2 update for this batch ───────────────────────────
                adam_t += 1;
                let bc1 = 1.0 - adam_beta1.powi(adam_t as i32);
                let bc2 = 1.0 - adam_beta2.powi(adam_t as i32);
                for l in 0..self.layers.len() {
                    let in_s = self.layers[l].in_size;
                    for o in 0..self.layers[l].out_size {
                        let off = o * in_s;
                        for idx in 0..in_s {
                            let g = ws_result.w_grad[l][off+idx] / b_count;
                            let m = adam_beta1 * self.layers[l].m_w[off+idx]
                                  + (1.0 - adam_beta1) * g;
                            let v = adam_beta2 * self.layers[l].v_w[off+idx]
                                  + (1.0 - adam_beta2) * g * g;
                            self.layers[l].m_w[off+idx] = m;
                            self.layers[l].v_w[off+idx] = v;
                            self.layers[l].w[off+idx] =
                                self.layers[l].w[off+idx] * l2_factor
                                - lr * (m / bc1) / ((v / bc2).sqrt() + adam_eps);
                        }
                        let gb = ws_result.b_grad[l][o] / b_count;
                        let mb = adam_beta1 * self.layers[l].m_b[o]
                               + (1.0 - adam_beta1) * gb;
                        let vb = adam_beta2 * self.layers[l].v_b[o]
                               + (1.0 - adam_beta2) * gb * gb;
                        self.layers[l].m_b[o] = mb;
                        self.layers[l].v_b[o] = vb;
                        self.layers[l].b[o] -=
                            lr * (mb / bc1) / ((vb / bc2).sqrt() + adam_eps);
                    }
                    self.layers[l].sync_transpose();
                }
                // Accumulate normalised per-batch gradient norms for epoch logging.
                for l in 0..self.layers.len() {
                    for (sq, &g) in epoch_grad_sq[l].iter_mut()
                                      .zip(ws_result.w_grad[l].iter()) {
                        *sq += (g / b_count).powi(2);
                    }
                }
                epoch_batch_count += 1;
            }
            // ── End mini-batch loop ───────────────────────────────────────────

            let epoch_loss_avg = epoch_loss / (n_samples * target_dim) as f64;
            let accuracy       = (epoch_correct as f64
                                  / (n_samples * target_dim) as f64) * 100.0;
            let epoch_secs     = epoch_start.elapsed().as_secs_f64();

            // ── Early stopping ────────────────────────────────────────────────
            if epoch >= warmup_epochs {
                if accuracy > best_acc {
                    best_acc          = accuracy;
                    epochs_no_improve = 0;
                    best_weights = self.layers.iter()
                        .map(|l| (l.w.clone(), l.b.clone())).collect();
                } else {
                    epochs_no_improve += 1;
                }
                if cfg.early_stop_patience > 0
                    && epochs_no_improve >= cfg.early_stop_patience
                {
                    let n = epoch_batch_count.max(1) as f64;
                    let h1g = epoch_grad_sq[0].iter()
                        .map(|&sq| sq / n).sum::<f64>().sqrt();
                    let og = epoch_grad_sq[self.layers.len()-1].iter()
                        .map(|&sq| sq / n).sum::<f64>().sqrt();
                    println!(
                        "  {:>3}/{:<3} │ {:>10.8} │ {:>6.2}% │ {:>8.2e} │ {:>5.2} \
                         │ {:>9.2e} │ {:>9.2e}  ⏹",
                        epoch + 1, epochs, epoch_loss_avg, accuracy, lr,
                        epoch_secs, h1g, og);
                    println!("  └─ early stop: no acc gain for {} epochs",
                             cfg.early_stop_patience);
                    history.push(EpochMetrics {
                        loss: epoch_loss_avg, accuracy_pct: accuracy, epoch_secs,
                    });
                    break;
                }
            }

            if (epoch + 1) % (epochs / 10).max(1) == 0 {
                let tag = if epoch < warmup_epochs { "W" } else { " " };
                let n = epoch_batch_count.max(1) as f64;
                let h1g = epoch_grad_sq[0].iter()
                    .map(|&sq| sq / n).sum::<f64>().sqrt();
                let og = epoch_grad_sq[self.layers.len()-1].iter()
                    .map(|&sq| sq / n).sum::<f64>().sqrt();
                println!(
                    "  {:>3}/{:<3} │ {:>10.8} │ {:>6.2}% │ {:>8.2e} │ {:>5.2} \
                     │ {:>9.2e} │ {:>9.2e}  {}",
                    epoch + 1, epochs, epoch_loss_avg, accuracy, lr,
                    epoch_secs, h1g, og, tag);
            }
            history.push(EpochMetrics {
                loss: epoch_loss_avg, accuracy_pct: accuracy, epoch_secs,
            });
        }

        // ── Restore best weights ──────────────────────────────────────────────
        for (l, (w, b)) in self.layers.iter_mut().zip(best_weights) {
            l.w = w; l.b = b; l.sync_transpose();
        }

        // ── Post-train sanity check ───────────────────────────────────────────
        let step = (n_samples / 5).max(1);
        let sample_preds: Vec<f64> = (0..5).map(|ci| {
            let i    = (ci * step).min(n_samples - 1);
            let mut acts = flat_inputs[i*input_dim..(i+1)*input_dim].to_vec();
            for layer in &self.layers {
                let mut out = vec![0.0f64; layer.out_size];
                layer.forward(&acts, &mut out);
                acts = out;
            }
            acts[0]
        }).collect();
        let p_mean = sample_preds.iter().sum::<f64>() / 5.0;
        let p_std  = (sample_preds.iter().map(|&p| (p - p_mean).powi(2)).sum::<f64>() / 5.0).sqrt();
        let p_min  = sample_preds.iter().cloned().fold(f64::INFINITY, f64::min);
        let p_max  = sample_preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let spread_flag = if p_std < 0.005 { "  ⚠ outputs near-identical (dead layers?)" } else { "" };
        println!("  {}┼{}┼{}┼{}┼{}┼{}┼{}",
            "─".repeat(7), "─".repeat(12), "─".repeat(9), "─".repeat(10),
            "─".repeat(7), "─".repeat(11), "─".repeat(10));
        println!("  ✓ best acc {:.2}%  |  pred spread: min={:.3} max={:.3} std={:.4}{}",
            best_acc, p_min, p_max, p_std, spread_flag);

        // Print output layer biases — if all direction biases are negative the model
        // has collapsed to a bearish prior; retrain with more epochs or higher LR.
        let out_layer = self.layers.last().unwrap();
        let dir_biases: Vec<String> = (0..out_layer.out_size)
            .filter(|o| o % 2 == 0)
            .map(|o| format!("{:>+.3}", out_layer.b[o]))
            .collect();
        let all_neg = dir_biases.iter().all(|s| s.starts_with('-'));
        println!("  Direction biases (output layer): [{}]{}",
            dir_biases.join(", "),
            if all_neg { "  ⚠ ALL NEGATIVE — model collapsed to bearish prior, try higher --dir-weight or more epochs" } else { "" });

        history
    }

    fn predict(&self, data: &[StockData], _ind: &[f64; INDICATOR_NF], cascade: &[f64], _anchor: f64) -> HashMap<usize, (f64, f64, f64)> {
        let mut inp = Vec::with_capacity(self.layers[0].in_size);
        let ind0    = compute_indicators(&indicators::bars_from_cascade(&data[..1]));
        inp.extend_from_slice(&extract(&data[0], &data[0]));
        inp.extend_from_slice(&ind0);
        for i in 1..self.lookback.min(data.len()) {
            let ind_i = compute_indicators(&indicators::bars_from_cascade(&data[..=i]));
            inp.extend_from_slice(&extract(&data[i], &data[i - 1]));
            inp.extend_from_slice(&ind_i);
        }
        inp.extend_from_slice(cascade);
        let mut ws = Workspace::new(&self.layers, self.layers[0].in_size);
        ws.acts[0].copy_from_slice(&inp);
        for (l, layer) in self.layers.iter().enumerate() {
            let (left, right) = ws.acts.split_at_mut(l + 1);
            layer.forward(&left[l], &mut right[0]);
        }
        let preds = &ws.acts[self.layers.len()];
        // output is interleaved: [dir_0, mag_0, dir_1, mag_1, ...]
        self.target_offsets.iter().enumerate().map(|(i, &off)| {
            let prob = preds[i * 2];       // direction: sigmoid output
            let mag  = preds[i * 2 + 1];  // magnitude: linear output (predicted % move)
            (off + 1, (prob - 0.5, prob, mag))
        }).collect()
    }

    /// Serialise via cascade_core so the .weights format stays identical.
    #[allow(dead_code)]
    fn serialize_weights(&self) -> Vec<u8> {
        self.to_core_net().serialize_weights()
    }
}

// ─────────────────────────────────────────────
//  Weight persistence — wraps cascade_core
// ─────────────────────────────────────────────

fn save_weights(path: &str, cfg: &Config, sniper: &Net) {
    let cn = sniper.to_core_net();
    // Pass dummy empty nets for scout/spotter slots; sniper_only=true in cfg skips them on load
    let blank_scout   = stock_tracker::cascade_core::net::Net::blank(vec![9],       0, cfg);
    let blank_spotter = stock_tracker::cascade_core::net::Net::blank(vec![0, 4, 9], 1, cfg);
    save_all_weights(path, cfg, &blank_scout, &blank_spotter, &cn);
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

fn main() -> io::Result<()> {
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
        println!("    --epochs3 N       Training epochs                 [300]");
        println!("    --lr3 F           Learning rate                   [0.001]");
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
        return Ok(());
    }

    let mut cfg = if let Ok(c) = fs::read_to_string("trainer_config.json") {
        serde_json::from_str(&c).unwrap_or_default()
    } else {
        Config::default()
    };

    let mut start_date        = String::new();
    let mut end_date          = String::new();
    let mut api_key           = String::new();
    let mut save_weights_path = String::new();
    let mut interval_mins: usize = 5;
    let mut symbols_str:   String = String::new();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--lookback"     => { cfg.lookback            = args[i+1].parse().unwrap(); i += 2; }
            "--hidden"       => { cfg.hidden              = args[i+1].parse().unwrap(); i += 2; }
            "--layers"       => { cfg.layers              = args[i+1].parse().unwrap(); i += 2; }
            "--epochs1"      => { cfg.epochs1             = args[i+1].parse().unwrap(); i += 2; }
            "--epochs2"      => { cfg.epochs2             = args[i+1].parse().unwrap(); i += 2; }
            "--epochs3"      => { cfg.epochs3             = args[i+1].parse().unwrap(); i += 2; }
            "--lr1"          => { cfg.lr1                 = args[i+1].parse().unwrap(); i += 2; }
            "--lr2"          => { cfg.lr2                 = args[i+1].parse().unwrap(); i += 2; }
            "--lr3"          => { cfg.lr3                 = args[i+1].parse().unwrap(); i += 2; }
            "--lr-decay"     => { cfg.lr_decay            = args[i+1].parse().unwrap(); i += 2; }
            "--batch-size"   => { cfg.batch_size          = args[i+1].parse().unwrap(); i += 2; }
            "--dir-weight"   => { cfg.dir_weight          = args[i+1].parse().unwrap(); i += 2; }
            "--l2-lambda"    => { cfg.l2_lambda           = args[i+1].parse().unwrap(); i += 2; }
            "--early-stop"   => { cfg.early_stop_patience = args[i+1].parse().unwrap(); i += 2; }
            "--out-prefix"   => { cfg.out_prefix          = args[i+1].clone();          i += 2; }
            "--start-date"   => { start_date              = args[i+1].clone();          i += 2; }
            "--end-date"     => { end_date                = args[i+1].clone();          i += 2; }
            "--api-key"      => { api_key                 = args[i+1].clone();          i += 2; }
            "--save-weights" => { save_weights_path       = args[i+1].clone();          i += 2; }
            "--interval"     => { interval_mins = args[i+1].parse().unwrap_or(5); cfg.bar_mins = interval_mins; i += 2; }
            "--symbols"      => { symbols_str             = args[i+1].clone();          i += 2; }
            _                => { i += 1; }
        }
    }
    fs::write("trainer_config.json", serde_json::to_string_pretty(&cfg).unwrap()).ok();

    cfg.bar_mins = interval_mins;

    // ── Multi-symbol batch mode ───────────────────────────────────────────────
    if !symbols_str.is_empty() {
        let sym_list: Vec<&str> = symbols_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        println!("━━━ Batch training {} symbols: {} ━━━", sym_list.len(), sym_list.join(", "));

        for sym in &sym_list {
            let sym_upper = sym.to_uppercase();
            println!("\n{}", "═".repeat(60));
            println!("  Symbol: {}  ({}/{})", sym_upper, sym_list.iter().position(|s| s == sym).unwrap_or(0) + 1, sym_list.len());
            println!("{}", "═".repeat(60));

            let csv_path = if !start_date.is_empty() && !end_date.is_empty() && !api_key.is_empty() {
                maybe_download(&sym_upper, &start_date, &end_date, &api_key)
            } else {
                let auto = format!("{}_data.csv", sym_upper);
                if std::path::Path::new(&auto).exists() { auto } else {
                    eprintln!("  ⚠  Skipping {} — no data file and no download flags provided", sym_upper);
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

            let mut sym_cfg = cfg.clone();
            sym_cfg.out_prefix = sym_upper.clone();

            println!("  Precomputing technical indicators...");
            let t_ind = Instant::now();
            let sym_indicators: Vec<[f64; INDICATOR_NF]> = (0..data.len())
                .map(|i| { let start = i.saturating_sub(sym_cfg.lookback - 1); compute_indicators(&indicators::bars_from_cascade(&data[start..=i])) })
                .collect();
            println!("  Done in {:.2?} ({} snapshots)", t_ind.elapsed(), sym_indicators.len());

            let r_end    = data.len() - 10;
            let train_d  = &data[..r_end];
            let live_w   = &data[r_end - sym_cfg.lookback..r_end];
            let anchor   = live_w.last().unwrap().close;
            let live_ind = &sym_indicators[r_end - 1];

            let mut sniper = Net::new((0..10).collect(), 0, &sym_cfg);
            sym_cfg.bar_mins = interval_mins;
            let sniper_history = sniper.train(train_d, &sym_indicators, &format!("SNIPER (+{}..{}m) [{}]", sym_cfg.bar_mins, 10*sym_cfg.bar_mins, sym_upper),
                sym_cfg.epochs3, sym_cfg.lr3, &sym_cfg, |_| vec![]);
            let sn_live = sniper.predict(live_w, live_ind, &[], anchor);

            let weights_out = format!("{}.weights", sym_upper);
            sym_cfg.sniper_only = true;
            save_weights(&weights_out, &sym_cfg, &sniper);

            println!("━━━ [{}] Results (last 10 bars) ━━━━━━━━━━━━━━━━━━━━", sym_upper);
            println!("  {:>4} | {:<32} | Actual", "Min", format!("Sniper ({}-{})", sym_cfg.bar_mins, 10*sym_cfg.bar_mins));
            println!("  ----|----------------------------------|--------");
            for m in 1..=10 {
                let actual_pct = (data[r_end + m - 1].close - anchor) / anchor;
                let pred = sn_live.get(&m).map(|(p, d, mag)| format!("p={:.3} ({:>+.3}) mag={:>+.4}", d, p, mag)).unwrap_or("        --          ".into());
                println!("  {:>4} | {:<32} | {:>+6.3}%", m * sym_cfg.bar_mins, pred, actual_pct * 100.0);
            }

            let mut tracker = TuningTracker::new(&format!("AI_Tuning_Log_{}.xlsx", sym_upper));
            let run_label = format!("lr={} l2={} es={} l={} h={} lb={}",
                sym_cfg.lr1, sym_cfg.l2_lambda, sym_cfg.early_stop_patience,
                sym_cfg.layers, sym_cfg.hidden, sym_cfg.lookback);
            tracker.add_run(RunRecord { run_label: run_label.clone(), cfg: sym_cfg.clone(), epochs: sniper_history });
            match tracker.save() {
                Ok(_)  => println!("📊 Excel charts saved to AI_Tuning_Log_{}.xlsx", sym_upper),
                Err(e) => eprintln!("⚠  Could not save Excel for {}: {}", sym_upper, e),
            }
        }

        println!("\n✅ Batch training complete. Weights saved:");
        for sym in &sym_list { println!("   {}.weights", sym.to_uppercase()); }
        return Ok(());
    }
    // ── End batch mode ────────────────────────────────────────────────────────

    let csv_path = if !start_date.is_empty() && !end_date.is_empty() && !api_key.is_empty() {
        let symbol = args[1].to_uppercase();
        maybe_download(&symbol, &start_date, &end_date, &api_key)
    } else {
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
        } else { raw }
    };
    if data.len() < cfg.lookback + 10 {
        println!("Not enough data ({} rows, need >= {}).", data.len(), cfg.lookback + 10);
        return Ok(());
    }
    println!("Loaded {} rows.\n", data.len());

    println!("Precomputing technical indicators...");
    let t_ind = Instant::now();
    let all_indicators: Vec<[f64; INDICATOR_NF]> = (0..data.len())
        .map(|i| { let start = i.saturating_sub(cfg.lookback - 1); compute_indicators(&indicators::bars_from_cascade(&data[start..=i])) })
        .collect();
    println!("  Done in {:.2?} ({} snapshots)\n", t_ind.elapsed(), all_indicators.len());

    let r_end    = data.len() - 10;
    let train_d  = &data[..r_end];
    let live_w   = &data[r_end - cfg.lookback..r_end];
    let anchor   = live_w.last().unwrap().close;
    let live_ind = &all_indicators[r_end - 1];

    let mut sniper = Net::new((0..10).collect(), 0, &cfg);
    let sniper_history = sniper.train(train_d, &all_indicators, &format!("SNIPER (+{}..{}m)", cfg.bar_mins, 10*cfg.bar_mins), cfg.epochs3, cfg.lr3, &cfg, |_| vec![]);
    let sn_live = sniper.predict(live_w, live_ind, &[], anchor);

    if !save_weights_path.is_empty() {
        cfg.sniper_only = true;
        save_weights(&save_weights_path, &cfg, &sniper);
    }

    println!("━━━ [{}] Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", cfg.out_prefix);
    println!("  {:>4} | {:<32} | Actual", "Min", format!("Sniper ({}-{})", cfg.bar_mins, 10*cfg.bar_mins));
    println!("  ----|----------------------------------|--------");
    for m in 1..=10 {
        let actual_pct = (data[r_end + m - 1].close - anchor) / anchor;
        let pred = sn_live.get(&m).map(|&(p, d, mag)| format!("p={:.3} ({:>+.3}) mag={:>+.4}", d, p, mag)).unwrap_or("        --          ".into());
        println!("  {:>4} | {:<32} | {:>+6.3}%", m * cfg.bar_mins, pred, actual_pct * 100.0);
    }

    let mut tracker = TuningTracker::new("AI_Tuning_Log.xlsx");
    let run_label = format!("lr={} l2={} es={} l={} h={} lb={}",
        cfg.lr1, cfg.l2_lambda, cfg.early_stop_patience, cfg.layers, cfg.hidden, cfg.lookback);
    tracker.add_run(RunRecord { run_label: run_label.clone(), cfg: cfg.clone(), epochs: sniper_history });
    match tracker.save() {
        Ok(_)  => println!("\n📊 Excel charts saved to AI_Tuning_Log.xlsx"),
        Err(e) => eprintln!("\n⚠  Could not save Excel: {}", e),
    }

    Ok(())
}
