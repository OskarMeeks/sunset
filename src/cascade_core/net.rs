// cascade_core/net.rs
//
// Shared neural-network types used by both cascade_trainer (training + inference)
// and scan_predict (inference only).
//
// Layer
//   • Holds weights, biases, and the transposed weight matrix used for fast
//     batched forward passes.
//   • `new_trained` (trainer only) additionally carries Adam moment accumulators.
//   • `blank` creates a zero-weight layer suitable for loading saved weights.
//
// Net
//   • Wraps a stack of Layers plus the target-offset mapping.
//   • `blank`      — creates the topology from Config; used by scan_predict.
//   • `predict`    — runs a full cascade forward pass, returns a per-offset map.
//   • `build_input`— builds the input vector without running the net; used for
//                    confidence/jitter scoring in scan_predict.
//   • `forward_raw`— runs a raw forward pass on a pre-built input vector.
//   • `stability`  — jitter test: add noise N times, measure directional agreement.
//   • `serialize_weights` / `deserialize_weights` — binary weight persistence
//     shared between the trainer's save and the predictor's load.

use std::collections::HashMap;
use crate::generator::StockData;
use crate::indicators::{compute_indicators, INDICATOR_NF};
use crate::cascade_core::features::{extract, NF};
use crate::cascade_core::config::Config;

// ─────────────────────────────────────────────
//  Layer
// ─────────────────────────────────────────────

#[derive(Clone)]
pub struct Layer {
    pub in_size:   usize,
    pub out_size:  usize,
    pub w:         Vec<f64>,   // row-major: w[o * in_size + i]
    pub w_t:       Vec<f64>,   // column-major transpose: w_t[i * out_size + o]
    pub b:         Vec<f64>,
    pub is_output: bool,
}

impl Layer {
    /// Create a zero-weight layer for loading pre-trained weights into.
    pub fn blank(in_size: usize, out_size: usize, is_output: bool) -> Self {
        let n_w = in_size * out_size;
        Layer { in_size, out_size, is_output,
            w:   vec![0.0; n_w],
            w_t: vec![0.0; n_w],
            b:   vec![0.0; out_size],
        }
    }

    /// Rebuild the transposed weight matrix from `w`.  Must be called after any
    /// weight mutation (deserialization, Adam update, best-weight restore).
    pub fn sync_transpose(&mut self) {
        for o in 0..self.out_size {
            for i in 0..self.in_size {
                self.w_t[i * self.out_size + o] = self.w[o * self.in_size + i];
            }
        }
    }

    /// Single-sample forward pass.  Applies leaky ReLU on hidden layers and
    /// sigmoid on the output layer.
    pub fn forward(&self, inp: &[f64], out: &mut [f64]) {
        for o in 0..self.out_size {
            let off = o * self.in_size;
            let z = self.b[o]
                + self.w[off..off + self.in_size]
                    .iter()
                    .zip(inp)
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>();
            out[o] = if self.is_output {
                if o % 2 == 0 { 1.0 / (1.0 + (-z).exp()) }  // direction: sigmoid
                else           { z.clamp(-0.1, 0.1) }         // magnitude: linear ±10%
            } else if z > 0.0 {
                z                           // ReLU
            } else {
                0.01 * z                    // leaky ReLU slope
            };
        }
    }
}

// ─────────────────────────────────────────────
//  Net  (inference-only subset)
// ─────────────────────────────────────────────

pub struct Net {
    pub layers:         Vec<Layer>,
    pub target_offsets: Vec<usize>,
    pub lookback:       usize,
}

impl Net {
    /// Build a zero-weight net with the topology described by `cfg`.
    /// `n_extra` is the number of cascade scalar inputs appended after the bar
    /// sequence (0 for scout, 1 for spotter, 4 for sniper).
    pub fn blank(target_offsets: Vec<usize>, n_extra: usize, cfg: &Config) -> Self {
        let input_size = cfg.lookback * (NF + INDICATOR_NF) + n_extra;
        let mut layers = Vec::new();
        let mut prev   = input_size;
        for _ in 0..cfg.layers {
            layers.push(Layer::blank(prev, cfg.hidden, false));
            prev = cfg.hidden;
        }
        layers.push(Layer::blank(prev, target_offsets.len(), true));
        Net { layers, target_offsets, lookback: cfg.lookback }
    }

    // ── Input construction ────────────────────────────────────────────────────

    /// Build the full flattened input vector for `data` (a lookback-length
    /// window) with `cascade` scalars appended.  Recomputes indicators per
    /// growing sub-window exactly as training did.
    pub fn build_input(
        &self,
        data:    &[StockData],
        _ind:    &[f64; INDICATOR_NF],
        cascade: &[f64],
    ) -> Vec<f64> {
        let mut inp = Vec::with_capacity(self.layers[0].in_size);
        let ind0 = compute_indicators(&crate::indicators::bars_from_cascade(&data[..1]));
        inp.extend_from_slice(&extract(&data[0], &data[0]));
        inp.extend_from_slice(&ind0);
        for i in 1..self.lookback.min(data.len()) {
            let ind_i = compute_indicators(&crate::indicators::bars_from_cascade(&data[..=i]));
            inp.extend_from_slice(&extract(&data[i], &data[i - 1]));
            inp.extend_from_slice(&ind_i);
        }
        inp.extend_from_slice(cascade);
        inp
    }

    // ── Forward passes ────────────────────────────────────────────────────────

    /// Run a forward pass on a pre-built input vector and return all output
    /// activations.
    pub fn forward_raw(&self, inp: &[f64]) -> Vec<f64> {
        let mut acts = inp.to_vec();
        let mut buf  = vec![0.0f64;
            self.layers.iter().map(|l| l.out_size).max().unwrap_or(1)];
        for layer in &self.layers {
            buf[..layer.out_size].iter_mut().for_each(|x| *x = 0.0);
            layer.forward(&acts, &mut buf[..layer.out_size]);
            acts.resize(layer.out_size, 0.0);
            acts.copy_from_slice(&buf[..layer.out_size]);
        }
        acts
    }

    /// Full cascade predict.  Returns a map of `(minute_offset → (direction_signal, probability, magnitude))`.
    ///
    /// * `direction_signal` = `probability − 0.5`  (positive = bullish, negative = bearish)
    /// * `probability`      = raw sigmoid output in (0, 1)
    /// * `magnitude`        = predicted % price move (linear output, clamped ±10%)
    pub fn predict(
        &self,
        data:    &[StockData],
        ind:     &[f64; INDICATOR_NF],
        cascade: &[f64],
        anchor:  f64,
    ) -> HashMap<usize, (f64, f64, f64)> {
        let _ = anchor; // retained for API compatibility; no longer used internally
        let inp   = self.build_input(data, ind, cascade);
        let preds = self.forward_raw(&inp);
        // output is interleaved: [dir_0, mag_0, dir_1, mag_1, ...]
        self.target_offsets.iter().enumerate().map(|(i, &off)| {
            let prob = preds[i * 2];       // direction: sigmoid
            let mag  = preds[i * 2 + 1];  // magnitude: linear ±10%
            (off + 1, (prob - 0.5, prob, mag))
        }).collect()
    }

    // ── Confidence: jitter stability ──────────────────────────────────────────

    /// Add small uniform noise to `inp` `n_passes` times and measure what
    /// fraction of noisy forward passes agree with the clean prediction direction
    /// for offset index `target_i` (index into `target_offsets`, not raw neuron index).
    ///
    /// Returns `(stability_fraction, per_pass_raw_direction_outputs)`.
    /// * `stability = 1.0` → decisive (all passes agree)
    /// * `stability = 0.5` → coin-flip (model is on a decision boundary)
    pub fn stability(
        &self,
        inp:       &[f64],
        target_i:  usize,
        n_passes:  usize,
        noise_std: f64,
    ) -> (f64, Vec<f64>) {
        // direction neuron for offset i is at index i*2 (magnitude is at i*2+1)
        let dir_neuron = target_i * 2;
        let clean      = self.forward_raw(inp);
        let direction  = clean[dir_neuron] >= 0.5; // sigmoid ≥ 0.5 = bullish
        let mut agree = 0usize;
        let mut pass_pcts: Vec<f64> = Vec::with_capacity(n_passes);
        let mut rng = 0xc0ffee_u64;
        for _ in 0..n_passes {
            let noisy: Vec<f64> = inp.iter().map(|&x| {
                rng = rng.wrapping_mul(6364136223846793005)
                         .wrapping_add(1442695040888963407);
                let u = (rng >> 1) as f64 / i64::MAX as f64 - 1.0; // [-1, 1]
                x + u * noise_std
            }).collect();
            let out = self.forward_raw(&noisy)[dir_neuron];
            pass_pcts.push(out);
            if (out >= 0.5) == direction { agree += 1; }
        }
        (agree as f64 / n_passes as f64, pass_pcts)
    }

    // ── Weight serialization ──────────────────────────────────────────────────
    //
    //  Binary format (little-endian):
    //    [n_layers: u64]
    //    for each layer:
    //      [in_size: u64] [out_size: u64] [is_output: u64]
    //      [weights: in_size × out_size × f64]
    //      [biases:  out_size × f64]

    /// Serialize all layer weights and biases to a flat binary blob.
    pub fn serialize_weights(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        macro_rules! push {
            ($v:expr, u64) => { buf.extend_from_slice(&($v as u64).to_le_bytes()); };
            ($v:expr, f64) => { buf.extend_from_slice(&($v as f64).to_le_bytes()); };
        }
        push!(self.layers.len(), u64);
        for l in &self.layers {
            push!(l.in_size,   u64);
            push!(l.out_size,  u64);
            push!(l.is_output, u64);
            for &w in &l.w { push!(w, f64); }
            for &b in &l.b { push!(b, f64); }
        }
        buf
    }

    /// Load weights from a blob produced by `serialize_weights`.
    /// Panics on topology mismatch.
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
            assert_eq!(in_sz,  l.in_size,  "Layer in_size mismatch");
            assert_eq!(is_out, l.is_output,"Layer is_output mismatch");
            // out_size may differ when loading old weights (e.g. pre-magnitude output
            // had 10 outputs; new format has 20). Resize the layer to match the file.
            if out_sz != l.out_size {
                l.out_size = out_sz;
                l.w   = vec![0.0; in_sz * out_sz];
                l.w_t = vec![0.0; in_sz * out_sz];
                l.b   = vec![0.0; out_sz];
            }
            for w in &mut l.w { *w = read_f64!(); }
            for b in &mut l.b { *b = read_f64!(); }
            l.sync_transpose();
        }
    }
}

// ─────────────────────────────────────────────
//  Weight file I/O  (magic header + three nets)
// ─────────────────────────────────────────────
//
//  File layout:
//    [magic: 8 bytes "CASC_W01"]
//    [config_json_len: u64]  [config_json: UTF-8]
//    [scout_blob_len:  u64]  [scout  weights]
//    [spotter_blob_len: u64] [spotter weights]
//    [sniper_blob_len:  u64] [sniper  weights]

/// Save a trained trio of nets plus their Config to a `.weights` file.
pub fn save_all_weights(path: &str, cfg: &Config, scout: &Net, spotter: &Net, sniper: &Net) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Cannot create weights file");
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

/// Load a `.weights` file and return `(Config, scout, spotter, sniper)`.
pub fn load_all_weights(path: &str) -> (Config, Net, Net, Net) {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("Cannot read '{}': {}", path, e));
    assert!(bytes.len() >= 8, "File too small");
    assert_eq!(&bytes[..8], b"CASC_W01",
        "Not a valid .weights file — use --save-weights when training");
    let mut pos = 8usize;
    let read_u64 = |p: &mut usize| -> u64 {
        let v = u64::from_le_bytes(bytes[*p..*p+8].try_into().unwrap());
        *p += 8; v
    };
    let cfg_len = read_u64(&mut pos) as usize;
    let cfg: Config = serde_json::from_slice(&bytes[pos..pos+cfg_len])
        .expect("Bad config JSON in weights file");
    pos += cfg_len;
    let mut scout   = Net::blank(vec![9],           0, &cfg);
    let mut spotter = Net::blank(vec![0, 4, 9],     1, &cfg);
    let mut sniper  = Net::blank((0..10).collect(), 0, &cfg);
    for net in [&mut scout, &mut spotter, &mut sniper] {
        let len = read_u64(&mut pos) as usize;
        net.deserialize_weights(&bytes[pos..pos+len]);
        pos += len;
    }
    (cfg, scout, spotter, sniper)
}
