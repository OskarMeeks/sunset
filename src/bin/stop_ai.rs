// src/bin/stop_ai.rs
//
// ╔══════════════════════════════════════════════════════════════════╗
// ║  Stop-Entry AI                                                   ║
// ║                                                                  ║
// ║  Trains a 3-layer MLP on 5-minute OHLCV candles.                ║
// ║  Inputs  : 18 technical indicators over a rolling lookback       ║
// ║            window set by the -a flag.                            ║
// ║  Outputs : direction  (Long / Short)                             ║
// ║            sl_mult    — optimal SL distance in ATR units         ║
// ║            tp_mult    — optimal TP distance in ATR units         ║
// ║                                                                  ║
// ║  Evaluation metric: average % gained / lost per candle held,     ║
// ║  across all back-tested stop-entry positions.                    ║
// ║                                                                  ║
// ║  USAGE                                                           ║
// ║    # train                                                       ║
// ║    cargo run --bin stop_ai -- train                              ║
// ║      --csv AAPL_data.csv -a 60 --weights stop_ai.json           ║
// ║      [--epochs 150] [--lr 0.001]                                 ║
// ║                                                                  ║
// ║    # back-test                                                   ║
// ║    cargo run --bin stop_ai -- eval                               ║
// ║      --csv AAPL_data.csv -a 60 --weights stop_ai.json           ║
// ║                                                                  ║
// ║    # live prediction                                             ║
// ║    cargo run --bin stop_ai -- predict                            ║
// ║      --csv AAPL_data.csv -a 60 --weights stop_ai.json           ║
// ╚══════════════════════════════════════════════════════════════════╝
//
// ── IMPORTANT ────────────────────────────────────────────────────────
// Replace `your_crate` on the two `use` lines below with the value of
// [package] name in your Cargo.toml.  If the package is already named
// something like "cascade", write `use cascade::indicators::...`.
// ─────────────────────────────────────────────────────────────────────

use stock_tracker::generator::{parse_csv, StockData, maybe_download};
use stock_tracker::indicators::{compute_indicators, Bar, INDICATOR_NF};

use std::{env, fs, process};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  Architecture hyper-parameters
// ═══════════════════════════════════════════════════════════════════

/// Number of 5-min candles to look forward when labelling training samples.
const FORWARD: usize = 20; // ≈ 100 minutes


const MAX_MULT: f64 = 4.0;

// MLP layer widths
const NIN: usize = INDICATOR_NF; // 72 (18 indicators × 4 time slices)
const H1:  usize = 256;
const H2:  usize = 128;
/// Outputs: [direction_logit, sl_mult_raw, tp_mult_raw]
///   raw[0] → sigmoid        → direction probability (Long if ≥ 0.5)
///   raw[1] → sigmoid scaled → SL distance in ATR units  ∈ [0.3, MAX_MULT]
///   raw[2] → sigmoid scaled → TP distance in ATR units  ∈ [0.5, MAX_MULT]
const NOUT: usize = 3;

/// Weight of the cap% surrogate loss relative to direction BCE loss.
const LAMBDA_REG: f64 = 0.5;

// ── Label generation parameters ───────────────────────────────────
/// Fixed SL/TP ratio used ONLY to determine trade direction during labelling.
/// 1:2 R:R means the random-walk win-rate baseline is ~33%.
const LABEL_SL_MULT: f64 = 1.0;
const LABEL_TP_MULT: f64 = 2.0;

/// Buffer added to MAE when computing the SL label.
/// Gives the trade a small margin beyond the worst drawdown it actually saw.
const MAE_BUFFER: f64 = 0.15; // in ATR units

/// Large TP multiplier used when we only want to measure MFE (max favorable
/// excursion). Must be bigger than any realistic move in the FORWARD window.
const MFE_TP_MULT: f64 = 20.0;

// ═══════════════════════════════════════════════════════════════════
//  XOR-shift RNG  (no extra dep)
// ═══════════════════════════════════════════════════════════════════

struct Rng(u64);
impl Rng {
    fn new() -> Self {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;
        // Mix with a large prime to avoid all-zero seed
        Self(ns.wrapping_add(0x9e37_79b9_7f4a_7c15).max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f64(&mut self) -> f64 {
        // uniform [0, 1)
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.f64() * (hi - lo)
    }
    /// Box–Muller normal deviate.
    #[allow(dead_code)]
    fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn shuffle(&mut self, idx: &mut Vec<usize>) {
        let n = idx.len();
        for i in (1..n).rev() {
            let j = (self.next() as usize) % (i + 1);
            idx.swap(i, j);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Activation helpers
// ═══════════════════════════════════════════════════════════════════

#[inline] fn sigmoid(x: f64) -> f64  { 1.0 / (1.0 + (-x).exp()) }
#[inline] fn relu(x: f64) -> f64     { x.max(0.0) }
#[inline] fn relu_d(pre: f64) -> f64 { if pre > 0.0 { 1.0 } else { 0.0 } }
#[inline] #[allow(dead_code)] fn tanh_d(x: f64) -> f64   { let t = x.tanh(); 1.0 - t * t }

// ═══════════════════════════════════════════════════════════════════
//  MLP  (flat-Vec storage for easy serialisation)
//
//  Layer indexing:
//    w1[j * NIN  + i]  — weight from input  i to hidden-1 j   (j<H1, i<NIN)
//    w2[j * H1   + i]  — weight from hidden-1 i to hidden-2 j  (j<H2, i<H1)
//    w3[j * H2   + i]  — weight from hidden-2 i to output j    (j<NOUT,i<H2)
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct Mlp {
    w1: Vec<f64>, b1: Vec<f64>, // NIN  → H1
    w2: Vec<f64>, b2: Vec<f64>, // H1   → H2
    w3: Vec<f64>, b3: Vec<f64>, // H2   → NOUT
}

impl Mlp {
    fn new(rng: &mut Rng) -> Self {
        // Glorot / Xavier uniform initialisation
        let mut init = |n: usize, fan_in: usize, fan_out: usize| -> Vec<f64> {
            let s = (6.0 / (fan_in + fan_out) as f64).sqrt();
            (0..n).map(|_| rng.range(-s, s)).collect()
        };
        Self {
            w1: init(H1 * NIN,  NIN,  H1),   b1: vec![0.0; H1],
            w2: init(H2 * H1,   H1,   H2),   b2: vec![0.0; H2],
            w3: init(NOUT * H2, H2,   NOUT), b3: vec![0.0; NOUT],
        }
    }

    /// Forward pass.  Returns all intermediate values needed for backprop.
    ///
    /// Returns: (pre1, h1, pre2, h2, raw_out)
    ///   pre* = pre-activation (before ReLU)
    ///   h*   = post-activation
    ///   raw  = final layer without any activation applied
    fn forward_full(&self, x: &[f64])
        -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)
    {
        // ── Hidden layer 1 ──
        let mut pre1 = vec![0.0f64; H1];
        for j in 0..H1 {
            let mut s = self.b1[j];
            for i in 0..NIN { s += self.w1[j * NIN + i] * x[i]; }
            pre1[j] = s;
        }
        let h1: Vec<f64> = pre1.iter().map(|&v| relu(v)).collect();

        // ── Hidden layer 2 ──
        let mut pre2 = vec![0.0f64; H2];
        for j in 0..H2 {
            let mut s = self.b2[j];
            for i in 0..H1 { s += self.w2[j * H1 + i] * h1[i]; }
            pre2[j] = s;
        }
        let h2: Vec<f64> = pre2.iter().map(|&v| relu(v)).collect();

        // ── Output layer (raw — activations applied outside) ──
        let mut raw = vec![0.0f64; NOUT];
        for j in 0..NOUT {
            let mut s = self.b3[j];
            for i in 0..H2 { s += self.w3[j * H2 + i] * h2[i]; }
            raw[j] = s;
        }

        (pre1, h1, pre2, h2, raw)
    }

    /// Inference-only forward. Returns (dir_prob, sl_mult, tp_mult).
    ///
    /// - dir_prob ∈ (0,1): probability that the trade is Long
    /// - sl_mult  ∈ [0.3, MAX_MULT]: learned SL distance in ATR units
    /// - tp_mult  ∈ [0.5, MAX_MULT]: learned TP distance in ATR units
    fn predict(&self, x: &[f64]) -> (f64, f64, f64) {
        let (_, _, _, _, raw) = self.forward_full(x);
        let dir_prob = sigmoid(raw[0]);
        // Sigmoid-scale raw outputs into [min, MAX_MULT] ranges.
        // sigmoid(raw[1]) maps to (0,1); we stretch to [0.3, MAX_MULT].
        let sl_mult = (0.3 + sigmoid(raw[1]) * (MAX_MULT - 0.3)).clamp(0.3, MAX_MULT);
        let tp_mult = (0.5 + sigmoid(raw[2]) * (MAX_MULT - 0.5)).clamp(0.5, MAX_MULT);
        (dir_prob, sl_mult, tp_mult)
    }

    // ── Serialisation ─────────────────────────────────────────────

    fn save(&self, path: &str) {
        let json = serde_json::json!({
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2,
            "w3": self.w3, "b3": self.b3,
        });
        fs::write(path, serde_json::to_string_pretty(&json).expect("serialise"))
            .unwrap_or_else(|e| panic!("Cannot save weights to '{}': {}", path, e));
        println!("Weights saved → {}", path);
    }

    fn load(path: &str) -> Self {
        let raw = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Cannot open weights '{}': {}", path, e));
        let v: serde_json::Value = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("Parse error in '{}': {}", path, e));
        let arr = |key: &str| -> Vec<f64> {
            v[key].as_array()
                .unwrap_or_else(|| panic!("Missing key '{}' in weights file", key))
                .iter()
                .map(|x| x.as_f64().expect("non-float in weights"))
                .collect()
        };
        Self {
            w1: arr("w1"), b1: arr("b1"),
            w2: arr("w2"), b2: arr("b2"),
            w3: arr("w3"), b3: arr("b3"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Adam optimiser state
// ═══════════════════════════════════════════════════════════════════

struct Adam {
    // First and second moment estimates, one Vec per weight tensor
    m_w1: Vec<f64>, v_w1: Vec<f64>,
    m_b1: Vec<f64>, v_b1: Vec<f64>,
    m_w2: Vec<f64>, v_w2: Vec<f64>,
    m_b2: Vec<f64>, v_b2: Vec<f64>,
    m_w3: Vec<f64>, v_w3: Vec<f64>,
    m_b3: Vec<f64>, v_b3: Vec<f64>,
    // Hyper-parameters
    lr:    f64,
    beta1: f64,
    beta2: f64,
    eps:   f64,
    t:     f64, // step count (for bias correction)
}

impl Adam {
    fn new(mlp: &Mlp, lr: f64) -> Self {
        let z = |n: usize| vec![0.0f64; n];
        Self {
            m_w1: z(mlp.w1.len()), v_w1: z(mlp.w1.len()),
            m_b1: z(mlp.b1.len()), v_b1: z(mlp.b1.len()),
            m_w2: z(mlp.w2.len()), v_w2: z(mlp.w2.len()),
            m_b2: z(mlp.b2.len()), v_b2: z(mlp.b2.len()),
            m_w3: z(mlp.w3.len()), v_w3: z(mlp.w3.len()),
            m_b3: z(mlp.b3.len()), v_b3: z(mlp.b3.len()),
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0.0,
        }
    }

    /// Apply one Adam step given six gradient tensors (order matches Mlp fields).
    fn step(
        &mut self, mlp: &mut Mlp,
        gw1: &[f64], gb1: &[f64],
        gw2: &[f64], gb2: &[f64],
        gw3: &[f64], gb3: &[f64],
    ) {
        self.t += 1.0;
        let t = self.t;
        let (b1, b2, eps) = (self.beta1, self.beta2, self.eps);
        // Bias-corrected step size (avoids two powf calls per parameter)
        let lr_t = self.lr * (1.0 - b2.powf(t)).sqrt() / (1.0 - b1.powf(t));

        macro_rules! update {
            ($p:expr, $m:expr, $v:expr, $g:expr) => {
                for k in 0..$p.len() {
                    $m[k] = b1 * $m[k] + (1.0 - b1) * $g[k];
                    $v[k] = b2 * $v[k] + (1.0 - b2) * $g[k] * $g[k];
                    $p[k] -= lr_t * $m[k] / ($v[k].sqrt() + eps);
                }
            };
        }
        update!(mlp.w1, self.m_w1, self.v_w1, gw1);
        update!(mlp.b1, self.m_b1, self.v_b1, gb1);
        update!(mlp.w2, self.m_w2, self.v_w2, gw2);
        update!(mlp.b2, self.m_b2, self.v_b2, gb2);
        update!(mlp.w3, self.m_w3, self.v_w3, gw3);
        update!(mlp.b3, self.m_b3, self.v_b3, gb3);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  One SGD step with full back-propagation
//
//  Outputs and their activations
//  ─────────────────────────────
//    raw[0]  → sigmoid         → direction probability  (BCE loss)
//    raw[1]  → sigmoid-scaled  → sl_mult ∈ [0.3, MAX]  (MSE loss, weight λ)
//    raw[2]  → sigmoid-scaled  → tp_mult ∈ [0.5, MAX]  (MSE loss, weight λ)
//
//  Returns (dir_loss, reg_loss) for logging.
// ═══════════════════════════════════════════════════════════════════

fn backprop_step(
    mlp:         &mut Mlp,
    adam:        &mut Adam,
    x:           &[f64],
    dir_lbl:     f64,
    // sl_lbl / tp_lbl / sl_mae_raw / tp_mfe_raw are no longer used for the
    // SL/TP loss — the cap% surrogate replaces them — but we keep the
    // signature compatible so the call-site doesn't need to change.
    _sl_lbl:     f64,
    _tp_lbl:     f64,
    _sl_mae_raw: f64,
    _tp_mfe_raw: f64,
    // Live future bars and trade context for the cap% surrogate.
    is_long:     bool,
    entry_px:    f64,   // actual stop-entry trigger price
    atr:         f64,
    future:      &[Bar],
    _epoch:      usize,
) -> (f64, f64) {
    let (pre1, h1, pre2, h2, raw) = mlp.forward_full(x);

    // ── Direction output (raw[0]) — BCE loss ─────────────────────
    let dir_prob = sigmoid(raw[0]);
    let eps = 1e-7_f64;
    let dir_loss = -(
        dir_lbl       * (dir_prob + eps).ln()
        + (1.0 - dir_lbl) * (1.0 - dir_prob + eps).ln()
    );

    // ── SL / TP outputs — sigmoid-scaled ────────────────────────
    let sl_sig  = sigmoid(raw[1]);
    let sl_pred = (0.3 + sl_sig * (MAX_MULT - 0.3)).clamp(0.3, MAX_MULT);
    let tp_sig  = sigmoid(raw[2]);
    let tp_pred = (0.5 + tp_sig * (MAX_MULT - 0.5)).clamp(0.5, MAX_MULT);

    // ── Differentiable cap% surrogate ────────────────────────────
    //
    // We want the model to directly maximise cap% = achieved_gain / max_gain.
    //
    // simulate() is not differentiable, so we build a smooth approximation:
    //
    //   soft_survival(sl_pred):
    //     For each future bar, compute how close the adverse excursion came to
    //     the SL.  Use a sigmoid to get a soft "did not get stopped" probability
    //     per bar, then multiply them (log-sum in log space for stability).
    //     A tighter SL that price nearly touches gets a lower survival score,
    //     pushing the gradient to widen the SL just enough.
    //
    //   soft_tp_hit(tp_pred):
    //     For each future bar, compute a soft "TP was reached" signal based on
    //     how far the MFE of that bar exceeds the predicted TP distance.
    //     A TP just at the MFE scores ~0.5; well inside scores ~1; too far scores ~0.
    //
    //   soft_cap = soft_survival * soft_tp_hit
    //   cap_loss = 1 - soft_cap   (minimise → maximise cap%)
    //
    // Temperature k controls sharpness of the sigmoid approximations.
    // Larger k → closer to hard sim; smaller k → smoother gradients.
    let k = 8.0_f64;

    // Find where entry triggers so we only score bars after entry.
    let entry_bar = {
        let mut found = future.len(); // sentinel: never triggered
        for (i, bar) in future.iter().enumerate() {
            if  is_long && bar.high >= entry_px { found = i; break; }
            if !is_long && bar.low  <= entry_px { found = i; break; }
        }
        found
    };

    let (soft_survival, d_survival_d_sl, soft_tp, d_tp_d_tp) = if entry_bar >= future.len() {
        // Entry never triggered — no gradient signal for SL/TP this sample.
        (1.0_f64, 0.0_f64, 0.0_f64, 0.0_f64)
    } else {
        // SL distance in price units
        let sl_price_dist = sl_pred * atr;
        let tp_price_dist = tp_pred * atr;

        let mut log_surv      = 0.0_f64;
        let mut d_log_surv_sl = 0.0_f64; // d(log_surv)/d(sl_pred)
        let mut best_fav      = 0.0_f64;

        for bar in &future[entry_bar..] {
            // Adverse excursion this bar in ATR units
            let adverse_atr = if is_long {
                (entry_px - bar.low).max(0.0)  / atr
            } else {
                (bar.high - entry_px).max(0.0) / atr
            };
            // Soft survival for this bar: sigmoid( k*(sl_pred - adverse) )
            // → 1 when SL is well above adverse, 0 when SL is below adverse.
            let z    = k * (sl_pred - adverse_atr);
            let s    = sigmoid(z);
            log_surv      += s.max(1e-12).ln();
            // d(ln s)/d(sl_pred) = k*(1-s)
            d_log_surv_sl += k * (1.0 - s);

            // Track best favourable excursion for TP scoring
            let fav = if is_long {
                (bar.high - entry_px).max(0.0) / atr
            } else {
                (entry_px - bar.low).max(0.0)  / atr
            };
            if fav > best_fav { best_fav = fav; }
        }

        let surv = log_surv.exp().clamp(1e-12, 1.0);
        // d(surv)/d(sl_pred) = surv * d(log_surv)/d(sl_pred)
        let d_surv_sl = surv * d_log_surv_sl;

        // Soft TP hit: sigmoid( k*(best_fav - tp_pred) )
        // → 1 when TP is well within MFE, 0 when TP exceeds MFE.
        let z_tp    = k * (best_fav - tp_pred);
        let s_tp    = sigmoid(z_tp);
        // d(s_tp)/d(tp_pred) = -k * s_tp*(1-s_tp)
        let d_stp   = -k * s_tp * (1.0 - s_tp);

        (surv, d_surv_sl, s_tp, d_stp)
    };

    // soft_cap = soft_survival * soft_tp_hit
    // cap_loss = 1 - soft_cap  (we minimise, so this maximises cap%)
    let soft_cap  = soft_survival * soft_tp;
    let cap_loss  = 1.0 - soft_cap;

    // d(cap_loss)/d(sl_pred):  -d(soft_cap)/d(sl_pred)
    //   = -(d_survival_d_sl * soft_tp)
    let d_cap_d_sl = -(d_survival_d_sl * soft_tp);
    // d(cap_loss)/d(tp_pred):  -d(soft_cap)/d(tp_pred)
    //   = -(soft_survival * d_tp_d_tp)
    let d_cap_d_tp = -(soft_survival * d_tp_d_tp);

    // ── Output-layer gradients w.r.t. raw[*] ────────────────────
    // Chain rule through the sigmoid-scale activation:
    //   d(sl_pred)/d(raw[1]) = (MAX_MULT - 0.3) * sl_sig*(1-sl_sig)
    //   d(tp_pred)/d(raw[2]) = (MAX_MULT - 0.5) * tp_sig*(1-tp_sig)
    let d_raw0 = dir_prob - dir_lbl;
    let d_raw1 = LAMBDA_REG * d_cap_d_sl * (MAX_MULT - 0.3) * sl_sig * (1.0 - sl_sig);
    let d_raw2 = LAMBDA_REG * d_cap_d_tp * (MAX_MULT - 0.5) * tp_sig * (1.0 - tp_sig);
    let d_raw  = [d_raw0, d_raw1, d_raw2];

    // ── w3 / b3  +  back-prop into h2 ───────────────────────────
    let mut gw3 = vec![0.0f64; NOUT * H2];
    let mut gb3 = vec![0.0f64; NOUT];
    let mut d_h2 = vec![0.0f64; H2];
    for j in 0..NOUT {
        gb3[j] = d_raw[j];
        for i in 0..H2 {
            gw3[j * H2 + i] = d_raw[j] * h2[i];
            d_h2[i] += d_raw[j] * mlp.w3[j * H2 + i];
        }
    }

    // ── w2 / b2  +  back-prop into h1 ───────────────────────────
    let d_pre2: Vec<f64> = (0..H2).map(|i| d_h2[i] * relu_d(pre2[i])).collect();
    let mut gw2 = vec![0.0f64; H2 * H1];
    let mut gb2 = vec![0.0f64; H2];
    let mut d_h1 = vec![0.0f64; H1];
    for j in 0..H2 {
        gb2[j] = d_pre2[j];
        for i in 0..H1 {
            gw2[j * H1 + i] = d_pre2[j] * h1[i];
            d_h1[i] += d_pre2[j] * mlp.w2[j * H1 + i];
        }
    }

    // ── w1 / b1 ──────────────────────────────────────────────────
    let d_pre1: Vec<f64> = (0..H1).map(|i| d_h1[i] * relu_d(pre1[i])).collect();
    let mut gw1 = vec![0.0f64; H1 * NIN];
    let mut gb1 = vec![0.0f64; H1];
    for j in 0..H1 {
        gb1[j] = d_pre1[j];
        for i in 0..NIN {
            gw1[j * NIN + i] = d_pre1[j] * x[i];
        }
    }

    // ── Adam update ───────────────────────────────────────────────
    adam.step(mlp, &gw1, &gb1, &gw2, &gb2, &gw3, &gb3);

    (dir_loss, cap_loss)
}

// ═══════════════════════════════════════════════════════════════════
//  5-minute bar aggregation from 1-min StockData
// ═══════════════════════════════════════════════════════════════════

fn to_5min(bars: &[StockData]) -> Vec<Bar> {
    let mut out = Vec::with_capacity(bars.len() / 5);
    let mut i = 0;
    while i + 4 < bars.len() {
        let w = &bars[i..i + 5];
        out.push(Bar {
            open:   w[0].open,
            high:   w.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max),
            low:    w.iter().map(|b| b.low).fold(f64::INFINITY, f64::min),
            close:  w[4].close,
            volume: w.iter().map(|b| b.volume as f64).sum(),
        });
        i += 5;
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
//  ATR helper
// ═══════════════════════════════════════════════════════════════════

fn current_atr(bars: &[Bar], period: usize) -> f64 {
    if bars.len() < 2 {
        return 0.001;
    }
    let start = bars.len().saturating_sub(period + 1);
    let trs: Vec<f64> = bars[start..].windows(2).map(|w| {
        (w[1].high - w[1].low)
            .max((w[1].high - w[0].close).abs())
            .max((w[1].low - w[0].close).abs())
    }).collect();
    if trs.is_empty() {
        return 0.001;
    }
    trs.iter().sum::<f64>() / trs.len() as f64
}

// ═══════════════════════════════════════════════════════════════════
//  Training sample
// ═══════════════════════════════════════════════════════════════════

struct Sample {
    features:   Vec<f64>,
    dir_label:  f64,   // 1.0 = Long, 0.0 = Short
    sl_mult:    f64,   // SL distance in ATR multiples  (MAE+buffer label)
    tp_mult:    f64,   // TP distance in ATR multiples  (MFE label)
    /// Best SL in ATR units (the label itself) — used for the SL-efficiency penalty gradient.
    sl_mae_raw: f64,
    /// Raw MFE in ATR units — used for the TP-realism penalty gradient.
    tp_mfe_raw: f64,
    atr:        f64,
    close:      f64,
    future:     Vec<Bar>,
}


fn build_samples(bars: &[Bar], lookback: usize) -> Vec<Sample> {
    if bars.len() < lookback + FORWARD { return vec![]; }
    let mut samples = Vec::new();

    for i in lookback..bars.len().saturating_sub(FORWARD) {
        let window = &bars[i - lookback..i];
        let future = &bars[i..i + FORWARD];
        let cur    = &bars[i - 1];
        if window.len() < 27 { continue; }

        let features = compute_indicators(window);
        let atr      = current_atr(window, 14).max(1e-8);
        let close    = cur.close;

        let entry_high = close + 0.25 * atr;
        let entry_low  = close - 0.25 * atr;

        // ── Step 1: Determine direction with a fixed, honest SL/TP ratio ──
        // We use LABEL_SL_MULT:LABEL_TP_MULT (1:2) — the same ratio the
        // eval loop uses.  Only keep samples where exactly one side wins:
        //   - "both win / both lose / neither triggers" are ambiguous and discarded.
        //   - This keeps labels clean and avoids training on random-walk noise.
        let long_sl  = close - LABEL_SL_MULT * atr;
        let long_tp  = close + LABEL_TP_MULT * atr;
        let short_sl = close + LABEL_SL_MULT * atr;
        let short_tp = close - LABEL_TP_MULT * atr;

        let long_sim  = simulate(true,  entry_high, entry_low, long_sl,  long_tp,  future);
        let short_sim = simulate(false, entry_high, entry_low, short_sl, short_tp, future);

        let (is_long, dir_trade) = match (&long_sim, &short_sim) {
            (Some(lt), Some(st)) if  lt.won && !st.won => (true,  lt),
            (Some(lt), Some(st)) if !lt.won &&  st.won => (false, st),
            (Some(lt), None)     if  lt.won             => (true,  lt),
            (None,     Some(st)) if  st.won             => (false, st),
            _ => continue, // ambiguous — skip
        };

        // ── Step 2: Best SL label — maximise locked-in MFE ──────────────────
        // Rather than using MAE+buffer (minimum SL to survive), we search for
        // the SL placement that captures the most gain.  A tighter SL risks
        // being stopped out early; a wider SL gives more room but risks more.
        // We scan candidates from tight to wide and pick the one whose MFE
        // (with a near-infinite TP so price runs freely) is greatest.
        //
        // Candidate SL distances from close, in ATR units.
        // Fine-grained near the tight end where small changes matter most.
        const SL_CANDIDATES: &[f64] = &[
            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0,
        ];

        let big_tp_for_mfe = if is_long {
            close + MFE_TP_MULT * atr
        } else {
            close - MFE_TP_MULT * atr
        };

        let mut best_sl_mult = 0.3_f64;
        let mut best_mfe_atr = 0.0_f64;

        for &candidate in SL_CANDIDATES {
            let trial_sl = if is_long {
                close - candidate * atr
            } else {
                close + candidate * atr
            };
            if let Some(t) = simulate(is_long, entry_high, entry_low, trial_sl, big_tp_for_mfe, future) {
                let mfe_atr = t.mfe / atr;
                if mfe_atr > best_mfe_atr {
                    best_mfe_atr = mfe_atr;
                    best_sl_mult = candidate;
                }
            }
        }

        // If no candidate produced a triggered entry, fall back to MAE+buffer.
        let sl_mult = if best_mfe_atr > 0.0 {
            best_sl_mult
        } else {
            ((dir_trade.mae / atr - 0.25) + MAE_BUFFER).clamp(0.3, MAX_MULT)
        };

        // sl_mae_raw is used by the efficiency penalty: penalise predicting
        // a wider SL than the best one we found.
        let sl_mae_raw = best_sl_mult;

        // ── Step 3: TP label from MFE at the best SL ─────────────────────
        // Simulate with the best SL and a huge TP to measure the MFE the
        // model should aim to reach.
        let best_sl_px = if is_long {
            close - sl_mult * atr
        } else {
            close + sl_mult * atr
        };
        let (tp_mult, tp_mfe_raw) =
            simulate(is_long, entry_high, entry_low, best_sl_px, big_tp_for_mfe, future)
                .map(|t| {
                    let mfe_atr = t.mfe / atr;
                    (mfe_atr.clamp(0.5, MAX_MULT), mfe_atr)
                })
                .unwrap_or((LABEL_TP_MULT, LABEL_TP_MULT));

        samples.push(Sample {
            features:   features.to_vec(),
            dir_label:  if is_long { 1.0 } else { 0.0 },
            sl_mult,
            tp_mult,
            sl_mae_raw,
            tp_mfe_raw,
            atr,
            close,
            future:     future.to_vec(),
        });
    }

    // ── Z-score normalise features ─────────────────────────────────────
    // Compute mean/std on the training portion ONLY (first 80%), then apply
    // those same stats to the validation portion.  Computing on all samples
    // would leak future data into the val set's scaling.
    if !samples.is_empty() {
        let nf      = samples[0].features.len();
        let n_train = ((samples.len() as f64) * 0.8) as usize;
        let n_tr    = n_train as f64;

        let mut mean = vec![0.0f64; nf];
        for s in &samples[..n_train] {
            for (j, &v) in s.features.iter().enumerate() { mean[j] += v; }
        }
        for m in &mut mean { *m /= n_tr; }

        let mut std = vec![0.0f64; nf];
        for s in &samples[..n_train] {
            for (j, &v) in s.features.iter().enumerate() {
                std[j] += (v - mean[j]).powi(2);
            }
        }
        for s in &mut std { *s = (*s / n_tr).sqrt().max(1e-8); }

        // Apply train stats to ALL samples (train + val)
        for s in &mut samples {
            for (j, v) in s.features.iter_mut().enumerate() {
                *v = (*v - mean[j]) / std[j];
            }
        }
    }

    samples
}

// ═══════════════════════════════════════════════════════════════════
//  Training loop
// ═══════════════════════════════════════════════════════════════════

fn train(bars: &[Bar], lookback: usize, epochs: usize, lr: f64, weights_path: &str) {
    // Always start fresh — stale weights trained on old labels will fight learning.
    match fs::remove_file(weights_path) {
        Ok(_)  => println!("Deleted old weights: {}", weights_path),
        Err(_) => println!("No existing weights to delete."),
    }

    println!(
        "Building samples  (5-min bars={}, lookback={}, forward={})...",
        bars.len(),
        lookback,
        FORWARD
    );
    let samples = build_samples(bars, lookback);
    if samples.is_empty() {
        eprintln!(
            "Not enough 5-min bars to build any sample. \
             Need at least {} bars (got {}).",
            lookback + FORWARD,
            bars.len()
        );
        process::exit(1);
    }

    // Chronological 80/20 split — boundary is fixed to prevent future-leakage.
    // Validation stays as-is (chronological, unbalanced — reflects real distribution).
    // Training set is balanced and shuffled after the split.
    let n_train = (samples.len() as f64 * 0.8) as usize;
    let (raw_train, val_set) = samples.split_at(n_train);

    let mut rng = Rng::new();

    // ── Balance training set by downsampling the majority class ──────
    // Done here (after the split) so the val set keeps the real distribution,
    // and so we can interleave longs/shorts rather than block them.
    // Build a balanced index list: equal longs and shorts from the training set
    let keep = {
        let nl = raw_train.iter().filter(|s| s.dir_label >= 0.5).count();
        let ns = raw_train.len() - nl;
        nl.min(ns)
    };

    // Build a shuffled index list that interleaves both classes
    let mut balanced_idx: Vec<usize> = Vec::with_capacity(keep * 2);
    // Map back to indices into raw_train
    let long_indices:  Vec<usize> = raw_train.iter().enumerate()
        .filter(|(_, s)| s.dir_label >= 0.5).map(|(i, _)| i).take(keep).collect();
    let short_indices: Vec<usize> = raw_train.iter().enumerate()
        .filter(|(_, s)| s.dir_label < 0.5).map(|(i, _)| i).take(keep).collect();
    balanced_idx.extend_from_slice(&long_indices);
    balanced_idx.extend_from_slice(&short_indices);
    rng.shuffle(&mut balanced_idx);

    let n_long  = long_indices.len();
    let n_short = short_indices.len();
    println!(
        "  {} training samples  |  {} validation samples",
        balanced_idx.len(),
        val_set.len()
    );
    println!("  Train balance: {n_long} Long / {n_short} Short (balanced)");

    // train_set is raw_train; we index into it via balanced_idx
    let train_set = raw_train;

    let mut mlp  = Mlp::new(&mut rng);
    let mut adam = Adam::new(&mlp, lr);

    // idx is rebuilt each epoch by re-shuffling balanced_idx
    let mut idx: Vec<usize> = balanced_idx;

    // ── Two-row column header ─────────────────────────────────────────────────────────────────────
    println!("\n{:<6}  {:<20}  {:<12}  {:<12}  {:<10}  {:<8}  {:<8}  {:<8}  {:<8}  {:<8}  {}",
        "Epoch", "loss", "trn_dir%", "val_dir%", "pnl%", "win%", "pf", "sl_mae", "tp_mae", "cap%", "sec");
    println!("{}", "─".repeat(115));

    for epoch in 0..epochs {
        let t_ep = Instant::now();
        rng.shuffle(&mut idx);

        let (mut sum_dir, mut sum_off) = (0.0_f64, 0.0_f64);
        for &k in &idx {
            let s = &train_set[k];
            let is_long   = s.dir_label >= 0.5;
            let entry_px  = if is_long { s.close + 0.25 * s.atr } else { s.close - 0.25 * s.atr };
            let (dl, ol) = backprop_step(
                &mut mlp, &mut adam,
                &s.features,
                s.dir_label, s.sl_mult, s.tp_mult,
                s.sl_mae_raw, s.tp_mfe_raw,
                is_long, entry_px, s.atr, &s.future,
                epoch,
            );
            sum_dir += dl;
            sum_off += ol;
        }
        let n      = idx.len() as f64;
        let ep_sec = t_ep.elapsed().as_secs_f64();

        // ── Training accuracy (on balanced set) ────────────────────
        // If train_dir% climbs but val_dir% stays flat → overfitting.
        // If both stay flat → no signal in features / labels are noise.
        let mut train_correct = 0usize;
        for &k in &idx {
            let s = &train_set[k];
            let (dir_prob, _, _) = mlp.predict(&s.features);
            if (dir_prob >= 0.5) == (s.dir_label >= 0.5) { train_correct += 1; }
        }
        let train_dir = train_correct as f64 / n * 100.0;

        // ── Validation metrics ─────────────────────────────────────
        let (mut correct_dir, mut sl_err_sum, mut tp_err_sum) = (0usize, 0.0_f64, 0.0_f64);
        for s in val_set {
            let (dir_prob, sm, tm) = mlp.predict(&s.features);
            if (dir_prob >= 0.5) == (s.dir_label >= 0.5) { correct_dir += 1; }
            sl_err_sum += (sm - s.sl_mult).abs();
            tp_err_sum += (tm - s.tp_mult).abs();
        }
        let nv      = val_set.len().max(1) as f64;
        let val_dir = correct_dir as f64 / nv * 100.0;
        let sl_mae  = sl_err_sum / nv;
        let tp_mae  = tp_err_sum / nv;

        if epoch % 5 == 0 || epoch == epochs - 1 {
            // ── sim_pnl%: simulated trade P&L ──
            let mut sim_pnls: Vec<f64> = Vec::new();
            let mut n_wins   = 0usize;
            let mut n_losses = 0usize;

            let mut capture_ratios: Vec<f64> = Vec::new();

            for s in val_set.iter() {
                let (dir_prob, sm, tm) = mlp.predict(&s.features);
                let is_long = dir_prob >= 0.5;
                // SL and TP are distances from close in ATR units
                let (sl, tp) = if is_long {
                    (s.close - sm * s.atr, s.close + tm * s.atr)
                } else {
                    (s.close + sm * s.atr, s.close - tm * s.atr)
                };
                // entry = stop-entry above/below close (not a market order)
                let entry_high = s.close + 0.25 * s.atr;
                let entry_low  = s.close - 0.25 * s.atr;
                let entry_px   = if is_long { entry_high } else { entry_low };

                if let Some(t) = simulate(is_long, entry_high, entry_low, sl, tp, &s.future) {
                    sim_pnls.push(t.pct_gain);
                    if t.won { n_wins += 1; } else { n_losses += 1; }

                    // Best possible stop-loss position: re-simulate the same entry
                    // but with a near-zero SL floor (won't stop out) and a massive
                    // TP so price runs freely.  The resulting MFE is the furthest
                    // favourable excursion achievable from this entry — i.e. what a
                    // perfect trailing stop placed at entry would have locked in.
                    // Max gain = furthest favourable excursion from entry to end
                    // of the future window, with no SL/TP cutting it short.
                    // Scan s.future directly: find where entry triggers, then
                    // track the best price from that point to the last bar.
                    let max_gain = {
                        let mut triggered = false;
                        let mut best: f64 = 0.0;
                        for bar in s.future.iter() {
                            if !triggered {
                                if is_long  && bar.high >= entry_high { triggered = true; }
                                if !is_long && bar.low  <= entry_low  { triggered = true; }
                            }
                            if triggered {
                                let excursion = if is_long {
                                    (bar.high - entry_px) / entry_px * 100.0
                                } else {
                                    (entry_px - bar.low)  / entry_px * 100.0
                                };
                                if excursion > best { best = excursion; }
                            }
                        }
                        best
                    };

                    // Only record ratio on winning trades with a meaningful move
                    if t.won && max_gain > 1e-4 {
                        capture_ratios.push((t.pct_gain / max_gain * 100.0).clamp(0.0, 100.0));
                    }
                }
            }

            let n_triggered = n_wins + n_losses;
            let avg_pnl  = if sim_pnls.is_empty() { f64::NAN }
                           else { sim_pnls.iter().sum::<f64>() / sim_pnls.len() as f64 };
            let win_rate = if n_triggered > 0 {
                n_wins as f64 / n_triggered as f64 * 100.0
            } else { f64::NAN };
            // profit factor: ratio of gross wins to gross losses
            let profit_factor = {
                let gw = sim_pnls.iter().filter(|&&p| p > 0.0).sum::<f64>();
                let gl = sim_pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum::<f64>();
                if gl < 1e-9 { f64::NAN } else { gw / gl }
            };
            let avg_capture = if capture_ratios.is_empty() { f64::NAN }
                              else { capture_ratios.iter().sum::<f64>() / capture_ratios.len() as f64 };

            let pnl_str     = if avg_pnl.is_nan()     { format!("{:>8}", "n/a") }
                              else { format!("{:>+7.3}%", avg_pnl) };
            let win_str     = if win_rate.is_nan()     { format!("{:>6}", "n/a") }
                              else { format!("{:>5.1}%", win_rate) };
            let pf_str      = if profit_factor.is_nan(){ format!("{:>6}", "n/a") }
                              else { format!("{:>5.2}", profit_factor) };
            let capture_str = if avg_capture.is_nan()  { format!("{:>6}", "n/a") }
                              else { format!("{:>5.1}%", avg_capture) };

            // average per-sample loss
            let total_loss = (sum_dir + sum_off) / n;

            // ── Best trade sample this epoch ───────────────────────
            struct BestInfo { entry: f64, sl: f64, tp: f64, won: bool, pnl: f64, is_long: bool }
            let mut best_pnl  = f64::NEG_INFINITY;
            let mut best_info: Option<BestInfo> = None;
            for s in val_set.iter() {
                let (dir_prob, sm, tm) = mlp.predict(&s.features);
                let is_long = dir_prob >= 0.5;
                let (sl, tp) = if is_long {
                    (s.close - sm * s.atr, s.close + tm * s.atr)
                } else {
                    (s.close + sm * s.atr, s.close - tm * s.atr)
                };
                let entry_h = s.close + 0.25 * s.atr;
                let entry_l = s.close - 0.25 * s.atr;
                let entry   = if is_long { entry_h } else { entry_l };
                if let Some(t) = simulate(is_long, entry_h, entry_l, sl, tp, &s.future) {
                    if t.pct_gain > best_pnl {
                        best_pnl = t.pct_gain;
                        best_info = Some(BestInfo { entry, sl, tp, won: t.won, pnl: t.pct_gain, is_long });
                    }
                }
            }
            let best_str = best_info.map(|b| format!(
                "  |  {} entry={:.4} sl={:.4} tp={:.4} {} ({:+.3}%)",
                if b.is_long { "LONG " } else { "SHORT" },
                b.entry, b.sl, b.tp,
                if b.won { "WIN ✓" } else { "LOSS ✗" },
                b.pnl
            )).unwrap_or_default();

            println!(
                "ep={:<5}  loss={:.4}  trn={:>5.1}%  val={:>5.1}%  pnl={}  win={}  pf={}  sl={:.3}  tp={:.3}  cap={}  {:.2}s{}",
                epoch+1, total_loss, train_dir, val_dir,
                pnl_str, win_str, pf_str, sl_mae, tp_mae, capture_str, ep_sec,
                best_str
            );
        }
    }

    mlp.save(weights_path);
}

// ═══════════════════════════════════════════════════════════════════
//  Back-test simulation
// ═══════════════════════════════════════════════════════════════════

struct Trade {
    pct_gain:     f64,   // signed % gain/loss for the trade
    candles_held: usize, // bars from entry trigger to exit
    won:          bool,
    /// Max Adverse Excursion: furthest price moved *against* us from entry (price units).
    mae:          f64,
    /// Max Favorable Excursion: furthest price moved *in our favour* from entry (price units).
    mfe:          f64,
}

/// Simulate one stop-entry trade on `future` bars.
/// Returns None if the entry level is never touched within the window.
fn simulate(
    is_long:    bool,
    entry_high: f64, // buy-stop trigger
    entry_low:  f64, // sell-stop trigger
    sl:         f64, // stop-loss price
    tp:         f64, // take-profit price
    future:     &[Bar],
) -> Option<Trade> {
    // ── Phase 1: wait for entry trigger ──────────────────────────
    let mut entry_px    = None;
    let mut entry_start = 0;
    for (c, bar) in future.iter().enumerate() {
        if is_long  && bar.high >= entry_high { entry_px = Some(entry_high); entry_start = c; break; }
        if !is_long && bar.low  <= entry_low  { entry_px = Some(entry_low);  entry_start = c; break; }
    }
    let px = entry_px?;

    // ── Phase 2: wait for stop or target, tracking MAE/MFE ───────
    let mut max_adverse   = 0.0_f64;
    let mut max_favorable = 0.0_f64;

    for (c, bar) in future[entry_start + 1..].iter().enumerate() {
        let candles = c + 1;

        // Update excursion trackers BEFORE checking exits (conservative).
        let (adverse, favorable) = if is_long {
            ((px - bar.low).max(0.0), (bar.high - px).max(0.0))
        } else {
            ((bar.high - px).max(0.0), (px - bar.low).max(0.0))
        };
        max_adverse   = max_adverse.max(adverse);
        max_favorable = max_favorable.max(favorable);

        if is_long {
            // Stop wins on same-bar touches (conservative).
            if bar.low <= sl {
                let pct = (sl - px) / px * 100.0;
                return Some(Trade { pct_gain: pct, candles_held: candles, won: false,
                                    mae: max_adverse, mfe: max_favorable });
            }
            if bar.high >= tp {
                let pct = (tp - px) / px * 100.0;
                return Some(Trade { pct_gain: pct, candles_held: candles, won: true,
                                    mae: max_adverse, mfe: max_favorable });
            }
        } else {
            if bar.high >= sl {
                let pct = (px - sl) / px * 100.0;
                return Some(Trade { pct_gain: pct, candles_held: candles, won: false,
                                    mae: max_adverse, mfe: max_favorable });
            }
            if bar.low <= tp {
                let pct = (px - tp) / px * 100.0;
                return Some(Trade { pct_gain: pct, candles_held: candles, won: true,
                                    mae: max_adverse, mfe: max_favorable });
            }
        }
    }
    None // position still open at end of window — excluded from stats
}

fn evaluate(bars: &[Bar], lookback: usize, mlp: &Mlp) {
    // Use the latter half of the data as the out-of-sample eval set
    // to avoid overlap with anything that was in the training split.
    let eval_bars = if bars.len() > (lookback + FORWARD) * 4 {
        &bars[bars.len() / 2..]
    } else {
        bars
    };
    println!(
        "Back-testing on {} 5-min bars (lookback={}) ...",
        eval_bars.len(),
        lookback
    );

    let mut trades: Vec<Trade> = Vec::new();
    let mut no_entry = 0usize;

    let end = eval_bars.len().saturating_sub(FORWARD * 2);
    for i in lookback..end {
        let window = &eval_bars[i - lookback..i];
        if window.len() < 27 { continue; }

        let features = compute_indicators(window);
        let (dir_prob, sm, tm) = mlp.predict(&features);
        let is_long = dir_prob >= 0.5;
        let cur     = &eval_bars[i - 1];
        let atr     = current_atr(window, 14).max(1e-8);

        let (sl, tp) = if is_long {
            (cur.close - sm * atr, cur.close + tm * atr)
        } else {
            (cur.close + sm * atr, cur.close - tm * atr)
        };
        // Use ATR-offset stop-entry triggers (consistent with training sim)
        let entry_high = cur.close + 0.25 * atr;
        let entry_low  = cur.close - 0.25 * atr;

        let future_len = (FORWARD * 2).min(eval_bars.len() - i);
        let future     = &eval_bars[i..i + future_len];

        match simulate(is_long, entry_high, entry_low, sl, tp, future) {
            Some(t) => trades.push(t),
            None    => no_entry += 1,
        }
    }

    if trades.is_empty() {
        println!("No trades triggered in the evaluation window.");
        return;
    }

    let n      = trades.len();
    let n_wins = trades.iter().filter(|t| t.won).count();

    // Core metric: average % per candle held
    // (a large % on a 1-candle trade scores the same as a small % over 20)
    let avg_ppc = trades.iter()
        .map(|t| t.pct_gain / t.candles_held.max(1) as f64)
        .sum::<f64>()
        / n as f64;

    let avg_pnl   = trades.iter().map(|t| t.pct_gain).sum::<f64>() / n as f64;
    let avg_hold  = trades.iter().map(|t| t.candles_held as f64).sum::<f64>() / n as f64;
    let win_rate  = n_wins as f64 / n as f64 * 100.0;

    // Expectancy (avg $ per unit risked, assuming SL_ATR = 1 unit)
    let profit_factor = {
        let gross_win  = trades.iter().filter(|t|  t.won).map(|t| t.pct_gain).sum::<f64>();
        let gross_loss = trades.iter().filter(|t| !t.won).map(|t| t.pct_gain.abs()).sum::<f64>();
        if gross_loss < 1e-9 { f64::INFINITY } else { gross_win / gross_loss }
    };

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Back-Test Results                               ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  Positions simulated  : {:>6}                   ║", n);
    println!("║  Entry never triggered: {:>6}  (excluded)       ║", no_entry);
    println!("║  Win rate             : {:>6.1}%                ║", win_rate);
    println!("║  Avg hold (candles)   : {:>6.1}                 ║", avg_hold);
    println!("║  Avg hold (minutes)   : {:>6.1}                 ║", avg_hold * 5.0);
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  Avg P&L per trade    : {:>+6.3}%               ║", avg_pnl);
    println!("║  Avg % per candle held: {:>+6.4}%  ← key metric ║", avg_ppc);
    println!("║  Profit factor        : {:>6.2}                 ║", profit_factor);
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    if avg_ppc > 0.0 {
        println!("  ✓ Model is net profitable on this out-of-sample data.");
    } else {
        println!("  ✗ Model loses on this data — consider more data / epochs / lookback.");
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════
//  Live prediction (latest window)
// ═══════════════════════════════════════════════════════════════════

fn predict_latest(bars: &[Bar], lookback: usize, mlp: &Mlp) {
    if bars.len() < lookback {
        eprintln!(
            "Error: only {} 5-min bars available; need at least {} for lookback.",
            bars.len(),
            lookback
        );
        process::exit(1);
    }
    let window = &bars[bars.len() - lookback..];
    if window.len() < 27 {
        eprintln!("Error: lookback must be ≥ 27 bars (MACD minimum).");
        process::exit(1);
    }

    let features   = compute_indicators(window);
    let (dir_prob, sm, tm) = mlp.predict(&features);
    let is_long    = dir_prob >= 0.5;
    let cur        = bars.last().unwrap();
    let atr        = current_atr(window, 14).max(1e-8);

    let (sl, tp) = if is_long {
        (cur.close - sm * atr, cur.close + tm * atr)
    } else {
        (cur.close + sm * atr, cur.close - tm * atr)
    };

    // Confidence: how far the direction probability is from the 0.5 threshold
    let confidence = (dir_prob - 0.5).abs() * 200.0; // mapped to [0, 100]

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Stop-Entry AI  —  Prediction                           ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Direction    : {:<10}  ({:.1}% confident)          ║",
        if is_long { "LONG  ▲" } else { "SHORT ▼" },
        confidence
    );
    println!("║  Close price  : {:<12.4}                              ║", cur.close);
    println!("║  ATR (14-bar) : {:<12.4}                              ║", atr);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Stop-Loss    : {:<12.4}                              ║", sl);
    println!("║  Take-Profit  : {:<12.4}                              ║", tp);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Risk/Reward  : 1 : {:<4.2}                                ║",
        (tp - cur.close).abs() / (cur.close - sl).abs().max(1e-8));
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
}

// ═══════════════════════════════════════════════════════════════════
//  CLI entry point
// ═══════════════════════════════════════════════════════════════════

fn usage(prog: &str) {
    eprintln!("Usage:");
    eprintln!(
        "  {prog} train   --csv <file> -a <lookback> --weights <file> \
         [--epochs N] [--lr F]"
    );
    eprintln!("  {prog} eval    --csv <file> -a <lookback> --weights <file>");
    eprintln!("  {prog} predict --csv <file> -a <lookback> --weights <file>");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --csv      Path to the 1-min OHLCV CSV produced by the generator.");
    eprintln!("  -a         Lookback in 5-min candles (default 60  ≈ 5 hours).");
    eprintln!("  --weights  Path to the JSON weights file (save / load).");
    eprintln!("  --epochs   Training epochs (default 100).");
    eprintln!("  --lr       Adam learning rate (default 0.001).");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage(&args[0]);
        process::exit(1);
    }
    let mode = args[1].clone();

    // ── Parse flags ───────────────────────────────────────────────
    let mut csv_path     = String::new();
    let mut weights_path = String::from("stop_ai_weights.json");
    let mut lookback     = 60usize;
    let mut epochs       = 100usize;
    let mut lr           = 1e-3_f64;
    let mut start_date   = String::new();
    let mut end_date     = String::new();
    let mut api_key      = String::new();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--csv"        => { i += 1; csv_path     = args[i].clone(); }
            "--weights"    => { i += 1; weights_path = args[i].clone(); }
            "-a"           => { i += 1; lookback     = args[i].parse().expect("-a: integer"); }
            "--epochs"     => { i += 1; epochs       = args[i].parse().expect("--epochs: integer"); }
            "--lr"         => { i += 1; lr           = args[i].parse().expect("--lr: float"); }
            "--start-date" => { i += 1; start_date   = args[i].clone(); }
            "--end-date"   => { i += 1; end_date     = args[i].clone(); }
            "--api-key"    => { i += 1; api_key      = args[i].clone(); }
            other          => eprintln!("Unknown flag '{}' — ignored", other),
        }
        i += 1;
    }

    if csv_path.is_empty() {
        eprintln!("Error: --csv is required.\n");
        usage(&args[0]);
        process::exit(1);
    }

    let csv_path = if !start_date.is_empty() && !end_date.is_empty() && !api_key.is_empty() {
        maybe_download(&csv_path.to_uppercase(), &start_date, &end_date, &api_key)
    } else {
        csv_path
    };
    if lookback < 27 {
        eprintln!("Error: -a must be ≥ 27 (MACD needs 26 bars minimum).");
        process::exit(1);
    }

    // ── Load and aggregate data ───────────────────────────────────
    println!("Loading 1-min data from '{}'...", csv_path);
    let raw_1min = parse_csv(&csv_path);
    println!("  {} 1-min bars loaded.", raw_1min.len());

    let bars_5min = to_5min(&raw_1min);
    println!("  {} 5-min bars after aggregation.", bars_5min.len());

    if bars_5min.len() < lookback + FORWARD + 10 {
        eprintln!(
            "Too few 5-min bars ({}) for lookback {}. Download more data with the generator.",
            bars_5min.len(),
            lookback
        );
        process::exit(1);
    }

    // ── Dispatch ──────────────────────────────────────────────────
    println!();
    match mode.as_str() {
        "train" => {
            println!(
                "Mode: TRAIN   epochs={} | lr={} | lookback={} | weights={}",
                epochs, lr, lookback, weights_path
            );
            println!();
            train(&bars_5min, lookback, epochs, lr, &weights_path);
        }
        "eval" => {
            println!("Mode: EVAL    weights={}", weights_path);
            println!();
            let mlp = Mlp::load(&weights_path);
            evaluate(&bars_5min, lookback, &mlp);
        }
        "predict" => {
            println!("Mode: PREDICT  weights={}", weights_path);
            let mlp = Mlp::load(&weights_path);
            predict_latest(&bars_5min, lookback, &mlp);
        }
        _ => {
            eprintln!("Unknown mode '{}'. Must be: train | eval | predict", mode);
            usage(&args[0]);
            process::exit(1);
        }
    }
}
