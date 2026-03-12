// indicators.rs
//
// Technical analysis feature extraction for stock trainers.
//
// Computes a dense feature vector from a lookback window of OHLCV bars.
// All outputs are normalised to roughly [-1, 1] so the network sees
// consistent magnitude regardless of price level.
//
// Feature groups (total: INDICATOR_NF = 18):
//
//   [0]  EMA(9)  distance from close  — trend short
//   [1]  EMA(21) distance from close  — trend medium
//   [2]  EMA(9) slope (last bar)      — trend direction
//   [3]  EMA(9) > EMA(21) crossover   — binary trend signal
//
//   [4]  RSI(14) normalised           — momentum (0=oversold, 1=overbought)
//   [5]  MACD histogram               — momentum divergence
//   [6]  MACD signal direction        — smoothed momentum sign
//
//   [7]  Bollinger %B                 — price within band (0=low, 1=high)
//   [8]  Bollinger bandwidth          — volatility regime
//   [9]  ATR ratio (14-bar)           — current volatility vs recent avg
//
//   [10] Pivot support distance       — how far above nearest support
//   [11] Pivot resistance distance    — how far below nearest resistance
//   [12] SR zone strength             — how many pivots cluster nearby
//
//   [13] VWAP distance                — price vs intraday VWAP
//   [14] VWAP slope                   — VWAP trend direction
//
//   [15] OBV momentum (normalised)    — volume-weighted price direction
//   [16] Volume ratio (vs 20-bar avg) — unusual volume flag
//   [17] Candle body ratio            — bar structure (doji vs engulfing)

pub const INDICATOR_NF: usize = 18;

/// A minimal bar type accepted by this module. Your StockData can be
/// converted via the `AsBar` trait below (or just pass slices directly).
#[derive(Clone, Debug)]
pub struct Bar {
    pub open:   f64,
    pub high:   f64,
    pub low:    f64,
    pub close:  f64,
    pub volume: f64,
}

// ─────────────────────────────────────────────
//  Moving averages
// ─────────────────────────────────────────────

/// Exponential moving average over a slice of closes.
/// Returns the EMA value at the last bar.
pub fn ema(closes: &[f64], period: usize) -> f64 {
    if closes.is_empty() { return 0.0; }
    let k = 2.0 / (period as f64 + 1.0);
    let mut e = closes[0];
    for &c in &closes[1..] { e = c * k + e * (1.0 - k); }
    e
}

/// EMA slope: (ema_now - ema_prev) / ema_prev, clamped.
pub fn ema_slope(closes: &[f64], period: usize) -> f64 {
    if closes.len() < 2 { return 0.0; }
    let now  = ema(closes, period);
    let prev = ema(&closes[..closes.len() - 1], period);
    ((now - prev) / prev.abs().max(1e-8)).clamp(-0.02, 0.02)
}

// ─────────────────────────────────────────────
//  RSI
// ─────────────────────────────────────────────

/// Wilder RSI over `period` bars. Returns value in [0, 1].
pub fn rsi(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 { return 0.5; }
    let recent = &closes[closes.len() - period - 1..];
    let (mut avg_gain, mut avg_loss) = (0.0_f64, 0.0_f64);
    for w in recent.windows(2) {
        let d = w[1] - w[0];
        if d > 0.0 { avg_gain += d; } else { avg_loss += d.abs(); }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;
    if avg_loss < 1e-10 { return 1.0; }
    let rs = avg_gain / avg_loss;
    (rs / (1.0 + rs)).clamp(0.0, 1.0) // same as RSI/100 in [0,1]
}

// ─────────────────────────────────────────────
//  MACD
// ─────────────────────────────────────────────

/// Returns (histogram, signal_sign) normalised by close price.
/// Histogram = MACD line - signal line.
pub fn macd(closes: &[f64]) -> (f64, f64) {
    if closes.len() < 26 { return (0.0, 0.0); }
    let fast   = ema(closes, 12);
    let slow   = ema(closes, 26);
    let line   = fast - slow;
    // Signal uses the last 9 bars of MACD — approximate with a shorter window
    let n      = closes.len();
    let macd_series: Vec<f64> = (9..=n).map(|end| {
        let f = ema(&closes[..end], 12);
        let s = ema(&closes[..end], 26);
        f - s
    }).collect();
    let signal = ema(&macd_series, 9);
    let hist   = (line - signal) / closes.last().unwrap_or(&1.0).abs().max(1e-8);
    let sign   = signal.signum();
    (hist.clamp(-0.01, 0.01), sign)
}

// ─────────────────────────────────────────────
//  Bollinger Bands
// ─────────────────────────────────────────────

/// Returns (%B, bandwidth) for a 20-period, 2-sigma Bollinger Band.
/// %B = (close - lower) / (upper - lower), clamped to [0,1].
/// Bandwidth = (upper - lower) / middle, normalised.
pub fn bollinger(closes: &[f64], period: usize) -> (f64, f64) {
    if closes.len() < period { return (0.5, 0.0); }
    let window = &closes[closes.len() - period..];
    let mean   = window.iter().sum::<f64>() / period as f64;
    let var    = window.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / period as f64;
    let std    = var.sqrt().max(1e-8);
    let upper  = mean + 2.0 * std;
    let lower  = mean - 2.0 * std;
    let close  = *closes.last().unwrap();
    let pct_b  = ((close - lower) / (upper - lower).max(1e-8)).clamp(0.0, 1.0);
    let bw     = ((upper - lower) / mean.abs().max(1e-8)).clamp(0.0, 0.1);
    (pct_b, bw)
}

// ─────────────────────────────────────────────
//  Average True Range
// ─────────────────────────────────────────────

/// ATR ratio: current bar's true range vs the rolling ATR average.
/// > 1.0 means the bar is more volatile than recent average.
pub fn atr_ratio(bars: &[Bar], period: usize) -> f64 {
    if bars.len() < period + 1 { return 1.0; }
    let trs: Vec<f64> = bars.windows(2).map(|w| {
        let tr = (w[1].high - w[1].low)
            .max((w[1].high - w[0].close).abs())
            .max((w[1].low  - w[0].close).abs());
        tr / w[1].close.max(1e-8)
    }).collect();
    if trs.is_empty() { return 1.0; }
    let avg = trs[trs.len().saturating_sub(period)..].iter().sum::<f64>()
        / period.min(trs.len()) as f64;
    let last = *trs.last().unwrap();
    (last / avg.max(1e-8)).clamp(0.0, 5.0) / 5.0   // normalise to [0,1]
}

// ─────────────────────────────────────────────
//  Pivot Support / Resistance
// ─────────────────────────────────────────────

/// Identifies swing pivots (local highs/lows) in the lookback window,
/// then returns:
///   [0] distance above nearest support  (positive = above support, clamped)
///   [1] distance below nearest resistance (positive = below resistance)
///   [2] SR cluster strength (0-1): fraction of pivots within 0.5% of current price
pub fn pivot_sr(bars: &[Bar]) -> (f64, f64, f64) {
    if bars.len() < 5 { return (0.0, 0.0, 0.0); }
    let close = bars.last().unwrap().close;

    // Collect swing highs and lows using a 2-bar lookaround
    let mut supports:    Vec<f64> = Vec::new();
    let mut resistances: Vec<f64> = Vec::new();
    for i in 2..bars.len() - 2 {
        let h = bars[i].high;
        let l = bars[i].low;
        // Swing high: higher than 2 bars each side
        if h > bars[i-1].high && h > bars[i-2].high && h > bars[i+1].high && h > bars[i+2].high {
            resistances.push(h);
        }
        // Swing low: lower than 2 bars each side
        if l < bars[i-1].low && l < bars[i-2].low && l < bars[i+1].low && l < bars[i+2].low {
            supports.push(l);
        }
    }

    // Nearest support below current close
    let support_dist = supports.iter()
        .filter(|&&s| s < close)
        .map(|&s| (close - s) / close)
        .fold(f64::MAX, f64::min);
    let support_dist = if support_dist == f64::MAX { 0.0 } else { support_dist.clamp(0.0, 0.05) / 0.05 };

    // Nearest resistance above current close
    let resist_dist = resistances.iter()
        .filter(|&&r| r > close)
        .map(|&r| (r - close) / close)
        .fold(f64::MAX, f64::min);
    let resist_dist = if resist_dist == f64::MAX { 0.0 } else { resist_dist.clamp(0.0, 0.05) / 0.05 };

    // Cluster strength: what fraction of all pivots are within 0.5% of close?
    let all_pivots: Vec<f64> = supports.iter().chain(resistances.iter()).copied().collect();
    let cluster = if all_pivots.is_empty() { 0.0 } else {
        let near = all_pivots.iter().filter(|&&p| ((p - close) / close).abs() < 0.005).count();
        (near as f64 / all_pivots.len() as f64).clamp(0.0, 1.0)
    };

    (support_dist, resist_dist, cluster)
}

// ─────────────────────────────────────────────
//  VWAP
// ─────────────────────────────────────────────

/// VWAP over the full window. Returns:
///   [0] (close - vwap) / close  — signed distance, clamped
///   [1] VWAP slope (last 5 bars of cumulative VWAP)
pub fn vwap_features(bars: &[Bar]) -> (f64, f64) {
    if bars.is_empty() { return (0.0, 0.0); }
    let (mut cum_pv, mut cum_vol) = (0.0, 0.0);
    let vwaps: Vec<f64> = bars.iter().map(|b| {
        let tp = (b.high + b.low + b.close) / 3.0;
        cum_pv  += tp * b.volume.max(1.0);
        cum_vol += b.volume.max(1.0);
        cum_pv / cum_vol
    }).collect();
    let vwap  = *vwaps.last().unwrap();
    let close = bars.last().unwrap().close;
    let dist  = ((close - vwap) / close.max(1e-8)).clamp(-0.05, 0.05) / 0.05;

    // VWAP slope over the last 5 bars
    let n = vwaps.len();
    let slope = if n >= 5 {
        let v_now  = vwaps[n - 1];
        let v_prev = vwaps[n - 5];
        ((v_now - v_prev) / v_prev.abs().max(1e-8)).clamp(-0.02, 0.02) / 0.02
    } else { 0.0 };

    (dist, slope)
}

// ─────────────────────────────────────────────
//  OBV
// ─────────────────────────────────────────────

/// On-Balance Volume momentum: rate of change of OBV over last 10 bars,
/// normalised by average OBV magnitude (so it's price-scale independent).
pub fn obv_momentum(bars: &[Bar]) -> f64 {
    if bars.len() < 2 { return 0.0; }
    let obvs: Vec<f64> = {
        let mut v = 0.0_f64;
        let mut out = vec![0.0; bars.len()];
        out[0] = 0.0;
        for i in 1..bars.len() {
            let d = bars[i].close - bars[i-1].close;
            v += if d > 0.0 { bars[i].volume } else if d < 0.0 { -bars[i].volume } else { 0.0 };
            out[i] = v;
        }
        out
    };
    let n = obvs.len();
    let lookback = 10.min(n - 1);
    let obv_now  = obvs[n - 1];
    let obv_prev = obvs[n - 1 - lookback];
    let scale    = obvs.iter().map(|x| x.abs()).fold(0.0_f64, f64::max).max(1.0);
    ((obv_now - obv_prev) / scale).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────
//  Volume ratio
// ─────────────────────────────────────────────

/// Current bar's volume vs 20-bar average. Normalised to [0,1] (cap at 5x).
pub fn volume_ratio(bars: &[Bar], period: usize) -> f64 {
    if bars.is_empty() { return 0.2; }
    let window = &bars[bars.len().saturating_sub(period)..];
    let avg    = window.iter().map(|b| b.volume).sum::<f64>() / window.len().max(1) as f64;
    let cur    = bars.last().unwrap().volume;
    (cur / avg.max(1.0)).clamp(0.0, 5.0) / 5.0
}

// ─────────────────────────────────────────────
//  Candle body ratio
// ─────────────────────────────────────────────

/// |close - open| / (high - low), scaled to [-1, 1] with sign = direction.
/// +1 = strong bullish engulfing; -1 = strong bearish engulfing; 0 = doji.
pub fn candle_body(bar: &Bar) -> f64 {
    let range = (bar.high - bar.low).max(1e-8);
    let body  = (bar.close - bar.open) / range;
    body.clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────
//  Master feature extractor
// ─────────────────────────────────────────────

/// Compute all INDICATOR_NF technical features from a lookback window of bars.
///
/// Call this once per training sample with the full lookback slice.
/// All features are normalised to roughly [-1, 1] or [0, 1].
///
/// # Panics
/// Requires `bars.len() >= 27` (26-period MACD minimum).
pub fn compute_indicators(bars: &[Bar]) -> [f64; INDICATOR_NF] {
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let n = closes.len();

    // ── Trend (EMA) ─────────────────────────────────────────────
    let ema9  = ema(&closes, 9);
    let ema21 = ema(&closes, 21);
    let close = *closes.last().unwrap();

    let ema9_dist  = ((close - ema9)  / close.max(1e-8)).clamp(-0.05, 0.05) / 0.05;
    let ema21_dist = ((close - ema21) / close.max(1e-8)).clamp(-0.05, 0.05) / 0.05;
    let ema9_slp   = ema_slope(&closes, 9) / 0.02;           // normalise slope
    let ema_cross  = if ema9 > ema21 { 1.0 } else { -1.0 };  // binary crossover

    // ── Momentum ────────────────────────────────────────────────
    let rsi14        = rsi(&closes, 14) * 2.0 - 1.0;         // remap [0,1] → [-1,1]
    let (macd_hist, macd_sign) = macd(&closes);
    let macd_h_norm  = macd_hist / 0.01;                      // normalise histogram

    // ── Volatility ──────────────────────────────────────────────
    let (pct_b, bw)  = bollinger(&closes, 20);
    let pct_b_norm   = pct_b * 2.0 - 1.0;                    // [0,1] → [-1,1]
    let bw_norm      = (bw / 0.1).clamp(0.0, 1.0);
    let atr_r        = atr_ratio(bars, 14) * 2.0 - 1.0;      // [0,1] → [-1,1]

    // ── Support / Resistance ─────────────────────────────────────
    let (sup_dist, res_dist, sr_strength) = pivot_sr(bars);
    let sup_norm = sup_dist * 2.0 - 1.0;                      // 0=at support, 1=far above
    let res_norm = 1.0 - res_dist * 2.0;                      // 1=at resistance, -1=far below

    // ── VWAP ────────────────────────────────────────────────────
    let (vwap_dist, vwap_slp) = vwap_features(bars);

    // ── Volume ──────────────────────────────────────────────────
    let obv_mom  = obv_momentum(bars);
    let vol_rat  = volume_ratio(bars, 20) * 2.0 - 1.0;        // [0,1] → [-1,1]

    // ── Candle ──────────────────────────────────────────────────
    let body = candle_body(bars.last().unwrap());

    [
        ema9_dist,    // [0]  EMA(9) distance
        ema21_dist,   // [1]  EMA(21) distance
        ema9_slp,     // [2]  EMA(9) slope
        ema_cross,    // [3]  EMA crossover signal
        rsi14,        // [4]  RSI(14) normalised
        macd_h_norm,  // [5]  MACD histogram
        macd_sign,    // [6]  MACD signal direction
        pct_b_norm,   // [7]  Bollinger %B
        bw_norm,      // [8]  Bollinger bandwidth
        atr_r,        // [9]  ATR ratio
        sup_norm,     // [10] Support distance
        res_norm,     // [11] Resistance distance
        sr_strength,  // [12] SR cluster strength
        vwap_dist,    // [13] VWAP distance
        vwap_slp,     // [14] VWAP slope
        obv_mom,      // [15] OBV momentum
        vol_rat,      // [16] Volume ratio
        body,         // [17] Candle body ratio
    ]
}

// ─────────────────────────────────────────────
//  Convenience: convert from cascade_trainer's StockData
// ─────────────────────────────────────────────

/// Helper to build a `Bar` slice from cascade_trainer's internal StockData.
/// Use this inside `extract()` or `build_samples()` to get indicator features.
///
/// Usage example (in cascade_trainer.rs):
/// ```
/// use indicators::{Bar, compute_indicators};
///
/// fn extract_with_indicators(window: &[StockData]) -> Vec<f64> {
///     let bars: Vec<Bar> = window.iter().map(|d| Bar {
///         open: d.open, high: d.high, low: d.low,
///         close: d.close, volume: d.volume as f64,
///     }).collect();
///     compute_indicators(&bars).to_vec()
/// }
/// ```
pub fn bars_from_cascade(data: &[impl AsCascadeBar]) -> Vec<Bar> {
    data.iter().map(|d| d.to_bar()).collect()
}

pub trait AsCascadeBar {
    fn to_bar(&self) -> Bar;
}
