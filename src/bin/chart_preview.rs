// src/bin/chart_preview.rs
//
// ╔══════════════════════════════════════════════════════════════════╗
// ║  Stop-Entry AI  —  Chart Preview                                ║
// ║                                                                  ║
// ║  Reads weights + CSV data, auto-downloads any missing weeks      ║
// ║  (same week-by-week system as the trainer), then renders a       ║
// ║  self-contained HTML candlestick chart with the SL / TP overlay. ║
// ║                                                                  ║
// ║  USAGE                                                           ║
// ║    cargo run --bin chart_preview --                              ║
// ║      --csv  AAPL_data.csv                                        ║
// ║      -a     60                                                   ║
// ║      --weights stop_ai.json                                      ║
// ║      --start-date 2025-04-16   ← download + scan range start    ║
// ║      [--end-date  2026-04-16]  ← download + scan range end      ║
// ║      [--api-key KEY]           ← or set ALPHAVANTAGE_API_KEY    ║
// ║      [--bar-index N]           ← explicit single bar            ║
// ║      [--chart-out chart.html]  ← default: chart.html            ║
// ║                                                                  ║
// ║  start-date is derived from the CSV's last row automatically.   ║
// ║  Only weeks strictly after that date are fetched.               ║
// ╚══════════════════════════════════════════════════════════════════╝

use stock_tracker::generator::{
    data_file_for, load_existing_timestamps, get_mondays,
    fetch_week, save_bars, parse_csv,
};
use stock_tracker::indicators::{compute_indicators, Bar, INDICATOR_NF};

use chrono::Duration;
use std::{env, fs, process, thread, time as std_time};

// ═══════════════════════════════════════════════════════════════════
//  Must match the constants in stop_ai.rs exactly
// ═══════════════════════════════════════════════════════════════════

const FORWARD:  usize = 10;
const MAX_MULT: f64   = 4.0;
const NIN:      usize = INDICATOR_NF;
const H1:       usize = 256;
const H2:       usize = 128;
const NOUT:     usize = 3;
const RATE_LIMIT_SECS: u64 = 8;

/// Minimum distance of dir_prob from the 0.5 decision boundary before we
/// consider a signal tradeable — mirrors the constant in stop_ai.rs.
/// Override at runtime with --confidence.
const CONFIDENCE_THRESHOLD: f64 = 0.09;

// ═══════════════════════════════════════════════════════════════════
//  Minimal MLP  (load + predict only)
// ═══════════════════════════════════════════════════════════════════

struct Mlp {
    w1: Vec<f64>, b1: Vec<f64>,
    w2: Vec<f64>, b2: Vec<f64>,
    w3: Vec<f64>, b3: Vec<f64>,
}

impl Mlp {
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

    #[inline(always)]
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
    }

    fn predict(&self, x: &[f64]) -> (f64, f64, f64) {
        let h1: Vec<f64> = (0..H1).map(|j| {
            (self.b1[j] + Self::dot(&self.w1[j * NIN..j * NIN + NIN], x)).max(0.0)
        }).collect();
        let h2: Vec<f64> = (0..H2).map(|j| {
            (self.b2[j] + Self::dot(&self.w2[j * H1..j * H1 + H1], &h1)).max(0.0)
        }).collect();
        let raw: Vec<f64> = (0..NOUT).map(|j| {
            self.b3[j] + Self::dot(&self.w3[j * H2..j * H2 + H2], &h2)
        }).collect();
        let sig  = |x: f64| 1.0 / (1.0 + (-x).exp());
        let dir  = sig(raw[0]);
        let sl_m = (0.3 + sig(raw[1]) * (MAX_MULT - 0.3)).clamp(0.3, MAX_MULT);
        let tp_m = (0.5 + sig(raw[2]) * (MAX_MULT - 0.5)).clamp(0.5, MAX_MULT);
        (dir, sl_m, tp_m)
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Bar aggregation + ATR
// ═══════════════════════════════════════════════════════════════════

fn to_5min(raw: &[stock_tracker::generator::StockData]) -> (Vec<Bar>, Vec<chrono::NaiveDate>) {
    let mut bars  = Vec::with_capacity(raw.len() / 5);
    let mut dates = Vec::with_capacity(raw.len() / 5);
    let mut i = 0;
    while i + 4 < raw.len() {
        let w = &raw[i..i + 5];
        bars.push(Bar {
            open:   w[0].open,
            high:   w.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max),
            low:    w.iter().map(|b| b.low ).fold(f64::INFINITY,     f64::min),
            close:  w[4].close,
            volume: w.iter().map(|b| b.volume as f64).sum(),
        });
        dates.push(w[0].ts.date_naive());
        i += 5;
    }
    (bars, dates)
}

fn current_atr(bars: &[Bar], period: usize) -> f64 {
    if bars.len() < 2 { return 0.001; }
    let start = bars.len().saturating_sub(period + 1);
    let trs: Vec<f64> = bars[start..].windows(2).map(|w| {
        (w[1].high - w[1].low)
            .max((w[1].high - w[0].close).abs())
            .max((w[1].low  - w[0].close).abs())
    }).collect();
    if trs.is_empty() { return 0.001; }
    trs.iter().sum::<f64>() / trs.len() as f64
}

// ═══════════════════════════════════════════════════════════════════
//  Trade simulation
// ═══════════════════════════════════════════════════════════════════

struct Trade { pct_gain: f64, candles_held: usize, won: bool }

fn simulate(
    is_long: bool, entry_high: f64, entry_low: f64,
    sl: f64, tp: f64, future: &[Bar],
) -> Option<Trade> {
    let mut entry_px    = None;
    let mut entry_start = 0;
    for (c, bar) in future.iter().enumerate() {
        if  is_long && bar.high >= entry_high { entry_px = Some(entry_high); entry_start = c; break; }
        if !is_long && bar.low  <= entry_low  { entry_px = Some(entry_low);  entry_start = c; break; }
    }
    let px = entry_px?;
    for (c, bar) in future[entry_start + 1..].iter().enumerate() {
        let candles = c + 1;
        if is_long {
            if bar.low  <= sl { return Some(Trade { pct_gain: (sl - px) / px * 100.0, candles_held: candles, won: false }); }
            if bar.high >= tp { return Some(Trade { pct_gain: (tp - px) / px * 100.0, candles_held: candles, won: true  }); }
        } else {
            if bar.high >= sl { return Some(Trade { pct_gain: (px - sl) / px * 100.0, candles_held: candles, won: false }); }
            if bar.low  <= tp { return Some(Trade { pct_gain: (px - tp) / px * 100.0, candles_held: candles, won: true  }); }
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════
//  last_monday_in_csv
//
//  Reads the last ~256 bytes of the CSV to find the most recent
//  timestamp without loading all rows.  Returns the Monday of that
//  week so the call site in main() knows exactly where to start.
// ═══════════════════════════════════════════════════════════════════

fn last_monday_in_csv(path: &str) -> Option<chrono::NaiveDate> {
    use std::io::{Read, Seek, SeekFrom};
    use chrono::Datelike;

    let mut f = std::fs::File::open(path).ok()?;
    let file_len = f.seek(SeekFrom::End(0)).ok()? as i64;
    if file_len == 0 { return None; }

    // Read the last 256 bytes — enough to always contain the final line.
    let read_start = (file_len - 256).max(0) as u64;
    f.seek(SeekFrom::Start(read_start)).ok()?;
    let mut tail = String::new();
    f.read_to_string(&mut tail).ok()?;

    // Last non-empty line.
    let last_line = tail.lines().filter(|l| !l.trim().is_empty()).last()?;
    // First CSV field is the timestamp: "2026-03-14 15:59:00" (with or without quotes).
    let ts_field = last_line.split(',').next()?.trim().trim_matches('"');
    // First 10 chars are always YYYY-MM-DD regardless of time suffix.
    let date_str = ts_field.get(..10)?;
    let date = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok()?;
    // Rewind to the Monday that starts this week.
    let days_since_monday = date.weekday().num_days_from_monday() as i64;
    Some(date - chrono::Duration::days(days_since_monday))
}

// ═══════════════════════════════════════════════════════════════════
//  sync_data
//
//  `start` must already point to the first Monday we actually need
//  (i.e. the Monday *after* the last week in the CSV, derived by
//  last_monday_in_csv at the call site).
//
//  week_already_downloaded is intentionally NOT called here — its
//  timestamp-format check (bare date vs. full datetime string) was
//  causing spurious re-fetches of every historical week.
// ═══════════════════════════════════════════════════════════════════

fn sync_data(symbol: &str, csv_path: &str, start: &str, end: &str, api_keys: &[String]) {
    assert!(!api_keys.is_empty(), "sync_data called with no API keys");

    let start_date = chrono::NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .unwrap_or_else(|_| panic!("--start-date must be YYYY-MM-DD, got '{}'", start));
    let end_date = chrono::NaiveDate::parse_from_str(end, "%Y-%m-%d")
        .unwrap_or_else(|_| panic!("--end-date must be YYYY-MM-DD, got '{}'", end));
    let today    = chrono::Local::now().date_naive();
    let end_date = end_date.min(today);

    let mut existing = load_existing_timestamps(csv_path);
    let mondays      = get_mondays(start_date, end_date);

    // Build a set of date-only prefixes ("YYYY-MM-DD") from whatever is in
    // `existing`.  Works regardless of whether load_existing_timestamps already
    // inserted them: it extracts the first 10 chars of every entry that looks
    // like a full timestamp, so the check below is never fooled by a version
    // of the library that stores only full datetime strings.
    let date_keys: std::collections::HashSet<String> = existing
        .iter()
        .filter(|s| s.len() >= 10 && s.as_bytes()[4] == b'-' && s.as_bytes()[7] == b'-')
        .map(|s| s[..10].to_string())
        .collect();

    // Returns true if any trading day in the Mon–Fri week has a row in the CSV.
    let week_present = |monday: chrono::NaiveDate| -> bool {
        (0..5).any(|d| date_keys.contains(&(monday + Duration::days(d)).to_string()))
    };

    let nkeys = api_keys.len();
    if nkeys > 1 {
        println!("Syncing {} — {} week(s) in range, checking each ({} key(s), round-robin):",
            symbol, mondays.len(), nkeys);
    } else {
        println!("Syncing {} — {} week(s) in range, checking each:", symbol, mondays.len());
    }

    let mut total_new  = 0usize;
    let mut fetch_idx  = 0usize;   // counts only actual API calls — drives round-robin + rate-limit

    for (wi, &monday) in mondays.iter().enumerate() {
        print!("[Week {:>3}/{}]  {}  ", wi + 1, mondays.len(), monday);

        // Skip future weeks.
        if monday > today {
            println!("future, skipping.");
            continue;
        }

        // Skip weeks whose data is already in the CSV — no API call needed.
        if week_present(monday) {
            println!("already downloaded.");
            continue;
        }

        // Genuine missing week — fetch from API.
        let key = &api_keys[fetch_idx % nkeys];
        print!("[key {}]  fetching... ", fetch_idx % nkeys + 1);
        let bars = fetch_week(symbol, monday, key);

        if !bars.is_empty() {
            let added = save_bars(&bars, csv_path, &existing);
            total_new += added;
            // Insert both the full timestamps and their date prefixes so that
            // subsequent iterations of this loop see the newly written rows.
            for b in &bars {
                existing.insert(b[0].clone());
                if b[0].len() >= 10 { existing.insert(b[0][..10].to_string()); }
            }
            println!("{} rows added.", added);
        } else {
            println!("no data returned.");
        }

        // Rate-limit: wait only when the same key slot will be reused on the
        // next fetch AND there are more genuinely missing weeks ahead.
        let more_missing = mondays[wi + 1..].iter()
            .any(|&m| m <= today && !week_present(m));
        let same_key_next = (fetch_idx + 1) % nkeys == 0;
        if more_missing && (nkeys == 1 || same_key_next) {
            println!("  Waiting {}s (rate limit on key {})...", RATE_LIMIT_SECS, fetch_idx % nkeys + 1);
            thread::sleep(std_time::Duration::from_secs(RATE_LIMIT_SECS));
        }

        fetch_idx += 1;
    }

    println!("Sync complete — {} new rows added to {}\n", total_new, csv_path);
}

// ═══════════════════════════════════════════════════════════════════
//  Chart generation
// ═══════════════════════════════════════════════════════════════════

/// Returns `true` if the signal passed the confidence threshold and a chart
/// was written; `false` if the bar was skipped due to low confidence.
fn run_chart(
    bars:                 &[Bar],
    lookback:             usize,
    mlp:                  &Mlp,
    bar_index:            Option<usize>,
    out_path:             &str,
    confidence_threshold: f64,
) -> bool {
    if bars.len() < lookback + FORWARD * 2 + 1 {
        eprintln!(
            "Error: only {} 5-min bars — need at least {} for lookback {} + forward window.",
            bars.len(), lookback + FORWARD * 2 + 1, lookback
        );
        process::exit(1);
    }

    // Valid range: [lookback, len - FORWARD*2).
    // Default = most recent valid bar (just past the last downloaded week).
    let max_i = bars.len().saturating_sub(FORWARD * 2);
    let i: usize = bar_index
        .map(|idx| idx.clamp(lookback, max_i - 1))
        .unwrap_or(max_i - 1);

    let window = &bars[i - lookback..i];
    if window.len() < 27 {
        eprintln!("Error: lookback must be ≥ 27 bars (MACD minimum).");
        process::exit(1);
    }

    // ── Prediction ───────────────────────────────────────────────────────────
    let features           = compute_indicators(window);
    let (dir_prob, sm, tm) = mlp.predict(&features);

    // Skip this bar if the model is not confident enough.
    let raw_confidence = (dir_prob - 0.5).abs();
    if raw_confidence < confidence_threshold {
        return false;
    }

    let is_long            = dir_prob >= 0.5;
    let cur                = &bars[i - 1];
    let atr                = current_atr(window, 14).max(1e-8);

    let (sl, tp) = if is_long {
        (cur.close - sm * atr, cur.close + tm * atr)
    } else {
        (cur.close + sm * atr, cur.close - tm * atr)
    };

    let entry_high = cur.close + 0.25 * atr;
    let entry_low  = cur.close - 0.25 * atr;
    let entry_px   = if is_long { entry_high } else { entry_low };
    let confidence = (dir_prob - 0.5).abs() * 200.0;
    let rr         = (tp - cur.close).abs() / (cur.close - sl).abs().max(1e-8);

    // ── Simulate outcome on the future bars ──────────────────────────────────
    let future_len = (FORWARD * 2).min(bars.len() - i);
    let future     = &bars[i..i + future_len];

    let trade = simulate(is_long, entry_high, entry_low, sl, tp, future);
    let (result_label, result_emoji, pct_gain, candles_held) = match &trade {
        Some(t) if  t.won => ("TP HIT", "✓", t.pct_gain, t.candles_held),
        Some(t)           => ("SL HIT", "✗", t.pct_gain, t.candles_held),
        None              => ("OPEN",   " ", 0.0,         0usize),
    };
    let result_color = match &trade {
        Some(t) if t.won => "#26a69a",
        Some(_)          => "#ef5350",
        None             => "#b0b0b0",
    };
    let result_full = format!("{} {}", result_label, result_emoji);

    // ── Build candle JS ──────────────────────────────────────────────────────
    let display_n   = 80.min(window.len());
    let disp_window = &window[window.len() - display_n..];

    // Fake Unix timestamps (5-min spacing). 2024-01-02 09:30 ET ≈ 1704203400 UTC.
    const BASE_TS: u64 = 1_704_203_400;

    let mut candle_js = String::with_capacity(2048);
    for (k, b) in disp_window.iter().enumerate() {
        let t = BASE_TS + k as u64 * 300;
        candle_js.push_str(&format!(
            "  {{time:{},open:{:.4},high:{:.4},low:{:.4},close:{:.4}}},\n",
            t, b.open, b.high, b.low, b.close
        ));
    }
    for (k, b) in future.iter().enumerate() {
        let t = BASE_TS + (display_n + k) as u64 * 300;
        candle_js.push_str(&format!(
            "  {{time:{},open:{:.4},high:{:.4},low:{:.4},close:{:.4}}},\n",
            t, b.open, b.high, b.low, b.close
        ));
    }

    let signal_ts   = BASE_TS + (display_n - 1) as u64 * 300;
    let entry_ts    = BASE_TS + display_n as u64 * 300;
    let band_end_ts = BASE_TS + (display_n + future_len.saturating_sub(1)) as u64 * 300;

    let dir_label    = if is_long { "LONG  ▲" } else { "SHORT ▼" };
    let dir_color    = if is_long { "#26a69a"  } else { "#ef5350"  };
    let marker_pos   = if is_long { "'belowBar'" } else { "'aboveBar'" };
    let marker_shape = if is_long { "arrowUp" } else { "arrowDown" };

    // ── Self-contained HTML ──────────────────────────────────────────────────
    let html = format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stop-Entry AI — Chart Preview</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{
    background:#131722;color:#d1d4dc;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    display:flex;flex-direction:column;height:100vh;
  }}
  #header{{
    display:flex;align-items:center;gap:16px;padding:10px 18px;
    background:#1e2130;border-bottom:1px solid #2a2e39;flex-shrink:0;
  }}
  #header h1{{font-size:14px;font-weight:600;color:#fff;letter-spacing:.04em}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:4px;font-size:12px;font-weight:700;letter-spacing:.06em}}
  #stats{{
    display:flex;gap:20px;padding:6px 18px;
    background:#1a1e2e;border-bottom:1px solid #2a2e39;
    font-size:11px;flex-shrink:0;align-items:center;
  }}
  .stat-lbl{{color:#787b86;margin-right:4px}}
  .stat-val{{color:#d1d4dc;font-weight:600}}
  .divider{{color:#2a2e39}}
  #chart-wrap{{flex:1;position:relative;overflow:hidden}}
  #chart{{width:100%;height:100%}}
  #info{{
    position:absolute;top:10px;left:10px;z-index:10;
    background:rgba(19,23,34,.88);border:1px solid #2a2e39;
    border-radius:6px;padding:10px 14px;font-size:12px;
    line-height:1.85;pointer-events:none;min-width:190px;
  }}
  .il{{color:#787b86;margin-right:6px}}
  .iv{{color:#d1d4dc;font-weight:600}}
  .isep{{border-top:1px solid #2a2e39;margin:5px 0}}
</style>
</head>
<body>
<div id="header">
  <h1>Stop-Entry AI — Chart Preview</h1>
  <span class="badge" style="background:{dir_color};color:#fff">{dir_label}</span>
  <span class="badge" style="background:{result_color};color:#fff">{result_full}</span>
</div>
<div id="stats">
  <span><span class="stat-lbl">Confidence</span><span class="stat-val">{confidence:.1}%</span></span>
  <span class="divider">|</span>
  <span><span class="stat-lbl">Entry</span><span class="stat-val">{entry_px:.4}</span></span>
  <span class="divider">|</span>
  <span><span class="stat-lbl">SL</span><span class="stat-val" style="color:#ef5350">{sl:.4}</span></span>
  <span class="divider">|</span>
  <span><span class="stat-lbl">TP</span><span class="stat-val" style="color:#26a69a">{tp:.4}</span></span>
  <span class="divider">|</span>
  <span><span class="stat-lbl">R:R</span><span class="stat-val">1 : {rr:.2}</span></span>
  <span class="divider">|</span>
  <span><span class="stat-lbl">P&amp;L</span><span class="stat-val" style="color:{result_color}">{pct_gain:+.3}%</span></span>
  <span class="divider">|</span>
  <span><span class="stat-lbl">Held</span><span class="stat-val">{candles_held} candles ({held_min} min)</span></span>
  <span style="margin-left:auto;color:#787b86">ATR {atr:.4} &nbsp;·&nbsp; SL {sm:.2}× &nbsp;·&nbsp; TP {tm:.2}×</span>
</div>
<div id="chart-wrap">
  <div id="chart"></div>
  <div id="info">
    <div><span class="il">Close at signal</span><span class="iv">{close:.4}</span></div>
    <div><span class="il">Entry trigger</span><span class="iv">{entry_px:.4}</span></div>
    <div class="isep"></div>
    <div><span class="il">Stop Loss</span><span class="iv" style="color:#ef5350">{sl:.4}</span><span style="color:#787b86;font-size:11px"> ({sm:.2}×ATR)</span></div>
    <div><span class="il">Take Profit</span><span class="iv" style="color:#26a69a">{tp:.4}</span><span style="color:#787b86;font-size:11px"> ({tm:.2}×ATR)</span></div>
    <div class="isep"></div>
    <div><span class="il">Direction</span><span class="iv" style="color:{dir_color}">{dir_label}</span></div>
    <div><span class="il">Result</span><span class="iv" style="color:{result_color}">{result_full}</span></div>
  </div>
</div>
<script>
(function(){{
  const chart = LightweightCharts.createChart(document.getElementById('chart'),{{
    layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc'}},
    grid:{{vertLines:{{color:'#1e2130'}},horzLines:{{color:'#1e2130'}}}},
    crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}},
    rightPriceScale:{{borderColor:'#2a2e39'}},
    timeScale:{{borderColor:'#2a2e39',timeVisible:true,secondsVisible:false}},
    width:document.getElementById('chart').offsetWidth,
    height:document.getElementById('chart').offsetHeight,
  }});
  const SIGNAL_TS={signal_ts}, ENTRY_TS={entry_ts}, BAND_END_TS={band_end_ts};
  const all=[
{candle_js}
  ];
  const hist=all.filter(c=>c.time<=SIGNAL_TS);
  const fut =all.filter(c=>c.time> SIGNAL_TS);

  const histS=chart.addCandlestickSeries({{
    upColor:'#26a69a',downColor:'#ef5350',
    borderUpColor:'#26a69a',borderDownColor:'#ef5350',
    wickUpColor:'#26a69a',wickDownColor:'#ef5350',
  }});
  histS.setData(hist);
  histS.setMarkers([{{time:SIGNAL_TS,position:{marker_pos},color:'{dir_color}',shape:'{marker_shape}',text:'Signal',size:1.5}}]);

  const futS=chart.addCandlestickSeries({{
    upColor:'#1a4d42',downColor:'#5c1f1f',
    borderUpColor:'#26a69a',borderDownColor:'#ef5350',
    wickUpColor:'#26a69a',wickDownColor:'#ef5350',
  }});
  futS.setData(fut);

  histS.createPriceLine({{price:{sl:.4},color:'#ef5350',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Solid,axisLabelVisible:true,title:'SL'}});
  histS.createPriceLine({{price:{tp:.4},color:'#26a69a',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Solid,axisLabelVisible:true,title:'TP'}});
  histS.createPriceLine({{price:{entry_px:.4},color:'#4e9af1',lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:'Entry'}});

  const zoneTop=Math.max({sl:.4},{tp:.4})*1.005;
  const zoneS=chart.addAreaSeries({{
    topColor:'rgba(78,154,241,0.07)',bottomColor:'rgba(78,154,241,0.07)',
    lineColor:'rgba(78,154,241,0.18)',lineWidth:1,
    lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false,
  }});
  const zd=[];
  for(let t=ENTRY_TS;t<=BAND_END_TS;t+=300) zd.push({{time:t,value:zoneTop}});
  if(zd.length) zoneS.setData(zd);

  chart.timeScale().fitContent();
  new ResizeObserver(()=>{{
    const w=document.getElementById('chart-wrap');
    chart.resize(w.clientWidth,w.clientHeight);
  }}).observe(document.getElementById('chart-wrap'));
}})();
</script>
</body>
</html>
"##,
        dir_label    = dir_label,   dir_color    = dir_color,
        result_full  = result_full, result_color = result_color,
        confidence   = confidence,  entry_px     = entry_px,
        sl           = sl,          tp           = tp,
        rr           = rr,          pct_gain     = pct_gain,
        candles_held = candles_held, held_min    = candles_held * 5,
        atr          = atr,         sm           = sm,   tm = tm,
        close        = cur.close,   candle_js    = candle_js,
        signal_ts    = signal_ts,   entry_ts     = entry_ts,
        band_end_ts  = band_end_ts, marker_pos   = marker_pos,
        marker_shape = marker_shape,
    );

    fs::write(out_path, &html)
        .unwrap_or_else(|e| panic!("Cannot write '{}': {}", out_path, e));

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Chart written                                           ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  File       : {:<43} ║", out_path);
    println!("║  Direction  : {:<10}  ({:.1}% confident)            ║",
        if is_long { "LONG  ▲" } else { "SHORT ▼" }, confidence);
    println!("║  Entry      : {:<12.4}                            ║", entry_px);
    println!("║  Stop Loss  : {:<12.4}  ({:.2}× ATR)              ║", sl, sm);
    println!("║  Take Profit: {:<12.4}  ({:.2}× ATR)              ║", tp, tm);
    println!("║  Result     : {:<10}  ({:+.3}% in {} candles)    ║",
        result_full, pct_gain, candles_held);
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  Open {} in any browser.", out_path);
    println!();

    true
}

// ═══════════════════════════════════════════════════════════════════
//  Entry point
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut csv_path   = String::new();
    let mut symbol     = String::new();
    let mut weights    = String::from("stop_ai_weights.json");
    let mut lookback   = 60usize;
    // api_key: --api-key flag takes priority, then ALPHAVANTAGE_API_KEY env var.
    let mut api_keys: Vec<String> = {
        let k = std::env::var("ALPHAVANTAGE_API_KEY").unwrap_or_default();
        if k.is_empty() { vec![
            String::from("8a72215b906449f29ba9e6063f5ccecb"),
            String::from("d4394d9216d149ce9b72adda686ab9d4"),
            String::from("0cd010b487004db0afa16b8a24dbf61e"),
            String::from("37b77cd8dad543b18b28f9dd70167669"),
        ]} else { vec![k] }
    };
    let mut start_date: Option<String> = None;
    let mut samples: usize = 1;                  // --samples N
    let mut end_date   = chrono::Local::now().date_naive().to_string();
    let mut bar_index: Option<usize> = None;
    let mut chart_out  = String::from("chart.html");
    let mut confidence_threshold: f64 = CONFIDENCE_THRESHOLD;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--csv"          => { i += 1; csv_path     = args[i].clone(); }
            "--symbol"       => { i += 1; symbol       = args[i].to_uppercase(); }
            "--weights"      => { i += 1; weights      = args[i].clone(); }
            "-a"             => { i += 1; lookback     = args[i].parse().expect("-a: integer"); }
            "--api-key"      => { i += 1;
                                  // First --api-key replaces the env-var default;
                                  // subsequent ones append.
                                  if api_keys.is_empty() { api_keys.push(args[i].clone()); }
                                  else { api_keys[0] = args[i].clone(); } }
            "--api-key-2"    => { i += 1; if api_keys.len() < 2 { api_keys.push(args[i].clone()) } else { api_keys[1] = args[i].clone(); } }
            "--api-key-3"    => { i += 1; if api_keys.len() < 3 { api_keys.push(args[i].clone()) } else { api_keys[2] = args[i].clone(); } }
            "--api-key-4"    => { i += 1; if api_keys.len() < 4 { api_keys.push(args[i].clone()) } else { api_keys[3] = args[i].clone(); } }
            "--api-key-5"    => { i += 1; if api_keys.len() < 5 { api_keys.push(args[i].clone()) } else { api_keys[4] = args[i].clone(); } }
            "--api-key-6"    => { i += 1; if api_keys.len() < 6 { api_keys.push(args[i].clone()) } else { api_keys[5] = args[i].clone(); } }
            "--start-date"   => { i += 1; start_date   = Some(args[i].clone()); }
            "--end-date"     => { i += 1; end_date     = args[i].clone(); }
            "--bar-index"    => { i += 1; bar_index    = args[i].parse().ok(); }
            "--samples"      => { i += 1; samples      = args[i].parse().expect("--samples: integer"); }
            "--chart-out"    => { i += 1; chart_out    = args[i].clone(); }
            "--confidence"   => { i += 1; confidence_threshold = args[i].parse().expect("--confidence: float [0, 0.5)"); }
            other            => eprintln!("Unknown flag '{}' — ignored", other),
        }
        i += 1;
    }

    // Derive csv_path from symbol if not given directly.
    if csv_path.is_empty() && !symbol.is_empty() {
        csv_path = data_file_for(&symbol);
    }
    if csv_path.is_empty() {
        eprintln!("Usage: chart_preview --csv <file>  (or --symbol AAPL)");
        eprintln!("                     -a <lookback> --weights <file>");
        eprintln!("                     [--api-key KEY]  [--api-key-2 K2] [--api-key-3 K3] [--api-key-4 K4]");
        eprintln!("                     --start-date YYYY-MM-DD   download + scan range start");
        eprintln!("                     [--end-date  YYYY-MM-DD]  download + scan range end (default: today)");
        eprintln!("                     [--samples N]             cap output at N charts (default: unlimited)");
        eprintln!("                     [--bar-index N]           explicit single bar (overrides range)");
        eprintln!("                     [--confidence F]          min |dir_prob-0.5| to enter (default {:.2})", CONFIDENCE_THRESHOLD);
        eprintln!("                     [--chart-out  file.html]  default: chart.html  (chart_N.html for multiple)");
        process::exit(1);
    }
    if lookback < 27 {
        eprintln!("Error: -a must be ≥ 27 (MACD minimum).");
        process::exit(1);
    }

    // Derive symbol from filename for the downloader if not passed explicitly.
    if symbol.is_empty() {
        symbol = csv_path
            .trim_end_matches(".csv")
            .trim_end_matches("_data")
            .to_uppercase();
    }

    // ── Sync any missing weeks before charting ───────────────────────────────
    if !api_keys.is_empty() {
        let effective_start = start_date.clone().unwrap_or_else(|| {
            eprintln!("Error: --start-date is required when API keys are present.");
            eprintln!("  e.g. --start-date 2025-04-16");
            process::exit(1);
        });
        println!("Checking for missing data ({} → {}) ...", effective_start, end_date);
        sync_data(&symbol, &csv_path, &effective_start, &end_date, &api_keys);
    } else {
        println!("No --api-key provided — using existing local data only.");
        println!("  Pass --api-key KEY (and optionally --api-key-2/3/4) to auto-download.");
        println!();
    }

    // ── Load and aggregate ───────────────────────────────────────────────────
    println!("Loading '{}' ...", csv_path);
    let raw  = parse_csv(&csv_path);
    let (bars, bar_dates) = to_5min(&raw);
    println!("  {} 1-min bars → {} 5-min bars", raw.len(), bars.len());
    println!();

    println!("Loading weights '{}' ...", weights);
    let mlp = Mlp::load(&weights);
    println!("  Weights loaded.");
    println!();

    // ── Build candidate bar list for random sampling ─────────────────────────
    // Done once; each sample iteration draws from it independently.
    let max_i = bars.len().saturating_sub(FORWARD * 2);

    let candidates: Option<Vec<usize>> = if bar_index.is_some() {
        None  // explicit index — no candidate list needed
    } else if start_date.is_some() {
        let parse_date = |s: &str| {
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .unwrap_or_else(|_| panic!("--start-date/--end-date must be YYYY-MM-DD, got '{}'", s))
        };
        let rs = start_date.as_deref().map(parse_date)
            .unwrap_or(chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
        let re = parse_date(&end_date);

        let cands: Vec<usize> = bar_dates.iter()
            .enumerate()
            .filter(|&(idx, d)| *d >= rs && *d <= re && idx >= lookback && idx < max_i)
            .map(|(idx, _)| idx)
            .collect();

        if cands.is_empty() {
            eprintln!("Warning: no bars found in range {} → {}, using most recent.", rs, re);
            None
        } else {
            println!("{} candidate bars in range {} → {}", cands.len(), rs, re);
            Some(cands)
        }
    } else {
        None
    };

    // ── Sequential scan: walk every candidate bar, emit only tradeable signals ─
    //
    // Instead of picking random bars, we step through the candidate range one
    // bar at a time (starting from the earliest), run the model, and only write
    // a chart when the signal's confidence exceeds the threshold.  --samples N
    // caps the total number of charts produced (default: unlimited).
    //
    // When --bar-index is given we run exactly that one bar (existing behaviour).

    if let Some(idx) = bar_index {
        // Explicit index — single chart, no confidence gate bypass (gate still
        // applies so the output is consistent with scan mode).
        let out_path = chart_out.clone();
        let taken = run_chart(&bars, lookback, &mlp, Some(idx), &out_path, confidence_threshold);
        if !taken {
            let conf_pct = confidence_threshold * 200.0;
            println!("Bar {} skipped — confidence below {:.0}% threshold.", idx, conf_pct);
        }
    } else {
        // Build the ordered list of bars to scan.
        let scan_indices: Vec<usize> = if let Some(ref cands) = candidates {
            // candidates is already filtered to the date range; sort ascending.
            let mut sorted = cands.clone();
            sorted.sort_unstable();
            sorted
        } else {
            // No range specified — scan every valid bar in the whole dataset.
            (lookback..max_i).collect()
        };

        let conf_pct   = confidence_threshold * 200.0;
        let cap        = if samples == 0 { usize::MAX } else { samples };
        let mut chart_n    = 0usize;
        let mut skipped    = 0usize;

        println!(
            "Scanning {} bars for signals with confidence ≥ {:.0}% (cap: {}) …",
            scan_indices.len(),
            conf_pct,
            if cap == usize::MAX { "unlimited".to_string() } else { cap.to_string() }
        );
        println!();

        for idx in scan_indices {
            if chart_n >= cap { break; }

            chart_n += 1;
            let out_path = {
                let stem = chart_out.trim_end_matches(".html");
                format!("{}_{}.html", stem, chart_n)
            };

            let taken = run_chart(&bars, lookback, &mlp, Some(idx), &out_path, confidence_threshold);
            if taken {
                // chart_n already incremented; message printed inside run_chart.
            } else {
                // Undo the counter increment — this bar didn't produce a chart.
                chart_n -= 1;
                skipped += 1;
            }
        }

        println!();
        println!("Scan complete — {} chart(s) written, {} bar(s) skipped (low confidence).", chart_n, skipped);
    }
}
