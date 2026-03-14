// generator.rs
//
// Shared data-fetching library used by cascade_trainer and scan_predict.
//
// Exposes:
//   - data_file_for       — canonical CSV filename for a symbol
//   - maybe_download      — week-by-week fetch for a date range (trainer path)
//   - ensure_data_for_date — fill in a specific missing date (predict --at path)
//   - fetch_or_load       — smart live loader: local CSV → API top-up (predict live path)
//   - parse_csv           — CSV → Vec<StockData>
//   - StockData           — shared bar struct used by both callers
//
// Can still be run as a standalone binary:
//   cargo run --bin generator -- AAPL 2024-01-01 2026-03-09 YOUR_API_KEY

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::thread;
use std::time;

// ─────────────────────────────────────────────
//  Config
// ─────────────────────────────────────────────

const TWELVE_DATA_URL: &str = "https://api.twelvedata.com/time_series";
const RATE_LIMIT_SECS: u64 = 8;
const POINTS_PER_CALL: usize = 1950;

// ─────────────────────────────────────────────
//  Shared bar struct
// ─────────────────────────────────────────────

/// Parsed OHLCV bar. Shared between cascade_trainer and scan_predict.
#[derive(Clone, Debug)]
pub struct StockData {
    pub ts:     DateTime<Utc>,
    pub open:   f64,
    pub high:   f64,
    pub low:    f64,
    pub close:  f64,
    pub volume: u64,
}

// ─────────────────────────────────────────────
//  File helpers
// ─────────────────────────────────────────────

/// Returns the canonical CSV filename for a symbol. e.g. "AAPL" -> "AAPL_data.csv"
pub fn data_file_for(symbol: &str) -> String {
    format!("{}_data.csv", symbol.to_uppercase())
}

/// Reads an existing CSV and returns all timestamps already saved.
pub fn load_existing_timestamps(filepath: &str) -> HashSet<String> {
    let mut timestamps = HashSet::new();
    if !Path::new(filepath).exists() { return timestamps; }
    let file = match File::open(filepath) { Ok(f) => f, Err(_) => return timestamps };
    for (i, line) in BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; }
        if let Ok(line) = line {
            if let Some(ts) = line.split(',').next() {
                let ts = ts.trim();
                timestamps.insert(ts.to_string());
                // Also insert the date-only prefix so week_already_downloaded
                // can check with a cheap HashSet lookup instead of a linear scan.
                if ts.len() >= 10 {
                    timestamps.insert(ts[..10].to_string());
                }
            }
        }
    }
    println!("  Found {} existing rows in {}", timestamps.len() / 2, filepath);
    timestamps
}

/// Returns true if the given week is already in the file.
/// Checks Mon–Fri date prefixes — avoids false misses on holidays or non-standard open times.
pub fn week_already_downloaded(monday: NaiveDate, existing: &HashSet<String>) -> bool {
    for day_offset in 0..5 {
        let day = monday + Duration::days(day_offset);
        if existing.contains(&day.to_string()) {
            return true;
        }
    }
    false
}
/// Returns all Mondays between start and end dates inclusive.
pub fn get_mondays(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut mondays = Vec::new();
    let mut current = start;
    while current.weekday().num_days_from_monday() != 0 { current += Duration::days(1); }
    while current <= end { mondays.push(current); current += Duration::weeks(1); }
    mondays
}

// ─────────────────────────────────────────────
//  API call
// ─────────────────────────────────────────────

/// Fetches one week of 1-minute bars from Twelve Data.
/// Returns rows as [timestamp, symbol, open, high, low, close, volume] string arrays,
/// sorted oldest-first. Returns empty vec on any error.
pub fn fetch_week(symbol: &str, monday: NaiveDate, api_key: &str) -> Vec<[String; 7]> {
    let friday    = monday + Duration::days(4);
    let start_str = format!("{} 00:00:00", monday).replace(' ', "%20");
    let end_str   = format!("{} 23:59:59", friday).replace(' ', "%20");
    let url = format!(
        "{url}?symbol={sym}&interval=1min&start_date={s}&end_date={e}&outputsize={n}&apikey={k}&format=JSON&timezone=UTC",
        url = TWELVE_DATA_URL, sym = symbol, s = start_str, e = end_str,
        n = POINTS_PER_CALL, k = api_key
    );
    let resp = match reqwest::blocking::get(&url) {
        Ok(r) => r,
        Err(e) => { eprintln!("  Request failed: {}", e); return vec![]; }
    };
    let json: serde_json::Value = match resp.json() {
        Ok(j) => j,
        Err(e) => { eprintln!("  JSON parse error: {}", e); return vec![]; }
    };
    if json["status"] == "error" {
        eprintln!("  API error: {}", json["message"].as_str().unwrap_or("?")); return vec![];
    }
    let values = match json["values"].as_array() {
        Some(v) => v,
        None => { println!("  No data (holiday week?)"); return vec![]; }
    };
    let mut bars: Vec<[String; 7]> = values.iter().map(|v| {
        let s = |k: &str| v[k].as_str().unwrap_or("0").to_string();
        [s("datetime"), symbol.to_uppercase(), s("open"), s("high"), s("low"), s("close"), s("volume")]
    }).collect();
    bars.sort_by(|a, b| a[0].cmp(&b[0]));
    bars
}

// ─────────────────────────────────────────────
//  CSV helpers
// ─────────────────────────────────────────────

/// Merges new bars into the CSV, skipping any timestamps already present.
/// Re-sorts and rewrites the whole file. Returns number of new rows added.
pub fn save_bars(new_bars: &[[String; 7]], filepath: &str, existing: &HashSet<String>) -> usize {
    let filtered: Vec<&[String; 7]> = new_bars.iter().filter(|b| !existing.contains(&b[0])).collect();
    if filtered.is_empty() { println!("  All rows already present."); return 0; }
    let mut all: Vec<Vec<String>> = Vec::new();
    if Path::new(filepath).exists() {
        let f = File::open(filepath).expect("open");
        for (i, line) in BufReader::new(f).lines().enumerate() {
            if i == 0 { continue; }
            if let Ok(l) = line {
                let cols: Vec<String> = l.split(',').map(|s| s.trim().to_string()).collect();
                if cols.len() >= 7 { all.push(cols); }
            }
        }
    }
    for b in &filtered { all.push(b.to_vec()); }
    all.sort_by(|a, b| a[0].cmp(&b[0]));
    let mut f = File::create(filepath).expect("create");
    writeln!(f, "Timestamp,Symbol,Open,High,Low,Close,Volume").unwrap();
    for row in &all { writeln!(f, "{}", row.join(",")).unwrap(); }
    println!("  Saved {} new rows  ({} total)", filtered.len(), all.len());
    filtered.len()
}

/// Parse a CSV file into a Vec<StockData>.
pub fn parse_csv(path: &str) -> Vec<StockData> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open '{}': {}", path, e));
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; }
        let l = line.unwrap();
        let c: Vec<&str> = l.split(',').collect();
        if c.len() < 7 { continue; }
        let ts = chrono::DateTime::parse_from_str(&format!("{} +0000", c[0]), "%Y-%m-%d %H:%M:%S %z")
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        rows.push(StockData {
            ts,
            open:   c[2].trim().parse().unwrap_or(0.0),
            high:   c[3].trim().parse().unwrap_or(0.0),
            low:    c[4].trim().parse().unwrap_or(0.0),
            close:  c[5].trim().parse().unwrap_or(0.0),
            volume: c[6].trim().parse().unwrap_or(0),
        });
    }
    rows
}

/// Write a Vec<StockData> back to a CSV file (full rewrite, sorted).
pub fn save_stock_data_to_csv(bars: &[StockData], path: &str, symbol: &str) {
    let mut file = File::create(path)
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

// ─────────────────────────────────────────────
//  maybe_download  (used by cascade_trainer)
//
//  Downloads a full date range week-by-week, skipping weeks already in the
//  CSV. Returns the path of the (now populated) CSV.
// ─────────────────────────────────────────────

pub fn maybe_download(symbol: &str, start: &str, end: &str, api_key: &str) -> String {
    let filepath   = data_file_for(symbol);
    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .expect("--start-date must be YYYY-MM-DD");
    let end_date   = NaiveDate::parse_from_str(end, "%Y-%m-%d")
        .expect("--end-date must be YYYY-MM-DD");
    let today      = chrono::Local::now().date_naive();
    let mut existing = load_existing_timestamps(&filepath);
    let mondays      = get_mondays(start_date, end_date);

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
            thread::sleep(time::Duration::from_secs(RATE_LIMIT_SECS));
        }
    }
    println!("Download complete: {} API calls, {} new rows, file: {}\n", calls, total_new, filepath);
    filepath
}

// ─────────────────────────────────────────────
//  ensure_data_for_date  (used by scan_predict --at path)
//
//  Checks whether a specific date is present in the CSV. If not, fetches
//  every missing week up to that date and appends them.
// ─────────────────────────────────────────────

pub fn ensure_data_for_date(symbol: &str, csv_path: &str, need_date: NaiveDate, api_key: &str) {
    let mut existing_ts: HashSet<String> = HashSet::new();
    let mut last_date: Option<NaiveDate> = None;

    if Path::new(csv_path).exists() {
        let file = File::open(csv_path).expect("Cannot open CSV");
        for (i, line) in BufReader::new(file).lines().enumerate() {
            if i == 0 { continue; }
            if let Ok(l) = line {
                if let Some(ts) = l.split(',').next() {
                    let ts = ts.trim();
                    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S") {
                        let d = dt.date();
                        last_date = Some(last_date.map_or(d, |prev: NaiveDate| prev.max(d)));
                        existing_ts.insert(d.to_string()); // date prefix for O(1) week check
                    }
                    existing_ts.insert(ts.to_string());
                }
            }
        }
    }

    if existing_ts.contains(&need_date.to_string()) { return; }

    if api_key.is_empty() {
        eprintln!("Error: no data for {} in '{}' and no --api-key provided.", need_date, csv_path);
        eprintln!("  Pass --api-key so scan_predict can auto-download the missing data.");
        std::process::exit(1);
    }

    let need_monday = {
        let days_from_mon = need_date.weekday().num_days_from_monday();
        need_date - Duration::days(days_from_mon as i64)
    };

    let start_monday = if let Some(ld) = last_date {
        if ld >= need_date { return; }
        let next_day  = ld + Duration::days(1);
        let days_back = next_day.weekday().num_days_from_monday() as i64;
        next_day - Duration::days(days_back)
    } else {
        need_monday
    };

    let mut mondays: Vec<NaiveDate> = Vec::new();
    let mut cur = start_monday;
    while cur <= need_monday { mondays.push(cur); cur += Duration::weeks(1); }

    if mondays.is_empty() { return; }

    println!("  Auto-downloading {} missing week(s) for {} ...", mondays.len(), symbol);

    for (i, &monday) in mondays.iter().enumerate() {
        println!("  Fetching week of {} ...", monday);
        let new_raw = fetch_week(symbol, monday, api_key);
        if !new_raw.is_empty() {
            save_bars(&new_raw, csv_path, &existing_ts);
            for b in &new_raw { existing_ts.insert(b[0].clone()); }
        }
        if i + 1 < mondays.len() {
            println!("  Waiting {}s (rate limit)...", RATE_LIMIT_SECS);
            thread::sleep(time::Duration::from_secs(RATE_LIMIT_SECS));
        }
    }
}

// ─────────────────────────────────────────────
//  fetch_or_load  (used by scan_predict live path)
//
//  Smart loader: tries the local CSV first; if stale or absent, tops up
//  from the Twelve Data API and rewrites the file.
// ─────────────────────────────────────────────

pub fn fetch_or_load(symbol: &str, n: usize, api_key: &str) -> Vec<StockData> {
    let csv_path = data_file_for(symbol);

    let mut existing: Vec<StockData> = if Path::new(&csv_path).exists() {
        println!("  Found local database '{}'", csv_path);
        let bars = parse_csv(&csv_path);
        println!("  Loaded {} existing bars", bars.len());
        bars
    } else {
        println!("  No local database found for {} — will download from API", symbol);
        vec![]
    };

    let now_utc = Utc::now();
    let needs_fetch = if let Some(last) = existing.last() {
        let age = (now_utc - last.ts).num_seconds();
        if age <= 120 {
            println!("  Data is fresh (last bar {} — {}s ago). Skipping API call.",
                last.ts.format("%Y-%m-%d %H:%M:%S UTC"), age);
            false
        } else {
            println!("  Last bar: {}  ({}s ago — stale, fetching updates)",
                last.ts.format("%Y-%m-%d %H:%M:%S UTC"), age);
            true
        }
    } else {
        true
    };

    if needs_fetch {
        if api_key.is_empty() {
            if existing.is_empty() {
                panic!("No local data and no --api-key provided. Cannot fetch live data.");
            }
            println!("  No --api-key provided; using stale local data as-is.");
        } else {
            let new_bars = fetch_live_raw(symbol, n, api_key);
            if !new_bars.is_empty() {
                let existing_ts: HashSet<String> = existing.iter()
                    .map(|b| b.ts.format("%Y-%m-%d %H:%M:%S").to_string())
                    .collect();
                let mut added = 0usize;
                for bar in new_bars {
                    let ts_key = bar.ts.format("%Y-%m-%d %H:%M:%S").to_string();
                    if !existing_ts.contains(&ts_key) { existing.push(bar); added += 1; }
                }
                existing.sort_by_key(|b| b.ts);
                println!("  Merged {} new bars into local database ({} total)", added, existing.len());
                save_stock_data_to_csv(&existing, &csv_path, symbol);
            }
        }
    }

    if existing.len() < n {
        eprintln!("  ⚠  Only {} bars available (need {}). Accuracy may be reduced.", existing.len(), n);
    }
    existing
}

/// Raw API fetch — returns bars sorted oldest-first. Does not touch the filesystem.
pub fn fetch_live_raw(symbol: &str, n: usize, api_key: &str) -> Vec<StockData> {
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
//  Standalone binary entry point
// ─────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("Usage:   {} <SYMBOL> <START_DATE> <END_DATE> <API_KEY>", args[0]);
        eprintln!("Example: {} AAPL 2024-01-01 2026-03-09 your_key_here", args[0]);
        eprintln!();
        eprintln!("Get a free API key at: https://twelvedata.com");
        std::process::exit(1);
    }
    let symbol  = args[1].to_uppercase();
    let start   = &args[2];
    let end     = &args[3];
    let api_key = &args[4];
    let filepath = maybe_download(&symbol, start, end, api_key);
    println!();
    println!("{}", "=".repeat(45));
    println!("Done.  Output file: {}", filepath);
    println!();
    println!("To train:");
    println!("  cargo run --bin cascade_trainer -- {} --save-weights {}_weights", filepath, symbol);
}
