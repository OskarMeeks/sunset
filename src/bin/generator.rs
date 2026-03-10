// fetch_stock_data.rs
//
// Downloads real minute-by-minute stock data from Twelve Data and saves it
// to a per-ticker CSV file in the exact format stock_trainer expects.
//
// Features:
//   - One API call per week of data (~1950 rows per call)
//   - Checks existing file before each call — skips weeks already downloaded
//   - Each ticker gets its own file: AAPL_data.csv, TSLA_data.csv etc.
//   - Appends new data and re-sorts the file oldest-first
//
// Add to Cargo.toml:
//   [dependencies]
//   chrono    = { version = "0.4", features = ["serde"] }
//   reqwest   = { version = "0.11", features = ["blocking", "json"] }
//   serde     = { version = "1", features = ["derive"] }
//   serde_json = "1"
//
// Usage:
//   cargo run --bin fetch_stock_data -- AAPL 2024-01-01 2026-03-09 YOUR_API_KEY
//   cargo run --bin fetch_stock_data -- TSLA 2024-01-01 2026-03-09 YOUR_API_KEY
//
// Get a free API key at: https://twelvedata.com

use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime};
use std::collections::HashSet;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::thread;
use std::time;

// ─────────────────────────────────────────────
//  Config
// ─────────────────────────────────────────────

const TWELVE_DATA_URL: &str = "https://api.twelvedata.com/time_series";

// 8 seconds between calls to stay safely under the 8 calls/minute free tier limit
const RATE_LIMIT_SECS: u64 = 8;

// ~1950 rows = 5 trading days x ~390 minutes per day = one full week
const POINTS_PER_CALL: usize = 1950;

// ─────────────────────────────────────────────
//  Data row
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Bar {
    timestamp: String, // "YYYY-MM-DD HH:MM:SS"
    symbol:    String,
    open:      String,
    high:      String,
    low:       String,
    close:     String,
    volume:    String,
}

// ─────────────────────────────────────────────
//  File helpers
// ─────────────────────────────────────────────

/// Returns the output filename for a ticker. e.g. "AAPL" -> "AAPL_data.csv"
fn data_file_for(symbol: &str) -> String {
    format!("{}_data.csv", symbol)
}

/// Reads an existing CSV and returns all timestamps already saved.
/// Used to decide whether a week needs downloading.
fn load_existing_timestamps(filepath: &str) -> HashSet<String> {
    let mut timestamps = HashSet::new();

    if !Path::new(filepath).exists() {
        return timestamps;
    }

    let file = match File::open(filepath) {
        Ok(f) => f,
        Err(_) => return timestamps,
    };

    for (i, line) in BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; } // skip header
        if let Ok(line) = line {
            // Timestamp is the first column
            if let Some(ts) = line.split(',').next() {
                timestamps.insert(ts.trim().to_string());
            }
        }
    }

    println!("  Found {} existing rows in {}", timestamps.len(), filepath);
    timestamps
}

/// Returns true if the given Monday's data is already in the file.
/// Checks the first 10 minutes of the trading day (13:30–13:39 UTC = 09:30 ET).
fn week_already_downloaded(monday: NaiveDate, existing: &HashSet<String>) -> bool {
    for minute in 0..10_u32 {
        let hour   = 13_u32;
        let min    = 30 + minute;
        let ts_str = format!("{} {:02}:{:02}:00", monday, hour, min);
        if existing.contains(&ts_str) {
            return true;
        }
    }
    false
}

/// Returns all Mondays between start and end dates inclusive.
fn get_mondays(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut mondays = Vec::new();
    // Advance to first Monday
    let mut current = start;
    while current.weekday().num_days_from_monday() != 0 {
        current += Duration::days(1);
    }
    while current <= end {
        mondays.push(current);
        current += Duration::weeks(1);
    }
    mondays
}

// ─────────────────────────────────────────────
//  API call
// ─────────────────────────────────────────────

/// Makes one API call to fetch one week of 1-minute bars starting on `monday`.
/// Returns a Vec of Bar rows, or empty Vec on error.
fn fetch_week(symbol: &str, monday: NaiveDate, api_key: &str) -> Vec<Bar> {
    let friday    = monday + Duration::days(4);
    let start_str = format!("{} 00:00:00", monday);
    let end_str   = format!("{} 23:59:59", friday);

    let url = format!(
        "{}?symbol={}&interval=1min&start_date={}&end_date={}&outputsize={}&apikey={}&format=JSON&timezone=UTC",
        TWELVE_DATA_URL,
        symbol,
        urlencoded(&start_str),
        urlencoded(&end_str),
        POINTS_PER_CALL,
        api_key,
    );

    let response = match reqwest::blocking::get(&url) {
        Ok(r)  => r,
        Err(e) => {
            eprintln!("  Request failed: {}", e);
            return vec![];
        }
    };

    let json: serde_json::Value = match response.json() {
        Ok(j)  => j,
        Err(e) => {
            eprintln!("  Failed to parse JSON: {}", e);
            return vec![];
        }
    };

    // Check for API-level errors
    if json["status"] == "error" {
        eprintln!("  API error: {}", json["message"].as_str().unwrap_or("unknown"));
        return vec![];
    }

    let values = match json["values"].as_array() {
        Some(v) => v,
        None    => {
            println!("  No data returned (may be a holiday week)");
            return vec![];
        }
    };

    let mut bars: Vec<Bar> = values.iter().map(|v| Bar {
        timestamp: v["datetime"].as_str().unwrap_or("").to_string(),
        symbol:    symbol.to_string(),
        open:      v["open"].as_str().unwrap_or("0").to_string(),
        high:      v["high"].as_str().unwrap_or("0").to_string(),
        low:       v["low"].as_str().unwrap_or("0").to_string(),
        close:     v["close"].as_str().unwrap_or("0").to_string(),
        volume:    v["volume"].as_str().unwrap_or("0").to_string(),
    }).collect();

    // Twelve Data returns newest-first — sort oldest-first to match our format
    bars.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    bars
}

/// Minimal URL encoding for date strings (replaces spaces with %20)
fn urlencoded(s: &str) -> String {
    s.replace(' ', "%20")
}

// ─────────────────────────────────────────────
//  Save helpers
// ─────────────────────────────────────────────

/// Saves new bars to the CSV file, skipping timestamps already present.
/// Loads all existing rows, merges with new ones, sorts, and rewrites the file.
/// Returns the number of new rows written.
fn save_bars(new_bars: &[Bar], filepath: &str, existing: &HashSet<String>) -> usize {
    // Filter to only rows we don't already have
    let filtered: Vec<&Bar> = new_bars
        .iter()
        .filter(|b| !existing.contains(&b.timestamp))
        .collect();

    if filtered.is_empty() {
        println!("  All rows already present, nothing to write.");
        return 0;
    }

    // Load existing rows from the file
    let mut all_rows: Vec<Vec<String>> = Vec::new();

    if Path::new(filepath).exists() {
        let file = File::open(filepath).expect("Cannot open file");
        for (i, line) in BufReader::new(file).lines().enumerate() {
            if i == 0 { continue; } // skip header
            if let Ok(line) = line {
                let cols: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
                if cols.len() >= 7 {
                    all_rows.push(cols);
                }
            }
        }
    }

    // Add new rows
    for bar in &filtered {
        all_rows.push(vec![
            bar.timestamp.clone(),
            bar.symbol.clone(),
            bar.open.clone(),
            bar.high.clone(),
            bar.low.clone(),
            bar.close.clone(),
            bar.volume.clone(),
        ]);
    }

    // Sort all rows by timestamp oldest-first
    all_rows.sort_by(|a, b| a[0].cmp(&b[0]));

    // Write back to file
    let mut file = File::create(filepath).expect("Cannot create file");
    writeln!(file, "Timestamp,Symbol,Open,High,Low,Close,Volume").unwrap();
    for row in &all_rows {
        writeln!(file, "{}", row.join(",")).unwrap();
    }

    println!(
        "  Saved {} new rows  ({} total in file)",
        filtered.len(),
        all_rows.len()
    );

    filtered.len()
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        eprintln!("Usage:   {} <SYMBOL> <START_DATE> <END_DATE> <API_KEY>", args[0]);
        eprintln!("Example: {} AAPL 2024-01-01 2026-03-09 your_key_here", args[0]);
        eprintln!();
        eprintln!("Get a free API key at: https://twelvedata.com");
        std::process::exit(1);
    }

    let symbol     = args[1].to_uppercase();
    let start_date = NaiveDate::parse_from_str(&args[2], "%Y-%m-%d")
        .expect("Invalid start date. Use YYYY-MM-DD");
    let end_date   = NaiveDate::parse_from_str(&args[3], "%Y-%m-%d")
        .expect("Invalid end date. Use YYYY-MM-DD");
    let api_key    = &args[4];
    let filepath   = data_file_for(&symbol);
    let today      = chrono::Local::now().date_naive();

    println!("Symbol:  {}", symbol);
    println!("Range:   {}  to  {}", start_date, end_date);
    println!("File:    {}", filepath);
    println!();

    let mut existing = load_existing_timestamps(&filepath);
    let mondays      = get_mondays(start_date, end_date);

    println!("{} weeks to check\n", mondays.len());

    let mut total_new  = 0_usize;
    let mut calls_made = 0_usize;
    let mut skipped    = 0_usize;

    for (i, &monday) in mondays.iter().enumerate() {
        print!("[Week {:>3}/{}]  {}  ", i + 1, mondays.len(), monday);

        // Skip if already downloaded
        if week_already_downloaded(monday, &existing) {
            println!("already downloaded, skipping.");
            skipped += 1;
            continue;
        }

        // Skip future weeks
        if monday > today {
            println!("future date, skipping.");
            skipped += 1;
            continue;
        }

        println!("fetching...");
        let bars = fetch_week(&symbol, monday, api_key);
        calls_made += 1;

        if !bars.is_empty() {
            let added = save_bars(&bars, &filepath, &existing);
            total_new += added;
            // Update in-memory set so later weeks don't re-add the same rows
            for bar in &bars {
                existing.insert(bar.timestamp.clone());
            }
        }

        // Rate limit — wait between calls (skip wait after the last call)
        let remaining_needed = mondays[i + 1..].iter()
            .filter(|&&m| !week_already_downloaded(m, &existing) && m <= today)
            .count();

        if remaining_needed > 0 {
            println!("  Waiting {}s (rate limit)...", RATE_LIMIT_SECS);
            thread::sleep(time::Duration::from_secs(RATE_LIMIT_SECS));
        }
    }

    println!();
    println!("{}", "=".repeat(45));
    println!("Done.");
    println!("  API calls made:  {}", calls_made);
    println!("  Weeks skipped:   {}", skipped);
    println!("  New rows added:  {}", total_new);
    println!("  Output file:     {}", filepath);
    println!();
    println!("To train:");
    println!("  cargo run --bin stock_trainer -- {} {}_weights.csv", filepath, symbol);
}
