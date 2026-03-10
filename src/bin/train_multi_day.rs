
// train_multi_day.rs
//
// FORTIFIED RETURNS VERSION:
// 1. Fixed NaN (Not a Number) issues with Gradient Clipping.
// 2. Small Random Weight Initialization.
// 3. Data Sanitizer for 0-price rows.

use chrono::{DateTime, Datelike, Duration, Local, NaiveDate, Utc};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::thread;
use std::time::{self, Instant};
use rand::seq::SliceRandom;
use rand::Rng; // Added for random initialization

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────

const TWELVE_DATA_URL: &str  = "https://api.twelvedata.com/time_series";
const RATE_LIMIT_SECS: u64   = 8;
const POINTS_PER_CALL: usize = 1950;
const TEST_DAYS: usize       = 10;
const DECAY_RATE: f64        = 0.005;

// ─────────────────────────────────────────────
//  Data types
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StockData {
    timestamp: DateTime<Utc>,
    symbol:    String,
    open:      f64,
    high:      f64,
    low:       f64,
    close:     f64,
    volume:    u64,
}

// ─────────────────────────────────────────────
//  Utility & Data Fetching
// ─────────────────────────────────────────────

fn data_file_for(symbol: &str) -> String { format!("{}_data.csv", symbol) }

fn load_existing_timestamps(filepath: &str) -> HashSet<String> {
    let mut ts = HashSet::new();
    if !Path::new(filepath).exists() { return ts; }
    let file = match File::open(filepath) { Ok(f) => f, Err(_) => return ts };
    for (i, line) in io::BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; }
        if let Ok(l) = line {
            if let Some(t) = l.split(',').next() { ts.insert(t.trim().to_string()); }
        }
    }
    ts
}

fn week_already_downloaded(monday: NaiveDate, existing: &HashSet<String>) -> bool {
    for day_offset in 0..5_i64 {
        let day = monday + Duration::days(day_offset);
        let prefix = day.to_string();
        if existing.iter().any(|ts| ts.starts_with(&prefix)) { return true; }
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

fn fetch_week(symbol: &str, monday: NaiveDate, api_key: &str) -> Vec<Vec<String>> {
    let friday = monday + Duration::days(4);
    let url = format!(
        "{}?symbol={}&interval=1min&start_date={}&end_date={}&outputsize={}&apikey={}&format=JSON&timezone=UTC",
        TWELVE_DATA_URL, symbol,
        format!("{} 00:00:00", monday).replace(' ', "%20"),
        format!("{} 23:59:59", friday).replace(' ', "%20"),
        POINTS_PER_CALL, api_key
    );
    let resp = match reqwest::blocking::get(&url) {
        Ok(r) => r, Err(e) => { eprintln!("  Request failed: {}", e); return vec![]; }
    };
    let json: serde_json::Value = match resp.json() {
        Ok(j) => j, Err(e) => { eprintln!("  JSON error: {}", e); return vec![]; }
    };
    if json["status"] == "error" {
        eprintln!("  API error: {}", json["message"].as_str().unwrap_or("unknown"));
        return vec![];
    }
    let values = match json["values"].as_array() {
        Some(v) => v, None => { return vec![]; }
    };
    let mut rows: Vec<Vec<String>> = values.iter().map(|v| vec![
        v["datetime"].as_str().unwrap_or("").to_string(),
        symbol.to_string(),
        v["open"].as_str().unwrap_or("0").to_string(),
        v["high"].as_str().unwrap_or("0").to_string(),
        v["low"].as_str().unwrap_or("0").to_string(),
        v["close"].as_str().unwrap_or("0").to_string(),
        v["volume"].as_str().unwrap_or("0").to_string(),
    ]).collect();
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    rows
}

fn save_raw_rows(new_rows: &[Vec<String>], filepath: &str, existing: &HashSet<String>) -> usize {
    let filtered: Vec<&Vec<String>> = new_rows.iter()
        .filter(|r| !existing.contains(&r[0])).collect();
    if filtered.is_empty() { return 0; }
    let mut all: Vec<Vec<String>> = Vec::new();
    if Path::new(filepath).exists() {
        let f = File::open(filepath).unwrap();
        for (i, line) in io::BufReader::new(f).lines().enumerate() {
            if i == 0 { continue; }
            if let Ok(l) = line {
                let cols: Vec<String> = l.split(',').map(|s| s.trim().to_string()).collect();
                if cols.len() >= 7 { all.push(cols); }
            }
        }
    }
    for r in &filtered { all.push((*r).clone()); }
    all.sort_by(|a, b| a[0].cmp(&b[0]));
    let mut f = File::create(filepath).unwrap();
    writeln!(f, "Timestamp,Symbol,Open,High,Low,Close,Volume").unwrap();
    for row in &all { writeln!(f, "{}", row.join(",")).unwrap(); }
    filtered.len()
}

fn ensure_data_available(symbol: &str, start: NaiveDate, end: NaiveDate, api_keys: &[String]) {
    let filepath = data_file_for(symbol);
    let today    = Local::now().date_naive();
    let mut existing = load_existing_timestamps(&filepath);
    let missing: Vec<NaiveDate> = get_mondays(start, end).into_iter()
        .filter(|&m| !week_already_downloaded(m, &existing) && m <= today)
        .collect();
    if missing.is_empty() { return; }

    let n_keys = api_keys.len();
    let effective_wait = (RATE_LIMIT_SECS / n_keys as u64).max(1);

    for (i, &monday) in missing.iter().enumerate() {
        let key = &api_keys[i % n_keys];
        let rows = fetch_week(symbol, monday, key);
        if !rows.is_empty() {
            save_raw_rows(&rows, &filepath, &existing);
            for r in &rows { existing.insert(r[0].clone()); }
        }
        if i < missing.len() - 1 { thread::sleep(time::Duration::from_secs(effective_wait)); }
    }
}

fn parse_csv(path: &str) -> Vec<StockData> {
    let file = File::open(path).expect("Cannot open CSV");
    let mut data = Vec::new();
    for (i, line) in io::BufReader::new(file).lines().enumerate() {
        let line = line.unwrap();
        if i == 0 { continue; }
        let cols: Vec<&str> = line.splitn(7, ',').collect();
        if cols.len() < 7 { continue; }

        let close_p: f64 = cols[5].trim().parse().unwrap_or(0.0);
        // DATA SANITIZER: Ignore broken API lines with $0 prices
        if close_p < 0.01 { continue; }

        let ts = DateTime::parse_from_str(&format!("{} +0000", cols[0].trim()), "%Y-%m-%d %H:%M:%S %z")
            .map(|dt| dt.with_timezone(&Utc)).unwrap_or(Utc::now());

        data.push(StockData {
            timestamp: ts, symbol: cols[1].trim().to_string(),
            open: cols[2].trim().parse().unwrap_or(0.0), high: cols[3].trim().parse().unwrap_or(0.0),
            low: cols[4].trim().parse().unwrap_or(0.0), close: close_p,
            volume: cols[6].trim().parse().unwrap_or(0),
        });
    }
    data
}

fn group_by_day(data: Vec<StockData>) -> Vec<(NaiveDate, Vec<StockData>)> {
    let mut map: HashMap<NaiveDate, Vec<StockData>> = HashMap::new();
    for row in data { map.entry(row.timestamp.date_naive()).or_default().push(row); }
    let mut days: Vec<_> = map.into_iter().collect();
    days.sort_by_key(|(d, _)| *d);
    for (_, rows) in &mut days { rows.sort_by_key(|r| r.timestamp); }
    days
}

// ─────────────────────────────────────────────
//  StockTrainer Implementation
// ─────────────────────────────────────────────

pub struct StockTrainer {
    weights:       Vec<Vec<f64>>,
    biases:        Vec<f64>,
    lookback:      usize,
    horizon:       usize,
    learning_rate: f64,
    pub epochs:    usize,
}

impl StockTrainer {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let lookback = 60;
        let horizon = 10;
        let nf = 3;

        // Random small weight initialization to prevent dead neurons and NaN
        let weights = (0..horizon).map(|_| {
            (0..lookback * nf).map(|_| rng.gen_range(-0.01..0.01)).collect()
        }).collect();

        StockTrainer {
            weights,
            biases: vec![0.0; horizon],
            lookback,
            horizon,
            learning_rate: 0.001, // Lower learning rate for more stability
            epochs:        40,
        }
    }

    pub fn build_samples(&self, data: &[StockData]) -> Option<Vec<(Vec<f64>, Vec<f64>)>> {
        if data.len() < self.lookback + self.horizon + 1 { return None; }
        let mut samples = Vec::new();
        for i in 1..(data.len() - self.lookback - self.horizon) {
            let mut input = Vec::with_capacity(self.lookback * 3);
            for j in 0..self.lookback {
                let curr = &data[i + j];
                let prev = &data[i + j - 1];

                // Safe returns calculation to prevent NaN
                let price_ret = (curr.close - prev.close) / prev.close.max(1e-6);
                let vola = (curr.high - curr.low) / curr.close.max(1e-6);
                let v_change = (curr.volume as f64 - prev.volume as f64) / (prev.volume as f64).max(1.0);

                input.push(price_ret);
                input.push(vola);
                input.push(v_change);
            }
            let current_price = data[i + self.lookback - 1].close;
            let targets: Vec<f64> = (0..self.horizon).map(|s| {
                let future_price = data[i + self.lookback + s].close;
                (future_price - current_price) / current_price.max(1e-6)
            }).collect();
            samples.push((input, targets));
        }
        Some(samples)
    }

    pub fn train_one_pass(&mut self, samples: &[(Vec<f64>, Vec<f64>)], weight: f64) -> f64 {
        let mut total_mse = 0.0;
        let adj_lr = self.learning_rate * weight;

        for (input, targets) in samples {
            for i in 0..self.horizon {
                let pred = self.weights[i].iter().zip(input).map(|(w, x)| w * x).sum::<f64>() + self.biases[i];

                if pred.is_nan() { continue; } // Skip if calculation broke

                let mut err = pred - targets[i];

                // GRADIENT CLIPPING: Prevents NaN by capping the error impact
                if err > 1.0 { err = 1.0; }
                if err < -1.0 { err = -1.0; }

                total_mse += err * err;
                let grad = adj_lr * 2.0 * err;

                for j in 0..self.weights[i].len() {
                    self.weights[i][j] -= grad * input[j];
                    // Weight capping for extra stability
                    if self.weights[i][j] > 10.0 { self.weights[i][j] = 10.0; }
                    if self.weights[i][j] < -10.0 { self.weights[i][j] = -10.0; }
                }
                self.biases[i] -= grad;
            }
        }
        total_mse / (samples.len() * self.horizon) as f64
    }

    pub fn test(&self, days: &[(NaiveDate, Vec<StockData>)]) -> (f64, f64) {
        let (mut dir_correct, mut dir_total) = (0, 0);
        let mut mse_sum = 0.0;
        for (_, data) in days {
            if let Some(samples) = self.build_samples(data) {
                for (input, targets) in samples {
                    let pred_ret = self.weights[0].iter().zip(&input).map(|(w, x)| w * x).sum::<f64>() + self.biases[0];
                    let actual_ret = targets[0];
                    mse_sum += (pred_ret - actual_ret).powi(2);

                    if (pred_ret > 0.0 && actual_ret > 0.0) || (pred_ret < 0.0 && actual_ret < 0.0) {
                        dir_correct += 1;
                    }
                    dir_total += 1;
                }
            }
        }
        (mse_sum / dir_total.max(1) as f64, (dir_correct as f64 / dir_total.max(1) as f64) * 100.0)
    }

    pub fn predict(&self, recent: &[StockData]) -> Vec<f64> {
        let mut input = Vec::new();
        if recent.len() <= self.lookback { return vec![0.0; self.horizon]; }

        for i in (recent.len() - self.lookback)..recent.len() {
            let curr = &recent[i];
            let prev = &recent[i - 1];
            input.push((curr.close - prev.close) / prev.close.max(1e-6));
            input.push((curr.high - curr.low) / curr.close.max(1e-6));
            input.push((curr.volume as f64 - prev.volume as f64) / (prev.volume as f64).max(1.0));
        }
        let current_price = recent.last().unwrap().close;
        (0..self.horizon).map(|i| {
            let pred_ret = self.weights[i].iter().zip(&input).map(|(w, x)| w * x).sum::<f64>() + self.biases[i];
            current_price * (1.0 + pred_ret)
        }).collect()
    }
}

// ─────────────────────────────────────────────
//  Main Execution Loop
// ─────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 { return; }
    let symbol = &args[1];
    let start = NaiveDate::parse_from_str(&args[2], "%Y-%m-%d").expect("Invalid Start Date");
    let end   = NaiveDate::parse_from_str(&args[3], "%Y-%m-%d").expect("Invalid End Date");
    let gens  = args[4].parse::<usize>().unwrap_or(5);

    let key1 = env::var("TWELVE_DATA_KEY").expect("Set TWELVE_DATA_KEY");
    ensure_data_available(symbol, start, end, &[key1]);

    let all_data = parse_csv(&data_file_for(symbol));
    let mut days = group_by_day(all_data.clone());
    let test_days = days.split_off(days.len().saturating_sub(TEST_DAYS));

    let mut trainer = StockTrainer::new();
    let mut samples_with_weights: Vec<_> = days.iter().enumerate().filter_map(|(i, (_, d))| {
        let s = trainer.build_samples(d)?;
        let weight = (-DECAY_RATE * (days.len() - 1 - i) as f64).exp();
        Some((s, weight))
    }).collect();

    let mut rng = rand::thread_rng();

    println!("\n[system] Training on {} days of data...", days.len());

    for g in 1..=gens {
        println!("\n━━━ Gen {}/{} ━━━━━━━━━━━━━━━━━━━━━━━━", g, gens);
        for _epoch in 1..=trainer.epochs {
            samples_with_weights.shuffle(&mut rng);
            for (samples, weight) in &samples_with_weights {
                trainer.train_one_pass(samples, *weight);
            }
        }
        let (mse, d_acc) = trainer.test(&test_days);
        println!("  Directional Acc: {:.2}% | MSE: {:.8}", d_acc, mse);
    }

    let predictions = trainer.predict(&all_data);
    println!("\nFinal 10-Min Forecast for {}:", symbol);
    for (i, p) in predictions.iter().enumerate() {
        println!("  +{:>2} min: ${:.2}", i + 1, p);
    }
}
