use chrono::{DateTime, Duration, NaiveDate, Utc};
use rand::Rng;
use std::env;

#[derive(Debug, Clone)]
struct StockData {
    timestamp: DateTime<Utc>,
    symbol: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}

impl StockData {
    fn generate(timestamp: DateTime<Utc>, symbol: &str) -> Self {
        let mut rng = rand::thread_rng();
        
        // Generate realistic stock data
        let open: f64 = rng.gen_range(100.0..200.0);
        let close: f64 = open + rng.gen_range(-5.0..5.0);
        let high: f64 = open.max(close) + rng.gen_range(0.0..3.0);
        let low: f64 = open.min(close) - rng.gen_range(0.0..3.0);
        let volume: u64 = rng.gen_range(100_000..1_000_000);
        
        StockData {
            timestamp,
            symbol: symbol.to_string(),
            open,
            high,
            low,
            close,
            volume,
        }
    }
}

fn generate_stock_data_for_window(
    date: NaiveDate,
    symbol: &str,
    start_hour: u32,
    start_minute: u32,
    duration_minutes: u32,
) -> Vec<StockData> {
    let mut data = Vec::new();
    
    // Create starting time
    let start_time = date
        .and_hms_opt(start_hour, start_minute, 0)
        .expect("Invalid time")
        .and_utc();
    
    // Generate data points for every minute in the window
    for minute in 0..duration_minutes {
        let timestamp = start_time + Duration::minutes(minute as i64);
        let stock_data = StockData::generate(timestamp, symbol);
        data.push(stock_data);
    }
    
    data
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <YYYY-MM-DD> [symbol] [start_hour] [start_minute]", args[0]);
        eprintln!("Example: {} 2026-03-09 AAPL 09 30", args[0]);
        eprintln!("\nDefaults: symbol=AAPL, start_hour=09, start_minute=30");
        std::process::exit(1);
    }
    
    // Parse date
    let date = NaiveDate::parse_from_str(&args[1], "%Y-%m-%d")
        .expect("Invalid date format. Use YYYY-MM-DD");
    
    // Parse optional arguments
    let symbol = args.get(2).map(|s| s.as_str()).unwrap_or("AAPL");
    let start_hour = args
        .get(3)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(9);
    let start_minute = args
        .get(4)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(30);
    
    // Generate 2 hours (120 minutes) of data
    let stock_data = generate_stock_data_for_window(date, symbol, start_hour, start_minute, 120);
    
    // Print header
    println!("Timestamp,Symbol,Open,High,Low,Close,Volume");
    
    // Print stock data
    for data in stock_data {
        println!(
            "{},{},{:.2},{:.2},{:.2},{:.2},{}",
            data.timestamp.format("%Y-%m-%d %H:%M:%S"),
            data.symbol,
            data.open,
            data.high,
            data.low,
            data.close,
            data.volume
        );
    }
}
