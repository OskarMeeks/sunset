use chrono::{DateTime, Utc};
use std::env;
use std::fs::File;
use std::io::{self, BufRead};

// ─────────────────────────────────────────────
//  Data structures
// ─────────────────────────────────────────────

// One row of stock data — one minute of price information.
// Mirrors exactly what the data generator in main.rs produces.
#[derive(Debug, Clone)]
struct StockData {
    timestamp: DateTime<Utc>, // what minute this data is from
    symbol: String,            // e.g. "AAPL"
    open: f64,                 // price at the start of this minute
    high: f64,                 // highest price during this minute
    low: f64,                  // lowest price during this minute
    close: f64,                // price at the end of this minute
    volume: u64,               // how many shares were traded this minute
}

// The 7 numbers we extract from each minute of stock data.
// These are fed into the model as inputs.
#[derive(Debug, Clone)]
struct Features {
    close: f64,
    open: f64,
    high: f64,
    low: f64,
    volume_norm: f64,  // volume scaled to 0.0-1.0 so it's comparable to prices
    price_range: f64,  // how wide the price swung this minute (high - low)
    price_change: f64, // did the price go up or down this minute (close - open)
}

// ─────────────────────────────────────────────
//  StockTrainer — single multi-output model
// ─────────────────────────────────────────────

// This is the AI model. It takes a 2-hour window of stock data as input
// and outputs 10 predicted close prices (one per future minute) all at once.
//
// Internally it's a multi-output linear regression:
//   inputs:  840 numbers  (120 minutes x 7 features)
//   outputs: 10 numbers   (predicted close price at +1, +2, ... +10 minutes)
//
// The weights are stored as a 2D grid: weights[output][input]
// So weights[0] is the row of 840 weights that produce the +1 min prediction,
//    weights[1] produces the +2 min prediction, and so on.
// All 10 output rows are trained together in one pass, so the model learns
// the relationship between all future minutes at the same time.
#[derive(Debug, Clone)]
pub struct StockTrainer {
    start_date: Option<String>,
    end_date: Option<String>,

    // Weight matrix: one row per output (10 rows), one column per input (840 columns).
    // weights[i][j] = "how much does input j contribute to the prediction at minute +i+1?"
    weights: Vec<Vec<f64>>,

    // One bias per output. Each output has its own baseline to shift from.
    biases: Vec<f64>,

    // Min and max values recorded during training, used to normalise new data.
    feature_min: Vec<f64>,
    feature_max: Vec<f64>,

    // 120 minutes of history as input (2 hours)
    lookback: usize,

    // 10 minutes of predictions as output
    horizon: usize,

    // How big each weight update step is. 0.001 is a safe default.
    learning_rate: f64,

    // How many full passes through the training data to do.
    epochs: usize,

    // Accuracy stats calculated from the last 20 epochs of training.
    // avg_dollar_error = on average, how many dollars off were the predictions?
    // accuracy_pct     = what percentage of the real price was the model correct to?
    //                    e.g. 98.5% means predictions were within $1.50 on a $100 stock
    avg_dollar_error: f64,
    accuracy_pct: f64,
}

impl StockTrainer {
    // Creates a new trainer with default settings.
    pub fn new() -> Self {
        let lookback  = 120; // 2 hours
        let horizon   = 10;  // predict next 10 minutes
        let n_features = 7;  // 7 numbers extracted per minute
        let input_size = lookback * n_features; // 840 total inputs

        StockTrainer {
            start_date: None,
            end_date:   None,
            // Weight matrix: 10 rows (one per future minute) x 840 columns (one per input).
            // All start at 0.0 and get updated during training.
            weights:     vec![vec![0.0; input_size]; horizon],
            biases:      vec![0.0; horizon],
            feature_min: vec![f64::MAX; n_features],
            feature_max: vec![f64::MIN; n_features],
            lookback,
            horizon,
            learning_rate: 0.001,
            epochs: 500,
            avg_dollar_error: 0.0,
            accuracy_pct: 0.0,
        }
    }

    pub fn set_start_date(&mut self, date: String) { self.start_date = Some(date); }
    pub fn set_end_date(&mut self, date: String)   { self.end_date   = Some(date); }

    // ── Feature extraction ──────────────────────────────────────────────

    // Converts one row of StockData into 7 numbers the model can learn from.
    fn extract_features(d: &StockData, max_volume: f64) -> Features {
        Features {
            close:        d.close,
            open:         d.open,
            high:         d.high,
            low:          d.low,
            volume_norm:  d.volume as f64 / max_volume.max(1.0),
            price_range:  d.high - d.low,
            price_change: d.close - d.open,
        }
    }

    // Flattens a Features struct into a plain list of numbers.
    fn features_to_vec(f: &Features) -> Vec<f64> {
        vec![f.close, f.open, f.high, f.low, f.volume_norm, f.price_range, f.price_change]
    }

    // ── Normalisation ───────────────────────────────────────────────────

    // Records the min and max of each feature across the whole training set.
    // Called once before training starts.
    fn fit_normalisation(&mut self, all_features: &[Vec<f64>]) {
        for fv in all_features {
            for (j, &v) in fv.iter().enumerate() {
                if v < self.feature_min[j] { self.feature_min[j] = v; }
                if v > self.feature_max[j] { self.feature_max[j] = v; }
            }
        }
    }

    // Scales each feature to a 0.0-1.0 range using the min/max from training.
    // This stops the model from treating large numbers as more important.
    fn normalise(&self, fv: &[f64]) -> Vec<f64> {
        fv.iter().enumerate().map(|(j, &v)| {
            let range = self.feature_max[j] - self.feature_min[j];
            if range < 1e-10 { 0.0 } else { (v - self.feature_min[j]) / range }
        }).collect()
    }

    // Reverses normalisation to turn a 0-1 prediction back into a real dollar price.
    fn denormalise_price(&self, norm_val: f64) -> f64 {
        let range = self.feature_max[0] - self.feature_min[0];
        norm_val * range + self.feature_min[0]
    }

    // ── Model forward pass ──────────────────────────────────────────────

    // Runs the model on one input window and returns all 10 predictions at once.
    // For each output i: prediction[i] = dot(weights[i], input) + biases[i]
    // This is the "forward pass" — one trip through the model.
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        (0..self.horizon).map(|i| {
            let dot: f64 = self.weights[i].iter().zip(input.iter()).map(|(w, x)| w * x).sum();
            dot + self.biases[i]
        }).collect()
    }

    // ── Training ────────────────────────────────────────────────────────

    // Trains the model on historical stock data.
    //
    // For each sliding 2-hour window in the data, we:
    //   1. Run the model to get 10 predicted prices
    //   2. Compare each prediction to the real price at that future minute
    //   3. Nudge all the weights slightly toward the right answer
    //   4. Repeat for every window, every epoch
    pub fn train(&mut self, data: &[StockData]) {
        self.train_inner(data, true);
    }

    // Like train() but does NOT reset the weights first.
    // Call this when training on multiple days one at a time —
    // the model keeps what it learned from previous days and builds on it.
    pub fn train_incremental(&mut self, data: &[StockData]) {
        self.train_inner(data, false);
    }

    // Internal training implementation.
    // reset_normalisation = true  on the first day (fresh start)
    // reset_normalisation = false on subsequent days (keep existing weights)
    fn train_inner(&mut self, data: &[StockData], reset_normalisation: bool) {
        if data.len() < self.lookback + self.horizon {
            eprintln!("[trainer] Need at least {} rows to train", self.lookback + self.horizon);
            return;
        }

        let max_volume = data.iter().map(|d| d.volume as f64).fold(0.0_f64, f64::max);

        // Turn every row into a feature vector
        let raw_features: Vec<Vec<f64>> = data.iter()
            .map(|d| Self::features_to_vec(&Self::extract_features(d, max_volume)))
            .collect();

        // On the first day, learn min/max from scratch.
        // On subsequent days, expand the min/max to include the new data —
        // this keeps the normalisation consistent across all days.
        if reset_normalisation {
            self.feature_min = vec![f64::MAX; 7];
            self.feature_max = vec![f64::MIN; 7];
        }
        self.fit_normalisation(&raw_features);

        // Scale every feature vector to 0-1
        let norm_features: Vec<Vec<f64>> = raw_features.iter()
            .map(|fv| self.normalise(fv))
            .collect();

        // Build training samples.
        // Each sample is:
        //   input  = 120 rows flattened into 840 numbers
        //   target = the real close prices at +1, +2, ... +10 minutes after the window
        let mut samples: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

        for i in 0..(data.len() - self.lookback - self.horizon + 1) {
            // Flatten the 2-hour window into one long input vector
            let input: Vec<f64> = norm_features[i..i + self.lookback]
                .iter()
                .flat_map(|v| v.clone())
                .collect();

            // The 10 target prices, each normalised to 0-1
            let targets: Vec<f64> = (0..self.horizon).map(|step| {
                let raw = data[i + self.lookback + step].close;
                (raw - self.feature_min[0]) / (self.feature_max[0] - self.feature_min[0]).max(1e-10)
            }).collect();

            samples.push((input, targets));
        }

        let n = samples.len();
        println!("[trainer] Training single model ({} inputs -> {} outputs) on {} samples, {} epochs ...",
            self.lookback * 7, self.horizon, n, self.epochs);

        // We'll collect the average dollar error from the last 20 epochs to measure accuracy.
        // Using 20 epochs instead of just the final one gives a more stable reading,
        // since the loss bounces around a little at the end of training.
        let last_n_epochs = 20_usize;
        let mut recent_dollar_errors: Vec<f64> = Vec::new();

        // Training loop — SGD (Stochastic Gradient Descent)
        for epoch in 0..self.epochs {
            let mut total_loss         = 0.0_f64;
            let mut total_dollar_error = 0.0_f64;
            let mut count              = 0_usize;

            for (input, targets) in &samples {
                // Run the model: one forward pass gives all 10 predictions at once
                let preds = self.forward(input);

                for i in 0..self.horizon {
                    let err = preds[i] - targets[i]; // positive = guessed too high
                    total_loss += err * err;

                    // Convert both prediction and target back to real dollar values
                    // so we can measure the error in dollars rather than 0-1 scale
                    let predicted_price = self.denormalise_price(preds[i]);
                    let real_price      = self.denormalise_price(targets[i]);
                    total_dollar_error += (predicted_price - real_price).abs();
                    count += 1;

                    // Nudge weights toward the right answer
                    for j in 0..self.weights[i].len() {
                        self.weights[i][j] -= self.learning_rate * 2.0 * err * input[j];
                    }
                    self.biases[i] -= self.learning_rate * 2.0 * err;
                }
            }

            // Print MSE every 100 epochs so you can watch the model improve
            if epoch % 100 == 0 {
                let mse = total_loss / (n * self.horizon) as f64;
                println!("  Epoch {:>4}  MSE = {:.6}", epoch, mse);
            }

            // Save the dollar error for the last 20 epochs
            if epoch >= self.epochs.saturating_sub(last_n_epochs) {
                recent_dollar_errors.push(total_dollar_error / count as f64);
            }
        }

        // Average dollar error across the last 20 epochs
        self.avg_dollar_error = recent_dollar_errors.iter().sum::<f64>()
            / recent_dollar_errors.len().max(1) as f64;

        // Accuracy % relative to the average close price in the training data.
        // e.g. avg price = $150, avg error = $1.50  =>  accuracy = 99.0%
        let avg_close = data.iter().map(|d| d.close).sum::<f64>() / data.len() as f64;
        self.accuracy_pct = (1.0 - (self.avg_dollar_error / avg_close)).max(0.0) * 100.0;

        println!("[trainer] Training complete.");
        println!("[trainer] Accuracy (last {} epochs): {:.2}%  |  Avg dollar error: ${:.4}",
            last_n_epochs, self.accuracy_pct, self.avg_dollar_error);
    }

    // ── Prediction ──────────────────────────────────────────────────────

    // Given the most recent 2 hours of stock data, returns 10 predicted close prices.
    // The model sees the whole 2-hour window at once and produces all 10 values
    // in a single forward pass — no looping, no separate models.
    pub fn predict(&self, recent_data: &[StockData]) -> Vec<f64> {
        if recent_data.len() < self.lookback {
            eprintln!("[trainer] Need at least {} rows (2 hours) to predict", self.lookback);
            return vec![];
        }

        let max_volume = recent_data.iter().map(|d| d.volume as f64).fold(0.0_f64, f64::max);

        // Take the most recent 120 rows and flatten them into one 840-number input
        let window_start = recent_data.len() - self.lookback;
        let input: Vec<f64> = recent_data[window_start..]
            .iter()
            .flat_map(|d| {
                let fv = Self::features_to_vec(&Self::extract_features(d, max_volume));
                self.normalise(&fv)
            })
            .collect();

        // One forward pass → 10 normalised predictions
        let norm_preds = self.forward(&input);

        // Convert each prediction from 0-1 scale back to a real dollar price
        let predictions: Vec<f64> = norm_preds.iter()
            .map(|&p| self.denormalise_price(p))
            .collect();

        for (i, price) in predictions.iter().enumerate() {
            println!("  +{:>2} min  predicted close: ${:.2}", i + 1, price);
        }

        predictions
    }

    // ── Save / Load ──────────────────────────────────────────────────────

    // Saves the trained weights, biases, and accuracy stats to a CSV file.
    // The first two lines are always the accuracy stats so they're easy to read.
    // Format for weights: "w,ROW,COL,VALUE"
    // Format for biases:  "bias,INDEX,VALUE"
    // Format for stats:   "accuracy_pct,VALUE" and "avg_dollar_error,VALUE"
    pub fn save_weights(&self, path: &str) -> io::Result<()> {
        use std::io::Write;
        let mut f = File::create(path)?;

        // Write accuracy stats first so they're at the top of the file
        writeln!(f, "accuracy_pct,{}", self.accuracy_pct)?;
        writeln!(f, "avg_dollar_error,{}", self.avg_dollar_error)?;

        for (i, bias) in self.biases.iter().enumerate() {
            writeln!(f, "bias,{},{}", i, bias)?;
        }
        for (i, row) in self.weights.iter().enumerate() {
            for (j, w) in row.iter().enumerate() {
                writeln!(f, "w,{},{},{}", i, j, w)?;
            }
        }
        println!("[trainer] Weights saved to {}  (accuracy: {:.2}%, avg error: ${:.4})",
            path, self.accuracy_pct, self.avg_dollar_error);
        Ok(())
    }

    // Loads weights, biases, and accuracy stats back from a saved file.
    pub fn load_weights(&mut self, path: &str) -> io::Result<()> {
        let f = File::open(path)?;
        for line in io::BufReader::new(f).lines() {
            let line  = line?;
            let parts: Vec<&str> = line.splitn(4, ',').collect();
            match parts[0] {
                "accuracy_pct"     => { self.accuracy_pct     = parts[1].parse().unwrap_or(0.0); }
                "avg_dollar_error" => { self.avg_dollar_error = parts[1].parse().unwrap_or(0.0); }
                "bias" => {
                    let i: usize = parts[1].parse().unwrap_or(0);
                    self.biases[i] = parts[2].parse().unwrap_or(0.0);
                }
                "w" => {
                    let i: usize = parts[1].parse().unwrap_or(0);
                    let j: usize = parts[2].parse().unwrap_or(0);
                    if i < self.weights.len() && j < self.weights[i].len() {
                        self.weights[i][j] = parts[3].parse().unwrap_or(0.0);
                    }
                }
                _ => {}
            }
        }
        println!("[trainer] Weights loaded from {}  (accuracy: {:.2}%, avg error: ${:.4})",
            path, self.accuracy_pct, self.avg_dollar_error);
        Ok(())
    }
}

// ─────────────────────────────────────────────
//  CSV parsing
// ─────────────────────────────────────────────

// Reads a CSV file from main.rs and returns a list of StockData rows.
fn parse_csv(path: &str) -> Vec<StockData> {
    let file   = File::open(path).expect("Cannot open CSV file");
    let reader = io::BufReader::new(file);
    let mut data = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("Read error");
        if i == 0 { continue; } // skip header row

        let cols: Vec<&str> = line.splitn(7, ',').collect();
        if cols.len() < 7 { continue; }

        let timestamp = DateTime::parse_from_str(
            &format!("{} +0000", cols[0].trim()),
            "%Y-%m-%d %H:%M:%S %z",
        )
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or(Utc::now());

        data.push(StockData {
            timestamp,
            symbol: cols[1].trim().to_string(),
            open:   cols[2].trim().parse().unwrap_or(0.0),
            high:   cols[3].trim().parse().unwrap_or(0.0),
            low:    cols[4].trim().parse().unwrap_or(0.0),
            close:  cols[5].trim().parse().unwrap_or(0.0),
            volume: cols[6].trim().parse().unwrap_or(0),
        });
    }

    data
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage:   {} <data.csv> [weights.csv]", args[0]);
        eprintln!("Example: {} stock_data.csv weights.csv", args[0]);
        std::process::exit(1);
    }

    let csv_path     = &args[1];
    let weights_path = args.get(2).map(|s| s.as_str()).unwrap_or("weights.csv");

    let data = parse_csv(csv_path);
    println!("[main] Loaded {} rows from {}", data.len(), csv_path);

    if data.is_empty() {
        eprintln!("[main] No data loaded. Check CSV format.");
        std::process::exit(1);
    }

    // Train on all the data.
    // Then predict using the last 120 rows (the most recent 2-hour window).
    // With only 130 rows we can't afford to hold back 20% — we need all of it to train.
    let recent_data = &data[data.len() - 120..]; // last 120 rows for prediction

    let mut trainer = StockTrainer::new();
    trainer.set_start_date(data.first().map(|d| d.timestamp.to_string()).unwrap_or_default());
    trainer.set_end_date(data.last().map(|d| d.timestamp.to_string()).unwrap_or_default());

    trainer.train(&data);
    trainer.save_weights(weights_path).expect("Failed to save weights");

    println!("\n[main] Predicting next 10 minutes of close prices:");
    let predictions = trainer.predict(recent_data);

    println!("\n-- Prediction Summary ---------------------");
    for (i, price) in predictions.iter().enumerate() {
        println!("  Minute +{:>2}: ${:.2}", i + 1, price);
    }
}
