//! Train Cell (plugin).
//!
//! Reads metrics history JSON, computes per-feature statistics and
//! feature-pair correlations, writes a trained anomaly model JSON file.
//! All math is pure Rust — no external ML dependencies.

use crawl_plugin_sdk::exports::crawl::plugin::plugin_api;
use crawl_plugin_sdk::host_tools;
use crawl_plugin_sdk::{emit_event, read_file_string};
use serde::{Deserialize, Serialize};

struct TrainerPlugin;

// ── Input types (from storage export) ───────────────────────────────

#[derive(Deserialize)]
struct MetricsRow {
    cpu_load_1m: f64,
    cpu_load_5m: f64,
    cpu_load_15m: f64,
    mem_total_kb: i64,
    mem_available_kb: i64,
    mem_used_percent: f64,
    uptime_secs: f64,
}

// ── Output types (consumed by inference.rs) ─────────────────────────

#[derive(Serialize)]
struct TrainedModel {
    features: Vec<FeatureStats>,
    correlations: Vec<(String, String, f64)>,
    trained_at: String,
    sample_count: u64,
}

#[derive(Serialize)]
struct FeatureStats {
    name: String,
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
    p5: f64,
    p25: f64,
    p50: f64,
    p75: f64,
    p95: f64,
    iqr: f64,
    lower_fence: f64,
    upper_fence: f64,
    ewma: f64,
    ewma_alpha: f64,
    sample_count: u64,
}

// ── Feature extraction ──────────────────────────────────────────────

const FEATURE_NAMES: &[&str] = &[
    "cpu_load_1m",
    "cpu_load_5m",
    "cpu_load_15m",
    "mem_total_kb",
    "mem_available_kb",
    "mem_used_percent",
    "uptime_secs",
];

fn extract_feature(row: &MetricsRow, idx: usize) -> f64 {
    match idx {
        0 => row.cpu_load_1m,
        1 => row.cpu_load_5m,
        2 => row.cpu_load_15m,
        3 => row.mem_total_kb as f64,
        4 => row.mem_available_kb as f64,
        5 => row.mem_used_percent,
        6 => row.uptime_secs,
        _ => 0.0,
    }
}

// ── Statistical computations ────────────────────────────────────────

fn compute_stats(name: &str, values: &[f64]) -> FeatureStats {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);
    let p5 = percentile(&sorted, 5.0);
    let p25 = percentile(&sorted, 25.0);
    let p50 = percentile(&sorted, 50.0);
    let p75 = percentile(&sorted, 75.0);
    let p95 = percentile(&sorted, 95.0);
    let iqr = p75 - p25;
    let lower_fence = p25 - 1.5 * iqr;
    let upper_fence = p75 + 1.5 * iqr;

    // Compute EWMA (exponentially weighted moving average).
    let alpha = 2.0 / (values.len().min(60) as f64 + 1.0);
    let mut ewma = values.first().copied().unwrap_or(0.0);
    for &v in &values[1..] {
        ewma = alpha * v + (1.0 - alpha) * ewma;
    }

    FeatureStats {
        name: name.to_string(),
        mean,
        stddev,
        min,
        max,
        p5,
        p25,
        p50,
        p75,
        p95,
        iqr,
        lower_fence,
        upper_fence,
        ewma,
        ewma_alpha: alpha,
        sample_count: values.len() as u64,
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = pct / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi || hi >= sorted.len() {
        return sorted[lo.min(sorted.len() - 1)];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Pearson correlation coefficient between two series.
fn pearson_r(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n < 3.0 {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..(n as usize) {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

// ── Plugin implementation ───────────────────────────────────────────

impl plugin_api::Guest for TrainerPlugin {
    fn init() -> Result<(), String> {
        Ok(())
    }

    fn execute(task: plugin_api::Task) -> plugin_api::TaskResult {
        match run_training(&task) {
            Ok(output) => plugin_api::TaskResult::Completed(output),
            Err(e) => plugin_api::TaskResult::Failed(format!("train error: {e}")),
        }
    }

    fn resume(task: plugin_api::Task, _checkpoint: plugin_api::Checkpoint) -> plugin_api::TaskResult {
        Self::execute(task)
    }

    fn describe() -> plugin_api::PluginInfo {
        plugin_api::PluginInfo {
            name: "train".to_string(),
            version: "0.1.0".to_string(),
            description: "Statistical model trainer — computes anomaly baselines from metrics history"
                .to_string(),
            supported_verbs: vec!["train".to_string()],
        }
    }
}

fn run_training(task: &plugin_api::Task) -> Result<String, String> {
    // Parse params to get training data path.
    let params: serde_json::Value =
        serde_json::from_str(&task.params).map_err(|e| format!("bad params: {e}"))?;
    let data_path = params
        .get("training_data_path")
        .and_then(|v| v.as_str())
        .unwrap_or("workspace/training_data.json");

    // Read training data.
    let data_str = read_file_string(data_path, 50 * 1024 * 1024)?;
    let rows: Vec<MetricsRow> =
        serde_json::from_str(&data_str).map_err(|e| format!("bad training data: {e}"))?;

    if rows.len() < 10 {
        return Err(format!(
            "insufficient training data: {} rows (need >= 10)",
            rows.len()
        ));
    }

    // Extract feature columns.
    let num_features = FEATURE_NAMES.len();
    let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(rows.len()); num_features];
    for row in &rows {
        for (i, col) in columns.iter_mut().enumerate() {
            col.push(extract_feature(row, i));
        }
    }

    // Compute per-feature stats.
    let features: Vec<FeatureStats> = (0..num_features)
        .map(|i| compute_stats(FEATURE_NAMES[i], &columns[i]))
        .collect();

    // Compute correlation matrix (upper triangle, skip self-correlation).
    let mut correlations = Vec::new();
    for i in 0..num_features {
        for j in (i + 1)..num_features {
            let r = pearson_r(&columns[i], &columns[j]);
            // Only record non-trivial correlations.
            if r.abs() > 0.3 {
                correlations.push((
                    FEATURE_NAMES[i].to_string(),
                    FEATURE_NAMES[j].to_string(),
                    (r * 1000.0).round() / 1000.0, // Round to 3 decimals.
                ));
            }
        }
    }

    let model = TrainedModel {
        features,
        correlations,
        trained_at: "now".to_string(),
        sample_count: rows.len() as u64,
    };

    // Serialize and write output.
    let model_json =
        serde_json::to_string_pretty(&model).map_err(|e| format!("serialize error: {e}"))?;
    let output_path = "workspace/trained_anomaly_model.json";
    host_tools::write_file(output_path, model_json.as_bytes())?;

    // Emit journal event.
    let _ = emit_event(
        "training_complete",
        &serde_json::json!({
            "samples": rows.len(),
            "features": FEATURE_NAMES.len(),
            "correlations": model.correlations.len(),
            "output_path": output_path,
        }),
    );

    Ok(serde_json::to_string(&serde_json::json!({
        "status": "trained",
        "samples": rows.len(),
        "features": FEATURE_NAMES.len(),
        "correlations_found": model.correlations.len(),
        "output_path": output_path,
    }))
    .unwrap_or_default())
}

crawl_plugin_sdk::export!(TrainerPlugin with_types_in crawl_plugin_sdk);
