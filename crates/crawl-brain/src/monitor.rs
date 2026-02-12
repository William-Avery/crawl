//! System metrics collection and anomaly detection loop.

use anyhow::Result;
use crawl_types::JournalEventKind;
use std::sync::Arc;
use std::time::Duration;

use crate::BrainState;

/// System metrics snapshot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub cpu_load_1m: f64,
    pub cpu_load_5m: f64,
    pub cpu_load_15m: f64,
    pub mem_total_kb: u64,
    pub mem_available_kb: u64,
    pub mem_used_percent: f64,
    pub uptime_secs: f64,
}

/// Run the continuous monitoring loop.
pub async fn run_monitor_loop(brain: Arc<BrainState>) -> Result<()> {
    let interval = Duration::from_millis(brain.config.monitor.metrics_interval_ms);
    let _threshold = brain.config.monitor.anomaly_threshold;

    // Ring buffer for anomaly detection (last 60 samples).
    let mut cpu_history: Vec<f64> = Vec::with_capacity(60);

    tracing::info!(
        interval_ms = brain.config.monitor.metrics_interval_ms,
        "monitor loop started"
    );

    let mut ticker = tokio::time::interval(interval);

    loop {
        ticker.tick().await;

        match collect_metrics() {
            Ok(snapshot) => {
                // Persist to metrics_history table.
                if let Err(e) = brain.storage.insert_metrics(&snapshot) {
                    tracing::debug!("failed to persist metrics: {e}");
                }

                // Track CPU history.
                if cpu_history.len() >= 60 {
                    cpu_history.remove(0);
                }
                cpu_history.push(snapshot.cpu_load_1m);

                // Run anomaly detection if we have enough history.
                if cpu_history.len() >= 10 {
                    if let Some(ref inference) = brain.inference {
                        match inference.infer_anomaly(&cpu_history) {
                            Ok(score) if score.is_anomalous => {
                                tracing::warn!(
                                    score = score.score,
                                    details = %score.details,
                                    "anomaly detected in system metrics"
                                );
                                let _ = brain.journal.emit(
                                    JournalEventKind::AnomalyDetected,
                                    None,
                                    None,
                                    serde_json::json!({
                                        "type": "system_metrics",
                                        "score": score.score,
                                        "details": score.details,
                                        "snapshot": snapshot,
                                    }),
                                );
                            }
                            Ok(_) => {}
                            Err(e) => {
                                tracing::debug!("anomaly detection error: {e}");
                            }
                        }
                    }
                }

                // Periodically log metrics.
                tracing::trace!(
                    cpu_1m = snapshot.cpu_load_1m,
                    mem_pct = snapshot.mem_used_percent,
                    "metrics snapshot"
                );
            }
            Err(e) => {
                tracing::warn!("failed to collect metrics: {e}");
            }
        }
    }
}

/// Collect system metrics from /proc (public for API use).
pub fn collect_metrics_snapshot() -> Result<MetricsSnapshot> {
    collect_metrics()
}

/// Collect system metrics from /proc.
fn collect_metrics() -> Result<MetricsSnapshot> {
    // Read /proc/loadavg.
    let loadavg = std::fs::read_to_string("/proc/loadavg")?;
    let parts: Vec<&str> = loadavg.split_whitespace().collect();
    let cpu_load_1m: f64 = parts.first().unwrap_or(&"0").parse().unwrap_or(0.0);
    let cpu_load_5m: f64 = parts.get(1).unwrap_or(&"0").parse().unwrap_or(0.0);
    let cpu_load_15m: f64 = parts.get(2).unwrap_or(&"0").parse().unwrap_or(0.0);

    // Read /proc/meminfo.
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let mut mem_total_kb = 0u64;
    let mut mem_available_kb = 0u64;
    for line in meminfo.lines() {
        if let Some(val) = line.strip_prefix("MemTotal:") {
            mem_total_kb = parse_meminfo_value(val);
        } else if let Some(val) = line.strip_prefix("MemAvailable:") {
            mem_available_kb = parse_meminfo_value(val);
        }
    }
    let mem_used_percent = if mem_total_kb > 0 {
        (1.0 - mem_available_kb as f64 / mem_total_kb as f64) * 100.0
    } else {
        0.0
    };

    // Read /proc/uptime.
    let uptime_str = std::fs::read_to_string("/proc/uptime")?;
    let uptime_secs: f64 = uptime_str
        .split_whitespace()
        .next()
        .unwrap_or("0")
        .parse()
        .unwrap_or(0.0);

    Ok(MetricsSnapshot {
        cpu_load_1m,
        cpu_load_5m,
        cpu_load_15m,
        mem_total_kb,
        mem_available_kb,
        mem_used_percent,
        uptime_secs,
    })
}

fn parse_meminfo_value(s: &str) -> u64 {
    s.trim()
        .split_whitespace()
        .next()
        .unwrap_or("0")
        .parse()
        .unwrap_or(0)
}
