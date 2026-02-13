//! Monitor Cell (plugin).
//!
//! Reads /proc/loadavg, /proc/meminfo, and process lists via host tools,
//! then scans system logs for anomalous patterns. Returns a combined
//! MONITOR result with system health metrics and log anomalies.

use crawl_plugin_sdk::exports::crawl::plugin::plugin_api;
use crawl_plugin_sdk::host_tools;
use crawl_plugin_sdk::{read_file_string, emit_event, memory_store};
use serde::Serialize;

struct MonitorPlugin;

// ── System metrics types ────────────────────────────────────────────

#[derive(Serialize)]
struct MonitorResult {
    system: SystemMetrics,
    logs: LogwatchResult,
}

#[derive(Serialize)]
struct SystemMetrics {
    load_1m: f64,
    load_5m: f64,
    load_15m: f64,
    mem_total_kb: u64,
    mem_available_kb: u64,
    mem_used_percent: f64,
    process_count: usize,
    top_cpu_processes: Vec<ProcessSummary>,
}

#[derive(Serialize)]
struct ProcessSummary {
    pid: u32,
    name: String,
    cpu_percent: f32,
    mem_kb: u64,
}

// ── Log anomaly types ───────────────────────────────────────────────

#[derive(Serialize)]
struct LogwatchResult {
    source: String,
    lines_scanned: usize,
    anomalies_detected: usize,
    anomalies: Vec<LogAnomaly>,
}

#[derive(Serialize)]
struct LogAnomaly {
    line: String,
    source: String,
    severity: String,
    pattern: String,
}

/// Known error patterns to look for.
const ERROR_PATTERNS: &[(&str, &str)] = &[
    ("OOM", "out_of_memory"),
    ("Out of memory", "out_of_memory"),
    ("oom-kill", "out_of_memory"),
    ("segfault", "segfault"),
    ("panic", "kernel_panic"),
    ("BUG:", "kernel_bug"),
    ("error", "general_error"),
    ("failed", "general_failure"),
    ("CRITICAL", "critical"),
    ("FATAL", "fatal"),
];

// ── Plugin implementation ───────────────────────────────────────────

impl plugin_api::Guest for MonitorPlugin {
    fn init() -> Result<(), String> {
        Ok(())
    }

    fn execute(task: plugin_api::Task) -> plugin_api::TaskResult {
        // Parse target as the log source to watch (default: "syslog").
        let log_source = if task.target.is_empty() {
            "syslog"
        } else {
            &task.target
        };

        let metrics = match collect_system_metrics() {
            Ok(m) => m,
            Err(e) => return plugin_api::TaskResult::Failed(format!("monitor error (metrics): {e}")),
        };

        let logs = match scan_logs(log_source) {
            Ok(l) => l,
            Err(e) => return plugin_api::TaskResult::Failed(format!("monitor error (logs): {e}")),
        };

        // Emit metrics event to journal.
        let _ = emit_event("metrics_snapshot", &serde_json::json!({
            "load_1m": metrics.load_1m,
            "mem_used_percent": metrics.mem_used_percent,
            "process_count": metrics.process_count,
        }));

        // Emit anomaly events and store in memory.
        if logs.anomalies_detected > 0 {
            let _ = emit_event("anomaly_detected", &serde_json::json!({
                "source": log_source,
                "count": logs.anomalies_detected,
            }));

            for anomaly in &logs.anomalies {
                let _ = memory_store(
                    &format!("Log anomaly in {}: {} ({})", anomaly.source, anomaly.line, anomaly.pattern),
                    &serde_json::json!({"source": log_source, "pattern": anomaly.pattern}),
                );
            }
        }

        let result = MonitorResult { system: metrics, logs };
        let output = serde_json::to_string(&result).unwrap_or_default();
        plugin_api::TaskResult::Completed(output)
    }

    fn resume(task: plugin_api::Task, _checkpoint: plugin_api::Checkpoint) -> plugin_api::TaskResult {
        // Monitor tasks are fast — no checkpointing needed.
        Self::execute(task)
    }

    fn describe() -> plugin_api::PluginInfo {
        plugin_api::PluginInfo {
            name: "monitor".to_string(),
            version: "0.1.0".to_string(),
            description: "System monitor — reads CPU, memory, process metrics and scans logs for anomalies"
                .to_string(),
            supported_verbs: vec!["monitor".to_string()],
        }
    }
}

// ── System metrics collection ───────────────────────────────────────

fn collect_system_metrics() -> Result<SystemMetrics, String> {
    // Read /proc/loadavg.
    let loadavg = read_file_string("/proc/loadavg", 256)?;
    let parts: Vec<&str> = loadavg.split_whitespace().collect();
    let load_1m: f64 = parts.first().unwrap_or(&"0").parse().unwrap_or(0.0);
    let load_5m: f64 = parts.get(1).unwrap_or(&"0").parse().unwrap_or(0.0);
    let load_15m: f64 = parts.get(2).unwrap_or(&"0").parse().unwrap_or(0.0);

    // Read /proc/meminfo.
    let meminfo = read_file_string("/proc/meminfo", 4096)?;
    let mut mem_total_kb = 0u64;
    let mut mem_available_kb = 0u64;
    for line in meminfo.lines() {
        if let Some(val) = line.strip_prefix("MemTotal:") {
            mem_total_kb = parse_kb(val);
        } else if let Some(val) = line.strip_prefix("MemAvailable:") {
            mem_available_kb = parse_kb(val);
        }
    }
    let mem_used_percent = if mem_total_kb > 0 {
        (1.0 - mem_available_kb as f64 / mem_total_kb as f64) * 100.0
    } else {
        0.0
    };

    // List processes.
    let processes = host_tools::list_processes().unwrap_or_default();
    let process_count = processes.len();

    // Get top 5 by CPU.
    let mut sorted = processes;
    sorted.sort_by(|a, b| b.cpu_percent.partial_cmp(&a.cpu_percent).unwrap_or(std::cmp::Ordering::Equal));
    let top_cpu_processes: Vec<ProcessSummary> = sorted
        .into_iter()
        .take(5)
        .map(|p| ProcessSummary {
            pid: p.pid,
            name: p.name,
            cpu_percent: p.cpu_percent,
            mem_kb: p.mem_kb,
        })
        .collect();

    Ok(SystemMetrics {
        load_1m,
        load_5m,
        load_15m,
        mem_total_kb,
        mem_available_kb,
        mem_used_percent,
        process_count,
        top_cpu_processes,
    })
}

fn parse_kb(s: &str) -> u64 {
    s.trim()
        .split_whitespace()
        .next()
        .unwrap_or("0")
        .parse()
        .unwrap_or(0)
}

// ── Log scanning ────────────────────────────────────────────────────

fn scan_logs(source: &str) -> Result<LogwatchResult, String> {
    let lines = host_tools::read_log(source, 200)?;
    let mut anomalies = Vec::new();

    for line in &lines {
        let lower = line.to_lowercase();
        for (pattern, category) in ERROR_PATTERNS {
            if lower.contains(&pattern.to_lowercase()) {
                let severity = match *category {
                    "out_of_memory" | "kernel_panic" | "fatal" | "critical" => "high",
                    "segfault" | "kernel_bug" => "high",
                    _ => "medium",
                };
                anomalies.push(LogAnomaly {
                    line: line.clone(),
                    source: source.to_string(),
                    severity: severity.to_string(),
                    pattern: category.to_string(),
                });
                break; // One match per line.
            }
        }
    }

    Ok(LogwatchResult {
        source: source.to_string(),
        lines_scanned: lines.len(),
        anomalies_detected: anomalies.len(),
        anomalies,
    })
}

crawl_plugin_sdk::export!(MonitorPlugin with_types_in crawl_plugin_sdk);
