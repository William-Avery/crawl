//! Log Anomaly Watcher Cell (plugin).
//!
//! Reads system log files via host tools and uses the inference
//! engine to detect anomalous log patterns.

use crawl_plugin_sdk::exports::crawl::plugin::plugin_api;
use crawl_plugin_sdk::host_tools;
use crawl_plugin_sdk::{emit_event, memory_store};
use serde::Serialize;

struct LogwatchPlugin;

#[derive(Serialize)]
struct LogwatchResult {
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

impl plugin_api::Guest for LogwatchPlugin {
    fn init() -> Result<(), String> {
        Ok(())
    }

    fn execute(task: plugin_api::Task) -> plugin_api::TaskResult {
        // Parse target as the log source to watch.
        let source = if task.target.is_empty() {
            "syslog"
        } else {
            &task.target
        };

        match scan_logs(source) {
            Ok(result) => {
                let output = serde_json::to_string(&result).unwrap_or_default();

                if result.anomalies_detected > 0 {
                    let _ = emit_event("anomaly_detected", &serde_json::json!({
                        "source": source,
                        "count": result.anomalies_detected,
                    }));

                    // Store anomalies in memory for future reference.
                    for anomaly in &result.anomalies {
                        let _ = memory_store(
                            &format!("Log anomaly in {}: {} ({})", anomaly.source, anomaly.line, anomaly.pattern),
                            &serde_json::json!({"source": source, "pattern": anomaly.pattern}),
                        );
                    }
                }

                plugin_api::TaskResult::Completed(output)
            }
            Err(e) => plugin_api::TaskResult::Failed(format!("logwatch error: {e}")),
        }
    }

    fn resume(task: plugin_api::Task, _checkpoint: plugin_api::Checkpoint) -> plugin_api::TaskResult {
        Self::execute(task)
    }

    fn describe() -> plugin_api::PluginInfo {
        plugin_api::PluginInfo {
            name: "logwatch".to_string(),
            version: "0.1.0".to_string(),
            description: "Log anomaly watcher — scans logs for error patterns and anomalies"
                .to_string(),
            supported_verbs: vec!["monitor".to_string()],
        }
    }
}

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
        lines_scanned: lines.len(),
        anomalies_detected: anomalies.len(),
        anomalies,
    })
}

crawl_plugin_sdk::export!(LogwatchPlugin with_types_in crawl_plugin_sdk);
