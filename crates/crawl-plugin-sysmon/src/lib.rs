//! System Monitor Cell (plugin).
//!
//! Reads /proc/loadavg, /proc/meminfo, and process lists via host tools.
//! Returns structured MONITOR results with system health metrics.

use crawl_plugin_sdk::exports::crawl::plugin::plugin_api;
use crawl_plugin_sdk::host_tools;
use crawl_plugin_sdk::{read_file_string, emit_event};
use serde::Serialize;

struct SysmonPlugin;

/// System metrics snapshot produced by this plugin.
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

impl plugin_api::Guest for SysmonPlugin {
    fn init() -> Result<(), String> {
        Ok(())
    }

    fn execute(_task: plugin_api::Task) -> plugin_api::TaskResult {
        match collect_system_metrics() {
            Ok(metrics) => {
                let output = serde_json::to_string(&metrics).unwrap_or_default();

                // Emit metrics event to journal.
                let _ = emit_event("metrics_snapshot", &serde_json::json!({
                    "load_1m": metrics.load_1m,
                    "mem_used_percent": metrics.mem_used_percent,
                    "process_count": metrics.process_count,
                }));

                plugin_api::TaskResult::Completed(output)
            }
            Err(e) => plugin_api::TaskResult::Failed(format!("sysmon error: {e}")),
        }
    }

    fn resume(task: plugin_api::Task, _checkpoint: plugin_api::Checkpoint) -> plugin_api::TaskResult {
        // Sysmon doesn't need checkpointing — tasks are fast.
        Self::execute(task)
    }

    fn describe() -> plugin_api::PluginInfo {
        plugin_api::PluginInfo {
            name: "sysmon".to_string(),
            version: "0.1.0".to_string(),
            description: "System monitor — reads CPU, memory, process metrics from /proc"
                .to_string(),
            supported_verbs: vec!["monitor".to_string()],
        }
    }
}

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

crawl_plugin_sdk::export!(SysmonPlugin with_types_in crawl_plugin_sdk);
