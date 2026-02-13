//! Maintain Cell (plugin).
//!
//! Housekeeping for the agent ecosystem: surveys workspace, identifies
//! stale files, checks disk usage, and reports recommendations.

use crawl_plugin_sdk::exports::crawl::plugin::plugin_api;
use crawl_plugin_sdk::host_tools;
use crawl_plugin_sdk::emit_event;
use serde::Serialize;

struct MaintainPlugin;

#[derive(Serialize)]
struct MaintainResult {
    scope: String,
    files_scanned: usize,
    stale_files: Vec<StaleFile>,
    disk_usage: String,
    recommendations: Vec<String>,
}

#[derive(Serialize)]
struct StaleFile {
    path: String,
    reason: String,
}

impl plugin_api::Guest for MaintainPlugin {
    fn init() -> Result<(), String> {
        Ok(())
    }

    fn execute(task: plugin_api::Task) -> plugin_api::TaskResult {
        match run_maintain(&task) {
            Ok(output) => plugin_api::TaskResult::Completed(output),
            Err(e) => plugin_api::TaskResult::Failed(format!("maintain error: {e}")),
        }
    }

    fn resume(task: plugin_api::Task, _checkpoint: plugin_api::Checkpoint) -> plugin_api::TaskResult {
        Self::execute(task)
    }

    fn describe() -> plugin_api::PluginInfo {
        plugin_api::PluginInfo {
            name: "maintain".to_string(),
            version: "0.1.0".to_string(),
            description: "Ecosystem housekeeper — workspace cleanup, disk checks, stale file detection"
                .to_string(),
            supported_verbs: vec!["maintain".to_string()],
        }
    }
}

fn run_maintain(task: &plugin_api::Task) -> Result<String, String> {
    let scope = if task.target.is_empty() {
        "full"
    } else {
        &task.target
    };

    let mut stale_files = Vec::new();
    let mut files_scanned = 0usize;
    let mut recommendations = Vec::new();

    // Survey workspace directory.
    let workspace_dir = match scope {
        "storage" => "workspace",
        "workspace" => "workspace",
        _ => "workspace", // full sweep starts with workspace
    };

    if let Ok(entries) = host_tools::list_dir(workspace_dir) {
        files_scanned += entries.len();
        for entry in &entries {
            // Detect stale training data exports.
            if entry.contains("training_data") && entry.ends_with(".json") {
                stale_files.push(StaleFile {
                    path: format!("{workspace_dir}/{entry}"),
                    reason: "training data export — can be regenerated".to_string(),
                });
            }
            // Detect old proposals.
            if entry.contains("proposal") || entry.contains("update_proposals") {
                stale_files.push(StaleFile {
                    path: format!("{workspace_dir}/{entry}"),
                    reason: "old proposal file".to_string(),
                });
            }
            // Detect temp files.
            if entry.ends_with(".tmp") || entry.ends_with(".bak") {
                stale_files.push(StaleFile {
                    path: format!("{workspace_dir}/{entry}"),
                    reason: "temporary file".to_string(),
                });
            }
        }

        // Scan subdirectories one level deep.
        for entry in &entries {
            let sub_path = format!("{workspace_dir}/{entry}");
            if let Ok(sub_entries) = host_tools::list_dir(&sub_path) {
                files_scanned += sub_entries.len();
                for sub_entry in &sub_entries {
                    if sub_entry.ends_with(".tmp") || sub_entry.ends_with(".bak") {
                        stale_files.push(StaleFile {
                            path: format!("{sub_path}/{sub_entry}"),
                            reason: "temporary file in subdirectory".to_string(),
                        });
                    }
                }
            }
        }
    }

    // Get disk usage.
    let disk_usage = match host_tools::exec_command("df", &["-h".to_string()]) {
        Ok(output) if output.exit_code == 0 => {
            // Extract the line for the root filesystem.
            let root_line = output
                .stdout
                .lines()
                .find(|l| l.ends_with('/') || l.contains(" /$"))
                .unwrap_or("unknown");
            root_line.to_string()
        }
        _ => "unavailable".to_string(),
    };

    // Check if disk usage is high.
    if let Some(pct) = extract_disk_percent(&disk_usage) {
        if pct > 85 {
            recommendations.push(format!(
                "Disk usage is {pct}% — consider cleaning stale files"
            ));
        }
    }

    // Add recommendations based on stale file count.
    if stale_files.len() > 5 {
        recommendations.push(format!(
            "{} stale files detected — workspace cleanup recommended",
            stale_files.len()
        ));
    }

    if stale_files.is_empty() && recommendations.is_empty() {
        recommendations.push("Workspace is clean — no action needed".to_string());
    }

    let _ = emit_event(
        "maintain_complete",
        &serde_json::json!({
            "scope": scope,
            "files_scanned": files_scanned,
            "stale_files": stale_files.len(),
            "recommendations": recommendations.len(),
        }),
    );

    let result = MaintainResult {
        scope: scope.to_string(),
        files_scanned,
        stale_files,
        disk_usage,
        recommendations,
    };

    serde_json::to_string(&result).map_err(|e| format!("serialize error: {e}"))
}

/// Try to extract disk usage percentage from a df -h output line.
fn extract_disk_percent(line: &str) -> Option<u32> {
    for part in line.split_whitespace() {
        if part.ends_with('%') {
            return part.trim_end_matches('%').parse().ok();
        }
    }
    None
}

crawl_plugin_sdk::export!(MaintainPlugin with_types_in crawl_plugin_sdk);
