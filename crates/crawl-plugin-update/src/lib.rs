//! Update Cell (plugin).
//!
//! Proposes modifications to existing files or skills by generating
//! unified diffs via LLM analysis. Produces UpdateProposal-compatible
//! JSON for the brain's update pipeline.

use crawl_plugin_sdk::exports::crawl::plugin::plugin_api;
use crawl_plugin_sdk::host_tools;
use crawl_plugin_sdk::{emit_event, llm_query, memory_search, read_file_string};
use serde::Serialize;

struct UpdatePlugin;

#[derive(Serialize)]
struct UpdateResult {
    proposal_id: String,
    description: String,
    diff: String,
    affected_files: Vec<String>,
    risk_assessment: String,
    rollback_plan: String,
    proposal_path: String,
}

impl plugin_api::Guest for UpdatePlugin {
    fn init() -> Result<(), String> {
        Ok(())
    }

    fn execute(task: plugin_api::Task) -> plugin_api::TaskResult {
        match run_update(&task) {
            Ok(output) => plugin_api::TaskResult::Completed(output),
            Err(e) => plugin_api::TaskResult::Failed(format!("update error: {e}")),
        }
    }

    fn resume(task: plugin_api::Task, _checkpoint: plugin_api::Checkpoint) -> plugin_api::TaskResult {
        Self::execute(task)
    }

    fn describe() -> plugin_api::PluginInfo {
        plugin_api::PluginInfo {
            name: "update".to_string(),
            version: "0.1.0".to_string(),
            description: "Skill updater — proposes file modifications as unified diffs with risk assessment"
                .to_string(),
            supported_verbs: vec!["update".to_string()],
        }
    }
}

fn run_update(task: &plugin_api::Task) -> Result<String, String> {
    let target_file = &task.target;
    let description = &task.description;

    // Read the target file.
    let file_content = read_file_string(target_file, 1024 * 1024)?;

    // Search memory for relevant context.
    let memory_context = memory_search(target_file, 3)
        .unwrap_or_default()
        .into_iter()
        .map(|m| m.content)
        .collect::<Vec<_>>()
        .join("\n");

    // Use LLM to analyze the file and generate proposed changes.
    let prompt = format!(
        r#"You are a code modification assistant. Analyze this file and propose changes.

## Target File: {target_file}
```
{file_content_preview}
```

## Requested Change
{description}

## Relevant Context
{memory_ctx}

## Instructions
Generate a unified diff (--- a/file, +++ b/file format) showing the proposed changes.
Also assess the risk and provide a rollback plan.

Respond with a JSON object:
{{
  "diff": "--- a/{target_file}\n+++ b/{target_file}\n@@ ... @@\n...",
  "risk_assessment": "low|medium|high — brief explanation",
  "rollback_plan": "how to revert this change"
}}

Respond ONLY with the JSON object."#,
        file_content_preview = truncate(&file_content, 3000),
        memory_ctx = if memory_context.is_empty() {
            "(no prior context)".to_string()
        } else {
            truncate(&memory_context, 500)
        },
    );

    let response = llm_query(&prompt, 2048)?;

    // Parse the LLM response.
    let (diff, risk_assessment, rollback_plan) = parse_update_response(&response.text);

    if diff.is_empty() {
        return Err("LLM did not produce a valid diff".to_string());
    }

    // Generate a proposal ID from the task ID.
    let proposal_id = format!("prop-{}", &task.id[..8.min(task.id.len())]);

    // Write proposal JSON to workspace.
    let proposal_path = format!("workspace/update_proposals/{proposal_id}.json");

    // Ensure the proposals directory exists by writing the file.
    let proposal_json = serde_json::json!({
        "proposal_id": proposal_id,
        "target_file": target_file,
        "description": description,
        "diff": diff,
        "risk_assessment": risk_assessment,
        "rollback_plan": rollback_plan,
        "status": "pending_review",
    });
    let proposal_bytes =
        serde_json::to_string_pretty(&proposal_json).map_err(|e| format!("serialize: {e}"))?;

    // Try to create proposals directory via list_dir (to check existence).
    let _ = host_tools::list_dir("workspace/update_proposals");
    host_tools::write_file(&proposal_path, proposal_bytes.as_bytes())?;

    let _ = emit_event(
        "update_proposed",
        &serde_json::json!({
            "proposal_id": proposal_id,
            "target_file": target_file,
            "risk": risk_assessment,
        }),
    );

    let result = UpdateResult {
        proposal_id,
        description: description.to_string(),
        diff,
        affected_files: vec![target_file.to_string()],
        risk_assessment,
        rollback_plan,
        proposal_path,
    };

    serde_json::to_string(&result).map_err(|e| format!("serialize error: {e}"))
}

/// Parse the LLM's update response JSON.
fn parse_update_response(text: &str) -> (String, String, String) {
    let trimmed = text.trim();

    // Try to extract JSON object.
    let json_str = if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
        let diff = val
            .get("diff")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let risk = val
            .get("risk_assessment")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let rollback = val
            .get("rollback_plan")
            .and_then(|v| v.as_str())
            .unwrap_or("revert file to previous version")
            .to_string();
        (diff, risk, rollback)
    } else {
        (String::new(), "unknown".to_string(), "revert file".to_string())
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...[truncated]", &s[..max])
    } else {
        s.to_string()
    }
}

crawl_plugin_sdk::export!(UpdatePlugin with_types_in crawl_plugin_sdk);
