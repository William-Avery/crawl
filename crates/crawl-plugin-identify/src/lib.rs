//! Process/File Identifier Cell (plugin).
//!
//! Uses the LLM (GLM-4 Flash via Ollama) to classify unknown processes,
//! files, services, and other entities.

use crawl_plugin_sdk::exports::crawl::plugin::plugin_api;
use crawl_plugin_sdk::host_tools;
use crawl_plugin_sdk::{llm_query, emit_event, memory_store, memory_search};
use serde::Serialize;

struct IdentifierPlugin;

#[derive(Serialize)]
struct IdentifyResult {
    target: String,
    hypothesis: String,
    confidence: f64,
    evidence: Vec<String>,
    category: String,
    suggested_followup: Vec<String>,
}

impl plugin_api::Guest for IdentifierPlugin {
    fn init() -> Result<(), String> {
        Ok(())
    }

    fn execute(task: plugin_api::Task) -> plugin_api::TaskResult {
        match identify_entity(&task.target, &task.description) {
            Ok(result) => {
                let output = serde_json::to_string(&result).unwrap_or_default();

                // Store identification in memory for future reference.
                let _ = memory_store(
                    &format!("Identified {}: {} (confidence: {:.0}%)",
                        result.target, result.hypothesis, result.confidence * 100.0),
                    &serde_json::json!({
                        "category": result.category,
                        "target": result.target,
                    }),
                );

                let _ = emit_event("task_completed", &serde_json::json!({
                    "verb": "identify",
                    "target": result.target,
                    "hypothesis": result.hypothesis,
                    "confidence": result.confidence,
                }));

                plugin_api::TaskResult::Completed(output)
            }
            Err(e) => plugin_api::TaskResult::Failed(format!("identify error: {e}")),
        }
    }

    fn resume(task: plugin_api::Task, _checkpoint: plugin_api::Checkpoint) -> plugin_api::TaskResult {
        Self::execute(task)
    }

    fn describe() -> plugin_api::PluginInfo {
        plugin_api::PluginInfo {
            name: "identify".to_string(),
            version: "0.1.0".to_string(),
            description: "Entity identifier — uses LLM to classify processes, files, and services"
                .to_string(),
            supported_verbs: vec!["identify".to_string()],
        }
    }
}

/// Infer a category from the target name using keyword heuristics.
/// Used as a fallback when the LLM returns "unknown".
fn infer_category_from_target(target: &str) -> &'static str {
    let t = target.to_lowercase();
    if t.contains("process") || t.contains("pid") || t.contains("daemon") || t.contains("workload") {
        "process"
    } else if t.contains("service") || t.contains("port") || t.contains("listen") || t.contains("server") {
        "service"
    } else if t.contains("config") || t.contains("setting") || t.contains("param") || t.contains("threshold") || t.contains("model") {
        "config"
    } else if t.contains("driver") || t.contains("cuda") || t.contains("gpu") || t.contains("thermal") || t.contains("hardware") || t.contains("memory") {
        "driver"
    } else if t.contains("log") || t.contains("error") || t.contains("warning") || t.contains("event") || t.contains("pattern") {
        "log_pattern"
    } else if t.contains("file") || t.contains("path") || t.contains("directory") {
        "file"
    } else {
        "unknown"
    }
}

fn identify_entity(target: &str, description: &str) -> Result<IdentifyResult, String> {
    // First, check if we already have a memory about this entity.
    let memory_results = memory_search(target, 3).unwrap_or_default();
    if let Some(best) = memory_results.first() {
        if best.similarity > 0.85 {
            // We've seen this before — return cached identification.
            // Try to recover original category from the stored metadata.
            let cached_category = serde_json::from_str::<serde_json::Value>(&best.metadata)
                .ok()
                .and_then(|v| v.get("category").and_then(|c| c.as_str().map(String::from)))
                .unwrap_or_default();
            // Use stored category if meaningful, otherwise infer from target name.
            let category = if cached_category.is_empty() || cached_category == "unknown" || cached_category == "cached" {
                infer_category_from_target(target).to_string()
            } else {
                cached_category
            };
            return Ok(IdentifyResult {
                target: target.to_string(),
                hypothesis: format!("Previously identified: {}", best.content),
                confidence: best.similarity.min(0.95),
                evidence: vec![format!("memory match (similarity: {:.2})", best.similarity)],
                category,
                suggested_followup: vec![],
            });
        }
    }

    // Gather evidence about the target.
    let mut evidence = Vec::new();

    // Try to find it in process list.
    if let Ok(processes) = host_tools::list_processes() {
        for proc in &processes {
            if proc.name.contains(target) || proc.cmdline.contains(target) {
                evidence.push(format!(
                    "Process found: PID={}, name={}, cmdline={}, CPU={:.1}%, MEM={}KB",
                    proc.pid, proc.name, proc.cmdline, proc.cpu_percent, proc.mem_kb
                ));
            }
        }
    }

    // Search recent logs for mentions of the target.
    let target_lower = target.to_lowercase();
    if let Ok(log_lines) = host_tools::read_log("syslog", 50) {
        for line in &log_lines {
            if line.to_lowercase().contains(&target_lower) {
                evidence.push(format!("Log mention: {}", line.trim()));
            }
        }
    }

    // Build the LLM prompt.
    let prompt = format!(
        r#"You are a system administrator analyzing an unknown entity on a Linux machine.

Target: {target}
Context: {description}

Evidence gathered:
{evidence_text}

Based on this information, identify what this entity is. Respond in this exact format:
HYPOTHESIS: <what this entity is>
CONFIDENCE: <0.0 to 1.0>
CATEGORY: <process|service|file|library|driver|config|unknown>
FOLLOWUP: <comma-separated list of suggested follow-up investigations>

Be concise and factual."#,
        evidence_text = if evidence.is_empty() {
            "No process-level evidence found.".to_string()
        } else {
            evidence.join("\n")
        }
    );

    let response = llm_query(&prompt, 512)?;

    // Parse the LLM response.
    let mut hypothesis = String::new();
    let mut confidence = 0.5;
    let mut category = "unknown".to_string();
    let mut suggested_followup = Vec::new();

    for line in response.text.lines() {
        let line = line.trim();
        let upper = line.to_uppercase();
        if let Some(pos) = upper.find("HYPOTHESIS:") {
            hypothesis = line[pos + "HYPOTHESIS:".len()..].trim().to_string();
        } else if let Some(pos) = upper.find("CONFIDENCE:") {
            confidence = line[pos + "CONFIDENCE:".len()..].trim().parse().unwrap_or(0.5);
        } else if let Some(pos) = upper.find("CATEGORY:") {
            category = line[pos + "CATEGORY:".len()..].trim().to_string();
        } else if let Some(pos) = upper.find("FOLLOWUP:") {
            suggested_followup = line[pos + "FOLLOWUP:".len()..].split(',')
                .map(|s| s.trim().to_string()).collect();
        }
    }

    if hypothesis.is_empty() {
        // Fallback: use the raw response as hypothesis.
        hypothesis = response.text.lines().next().unwrap_or("Unknown entity").to_string();
    }

    // If the LLM returned "unknown" category, try keyword heuristic on the target name.
    if category == "unknown" {
        category = infer_category_from_target(target).to_string();
    }

    Ok(IdentifyResult {
        target: target.to_string(),
        hypothesis,
        confidence,
        evidence,
        category,
        suggested_followup,
    })
}

crawl_plugin_sdk::export!(IdentifierPlugin with_types_in crawl_plugin_sdk);
