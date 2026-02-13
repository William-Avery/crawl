//! Procure Cell (plugin).
//!
//! Collects minimal diagnostic evidence about a target entity.
//! Each evidence item is hashed (SHA-256) for provenance tracking.
//! Results are stored in memory for future reference.

use crawl_plugin_sdk::exports::crawl::plugin::plugin_api;
use crawl_plugin_sdk::host_tools;
use crawl_plugin_sdk::{emit_event, llm_query, memory_store, read_file_string};
use serde::Serialize;

struct ProcurePlugin;

#[derive(Serialize)]
struct ProcureResult {
    target: String,
    evidence: Vec<EvidenceItem>,
    summary: String,
}

#[derive(Serialize)]
struct EvidenceItem {
    source: String,
    content: String,
    content_hash: String,
}

impl plugin_api::Guest for ProcurePlugin {
    fn init() -> Result<(), String> {
        Ok(())
    }

    fn execute(task: plugin_api::Task) -> plugin_api::TaskResult {
        match run_procure(&task) {
            Ok(output) => plugin_api::TaskResult::Completed(output),
            Err(e) => plugin_api::TaskResult::Failed(format!("procure error: {e}")),
        }
    }

    fn resume(task: plugin_api::Task, _checkpoint: plugin_api::Checkpoint) -> plugin_api::TaskResult {
        Self::execute(task)
    }

    fn describe() -> plugin_api::PluginInfo {
        plugin_api::PluginInfo {
            name: "procure".to_string(),
            version: "0.1.0".to_string(),
            description: "Evidence collector — gathers minimal diagnostic data with provenance hashing"
                .to_string(),
            supported_verbs: vec!["procure".to_string()],
        }
    }
}

fn run_procure(task: &plugin_api::Task) -> Result<String, String> {
    let target = &task.target;
    let description = &task.description;
    let mut evidence = Vec::new();

    // Use LLM to plan a collection strategy.
    let plan_prompt = format!(
        r#"You are a diagnostic evidence collector on a Linux system.

Target: {target}
Context: {description}

Plan which commands and files to examine to gather evidence about this target.
Respond with a JSON object:
{{
  "commands": ["command arg1 arg2", ...],
  "files": ["/path/to/file", ...],
  "log_sources": ["syslog", ...]
}}

Keep it minimal — only the most relevant sources. Max 5 commands, 3 files, 2 log sources.
Respond ONLY with the JSON object."#
    );

    let plan_response = llm_query(&plan_prompt, 512);

    // Parse plan or use defaults.
    let (commands, files, log_sources) = match plan_response {
        Ok(resp) => parse_plan(&resp.text),
        Err(_) => default_plan(target),
    };

    // Execute planned commands.
    for cmd_str in &commands {
        let parts: Vec<&str> = cmd_str.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let cmd = parts[0];
        let args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();

        match host_tools::exec_command(cmd, &args) {
            Ok(output) => {
                if output.exit_code == 0 && !output.stdout.is_empty() {
                    let content = truncate(&output.stdout, 4096);
                    let hash = sha256_hex(content.as_bytes());
                    evidence.push(EvidenceItem {
                        source: format!("exec:{cmd_str}"),
                        content,
                        content_hash: hash,
                    });
                }
            }
            Err(_) => {} // command not allowed or failed — skip
        }
    }

    // Read planned files.
    for file_path in &files {
        match read_file_string(file_path, 64 * 1024) {
            Ok(content) => {
                let content = truncate(&content, 4096);
                let hash = sha256_hex(content.as_bytes());
                evidence.push(EvidenceItem {
                    source: format!("file:{file_path}"),
                    content,
                    content_hash: hash,
                });
            }
            Err(_) => {} // file not readable — skip
        }
    }

    // Read planned log sources.
    for log_source in &log_sources {
        match host_tools::read_log(log_source, 30) {
            Ok(lines) => {
                let content = lines.join("\n");
                let content = truncate(&content, 4096);
                let hash = sha256_hex(content.as_bytes());
                evidence.push(EvidenceItem {
                    source: format!("log:{log_source}"),
                    content,
                    content_hash: hash,
                });
            }
            Err(_) => {}
        }
    }

    // Also gather process context.
    if let Ok(processes) = host_tools::list_processes() {
        let target_lower = target.to_lowercase();
        let mut proc_lines = Vec::new();
        for proc in &processes {
            if proc.name.to_lowercase().contains(&target_lower)
                || proc.cmdline.to_lowercase().contains(&target_lower)
            {
                proc_lines.push(format!(
                    "PID={} name={} cmd={} cpu={:.1}% mem={}KB",
                    proc.pid, proc.name, proc.cmdline, proc.cpu_percent, proc.mem_kb
                ));
            }
        }
        if !proc_lines.is_empty() {
            let content = proc_lines.join("\n");
            let hash = sha256_hex(content.as_bytes());
            evidence.push(EvidenceItem {
                source: "process_list".to_string(),
                content,
                content_hash: hash,
            });
        }
    }

    // Build summary.
    let summary = format!(
        "Procured {} evidence items for target '{}'",
        evidence.len(),
        target
    );

    // Store summary in memory.
    let _ = memory_store(
        &format!("Procured evidence for {target}: {} items collected", evidence.len()),
        &serde_json::json!({
            "type": "procure",
            "target": target,
            "evidence_count": evidence.len(),
        }),
    );

    let _ = emit_event(
        "procure_complete",
        &serde_json::json!({
            "target": target,
            "evidence_count": evidence.len(),
        }),
    );

    let result = ProcureResult {
        target: target.to_string(),
        evidence,
        summary,
    };

    serde_json::to_string(&result).map_err(|e| format!("serialize error: {e}"))
}

/// Parse the LLM's collection plan.
fn parse_plan(text: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let trimmed = text.trim();
    // Try to extract JSON from text.
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
        let commands = val
            .get("commands")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let files = val
            .get("files")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let log_sources = val
            .get("log_sources")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        (commands, files, log_sources)
    } else {
        default_plan("")
    }
}

/// Fallback collection plan when LLM is unavailable.
fn default_plan(target: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut commands = vec![
        "ss -tlnp".to_string(),
        "df -h".to_string(),
        "free -m".to_string(),
    ];
    if !target.is_empty() {
        commands.push(format!("ps -eo pid,pcpu,pmem,comm --sort=-pcpu"));
    }
    let files = vec!["/etc/os-release".to_string()];
    let log_sources = vec!["syslog".to_string()];
    (commands, files, log_sources)
}

/// Simple SHA-256 implementation for provenance hashing.
/// Uses the standard bitwise operations on u32 words.
fn sha256_hex(data: &[u8]) -> String {
    // Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Round constants.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    // Pre-processing: pad message.
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block.
    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    format!(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    )
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...[truncated]", &s[..max])
    } else {
        s.to_string()
    }
}

crawl_plugin_sdk::export!(ProcurePlugin with_types_in crawl_plugin_sdk);
