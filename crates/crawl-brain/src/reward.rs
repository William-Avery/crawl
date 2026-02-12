//! Reward system — scores completed tasks, extracts entities, adapts curiosity interval.
//!
//! The reward engine runs as part of the curiosity loop. Each think cycle:
//! 1. Score any unscored completed tasks on 4 axes (novelty, anomaly, confidence, actionability)
//! 2. Extract entities from IDENTIFY task outputs into the entities table
//! 3. Build a scoreboard + entity summary for the LLM prompt
//! 4. Periodically run LLM reflection to assess investigation quality
//! 5. Compute adaptive think interval based on EWMA of composite scores

use anyhow::{Context, Result};
use chrono::Utc;
use crawl_types::{JournalEventKind, LlmRequest, TaskVerb};
use parking_lot::Mutex;
use rusqlite::Connection;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

use crate::config::RewardConfig;
use crate::journal::Journal;
use crate::memory::MemorySystem;
use crate::ollama::OllamaClient;
use crate::soul::{ReflectionContext, Soul};

/// A scored task row from SQLite.
struct ScoredTask {
    task_id: String,
    verb: String,
    target: String,
    description: String,
    composite: f64,
    novelty: f64,
    anomaly: f64,
    confidence: f64,
    actionability: f64,
}

/// An unscored completed task ready for evaluation.
struct UnscoredTask {
    id: String,
    verb: String,
    target: String,
    #[allow(dead_code)]
    description: String,
    result: Option<String>,
    budget: String,
    created_at: String,
    completed_at: Option<String>,
}

/// Axis scores for a single task.
struct AxisScores {
    novelty: f64,
    anomaly: f64,
    confidence: f64,
    actionability: f64,
}

pub struct RewardEngine {
    db: Arc<Mutex<Connection>>,
    ollama: Arc<OllamaClient>,
    journal: Arc<Journal>,
    memory: Arc<MemorySystem>,
    config: RewardConfig,
    ewma_composite: f64,
    cycle_count: u64,
    soul: Soul,
}

impl RewardEngine {
    pub fn new(
        db: Arc<Mutex<Connection>>,
        ollama: Arc<OllamaClient>,
        journal: Arc<Journal>,
        memory: Arc<MemorySystem>,
        config: RewardConfig,
        soul: Soul,
    ) -> Self {
        Self {
            db,
            ollama,
            journal,
            memory,
            config,
            ewma_composite: 0.5, // avoid cold-start stall
            cycle_count: 0,
            soul,
        }
    }

    // ── Public API ──────────────────────────────────────────────────

    /// Score all unscored completed autonomy tasks. Returns count scored.
    pub fn score_unscored_tasks(&self) -> Result<(u32, f64)> {
        let unscored = self.fetch_unscored_tasks()?;
        if unscored.is_empty() {
            return Ok((0, 0.0));
        }

        let mut total_composite = 0.0;
        let mut count = 0u32;

        for task in &unscored {
            match self.score_task(task) {
                Ok(composite) => {
                    total_composite += composite;
                    count += 1;
                }
                Err(e) => {
                    tracing::debug!(task_id = %task.id, error = %e, "failed to score task");
                }
            }
        }

        let avg = if count > 0 {
            total_composite / count as f64
        } else {
            0.0
        };

        Ok((count, avg))
    }

    /// Score a single task: compute axis scores, persist, extract entities, emit journal event.
    fn score_task(&self, task: &UnscoredTask) -> Result<f64> {
        let output = task
            .result
            .as_deref()
            .and_then(|r| serde_json::from_str::<Value>(r).ok())
            .unwrap_or(Value::Null);

        let verb = parse_verb(&task.verb);
        let axes = self.compute_axes(task, &output, verb);
        let composite = self.weighted_composite(&axes);

        // Extract entities from IDENTIFY tasks.
        let entities_found = if verb == Some(TaskVerb::Identify) {
            self.extract_entities(&task.id, &output).unwrap_or(0)
        } else {
            0
        };

        // Persist to task_rewards.
        let detail = serde_json::json!({
            "verb": task.verb,
            "target": task.target,
        });

        {
            let db = self.db.lock();
            db.execute(
                "INSERT OR REPLACE INTO task_rewards (task_id, novelty, anomaly, confidence, actionability, composite, scoring_method, scoring_detail, entities_found, scored_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                rusqlite::params![
                    task.id,
                    axes.novelty,
                    axes.anomaly,
                    axes.confidence,
                    axes.actionability,
                    composite,
                    "rule",
                    serde_json::to_string(&detail)?,
                    entities_found as i64,
                    Utc::now().to_rfc3339(),
                ],
            )?;
        }

        // Journal event.
        let _ = self.journal.emit(
            JournalEventKind::AutonomyRewardScored,
            Some(Uuid::parse_str(&task.id).unwrap_or(Uuid::nil())),
            None,
            serde_json::json!({
                "novelty": axes.novelty,
                "anomaly": axes.anomaly,
                "confidence": axes.confidence,
                "actionability": axes.actionability,
                "composite": composite,
                "entities_found": entities_found,
            }),
        );

        tracing::debug!(
            task_id = %task.id,
            composite = format!("{composite:.3}"),
            entities = entities_found,
            "scored task"
        );

        Ok(composite)
    }

    /// Build a formatted scoreboard of recent scored tasks for the LLM prompt.
    pub fn build_scoreboard(&self) -> Result<String> {
        let db = self.db.lock();
        let mut stmt = db.prepare(
            "SELECT tr.task_id, t.verb, t.target, t.description, tr.composite, tr.novelty, tr.anomaly, tr.confidence, tr.actionability
             FROM task_rewards tr
             JOIN tasks t ON t.id = tr.task_id
             ORDER BY tr.scored_at DESC
             LIMIT ?1",
        )?;

        let tasks: Vec<ScoredTask> = stmt
            .query_map(rusqlite::params![self.config.scoreboard_size as i64], |row| {
                Ok(ScoredTask {
                    task_id: row.get(0)?,
                    verb: row.get(1)?,
                    target: row.get(2)?,
                    description: row.get(3)?,
                    composite: row.get(4)?,
                    novelty: row.get(5)?,
                    anomaly: row.get(6)?,
                    confidence: row.get(7)?,
                    actionability: row.get(8)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        drop(stmt);
        drop(db);

        if tasks.is_empty() {
            return Ok(String::new());
        }

        let mut out = String::from("Task | Verb | Target | Score | N/A/C/Act\n");
        out.push_str("---|---|---|---|---\n");

        for t in &tasks {
            let short_id = if t.task_id.len() > 8 {
                &t.task_id[..8]
            } else {
                &t.task_id
            };
            let short_desc = if t.description.len() > 40 {
                format!("{}...", &t.description[..40])
            } else {
                t.description.clone()
            };
            out.push_str(&format!(
                "{}.. | {} | {} | {:.2} | {:.1}/{:.1}/{:.1}/{:.1}\n",
                short_id,
                t.verb.trim_matches('"'),
                truncate_str(&t.target, 20),
                t.composite,
                t.novelty,
                t.anomaly,
                t.confidence,
                t.actionability,
            ));
            let _ = short_desc; // description available but scoreboard uses target for brevity
        }

        // Per-verb averages.
        let mut verb_scores: std::collections::HashMap<String, (f64, u32)> =
            std::collections::HashMap::new();
        for t in &tasks {
            let entry = verb_scores
                .entry(t.verb.trim_matches('"').to_string())
                .or_insert((0.0, 0));
            entry.0 += t.composite;
            entry.1 += 1;
        }

        out.push_str("\nVerb averages: ");
        let avgs: Vec<String> = verb_scores
            .iter()
            .map(|(verb, (sum, count))| format!("{}={:.2}", verb, sum / *count as f64))
            .collect();
        out.push_str(&avgs.join(", "));

        Ok(out)
    }

    /// Build a summary of known entities for the LLM prompt.
    pub fn build_entity_summary(&self) -> Result<String> {
        let db = self.db.lock();
        let mut stmt = db.prepare(
            "SELECT kind, name, confidence, last_seen FROM entities ORDER BY last_seen DESC LIMIT 30",
        )?;

        let entities: Vec<(String, String, Option<f64>, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        if entities.is_empty() {
            return Ok(String::new());
        }

        let mut out = String::from("Kind | Name | Confidence | Last Seen\n");
        out.push_str("---|---|---|---\n");

        for (kind, name, conf, last_seen) in &entities {
            let conf_str = conf.map(|c| format!("{c:.2}")).unwrap_or_else(|| "—".into());
            // Show only the time portion of last_seen for brevity.
            let time = if last_seen.len() > 11 {
                &last_seen[11..19.min(last_seen.len())]
            } else {
                last_seen
            };
            out.push_str(&format!(
                "{} | {} | {} | {}\n",
                kind, truncate_str(name, 30), conf_str, time,
            ));
        }

        Ok(out)
    }

    /// Return current soul content for prompt injection.
    pub fn soul_content(&self) -> &str {
        self.soul.content()
    }

    /// Run LLM reflection if due this cycle. Returns reflection text if run.
    pub async fn maybe_reflect(&mut self) -> Result<Option<String>> {
        if self.cycle_count == 0
            || self.cycle_count % self.config.llm_reflect_every_n_cycles as u64 != 0
        {
            return Ok(None);
        }

        let recent_scored = self.fetch_recent_scored(10)?;
        if recent_scored.is_empty() {
            return Ok(None);
        }

        let entity_summary = self.build_entity_summary()?;

        // Build reflection prompt.
        let mut task_lines = String::new();
        for t in &recent_scored {
            let output_preview = self.fetch_task_output_preview(&t.task_id, 500)?;
            task_lines.push_str(&format!(
                "- [{}] {} target={} composite={:.2} (N={:.1} A={:.1} C={:.1} Act={:.1})\n  Output: {}\n",
                t.verb.trim_matches('"'),
                truncate_str(&t.description, 60),
                truncate_str(&t.target, 30),
                t.composite,
                t.novelty,
                t.anomaly,
                t.confidence,
                t.actionability,
                output_preview,
            ));
        }

        let prompt = format!(
            r#"You are reviewing the quality of recent autonomous investigations by the crawl-brain system.

## Recently Scored Tasks
{task_lines}

## Known Entities
{entities}

## Your Task
Analyze the quality of these investigations and respond with a JSON object:
{{
  "most_valuable_tasks": ["task description or target that was most useful"],
  "least_valuable_tasks": ["task description or target that was wasteful"],
  "knowledge_gaps": ["things the system should investigate next"],
  "strategy_notes": "brief recommendation for improving investigation quality",
  "understanding_score": 0.0 to 1.0 rating of overall system understanding
}}

Respond ONLY with the JSON object."#,
            task_lines = task_lines,
            entities = if entity_summary.is_empty() {
                "(none yet)".to_string()
            } else {
                entity_summary.clone()
            },
        );

        let request = LlmRequest {
            prompt,
            max_tokens: self.config.max_reflection_tokens,
            temperature: Some(0.3),
            tainted: false,
        };

        let ollama = self.ollama.clone();
        let response = ollama
            .query(&request)
            .await
            .context("reflection LLM query failed")?;

        // Try to parse the response as JSON.
        let reflection_text = response.text.trim().to_string();
        let reflection_json = extract_json_object(&reflection_text);

        // Store strategy in memory if we got valid JSON.
        if let Some(ref json) = reflection_json {
            // Extract knowledge gaps and strategy for future prompts.
            if let Some(strategy) = json.get("strategy_notes").and_then(|v| v.as_str()) {
                let content = format!("Reflection: {}", strategy);
                let _ = self.memory.store(
                    &content,
                    serde_json::json!({"source": "autonomy", "type": "reflection"}),
                );
            }

            // Apply score adjustments if understanding_score is provided.
            if let Some(understanding) = json.get("understanding_score").and_then(|v| v.as_f64()) {
                tracing::info!(understanding = format!("{understanding:.2}"), "LLM reflection complete");
            }
        }

        // Evolve the soul document with the reflection context.
        let mut soul_updated = false;
        if self.soul.enabled() {
            if let Some(ref json) = reflection_json {
                let reflection_ctx = ReflectionContext {
                    understanding_score: json
                        .get("understanding_score")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5),
                    most_valuable: json
                        .get("most_valuable_tasks")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default(),
                    least_valuable: json
                        .get("least_valuable_tasks")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default(),
                    knowledge_gaps: json
                        .get("knowledge_gaps")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default(),
                    strategy: json
                        .get("strategy_notes")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    entity_summary: entity_summary,
                };

                match self.soul.evolve(&reflection_ctx).await {
                    Ok(()) => {
                        soul_updated = true;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "soul evolution failed");
                    }
                }
            }
        }

        // Journal the reflection.
        let _ = self.journal.emit(
            JournalEventKind::AutonomyReflection,
            None,
            None,
            serde_json::json!({
                "cycle": self.cycle_count,
                "tasks_reviewed": recent_scored.len(),
                "reflection": reflection_json.unwrap_or(Value::String(reflection_text.clone())),
                "llm_tokens": response.tokens_used,
                "soul_updated": soul_updated,
            }),
        );

        Ok(Some(reflection_text))
    }

    /// Get the latest reflection notes from memory for the prompt.
    pub fn latest_reflection_notes(&self) -> Result<String> {
        let entries = self.memory.search("autonomy reflection strategy", 3)?;
        let mut notes = Vec::new();
        for entry in entries {
            if let Some(meta) = entry.metadata.as_object() {
                if meta.get("type").and_then(|v| v.as_str()) == Some("reflection") {
                    notes.push(entry.content);
                }
            }
        }
        Ok(notes.join("\n"))
    }

    /// Compute the adaptive think interval based on EWMA composite score.
    pub fn compute_adaptive_interval(&self) -> Duration {
        if !self.config.enabled {
            // When reward is disabled, return a fixed interval (caller uses config default).
            return Duration::from_millis(60_000);
        }

        let min = self.config.adaptive_min_interval_ms as f64;
        let max = self.config.adaptive_max_interval_ms as f64;
        let clamped = self.ewma_composite.clamp(0.0, 1.0);

        // High EWMA → short interval (exciting, speed up), Low EWMA → long interval (boring, slow down).
        let interval_ms = max - clamped * (max - min);
        Duration::from_millis(interval_ms as u64)
    }

    /// Update EWMA with a new cycle's average composite score.
    pub fn update_ewma(&mut self, cycle_composite: f64) {
        let alpha = self.config.ewma_alpha as f64;
        self.ewma_composite = alpha * cycle_composite + (1.0 - alpha) * self.ewma_composite;
    }

    /// Increment cycle counter.
    pub fn tick_cycle(&mut self) {
        self.cycle_count += 1;
    }

    /// Current EWMA value.
    pub fn ewma(&self) -> f64 {
        self.ewma_composite
    }

    /// Current cycle count.
    #[allow(dead_code)]
    pub fn cycle(&self) -> u64 {
        self.cycle_count
    }

    /// Total entity count.
    pub fn entity_count(&self) -> Result<u64> {
        let db = self.db.lock();
        let count: i64 =
            db.query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;
        Ok(count as u64)
    }

    // ── Scoring Internals ───────────────────────────────────────────

    /// Compute axis scores for a task.
    fn compute_axes(&self, task: &UnscoredTask, output: &Value, verb: Option<TaskVerb>) -> AxisScores {
        AxisScores {
            novelty: self.score_novelty(task, output, verb).clamp(0.0, 1.0),
            anomaly: self.score_anomaly(output, verb).clamp(0.0, 1.0),
            confidence: self.score_confidence(task, output).clamp(0.0, 1.0),
            actionability: self.score_actionability(task, output, verb).clamp(0.0, 1.0),
        }
    }

    /// Novelty: Is this new?
    fn score_novelty(&self, task: &UnscoredTask, output: &Value, _verb: Option<TaskVerb>) -> f64 {
        let mut score = 0.0;

        // Check if target entity is already known.
        let entity_exists = {
            let db = self.db.lock();
            let count: i64 = db
                .query_row(
                    "SELECT COUNT(*) FROM entities WHERE name = ?1",
                    rusqlite::params![task.target],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            count > 0
        };

        if !entity_exists {
            score += 0.5; // target not in entities table
        }

        // Check if same target was investigated recently.
        let recent_same_target = {
            let db = self.db.lock();
            let count: i64 = db
                .query_row(
                    "SELECT COUNT(*) FROM tasks WHERE target = ?1 AND status = 'completed' AND completed_at > datetime('now', '-10 minutes') AND id != ?2",
                    rusqlite::params![task.target, task.id],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            count
        };

        if recent_same_target > 0 {
            score = 0.0; // redundant, recently investigated
        } else {
            // Check if target is stale (investigated but >1hr ago).
            let stale = {
                let db = self.db.lock();
                let count: i64 = db
                    .query_row(
                        "SELECT COUNT(*) FROM tasks WHERE target = ?1 AND status = 'completed' AND completed_at < datetime('now', '-60 minutes')",
                        rusqlite::params![task.target],
                        |row| row.get(0),
                    )
                    .unwrap_or(0);
                count
            };
            if stale > 0 {
                score += 0.3; // stale, worth re-investigating
            }
        }

        // Check if output has new keys compared to recent same-verb tasks.
        if let Some(obj) = output.as_object() {
            if obj.len() > 3 {
                score += 0.2; // structurally rich output
            }
        }

        score
    }

    /// Anomaly: Is this unusual?
    fn score_anomaly(&self, output: &Value, verb: Option<TaskVerb>) -> f64 {
        let mut score = 0.0;

        if let Some(obj) = output.as_object() {
            match verb {
                Some(TaskVerb::Monitor) => {
                    // Look for anomaly indicators.
                    for key in ["anomalies", "warnings", "threshold_exceeded"] {
                        if obj.contains_key(key) {
                            score += 0.3;
                        }
                    }
                    // Check status field.
                    if let Some(status) = obj.get("status").and_then(|v| v.as_str()) {
                        match status {
                            "warning" | "error" | "critical" => score += 0.4,
                            _ => {}
                        }
                    }
                }
                Some(TaskVerb::Identify) => {
                    // Low confidence means something unexpected.
                    if let Some(conf) = obj.get("confidence").and_then(|v| v.as_f64()) {
                        if conf < 0.5 {
                            score += 0.3;
                        }
                    }
                    // Check for unknown/unrecognized indicators.
                    let output_str = output.to_string().to_lowercase();
                    if output_str.contains("unknown") || output_str.contains("unrecognized") {
                        score += 0.4;
                    }
                }
                _ => {}
            }
        }

        score
    }

    /// Confidence: Is the result reliable?
    fn score_confidence(&self, task: &UnscoredTask, output: &Value) -> f64 {
        let mut score = 0.5; // base for completed task

        // Use output confidence field if present.
        if let Some(conf) = output
            .as_object()
            .and_then(|o| o.get("confidence"))
            .and_then(|v| v.as_f64())
        {
            score = conf;
        }

        // Check budget efficiency.
        if let Ok(budget) = serde_json::from_str::<Value>(&task.budget) {
            if let (Some(budget_ms), Some(created), Some(completed)) = (
                budget.get("time_budget_ms").and_then(|v| v.as_u64()),
                chrono::DateTime::parse_from_rfc3339(&task.created_at).ok(),
                task.completed_at
                    .as_deref()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok()),
            ) {
                let actual_ms = (completed - created).num_milliseconds().unsigned_abs();
                if budget_ms > 0 && actual_ms < budget_ms / 2 {
                    score += 0.1; // efficient
                }
            }
        }

        // Rich output.
        if let Some(obj) = output.as_object() {
            if obj.len() > 3 {
                score += 0.1;
            }
            if obj.is_empty() {
                score -= 0.2; // empty output
            }
        } else if output.is_null() {
            score -= 0.2; // null output
        }

        score
    }

    /// Actionability: Is this useful?
    fn score_actionability(&self, task: &UnscoredTask, output: &Value, verb: Option<TaskVerb>) -> f64 {
        let mut score = 0.0;

        if let Some(obj) = output.as_object() {
            // Look for actionable keys.
            for key in ["suggested_actions", "follow_up", "recommendations"] {
                if obj.contains_key(key) {
                    score += 0.3;
                    break;
                }
            }

            for key in ["risk", "severity"] {
                if obj.contains_key(key) {
                    score += 0.2;
                    break;
                }
            }

            match verb {
                Some(TaskVerb::Identify) => {
                    if obj.contains_key("kind") {
                        score += 0.2; // has classification
                    }
                }
                Some(TaskVerb::Monitor) => {
                    if obj.contains_key("baseline") || obj.contains_key("trend") {
                        score += 0.2;
                    }
                }
                _ => {}
            }
        }

        // New entity extracted is always actionable.
        let entity_exists = {
            let db = self.db.lock();
            let count: i64 = db
                .query_row(
                    "SELECT COUNT(*) FROM entities WHERE name = ?1",
                    rusqlite::params![task.target],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            count
        };
        if entity_exists == 0 {
            score += 0.2; // will create new entity
        }

        score
    }

    /// Weighted composite from axis scores.
    fn weighted_composite(&self, axes: &AxisScores) -> f64 {
        let c = &self.config;
        (axes.novelty * c.novelty_weight as f64
            + axes.anomaly * c.anomaly_weight as f64
            + axes.confidence * c.confidence_weight as f64
            + axes.actionability * c.actionability_weight as f64)
            .clamp(0.0, 1.0)
    }

    // ── Entity Extraction ───────────────────────────────────────────

    /// Extract entities from an IDENTIFY task output and upsert into the entities table.
    fn extract_entities(&self, task_id: &str, output: &Value) -> Result<usize> {
        let mut count = 0usize;

        if let Some(obj) = output.as_object() {
            // Pattern 1: Direct entity {kind, name, confidence}
            if let (Some(kind), Some(name)) = (
                obj.get("kind").and_then(|v| v.as_str()),
                obj.get("name").and_then(|v| v.as_str()),
            ) {
                let conf = obj.get("confidence").and_then(|v| v.as_f64());
                if self.upsert_entity(kind, name, conf, obj)? {
                    count += 1;
                }
            }

            // Pattern 2: Items list {items: [{kind, name, ...}]}
            if let Some(items) = obj.get("items").and_then(|v| v.as_array()) {
                for item in items {
                    if let Some(item_obj) = item.as_object() {
                        if let (Some(kind), Some(name)) = (
                            item_obj.get("kind").and_then(|v| v.as_str()),
                            item_obj.get("name").and_then(|v| v.as_str()),
                        ) {
                            let conf = item_obj.get("confidence").and_then(|v| v.as_f64());
                            if self.upsert_entity(kind, name, conf, item_obj)? {
                                count += 1;
                            }
                        }
                    }
                }
            }

            // Pattern 3: Hypothesis {hypothesis, confidence}
            if let Some(hypothesis) = obj.get("hypothesis").and_then(|v| v.as_str()) {
                let conf = obj.get("confidence").and_then(|v| v.as_f64());
                if self.upsert_entity("hypothesis", hypothesis, conf, obj)? {
                    count += 1;
                }
            }
        }

        if count > 0 {
            tracing::debug!(task_id, entities = count, "extracted entities from task output");
        }

        let _ = task_id; // used in debug log above
        Ok(count)
    }

    /// Upsert an entity into the entities table. Returns true if a new entity was created.
    fn upsert_entity(
        &self,
        kind: &str,
        name: &str,
        confidence: Option<f64>,
        metadata: &serde_json::Map<String, Value>,
    ) -> Result<bool> {
        let db = self.db.lock();
        let now = Utc::now().to_rfc3339();

        // Check if entity exists by kind+name.
        let existing: Option<(String, Option<f64>)> = db
            .query_row(
                "SELECT id, confidence FROM entities WHERE kind = ?1 AND name = ?2",
                rusqlite::params![kind, name],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();

        if let Some((id, old_conf)) = existing {
            // Update: bump last_seen, update confidence if higher.
            let new_conf = match (confidence, old_conf) {
                (Some(new), Some(old)) if new > old => Some(new),
                (Some(new), None) => Some(new),
                (_, old) => old,
            };
            db.execute(
                "UPDATE entities SET last_seen = ?1, confidence = ?2 WHERE id = ?3",
                rusqlite::params![now, new_conf, id],
            )?;
            Ok(false)
        } else {
            // Insert new entity.
            let id = Uuid::now_v7().to_string();
            let desc = metadata
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let meta_json = serde_json::to_string(metadata)?;

            db.execute(
                "INSERT INTO entities (id, kind, name, description, confidence, metadata, first_seen, last_seen) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![id, kind, name, desc, confidence, meta_json, now, now],
            )?;

            // Emit EntityDiscovered journal event (drop lock first).
            drop(db);
            let _ = self.journal.emit(
                JournalEventKind::EntityDiscovered,
                None,
                None,
                serde_json::json!({
                    "kind": kind,
                    "name": name,
                    "confidence": confidence,
                }),
            );

            Ok(true)
        }
    }

    // ── SQL Helpers ─────────────────────────────────────────────────

    /// Fetch completed autonomy tasks that have no reward entry yet.
    fn fetch_unscored_tasks(&self) -> Result<Vec<UnscoredTask>> {
        let db = self.db.lock();
        let mut stmt = db.prepare(
            "SELECT t.id, t.verb, t.target, t.description, t.result, t.budget, t.created_at, t.completed_at
             FROM tasks t
             LEFT JOIN task_rewards tr ON tr.task_id = t.id
             WHERE tr.task_id IS NULL
               AND t.status = 'completed'
               AND t.params LIKE '%autonomy%'
             ORDER BY t.completed_at ASC
             LIMIT 20",
        )?;

        let tasks = stmt
            .query_map([], |row| {
                Ok(UnscoredTask {
                    id: row.get(0)?,
                    verb: row.get(1)?,
                    target: row.get(2)?,
                    description: row.get(3)?,
                    result: row.get(4)?,
                    budget: row.get(5)?,
                    created_at: row.get(6)?,
                    completed_at: row.get(7)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(tasks)
    }

    /// Fetch recent scored tasks for reflection.
    fn fetch_recent_scored(&self, limit: usize) -> Result<Vec<ScoredTask>> {
        let db = self.db.lock();
        let mut stmt = db.prepare(
            "SELECT tr.task_id, t.verb, t.target, t.description, tr.composite, tr.novelty, tr.anomaly, tr.confidence, tr.actionability
             FROM task_rewards tr
             JOIN tasks t ON t.id = tr.task_id
             ORDER BY tr.scored_at DESC
             LIMIT ?1",
        )?;

        let tasks = stmt
            .query_map(rusqlite::params![limit as i64], |row| {
                Ok(ScoredTask {
                    task_id: row.get(0)?,
                    verb: row.get(1)?,
                    target: row.get(2)?,
                    description: row.get(3)?,
                    composite: row.get(4)?,
                    novelty: row.get(5)?,
                    anomaly: row.get(6)?,
                    confidence: row.get(7)?,
                    actionability: row.get(8)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(tasks)
    }

    /// Fetch a truncated preview of a task's output for reflection prompts.
    fn fetch_task_output_preview(&self, task_id: &str, max_len: usize) -> Result<String> {
        let db = self.db.lock();
        let result: Option<String> = db
            .query_row(
                "SELECT result FROM tasks WHERE id = ?1",
                rusqlite::params![task_id],
                |row| row.get(0),
            )
            .ok()
            .flatten();

        let preview = result
            .and_then(|r| {
                serde_json::from_str::<Value>(&r)
                    .ok()
                    .map(|v| {
                        let s = v.to_string();
                        if s.len() > max_len {
                            format!("{}...", &s[..max_len])
                        } else {
                            s
                        }
                    })
            })
            .unwrap_or_else(|| "(no output)".to_string());

        Ok(preview)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn parse_verb(s: &str) -> Option<TaskVerb> {
    // Verb is stored as JSON-serialized string, e.g. "\"IDENTIFY\""
    let cleaned = s.trim_matches('"');
    match cleaned {
        "IDENTIFY" => Some(TaskVerb::Identify),
        "MONITOR" => Some(TaskVerb::Monitor),
        "PROCURE" => Some(TaskVerb::Procure),
        "MAINTAIN" => Some(TaskVerb::Maintain),
        "BUILD" => Some(TaskVerb::Build),
        "UPDATE" => Some(TaskVerb::Update),
        "CRUD" => Some(TaskVerb::Crud),
        _ => None,
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}

/// Try to extract a JSON object from text that might have surrounding commentary.
fn extract_json_object(text: &str) -> Option<Value> {
    let trimmed = text.trim();

    // Try direct parse.
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        if v.is_object() {
            return Some(v);
        }
    }

    // Try extracting from braces.
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            let slice = &trimmed[start..=end];
            if let Ok(v) = serde_json::from_str::<Value>(slice) {
                if v.is_object() {
                    return Some(v);
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_verb() {
        assert_eq!(parse_verb("\"IDENTIFY\""), Some(TaskVerb::Identify));
        assert_eq!(parse_verb("MONITOR"), Some(TaskVerb::Monitor));
        assert_eq!(parse_verb("unknown"), None);
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world foo", 5), "hello...");
    }

    #[test]
    fn test_extract_json_object() {
        let text = r#"Here is the result: {"key": "value"} and more text"#;
        let v = extract_json_object(text).unwrap();
        assert_eq!(v.get("key").unwrap().as_str().unwrap(), "value");
    }

    #[test]
    fn test_extract_json_object_direct() {
        let text = r#"{"a": 1, "b": 2}"#;
        let v = extract_json_object(text).unwrap();
        assert_eq!(v.get("a").unwrap().as_i64().unwrap(), 1);
    }

    #[test]
    fn test_weighted_composite() {
        let config = RewardConfig::default();
        let engine_axes = AxisScores {
            novelty: 0.8,
            anomaly: 0.6,
            confidence: 0.7,
            actionability: 0.5,
        };
        let composite = (engine_axes.novelty * config.novelty_weight as f64
            + engine_axes.anomaly * config.anomaly_weight as f64
            + engine_axes.confidence * config.confidence_weight as f64
            + engine_axes.actionability * config.actionability_weight as f64)
            .clamp(0.0, 1.0);
        // 0.8*0.3 + 0.6*0.25 + 0.7*0.2 + 0.5*0.25 = 0.24 + 0.15 + 0.14 + 0.125 = 0.655
        assert!((composite - 0.655).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_interval_extremes() {
        let config = RewardConfig::default();

        // EWMA = 0.0 → max interval (300s)
        let min = config.adaptive_min_interval_ms as f64;
        let max = config.adaptive_max_interval_ms as f64;

        let interval_at_zero = max - 0.0_f64.clamp(0.0, 1.0) * (max - min);
        assert_eq!(interval_at_zero as u64, 300_000);

        // EWMA = 1.0 → min interval (20s)
        let interval_at_one = max - 1.0_f64.clamp(0.0, 1.0) * (max - min);
        assert_eq!(interval_at_one as u64, 20_000);

        // EWMA = 0.5 → midpoint (160s)
        let interval_at_half = max - 0.5_f64.clamp(0.0, 1.0) * (max - min);
        assert_eq!(interval_at_half as u64, 160_000);
    }
}
