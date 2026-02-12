//! Wisdom system — persistent learned constraints extracted from experience.
//!
//! The brain accumulates wisdom entries (constraints, preferences, patterns) from
//! task results. These gate behavior via pre-flight checks and inform the LLM
//! prompt. A maturity system unlocks progressively powerful verbs as the brain
//! demonstrates competence.

use anyhow::{Context, Result};
use chrono::Utc;
use crawl_types::LlmRequest;
use parking_lot::{Mutex, RwLock};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use uuid::Uuid;

use crate::config::WisdomConfig;
use crate::ollama::OllamaClient;

// ── Types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WisdomKind {
    Constraint,
    Preference,
    Pattern,
}

impl fmt::Display for WisdomKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WisdomKind::Constraint => write!(f, "constraint"),
            WisdomKind::Preference => write!(f, "preference"),
            WisdomKind::Pattern => write!(f, "pattern"),
        }
    }
}

impl WisdomKind {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "constraint" => Some(WisdomKind::Constraint),
            "preference" => Some(WisdomKind::Preference),
            "pattern" => Some(WisdomKind::Pattern),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WisdomEntry {
    pub id: String,
    pub kind: WisdomKind,
    pub content: String,
    pub evidence: Vec<String>,
    pub confidence: f64,
    pub times_confirmed: u32,
    pub times_contradicted: u32,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
    pub active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaturityLevel {
    Observer = 0,
    Investigator = 1,
    Caretaker = 2,
    Builder = 3,
}

impl MaturityLevel {
    pub fn name(&self) -> &'static str {
        match self {
            MaturityLevel::Observer => "Observer",
            MaturityLevel::Investigator => "Investigator",
            MaturityLevel::Caretaker => "Caretaker",
            MaturityLevel::Builder => "Builder",
        }
    }

    /// Verbs unlocked at this maturity level (cumulative).
    pub fn unlocked_verbs(&self) -> Vec<String> {
        let mut verbs = vec!["IDENTIFY".to_string(), "MONITOR".to_string()];
        if *self >= MaturityLevel::Investigator {
            verbs.push("PROCURE".to_string());
        }
        if *self >= MaturityLevel::Caretaker {
            verbs.push("MAINTAIN".to_string());
        }
        if *self >= MaturityLevel::Builder {
            verbs.push("BUILD".to_string());
        }
        verbs
    }
}

impl fmt::Display for MaturityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug)]
pub enum PreflightResult {
    Allowed,
    Blocked { reason: String, wisdom_id: String },
    Warned { reason: String, wisdom_id: String },
}

/// Input data for wisdom distillation from scored tasks.
#[allow(dead_code)]
pub struct DistillationInput {
    pub task_id: String,
    pub verb: String,
    pub target: String,
    pub composite: f64,
    pub novelty: f64,
    pub anomaly: f64,
    pub confidence: f64,
    pub actionability: f64,
    pub outcome_summary: String,
}

/// A wisdom entry proposed by the LLM during distillation.
#[derive(Debug, Deserialize)]
pub(crate) struct DistilledWisdom {
    kind: String,
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    confirms_existing: Option<String>,
    #[serde(default)]
    contradicts_existing: Option<String>,
}

// ── WisdomSystem ────────────────────────────────────────────────────

pub struct WisdomSystem {
    db: Arc<Mutex<Connection>>,
    config: WisdomConfig,
    cache: RwLock<Vec<WisdomEntry>>,
}

impl WisdomSystem {
    /// Create and initialize the wisdom system, loading active entries into cache.
    pub fn new(db: Arc<Mutex<Connection>>, config: WisdomConfig) -> Result<Self> {
        let entries = Self::load_active_entries(&db)?;
        Ok(Self {
            db,
            config,
            cache: RwLock::new(entries),
        })
    }

    /// Number of active wisdom entries in cache.
    pub fn active_count(&self) -> usize {
        self.cache.read().len()
    }

    /// Clone active entries for distillation context.
    pub fn active_entries(&self) -> Vec<WisdomEntry> {
        self.cache.read().clone()
    }

    /// Compute the current maturity level based on DB state.
    pub fn compute_maturity(&self) -> Result<MaturityLevel> {
        let thresholds = &self.config.maturity;
        let db = self.db.lock();

        let entity_count: i64 = db
            .query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))
            .unwrap_or(0);

        let wisdom_count = self.cache.read().len() as u32;

        // Count successful PROCURE tasks.
        let procure_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM tasks WHERE status = 'completed' AND verb LIKE '%PROCURE%'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        // Count successful MAINTAIN tasks.
        let maintain_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM tasks WHERE status = 'completed' AND verb LIKE '%MAINTAIN%'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        drop(db);

        // Builder: requires successful MAINTAIN tasks.
        if maintain_count >= thresholds.builder_maintain_count as i64 {
            return Ok(MaturityLevel::Builder);
        }

        // Caretaker: requires successful PROCURE tasks.
        if procure_count >= thresholds.caretaker_procure_count as i64 {
            return Ok(MaturityLevel::Caretaker);
        }

        // Investigator: requires sufficient entities and wisdom entries.
        if entity_count >= thresholds.investigator_entity_count as i64
            && wisdom_count >= thresholds.investigator_wisdom_count
        {
            return Ok(MaturityLevel::Investigator);
        }

        Ok(MaturityLevel::Observer)
    }

    /// Compute effective verbs: intersection of maturity-unlocked verbs and config ceiling.
    pub fn effective_verbs(&self, config_ceiling: &[String]) -> Result<Vec<String>> {
        let maturity = self.compute_maturity()?;
        let unlocked = maturity.unlocked_verbs();

        // Config is the ceiling, maturity is the floor.
        let effective: Vec<String> = unlocked
            .into_iter()
            .filter(|v| {
                config_ceiling
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(v))
            })
            .collect();

        Ok(effective)
    }

    /// Check a proposed task against active wisdom constraints.
    pub fn preflight_check(&self, verb: &str, target: &str, description: &str) -> PreflightResult {
        let cache = self.cache.read();
        let verb_lower = verb.to_lowercase();
        let target_lower = target.to_lowercase();
        let desc_lower = description.to_lowercase();

        for entry in cache.iter() {
            if entry.kind != WisdomKind::Constraint {
                continue;
            }
            if entry.confidence < self.config.preflight_min_confidence {
                continue;
            }

            let content_lower = entry.content.to_lowercase();

            // Match if the constraint mentions the verb AND (target OR description keywords).
            let verb_match = content_lower.contains(&verb_lower);
            let target_match = content_lower.contains(&target_lower)
                || target_lower
                    .split_whitespace()
                    .any(|w| w.len() > 3 && content_lower.contains(w));
            let desc_match = desc_lower
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .any(|w| content_lower.contains(w));

            if verb_match && (target_match || desc_match) {
                if entry.confidence >= self.config.preflight_block_confidence {
                    return PreflightResult::Blocked {
                        reason: entry.content.clone(),
                        wisdom_id: entry.id.clone(),
                    };
                } else {
                    return PreflightResult::Warned {
                        reason: entry.content.clone(),
                        wisdom_id: entry.id.clone(),
                    };
                }
            }
        }

        PreflightResult::Allowed
    }

    /// Run LLM distillation on recent scored tasks to extract wisdom.
    pub(crate) async fn distill(
        &self,
        ollama: &OllamaClient,
        inputs: &[DistillationInput],
        existing: &[WisdomEntry],
    ) -> Result<Vec<DistilledWisdom>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Build task results section.
        let mut task_lines = String::new();
        for input in inputs {
            task_lines.push_str(&format!(
                "- [{}] target={} score={:.2} (N={:.1}/A={:.1}/C={:.1}/Act={:.1}) outcome={}\n",
                input.verb,
                input.target,
                input.composite,
                input.novelty,
                input.anomaly,
                input.confidence,
                input.actionability,
                truncate_str(&input.outcome_summary, 120),
            ));
        }

        // Build existing wisdom section.
        let mut wisdom_lines = String::new();
        for entry in existing {
            wisdom_lines.push_str(&format!(
                "- [{}] id={} (conf={:.2}, confirmed={}x, contradicted={}x): {}\n",
                entry.kind,
                &entry.id[..8.min(entry.id.len())],
                entry.confidence,
                entry.times_confirmed,
                entry.times_contradicted,
                truncate_str(&entry.content, 100),
            ));
        }

        let prompt = format!(
            r#"You are the wisdom extraction engine of crawl-brain. Analyze recent task results and extract reusable rules.

## Recent Task Results (with reward scores)
{task_lines}
## Existing Wisdom Entries
{wisdom_section}
## Instructions
Extract 0-5 wisdom entries. Types:
- "constraint": Things to avoid (wasteful investigations, repeated targets, ineffective verbs)
- "preference": Better approaches discovered (preferred targets, timing, methods)
- "pattern": Correlations found (resource spikes at certain times, related processes, etc.)

For each new entry, check if it confirms or contradicts an existing entry (use the id prefix).

Respond with a JSON array (may be empty):
[{{"kind": "constraint", "content": "...", "tags": ["..."], "confirms_existing": null, "contradicts_existing": null}}]

Respond ONLY with the JSON array."#,
            task_lines = task_lines,
            wisdom_section = if wisdom_lines.is_empty() {
                "(none yet)".to_string()
            } else {
                wisdom_lines
            },
        );

        let request = LlmRequest {
            prompt,
            max_tokens: self.config.max_distillation_tokens,
            temperature: Some(0.3),
            tainted: false,
        };

        let response = ollama
            .query(&request)
            .await
            .context("wisdom distillation LLM query failed")?;

        // Parse response as JSON array of DistilledWisdom.
        let text = response.text.trim();
        let distilled = parse_distilled_array(text);

        Ok(distilled)
    }

    /// Apply distilled wisdom: insert new entries, reinforce confirmed, decay contradicted.
    /// Returns the count of new entries created.
    pub(crate) fn apply_distilled(
        &self,
        distilled: &[DistilledWisdom],
        task_ids: &[String],
    ) -> Result<u32> {
        if distilled.is_empty() {
            return Ok(0);
        }

        let now = Utc::now().to_rfc3339();
        let evidence_json = serde_json::to_string(task_ids)?;
        let mut new_count = 0u32;

        let db = self.db.lock();

        for entry in distilled {
            // Handle confirmations.
            if let Some(ref confirms_id) = entry.confirms_existing {
                self.reinforce_entry_locked(&db, confirms_id, task_ids, &now)?;
            }

            // Handle contradictions.
            if let Some(ref contradicts_id) = entry.contradicts_existing {
                self.decay_entry_locked(&db, contradicts_id, &now)?;
            }

            // Insert new entry if it has valid kind.
            let kind = match WisdomKind::from_str(&entry.kind) {
                Some(k) => k,
                None => continue,
            };

            // Skip if content is too short or empty.
            if entry.content.trim().len() < 10 {
                continue;
            }

            let id = Uuid::now_v7().to_string();
            let tags_json = serde_json::to_string(&entry.tags)?;

            db.execute(
                "INSERT INTO wisdom_entries (id, kind, content, evidence, confidence, tags, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    id,
                    kind.to_string(),
                    entry.content,
                    evidence_json,
                    0.5, // initial confidence
                    tags_json,
                    now,
                    now,
                ],
            )?;

            new_count += 1;
        }

        // Deactivate entries below threshold.
        db.execute(
            "UPDATE wisdom_entries SET active = 0 WHERE active = 1 AND confidence < ?1",
            rusqlite::params![self.config.deactivation_threshold],
        )?;

        drop(db);

        // Refresh cache.
        self.refresh_cache()?;

        Ok(new_count)
    }

    /// Build prompt section with active wisdom grouped by kind.
    pub fn build_prompt_section(&self) -> String {
        let cache = self.cache.read();
        if cache.is_empty() {
            return String::new();
        }

        let max = self.config.max_prompt_entries;
        let mut constraints = Vec::new();
        let mut preferences = Vec::new();
        let mut patterns = Vec::new();

        for entry in cache.iter().take(max) {
            let line = format!(
                "- (conf={:.2}) {}",
                entry.confidence, entry.content,
            );
            match entry.kind {
                WisdomKind::Constraint => constraints.push(line),
                WisdomKind::Preference => preferences.push(line),
                WisdomKind::Pattern => patterns.push(line),
            }
        }

        let mut out = String::from("## Learned Wisdom\n");

        if !constraints.is_empty() {
            out.push_str("### Constraints (things to avoid)\n");
            for line in &constraints {
                out.push_str(line);
                out.push('\n');
            }
        }
        if !preferences.is_empty() {
            out.push_str("### Preferences (better approaches)\n");
            for line in &preferences {
                out.push_str(line);
                out.push('\n');
            }
        }
        if !patterns.is_empty() {
            out.push_str("### Patterns (correlations discovered)\n");
            for line in &patterns {
                out.push_str(line);
                out.push('\n');
            }
        }

        out
    }

    // ── Internals ───────────────────────────────────────────────────

    fn load_active_entries(db: &Arc<Mutex<Connection>>) -> Result<Vec<WisdomEntry>> {
        let db = db.lock();
        let mut stmt = db.prepare(
            "SELECT id, kind, content, evidence, confidence, times_confirmed, times_contradicted, tags, created_at, updated_at
             FROM wisdom_entries
             WHERE active = 1
             ORDER BY confidence DESC",
        )?;

        let entries = stmt
            .query_map([], |row| {
                let kind_str: String = row.get(1)?;
                let evidence_str: String = row.get(3)?;
                let tags_str: String = row.get(7)?;
                Ok(WisdomEntry {
                    id: row.get(0)?,
                    kind: WisdomKind::from_str(&kind_str).unwrap_or(WisdomKind::Pattern),
                    content: row.get(2)?,
                    evidence: serde_json::from_str(&evidence_str).unwrap_or_default(),
                    confidence: row.get(4)?,
                    times_confirmed: row.get::<_, i64>(5)? as u32,
                    times_contradicted: row.get::<_, i64>(6)? as u32,
                    tags: serde_json::from_str(&tags_str).unwrap_or_default(),
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                    active: true,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }

    fn refresh_cache(&self) -> Result<()> {
        let entries = Self::load_active_entries(&self.db)?;
        *self.cache.write() = entries;
        Ok(())
    }

    fn reinforce_entry_locked(
        &self,
        db: &Connection,
        id_prefix: &str,
        task_ids: &[String],
        now: &str,
    ) -> Result<()> {
        // Find entry by ID prefix.
        let entry: Option<(String, f64, String)> = db
            .query_row(
                "SELECT id, confidence, evidence FROM wisdom_entries WHERE id LIKE ?1 AND active = 1",
                rusqlite::params![format!("{id_prefix}%")],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .ok();

        if let Some((id, confidence, evidence_str)) = entry {
            let new_conf = (confidence + self.config.reinforce_delta).min(1.0);

            // Append task IDs to evidence.
            let mut evidence: Vec<String> =
                serde_json::from_str(&evidence_str).unwrap_or_default();
            evidence.extend(task_ids.iter().cloned());
            let evidence_json = serde_json::to_string(&evidence)?;

            db.execute(
                "UPDATE wisdom_entries SET confidence = ?1, times_confirmed = times_confirmed + 1, evidence = ?2, updated_at = ?3 WHERE id = ?4",
                rusqlite::params![new_conf, evidence_json, now, id],
            )?;
        }

        Ok(())
    }

    fn decay_entry_locked(&self, db: &Connection, id_prefix: &str, now: &str) -> Result<()> {
        let entry_id: Option<String> = db
            .query_row(
                "SELECT id FROM wisdom_entries WHERE id LIKE ?1 AND active = 1",
                rusqlite::params![format!("{id_prefix}%")],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = entry_id {
            db.execute(
                "UPDATE wisdom_entries SET confidence = MAX(0.0, confidence - ?1), times_contradicted = times_contradicted + 1, updated_at = ?2 WHERE id = ?3",
                rusqlite::params![self.config.decay_delta, now, id],
            )?;
        }

        Ok(())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn parse_distilled_array(text: &str) -> Vec<DistilledWisdom> {
    let trimmed = text.trim();

    // Try direct parse.
    if let Ok(arr) = serde_json::from_str::<Vec<DistilledWisdom>>(trimmed) {
        return arr;
    }

    // Try extracting from brackets.
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            let slice = &trimmed[start..=end];
            if let Ok(arr) = serde_json::from_str::<Vec<DistilledWisdom>>(slice) {
                return arr;
            }
        }
    }

    Vec::new()
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maturity_level_ordering() {
        assert!(MaturityLevel::Observer < MaturityLevel::Investigator);
        assert!(MaturityLevel::Investigator < MaturityLevel::Caretaker);
        assert!(MaturityLevel::Caretaker < MaturityLevel::Builder);
    }

    #[test]
    fn test_maturity_verbs() {
        let obs = MaturityLevel::Observer.unlocked_verbs();
        assert_eq!(obs, vec!["IDENTIFY", "MONITOR"]);

        let inv = MaturityLevel::Investigator.unlocked_verbs();
        assert_eq!(inv, vec!["IDENTIFY", "MONITOR", "PROCURE"]);

        let ct = MaturityLevel::Caretaker.unlocked_verbs();
        assert_eq!(ct, vec!["IDENTIFY", "MONITOR", "PROCURE", "MAINTAIN"]);

        let bld = MaturityLevel::Builder.unlocked_verbs();
        assert_eq!(bld, vec!["IDENTIFY", "MONITOR", "PROCURE", "MAINTAIN", "BUILD"]);
    }

    #[test]
    fn test_wisdom_kind_display() {
        assert_eq!(WisdomKind::Constraint.to_string(), "constraint");
        assert_eq!(WisdomKind::Preference.to_string(), "preference");
        assert_eq!(WisdomKind::Pattern.to_string(), "pattern");
    }

    #[test]
    fn test_wisdom_kind_from_str() {
        assert_eq!(WisdomKind::from_str("constraint"), Some(WisdomKind::Constraint));
        assert_eq!(WisdomKind::from_str("preference"), Some(WisdomKind::Preference));
        assert_eq!(WisdomKind::from_str("pattern"), Some(WisdomKind::Pattern));
        assert_eq!(WisdomKind::from_str("unknown"), None);
    }

    #[test]
    fn test_parse_distilled_array_direct() {
        let text = r#"[{"kind": "constraint", "content": "avoid re-identifying sshd", "tags": ["sshd"]}]"#;
        let result = parse_distilled_array(text);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].kind, "constraint");
        assert_eq!(result[0].content, "avoid re-identifying sshd");
    }

    #[test]
    fn test_parse_distilled_array_wrapped() {
        let text = r#"Here are the results: [{"kind": "pattern", "content": "GPU temps spike at noon", "tags": []}] end"#;
        let result = parse_distilled_array(text);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].kind, "pattern");
    }

    #[test]
    fn test_parse_distilled_array_empty() {
        let result = parse_distilled_array("[]");
        assert!(result.is_empty());
    }

    #[test]
    fn test_preflight_check_no_entries() {
        let db = Arc::new(Mutex::new(Connection::open_in_memory().unwrap()));
        let config = WisdomConfig::default();
        // Create table so load works.
        db.lock()
            .execute_batch(
                "CREATE TABLE wisdom_entries (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL,
                    evidence TEXT NOT NULL DEFAULT '[]', confidence REAL NOT NULL DEFAULT 0.5,
                    times_confirmed INTEGER NOT NULL DEFAULT 1,
                    times_contradicted INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
                )",
            )
            .unwrap();
        let ws = WisdomSystem::new(db, config).unwrap();
        match ws.preflight_check("IDENTIFY", "sshd", "identify sshd process") {
            PreflightResult::Allowed => {} // expected
            _ => panic!("expected Allowed with no entries"),
        }
    }

    #[test]
    fn test_build_prompt_section_empty() {
        let db = Arc::new(Mutex::new(Connection::open_in_memory().unwrap()));
        let config = WisdomConfig::default();
        db.lock()
            .execute_batch(
                "CREATE TABLE wisdom_entries (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL,
                    evidence TEXT NOT NULL DEFAULT '[]', confidence REAL NOT NULL DEFAULT 0.5,
                    times_confirmed INTEGER NOT NULL DEFAULT 1,
                    times_contradicted INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
                )",
            )
            .unwrap();
        let ws = WisdomSystem::new(db, config).unwrap();
        assert!(ws.build_prompt_section().is_empty());
    }

    #[test]
    fn test_build_prompt_section_with_entries() {
        let db = Arc::new(Mutex::new(Connection::open_in_memory().unwrap()));
        let config = WisdomConfig::default();
        db.lock()
            .execute_batch(
                "CREATE TABLE wisdom_entries (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL,
                    evidence TEXT NOT NULL DEFAULT '[]', confidence REAL NOT NULL DEFAULT 0.5,
                    times_confirmed INTEGER NOT NULL DEFAULT 1,
                    times_contradicted INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
                );
                INSERT INTO wisdom_entries (id, kind, content, confidence, created_at, updated_at)
                    VALUES ('w1', 'constraint', 'avoid repeated IDENTIFY on sshd', 0.7, '2025-01-01', '2025-01-01');
                INSERT INTO wisdom_entries (id, kind, content, confidence, created_at, updated_at)
                    VALUES ('w2', 'preference', 'prefer MONITOR for gpu_temp', 0.6, '2025-01-01', '2025-01-01');",
            )
            .unwrap();
        let ws = WisdomSystem::new(db, config).unwrap();
        let section = ws.build_prompt_section();
        assert!(section.contains("## Learned Wisdom"));
        assert!(section.contains("Constraints"));
        assert!(section.contains("avoid repeated IDENTIFY on sshd"));
        assert!(section.contains("Preferences"));
        assert!(section.contains("prefer MONITOR for gpu_temp"));
    }
}
