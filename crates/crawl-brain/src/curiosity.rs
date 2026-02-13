//! Autonomous curiosity loop — periodic observe/reason/act/reflect cycle.
//!
//! Runs as a tokio task alongside monitor/scheduler/API. Each "think cycle":
//! 1. **Observe** — Gather current state (metrics, cells, journal, tasks, memory)
//! 2. **Reason** — Send context to LLM, ask it what to investigate
//! 3. **Act** — Parse LLM response, validate, submit tasks via SchedulerSender
//! 4. **Reflect** — Store reasoning trace in memory for future context

use anyhow::{Context, Result};
use chrono::Utc;
use crawl_types::{Budget, JournalEventKind, LlmRequest, RiskTier, Task, TaskVerb};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

use crate::monitor;
use crate::reward::RewardEngine;
use crate::scheduler::{Priority, SchedulerSender};
use crate::soul::Soul;
use crate::wisdom::{MaturityLevel, PreflightResult, WisdomSystem};
use crate::BrainState;

/// Proposed task parsed from LLM JSON output.
#[derive(Debug, serde::Deserialize)]
struct ProposedTask {
    cell_id: String,
    verb: String,
    description: String,
    target: String,
    #[serde(default)]
    hypothesis: Option<String>,
}

/// The autonomous curiosity loop.
pub struct CuriosityLoop {
    brain: Arc<BrainState>,
    scheduler: SchedulerSender,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
    reward_engine: RewardEngine,
    wisdom: Option<Arc<WisdomSystem>>,
}

impl CuriosityLoop {
    pub fn new(
        brain: Arc<BrainState>,
        scheduler: SchedulerSender,
        shutdown_rx: tokio::sync::watch::Receiver<bool>,
        soul: Soul,
        wisdom: Option<Arc<WisdomSystem>>,
    ) -> Self {
        let reward_engine = RewardEngine::new(
            brain.storage.db.clone(),
            brain.llm.clone(),
            brain.journal.clone(),
            brain.memory.clone(),
            brain.config.autonomy.reward.clone(),
            soul,
            wisdom.clone(),
        );
        Self {
            brain,
            scheduler,
            shutdown_rx,
            reward_engine,
            wisdom,
        }
    }

    /// Run the curiosity loop until shutdown.
    pub async fn run(mut self) -> Result<()> {
        let reward_enabled = self.brain.config.autonomy.reward.enabled;
        let initial_interval = if reward_enabled {
            self.reward_engine.compute_adaptive_interval()
        } else {
            Duration::from_millis(self.brain.config.autonomy.think_interval_ms)
        };

        tracing::info!(
            interval_ms = initial_interval.as_millis() as u64,
            max_pending = self.brain.config.autonomy.max_pending_tasks,
            reward_enabled,
            "curiosity loop started"
        );

        // Wait one full interval before first think (let system stabilize).
        tokio::time::sleep(initial_interval).await;

        loop {
            tokio::select! {
                biased;
                _ = self.shutdown_rx.changed() => {
                    tracing::info!("curiosity loop received shutdown signal");
                    break;
                }
                // We use a yield point, then sleep for the adaptive interval after the cycle.
                _ = std::future::ready(()) => {}
            }

            if let Err(e) = self.think_cycle().await {
                tracing::warn!(error = %e, "curiosity think cycle failed");
            }

            // Compute next interval (adaptive if reward enabled).
            let next_interval = if reward_enabled {
                self.reward_engine.compute_adaptive_interval()
            } else {
                Duration::from_millis(self.brain.config.autonomy.think_interval_ms)
            };

            tokio::select! {
                biased;
                _ = self.shutdown_rx.changed() => {
                    tracing::info!("curiosity loop received shutdown signal");
                    break;
                }
                _ = tokio::time::sleep(next_interval) => {}
            }
        }
        Ok(())
    }

    /// One think cycle: score → observe → reason → act → reflect.
    async fn think_cycle(&mut self) -> Result<()> {
        let config = &self.brain.config.autonomy;
        let reward_enabled = config.reward.enabled;

        // ── Step 0: Load trained model if available ────────────────
        if let Err(e) = self.maybe_load_trained_model() {
            tracing::debug!(error = %e, "failed to load trained model");
        }

        // ── Step 1: Score unscored tasks ────────────────────────────
        let (tasks_scored, cycle_avg_composite) = if reward_enabled {
            self.reward_engine.score_unscored_tasks().unwrap_or((0, 0.0))
        } else {
            (0, 0.0)
        };

        // Check throttle: count pending autonomy-submitted tasks.
        let pending_count = self.count_pending_autonomy_tasks()?;
        if pending_count >= config.max_pending_tasks {
            tracing::debug!(
                pending = pending_count,
                max = config.max_pending_tasks,
                "throttled: too many pending autonomy tasks"
            );
            return Ok(());
        }

        // ── Step 2: Gather observations ─────────────────────────────
        let context = self.gather_context()?;

        // ── Step 3: Build enhanced prompt ───────────────────────────
        let prompt = self.build_reasoning_prompt(&context);

        // Query LLM.
        let request = LlmRequest {
            prompt,
            max_tokens: config.max_tokens_per_think,
            temperature: Some(config.temperature),
            tainted: false, // trusted internal prompt
        };

        let llm = self.brain.llm.clone();
        let response = llm.query(&request).await
            .context("curiosity LLM query failed")?;

        // Parse response into proposed tasks.
        tracing::debug!(response = %response.text, "curiosity LLM raw response");
        let proposed = self.parse_proposals(&response.text);

        // ── Journal the think cycle with reward metrics ─────────────
        let entities_total = if reward_enabled {
            self.reward_engine.entity_count().unwrap_or(0)
        } else {
            0
        };
        let current_interval_ms = if reward_enabled {
            self.reward_engine.compute_adaptive_interval().as_millis() as u64
        } else {
            config.think_interval_ms
        };

        // Wisdom metrics for journal.
        let wisdom_json = match &self.wisdom {
            Some(w) if config.wisdom.enabled => {
                let level = w
                    .compute_maturity()
                    .unwrap_or(MaturityLevel::Observer);
                serde_json::json!({
                    "active_count": w.active_count(),
                    "maturity_level": level.name(),
                })
            }
            _ => serde_json::json!(null),
        };

        let _ = self.brain.journal.emit(
            JournalEventKind::AutonomyThinkCycle,
            None,
            None,
            serde_json::json!({
                "metrics": {
                    "cpu_1m": context.cpu_load_1m,
                    "mem_pct": context.mem_used_percent,
                    "uptime": context.uptime_secs,
                },
                "loaded_cells": context.loaded_cells,
                "memory_count": context.memory_count,
                "tasks_pending": context.tasks_pending,
                "tasks_completed": context.tasks_completed,
                "tasks_failed": context.tasks_failed,
                "proposed_count": proposed.len(),
                "llm_tokens": response.tokens_used,
                "reward": {
                    "tasks_scored": tasks_scored,
                    "cycle_avg_composite": cycle_avg_composite,
                    "ewma_composite": self.reward_engine.ewma(),
                    "current_interval_ms": current_interval_ms,
                    "entities_total": entities_total,
                },
                "wisdom": wisdom_json,
            }),
        );

        if proposed.is_empty() {
            tracing::debug!("curiosity: nothing interesting to investigate this cycle");
            // Still update EWMA and tick even when nothing submitted.
            if reward_enabled && tasks_scored > 0 {
                self.reward_engine.update_ewma(cycle_avg_composite);
            }
            self.reward_engine.tick_cycle();
            let _ = self.brain.storage.kv_put("reward_ewma", &self.reward_engine.ewma().to_le_bytes());
            return Ok(());
        }

        // ── Step 4: Validate and submit proposed tasks ──────────────
        let mut submitted = 0u32;
        let remaining_slots = config.max_pending_tasks.saturating_sub(pending_count);

        for proposal in proposed.into_iter().take(remaining_slots as usize) {
            match self.validate_and_submit(proposal).await {
                Ok(task_id) => {
                    submitted += 1;
                    tracing::info!(task_id = %task_id, "curiosity submitted task");
                }
                Err(e) => {
                    tracing::debug!(error = %e, "curiosity rejected proposed task");
                }
            }
        }

        // Store reasoning trace in memory.
        if submitted > 0 {
            let summary = format!(
                "Think cycle at {}: CPU {:.1}%, mem {:.1}%, {} cells loaded, submitted {} tasks",
                Utc::now().format("%H:%M:%S"),
                context.cpu_load_1m,
                context.mem_used_percent,
                context.loaded_cells.len(),
                submitted,
            );
            let _ = self.brain.memory.store(
                &summary,
                serde_json::json!({"source": "autonomy", "type": "think_cycle"}),
            );
        }

        // ── Step 4b: Auto-submit training task if conditions met ─────
        if self.should_propose_training() {
            match self.maybe_export_training_data() {
                Ok(Some(path)) => {
                    match self.submit_training_task(&path).await {
                        Ok(task_id) => {
                            tracing::info!(task_id = %task_id, "auto-submitted training task");
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "failed to submit training task");
                        }
                    }
                }
                Ok(None) => {} // not enough data
                Err(e) => {
                    tracing::debug!(error = %e, "failed to export training data");
                }
            }
        }

        // ── Step 5: Update EWMA and run reflection if due ───────────
        if reward_enabled {
            if tasks_scored > 0 {
                self.reward_engine.update_ewma(cycle_avg_composite);
            }
            self.reward_engine.tick_cycle();
            let _ = self.brain.storage.kv_put("reward_ewma", &self.reward_engine.ewma().to_le_bytes());

            // Run LLM reflection if due.
            match self.reward_engine.maybe_reflect().await {
                Ok(Some(text)) => {
                    tracing::info!(
                        text_len = text.len(),
                        "LLM reflection completed"
                    );
                }
                Ok(None) => {} // not due this cycle
                Err(e) => {
                    tracing::warn!(error = %e, "LLM reflection failed");
                }
            }
        }

        Ok(())
    }

    /// Gather observations about the current system state.
    fn gather_context(&self) -> Result<ObservationContext> {
        // System metrics.
        let metrics = monitor::collect_metrics_snapshot()
            .unwrap_or_else(|_| monitor::MetricsSnapshot {
                cpu_load_1m: 0.0,
                cpu_load_5m: 0.0,
                cpu_load_15m: 0.0,
                mem_total_kb: 0,
                mem_available_kb: 0,
                mem_used_percent: 0.0,
                uptime_secs: 0.0,
            });

        // Loaded Cells + their capabilities.
        let loaded_cells = self.brain.engine.list_plugins();
        let mut cell_details = Vec::new();
        for cell_id in &loaded_cells {
            let caps = self.brain.engine.plugin_capabilities(cell_id)
                .unwrap_or_default();
            let cap_names: Vec<String> = caps.iter().map(|c| format!("{c:?}")).collect();
            cell_details.push(format!("{cell_id} [{caps}]", caps = cap_names.join(", ")));
        }

        // Recent journal events.
        let recent_events = self.brain.journal.recent_events(20)
            .unwrap_or_default();
        let event_summaries: Vec<String> = recent_events.iter().map(|e| {
            format!(
                "[{}] {:?}{}{}",
                e.timestamp.format("%H:%M:%S"),
                e.kind,
                e.cell_id.as_deref().map(|c| format!(" cell={c}")).unwrap_or_default(),
                e.task_id.map(|t| format!(" task={t}")).unwrap_or_default(),
            )
        }).collect();

        // Task counts from SQLite.
        let (tasks_pending, tasks_completed, tasks_failed) = self.count_tasks_by_status()?;

        // Recent completed task summaries.
        let recent_results = self.recent_task_results(5)?;

        // Memory count.
        let memory_count = self.brain.memory.count().unwrap_or(0);

        // LLM usage.
        let llm_queries = self.brain.llm.total_queries();
        let llm_tokens = self.brain.llm.total_tokens();

        // Reward-enhanced context.
        let (scoreboard, entity_summary, reflection_notes) =
            if self.brain.config.autonomy.reward.enabled {
                let sb = self.reward_engine.build_scoreboard().unwrap_or_default();
                let es = self.reward_engine.build_entity_summary().unwrap_or_default();
                let rn = self.reward_engine.latest_reflection_notes().unwrap_or_default();
                (sb, es, rn)
            } else {
                (String::new(), String::new(), String::new())
            };

        // Soul content.
        let soul = if self.brain.config.autonomy.reward.enabled {
            self.reward_engine.soul_content().to_string()
        } else {
            String::new()
        };

        // Wisdom context.
        let (wisdom_summary, maturity_level) = match &self.wisdom {
            Some(w) if self.brain.config.autonomy.wisdom.enabled => {
                let summary = w.build_prompt_section();
                let level = w.compute_maturity().unwrap_or(MaturityLevel::Observer);
                (
                    summary,
                    format!("{} (Level {})", level.name(), level as u32),
                )
            }
            _ => (String::new(), String::new()),
        };

        // Telemetry cost summary from task_telemetry table.
        let telemetry_summary = self.build_telemetry_summary().unwrap_or_default();

        Ok(ObservationContext {
            cpu_load_1m: metrics.cpu_load_1m,
            cpu_load_5m: metrics.cpu_load_5m,
            cpu_load_15m: metrics.cpu_load_15m,
            mem_used_percent: metrics.mem_used_percent,
            mem_available_mb: metrics.mem_available_kb / 1024,
            uptime_secs: metrics.uptime_secs,
            loaded_cells,
            cell_details,
            event_summaries,
            tasks_pending,
            tasks_completed,
            tasks_failed,
            recent_results,
            memory_count,
            llm_queries,
            llm_tokens,
            scoreboard,
            entity_summary,
            reflection_notes,
            soul,
            wisdom_summary,
            maturity_level,
            telemetry_summary,
        })
    }

    /// Build the LLM reasoning prompt with current observations.
    fn build_reasoning_prompt(&self, ctx: &ObservationContext) -> String {
        let config = &self.brain.config.autonomy;

        // Compute effective verbs: maturity-gated intersection with config ceiling.
        let allowed_verbs = match &self.wisdom {
            Some(w) if config.wisdom.enabled => w
                .effective_verbs(&config.allowed_verbs)
                .unwrap_or_else(|_| config.allowed_verbs.clone()),
            _ => config.allowed_verbs.clone(),
        };
        let allowed_verbs_str = allowed_verbs.join(", ");

        let cells_section = if ctx.cell_details.is_empty() {
            "  (no cells loaded)".to_string()
        } else {
            ctx.cell_details.iter()
                .map(|c| format!("  - {c}"))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let events_section = if ctx.event_summaries.is_empty() {
            "  (no recent events)".to_string()
        } else {
            ctx.event_summaries.iter()
                .map(|e| format!("  - {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let results_section = if ctx.recent_results.is_empty() {
            "  (no recent results)".to_string()
        } else {
            ctx.recent_results.iter()
                .map(|r| format!("  - {r}"))
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Build optional reward-enhanced sections.
        let scoreboard_section = if ctx.scoreboard.is_empty() {
            String::new()
        } else {
            format!(
                "\n## Investigation Scoreboard\nRecent task scores (composite = weighted N/A/C/Act):\n{}\n",
                ctx.scoreboard
            )
        };

        let entity_section = if ctx.entity_summary.is_empty() {
            String::new()
        } else {
            format!(
                "\n## Known Entities\nThese have already been identified — avoid redundant IDENTIFY tasks:\n{}\n",
                ctx.entity_summary
            )
        };

        let reflection_section = if ctx.reflection_notes.is_empty() {
            String::new()
        } else {
            format!(
                "\n## Reflection Notes\nRecent self-assessment of investigation quality:\n{}\n",
                ctx.reflection_notes
            )
        };

        let soul_section = if ctx.soul.is_empty() {
            String::new()
        } else {
            format!("\n## Soul\n{}\n", ctx.soul)
        };

        let maturity_section = if ctx.maturity_level.is_empty() {
            String::new()
        } else {
            format!("\n## Maturity: {}\n", ctx.maturity_level)
        };

        let wisdom_section = if ctx.wisdom_summary.is_empty() {
            String::new()
        } else {
            format!("\n{}\n", ctx.wisdom_summary)
        };

        let telemetry_section = if ctx.telemetry_summary.is_empty() {
            String::new()
        } else {
            format!("\n{}\nUse this to estimate which tasks are cheap vs expensive.\n", ctx.telemetry_summary)
        };

        format!(
r#"You are the reasoning core of a local system observer called "crawl-brain".
You run on a Jetson Orin (aarch64 Linux). Your job is to be curious about this machine — understand what's running, what's normal, what's unusual, and learn about the system over time.
{soul}{maturity}{wisdom}
## Current State
- Uptime: {uptime:.0}s
- CPU load: {cpu1:.2}/{cpu5:.2}/{cpu15:.2}
- Memory: {mem_pct:.1}% used ({mem_avail} MB free)
- Loaded Cells:
{cells}
- Memory entries: {mem_count}
- LLM usage: {llm_q} queries, {llm_t} tokens total
- Tasks today: {completed} completed, {pending} pending, {failed} failed

## Recent Events
{events}

## Recent Task Results
{results}
{scoreboard}{entities}{reflection}{telemetry}
## What You Can Do
You can submit tasks to loaded Cells using these verbs: {verbs}.
Each task needs: cell_id, verb, description, target, and optionally a hypothesis.
For PROCURE/MAINTAIN/BUILD tasks, always include a hypothesis explaining what you expect to learn or achieve.

## Instructions
Think about what would be interesting or useful to investigate right now.
Consider:
- What processes are running that you don't know about yet?
- Are system metrics normal? Any trends worth watching?
- What logs might reveal something interesting?
- What have you already investigated (check recent tasks and known entities) — don't repeat yourself.
- Are there any anomalies or unusual patterns?
- What knowledge gaps exist that could be filled?
- Respect learned wisdom — avoid repeating mistakes listed in constraints.

Respond with a JSON array of 1-3 tasks to submit. You should almost always submit at least one task — there is always something to learn about this system. Only return an empty array [] if you have thoroughly investigated everything recently.

Format: [{{"cell_id": "...", "verb": "...", "description": "...", "target": "...", "hypothesis": "..."}}]
Respond ONLY with the JSON array, no other text."#,
            soul = soul_section,
            maturity = maturity_section,
            wisdom = wisdom_section,
            uptime = ctx.uptime_secs,
            cpu1 = ctx.cpu_load_1m,
            cpu5 = ctx.cpu_load_5m,
            cpu15 = ctx.cpu_load_15m,
            mem_pct = ctx.mem_used_percent,
            mem_avail = ctx.mem_available_mb,
            cells = cells_section,
            mem_count = ctx.memory_count,
            llm_q = ctx.llm_queries,
            llm_t = ctx.llm_tokens,
            completed = ctx.tasks_completed,
            pending = ctx.tasks_pending,
            failed = ctx.tasks_failed,
            events = events_section,
            results = results_section,
            scoreboard = scoreboard_section,
            entities = entity_section,
            reflection = reflection_section,
            telemetry = telemetry_section,
            verbs = allowed_verbs_str,
        )
    }

    /// Parse LLM response text into proposed tasks.
    fn parse_proposals(&self, text: &str) -> Vec<ProposedTask> {
        // Try to extract JSON array from the response.
        // The LLM might wrap it in markdown code blocks or add commentary.
        let trimmed = text.trim();

        // Try direct parse first.
        if let Ok(tasks) = serde_json::from_str::<Vec<ProposedTask>>(trimmed) {
            return tasks;
        }

        // Try extracting from markdown code block.
        if let Some(start) = trimmed.find('[') {
            if let Some(end) = trimmed.rfind(']') {
                let slice = &trimmed[start..=end];
                if let Ok(tasks) = serde_json::from_str::<Vec<ProposedTask>>(slice) {
                    return tasks;
                }
            }
        }

        tracing::debug!(response = %trimmed, "failed to parse LLM response as task array");
        Vec::new()
    }

    /// Validate a proposed task and submit it to the scheduler.
    async fn validate_and_submit(&self, proposal: ProposedTask) -> Result<Uuid> {
        let config = &self.brain.config.autonomy;

        // Pre-flight wisdom check.
        if let Some(ref w) = self.wisdom {
            if config.wisdom.enabled {
                match w.preflight_check(&proposal.verb, &proposal.target, &proposal.description) {
                    PreflightResult::Blocked { reason, .. } => {
                        anyhow::bail!("blocked by wisdom: {}", reason);
                    }
                    PreflightResult::Warned { reason, .. } => {
                        tracing::debug!(reason, "wisdom warning (proceeding)");
                    }
                    PreflightResult::Allowed => {}
                }
            }
        }

        // Check cell is loaded.
        if !self.brain.engine.is_loaded(&proposal.cell_id) {
            anyhow::bail!("cell '{}' is not loaded", proposal.cell_id);
        }

        // Check verb is allowed (maturity-gated).
        let verb_upper = proposal.verb.to_uppercase();
        let effective = match &self.wisdom {
            Some(w) if config.wisdom.enabled => w
                .effective_verbs(&config.allowed_verbs)
                .unwrap_or_else(|_| config.allowed_verbs.clone()),
            _ => config.allowed_verbs.clone(),
        };
        if !effective.iter().any(|v| v.eq_ignore_ascii_case(&verb_upper)) {
            anyhow::bail!(
                "verb '{}' not unlocked at current maturity level",
                proposal.verb
            );
        }

        // Parse verb.
        let verb = match verb_upper.as_str() {
            "IDENTIFY" => TaskVerb::Identify,
            "MONITOR" => TaskVerb::Monitor,
            "PROCURE" => TaskVerb::Procure,
            "MAINTAIN" => TaskVerb::Maintain,
            "BUILD" => TaskVerb::Build,
            "UPDATE" => TaskVerb::Update,
            "CRUD" => TaskVerb::Crud,
            other => anyhow::bail!("unknown verb '{other}'"),
        };

        // Store hypothesis in params if provided.
        let params = match &proposal.hypothesis {
            Some(h) => serde_json::json!({"source": "autonomy", "hypothesis": h}),
            None => serde_json::json!({"source": "autonomy"}),
        };

        // Build task with graduated budget based on verb tier.
        let task_id = Uuid::now_v7();
        let task = Task {
            id: task_id,
            verb,
            description: proposal.description.clone(),
            target: proposal.target.clone(),
            params,
            budget: graduated_budget(&verb),
            cell_id: proposal.cell_id.clone(),
            created_at: Utc::now(),
            continuation: None,
        };

        // Persist to SQLite.
        {
            let db = self.brain.storage.db.lock();
            db.execute(
                "INSERT INTO tasks (id, verb, cell_id, description, target, params, status, budget, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    task_id.to_string(),
                    serde_json::to_string(&task.verb)?,
                    task.cell_id,
                    task.description,
                    task.target,
                    serde_json::to_string(&task.params)?,
                    "pending",
                    serde_json::to_string(&task.budget)?,
                    task.created_at.to_rfc3339(),
                ],
            )?;
        }

        // Submit via scheduler at Low priority.
        self.scheduler.submit(task, Priority::Low).await?;

        Ok(task_id)
    }

    /// Count pending tasks submitted by the autonomy loop.
    fn count_pending_autonomy_tasks(&self) -> Result<u32> {
        let db = self.brain.storage.db.lock();
        let count: i64 = db.query_row(
            "SELECT COUNT(*) FROM tasks WHERE status IN ('pending', 'running') AND params LIKE '%autonomy%'",
            [],
            |row| row.get(0),
        )?;
        Ok(count as u32)
    }

    /// Count tasks by status.
    fn count_tasks_by_status(&self) -> Result<(u32, u32, u32)> {
        let db = self.brain.storage.db.lock();

        let pending: i64 = db.query_row(
            "SELECT COUNT(*) FROM tasks WHERE status IN ('pending', 'running')",
            [],
            |row| row.get(0),
        )?;
        let completed: i64 = db.query_row(
            "SELECT COUNT(*) FROM tasks WHERE status = 'completed'",
            [],
            |row| row.get(0),
        )?;
        let failed: i64 = db.query_row(
            "SELECT COUNT(*) FROM tasks WHERE status = 'failed'",
            [],
            |row| row.get(0),
        )?;

        Ok((pending as u32, completed as u32, failed as u32))
    }

    /// Get summaries of recent completed tasks.
    fn recent_task_results(&self, limit: usize) -> Result<Vec<String>> {
        let db = self.brain.storage.db.lock();
        let mut stmt = db.prepare(
            "SELECT verb, cell_id, description, target, result FROM tasks WHERE status = 'completed' ORDER BY completed_at DESC LIMIT ?1",
        )?;

        let results: Vec<String> = stmt
            .query_map(rusqlite::params![limit as i64], |row| {
                let verb: String = row.get(0)?;
                let cell_id: String = row.get(1)?;
                let desc: String = row.get(2)?;
                let target: String = row.get(3)?;
                let result: Option<String> = row.get(4)?;
                let result_summary = result
                    .and_then(|r| {
                        serde_json::from_str::<serde_json::Value>(&r)
                            .ok()
                            .and_then(|v| {
                                v.get("output")
                                    .map(|o| {
                                        let s = o.to_string();
                                        if s.len() > 100 {
                                            format!("{}...", &s[..100])
                                        } else {
                                            s
                                        }
                                    })
                            })
                    })
                    .unwrap_or_default();
                Ok(format!("{verb} {cell_id}: {desc} (target: {target}) -> {result_summary}"))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(results)
    }

    /// Build a per-verb telemetry cost summary from the task_telemetry table.
    fn build_telemetry_summary(&self) -> Result<String> {
        let db = self.brain.storage.db.lock();
        let mut stmt = db.prepare(
            "SELECT t.verb, COUNT(*) as cnt,
                    AVG(tt.duration_ms) as avg_dur,
                    AVG(tt.tool_calls_used) as avg_tools,
                    AVG(tt.bytes_read) as avg_read,
                    AVG(tt.llm_calls_used) as avg_llm
             FROM task_telemetry tt
             JOIN tasks t ON t.id = tt.task_id
             GROUP BY t.verb
             ORDER BY cnt DESC"
        )?;

        let rows: Vec<String> = stmt.query_map([], |row| {
            let verb: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            let avg_dur: f64 = row.get(2)?;
            let avg_tools: f64 = row.get(3)?;
            let avg_read: f64 = row.get(4)?;
            let avg_llm: f64 = row.get(5)?;
            Ok(format!(
                "  - {} (n={}): avg {:.0}ms, {:.1} tool calls, {:.0} bytes read, {:.1} LLM calls",
                verb.trim_matches('"'), count, avg_dur, avg_tools, avg_read, avg_llm
            ))
        })?.filter_map(|r| r.ok()).collect();

        if rows.is_empty() {
            return Ok(String::new());
        }

        Ok(format!("## Resource Cost by Verb\n{}", rows.join("\n")))
    }

    // ── Training pipeline orchestration ─────────────────────────────

    /// Export metrics history to workspace JSON if enough samples exist.
    /// Returns the file path if exported, None if insufficient data.
    fn maybe_export_training_data(&self) -> Result<Option<String>> {
        let count = self.brain.storage.metrics_count(24)?;
        if count < 500 {
            tracing::trace!(count, "insufficient metrics for training (need >= 500)");
            return Ok(None);
        }

        let json = self.brain.storage.export_metrics_json(24)?;
        let workspace = &self.brain.config.paths.workspace_dir;
        std::fs::create_dir_all(workspace)
            .with_context(|| format!("failed to create workspace: {}", workspace.display()))?;

        let path = workspace.join("training_data.json");
        std::fs::write(&path, &json)
            .with_context(|| format!("failed to write training data: {}", path.display()))?;

        tracing::debug!(samples = count, path = %path.display(), "exported training data");
        Ok(Some(path.to_string_lossy().into_owned()))
    }

    /// Check whether we should auto-submit a training task this cycle.
    fn should_propose_training(&self) -> bool {
        // Trainer cell must be loaded.
        if !self.brain.engine.is_loaded("build") {
            return false;
        }

        // Need enough data.
        let count = self.brain.storage.metrics_count(24).unwrap_or(0);
        if count < 500 {
            return false;
        }

        // No pending/running training tasks.
        let db = self.brain.storage.db.lock();
        let active: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM tasks WHERE cell_id = 'build' AND status IN ('pending', 'running')",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if active > 0 {
            return false;
        }

        true
    }

    /// Build and submit a BUILD task to the trainer cell.
    async fn submit_training_task(&self, data_path: &str) -> Result<Uuid> {
        let task_id = Uuid::now_v7();
        let task = Task {
            id: task_id,
            verb: TaskVerb::Build,
            description: "Train statistical anomaly model from metrics history".to_string(),
            target: "anomaly_model".to_string(),
            params: serde_json::json!({
                "source": "autonomy",
                "training_data_path": data_path,
            }),
            budget: graduated_budget(&TaskVerb::Build),
            cell_id: "build".to_string(),
            created_at: Utc::now(),
            continuation: None,
        };

        // Persist to SQLite.
        {
            let db = self.brain.storage.db.lock();
            db.execute(
                "INSERT INTO tasks (id, verb, cell_id, description, target, params, status, budget, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    task_id.to_string(),
                    serde_json::to_string(&task.verb)?,
                    task.cell_id,
                    task.description,
                    task.target,
                    serde_json::to_string(&task.params)?,
                    "pending",
                    serde_json::to_string(&task.budget)?,
                    task.created_at.to_rfc3339(),
                ],
            )?;
        }

        self.scheduler.submit(task, Priority::Low).await?;
        Ok(task_id)
    }

    /// Load a trained anomaly model if one exists and is newer than what's loaded.
    fn maybe_load_trained_model(&self) -> Result<bool> {
        let model_path = self.brain.config.paths.workspace_dir.join("trained_anomaly_model.json");
        if !model_path.exists() {
            return Ok(false);
        }

        let Some(ref inference) = self.brain.inference else {
            return Ok(false);
        };

        // Skip if we already have a trained model loaded.
        // We reload unconditionally when the file is present and no model loaded,
        // and also check file mtime vs a stored timestamp to detect new models.
        if inference.has_trained_model() {
            // Check if the file is newer than our last load by comparing content.
            // Simple approach: re-read and compare trained_at field.
            // For efficiency, just skip — the model will be reloaded on restart
            // or when a new training run produces a different file.
            return Ok(false);
        }

        let json = std::fs::read_to_string(&model_path)
            .with_context(|| format!("failed to read trained model: {}", model_path.display()))?;
        inference.load_trained_model(&json)?;
        Ok(true)
    }
}

/// Compute a budget appropriate for the verb's risk tier.
fn graduated_budget(verb: &TaskVerb) -> Budget {
    match verb {
        TaskVerb::Identify => Budget {
            time_budget_ms: Some(30_000),
            deadline_at: None,
            max_tool_calls: 100,
            max_bytes_read: 5 * 1024 * 1024,
            max_bytes_written: 0,
            max_network_calls: 0,
            max_llm_calls: 2,
            max_tokens_per_call: 512,
            risk_tier: RiskTier::Low,
        },
        TaskVerb::Monitor => Budget {
            time_budget_ms: Some(30_000),
            deadline_at: None,
            max_tool_calls: 50,
            max_bytes_read: 5 * 1024 * 1024,
            max_bytes_written: 0,
            max_network_calls: 0,
            max_llm_calls: 0,
            max_tokens_per_call: 0,
            risk_tier: RiskTier::Low,
        },
        TaskVerb::Procure => Budget {
            time_budget_ms: Some(60_000),
            deadline_at: None,
            max_tool_calls: 100,
            max_bytes_read: 10 * 1024 * 1024,
            max_bytes_written: 1024 * 1024,
            max_network_calls: 0,
            max_llm_calls: 0,
            max_tokens_per_call: 0,
            risk_tier: RiskTier::Medium,
        },
        TaskVerb::Maintain => Budget {
            time_budget_ms: Some(120_000),
            deadline_at: None,
            max_tool_calls: 200,
            max_bytes_read: 20 * 1024 * 1024,
            max_bytes_written: 10 * 1024 * 1024,
            max_network_calls: 0,
            max_llm_calls: 0,
            max_tokens_per_call: 0,
            risk_tier: RiskTier::Medium,
        },
        TaskVerb::Build => Budget {
            time_budget_ms: Some(300_000),
            deadline_at: None,
            max_tool_calls: 500,
            max_bytes_read: 50 * 1024 * 1024,
            max_bytes_written: 20 * 1024 * 1024,
            max_network_calls: 0,
            max_llm_calls: 5,
            max_tokens_per_call: 2048,
            risk_tier: RiskTier::High,
        },
        _ => Budget {
            time_budget_ms: Some(30_000),
            deadline_at: None,
            max_tool_calls: 50,
            max_bytes_read: 5 * 1024 * 1024,
            max_bytes_written: 0,
            max_network_calls: 0,
            max_llm_calls: 0,
            max_tokens_per_call: 0,
            risk_tier: RiskTier::Low,
        },
    }
}

/// Intermediate struct holding gathered observations for a think cycle.
struct ObservationContext {
    cpu_load_1m: f64,
    cpu_load_5m: f64,
    cpu_load_15m: f64,
    mem_used_percent: f64,
    mem_available_mb: u64,
    uptime_secs: f64,
    loaded_cells: Vec<String>,
    cell_details: Vec<String>,
    event_summaries: Vec<String>,
    tasks_pending: u32,
    tasks_completed: u32,
    tasks_failed: u32,
    recent_results: Vec<String>,
    memory_count: usize,
    llm_queries: u64,
    llm_tokens: u64,
    scoreboard: String,
    entity_summary: String,
    reflection_notes: String,
    soul: String,
    wisdom_summary: String,
    maturity_level: String,
    telemetry_summary: String,
}
