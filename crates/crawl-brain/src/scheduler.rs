//! Task scheduling, priority queues, and budget tracking.
#![allow(unused)]

use anyhow::Result;
use crawl_types::{Budget, Checkpoint, JournalEventKind, RiskTier, Task, TaskResult, TaskVerb};
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::engine::{wit_task_types, SubsystemRefs};
use crate::BrainState;

/// Task priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Normal,
    High,
}

/// A queued task with its priority.
struct QueuedTask {
    task: Task,
    priority: Priority,
}

/// The task scheduler.
pub struct Scheduler {
    brain: Arc<BrainState>,
    high_rx: mpsc::Receiver<QueuedTask>,
    normal_rx: mpsc::Receiver<QueuedTask>,
    low_rx: mpsc::Receiver<QueuedTask>,
    high_tx: mpsc::Sender<QueuedTask>,
    normal_tx: mpsc::Sender<QueuedTask>,
    low_tx: mpsc::Sender<QueuedTask>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
}

impl Scheduler {
    pub fn new(brain: Arc<BrainState>, shutdown_rx: tokio::sync::watch::Receiver<bool>) -> Self {
        let (high_tx, high_rx) = mpsc::channel(64);
        let (normal_tx, normal_rx) = mpsc::channel(256);
        let (low_tx, low_rx) = mpsc::channel(256);

        Self {
            brain,
            high_rx,
            normal_rx,
            low_rx,
            high_tx,
            normal_tx,
            low_tx,
            shutdown_rx,
        }
    }

    /// Submit a task to the scheduler.
    pub async fn submit(&self, task: Task, priority: Priority) -> Result<()> {
        let queued = QueuedTask { task, priority };
        let tx = match priority {
            Priority::High => &self.high_tx,
            Priority::Normal => &self.normal_tx,
            Priority::Low => &self.low_tx,
        };
        tx.send(queued)
            .await
            .map_err(|_| anyhow::anyhow!("scheduler channel closed"))?;
        Ok(())
    }

    /// Get a sender handle for submitting tasks externally.
    pub fn sender(&self) -> SchedulerSender {
        SchedulerSender {
            high_tx: self.high_tx.clone(),
            normal_tx: self.normal_tx.clone(),
            low_tx: self.low_tx.clone(),
        }
    }

    /// Run the scheduler loop, dispatching tasks to Cells.
    pub async fn run(mut self) -> Result<()> {
        tracing::info!("scheduler started");

        loop {
            // Biased select: shutdown first, then high priority.
            let queued = tokio::select! {
                biased;
                _ = self.shutdown_rx.changed() => {
                    tracing::info!("scheduler received shutdown signal, draining");
                    break;
                }
                Some(task) = self.high_rx.recv() => task,
                Some(task) = self.normal_rx.recv() => task,
                Some(task) = self.low_rx.recv() => task,
                else => {
                    tracing::info!("all scheduler channels closed");
                    break;
                }
            };

            let task = queued.task;
            let brain = self.brain.clone();

            // Spawn task execution with panic catching.
            tokio::spawn(async move {
                let task_id = task.id;
                let cell_id = task.cell_id.clone();
                let verb = task.verb;

                tracing::info!(
                    task_id = %task_id,
                    cell_id = %cell_id,
                    verb = ?verb,
                    "dispatching task"
                );

                // Mark task as running in SQLite.
                {
                    let db = brain.storage.db.lock();
                    let _ = db.execute(
                        "UPDATE tasks SET status = 'running', started_at = ?1 WHERE id = ?2",
                        rusqlite::params![chrono::Utc::now().to_rfc3339(), task_id.to_string()],
                    );
                }

                let _ = brain.journal.emit(
                    JournalEventKind::TaskStarted,
                    Some(task_id),
                    Some(cell_id.clone()),
                    serde_json::json!({
                        "verb": verb,
                        "target": task.target,
                    }),
                );

                // Execute the task with timeout.
                let timeout_ms = task.budget.time_budget_ms.unwrap_or(30_000);
                tracing::debug!(task_id = %task_id, timeout_ms, "starting task execution");
                let result = tokio::time::timeout(
                    std::time::Duration::from_millis(timeout_ms),
                    execute_task(&brain, &task),
                )
                .await;
                tracing::debug!(task_id = %task_id, "task execution returned");

                match result {
                    Ok(Ok(task_result)) => {
                        let event_kind = match &task_result {
                            TaskResult::Completed { .. } => JournalEventKind::TaskCompleted,
                            TaskResult::Checkpointed { .. } => JournalEventKind::TaskCheckpointed,
                            TaskResult::Failed { .. } => JournalEventKind::TaskFailed,
                        };
                        // Persist final status to SQLite.
                        persist_task_result(&brain, task_id, &task_result);
                        let _ = brain.journal.emit(
                            event_kind,
                            Some(task_id),
                            Some(cell_id),
                            serde_json::to_value(&task_result).unwrap_or_default(),
                        );
                    }
                    Ok(Err(e)) => {
                        tracing::error!(task_id = %task_id, error = %e, "task execution failed");
                        let failed = TaskResult::Failed {
                            task_id,
                            error: e.to_string(),
                            retryable: true,
                        };
                        persist_task_result(&brain, task_id, &failed);
                        let _ = brain.journal.emit(
                            JournalEventKind::TaskFailed,
                            Some(task_id),
                            Some(cell_id),
                            serde_json::json!({"error": e.to_string()}),
                        );
                    }
                    Err(_) => {
                        tracing::warn!(task_id = %task_id, "task timed out, checkpointing");
                        let checkpoint = Checkpoint {
                            continuation_id: Uuid::now_v7(),
                            cursor: String::new(),
                            state_blob_ref: None,
                            completed_steps: vec![],
                            findings: vec![],
                            current_stage: "timed_out".into(),
                            continuation_plan: "resume from last known state".into(),
                            remaining_estimate_ms: None,
                        };
                        let checkpointed = TaskResult::Checkpointed {
                            task_id,
                            checkpoint: checkpoint.clone(),
                            partial_output: None,
                        };
                        persist_task_result(&brain, task_id, &checkpointed);
                        let _ = brain.journal.emit(
                            JournalEventKind::TaskCheckpointed,
                            Some(task_id),
                            Some(cell_id),
                            serde_json::to_value(&checkpoint).unwrap_or_default(),
                        );
                    }
                }
            });
        }

        Ok(())
    }
}

/// Handle for submitting tasks to the scheduler.
#[derive(Clone)]
pub struct SchedulerSender {
    high_tx: mpsc::Sender<QueuedTask>,
    normal_tx: mpsc::Sender<QueuedTask>,
    low_tx: mpsc::Sender<QueuedTask>,
}

impl SchedulerSender {
    pub async fn submit(&self, task: Task, priority: Priority) -> Result<()> {
        let queued = QueuedTask { task, priority };
        let tx = match priority {
            Priority::High => &self.high_tx,
            Priority::Normal => &self.normal_tx,
            Priority::Low => &self.low_tx,
        };
        tx.send(queued)
            .await
            .map_err(|_| anyhow::anyhow!("scheduler channel closed"))?;
        Ok(())
    }
}

// ── Type Conversion Helpers ─────────────────────────────────────────

fn convert_verb(verb: TaskVerb) -> wit_task_types::TaskVerb {
    match verb {
        TaskVerb::Identify => wit_task_types::TaskVerb::Identify,
        TaskVerb::Monitor => wit_task_types::TaskVerb::Monitor,
        TaskVerb::Procure => wit_task_types::TaskVerb::Procure,
        TaskVerb::Maintain => wit_task_types::TaskVerb::Maintain,
        TaskVerb::Build => wit_task_types::TaskVerb::Build,
        TaskVerb::Update => wit_task_types::TaskVerb::Update,
        TaskVerb::Crud => wit_task_types::TaskVerb::Crud,
    }
}

fn convert_risk_tier(tier: RiskTier) -> wit_task_types::RiskTier {
    match tier {
        RiskTier::Low => wit_task_types::RiskTier::Low,
        RiskTier::Medium => wit_task_types::RiskTier::Medium,
        RiskTier::High => wit_task_types::RiskTier::High,
        RiskTier::Critical => wit_task_types::RiskTier::Critical,
    }
}

fn convert_budget(budget: &Budget) -> wit_task_types::Budget {
    wit_task_types::Budget {
        time_budget_ms: budget.time_budget_ms,
        max_tool_calls: budget.max_tool_calls,
        max_bytes_read: budget.max_bytes_read,
        max_bytes_written: budget.max_bytes_written,
        max_network_calls: budget.max_network_calls,
        max_llm_calls: budget.max_llm_calls,
        max_tokens_per_call: budget.max_tokens_per_call,
        risk_tier: convert_risk_tier(budget.risk_tier),
    }
}

// ── Task Execution ──────────────────────────────────────────────────

/// Execute a task by dispatching to the appropriate Cell.
async fn execute_task(brain: &BrainState, task: &Task) -> Result<TaskResult> {
    let start = std::time::Instant::now();

    // Check if the Cell is loaded.
    if !brain.engine.is_loaded(&task.cell_id) {
        return Ok(TaskResult::Failed {
            task_id: task.id,
            error: format!("cell '{}' is not loaded", task.cell_id),
            retryable: false,
        });
    }

    // Check risk tier approval for BUILD/UPDATE.
    if task.verb.requires_approval() && task.budget.risk_tier >= RiskTier::High {
        tracing::warn!(
            task_id = %task.id,
            verb = ?task.verb,
            "task requires elevated approval — auto-approving in dev mode"
        );
        // TODO: actual approval gate
    }

    // Resolve capabilities via policy.
    let policy = brain.policy.load();
    let requested_caps = brain.engine.plugin_capabilities(&task.cell_id)
        .unwrap_or_default();
    let capabilities = crate::policy::resolve_capabilities(
        &task.cell_id,
        &requested_caps,
        &policy,
    )?;

    // Convert crawl_types::Task to WIT task_types::Task.
    let wit_task = wit_task_types::Task {
        id: task.id.to_string(),
        verb: convert_verb(task.verb),
        description: task.description.clone(),
        target: task.target.clone(),
        params: serde_json::to_string(&task.params).unwrap_or_default(),
        budget: convert_budget(&task.budget),
    };

    // Build subsystem refs for the Cell.
    let subsystems = SubsystemRefs {
        memory: brain.memory.clone(),
        ollama: brain.ollama.clone(),
        inference: brain.inference.clone(),
        journal: brain.journal.clone(),
        policy: policy.clone(),
        config: brain.config.clone(),
    };

    // Execute the plugin.
    let wit_result = brain.engine.execute_plugin(
        &task.cell_id,
        wit_task,
        capabilities,
        task.budget.clone(),
        subsystems,
    ).await?;

    let duration = start.elapsed();

    // Convert WIT result back to crawl_types::TaskResult.
    match wit_result {
        wit_task_types::TaskResult::Completed(output_json) => {
            let output: serde_json::Value = serde_json::from_str(&output_json).unwrap_or_default();
            Ok(TaskResult::Completed {
                task_id: task.id,
                verb: task.verb,
                output,
                duration_ms: duration.as_millis() as u64,
                tool_calls_used: 0, // TODO: read from CellState after execution
                bytes_read: 0,
                bytes_written: 0,
            })
        }
        wit_task_types::TaskResult::Checkpointed(cp) => {
            Ok(TaskResult::Checkpointed {
                task_id: task.id,
                checkpoint: Checkpoint {
                    continuation_id: Uuid::parse_str(&cp.continuation_id)
                        .unwrap_or_else(|_| Uuid::now_v7()),
                    cursor: cp.cursor,
                    state_blob_ref: cp.state_blob_ref,
                    completed_steps: cp.completed_steps,
                    findings: vec![],
                    current_stage: cp.current_stage,
                    continuation_plan: cp.continuation_plan,
                    remaining_estimate_ms: cp.remaining_estimate_ms,
                },
                partial_output: None,
            })
        }
        wit_task_types::TaskResult::Failed(error) => {
            Ok(TaskResult::Failed {
                task_id: task.id,
                error,
                retryable: true,
            })
        }
    }
}

/// Persist task result status to SQLite.
fn persist_task_result(brain: &BrainState, task_id: Uuid, result: &TaskResult) {
    let (status, result_json, completed_at) = match result {
        TaskResult::Completed { .. } => (
            "completed",
            serde_json::to_string(result).ok(),
            Some(chrono::Utc::now().to_rfc3339()),
        ),
        TaskResult::Checkpointed { .. } => (
            "checkpointed",
            serde_json::to_string(result).ok(),
            None,
        ),
        TaskResult::Failed { .. } => (
            "failed",
            serde_json::to_string(result).ok(),
            Some(chrono::Utc::now().to_rfc3339()),
        ),
    };
    let db = brain.storage.db.lock();
    let _ = db.execute(
        "UPDATE tasks SET status = ?1, result = ?2, completed_at = ?3 WHERE id = ?4",
        rusqlite::params![status, result_json, completed_at, task_id.to_string()],
    );
}
