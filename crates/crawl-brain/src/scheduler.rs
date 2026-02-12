//! Task scheduling, priority queues, and budget tracking.
#![allow(unused)]

use anyhow::Result;
use crawl_types::{Checkpoint, JournalEventKind, RiskTier, Task, TaskResult};
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

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
}

impl Scheduler {
    pub fn new(brain: Arc<BrainState>) -> Self {
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
            // Biased select: high priority first.
            let queued = tokio::select! {
                biased;
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

            // Spawn task execution.
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
                let result = tokio::time::timeout(
                    std::time::Duration::from_millis(timeout_ms),
                    execute_task(&brain, &task),
                )
                .await;

                match result {
                    Ok(Ok(task_result)) => {
                        let event_kind = match &task_result {
                            TaskResult::Completed { .. } => JournalEventKind::TaskCompleted,
                            TaskResult::Checkpointed { .. } => JournalEventKind::TaskCheckpointed,
                            TaskResult::Failed { .. } => JournalEventKind::TaskFailed,
                        };
                        let _ = brain.journal.emit(
                            event_kind,
                            Some(task_id),
                            Some(cell_id),
                            serde_json::to_value(&task_result).unwrap_or_default(),
                        );
                    }
                    Ok(Err(e)) => {
                        tracing::error!(task_id = %task_id, error = %e, "task execution failed");
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

    // TODO: actually instantiate the WASM component and call execute.
    // For now, return a placeholder.
    let duration = start.elapsed();

    Ok(TaskResult::Completed {
        task_id: task.id,
        verb: task.verb,
        output: serde_json::json!({
            "message": "task executed (placeholder)",
            "cell_id": task.cell_id,
        }),
        duration_ms: duration.as_millis() as u64,
        tool_calls_used: 0,
        bytes_read: 0,
        bytes_written: 0,
    })
}
