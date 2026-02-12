//! Self-update pipeline: git diff → sandbox build → test → approval → deploy.
#![allow(unused)]

use anyhow::Result;
use crawl_types::JournalEventKind;
use std::path::PathBuf;
use std::sync::Arc;

use crate::journal::Journal;
use crate::sandbox;

/// An update proposal from a Cell.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UpdateProposal {
    /// Unique ID for this proposal.
    pub id: String,
    /// Cell that proposed the update.
    pub cell_id: String,
    /// Description of what the update does.
    pub description: String,
    /// Git diff content.
    pub diff: String,
    /// Files affected.
    pub affected_files: Vec<String>,
    /// Risk assessment.
    pub risk_assessment: String,
    /// Rollback plan.
    pub rollback_plan: String,
}

/// Status of an update in the pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum UpdateStatus {
    Proposed,
    Building,
    Testing,
    AwaitingApproval,
    Approved,
    Deploying,
    Deployed,
    Rejected,
    Failed { error: String },
    RolledBack,
}

/// The self-update pipeline manager.
pub struct UpdatePipeline {
    workspace_dir: PathBuf,
    journal: Arc<Journal>,
}

impl UpdatePipeline {
    pub fn new(workspace_dir: PathBuf, journal: Arc<Journal>) -> Self {
        Self {
            workspace_dir,
            journal,
        }
    }

    /// Process an update proposal through the full pipeline.
    pub async fn process_proposal(&self, proposal: &UpdateProposal) -> Result<UpdateStatus> {
        tracing::info!(
            proposal_id = %proposal.id,
            cell_id = %proposal.cell_id,
            "processing update proposal"
        );

        self.journal.emit(
            JournalEventKind::UpdateProposed,
            None,
            Some(proposal.cell_id.clone()),
            serde_json::to_value(proposal)?,
        )?;

        // Step 1: Apply diff in sandbox.
        let sandbox_dir = self.workspace_dir.join("sandbox").join(&proposal.id);
        std::fs::create_dir_all(&sandbox_dir)?;

        // Step 2: Build in sandbox.
        let build_result = sandbox::sandboxed_exec(
            "cargo",
            &["build", "--manifest-path", &sandbox_dir.join("Cargo.toml").to_string_lossy()],
            &[sandbox_dir.to_str().unwrap_or(".")],
            &[sandbox_dir.to_str().unwrap_or(".")],
            120_000, // 2 minutes.
        )
        .await;

        match build_result {
            Ok(output) if output.exit_code == 0 => {
                tracing::info!(proposal_id = %proposal.id, "sandbox build succeeded");
            }
            Ok(output) => {
                let error = format!("build failed (exit {}): {}", output.exit_code, output.stderr);
                tracing::warn!(proposal_id = %proposal.id, %error);
                return Ok(UpdateStatus::Failed { error });
            }
            Err(e) => {
                return Ok(UpdateStatus::Failed {
                    error: format!("build error: {e}"),
                });
            }
        }

        // Step 3: Run tests in sandbox.
        let test_result = sandbox::sandboxed_exec(
            "cargo",
            &["test", "--manifest-path", &sandbox_dir.join("Cargo.toml").to_string_lossy()],
            &[sandbox_dir.to_str().unwrap_or(".")],
            &[sandbox_dir.to_str().unwrap_or(".")],
            120_000,
        )
        .await;

        match test_result {
            Ok(output) if output.exit_code == 0 => {
                tracing::info!(proposal_id = %proposal.id, "sandbox tests passed");
            }
            Ok(output) => {
                let error = format!("tests failed (exit {}): {}", output.exit_code, output.stderr);
                tracing::warn!(proposal_id = %proposal.id, %error);
                return Ok(UpdateStatus::Failed { error });
            }
            Err(e) => {
                return Ok(UpdateStatus::Failed {
                    error: format!("test error: {e}"),
                });
            }
        }

        // Step 4: Await approval.
        tracing::info!(
            proposal_id = %proposal.id,
            "update passed build & tests, awaiting approval"
        );

        Ok(UpdateStatus::AwaitingApproval)
    }

    /// Approve and deploy an update.
    pub async fn approve_and_deploy(&self, proposal_id: &str) -> Result<UpdateStatus> {
        self.journal.emit(
            JournalEventKind::UpdateApproved,
            None,
            None,
            serde_json::json!({"proposal_id": proposal_id}),
        )?;

        // TODO: Atomic symlink switch, retain rollback.
        tracing::info!(proposal_id, "update deployed (placeholder)");

        Ok(UpdateStatus::Deployed)
    }

    /// Reject an update proposal.
    pub fn reject(&self, proposal_id: &str, reason: &str) -> Result<()> {
        self.journal.emit(
            JournalEventKind::UpdateRejected,
            None,
            None,
            serde_json::json!({
                "proposal_id": proposal_id,
                "reason": reason,
            }),
        )?;
        Ok(())
    }
}
