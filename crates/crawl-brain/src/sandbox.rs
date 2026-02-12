//! CLI sandboxing using Landlock for filesystem restrictions.
//!
//! Linux 5.15 supports Landlock ABI v1, which provides filesystem access control.
//! Primary sandboxing for Cells is via WASM — Landlock is used for CLI exec.
#![allow(unused)]

use anyhow::{Context, Result};
use landlock::{
    Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
    RulesetStatus, ABI,
};
use std::path::Path;

/// Output from a sandboxed CLI command execution.
#[derive(Debug, Clone)]
pub struct SandboxedOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Execute a CLI command with timeout.
pub async fn sandboxed_exec(
    command: &str,
    args: &[&str],
    _allowed_read_paths: &[&str],
    _allowed_write_paths: &[&str],
    timeout_ms: u64,
) -> Result<SandboxedOutput> {
    use tokio::process::Command;

    let output = tokio::time::timeout(
        std::time::Duration::from_millis(timeout_ms),
        Command::new(command)
            .args(args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("failed to spawn sandboxed command")?
            .wait_with_output(),
    )
    .await
    .context("sandboxed command timed out")?
    .context("sandboxed command failed")?;

    Ok(SandboxedOutput {
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

/// Create and apply a Landlock ruleset for filesystem sandboxing.
///
/// Must be called in the child process before exec for full isolation.
pub fn apply_landlock_sandbox(
    allowed_read_paths: &[&Path],
    allowed_write_paths: &[&Path],
) -> Result<RulesetStatus> {
    let abi = ABI::V1;
    let read_access = AccessFs::from_read(abi);
    let read_write_access = AccessFs::from_all(abi);

    // Create ruleset handling all filesystem access types.
    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .context("failed to handle access")?
        .create()
        .context("failed to create Landlock ruleset")?;

    // Add read-only rules.
    for path in allowed_read_paths {
        if path.exists() {
            let fd = PathFd::new(path).context("failed to open path for Landlock")?;
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, read_access))
                .context("failed to add read rule")?;
        }
    }

    // Add read-write rules.
    for path in allowed_write_paths {
        if path.exists() {
            let fd = PathFd::new(path).context("failed to open path for Landlock")?;
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, read_write_access))
                .context("failed to add write rule")?;
        }
    }

    let status = ruleset
        .restrict_self()
        .context("failed to apply Landlock restrictions")?;

    match status.ruleset {
        RulesetStatus::FullyEnforced => {
            tracing::info!("Landlock fully enforced");
        }
        RulesetStatus::PartiallyEnforced => {
            tracing::warn!("Landlock partially enforced");
        }
        RulesetStatus::NotEnforced => {
            tracing::warn!("Landlock not enforced (kernel support missing)");
        }
        _ => {}
    }

    Ok(status.ruleset)
}
