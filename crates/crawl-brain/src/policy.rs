//! Policy enforcement, capability gating, and signature verification.
#![allow(unused)]

use anyhow::{bail, Context, Result};
use crawl_types::{Capability, FeatureFlags, HumanApprovalAction, PolicyConfig, ProhibitedAction};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::collections::HashSet;
use std::path::Path;

/// Verify that a policy file's Ed25519 signature is valid.
pub fn verify_policy_signature(
    policy_path: &Path,
    sig_path: &Path,
    pubkey_path: &Path,
) -> Result<()> {
    let policy_bytes =
        std::fs::read(policy_path).with_context(|| "failed to read policy file")?;
    let sig_hex =
        std::fs::read_to_string(sig_path).with_context(|| "failed to read signature file")?;
    let sig_bytes = hex::decode(sig_hex.trim()).with_context(|| "invalid hex in signature")?;
    let signature =
        Signature::from_slice(&sig_bytes).with_context(|| "invalid Ed25519 signature format")?;

    let pubkey_bytes =
        std::fs::read(pubkey_path).with_context(|| "failed to read public key file")?;
    let pubkey_array: [u8; 32] = pubkey_bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("public key must be exactly 32 bytes"))?;
    let verifying_key =
        VerifyingKey::from_bytes(&pubkey_array).with_context(|| "invalid Ed25519 public key")?;

    verifying_key
        .verify(&policy_bytes, &signature)
        .with_context(|| "policy signature verification failed")?;

    tracing::info!("policy signature verified successfully");
    Ok(())
}

/// Resolve the capabilities available to a specific Cell, based on:
/// 1. The Cell's requested capabilities (from its manifest)
/// 2. The policy's per-Cell grants
/// 3. The global feature flags (disabled features do not exist)
pub fn resolve_capabilities(
    cell_id: &str,
    requested: &[Capability],
    policy: &PolicyConfig,
) -> Result<HashSet<Capability>> {
    let available = capabilities_from_features(&policy.features);

    // Get the policy-granted capabilities for this Cell.
    let granted: HashSet<Capability> = policy
        .cell_capabilities
        .get(cell_id)
        .map(|caps| caps.iter().copied().collect())
        .unwrap_or_default();

    let mut resolved = HashSet::new();
    let mut denied = Vec::new();

    for cap in requested {
        if !available.contains(cap) {
            // Feature is globally disabled — capability does not exist.
            denied.push((*cap, "feature disabled globally"));
        } else if !granted.contains(cap) {
            // Cell is not granted this capability.
            denied.push((*cap, "not granted to this cell"));
        } else {
            resolved.insert(*cap);
        }
    }

    if !denied.is_empty() {
        for (cap, reason) in &denied {
            tracing::debug!(
                cell_id,
                capability = ?cap,
                reason,
                "capability denied"
            );
        }
    }

    Ok(resolved)
}

/// Derive the set of capabilities that exist given the feature flags.
/// Disabled features → their capabilities simply don't exist.
fn capabilities_from_features(flags: &FeatureFlags) -> HashSet<Capability> {
    let mut caps = HashSet::new();

    // Always available (not gated by feature flags).
    caps.insert(Capability::FilesystemRead);
    caps.insert(Capability::ProcessList);
    caps.insert(Capability::LogRead);
    caps.insert(Capability::JournalEmit);
    caps.insert(Capability::MetricsRead);
    caps.insert(Capability::InferenceRun);
    caps.insert(Capability::MemoryAccess);
    caps.insert(Capability::LlmQuery);

    if flags.cli_exec {
        caps.insert(Capability::CliExec);
    }
    if flags.network {
        caps.insert(Capability::NetworkGet);
    }

    // FilesystemWrite is always potentially available but scoped by paths.
    caps.insert(Capability::FilesystemWrite);

    caps
}

/// Check whether a given path is within the allowed read paths.
pub fn is_path_readable(path: &str, policy: &PolicyConfig) -> bool {
    policy
        .allowed_read_paths
        .iter()
        .any(|allowed| path.starts_with(allowed))
}

/// Check whether a given path is within the allowed write paths.
pub fn is_path_writable(path: &str, policy: &PolicyConfig) -> bool {
    policy
        .allowed_write_paths
        .iter()
        .any(|allowed| path.starts_with(allowed))
}

/// Check whether a CLI command is in the allowlist.
pub fn is_command_allowed(cmd: &str, policy: &PolicyConfig) -> bool {
    // Extract the base command (first word).
    let base = cmd.split_whitespace().next().unwrap_or("");
    policy.allowed_cli_commands.iter().any(|c| c == base)
}

/// Check whether a domain is in the network allowlist.
pub fn is_domain_allowed(domain: &str, policy: &PolicyConfig) -> bool {
    policy
        .allowed_network_domains
        .iter()
        .any(|d| domain == d || domain.ends_with(&format!(".{d}")))
}

/// Check whether a category is blocked.
pub fn is_category_blocked(category: &str, policy: &PolicyConfig) -> bool {
    policy.blocked_categories.iter().any(|c| c == category)
}

/// Check whether an action is absolutely prohibited.
/// Prohibited actions have NO override, NO escalation path, NO exception.
/// Returns the prohibition if blocked, None if not prohibited.
pub fn check_prohibited(action: ProhibitedAction, policy: &PolicyConfig) -> Option<ProhibitedAction> {
    if policy.prohibited_actions.contains(&action) {
        tracing::warn!(action = ?action, "PROHIBITED action attempted — permanently blocked");
        Some(action)
    } else {
        None
    }
}

/// Check whether an action requires human approval.
/// Returns true if the action needs approval before execution.
pub fn requires_human_approval(action: HumanApprovalAction, policy: &PolicyConfig) -> bool {
    let needed = policy.human_approval_required.contains(&action);
    if needed {
        tracing::info!(action = ?action, "action requires human approval");
    }
    needed
}

/// Validate that a proposed action is neither prohibited nor unapproved.
/// Returns Ok(()) if the action can proceed, Err with reason if not.
pub fn validate_action_gate(
    prohibited_check: Option<ProhibitedAction>,
    approval_check: Option<HumanApprovalAction>,
    has_approval: bool,
    policy: &PolicyConfig,
) -> Result<()> {
    // Absolute prohibitions — no override possible.
    if let Some(action) = prohibited_check {
        if check_prohibited(action, policy).is_some() {
            bail!("action {:?} is permanently prohibited by policy", action);
        }
    }

    // Human approval gate.
    if let Some(action) = approval_check {
        if requires_human_approval(action, policy) && !has_approval {
            bail!("action {:?} requires human approval", action);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_policy() -> PolicyConfig {
        PolicyConfig {
            features: FeatureFlags {
                network: false,
                cli_exec: true,
                ..FeatureFlags::default()
            },
            cell_capabilities: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "sysmon".into(),
                    vec![
                        Capability::FilesystemRead,
                        Capability::ProcessList,
                        Capability::MetricsRead,
                    ],
                );
                m.insert(
                    "researcher".into(),
                    vec![
                        Capability::NetworkGet,
                        Capability::LlmQuery,
                        Capability::MemoryAccess,
                    ],
                );
                m
            },
            allowed_read_paths: vec!["/proc".into(), "/sys".into()],
            allowed_write_paths: vec!["/tmp/crawl".into()],
            allowed_cli_commands: vec!["uname".into(), "df".into()],
            allowed_network_domains: vec!["docs.rs".into(), "crates.io".into()],
            blocked_categories: vec!["banking".into()],
            blocked_domains: vec![],
            ..PolicyConfig::default()
        }
    }

    #[test]
    fn resolve_caps_basic() {
        let policy = test_policy();
        let caps = resolve_capabilities(
            "sysmon",
            &[
                Capability::FilesystemRead,
                Capability::ProcessList,
                Capability::MetricsRead,
            ],
            &policy,
        )
        .unwrap();
        assert_eq!(caps.len(), 3);
        assert!(caps.contains(&Capability::FilesystemRead));
    }

    #[test]
    fn resolve_caps_denied_by_feature_flag() {
        let policy = test_policy();
        // Network is disabled globally, so NetworkGet doesn't exist.
        let caps = resolve_capabilities(
            "researcher",
            &[Capability::NetworkGet, Capability::LlmQuery],
            &policy,
        )
        .unwrap();
        assert!(!caps.contains(&Capability::NetworkGet));
        assert!(caps.contains(&Capability::LlmQuery));
    }

    #[test]
    fn resolve_caps_denied_by_cell_grant() {
        let policy = test_policy();
        // sysmon is not granted LlmQuery.
        let caps = resolve_capabilities(
            "sysmon",
            &[Capability::FilesystemRead, Capability::LlmQuery],
            &policy,
        )
        .unwrap();
        assert!(caps.contains(&Capability::FilesystemRead));
        assert!(!caps.contains(&Capability::LlmQuery));
    }

    #[test]
    fn path_checks() {
        let policy = test_policy();
        assert!(is_path_readable("/proc/loadavg", &policy));
        assert!(!is_path_readable("/etc/passwd", &policy));
        assert!(is_path_writable("/tmp/crawl/data", &policy));
        assert!(!is_path_writable("/tmp/other", &policy));
    }

    #[test]
    fn command_allowlist() {
        let policy = test_policy();
        assert!(is_command_allowed("uname -a", &policy));
        assert!(is_command_allowed("df -h", &policy));
        assert!(!is_command_allowed("rm -rf /", &policy));
    }

    #[test]
    fn domain_allowlist() {
        let policy = test_policy();
        assert!(is_domain_allowed("docs.rs", &policy));
        assert!(is_domain_allowed("sub.docs.rs", &policy));
        assert!(!is_domain_allowed("evil.com", &policy));
    }

    #[test]
    fn prohibited_actions_enforced() {
        let policy = test_policy();
        // Financial transactions are always prohibited.
        assert!(check_prohibited(ProhibitedAction::FinancialTransaction, &policy).is_some());
        assert!(check_prohibited(ProhibitedAction::InfrastructureControl, &policy).is_some());
        assert!(check_prohibited(ProhibitedAction::LegalSubmission, &policy).is_some());
    }

    #[test]
    fn human_approval_enforced() {
        let policy = test_policy();
        assert!(requires_human_approval(HumanApprovalAction::Purchase, &policy));
        assert!(requires_human_approval(HumanApprovalAction::CredentialUsage, &policy));
        assert!(requires_human_approval(HumanApprovalAction::EmailSending, &policy));
    }

    #[test]
    fn validate_action_gate_blocks_prohibited() {
        let policy = test_policy();
        let result = validate_action_gate(
            Some(ProhibitedAction::FinancialTransaction),
            None,
            false,
            &policy,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("permanently prohibited"));
    }

    #[test]
    fn validate_action_gate_blocks_unapproved() {
        let policy = test_policy();
        let result = validate_action_gate(
            None,
            Some(HumanApprovalAction::Purchase),
            false, // no approval
            &policy,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires human approval"));
    }

    #[test]
    fn validate_action_gate_allows_approved() {
        let policy = test_policy();
        let result = validate_action_gate(
            None,
            Some(HumanApprovalAction::Purchase),
            true, // has approval
            &policy,
        );
        assert!(result.is_ok());
    }
}
