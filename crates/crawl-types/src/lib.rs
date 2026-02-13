//! Shared domain types for the crawl Brain + Cells system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Task Verbs ──────────────────────────────────────────────────────

/// Every Cell task must declare exactly one verb.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TaskVerb {
    /// Determine what an entity is (process, file, service, dependency, error pattern).
    Identify,
    /// Observe state over time (resources, logs, file drift, network listeners).
    Monitor,
    /// Collect minimal diagnostic evidence.
    Procure,
    /// Keep agent ecosystem healthy (cache pruning, index rebuild, GC, etc).
    Maintain,
    /// Create new artifacts in sandbox (models, parsers, plugins).
    Train,
    /// Propose modifications to skills or Brain code.
    Update,
    /// Data operations on agent workspace / permitted dirs / structured stores.
    Crud,
}

impl TaskVerb {
    /// Whether this verb requires elevated approval before execution.
    pub fn requires_approval(&self) -> bool {
        matches!(self, TaskVerb::Train | TaskVerb::Update)
    }
}

// ── Risk Tier ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskTier {
    /// Read-only, no side effects.
    Low,
    /// Writes to agent-owned storage only.
    Medium,
    /// Writes to shared storage or executes commands.
    High,
    /// Modifies Brain code or system configuration.
    Critical,
}

// ── Budget ──────────────────────────────────────────────────────────

/// Resource budget assigned to every Brain→Cell task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    /// Maximum wall-clock time in milliseconds.
    pub time_budget_ms: Option<u64>,
    /// Absolute deadline (alternative to time_budget_ms).
    pub deadline_at: Option<DateTime<Utc>>,
    /// Maximum number of host tool calls.
    pub max_tool_calls: u32,
    /// Maximum bytes the Cell may read.
    pub max_bytes_read: u64,
    /// Maximum bytes the Cell may write.
    pub max_bytes_written: u64,
    /// Maximum outbound network requests.
    pub max_network_calls: u32,
    /// Maximum LLM query calls.
    pub max_llm_calls: u32,
    /// Maximum tokens per LLM call.
    pub max_tokens_per_call: u32,
    /// Risk tier for this task.
    pub risk_tier: RiskTier,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            time_budget_ms: Some(30_000),
            deadline_at: None,
            max_tool_calls: 100,
            max_bytes_read: 10 * 1024 * 1024,   // 10 MiB
            max_bytes_written: 1024 * 1024,       // 1 MiB
            max_network_calls: 0,
            max_llm_calls: 0,
            max_tokens_per_call: 2048,
            risk_tier: RiskTier::Low,
        }
    }
}

// ── Checkpoint ──────────────────────────────────────────────────────

/// Returned by a Cell when it cannot complete within its budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique ID for this checkpoint (used for continuation).
    pub continuation_id: Uuid,
    /// Opaque cursor for the Cell to resume from.
    pub cursor: String,
    /// Serialized state blob reference (stored in KV).
    pub state_blob_ref: Option<String>,
    /// Steps completed so far.
    pub completed_steps: Vec<String>,
    /// Findings so far.
    pub findings: Vec<serde_json::Value>,
    /// Current stage description.
    pub current_stage: String,
    /// Plan for what remains.
    pub continuation_plan: String,
    /// Estimated remaining time in ms.
    pub remaining_estimate_ms: Option<u64>,
}

// ── Task ────────────────────────────────────────────────────────────

/// A task dispatched from Brain to a Cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task ID.
    pub id: Uuid,
    /// The verb this task performs.
    pub verb: TaskVerb,
    /// Human-readable description of what to do.
    pub description: String,
    /// Target entity (process name, file path, service, etc).
    pub target: String,
    /// Additional structured parameters.
    pub params: serde_json::Value,
    /// Resource budget.
    pub budget: Budget,
    /// Which Cell (plugin) should handle this task.
    pub cell_id: String,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Optional continuation from a previous checkpoint.
    pub continuation: Option<Checkpoint>,
}

// ── Task Result ─────────────────────────────────────────────────────

/// The outcome of a Cell executing a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum TaskResult {
    /// Task completed successfully.
    Completed {
        task_id: Uuid,
        verb: TaskVerb,
        output: serde_json::Value,
        duration_ms: u64,
        tool_calls_used: u32,
        bytes_read: u64,
        bytes_written: u64,
        network_calls_used: u32,
        llm_calls_used: u32,
    },
    /// Task was interrupted and checkpointed.
    Checkpointed {
        task_id: Uuid,
        checkpoint: Checkpoint,
        partial_output: Option<serde_json::Value>,
    },
    /// Task failed with an error.
    Failed {
        task_id: Uuid,
        error: String,
        retryable: bool,
    },
}

// ── Policy Types ────────────────────────────────────────────────────

/// Feature flags controlling what capabilities exist in the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    pub network: bool,
    pub cli_exec: bool,
    pub ui_automation: bool,
    pub credentials_access: bool,
    pub self_update: bool,
    pub peer_sync: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            network: false,
            cli_exec: true,
            ui_automation: false,
            credentials_access: false,
            self_update: true,
            peer_sync: false,
        }
    }
}

/// Capability that can be granted to a Cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Read files within scoped paths.
    FilesystemRead,
    /// Write files within scoped paths.
    FilesystemWrite,
    /// List running processes.
    ProcessList,
    /// Read system logs.
    LogRead,
    /// Execute allowlisted CLI commands.
    CliExec,
    /// Make outbound HTTP GET requests (allowlisted).
    NetworkGet,
    /// Query the LLM.
    LlmQuery,
    /// Store/search semantic memory.
    MemoryAccess,
    /// Emit events to the journal.
    JournalEmit,
    /// Access system metrics.
    MetricsRead,
    /// Run inference on ONNX models.
    InferenceRun,
}

/// Actions that are absolutely prohibited — no capability exists, no override possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProhibitedAction {
    /// No autonomous financial transactions of any kind.
    FinancialTransaction,
    /// No autonomous identity authentication or login.
    IdentityAuthentication,
    /// No autonomous legal submissions, filings, or signatures.
    LegalSubmission,
    /// No autonomous infrastructure control (SCADA, power grid, IoT).
    InfrastructureControl,
    /// No autonomous medical record access or modification.
    MedicalRecordAccess,
    /// No autonomous weapon/ammunition procurement.
    WeaponProcurement,
    /// No autonomous surveillance tool operation.
    SurveillanceOperation,
}

/// Actions that require explicit human approval before execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanApprovalAction {
    /// Any purchase or payment.
    Purchase,
    /// Any public posting (social media, forums, etc).
    PublicPosting,
    /// Any file upload to external services.
    FileUpload,
    /// Any email or message sending.
    EmailSending,
    /// Any use of stored credentials.
    CredentialUsage,
    /// Code deployment to production.
    CodeDeployment,
    /// System configuration changes.
    SystemConfigChange,
}

/// Autonomous allowed actions — no approval needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllowedAutonomousAction {
    /// Read-only public web research.
    PublicResearch,
    /// Open data retrieval (APIs, public datasets).
    OpenDataRetrieval,
    /// Reading documentation and man pages.
    DocumentationReading,
    /// Accessing academic resources (arxiv, papers).
    AcademicResources,
    /// Local filesystem reads within policy scope.
    ScopedFileRead,
    /// Local system observation (metrics, logs).
    SystemObservation,
    /// Writing to agent-owned workspace.
    WorkspaceWrite,
    /// Running allowlisted CLI commands.
    AllowlistedCommands,
}

/// Policy configuration loaded from RON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Global feature flags.
    pub features: FeatureFlags,
    /// Per-cell capability grants.
    pub cell_capabilities: std::collections::HashMap<String, Vec<Capability>>,
    /// Filesystem paths cells are allowed to read.
    pub allowed_read_paths: Vec<String>,
    /// Filesystem paths cells are allowed to write.
    pub allowed_write_paths: Vec<String>,
    /// Allowlisted CLI commands.
    pub allowed_cli_commands: Vec<String>,
    /// Allowlisted network domains (for Tier 3 research).
    pub allowed_network_domains: Vec<String>,
    /// Blocked categories (permanently denied).
    pub blocked_categories: Vec<String>,
    /// Blocked domains (never fetch from these).
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    /// Absolute prohibitions — these actions can never be performed.
    #[serde(default)]
    pub prohibited_actions: Vec<ProhibitedAction>,
    /// Actions requiring human approval before execution.
    #[serde(default)]
    pub human_approval_required: Vec<HumanApprovalAction>,
    /// Actions the system may perform autonomously.
    #[serde(default)]
    pub allowed_autonomous: Vec<AllowedAutonomousAction>,
    /// Protected file patterns — these can NEVER be edited or deleted,
    /// even if they fall within an allowed write path.
    /// Supports exact paths and glob-like suffix matching (e.g. "*.ron", ".env*").
    #[serde(default)]
    pub protected_paths: Vec<String>,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            features: FeatureFlags::default(),
            cell_capabilities: std::collections::HashMap::new(),
            allowed_read_paths: vec![
                "/proc".into(),
                "/sys".into(),
                "/var/log".into(),
            ],
            allowed_write_paths: vec![],
            allowed_cli_commands: vec![
                "uname".into(),
                "df".into(),
                "free".into(),
                "ps".into(),
                "top".into(),
                "git".into(),
            ],
            allowed_network_domains: vec![],
            blocked_categories: vec![
                "banking".into(),
                "brokerage".into(),
                "crypto_exchange".into(),
                "payment_processor".into(),
                "tax_portal".into(),
                "login_flow".into(),
            ],
            blocked_domains: vec![],
            prohibited_actions: vec![
                ProhibitedAction::FinancialTransaction,
                ProhibitedAction::IdentityAuthentication,
                ProhibitedAction::LegalSubmission,
                ProhibitedAction::InfrastructureControl,
                ProhibitedAction::MedicalRecordAccess,
                ProhibitedAction::WeaponProcurement,
                ProhibitedAction::SurveillanceOperation,
            ],
            human_approval_required: vec![
                HumanApprovalAction::Purchase,
                HumanApprovalAction::PublicPosting,
                HumanApprovalAction::FileUpload,
                HumanApprovalAction::EmailSending,
                HumanApprovalAction::CredentialUsage,
                HumanApprovalAction::CodeDeployment,
                HumanApprovalAction::SystemConfigChange,
            ],
            allowed_autonomous: vec![
                AllowedAutonomousAction::PublicResearch,
                AllowedAutonomousAction::OpenDataRetrieval,
                AllowedAutonomousAction::DocumentationReading,
                AllowedAutonomousAction::AcademicResources,
                AllowedAutonomousAction::ScopedFileRead,
                AllowedAutonomousAction::SystemObservation,
                AllowedAutonomousAction::WorkspaceWrite,
                AllowedAutonomousAction::AllowlistedCommands,
            ],
            protected_paths: vec![
                // ── Policy & Security ──
                "policy/".into(),
                "*.ron".into(),
                "*.sig".into(),
                "*.pub".into(),
                // ── Secrets & Environment ──
                ".env".into(),
                ".env.*".into(),
                "*.pem".into(),
                "*.key".into(),
                "*.crt".into(),
                "*.p12".into(),
                "credentials".into(),
                "secrets".into(),
                // ── Git & VCS ──
                ".git/".into(),
                ".gitignore".into(),
                ".gitmodules".into(),
                ".gitattributes".into(),
                // ── Source code ──
                "*.rs".into(),
                "*.toml".into(),
                "Cargo.lock".into(),
                "*.proto".into(),
                "*.wit".into(),
                // ── Build & CI ──
                "build.rs".into(),
                ".github/".into(),
                ".gitlab-ci.yml".into(),
                "Makefile".into(),
                "Dockerfile".into(),
                "*.dockerfile".into(),
                // ── Network & System Config ──
                "/etc/".into(),
                "/sys/".into(),
                "*.conf".into(),
                "*.cfg".into(),
                "*.ini".into(),
                "*.yaml".into(),
                "*.yml".into(),
                // ── Documentation that governs behavior ──
                "CLAUDE.md".into(),
                "SECURITY.md".into(),
                // ── Shell scripts ──
                "*.sh".into(),
                "*.bash".into(),
                // ── WASM plugins (compiled) ──
                "*.wasm".into(),
            ],
        }
    }
}

// ── Journal Event ───────────────────────────────────────────────────

/// An event recorded in the append-only journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEvent {
    /// Event ID.
    pub id: Uuid,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Event kind.
    pub kind: JournalEventKind,
    /// Associated task ID (if any).
    pub task_id: Option<Uuid>,
    /// Cell that produced this event (if any).
    pub cell_id: Option<String>,
    /// Structured payload.
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JournalEventKind {
    TaskStarted,
    TaskCompleted,
    TaskFailed,
    TaskCheckpointed,
    ToolCall,
    LlmQuery,
    AnomalyDetected,
    PolicyViolation,
    CellLoaded,
    CellUnloaded,
    MetricsSnapshot,
    MemoryStored,
    ResearchEscalation,
    InjectionDetected,
    UpdateProposed,
    UpdateApproved,
    UpdateRejected,
    AutonomyThinkCycle,
    AutonomyRewardScored,
    AutonomyReflection,
    EntityDiscovered,
    SystemStartup,
    SystemShutdown,
}

// ── LLM Types ───────────────────────────────────────────────────────

/// Request to the LLM (Ollama).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    /// The prompt to send.
    pub prompt: String,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature (None = model default).
    pub temperature: Option<f32>,
    /// Whether the prompt contains tainted content.
    pub tainted: bool,
}

/// Response from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Generated text.
    pub text: String,
    /// Tokens consumed.
    pub tokens_used: u32,
    /// Whether this response is tainted (true if prompt was tainted).
    pub tainted: bool,
    /// Cost in microdollars (1 USD = 1,000,000). 0 for free/local providers.
    #[serde(default)]
    pub cost_microdollars: u64,
}

// ── Taint Tracking ──────────────────────────────────────────────────

/// Wraps a value with taint tracking for prompt injection defense.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tainted<T> {
    pub value: T,
    pub source: TaintSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaintSource {
    /// Content fetched from the network.
    Network { url: String },
    /// Content derived from LLM output on tainted input.
    LlmDerived,
    /// Content from an external file not in the trusted set.
    ExternalFile { path: String },
    /// Explicitly trusted (e.g., user input, policy file).
    Trusted,
}

impl TaintSource {
    pub fn is_tainted(&self) -> bool {
        !matches!(self, TaintSource::Trusted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_verb_approval() {
        assert!(!TaskVerb::Identify.requires_approval());
        assert!(!TaskVerb::Monitor.requires_approval());
        assert!(TaskVerb::Train.requires_approval());
        assert!(TaskVerb::Update.requires_approval());
    }

    #[test]
    fn risk_tier_ordering() {
        assert!(RiskTier::Low < RiskTier::Medium);
        assert!(RiskTier::Medium < RiskTier::High);
        assert!(RiskTier::High < RiskTier::Critical);
    }

    #[test]
    fn default_budget() {
        let b = Budget::default();
        assert_eq!(b.time_budget_ms, Some(30_000));
        assert_eq!(b.max_network_calls, 0);
        assert_eq!(b.risk_tier, RiskTier::Low);
    }

    #[test]
    fn policy_config_defaults() {
        let p = PolicyConfig::default();
        assert!(!p.features.network);
        assert!(p.features.cli_exec);
        assert!(!p.features.ui_automation);
        assert!(!p.features.credentials_access);
        assert!(p.features.self_update);
        assert_eq!(p.blocked_categories.len(), 6);
        // All absolute prohibitions are present by default.
        assert_eq!(p.prohibited_actions.len(), 7);
        assert!(p.prohibited_actions.contains(&ProhibitedAction::FinancialTransaction));
        assert!(p.prohibited_actions.contains(&ProhibitedAction::InfrastructureControl));
        // All human-approval gates are present by default.
        assert_eq!(p.human_approval_required.len(), 7);
        assert!(p.human_approval_required.contains(&HumanApprovalAction::Purchase));
        assert!(p.human_approval_required.contains(&HumanApprovalAction::CredentialUsage));
        // Autonomous actions are defined.
        assert_eq!(p.allowed_autonomous.len(), 8);
        assert!(p.allowed_autonomous.contains(&AllowedAutonomousAction::PublicResearch));
    }

    #[test]
    fn task_result_serialization() {
        let result = TaskResult::Completed {
            task_id: Uuid::nil(),
            verb: TaskVerb::Monitor,
            output: serde_json::json!({"load": 0.5}),
            duration_ms: 100,
            tool_calls_used: 3,
            bytes_read: 1024,
            bytes_written: 0,
            network_calls_used: 1,
            llm_calls_used: 0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"status\":\"Completed\""));
    }

    #[test]
    fn taint_source_check() {
        assert!(TaintSource::Network { url: "http://example.com".into() }.is_tainted());
        assert!(TaintSource::LlmDerived.is_tainted());
        assert!(!TaintSource::Trusted.is_tainted());
    }
}
