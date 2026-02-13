//! Configuration loading and management for the Brain daemon.

use anyhow::{Context, Result};
use crawl_types::PolicyConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level daemon configuration (loaded from TOML).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    /// General daemon settings.
    #[serde(default)]
    pub daemon: DaemonConfig,

    /// Paths configuration.
    #[serde(default)]
    pub paths: PathsConfig,

    /// Ollama LLM settings (legacy, used as fallback if [llm] absent).
    #[serde(default)]
    pub ollama: OllamaConfig,

    /// LLM provider pool settings (preferred over [ollama]).
    #[serde(default)]
    pub llm: Option<LlmConfig>,

    /// ONNX inference settings.
    #[serde(default)]
    pub inference: InferenceConfig,

    /// Storage settings.
    #[serde(default)]
    pub storage: StorageConfig,

    /// API settings.
    #[serde(default)]
    pub api: ApiConfig,

    /// Monitoring settings.
    #[serde(default)]
    pub monitor: MonitorConfig,

    /// Autonomy / curiosity loop settings.
    #[serde(default)]
    pub autonomy: AutonomyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Path to the policy RON file.
    pub policy_path: PathBuf,
    /// Path to the policy signature file.
    pub policy_sig_path: PathBuf,
    /// Path to the policy signing public key.
    pub policy_pubkey_path: Option<PathBuf>,
    /// Log level filter.
    pub log_level: String,
    /// Whether to require policy signature verification.
    pub require_policy_signature: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            policy_path: PathBuf::from("policy/default.ron"),
            policy_sig_path: PathBuf::from("policy/default.ron.sig"),
            policy_pubkey_path: None,
            log_level: "info".into(),
            require_policy_signature: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    /// Directory containing compiled .wasm plugin files.
    pub plugins_dir: PathBuf,
    /// Directory for plugin workspace / scratch data.
    pub workspace_dir: PathBuf,
    /// Directory for ONNX model files.
    pub models_dir: PathBuf,
    /// Path to the soul.md file (brain's evolving identity document).
    #[serde(default = "default_soul_path")]
    pub soul_path: PathBuf,
}

fn default_soul_path() -> PathBuf {
    PathBuf::from("data/soul.md")
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            plugins_dir: PathBuf::from("plugins"),
            workspace_dir: PathBuf::from("workspace"),
            models_dir: PathBuf::from("models"),
            soul_path: default_soul_path(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Ollama API base URL.
    pub base_url: String,
    /// Model name to use.
    pub model: String,
    /// Maximum requests per second.
    pub rate_limit_rps: f64,
    /// Default max tokens per call.
    pub default_max_tokens: u32,
    /// Connection timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".into(),
            model: "glm-4.7-flash:latest".into(),
            rate_limit_rps: 5.0,
            default_max_tokens: 2048,
            timeout_ms: 60_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Whether to use TensorRT execution provider.
    pub use_tensorrt: bool,
    /// Whether to use CUDA execution provider (fallback).
    pub use_cuda: bool,
    /// Number of intra-op threads.
    pub intra_op_threads: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            use_tensorrt: true,
            use_cuda: true,
            intra_op_threads: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Path to the redb database file.
    pub redb_path: PathBuf,
    /// Path to the rusqlite database file.
    pub sqlite_path: PathBuf,
    /// Path to the journal directory.
    pub journal_dir: PathBuf,
    /// Maximum journal file size before rotation (bytes).
    pub journal_max_size: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            redb_path: PathBuf::from("data/brain.redb"),
            sqlite_path: PathBuf::from("data/brain.sqlite"),
            journal_dir: PathBuf::from("data/journal"),
            journal_max_size: 64 * 1024 * 1024, // 64 MiB
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Unix domain socket path for gRPC.
    pub uds_path: PathBuf,
    /// Optional TCP port for gRPC-Web.
    pub grpc_web_port: Option<u16>,
    /// Prometheus metrics port.
    pub metrics_port: u16,
    /// Optional TCP port for the web portal dashboard.
    #[serde(default = "default_portal_port")]
    pub portal_port: Option<u16>,
}

fn default_portal_port() -> Option<u16> {
    Some(9080)
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            uds_path: PathBuf::from("data/crawl-brain.sock"),
            grpc_web_port: Some(9090),
            metrics_port: 9091,
            portal_port: default_portal_port(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Interval for system metrics collection (milliseconds).
    pub metrics_interval_ms: u64,
    /// Interval for log scanning (milliseconds).
    pub log_scan_interval_ms: u64,
    /// Anomaly score threshold for alerting.
    pub anomaly_threshold: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            metrics_interval_ms: 5_000,
            log_scan_interval_ms: 10_000,
            anomaly_threshold: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyConfig {
    /// Whether the autonomy loop is enabled.
    pub enabled: bool,
    /// How often (ms) the Brain "thinks" about what to do next.
    pub think_interval_ms: u64,
    /// Max pending autonomy tasks before pausing submission.
    pub max_pending_tasks: u32,
    /// Max LLM tokens per think cycle.
    pub max_tokens_per_think: u32,
    /// Temperature for reasoning queries (lower = more focused).
    pub temperature: f32,
    /// Which verbs the autonomy loop is allowed to use.
    pub allowed_verbs: Vec<String>,
    /// Reward system settings.
    #[serde(default)]
    pub reward: RewardConfig,
    /// Soul (persistent identity) settings.
    #[serde(default)]
    pub soul: SoulConfig,
    /// Wisdom (learned constraints) settings.
    #[serde(default)]
    pub wisdom: WisdomConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Whether the reward system is enabled.
    pub enabled: bool,
    /// Weight for novelty axis in composite score.
    pub novelty_weight: f32,
    /// Weight for anomaly axis in composite score.
    pub anomaly_weight: f32,
    /// Weight for confidence axis in composite score.
    pub confidence_weight: f32,
    /// Weight for actionability axis in composite score.
    pub actionability_weight: f32,
    /// Weight for resource efficiency axis in composite score.
    #[serde(default = "default_efficiency_weight")]
    pub efficiency_weight: f32,
    /// Run LLM reflection every N think cycles.
    pub llm_reflect_every_n_cycles: u32,
    /// Max tokens for LLM reflection response.
    pub max_reflection_tokens: u32,
    /// Number of recent scored tasks to show in scoreboard.
    pub scoreboard_size: usize,
    /// Minimum adaptive interval (ms) — fastest think rate.
    pub adaptive_min_interval_ms: u64,
    /// Maximum adaptive interval (ms) — slowest think rate.
    pub adaptive_max_interval_ms: u64,
    /// EWMA smoothing factor (higher = more responsive to recent scores).
    pub ewma_alpha: f32,
    /// Similarity threshold for entity deduplication.
    pub entity_dedup_threshold: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            novelty_weight: 0.25,
            anomaly_weight: 0.20,
            confidence_weight: 0.20,
            actionability_weight: 0.20,
            efficiency_weight: 0.15,
            llm_reflect_every_n_cycles: 10,
            max_reflection_tokens: 512,
            scoreboard_size: 15,
            adaptive_min_interval_ms: 20_000,
            adaptive_max_interval_ms: 300_000,
            ewma_alpha: 0.3,
            entity_dedup_threshold: 0.85,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoulConfig {
    /// Whether the soul system is enabled.
    pub enabled: bool,
    /// Maximum size of soul.md in bytes (prevents prompt bloat).
    pub max_bytes: usize,
    /// Max tokens for the soul update LLM call.
    pub max_update_tokens: u32,
}

impl Default for SoulConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_bytes: 4096,
            max_update_tokens: 768,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomConfig {
    /// Whether the wisdom system is enabled.
    pub enabled: bool,
    /// Max tokens for distillation LLM call.
    pub max_distillation_tokens: u32,
    /// Minimum confidence for a pre-flight wisdom match to be considered.
    pub preflight_min_confidence: f64,
    /// Above this confidence, pre-flight blocks the proposal; below, it warns.
    pub preflight_block_confidence: f64,
    /// Confidence boost per confirmation.
    pub reinforce_delta: f64,
    /// Confidence penalty per contradiction.
    pub decay_delta: f64,
    /// Below this confidence, entry is deactivated.
    pub deactivation_threshold: f64,
    /// Max wisdom entries to include in LLM prompt.
    pub max_prompt_entries: usize,
    /// Maturity thresholds for verb graduation.
    #[serde(default)]
    pub maturity: MaturityThresholds,
}

impl Default for WisdomConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_distillation_tokens: 512,
            preflight_min_confidence: 0.4,
            preflight_block_confidence: 0.8,
            reinforce_delta: 0.1,
            decay_delta: 0.15,
            deactivation_threshold: 0.1,
            max_prompt_entries: 30,
            maturity: MaturityThresholds::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaturityThresholds {
    /// Entity count required to graduate to Investigator (PROCURE unlocked).
    pub investigator_entity_count: u32,
    /// Wisdom entry count required to graduate to Investigator.
    pub investigator_wisdom_count: u32,
    /// Successful PROCURE task count required to graduate to Caretaker (MAINTAIN unlocked).
    pub caretaker_procure_count: u32,
    /// Successful MAINTAIN task count required to graduate to Builder (TRAIN unlocked).
    pub builder_maintain_count: u32,
}

impl Default for MaturityThresholds {
    fn default() -> Self {
        Self {
            investigator_entity_count: 10,
            investigator_wisdom_count: 5,
            caretaker_procure_count: 5,
            builder_maintain_count: 5,
        }
    }
}

// ── LLM Provider Pool Config ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Daily budget in USD. Cloud providers are skipped when exceeded.
    #[serde(default = "default_daily_budget")]
    pub daily_budget_usd: f64,
    /// Ordered list of providers. First available provider is used.
    pub providers: Vec<LlmProviderConfig>,
}

fn default_daily_budget() -> f64 {
    5.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderConfig {
    /// Provider kind: "anthropic" or "ollama".
    pub kind: String,
    /// Model name (e.g. "claude-haiku-4-5-20251001" or "qwen2.5:14b").
    pub model: String,
    /// Base URL (required for ollama, ignored for anthropic).
    #[serde(default)]
    pub base_url: Option<String>,
    /// Maximum requests per second.
    #[serde(default = "default_rps")]
    pub rate_limit_rps: f64,
    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Environment variable name holding the API key (for anthropic).
    #[serde(default)]
    pub api_key_env: Option<String>,
}

fn default_efficiency_weight() -> f32 {
    0.15
}

fn default_rps() -> f64 {
    5.0
}

fn default_timeout() -> u64 {
    30_000
}

impl Default for AutonomyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            think_interval_ms: 60_000,
            max_pending_tasks: 5,
            max_tokens_per_think: 1024,
            temperature: 0.4,
            allowed_verbs: vec![
                "IDENTIFY".into(), "MONITOR".into(), "RESEARCH".into(),
                "PROCURE".into(), "MAINTAIN".into(), "TRAIN".into(), "UPDATE".into(),
            ],
            reward: RewardConfig::default(),
            soul: SoulConfig::default(),
            wisdom: WisdomConfig::default(),
        }
    }
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            daemon: DaemonConfig::default(),
            paths: PathsConfig::default(),
            ollama: OllamaConfig::default(),
            llm: None,
            inference: InferenceConfig::default(),
            storage: StorageConfig::default(),
            api: ApiConfig::default(),
            monitor: MonitorConfig::default(),
            autonomy: AutonomyConfig::default(),
        }
    }
}

impl BrainConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file: {}", path.display()))?;
        let config: Self =
            toml::from_str(&content).with_context(|| "failed to parse config TOML")?;
        Ok(config)
    }

    /// Load or create default configuration.
    pub fn load_or_default(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            tracing::warn!("Config file not found at {}, using defaults", path.display());
            Ok(Self::default())
        }
    }

    /// Return the effective LLM config: use `[llm]` if present, otherwise
    /// synthesize a single-provider config from the legacy `[ollama]` section.
    pub fn effective_llm_config(&self) -> LlmConfig {
        if let Some(ref llm) = self.llm {
            llm.clone()
        } else {
            LlmConfig {
                daily_budget_usd: 0.0,
                providers: vec![LlmProviderConfig {
                    kind: "ollama".into(),
                    model: self.ollama.model.clone(),
                    base_url: Some(self.ollama.base_url.clone()),
                    rate_limit_rps: self.ollama.rate_limit_rps,
                    timeout_ms: self.ollama.timeout_ms,
                    api_key_env: None,
                }],
            }
        }
    }
}

/// Load policy from a RON file.
pub fn load_policy(path: &Path) -> Result<PolicyConfig> {
    if !path.exists() {
        tracing::warn!("Policy file not found at {}, using defaults", path.display());
        return Ok(PolicyConfig::default());
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read policy file: {}", path.display()))?;
    let policy: PolicyConfig =
        ron::from_str(&content).with_context(|| "failed to parse policy RON")?;
    Ok(policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_serializes() {
        let config = BrainConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        assert!(toml_str.contains("[daemon]"));
        assert!(toml_str.contains("[ollama]"));
    }

    #[test]
    fn effective_llm_config_fallback_to_ollama() {
        let config = BrainConfig::default();
        assert!(config.llm.is_none());
        let llm = config.effective_llm_config();
        assert_eq!(llm.providers.len(), 1);
        assert_eq!(llm.providers[0].kind, "ollama");
        assert_eq!(llm.providers[0].model, config.ollama.model);
    }

    #[test]
    fn effective_llm_config_uses_explicit() {
        let mut config = BrainConfig::default();
        config.llm = Some(LlmConfig {
            daily_budget_usd: 5.0,
            providers: vec![
                LlmProviderConfig {
                    kind: "anthropic".into(),
                    model: "claude-haiku-4-5-20251001".into(),
                    base_url: None,
                    rate_limit_rps: 5.0,
                    timeout_ms: 30000,
                    api_key_env: Some("ANTHROPIC_API_KEY".into()),
                },
            ],
        });
        let llm = config.effective_llm_config();
        assert_eq!(llm.providers.len(), 1);
        assert_eq!(llm.providers[0].kind, "anthropic");
    }

    #[test]
    fn default_policy_serializes_to_ron() {
        let policy = PolicyConfig::default();
        let ron_str = ron::ser::to_string_pretty(&policy, ron::ser::PrettyConfig::default()).unwrap();
        assert!(ron_str.contains("network"));
    }
}
