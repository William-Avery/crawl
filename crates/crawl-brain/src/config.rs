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

    /// Ollama LLM settings.
    #[serde(default)]
    pub ollama: OllamaConfig,

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
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            plugins_dir: PathBuf::from("plugins"),
            workspace_dir: PathBuf::from("workspace"),
            models_dir: PathBuf::from("models"),
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
            model: "glm4:latest".into(),
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
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            uds_path: PathBuf::from("data/crawl-brain.sock"),
            grpc_web_port: Some(9090),
            metrics_port: 9091,
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
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            daemon: DaemonConfig::default(),
            paths: PathsConfig::default(),
            ollama: OllamaConfig::default(),
            inference: InferenceConfig::default(),
            storage: StorageConfig::default(),
            api: ApiConfig::default(),
            monitor: MonitorConfig::default(),
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
    fn default_policy_serializes_to_ron() {
        let policy = PolicyConfig::default();
        let ron_str = ron::ser::to_string_pretty(&policy, ron::ser::PrettyConfig::default()).unwrap();
        assert!(ron_str.contains("network"));
    }
}
