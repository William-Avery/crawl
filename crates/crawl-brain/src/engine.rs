//! Wasmtime WASM Component Model engine for loading and running Cells (plugins).
#![allow(unused)]

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{Config, Engine, Store};

use crate::config::BrainConfig;
use crate::inference::InferenceEngine;
use crate::journal::Journal;
use crate::llm::LlmPool;
use crate::memory::MemorySystem;
use crawl_types::{Budget, Capability, JournalEventKind, PolicyConfig};

// ── Host-side WIT Bindings ──────────────────────────────────────────

wasmtime::component::bindgen!({
    world: "cell",
    path: "../../wit",
    async: true,
    trappable_imports: true,
});

// Re-export WIT types for use by scheduler and other modules.
pub(crate) use crawl::plugin::task_types as wit_task_types;

// ── Subsystem References ────────────────────────────────────────────

/// Shared subsystem references passed into each Cell execution.
/// All fields are Arc-wrapped so cloning is cheap.
#[derive(Clone)]
pub struct SubsystemRefs {
    pub memory: Arc<MemorySystem>,
    pub llm: Arc<LlmPool>,
    pub inference: Option<Arc<InferenceEngine>>,
    pub journal: Arc<Journal>,
    pub policy: Arc<PolicyConfig>,
    pub config: BrainConfig,
}

// ── Cell State ──────────────────────────────────────────────────────

/// Per-Cell instance state stored in the Wasmtime Store.
pub struct CellState {
    pub cell_id: String,
    pub capabilities: std::collections::HashSet<Capability>,
    pub tool_calls_used: u32,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub network_calls_used: u32,
    pub llm_calls_used: u32,
    pub budget: Budget,
    pub output_buffer: Vec<u8>,
    pub table: ResourceTable,
    pub subsystems: SubsystemRefs,
    pub wasi_ctx: wasmtime_wasi::WasiCtx,
}

impl CellState {
    pub fn new(
        cell_id: String,
        capabilities: std::collections::HashSet<Capability>,
        budget: Budget,
        subsystems: SubsystemRefs,
    ) -> Self {
        Self {
            cell_id,
            capabilities,
            tool_calls_used: 0,
            bytes_read: 0,
            bytes_written: 0,
            network_calls_used: 0,
            llm_calls_used: 0,
            budget,
            output_buffer: Vec::new(),
            table: ResourceTable::new(),
            subsystems,
            wasi_ctx: wasmtime_wasi::WasiCtxBuilder::new().build(),
        }
    }

}

impl wasmtime_wasi::IoView for CellState {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
}

impl wasmtime_wasi::WasiView for CellState {
    fn ctx(&mut self) -> &mut wasmtime_wasi::WasiCtx {
        &mut self.wasi_ctx
    }
}

impl CellState {
    fn check_tool_budget(&mut self) -> std::result::Result<(), String> {
        if self.tool_calls_used >= self.budget.max_tool_calls {
            return Err(format!("tool call budget exhausted ({}/{})", self.tool_calls_used, self.budget.max_tool_calls));
        }
        self.tool_calls_used += 1;
        Ok(())
    }

    fn has_capability(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }

    fn require_cap(&mut self, cap: Capability) -> std::result::Result<(), String> {
        self.check_tool_budget()?;
        if !self.has_capability(cap) {
            return Err(format!("capability denied: {cap:?}"));
        }
        Ok(())
    }
}

// ── Host Tool Implementations ───────────────────────────────────────
// With trappable_imports, return type is anyhow::Result<Result<T, String>>
// Outer Result: host trap (anyhow::Error). Inner Result: WIT-level error (String).

impl crawl::plugin::host_tools::Host for CellState {
    async fn read_file(&mut self, path: String, max_bytes: u32) -> anyhow::Result<std::result::Result<Vec<u8>, String>> {
        if let Err(e) = self.require_cap(Capability::FilesystemRead) {
            return Ok(Err(e));
        }
        // Check path against policy allowlist.
        if !crate::policy::is_path_readable(&path, &self.subsystems.policy) {
            return Ok(Err(format!("path not in allowed read paths: {path}")));
        }
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => return Ok(Err(format!("read error: {e}"))),
        };
        let data = if data.len() > max_bytes as usize {
            data[..max_bytes as usize].to_vec()
        } else {
            data
        };
        self.bytes_read += data.len() as u64;
        if self.bytes_read > self.budget.max_bytes_read {
            return Ok(Err(format!("read budget exhausted ({}/{})", self.bytes_read, self.budget.max_bytes_read)));
        }
        Ok(Ok(data))
    }

    async fn list_dir(&mut self, path: String) -> anyhow::Result<std::result::Result<Vec<String>, String>> {
        if let Err(e) = self.require_cap(Capability::FilesystemRead) {
            return Ok(Err(e));
        }
        if !crate::policy::is_path_readable(&path, &self.subsystems.policy) {
            return Ok(Err(format!("path not in allowed read paths: {path}")));
        }
        match std::fs::read_dir(&path) {
            Ok(entries) => Ok(Ok(entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect())),
            Err(e) => Ok(Err(format!("list_dir error: {e}"))),
        }
    }

    async fn write_file(&mut self, path: String, data: Vec<u8>) -> anyhow::Result<std::result::Result<(), String>> {
        if let Err(e) = self.require_cap(Capability::FilesystemWrite) {
            return Ok(Err(e));
        }
        if !crate::policy::is_path_writable(&path, &self.subsystems.policy) {
            return Ok(Err(format!("path not in allowed write paths: {path}")));
        }
        let len = data.len() as u64;
        if self.bytes_written + len > self.budget.max_bytes_written {
            return Ok(Err("write budget exhausted".into()));
        }
        match std::fs::write(&path, &data) {
            Ok(()) => {
                self.bytes_written += len;
                Ok(Ok(()))
            }
            Err(e) => Ok(Err(format!("write error: {e}"))),
        }
    }

    async fn list_processes(&mut self) -> anyhow::Result<std::result::Result<Vec<crawl::plugin::host_tools::ProcessInfo>, String>> {
        if let Err(e) = self.require_cap(Capability::ProcessList) {
            return Ok(Err(e));
        }
        let mut processes = Vec::new();
        let entries = match std::fs::read_dir("/proc") {
            Ok(e) => e,
            Err(e) => return Ok(Err(format!("proc read error: {e}"))),
        };
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Ok(pid) = name_str.parse::<u32>() {
                let comm = std::fs::read_to_string(format!("/proc/{pid}/comm"))
                    .unwrap_or_default().trim().to_string();
                let cmdline = std::fs::read_to_string(format!("/proc/{pid}/cmdline"))
                    .unwrap_or_default().replace('\0', " ").trim().to_string();
                processes.push(crawl::plugin::host_tools::ProcessInfo {
                    pid, name: comm, cmdline, cpu_percent: 0.0, mem_kb: 0,
                });
            }
        }
        Ok(Ok(processes))
    }

    async fn read_log(&mut self, source: String, lines: u32) -> anyhow::Result<std::result::Result<Vec<String>, String>> {
        if let Err(e) = self.require_cap(Capability::LogRead) {
            return Ok(Err(e));
        }
        let log_path = match source.as_str() {
            "syslog" => "/var/log/syslog",
            "kern" => "/var/log/kern.log",
            "dmesg" => "/var/log/dmesg",
            other => other,
        };
        match std::fs::read_to_string(log_path) {
            Ok(content) => {
                self.bytes_read += content.len() as u64;
                Ok(Ok(content.lines().rev().take(lines as usize).map(String::from).collect()))
            }
            Err(e) => Ok(Err(format!("log read error: {e}"))),
        }
    }

    async fn exec_command(&mut self, cmd: String, args: Vec<String>) -> anyhow::Result<std::result::Result<crawl::plugin::host_tools::CommandOutput, String>> {
        if let Err(e) = self.require_cap(Capability::CliExec) {
            return Ok(Err(e));
        }
        if !crate::policy::is_command_allowed(&cmd, &self.subsystems.policy) {
            return Ok(Err(format!("command not in allowlist: {cmd}")));
        }
        match std::process::Command::new(&cmd).args(&args).output() {
            Ok(output) => Ok(Ok(crawl::plugin::host_tools::CommandOutput {
                exit_code: output.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            })),
            Err(e) => Ok(Err(format!("exec error: {e}"))),
        }
    }

    async fn http_get(&mut self, url: String) -> anyhow::Result<std::result::Result<crawl::plugin::host_tools::HttpResponse, String>> {
        if let Err(e) = self.require_cap(Capability::NetworkGet) {
            return Ok(Err(e));
        }
        self.network_calls_used += 1;
        if self.network_calls_used > self.budget.max_network_calls {
            return Ok(Err("network call budget exhausted".into()));
        }
        // Extract domain and check against policy allowlist.
        let domain = extract_domain(&url).unwrap_or_default();
        if !crate::policy::is_domain_allowed(&domain, &self.subsystems.policy) {
            return Ok(Err(format!("domain '{domain}' not in allowlist")));
        }
        // Perform the request. Clone nothing from &mut self across await.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build HTTP client: {e}"))?;
        match client.get(&url).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let headers: Vec<(String, String)> = resp.headers().iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();
                let body = resp.text().await.unwrap_or_default();
                self.bytes_read += body.len() as u64;
                Ok(Ok(crawl::plugin::host_tools::HttpResponse { status, body, headers }))
            }
            Err(e) => Ok(Err(format!("HTTP GET failed: {e}"))),
        }
    }

    async fn emit_event(&mut self, kind: String, payload: String) -> anyhow::Result<std::result::Result<(), String>> {
        if let Err(e) = self.require_cap(Capability::JournalEmit) {
            return Ok(Err(e));
        }
        let event_kind = match kind.as_str() {
            "anomaly_detected" => JournalEventKind::AnomalyDetected,
            "tool_call" => JournalEventKind::ToolCall,
            "llm_query" => JournalEventKind::LlmQuery,
            "memory_stored" => JournalEventKind::MemoryStored,
            "injection_detected" => JournalEventKind::InjectionDetected,
            _ => JournalEventKind::ToolCall,
        };
        let payload_val: serde_json::Value = serde_json::from_str(&payload).unwrap_or_default();
        let journal = self.subsystems.journal.clone();
        match journal.emit(event_kind, None, Some(self.cell_id.clone()), payload_val) {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(format!("journal emit failed: {e}"))),
        }
    }

    async fn get_config(&mut self, key: String) -> anyhow::Result<std::result::Result<String, String>> {
        if let Err(e) = self.check_tool_budget() {
            return Ok(Err(e));
        }
        let config = &self.subsystems.config;
        match key.as_str() {
            "ollama.model" => {
                let model = self.subsystems.llm.ollama_model()
                    .unwrap_or(&config.ollama.model);
                Ok(Ok(model.to_string()))
            }
            "ollama.base_url" => Ok(Ok(config.ollama.base_url.clone())),
            "paths.plugins_dir" => Ok(Ok(config.paths.plugins_dir.display().to_string())),
            "paths.workspace_dir" => Ok(Ok(config.paths.workspace_dir.display().to_string())),
            "paths.models_dir" => Ok(Ok(config.paths.models_dir.display().to_string())),
            "monitor.anomaly_threshold" => Ok(Ok(config.monitor.anomaly_threshold.to_string())),
            "monitor.metrics_interval_ms" => Ok(Ok(config.monitor.metrics_interval_ms.to_string())),
            _ => Ok(Err(format!("config key '{key}' not found"))),
        }
    }

    async fn memory_store(&mut self, content: String, metadata: String) -> anyhow::Result<std::result::Result<String, String>> {
        if let Err(e) = self.require_cap(Capability::MemoryAccess) {
            return Ok(Err(e));
        }
        let metadata_val: serde_json::Value = serde_json::from_str(&metadata).unwrap_or_default();
        let memory = self.subsystems.memory.clone();
        match memory.store(&content, metadata_val) {
            Ok(id) => Ok(Ok(id)),
            Err(e) => Ok(Err(format!("memory store failed: {e}"))),
        }
    }

    async fn memory_search(&mut self, query: String, top_k: u32) -> anyhow::Result<std::result::Result<Vec<crawl::plugin::host_tools::MemoryResult>, String>> {
        if let Err(e) = self.require_cap(Capability::MemoryAccess) {
            return Ok(Err(e));
        }
        let memory = self.subsystems.memory.clone();
        match memory.search(&query, top_k as usize) {
            Ok(entries) => {
                let results = entries.into_iter().map(|e| {
                    crawl::plugin::host_tools::MemoryResult {
                        id: e.id,
                        content: e.content,
                        similarity: e.similarity.unwrap_or(0.0),
                        metadata: serde_json::to_string(&e.metadata).unwrap_or_default(),
                    }
                }).collect();
                Ok(Ok(results))
            }
            Err(e) => Ok(Err(format!("memory search failed: {e}"))),
        }
    }

    async fn infer_anomaly(&mut self, metrics: Vec<f64>) -> anyhow::Result<std::result::Result<crawl::plugin::host_tools::AnomalyResult, String>> {
        if let Err(e) = self.require_cap(Capability::InferenceRun) {
            return Ok(Err(e));
        }
        match &self.subsystems.inference {
            Some(engine) => {
                match engine.infer_anomaly(&metrics) {
                    Ok(score) => Ok(Ok(crawl::plugin::host_tools::AnomalyResult {
                        score: score.score,
                        is_anomalous: score.is_anomalous,
                        details: score.details,
                    })),
                    Err(e) => Ok(Err(format!("anomaly inference failed: {e}"))),
                }
            }
            None => Ok(Ok(crawl::plugin::host_tools::AnomalyResult {
                score: 0.0,
                is_anomalous: false,
                details: "inference engine not available".into(),
            })),
        }
    }

    async fn embed_text(&mut self, text: String) -> anyhow::Result<std::result::Result<Vec<f32>, String>> {
        if let Err(e) = self.require_cap(Capability::InferenceRun) {
            return Ok(Err(e));
        }
        match &self.subsystems.inference {
            Some(engine) => {
                match engine.embed_text(&text) {
                    Ok(embedding) => Ok(Ok(embedding)),
                    Err(e) => Ok(Err(format!("embedding failed: {e}"))),
                }
            }
            None => {
                // Fallback to hash-based embedding.
                Ok(Ok(crate::inference::simple_hash_embedding(&text, 384)))
            }
        }
    }
}

// task-types only defines types; the Host trait has no methods.
impl crawl::plugin::task_types::Host for CellState {}

impl crawl::plugin::llm_api::Host for CellState {
    async fn query(
        &mut self,
        req: crawl::plugin::llm_api::LlmRequest,
    ) -> anyhow::Result<std::result::Result<crawl::plugin::llm_api::LlmResponse, String>> {
        if let Err(e) = self.require_cap(Capability::LlmQuery) {
            return Ok(Err(e));
        }
        if self.llm_calls_used >= self.budget.max_llm_calls {
            return Ok(Err(format!("LLM budget exhausted ({}/{})", self.llm_calls_used, self.budget.max_llm_calls)));
        }
        self.llm_calls_used += 1;

        let llm_req = crawl_types::LlmRequest {
            prompt: req.prompt,
            max_tokens: req.max_tokens.min(self.budget.max_tokens_per_call),
            temperature: req.temperature,
            tainted: false, // Cell-originated prompts are not tainted.
        };

        // Clone the Arc before awaiting to avoid holding &mut self across .await.
        let llm = self.subsystems.llm.clone();
        match llm.query(&llm_req).await {
            Ok(resp) => Ok(Ok(crawl::plugin::llm_api::LlmResponse {
                text: resp.text,
                tokens_used: resp.tokens_used,
                tainted: resp.tainted,
            })),
            Err(e) => Ok(Err(format!("LLM query failed: {e}"))),
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Extract domain from a URL string without pulling in the `url` crate.
fn extract_domain(url: &str) -> Option<String> {
    let after_scheme = url.split("://").nth(1)?;
    let host_port = after_scheme.split('/').next()?;
    let host = host_port.split(':').next()?;
    Some(host.to_string())
}

// ── Plugin Metadata ─────────────────────────────────────────────────

pub struct PluginManifest {
    pub cell_id: String,
    pub name: String,
    pub version: String,
    pub requested_capabilities: Vec<Capability>,
    pub wasm_path: PathBuf,
}

struct LoadedPlugin {
    manifest: PluginManifest,
    component: Component,
}

// ── Plugin Engine ───────────────────────────────────────────────────

pub struct PluginEngine {
    engine: Engine,
    plugins: parking_lot::RwLock<HashMap<String, Arc<LoadedPlugin>>>,
}

impl PluginEngine {
    pub fn new(_config: &BrainConfig) -> Result<Self> {
        let mut wasm_config = Config::new();
        wasm_config.wasm_component_model(true);
        wasm_config.async_support(true);
        wasm_config.epoch_interruption(true);
        wasm_config.cranelift_opt_level(wasmtime::OptLevel::Speed);

        // Enable compilation cache — compiled WASM is cached to disk so
        // subsequent loads skip Cranelift compilation entirely.
        if let Err(e) = wasm_config.cache_config_load_default() {
            tracing::warn!(error = %e, "failed to enable wasmtime compilation cache, continuing without");
        }

        let engine = Engine::new(&wasm_config).with_context(|| "failed to create wasmtime engine")?;

        Ok(Self {
            engine,
            plugins: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    pub fn load_plugin(&self, manifest: PluginManifest) -> Result<()> {
        let component = Component::from_file(&self.engine, &manifest.wasm_path)
            .with_context(|| format!("failed to load WASM component: {}", manifest.wasm_path.display()))?;
        let cell_id = manifest.cell_id.clone();
        self.plugins.write().insert(cell_id.clone(), Arc::new(LoadedPlugin { manifest, component }));
        tracing::info!(cell_id, "plugin loaded");
        Ok(())
    }

    pub fn unload_plugin(&self, cell_id: &str) -> bool {
        let removed = self.plugins.write().remove(cell_id).is_some();
        if removed { tracing::info!(cell_id, "plugin unloaded"); }
        removed
    }

    /// Unload all plugins. Called during shutdown.
    pub fn unload_all(&self) {
        let mut plugins = self.plugins.write();
        let count = plugins.len();
        plugins.clear();
        tracing::info!(count, "all plugins unloaded");
    }

    /// Advance the wasmtime epoch to trap all running Cells.
    pub fn kill_all_cells(&self) {
        for _ in 0..1000 {
            self.engine.increment_epoch();
        }
        tracing::info!("wasmtime epoch advanced — all running Cells will trap");
    }

    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.read().keys().cloned().collect()
    }

    pub fn is_loaded(&self, cell_id: &str) -> bool {
        self.plugins.read().contains_key(cell_id)
    }

    pub fn wasmtime_engine(&self) -> &Engine {
        &self.engine
    }

    pub fn plugin_capabilities(&self, cell_id: &str) -> Option<Vec<Capability>> {
        self.plugins.read().get(cell_id).map(|p| p.manifest.requested_capabilities.clone())
    }

    /// Instantiate a plugin and execute a task.
    pub async fn execute_plugin(
        &self,
        cell_id: &str,
        task: crawl::plugin::task_types::Task,
        capabilities: std::collections::HashSet<Capability>,
        budget: Budget,
        subsystems: SubsystemRefs,
    ) -> Result<crawl::plugin::task_types::TaskResult> {
        let plugin = self.plugins.read().get(cell_id).cloned()
            .ok_or_else(|| anyhow::anyhow!("plugin '{}' not loaded", cell_id))?;

        tracing::debug!(cell_id, "creating WASM store");
        let state = CellState::new(cell_id.to_string(), capabilities, budget, subsystems);
        let mut store = Store::new(&self.engine, state);
        store.epoch_deadline_async_yield_and_update(1);

        tracing::debug!(cell_id, "setting up linker");
        let mut linker = Linker::new(&self.engine);
        wasmtime_wasi::add_to_linker_async(&mut linker)
            .with_context(|| format!("failed to add WASI to linker for {cell_id}"))?;
        Cell::add_to_linker(&mut linker, |state: &mut CellState| state)
            .with_context(|| format!("failed to add Cell bindings to linker for {cell_id}"))?;

        tracing::debug!(cell_id, "instantiating WASM component");
        let instance = Cell::instantiate_async(&mut store, &plugin.component, &linker).await
            .with_context(|| format!("failed to instantiate WASM component for {cell_id}"))?;

        tracing::debug!(cell_id, "calling execute");
        let result = instance.crawl_plugin_plugin_api().call_execute(&mut store, &task).await
            .with_context(|| format!("WASM execute call failed for {cell_id}"))?;

        tracing::debug!(cell_id, "execute completed");
        Ok(result)
    }

    pub fn scan_plugins_dir(&self, dir: &Path) -> Result<usize> {
        if !dir.exists() {
            tracing::warn!("plugins directory does not exist: {}", dir.display());
            return Ok(0);
        }
        let mut count = 0;
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "wasm") {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
                let cell_id = stem.strip_prefix("crawl_plugin_").unwrap_or(&stem).to_string();
                // Request all capabilities — the policy resolver will
                // grant only what's allowed for this cell_id.
                let manifest = PluginManifest {
                    cell_id: cell_id.clone(), name: cell_id.clone(), version: "0.1.0".into(),
                    requested_capabilities: vec![
                        Capability::FilesystemRead, Capability::FilesystemWrite,
                        Capability::ProcessList, Capability::LogRead,
                        Capability::CliExec, Capability::NetworkGet,
                        Capability::LlmQuery, Capability::MemoryAccess,
                        Capability::JournalEmit, Capability::MetricsRead,
                        Capability::InferenceRun,
                    ],
                    wasm_path: path,
                };
                match self.load_plugin(manifest) {
                    Ok(()) => count += 1,
                    Err(e) => tracing::warn!(cell_id = cell_id, error = %e, "failed to load plugin"),
                }
            }
        }
        Ok(count)
    }
}
