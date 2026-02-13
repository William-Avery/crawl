//! crawl-brain: Supervisor daemon for the Policy-Driven Local Brain + Cells system.

mod api;
mod config;
mod curiosity;
mod engine;
mod inference;
mod journal;
mod llm;
mod memory;
mod monitor;
mod policy;
mod research;
mod reward;
mod sandbox;
mod soul;
mod wisdom;
mod scheduler;
mod storage;
mod updater;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal::unix::{signal, SignalKind};
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(name = "crawl-brain", about = "Policy-driven local Brain supervisor daemon")]
struct Cli {
    /// Path to the configuration file.
    #[arg(short, long, default_value = "crawl.toml")]
    config: PathBuf,

    /// Override log level.
    #[arg(long)]
    log_level: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load config first (before async runtime) for tracing setup.
    let config = config::BrainConfig::load_or_default(&cli.config)?;

    // Initialize tracing.
    let log_level = cli
        .log_level
        .as_deref()
        .unwrap_or(&config.daemon.log_level);
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!(
        version = env!("CARGO_PKG_VERSION"),
        "crawl-brain starting"
    );

    // Install panic hook that uses tracing (tokio::spawn panics go to stderr by default).
    std::panic::set_hook(Box::new(|info| {
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".into()
        };
        let location = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())).unwrap_or_default();
        tracing::error!(payload = %payload, location = %location, "PANIC");
    }));

    // Build the tokio runtime.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("crawl-brain")
        .build()?;

    runtime.block_on(async_main(config))
}

async fn async_main(config: config::BrainConfig) -> Result<()> {
    // Load policy.
    let policy = config::load_policy(&config.daemon.policy_path)?;
    info!(?policy.features, "policy loaded");

    // Initialize storage.
    let storage = storage::Storage::init(&config.storage)?;
    let sqlite_db = storage.db.clone();
    info!("storage initialized");

    // Initialize journal.
    let journal = journal::Journal::open(&config.storage.journal_dir)?;
    let journal = Arc::new(journal);

    // Log startup event.
    journal.emit(crawl_types::JournalEventKind::SystemStartup, None, None, serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "config_path": config.daemon.policy_path.display().to_string(),
    }))?;

    // Initialize the WASM engine and load plugins.
    let engine = engine::PluginEngine::new(&config)?;
    let plugins_loaded = engine.scan_plugins_dir(&config.paths.plugins_dir)?;
    info!(plugins_loaded, "WASM engine initialized");

    // Initialize the LLM provider pool.
    let llm = Arc::new(llm::LlmPool::new(&config.effective_llm_config())?);
    info!(
        primary = %llm.primary_provider_label(),
        "LLM provider pool initialized"
    );

    // Initialize inference (ONNX models) — non-fatal if models aren't present.
    let inference = match inference::InferenceEngine::new(&config) {
        Ok(eng) => {
            info!("ONNX inference engine initialized");
            Some(Arc::new(eng))
        }
        Err(e) => {
            tracing::warn!("ONNX inference engine unavailable: {e}");
            None
        }
    };

    // Initialize memory system (shares the SQLite connection from storage).
    let memory = Arc::new(memory::MemorySystem::new(sqlite_db, inference.clone())?);
    info!("memory system initialized");

    // Initialize research engine.
    let research = Arc::new(research::ResearchEngine::new(
        memory.clone(),
        llm.clone(),
        inference.clone(),
        policy.allowed_network_domains.clone(),
    ));
    info!("research engine initialized");

    // Initialize wisdom system (shares the SQLite connection from storage).
    let wisdom = if config.autonomy.wisdom.enabled {
        let ws = wisdom::WisdomSystem::new(storage.db.clone(), config.autonomy.wisdom.clone())?;
        info!(entries = ws.active_count(), "wisdom system initialized");
        Some(Arc::new(ws))
    } else {
        info!("wisdom system disabled by config");
        None
    };

    // Shutdown channel: Brain signals all subsystems to stop.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    // Build the shared brain state.
    let brain = Arc::new(BrainState {
        config: config.clone(),
        policy: arc_swap::ArcSwap::new(Arc::new(policy)),
        storage,
        journal: journal.clone(),
        engine,
        llm,
        inference,
        memory,
        research,
        wisdom,
        shutdown_tx,
    });

    // Initialize scheduler (receives shutdown signal).
    let scheduler = scheduler::Scheduler::new(brain.clone(), shutdown_rx.clone());
    let scheduler_sender = scheduler.sender();

    // Start the monitoring loop.
    let monitor_handle = tokio::spawn({
        let brain = brain.clone();
        let mut shutdown = shutdown_rx.clone();
        async move {
            tokio::select! {
                biased;
                _ = shutdown.changed() => {
                    tracing::info!("monitor loop received shutdown signal");
                }
                result = monitor::run_monitor_loop(brain) => {
                    if let Err(e) = result {
                        error!("monitor loop exited with error: {e}");
                    }
                }
            }
        }
    });

    // Clone sender for the curiosity loop before API takes ownership.
    let curiosity_sender = scheduler_sender.clone();

    // Start the gRPC API server.
    let api_handle = tokio::spawn({
        let brain = brain.clone();
        async move {
            if let Err(e) = api::serve(brain, scheduler_sender).await {
                error!("API server exited with error: {e}");
            }
        }
    });

    // Start the scheduler.
    let scheduler_handle = tokio::spawn(async move {
        if let Err(e) = scheduler.run().await {
            error!("scheduler exited with error: {e}");
        }
    });

    // Start the autonomy / curiosity loop.
    let curiosity_handle = if brain.config.autonomy.enabled {
        let soul = soul::Soul::load(
            config.paths.soul_path.clone(),
            config.autonomy.soul.clone(),
            brain.llm.clone(),
        )?;
        let loop_ = curiosity::CuriosityLoop::new(
            brain.clone(),
            curiosity_sender,
            shutdown_rx.clone(),
            soul,
            brain.wisdom.clone(),
        );
        Some(tokio::spawn(async move {
            if let Err(e) = loop_.run().await {
                error!("curiosity loop exited with error: {e}");
            }
        }))
    } else {
        info!("autonomy loop disabled by config");
        None
    };

    // Wait for shutdown signal.
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;
    let mut sighup = signal(SignalKind::hangup())?;

    tokio::select! {
        _ = sigterm.recv() => info!("received SIGTERM, shutting down"),
        _ = sigint.recv() => info!("received SIGINT, shutting down"),
        _ = sighup.recv() => {
            info!("received SIGHUP, reloading config");
            // TODO: implement hot-reload
            return Ok(());
        }
    }

    // ── Graceful Shutdown Sequence ────────────────────────────────────
    info!("initiating graceful shutdown");

    // Step 1: Journal the shutdown event.
    journal.emit(crawl_types::JournalEventKind::SystemShutdown, None, None, serde_json::json!({}))?;

    // Step 2: Signal all subsystems to stop (scheduler drains, monitor stops).
    let _ = brain.shutdown_tx.send(true);
    info!("shutdown signal sent to all subsystems");

    // Step 3: Kill all running WASM Cells by advancing the wasmtime epoch.
    // Any Cell in-flight will trap at its next yield point.
    brain.engine.kill_all_cells();

    // Step 4: Give in-flight tasks a brief grace period to complete/checkpoint.
    info!("waiting for in-flight tasks to complete (2s grace period)...");
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Step 5: Explicitly unload all plugins (drops compiled Components).
    brain.engine.unload_all();

    // Step 6: Abort any remaining background tasks.
    monitor_handle.abort();
    api_handle.abort();
    scheduler_handle.abort();
    if let Some(ref h) = curiosity_handle {
        h.abort();
    }

    // Wait briefly for abort to propagate.
    let _ = tokio::time::timeout(
        std::time::Duration::from_millis(500),
        async {
            let _ = monitor_handle.await;
            let _ = api_handle.await;
            let _ = scheduler_handle.await;
            if let Some(h) = curiosity_handle {
                let _ = h.await;
            }
        },
    ).await;

    info!("crawl-brain shut down cleanly");
    Ok(())
}

/// Shared state accessible to all Brain subsystems.
pub struct BrainState {
    pub config: config::BrainConfig,
    pub policy: arc_swap::ArcSwap<crawl_types::PolicyConfig>,
    pub storage: storage::Storage,
    pub journal: Arc<journal::Journal>,
    pub engine: engine::PluginEngine,
    pub llm: Arc<llm::LlmPool>,
    pub inference: Option<Arc<inference::InferenceEngine>>,
    pub memory: Arc<memory::MemorySystem>,
    pub research: Arc<research::ResearchEngine>,
    pub wisdom: Option<Arc<wisdom::WisdomSystem>>,
    /// Send `true` to trigger graceful shutdown of all subsystems.
    pub shutdown_tx: tokio::sync::watch::Sender<bool>,
}
