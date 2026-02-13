//! Lightweight web portal serving a real-time dashboard for crawl-brain.

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use serde::Serialize;
use tokio::sync::watch;
use tracing::info;

use crate::BrainState;

/// Shared state for the portal server.
#[derive(Clone)]
struct PortalState {
    brain: Arc<BrainState>,
    start_time: std::time::Instant,
}

/// JSON snapshot of the entire brain state, sent to the dashboard.
#[derive(Serialize)]
struct PortalSnapshot {
    version: String,
    uptime_secs: u64,
    ewma: f64,
    think_interval_ms: u64,

    cells: Vec<CellSnapshot>,
    tasks: TaskSummary,
    recent_tasks: Vec<TaskSnapshot>,
    metrics: Option<MetricsSnapshot>,
    journal: Vec<JournalSnapshot>,
    llm: LlmSnapshot,
    memory_count: u64,
    wisdom: Option<WisdomSnapshot>,
    soul: String,
    scoreboard: Vec<ScoreboardSnapshot>,
    entities: Vec<EntitySnapshot>,
}

#[derive(Serialize)]
struct CellSnapshot {
    id: String,
    capabilities: Vec<String>,
}

#[derive(Serialize)]
struct TaskSummary {
    completed: u64,
    pending: u64,
    running: u64,
    failed: u64,
}

#[derive(Serialize)]
struct TaskSnapshot {
    id: String,
    verb: String,
    cell_id: String,
    description: String,
    target: String,
    status: String,
    created_at: String,
    completed_at: String,
}

#[derive(Serialize)]
struct MetricsSnapshot {
    cpu_load_1m: f64,
    cpu_load_5m: f64,
    cpu_load_15m: f64,
    mem_total_kb: u64,
    mem_available_kb: u64,
    mem_used_percent: f64,
    uptime_secs: f64,
}

#[derive(Serialize)]
struct JournalSnapshot {
    timestamp: String,
    kind: String,
    cell_id: String,
    task_id: String,
}

#[derive(Serialize)]
struct LlmSnapshot {
    provider: String,
    queries: u64,
    tokens: u64,
    budget_spent_usd: f64,
}

#[derive(Serialize)]
struct WisdomSnapshot {
    entries: Vec<WisdomEntrySnapshot>,
    maturity: String,
}

#[derive(Serialize)]
struct WisdomEntrySnapshot {
    kind: String,
    content: String,
    confidence: f64,
}

#[derive(Serialize)]
struct ScoreboardSnapshot {
    task_id: String,
    verb: String,
    target: String,
    composite: f64,
    novelty: f64,
    anomaly: f64,
    confidence: f64,
    actionability: f64,
    efficiency: f64,
    scored_at: String,
}

#[derive(Serialize)]
struct EntitySnapshot {
    name: String,
    kind: String,
    description: String,
    confidence: f64,
    last_seen: String,
}

/// Serve the portal on the given port with graceful shutdown.
pub async fn serve(
    brain: Arc<BrainState>,
    mut shutdown_rx: watch::Receiver<bool>,
    port: u16,
) -> anyhow::Result<()> {
    let state = PortalState {
        brain,
        start_time: std::time::Instant::now(),
    };

    let app = Router::new()
        .route("/", get(serve_html))
        .route("/api/state", get(serve_state))
        .with_state(state);

    let addr: std::net::SocketAddr = ([0, 0, 0, 0], port).into();
    info!(%addr, "portal server starting");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx.changed().await;
        })
        .await?;

    Ok(())
}

async fn serve_html() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        Html(include_str!("portal.html")),
    )
}

async fn serve_state(State(state): State<PortalState>) -> Result<Json<PortalSnapshot>, StatusCode> {
    let brain = &state.brain;
    let uptime = state.start_time.elapsed().as_secs();

    // Cells
    let plugins = brain.engine.list_plugins();
    let cells: Vec<CellSnapshot> = plugins
        .iter()
        .map(|id| CellSnapshot {
            id: id.clone(),
            capabilities: brain
                .engine
                .plugin_capabilities(id)
                .unwrap_or_default()
                .iter()
                .map(|c| format!("{c:?}"))
                .collect(),
        })
        .collect();

    // Task counts
    let (completed, pending, running, failed) = {
        let db = brain.storage.db.lock();
        let count = |status: &str| -> u64 {
            db.query_row(
                "SELECT COUNT(*) FROM tasks WHERE status = ?1",
                rusqlite::params![status],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0) as u64
        };
        (count("completed"), count("pending"), count("running"), count("failed"))
    };

    // Recent tasks
    let recent_tasks = {
        let db = brain.storage.db.lock();
        let mut stmt = db
            .prepare(
                "SELECT id, verb, cell_id, description, target, status, created_at, \
                 COALESCE(completed_at, '') FROM tasks ORDER BY created_at DESC LIMIT 20",
            )
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        stmt.query_map([], |row| {
            Ok(TaskSnapshot {
                id: row.get(0)?,
                verb: row.get(1)?,
                cell_id: row.get(2)?,
                description: row.get(3)?,
                target: row.get(4)?,
                status: row.get(5)?,
                created_at: row.get(6)?,
                completed_at: row.get(7)?,
            })
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .filter_map(|r| r.ok())
        .collect::<Vec<_>>()
    };

    // System metrics
    let metrics = crate::monitor::collect_metrics_snapshot().ok().map(|s| MetricsSnapshot {
        cpu_load_1m: s.cpu_load_1m,
        cpu_load_5m: s.cpu_load_5m,
        cpu_load_15m: s.cpu_load_15m,
        mem_total_kb: s.mem_total_kb,
        mem_available_kb: s.mem_available_kb,
        mem_used_percent: s.mem_used_percent,
        uptime_secs: s.uptime_secs,
    });

    // Journal events
    let journal = brain
        .journal
        .recent_events(30)
        .unwrap_or_default()
        .into_iter()
        .map(|e| JournalSnapshot {
            timestamp: e.timestamp.to_rfc3339(),
            kind: format!("{:?}", e.kind),
            cell_id: e.cell_id.unwrap_or_default(),
            task_id: e.task_id.map(|id| id.to_string()).unwrap_or_default(),
        })
        .collect();

    // LLM stats
    let llm = LlmSnapshot {
        provider: brain.llm.primary_provider_label().to_string(),
        queries: brain.llm.total_queries(),
        tokens: brain.llm.total_tokens(),
        budget_spent_usd: brain.llm.budget_spent_usd(),
    };

    // Memory count
    let memory_count = brain.memory.count().unwrap_or(0) as u64;

    // Wisdom
    let wisdom = brain.wisdom.as_ref().map(|w| {
        let entries = w
            .active_entries()
            .into_iter()
            .map(|e| WisdomEntrySnapshot {
                kind: format!("{}", e.kind),
                content: e.content.clone(),
                confidence: e.confidence,
            })
            .collect();
        let maturity = w
            .compute_maturity()
            .map(|l| l.name().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        WisdomSnapshot { entries, maturity }
    });

    // Soul
    let soul = std::fs::read_to_string(&brain.config.paths.soul_path).unwrap_or_default();

    // Scoreboard
    let scoreboard = {
        let db = brain.storage.db.lock();
        let mut stmt = db
            .prepare(
                "SELECT r.task_id, t.verb, t.target, r.composite, r.novelty, r.anomaly, \
                 r.confidence, r.actionability, r.scored_at, \
                 COALESCE(r.efficiency, 0.0) \
                 FROM task_rewards r \
                 JOIN tasks t ON r.task_id = t.id \
                 ORDER BY r.composite DESC LIMIT 10",
            )
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        stmt.query_map([], |row| {
            Ok(ScoreboardSnapshot {
                task_id: row.get(0)?,
                verb: row.get(1)?,
                target: row.get(2)?,
                composite: row.get(3)?,
                novelty: row.get(4)?,
                anomaly: row.get(5)?,
                confidence: row.get(6)?,
                actionability: row.get(7)?,
                scored_at: row.get(8)?,
                efficiency: row.get(9)?,
            })
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .filter_map(|r| r.ok())
        .collect::<Vec<_>>()
    };

    // Entities
    let entities = {
        let db = brain.storage.db.lock();
        let mut stmt = db
            .prepare(
                "SELECT name, kind, COALESCE(description, ''), COALESCE(confidence, 0.0), \
                 last_seen FROM entities ORDER BY last_seen DESC LIMIT 20",
            )
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        stmt.query_map([], |row| {
            Ok(EntitySnapshot {
                name: row.get(0)?,
                kind: row.get(1)?,
                description: row.get(2)?,
                confidence: row.get(3)?,
                last_seen: row.get(4)?,
            })
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .filter_map(|r| r.ok())
        .collect::<Vec<_>>()
    };

    // EWMA
    let ewma = brain
        .storage
        .kv_get("reward_ewma")
        .ok()
        .flatten()
        .and_then(|bytes| bytes.try_into().ok().map(f64::from_le_bytes))
        .unwrap_or(0.0);

    // Think interval
    let reward_cfg = &brain.config.autonomy.reward;
    let think_interval_ms = if reward_cfg.enabled {
        let min = reward_cfg.adaptive_min_interval_ms;
        let max = reward_cfg.adaptive_max_interval_ms;
        let ratio = ewma.clamp(0.0, 1.0);
        let interval = max as f64 - ratio * (max - min) as f64;
        interval as u64
    } else {
        brain.config.autonomy.think_interval_ms
    };

    Ok(Json(PortalSnapshot {
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: uptime,
        ewma,
        think_interval_ms,
        cells,
        tasks: TaskSummary {
            completed,
            pending,
            running,
            failed,
        },
        recent_tasks,
        metrics,
        journal,
        llm,
        memory_count,
        wisdom,
        soul,
        scoreboard,
        entities,
    }))
}
