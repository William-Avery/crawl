//! gRPC API server for the Brain daemon.

use anyhow::Result;
use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::research::ResearchTier;
use crate::scheduler::{Priority, SchedulerSender};
use crate::BrainState;

// Include the generated protobuf code.
pub mod pb {
    tonic::include_proto!("crawl.v1");

    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("crawl_descriptor");
}

use pb::brain_service_server::{BrainService, BrainServiceServer};

/// gRPC service implementation.
struct BrainServiceImpl {
    brain: Arc<BrainState>,
    scheduler_sender: SchedulerSender,
    start_time: std::time::Instant,
}

#[tonic::async_trait]
impl BrainService for BrainServiceImpl {
    async fn health_check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        Ok(Response::new(pb::HealthCheckResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            status: "ok".to_string(),
            uptime_secs: self.start_time.elapsed().as_secs(),
            loaded_cells: self.brain.engine.list_plugins().len() as u32,
            total_tasks_completed: count_tasks_by_status(&self.brain, "completed"),
            ollama_queries: self.brain.llm.total_queries(),
            ollama_tokens: self.brain.llm.total_tokens(),
        }))
    }

    async fn submit_task(
        &self,
        request: Request<pb::SubmitTaskRequest>,
    ) -> Result<Response<pb::SubmitTaskResponse>, Status> {
        let req = request.into_inner();
        let task_id = uuid::Uuid::now_v7();
        let now = chrono::Utc::now();

        // Parse verb.
        let verb = match req.verb.to_uppercase().as_str() {
            "IDENTIFY" => crawl_types::TaskVerb::Identify,
            "MONITOR" => crawl_types::TaskVerb::Monitor,
            "PROCURE" => crawl_types::TaskVerb::Procure,
            "MAINTAIN" => crawl_types::TaskVerb::Maintain,
            "BUILD" => crawl_types::TaskVerb::Build,
            "UPDATE" => crawl_types::TaskVerb::Update,
            "CRUD" => crawl_types::TaskVerb::Crud,
            other => return Err(Status::invalid_argument(format!("unknown verb: {other}"))),
        };

        // Parse priority.
        let priority = match req.priority.to_lowercase().as_str() {
            "high" => Priority::High,
            "low" => Priority::Low,
            _ => Priority::Normal,
        };

        // Parse budget (from proto Budget or use defaults).
        let budget = if let Some(pb_budget) = req.budget {
            crawl_types::Budget {
                time_budget_ms: if pb_budget.time_budget_ms > 0 { Some(pb_budget.time_budget_ms) } else { Some(30_000) },
                deadline_at: None,
                max_tool_calls: if pb_budget.max_tool_calls > 0 { pb_budget.max_tool_calls } else { 100 },
                max_bytes_read: if pb_budget.max_bytes_read > 0 { pb_budget.max_bytes_read } else { 10 * 1024 * 1024 },
                max_bytes_written: if pb_budget.max_bytes_written > 0 { pb_budget.max_bytes_written } else { 1024 * 1024 },
                max_network_calls: pb_budget.max_network_calls,
                max_llm_calls: pb_budget.max_llm_calls,
                max_tokens_per_call: if pb_budget.max_tokens_per_call > 0 { pb_budget.max_tokens_per_call } else { 2048 },
                risk_tier: match pb_budget.risk_tier.as_str() {
                    "medium" => crawl_types::RiskTier::Medium,
                    "high" => crawl_types::RiskTier::High,
                    "critical" => crawl_types::RiskTier::Critical,
                    _ => crawl_types::RiskTier::Low,
                },
            }
        } else {
            crawl_types::Budget::default()
        };

        let params: serde_json::Value = serde_json::from_str(&req.params_json).unwrap_or_default();

        let task = crawl_types::Task {
            id: task_id,
            verb,
            description: req.description.clone(),
            target: req.target.clone(),
            params: params.clone(),
            budget: budget.clone(),
            cell_id: req.cell_id.clone(),
            created_at: now,
            continuation: None,
        };

        // Persist to SQLite.
        {
            let db = self.brain.storage.db.lock();
            db.execute(
                "INSERT INTO tasks (id, verb, cell_id, description, target, params, status, budget, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    task_id.to_string(),
                    format!("{verb:?}"),
                    &req.cell_id,
                    &req.description,
                    &req.target,
                    serde_json::to_string(&params).unwrap_or_default(),
                    "pending",
                    serde_json::to_string(&budget).unwrap_or_default(),
                    now.to_rfc3339(),
                ],
            ).map_err(|e| Status::internal(format!("failed to persist task: {e}")))?;
        }

        // Submit to scheduler.
        self.scheduler_sender.submit(task, priority).await
            .map_err(|e| Status::internal(format!("failed to submit task: {e}")))?;

        tracing::info!(
            task_id = %task_id,
            verb = ?verb,
            cell_id = %req.cell_id,
            "task submitted via API"
        );

        Ok(Response::new(pb::SubmitTaskResponse {
            task_id: task_id.to_string(),
        }))
    }

    async fn get_task(
        &self,
        request: Request<pb::GetTaskRequest>,
    ) -> Result<Response<pb::GetTaskResponse>, Status> {
        let req = request.into_inner();

        let db = self.brain.storage.db.lock();
        let result = db.query_row(
            "SELECT id, verb, cell_id, description, target, status, created_at, completed_at, result FROM tasks WHERE id = ?1",
            rusqlite::params![req.task_id],
            |row| {
                Ok(pb::TaskInfo {
                    id: row.get::<_, String>(0)?,
                    verb: row.get::<_, String>(1)?,
                    cell_id: row.get::<_, String>(2)?,
                    description: row.get::<_, String>(3)?,
                    target: row.get::<_, String>(4)?,
                    status: row.get::<_, String>(5)?,
                    created_at: row.get::<_, String>(6)?,
                    completed_at: row.get::<_, Option<String>>(7)?.unwrap_or_default(),
                    result_json: row.get::<_, Option<String>>(8)?.unwrap_or_default(),
                })
            },
        );

        match result {
            Ok(task_info) => Ok(Response::new(pb::GetTaskResponse { task: Some(task_info) })),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(Status::not_found(format!("task '{}' not found", req.task_id)))
            }
            Err(e) => Err(Status::internal(format!("database error: {e}"))),
        }
    }

    async fn list_tasks(
        &self,
        request: Request<pb::ListTasksRequest>,
    ) -> Result<Response<pb::ListTasksResponse>, Status> {
        let req = request.into_inner();
        let limit = if req.limit == 0 { 50u32 } else { req.limit };

        let db = self.brain.storage.db.lock();

        let tasks = if req.status_filter.is_empty() {
            let mut stmt = db.prepare(
                "SELECT id, verb, cell_id, description, target, status, created_at, completed_at, result FROM tasks ORDER BY created_at DESC LIMIT ?1"
            ).map_err(|e| Status::internal(e.to_string()))?;
            query_task_rows(&mut stmt, rusqlite::params![limit])
        } else {
            let mut stmt = db.prepare(
                "SELECT id, verb, cell_id, description, target, status, created_at, completed_at, result FROM tasks WHERE status = ?1 ORDER BY created_at DESC LIMIT ?2"
            ).map_err(|e| Status::internal(e.to_string()))?;
            query_task_rows(&mut stmt, rusqlite::params![req.status_filter, limit])
        };

        Ok(Response::new(pb::ListTasksResponse { tasks }))
    }

    async fn list_cells(
        &self,
        _request: Request<pb::ListCellsRequest>,
    ) -> Result<Response<pb::ListCellsResponse>, Status> {
        let plugins = self.brain.engine.list_plugins();
        let cells: Vec<pb::CellInfo> = plugins
            .into_iter()
            .map(|cell_id| {
                let caps = self
                    .brain
                    .engine
                    .plugin_capabilities(&cell_id)
                    .unwrap_or_default()
                    .iter()
                    .map(|c| format!("{c:?}"))
                    .collect();
                pb::CellInfo {
                    cell_id: cell_id.clone(),
                    name: cell_id,
                    version: "0.1.0".to_string(),
                    capabilities: caps,
                    supported_verbs: vec![],
                }
            })
            .collect();

        Ok(Response::new(pb::ListCellsResponse { cells }))
    }

    async fn search_memory(
        &self,
        request: Request<pb::SearchMemoryRequest>,
    ) -> Result<Response<pb::SearchMemoryResponse>, Status> {
        let req = request.into_inner();
        let top_k = if req.top_k == 0 { 10 } else { req.top_k as usize };

        match self.brain.memory.search(&req.query, top_k) {
            Ok(results) => {
                let entries = results
                    .into_iter()
                    .map(|e| pb::MemoryEntry {
                        id: e.id,
                        content: e.content,
                        similarity: e.similarity.unwrap_or(0.0),
                        metadata_json: serde_json::to_string(&e.metadata).unwrap_or_default(),
                        created_at: e.created_at,
                    })
                    .collect();
                Ok(Response::new(pb::SearchMemoryResponse { entries }))
            }
            Err(e) => Err(Status::internal(format!("memory search failed: {e}"))),
        }
    }

    async fn store_memory(
        &self,
        request: Request<pb::StoreMemoryRequest>,
    ) -> Result<Response<pb::StoreMemoryResponse>, Status> {
        let req = request.into_inner();
        let metadata: serde_json::Value =
            serde_json::from_str(&req.metadata_json).unwrap_or_default();

        match self.brain.memory.store(&req.content, metadata) {
            Ok(id) => Ok(Response::new(pb::StoreMemoryResponse { id })),
            Err(e) => Err(Status::internal(format!("memory store failed: {e}"))),
        }
    }

    async fn get_journal_events(
        &self,
        request: Request<pb::GetJournalEventsRequest>,
    ) -> Result<Response<pb::GetJournalEventsResponse>, Status> {
        let req = request.into_inner();
        let limit = if req.limit == 0 { 50 } else { req.limit as usize };

        match self.brain.journal.recent_events(limit) {
            Ok(events) => {
                let pb_events = events
                    .into_iter()
                    .map(|e| pb::JournalEvent {
                        id: e.id.to_string(),
                        timestamp: e.timestamp.to_rfc3339(),
                        kind: format!("{:?}", e.kind),
                        task_id: e.task_id.map(|id| id.to_string()).unwrap_or_default(),
                        cell_id: e.cell_id.unwrap_or_default(),
                        payload_json: serde_json::to_string(&e.payload).unwrap_or_default(),
                    })
                    .collect();
                Ok(Response::new(pb::GetJournalEventsResponse {
                    events: pb_events,
                }))
            }
            Err(e) => Err(Status::internal(format!("journal read failed: {e}"))),
        }
    }

    async fn get_metrics(
        &self,
        _request: Request<pb::GetMetricsRequest>,
    ) -> Result<Response<pb::GetMetricsResponse>, Status> {
        match crate::monitor::collect_metrics_snapshot() {
            Ok(snap) => Ok(Response::new(pb::GetMetricsResponse {
                cpu_load_1m: snap.cpu_load_1m,
                cpu_load_5m: snap.cpu_load_5m,
                cpu_load_15m: snap.cpu_load_15m,
                mem_total_kb: snap.mem_total_kb,
                mem_available_kb: snap.mem_available_kb,
                mem_used_percent: snap.mem_used_percent,
                uptime_secs: snap.uptime_secs,
            })),
            Err(e) => Err(Status::internal(format!("metrics collection failed: {e}"))),
        }
    }

    async fn research(
        &self,
        request: Request<pb::ResearchRequest>,
    ) -> Result<Response<pb::ResearchResponse>, Status> {
        let req = request.into_inner();

        let max_tier = match req.max_tier {
            0 | 1 => ResearchTier::LocalMemory,
            2 => ResearchTier::OfflineRefs,
            3 => ResearchTier::Web,
            4 => ResearchTier::ExternalModel,
            _ => ResearchTier::Books,
        };

        match self.brain.research.research(&req.query, max_tier).await {
            Ok(result) => Ok(Response::new(pb::ResearchResponse {
                tier_used: result.tier as u32,
                answer: result.answer,
                confidence: result.confidence,
                sources: result.sources,
                tainted: result.tainted,
            })),
            Err(e) => Err(Status::internal(format!("research failed: {e}"))),
        }
    }
}

// ── Helper Functions ────────────────────────────────────────────────

fn query_task_rows(stmt: &mut rusqlite::Statement, params: impl rusqlite::Params) -> Vec<pb::TaskInfo> {
    stmt.query_map(params, |row| {
        Ok(pb::TaskInfo {
            id: row.get(0)?,
            verb: row.get(1)?,
            cell_id: row.get(2)?,
            description: row.get(3)?,
            target: row.get(4)?,
            status: row.get(5)?,
            created_at: row.get(6)?,
            completed_at: row.get::<_, Option<String>>(7)?.unwrap_or_default(),
            result_json: row.get::<_, Option<String>>(8)?.unwrap_or_default(),
        })
    })
    .map(|rows| rows.filter_map(|r| r.ok()).collect())
    .unwrap_or_default()
}

fn count_tasks_by_status(brain: &BrainState, status: &str) -> u64 {
    let db = brain.storage.db.lock();
    db.query_row(
        "SELECT COUNT(*) FROM tasks WHERE status = ?1",
        rusqlite::params![status],
        |row| row.get::<_, i64>(0),
    )
    .unwrap_or(0) as u64
}

/// Start the gRPC API server.
pub async fn serve(brain: Arc<BrainState>, scheduler_sender: SchedulerSender) -> Result<()> {
    let port = brain.config.api.grpc_web_port.unwrap_or(9090);

    let service = BrainServiceImpl {
        brain: brain.clone(),
        scheduler_sender,
        start_time: std::time::Instant::now(),
    };

    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse()?;

    tracing::info!(%addr, "gRPC API server starting");

    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(pb::FILE_DESCRIPTOR_SET)
        .build_v1()?;

    tonic::transport::Server::builder()
        .accept_http1(true)
        .layer(tonic_web::GrpcWebLayer::new())
        .add_service(reflection)
        .add_service(BrainServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
