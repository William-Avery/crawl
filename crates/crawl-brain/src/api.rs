//! gRPC API server for the Brain daemon.

use anyhow::Result;
use std::sync::Arc;
use tonic::{Request, Response, Status};

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
            total_tasks_completed: 0,
            ollama_queries: self.brain.ollama.total_queries(),
            ollama_tokens: self.brain.ollama.total_tokens(),
        }))
    }

    async fn submit_task(
        &self,
        request: Request<pb::SubmitTaskRequest>,
    ) -> Result<Response<pb::SubmitTaskResponse>, Status> {
        let req = request.into_inner();
        let task_id = uuid::Uuid::now_v7().to_string();

        tracing::info!(
            task_id,
            verb = req.verb,
            cell_id = req.cell_id,
            "task submitted via API"
        );

        Ok(Response::new(pb::SubmitTaskResponse { task_id }))
    }

    async fn get_task(
        &self,
        request: Request<pb::GetTaskRequest>,
    ) -> Result<Response<pb::GetTaskResponse>, Status> {
        let req = request.into_inner();
        Ok(Response::new(pb::GetTaskResponse {
            task: Some(pb::TaskInfo {
                id: req.task_id,
                verb: String::new(),
                cell_id: String::new(),
                description: String::new(),
                target: String::new(),
                status: "unknown".to_string(),
                created_at: String::new(),
                completed_at: String::new(),
                result_json: String::new(),
            }),
        }))
    }

    async fn list_tasks(
        &self,
        _request: Request<pb::ListTasksRequest>,
    ) -> Result<Response<pb::ListTasksResponse>, Status> {
        Ok(Response::new(pb::ListTasksResponse { tasks: vec![] }))
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
        _request: Request<pb::ResearchRequest>,
    ) -> Result<Response<pb::ResearchResponse>, Status> {
        Ok(Response::new(pb::ResearchResponse {
            tier_used: 1,
            answer: "research not yet connected".to_string(),
            confidence: 0.0,
            sources: vec![],
            tainted: false,
        }))
    }
}

/// Start the gRPC API server.
pub async fn serve(brain: Arc<BrainState>) -> Result<()> {
    let port = brain.config.api.grpc_web_port.unwrap_or(9090);

    let service = BrainServiceImpl {
        brain: brain.clone(),
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
