//! Ollama HTTP client with rate limiting and taint tracking.

use anyhow::{bail, Context, Result};
use crawl_types::{LlmRequest, LlmResponse};
use governor::{Quota, RateLimiter};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::config::OllamaConfig;

/// Client for querying the local Ollama instance.
pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
    rate_limiter: RateLimiter<
        governor::state::NotKeyed,
        governor::state::InMemoryState,
        governor::clock::DefaultClock,
    >,
    total_queries: AtomicU64,
    total_tokens: AtomicU64,
}

/// Ollama /api/generate request body.
#[derive(Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    num_predict: u32,
}

/// Ollama /api/generate response body.
#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    prompt_eval_count: u32,
}

impl OllamaClient {
    pub fn new(config: &OllamaConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .expect("failed to build HTTP client");

        // Set up rate limiter.
        let rps = (config.rate_limit_rps.max(0.1)) as u32;
        let rps = rps.max(1);
        let quota = Quota::per_second(NonZeroU32::new(rps).unwrap());
        let rate_limiter = RateLimiter::direct(quota);

        Self {
            client,
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            rate_limiter,
            total_queries: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
        }
    }

    /// Query the Ollama LLM. Respects rate limiting.
    pub async fn query(&self, request: &LlmRequest) -> Result<LlmResponse> {
        // Wait for rate limiter.
        self.rate_limiter
            .until_ready()
            .await;

        let url = format!("{}/api/generate", self.base_url);

        let body = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt: request.prompt.clone(),
            stream: false,
            options: OllamaOptions {
                temperature: request.temperature,
                num_predict: request.max_tokens,
            },
        };

        tracing::debug!(
            model = %self.model,
            prompt_len = request.prompt.len(),
            max_tokens = request.max_tokens,
            tainted = request.tainted,
            "sending LLM query"
        );

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| "failed to send request to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Ollama returned {status}: {body}");
        }

        let ollama_resp: OllamaGenerateResponse = resp
            .json()
            .await
            .with_context(|| "failed to parse Ollama response")?;

        let tokens_used = ollama_resp.eval_count + ollama_resp.prompt_eval_count;

        // Update counters.
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        self.total_tokens
            .fetch_add(tokens_used as u64, Ordering::Relaxed);

        tracing::debug!(
            tokens_used,
            response_len = ollama_resp.response.len(),
            "LLM query completed"
        );

        Ok(LlmResponse {
            text: ollama_resp.response,
            tokens_used,
            tainted: request.tainted,
        })
    }

    /// Check if Ollama is reachable and the model is available.
    pub async fn health_check(&self) -> Result<()> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| "failed to reach Ollama")?;
        if !resp.status().is_success() {
            bail!("Ollama health check failed: {}", resp.status());
        }
        Ok(())
    }

    pub fn total_queries(&self) -> u64 {
        self.total_queries.load(Ordering::Relaxed)
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }
}
