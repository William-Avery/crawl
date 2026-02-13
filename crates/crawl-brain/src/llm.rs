//! LLM provider pool with cloud-first, local-fallback strategy.
//!
//! Supports multiple providers (Anthropic, Ollama) with ordered fallback.
//! Cloud providers are skipped when the daily budget is exhausted.

use anyhow::{bail, Context, Result};
use crawl_types::{LlmRequest, LlmResponse};
use governor::{Quota, RateLimiter};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

use crate::config::LlmConfig;

// ── Anthropic API types ─────────────────────────────────────────────

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ── Ollama API types ────────────────────────────────────────────────

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

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    prompt_eval_count: u32,
}

// ── Rate limiter type alias ─────────────────────────────────────────

type DirectRateLimiter = RateLimiter<
    governor::state::NotKeyed,
    governor::state::InMemoryState,
    governor::clock::DefaultClock,
>;

fn build_rate_limiter(rps: f64) -> DirectRateLimiter {
    let rps = (rps.max(0.1)) as u32;
    let rps = rps.max(1);
    let quota = Quota::per_second(NonZeroU32::new(rps).unwrap());
    RateLimiter::direct(quota)
}

// ── Provider ────────────────────────────────────────────────────────

enum Provider {
    Anthropic {
        client: Client,
        api_key: String,
        model: String,
        rate_limiter: DirectRateLimiter,
        input_price_per_mtok: u64,
        output_price_per_mtok: u64,
    },
    Ollama {
        client: Client,
        base_url: String,
        model: String,
        rate_limiter: DirectRateLimiter,
    },
}

impl Provider {
    fn is_paid(&self) -> bool {
        matches!(self, Provider::Anthropic { .. })
    }

    async fn query(&self, request: &LlmRequest) -> Result<LlmResponse> {
        match self {
            Provider::Anthropic {
                client,
                api_key,
                model,
                rate_limiter,
                input_price_per_mtok,
                output_price_per_mtok,
            } => {
                rate_limiter.until_ready().await;

                let body = AnthropicRequest {
                    model: model.clone(),
                    max_tokens: request.max_tokens,
                    messages: vec![AnthropicMessage {
                        role: "user",
                        content: request.prompt.clone(),
                    }],
                    temperature: request.temperature,
                };

                tracing::debug!(
                    model = %model,
                    prompt_len = request.prompt.len(),
                    max_tokens = request.max_tokens,
                    "sending Anthropic LLM query"
                );

                let resp = client
                    .post("https://api.anthropic.com/v1/messages")
                    .header("x-api-key", api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json")
                    .json(&body)
                    .send()
                    .await
                    .context("failed to send request to Anthropic")?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    bail!("Anthropic returned {status}: {body}");
                }

                let api_resp: AnthropicResponse = resp
                    .json()
                    .await
                    .context("failed to parse Anthropic response")?;

                let text = api_resp
                    .content
                    .into_iter()
                    .map(|c| c.text)
                    .collect::<Vec<_>>()
                    .join("");

                let tokens_used = api_resp.usage.input_tokens + api_resp.usage.output_tokens;

                // Compute cost in microdollars.
                let input_cost =
                    (api_resp.usage.input_tokens as u64 * input_price_per_mtok) / 1_000_000;
                let output_cost =
                    (api_resp.usage.output_tokens as u64 * output_price_per_mtok) / 1_000_000;
                let cost_microdollars = input_cost + output_cost;

                tracing::debug!(
                    tokens_used,
                    cost_microdollars,
                    response_len = text.len(),
                    "Anthropic query completed"
                );

                Ok(LlmResponse {
                    text,
                    tokens_used,
                    tainted: request.tainted,
                    cost_microdollars,
                })
            }

            Provider::Ollama {
                client,
                base_url,
                model,
                rate_limiter,
            } => {
                rate_limiter.until_ready().await;

                let url = format!("{}/api/generate", base_url);

                let body = OllamaGenerateRequest {
                    model: model.clone(),
                    prompt: request.prompt.clone(),
                    stream: false,
                    options: OllamaOptions {
                        temperature: request.temperature,
                        num_predict: request.max_tokens,
                    },
                };

                tracing::debug!(
                    model = %model,
                    prompt_len = request.prompt.len(),
                    max_tokens = request.max_tokens,
                    "sending Ollama LLM query"
                );

                let resp = client
                    .post(&url)
                    .json(&body)
                    .send()
                    .await
                    .context("failed to send request to Ollama")?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    bail!("Ollama returned {status}: {body}");
                }

                let ollama_resp: OllamaGenerateResponse = resp
                    .json()
                    .await
                    .context("failed to parse Ollama response")?;

                let tokens_used = ollama_resp.eval_count + ollama_resp.prompt_eval_count;

                tracing::debug!(
                    tokens_used,
                    response_len = ollama_resp.response.len(),
                    "Ollama query completed"
                );

                Ok(LlmResponse {
                    text: ollama_resp.response,
                    tokens_used,
                    tainted: request.tainted,
                    cost_microdollars: 0,
                })
            }
        }
    }
}

// ── Anthropic pricing ───────────────────────────────────────────────

fn anthropic_pricing(model: &str) -> (u64, u64) {
    // (input_price_per_mtok, output_price_per_mtok) in microdollars per 1M tokens
    if model.contains("opus") {
        (15_000, 75_000)
    } else if model.contains("haiku") {
        (250, 1_250)
    } else {
        // Default to Sonnet pricing
        (3_000, 15_000)
    }
}

// ── Budget Tracker ──────────────────────────────────────────────────

struct BudgetTracker {
    daily_limit_microdollars: u64,
    spent_microdollars: AtomicU64,
    reset_date: parking_lot::Mutex<chrono::NaiveDate>,
}

impl BudgetTracker {
    fn new(daily_budget_usd: f64) -> Self {
        let limit = (daily_budget_usd * 1_000_000.0) as u64;
        Self {
            daily_limit_microdollars: limit,
            spent_microdollars: AtomicU64::new(0),
            reset_date: parking_lot::Mutex::new(chrono::Utc::now().date_naive()),
        }
    }

    fn maybe_reset(&self) {
        let today = chrono::Utc::now().date_naive();
        let mut date = self.reset_date.lock();
        if *date < today {
            *date = today;
            self.spent_microdollars.store(0, Relaxed);
            tracing::info!("LLM daily budget reset");
        }
    }

    fn is_exhausted(&self) -> bool {
        if self.daily_limit_microdollars == 0 {
            return false; // no limit set (local-only config)
        }
        self.maybe_reset();
        self.spent_microdollars.load(Relaxed) >= self.daily_limit_microdollars
    }

    fn record_spend(&self, microdollars: u64) {
        self.spent_microdollars.fetch_add(microdollars, Relaxed);
    }

    fn spent_usd(&self) -> f64 {
        self.maybe_reset();
        self.spent_microdollars.load(Relaxed) as f64 / 1_000_000.0
    }
}

// ── LlmPool ────────────────────────────────────────────────────────

pub struct LlmPool {
    providers: Vec<(String, Provider)>,
    budget: BudgetTracker,
    total_queries: AtomicU64,
    total_tokens: AtomicU64,
}

impl LlmPool {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        let mut providers = Vec::new();

        for pc in &config.providers {
            let label = format!("{}:{}", pc.kind, pc.model);
            let provider = match pc.kind.as_str() {
                "anthropic" => {
                    let env_var = pc
                        .api_key_env
                        .as_deref()
                        .unwrap_or("ANTHROPIC_API_KEY");
                    let api_key = match std::env::var(env_var) {
                        Ok(k) if !k.is_empty() => k,
                        _ => {
                            tracing::warn!(
                                provider = %label,
                                env_var,
                                "API key not found, skipping provider"
                            );
                            continue;
                        }
                    };

                    let client = Client::builder()
                        .timeout(Duration::from_millis(pc.timeout_ms))
                        .build()
                        .context("failed to build HTTP client for Anthropic")?;

                    let (input_price, output_price) = anthropic_pricing(&pc.model);

                    Provider::Anthropic {
                        client,
                        api_key,
                        model: pc.model.clone(),
                        rate_limiter: build_rate_limiter(pc.rate_limit_rps),
                        input_price_per_mtok: input_price,
                        output_price_per_mtok: output_price,
                    }
                }
                "ollama" => {
                    let base_url = pc
                        .base_url
                        .clone()
                        .unwrap_or_else(|| "http://localhost:11434".into());

                    let client = Client::builder()
                        .timeout(Duration::from_millis(pc.timeout_ms))
                        .build()
                        .context("failed to build HTTP client for Ollama")?;

                    Provider::Ollama {
                        client,
                        base_url,
                        model: pc.model.clone(),
                        rate_limiter: build_rate_limiter(pc.rate_limit_rps),
                    }
                }
                other => {
                    tracing::warn!(kind = other, "unknown LLM provider kind, skipping");
                    continue;
                }
            };

            tracing::info!(provider = %label, "LLM provider registered");
            providers.push((label, provider));
        }

        if providers.is_empty() {
            bail!("no LLM providers could be initialized");
        }

        Ok(Self {
            providers,
            budget: BudgetTracker::new(config.daily_budget_usd),
            total_queries: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
        })
    }

    /// Query the first available LLM provider, falling back on failure or budget exhaustion.
    pub async fn query(&self, request: &LlmRequest) -> Result<LlmResponse> {
        let mut last_err = None;
        for (label, provider) in &self.providers {
            if provider.is_paid() && self.budget.is_exhausted() {
                tracing::debug!(provider = %label, "skipping: daily budget exhausted");
                continue;
            }
            match provider.query(request).await {
                Ok(resp) => {
                    if resp.cost_microdollars > 0 {
                        self.budget.record_spend(resp.cost_microdollars);
                    }
                    self.total_queries.fetch_add(1, Relaxed);
                    self.total_tokens.fetch_add(resp.tokens_used as u64, Relaxed);
                    tracing::debug!(
                        provider = %label,
                        tokens = resp.tokens_used,
                        cost_usd = format!("{:.6}", resp.cost_microdollars as f64 / 1_000_000.0),
                        "LLM query ok"
                    );
                    return Ok(resp);
                }
                Err(e) => {
                    tracing::warn!(provider = %label, error = %e, "LLM query failed, trying next");
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no LLM providers available")))
    }

    /// Check if any provider is reachable.
    pub async fn health_check(&self) -> Result<()> {
        for (label, provider) in &self.providers {
            match provider {
                Provider::Ollama { client, base_url, .. } => {
                    let url = format!("{}/api/tags", base_url);
                    match client.get(&url).send().await {
                        Ok(resp) if resp.status().is_success() => {
                            tracing::debug!(provider = %label, "health check ok");
                            return Ok(());
                        }
                        Ok(resp) => {
                            tracing::debug!(provider = %label, status = %resp.status(), "health check failed");
                        }
                        Err(e) => {
                            tracing::debug!(provider = %label, error = %e, "health check failed");
                        }
                    }
                }
                Provider::Anthropic { .. } => {
                    // Anthropic doesn't have a free health endpoint; assume available if key is set.
                    return Ok(());
                }
            }
        }
        bail!("no LLM providers are reachable")
    }

    pub fn total_queries(&self) -> u64 {
        self.total_queries.load(Relaxed)
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Relaxed)
    }

    pub fn budget_spent_usd(&self) -> f64 {
        self.budget.spent_usd()
    }

    /// Label of the first provider (for telemetry/config queries).
    pub fn primary_provider_label(&self) -> &str {
        self.providers
            .first()
            .map(|(l, _)| l.as_str())
            .unwrap_or("none")
    }

    /// Model name of the first Ollama provider (for backward compat with get_config).
    pub fn ollama_model(&self) -> Option<&str> {
        for (_, p) in &self.providers {
            if let Provider::Ollama { model, .. } = p {
                return Some(model.as_str());
            }
        }
        None
    }
}
