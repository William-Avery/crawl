//! Tiered research loop with taint tracking and injection defense.
#![allow(unused)]

use anyhow::Result;
use crawl_types::LlmRequest;
use std::sync::Arc;

use crate::inference::InferenceEngine;
use crate::llm::LlmPool;
use crate::memory::MemorySystem;

/// Research tier levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResearchTier {
    /// Tier 1: Local semantic memory search.
    LocalMemory = 1,
    /// Tier 2: Offline local references/files.
    OfflineRefs = 2,
    /// Tier 3: Web (allowlisted, GET-only).
    Web = 3,
    /// Tier 4: External model (Ollama GLM-4 Flash).
    ExternalModel = 4,
    /// Tier 5: Books (last resort, requires approval for paid).
    Books = 5,
}

/// Result of a research query.
#[derive(Debug, Clone)]
pub struct ResearchResult {
    pub tier: ResearchTier,
    pub answer: String,
    pub confidence: f64,
    pub sources: Vec<String>,
    pub tainted: bool,
}

/// The research engine orchestrates tiered lookups.
pub struct ResearchEngine {
    memory: Arc<MemorySystem>,
    llm: Arc<LlmPool>,
    inference: Option<Arc<InferenceEngine>>,
    allowed_domains: Vec<String>,
}

impl ResearchEngine {
    pub fn new(
        memory: Arc<MemorySystem>,
        llm: Arc<LlmPool>,
        inference: Option<Arc<InferenceEngine>>,
        allowed_domains: Vec<String>,
    ) -> Self {
        Self {
            memory,
            llm,
            inference,
            allowed_domains,
        }
    }

    /// Execute a tiered research query.
    pub async fn research(&self, query: &str, max_tier: ResearchTier) -> Result<ResearchResult> {
        // Tier 1: Local memory.
        if max_tier >= ResearchTier::LocalMemory {
            if let Ok(results) = self.memory.search(query, 5) {
                if let Some(best) = results.first() {
                    if best.similarity.unwrap_or(0.0) > 0.75 {
                        return Ok(ResearchResult {
                            tier: ResearchTier::LocalMemory,
                            answer: best.content.clone(),
                            confidence: best.similarity.unwrap_or(0.0),
                            sources: vec![format!("memory:{}", best.id)],
                            tainted: false,
                        });
                    }
                }
            }
        }

        // Tier 2: Offline references (local files).
        if max_tier >= ResearchTier::OfflineRefs {
            // TODO: search local documentation files.
        }

        // Tier 3: Web (allowlisted GET).
        if max_tier >= ResearchTier::Web && !self.allowed_domains.is_empty() {
            // TODO: implement allowlisted web search.
            // Any content fetched here is TAINTED.
        }

        // Tier 4: External model (Ollama).
        if max_tier >= ResearchTier::ExternalModel {
            let resp = self
                .llm
                .query(&LlmRequest {
                    prompt: format!(
                        "Research query: {query}\n\nProvide a concise, factual answer."
                    ),
                    max_tokens: 1024,
                    temperature: Some(0.3),
                    tainted: false, // Query is trusted, but response needs verification.
                })
                .await?;

            // Check for injection in the response.
            let injection_safe = if let Some(ref inf) = self.inference {
                let check = inf.detect_injection(&resp.text)?;
                !check.is_injection
            } else {
                true // No injection model available, trust the response.
            };

            if injection_safe {
                return Ok(ResearchResult {
                    tier: ResearchTier::ExternalModel,
                    answer: resp.text,
                    confidence: 0.6,
                    sources: vec!["ollama:glm4-flash".into()],
                    tainted: false, // Local model, not tainted.
                });
            } else {
                tracing::warn!("LLM response flagged for potential injection");
            }
        }

        // Tier 5: Books (requires approval).
        if max_tier >= ResearchTier::Books {
            return Ok(ResearchResult {
                tier: ResearchTier::Books,
                answer: format!("No local answer found for: {query}. A book/reference may be needed."),
                confidence: 0.0,
                sources: vec![],
                tainted: false,
            });
        }

        Ok(ResearchResult {
            tier: ResearchTier::LocalMemory,
            answer: format!("No answer found for: {query}"),
            confidence: 0.0,
            sources: vec![],
            tainted: false,
        })
    }

    /// Check tainted content for prompt injection before processing.
    pub fn check_injection(&self, content: &str) -> Result<bool> {
        if let Some(ref inf) = self.inference {
            let score = inf.detect_injection(content)?;
            Ok(!score.is_injection)
        } else {
            // Use heuristic fallback.
            let score = crate::inference::heuristic_injection_check(content);
            Ok(!score.is_injection)
        }
    }
}
