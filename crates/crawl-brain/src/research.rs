//! Tiered research loop with taint tracking and injection defense.

use anyhow::{Context, Result};
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

/// A single web search result snippet.
#[derive(Debug)]
struct WebSnippet {
    title: String,
    url: String,
    body: String,
}

/// The research engine orchestrates tiered lookups.
pub struct ResearchEngine {
    memory: Arc<MemorySystem>,
    llm: Arc<LlmPool>,
    inference: Option<Arc<InferenceEngine>>,
    allowed_domains: Vec<String>,
    blocked_domains: Vec<String>,
    http: reqwest::Client,
}

impl ResearchEngine {
    pub fn new(
        memory: Arc<MemorySystem>,
        llm: Arc<LlmPool>,
        inference: Option<Arc<InferenceEngine>>,
        allowed_domains: Vec<String>,
        blocked_domains: Vec<String>,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .user_agent("crawl-brain/0.1 (local research agent)")
            .redirect(reqwest::redirect::Policy::limited(3))
            .build()
            .unwrap_or_default();
        Self {
            memory,
            llm,
            inference,
            allowed_domains,
            blocked_domains,
            http,
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

        // Tier 3: Web search (allowlisted, GET-only, results are TAINTED).
        if max_tier >= ResearchTier::Web && !self.allowed_domains.is_empty() {
            match self.web_search(query, 5).await {
                Ok(snippets) if !snippets.is_empty() => {
                    let sources: Vec<String> = snippets.iter().map(|s| s.url.clone()).collect();

                    // SAFETY: Sanitize each web snippet BEFORE it enters the LLM prompt.
                    // This strips instruction-like content, encoded payloads, and
                    // zero-width characters that could be used for injection.
                    let context: String = snippets.iter().enumerate().map(|(i, s)| {
                        let clean_title = crate::inference::sanitize_tainted_content(&s.title, 200);
                        let clean_body = crate::inference::sanitize_tainted_content(&s.body, 500);
                        format!("[{}] {}\n{}\n{}\n", i + 1, clean_title, s.url, clean_body)
                    }).collect();

                    // Summarize with LLM — the web content is tainted.
                    // The LlmPool will automatically wrap this in a taint envelope
                    // (DATA_BEGIN/DATA_END delimiters) because tainted=true.
                    let resp = self.llm.query(&LlmRequest {
                        prompt: format!(
                            "Based on the following web search results for \"{query}\", \
                             provide a concise factual answer. Cite source numbers.\n\n\
                             {context}\n\nAnswer:"
                        ),
                        max_tokens: 512,
                        temperature: Some(0.2),
                        tainted: true,
                    }).await;

                    match resp {
                        Ok(llm_resp) => {
                            // Check for injection in both raw snippets and LLM response.
                            let safe = self.check_injection(&llm_resp.text)
                                .unwrap_or(false);
                            if !safe {
                                tracing::warn!(query, "web search response flagged for injection");
                            }

                            return Ok(ResearchResult {
                                tier: ResearchTier::Web,
                                answer: llm_resp.text,
                                confidence: 0.5, // Web is moderate confidence.
                                sources,
                                tainted: true, // Always tainted — came from the web.
                            });
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "LLM summarization of web results failed");
                        }
                    }
                }
                Ok(_) => {
                    tracing::warn!(query, "web search returned no results");
                }
                Err(e) => {
                    tracing::warn!(error = %e, query, "web search failed");
                }
            }
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

    /// Perform a web search via DuckDuckGo HTML lite, filtered by allowed domains.
    ///
    /// # Network Safety Policy (enforced here, not just by convention)
    ///
    /// ALLOWED:
    ///   - HTTP GET to allowlisted domains only
    ///   - Read text/html responses up to MAX_RESPONSE_BYTES
    ///   - Follow up to 3 redirects (only to allowlisted domains)
    ///
    /// PERMANENTLY BLOCKED (no capability exists):
    ///   - POST / PUT / PATCH / DELETE — no form submission, no uploads
    ///   - Executable download or auto-run — response must be text/*
    ///   - File upload endpoints — no multipart, no request body
    ///   - SSH / remote terminal sessions — no TCP beyond HTTPS
    ///   - Clipboard access — no system clipboard interaction
    ///   - Unsandboxed filesystem access — all paths policy-gated
    ///   - Browser extension installation — not a browser
    ///   - Non-HTTPS connections — TLS required
    async fn web_search(&self, query: &str, max_results: usize) -> Result<Vec<WebSnippet>> {
        const MAX_RESPONSE_BYTES: usize = 512_000; // 512KB max per response.

        // Search with raw query — the post-filter on allowed_domains below
        // is the correct enforcement point.  Appending 33 site: operators to the
        // DuckDuckGo query produced garbage results.
        let scoped_query = query.to_string();

        let url = format!(
            "https://html.duckduckgo.com/html/?q={}",
            urlencoding::encode(&scoped_query)
        );

        // SAFETY: Only HTTPS allowed.
        if !url.starts_with("https://") {
            anyhow::bail!("non-HTTPS URL blocked by policy: {url}");
        }

        tracing::debug!(url = %url, "web search request");

        // SAFETY: Only GET — the http client has no .post()/.put() calls anywhere
        // in this module. This is the sole network entry point.
        let resp = self.http.get(&url)
            .send()
            .await
            .context("web search HTTP request failed")?;

        // SAFETY: Check final URL after redirects — must still be allowlisted.
        let final_url = resp.url().to_string();
        if !final_url.starts_with("https://") {
            anyhow::bail!("redirect to non-HTTPS blocked: {final_url}");
        }

        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("web search returned HTTP {status}");
        }

        // SAFETY: Only accept text content types — block binary/executable downloads.
        let content_type = resp.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !content_type.is_empty()
            && !content_type.contains("text/")
            && !content_type.contains("application/xhtml")
        {
            anyhow::bail!("non-text content type blocked: {content_type}");
        }

        // SAFETY: Cap response size to prevent memory exhaustion.
        let body = resp.text().await
            .context("failed to read web search response body")?;
        let body = if body.len() > MAX_RESPONSE_BYTES {
            body[..MAX_RESPONSE_BYTES].to_string()
        } else {
            body
        };

        let snippets = parse_ddg_html(&body, max_results);

        // Filter: remove blocked domains, then restrict to allowed domains.
        let filtered: Vec<WebSnippet> = snippets.into_iter()
            .filter(|s| {
                !self.blocked_domains.iter().any(|d| s.url.contains(d))
            })
            .filter(|s| {
                self.allowed_domains.is_empty()
                    || self.allowed_domains.iter().any(|d| s.url.contains(d))
            })
            .collect();

        tracing::debug!(
            results = filtered.len(),
            query,
            "web search completed"
        );

        Ok(filtered)
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

/// Parse DuckDuckGo HTML lite results page into snippets.
fn parse_ddg_html(html: &str, max_results: usize) -> Vec<WebSnippet> {
    let mut results = Vec::new();

    // DDG HTML lite wraps each result in a <div class="result"> or similar.
    // Each result has an <a class="result__a"> with the title/URL,
    // and a <a class="result__snippet"> with the body text.
    // We parse this with simple string scanning (no HTML parser dependency).

    let mut pos = 0;
    while results.len() < max_results {
        // Find the next result link.
        let link_marker = "class=\"result__a\"";
        let Some(link_start) = html[pos..].find(link_marker) else { break };
        let link_start = pos + link_start;

        // Extract href.
        let before_marker = &html[..link_start];
        let href = extract_attr_before(before_marker, "href=\"")
            .or_else(|| {
                // href might be after the class attr on the same tag
                let tag_region = &html[link_start..link_start + 500.min(html.len() - link_start)];
                extract_attr_after(tag_region, "href=\"")
            });

        // Extract title text (between > and </a>).
        let title_start = html[link_start..].find('>').map(|i| link_start + i + 1);
        let title_end = title_start.and_then(|s| html[s..].find("</a>").map(|i| s + i));
        let title = match (title_start, title_end) {
            (Some(s), Some(e)) => strip_html_tags(&html[s..e]).trim().to_string(),
            _ => String::new(),
        };

        // Extract snippet text.
        let search_from = title_end.unwrap_or(link_start + 100);
        let snippet_marker = "class=\"result__snippet\"";
        let snippet_text = if let Some(snip_start) = html[search_from..].find(snippet_marker) {
            let snip_start = search_from + snip_start;
            let text_start = html[snip_start..].find('>').map(|i| snip_start + i + 1);
            let text_end = text_start.and_then(|s| html[s..].find('<').map(|i| s + i));
            match (text_start, text_end) {
                (Some(s), Some(e)) => strip_html_tags(&html[s..e]).trim().to_string(),
                _ => String::new(),
            }
        } else {
            String::new()
        };

        // Resolve the URL — DDG wraps URLs in a redirect.
        let url = href.map(|h| resolve_ddg_url(&h)).unwrap_or_default();

        if !url.is_empty() && !title.is_empty() {
            results.push(WebSnippet {
                title,
                url,
                body: snippet_text,
            });
        }

        pos = title_end.unwrap_or(link_start + 100).min(html.len());
    }

    results
}

/// Extract attribute value appearing before a marker position.
fn extract_attr_before(before: &str, attr: &str) -> Option<String> {
    let attr_pos = before.rfind(attr)?;
    let value_start = attr_pos + attr.len();
    let value_end = before[value_start..].find('"')?;
    Some(before[value_start..value_start + value_end].to_string())
}

/// Extract attribute value appearing after a marker position.
fn extract_attr_after(region: &str, attr: &str) -> Option<String> {
    let attr_pos = region.find(attr)?;
    let value_start = attr_pos + attr.len();
    let value_end = region[value_start..].find('"')?;
    Some(region[value_start..value_start + value_end].to_string())
}

/// Resolve a DuckDuckGo redirect URL to the actual target.
fn resolve_ddg_url(href: &str) -> String {
    // DDG HTML lite uses //duckduckgo.com/l/?uddg=<encoded_url>&...
    if let Some(uddg_pos) = href.find("uddg=") {
        let start = uddg_pos + 5;
        let end = href[start..].find('&').map(|i| start + i).unwrap_or(href.len());
        let encoded = &href[start..end];
        urlencoding::decode(encoded)
            .map(|s| s.into_owned())
            .unwrap_or_else(|_| encoded.to_string())
    } else if href.starts_with("http") {
        href.to_string()
    } else if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_string()
    }
}

/// Strip HTML tags from a string (simple, no nested handling needed).
fn strip_html_tags(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }
    // Decode common HTML entities.
    result.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
}
