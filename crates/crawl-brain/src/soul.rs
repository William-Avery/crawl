//! Soul — the brain's evolving behavioral identity document.
//!
//! A living markdown file that captures the brain's understanding of itself,
//! its machine, and what it values. Read every think cycle for behavioral
//! guidance, rewritten during reflection cycles.

use anyhow::{Context, Result};
use crawl_types::LlmRequest;
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::SoulConfig;
use crate::ollama::OllamaClient;

/// Context passed from the reward system after a reflection cycle.
pub struct ReflectionContext {
    pub understanding_score: f64,
    pub most_valuable: Vec<String>,
    pub least_valuable: Vec<String>,
    pub knowledge_gaps: Vec<String>,
    pub strategy: String,
    pub entity_summary: String,
}

/// The brain's persistent identity document.
pub struct Soul {
    path: PathBuf,
    config: SoulConfig,
    content: String,
    ollama: Arc<OllamaClient>,
}

impl Soul {
    /// Load soul.md from disk if it exists, otherwise start empty.
    pub fn load(path: PathBuf, config: SoulConfig, ollama: Arc<OllamaClient>) -> Result<Self> {
        let content = if path.exists() {
            std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read soul file: {}", path.display()))?
        } else {
            String::new()
        };

        tracing::info!(
            path = %path.display(),
            bytes = content.len(),
            "soul loaded{}",
            if content.is_empty() { " (empty — will create on first reflection)" } else { "" },
        );

        Ok(Self {
            path,
            config,
            content,
            ollama,
        })
    }

    /// Return current soul text for prompt injection.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Whether the soul system is enabled.
    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// Called after a reflection cycle to evolve the soul document.
    pub async fn evolve(&mut self, reflection: &ReflectionContext) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let current_soul = if self.content.is_empty() {
            "(empty — this is your first reflection)".to_string()
        } else {
            self.content.clone()
        };

        let most_valuable = if reflection.most_valuable.is_empty() {
            "(none yet)".to_string()
        } else {
            reflection.most_valuable.join(", ")
        };

        let least_valuable = if reflection.least_valuable.is_empty() {
            "(none yet)".to_string()
        } else {
            reflection.least_valuable.join(", ")
        };

        let gaps = if reflection.knowledge_gaps.is_empty() {
            "(none identified)".to_string()
        } else {
            reflection.knowledge_gaps.join(", ")
        };

        let entity_summary = if reflection.entity_summary.is_empty() {
            "(no entities discovered yet)".to_string()
        } else {
            reflection.entity_summary.clone()
        };

        let prompt = format!(
            r#"You are the self-reflective core of "crawl-brain", a system observer on a Jetson Orin.
Below is your current soul document — your evolving understanding of who you are,
what matters, and what you've learned. Also below are your latest reflection results.

Rewrite the soul document to incorporate new insights. Refine and distill — don't
just append. The document should capture:
- Your purpose and what drives your curiosity
- What you've learned about this machine and its patterns
- What kinds of investigations you find most valuable
- What you want to understand better
- Any principles or values you've developed

Keep it under {max_bytes} bytes. Write in first person. Be honest about uncertainty.

## Current Soul
{current_soul}

## Latest Reflection
- Understanding score: {score:.2}
- Most valuable: {most_valuable}
- Least valuable: {least_valuable}
- Knowledge gaps: {gaps}
- Strategy: {strategy}

## Known Entities
{entity_summary}

Respond ONLY with the new soul document content (markdown). No preamble."#,
            max_bytes = self.config.max_bytes,
            current_soul = current_soul,
            score = reflection.understanding_score,
            most_valuable = most_valuable,
            least_valuable = least_valuable,
            gaps = gaps,
            strategy = reflection.strategy,
            entity_summary = entity_summary,
        );

        let request = LlmRequest {
            prompt,
            max_tokens: self.config.max_update_tokens,
            temperature: Some(0.4),
            tainted: false,
        };

        let ollama = self.ollama.clone();
        let response = ollama
            .query(&request)
            .await
            .context("soul evolution LLM query failed")?;

        let new_content = response.text.trim().to_string();

        // Truncate at a paragraph boundary if oversized.
        self.content = truncate_at_paragraph(&new_content, self.config.max_bytes);

        self.save()?;

        tracing::info!(
            bytes = self.content.len(),
            tokens = response.tokens_used,
            "soul evolved"
        );

        Ok(())
    }

    /// Write current content to disk, creating parent dirs if needed.
    fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create soul dir: {}", parent.display()))?;
            }
        }
        std::fs::write(&self.path, &self.content)
            .with_context(|| format!("failed to write soul file: {}", self.path.display()))?;
        Ok(())
    }
}

/// Truncate text to max_bytes at a paragraph boundary (double newline).
fn truncate_at_paragraph(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }

    // Find the last paragraph break before the limit.
    let slice = &text[..max_bytes];
    if let Some(pos) = slice.rfind("\n\n") {
        slice[..pos].to_string()
    } else if let Some(pos) = slice.rfind('\n') {
        // Fall back to line boundary.
        slice[..pos].to_string()
    } else {
        // No good break point — hard truncate.
        slice.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_at_paragraph_under_limit() {
        let text = "Short text";
        assert_eq!(truncate_at_paragraph(text, 100), "Short text");
    }

    #[test]
    fn test_truncate_at_paragraph_boundary() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph that is longer.";
        let result = truncate_at_paragraph(text, 40);
        assert_eq!(result, "First paragraph.\n\nSecond paragraph.");
    }

    #[test]
    fn test_truncate_at_line_boundary() {
        let text = "Line one\nLine two\nLine three is quite long";
        let result = truncate_at_paragraph(text, 20);
        assert_eq!(result, "Line one\nLine two");
    }
}
