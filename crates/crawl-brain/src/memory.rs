//! Semantic memory system backed by embeddings + vector search.

use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use rusqlite::Connection;
use std::sync::Arc;
use uuid::Uuid;

use crate::inference::InferenceEngine;

/// Semantic memory system for Tier 1 research (local memory).
pub struct MemorySystem {
    db: Arc<Mutex<Connection>>,
    inference: Option<Arc<InferenceEngine>>,
}

/// A memory entry with its embedding.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub metadata: serde_json::Value,
    pub created_at: String,
    pub similarity: Option<f64>,
}

impl MemorySystem {
    /// Create a new MemorySystem sharing the SQLite connection from Storage.
    pub fn new(
        db: Arc<Mutex<Connection>>,
        inference: Option<Arc<InferenceEngine>>,
    ) -> Result<Self> {
        Ok(Self { db, inference })
    }

    /// Store a memory entry with its embedding.
    pub fn store(&self, content: &str, metadata: serde_json::Value) -> Result<String> {
        let id = Uuid::now_v7().to_string();
        let now = Utc::now().to_rfc3339();

        // Generate embedding.
        let embedding = self.embed(content)?;
        let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        let db = self.db.lock();
        db.execute(
            "INSERT INTO memory_entries (id, content, embedding, metadata, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![id, content, embedding_bytes, serde_json::to_string(&metadata)?, now],
        )?;

        Ok(id)
    }

    /// Search memory by semantic similarity.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<MemoryEntry>> {
        let query_embedding = self.embed(query)?;

        let db = self.db.lock();
        let mut stmt =
            db.prepare("SELECT id, content, embedding, metadata, created_at FROM memory_entries")?;

        let mut entries: Vec<(MemoryEntry, f64)> = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let content: String = row.get(1)?;
                let embedding_bytes: Vec<u8> = row.get(2)?;
                let metadata_str: String = row.get(3)?;
                let created_at: String = row.get(4)?;
                Ok((id, content, embedding_bytes, metadata_str, created_at))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, content, embedding_bytes, metadata_str, created_at)| {
                let stored_embedding = bytes_to_f32_vec(&embedding_bytes);
                let similarity = cosine_similarity(&query_embedding, &stored_embedding);
                let metadata = serde_json::from_str(&metadata_str).unwrap_or_default();
                (
                    MemoryEntry {
                        id,
                        content,
                        metadata,
                        created_at,
                        similarity: Some(similarity),
                    },
                    similarity,
                )
            })
            .collect();

        // Sort by similarity descending, take top_k.
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(top_k);

        Ok(entries.into_iter().map(|(entry, _)| entry).collect())
    }

    /// Get total memory entries count.
    pub fn count(&self) -> Result<usize> {
        let db = self.db.lock();
        let count: i64 =
            db.query_row("SELECT COUNT(*) FROM memory_entries", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(ref eng) = self.inference {
            eng.embed_text(text)
        } else {
            Ok(simple_hash_embedding(text, 384))
        }
    }
}

/// Simple hash-based pseudo-embedding for fallback when ONNX models aren't loaded.
pub fn simple_hash_embedding(text: &str, dims: usize) -> Vec<f32> {
    use sha2::{Digest, Sha256};

    let mut result = vec![0.0f32; dims];
    let hash = Sha256::digest(text.as_bytes());

    for i in 0..dims {
        let chunk_idx = i % hash.len();
        let sign = if hash[chunk_idx] & (1 << (i % 8)) != 0 {
            1.0
        } else {
            -1.0
        };
        result[i] = sign * (hash[(i + chunk_idx) % hash.len()] as f32 / 255.0);
    }

    // L2 normalize.
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut result {
            *v /= norm;
        }
    }

    result
}

/// Convert byte slice to f32 vector.
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_same_vector() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn bytes_roundtrip() {
        let original = vec![1.0f32, 2.5, -3.7, 0.0];
        let bytes: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();
        let recovered = bytes_to_f32_vec(&bytes);
        assert_eq!(original, recovered);
    }

    #[test]
    fn hash_embedding_dimensions() {
        let emb = simple_hash_embedding("hello world", 384);
        assert_eq!(emb.len(), 384);
    }

    #[test]
    fn hash_embedding_normalized() {
        let emb = simple_hash_embedding("test string", 384);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
