//! Append-only event journal with postcard serialization.

use anyhow::{Context, Result};
use chrono::Utc;
use crawl_types::{JournalEvent, JournalEventKind};
use parking_lot::Mutex;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Append-only event journal.
pub struct Journal {
    dir: PathBuf,
    writer: Mutex<BufWriter<File>>,
    current_file: Mutex<PathBuf>,
    max_file_size: u64,
}

impl Journal {
    /// Open or create a journal in the given directory.
    pub fn open(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create journal dir: {}", dir.display()))?;

        let file_path = dir.join(format!("journal-{}.jsonl", Utc::now().format("%Y%m%d-%H%M%S")));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .with_context(|| format!("failed to open journal file: {}", file_path.display()))?;

        Ok(Self {
            dir: dir.to_path_buf(),
            writer: Mutex::new(BufWriter::new(file)),
            current_file: Mutex::new(file_path),
            max_file_size: 64 * 1024 * 1024, // 64 MiB
        })
    }

    /// Emit a journal event.
    pub fn emit(
        &self,
        kind: JournalEventKind,
        task_id: Option<Uuid>,
        cell_id: Option<String>,
        payload: serde_json::Value,
    ) -> Result<()> {
        let event = JournalEvent {
            id: Uuid::now_v7(),
            timestamp: Utc::now(),
            kind,
            task_id,
            cell_id,
            payload,
        };

        let line = serde_json::to_string(&event)?;

        let mut writer = self.writer.lock();
        writeln!(writer, "{line}")?;
        writer.flush()?;

        // Check if rotation is needed.
        drop(writer);
        self.maybe_rotate()?;

        Ok(())
    }

    /// Rotate the journal file if it exceeds max size.
    fn maybe_rotate(&self) -> Result<()> {
        let current = self.current_file.lock();
        let metadata = fs::metadata(&*current)?;
        if metadata.len() < self.max_file_size {
            return Ok(());
        }
        drop(current);

        let new_path = self.dir.join(format!(
            "journal-{}.jsonl",
            Utc::now().format("%Y%m%d-%H%M%S")
        ));
        let new_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&new_path)?;

        let mut writer = self.writer.lock();
        *writer = BufWriter::new(new_file);
        let mut current = self.current_file.lock();
        *current = new_path;

        tracing::info!("journal rotated");
        Ok(())
    }

    /// Read recent events from the journal (most recent file, last N lines).
    pub fn recent_events(&self, limit: usize) -> Result<Vec<JournalEvent>> {
        let current = self.current_file.lock().clone();
        let content = fs::read_to_string(&current)?;
        let events: Vec<JournalEvent> = content
            .lines()
            .rev()
            .take(limit)
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        Ok(events)
    }
}
