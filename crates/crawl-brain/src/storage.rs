//! Storage layer: redb (KV) + rusqlite (relational).

use anyhow::{Context, Result};
use redb::{Database, TableDefinition};
use rusqlite::Connection;
use std::sync::Arc;
use parking_lot::Mutex;

use crate::config::StorageConfig;

// ── redb table definitions ──────────────────────────────────────────

/// Cell state / checkpoint storage.
const CELL_STATE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("cell_state");

/// Configuration cache.
const CONFIG_CACHE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("config_cache");

/// Generic KV store.
const KV_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("kv");

/// The unified storage backend.
pub struct Storage {
    /// redb key-value database.
    pub kv: Database,
    /// rusqlite relational database.
    pub db: Arc<Mutex<Connection>>,
}

impl Storage {
    /// Initialize storage, creating databases and tables as needed.
    pub fn init(config: &StorageConfig) -> Result<Self> {
        // Ensure parent directories exist.
        if let Some(parent) = config.redb_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create dir: {}", parent.display()))?;
        }
        if let Some(parent) = config.sqlite_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create dir: {}", parent.display()))?;
        }

        // Open redb.
        let kv = Database::create(&config.redb_path)
            .with_context(|| format!("failed to open redb at {}", config.redb_path.display()))?;

        // Create redb tables.
        {
            let write_txn = kv.begin_write()?;
            {
                let _ = write_txn.open_table(CELL_STATE_TABLE)?;
                let _ = write_txn.open_table(CONFIG_CACHE_TABLE)?;
                let _ = write_txn.open_table(KV_TABLE)?;
            }
            write_txn.commit()?;
        }

        // Open SQLite.
        let db = Connection::open(&config.sqlite_path)
            .with_context(|| format!("failed to open sqlite at {}", config.sqlite_path.display()))?;

        // Enable WAL mode for better concurrency.
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;

        // Create relational tables.
        db.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                verb TEXT NOT NULL,
                cell_id TEXT NOT NULL,
                description TEXT NOT NULL,
                target TEXT NOT NULL,
                params TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                budget TEXT NOT NULL DEFAULT '{}',
                result TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                confidence REAL,
                metadata TEXT NOT NULL DEFAULT '{}',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_results (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                tier INTEGER NOT NULL,
                result TEXT NOT NULL,
                tainted INTEGER NOT NULL DEFAULT 0,
                source TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS task_rewards (
                task_id TEXT PRIMARY KEY REFERENCES tasks(id),
                novelty REAL NOT NULL DEFAULT 0.0,
                anomaly REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 0.0,
                actionability REAL NOT NULL DEFAULT 0.0,
                composite REAL NOT NULL DEFAULT 0.0,
                scoring_method TEXT NOT NULL DEFAULT 'rule',
                scoring_detail TEXT NOT NULL DEFAULT '{}',
                entities_found INTEGER NOT NULL DEFAULT 0,
                scored_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_cell_id ON tasks(cell_id);
            CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_task_rewards_composite ON task_rewards(composite);
            CREATE INDEX IF NOT EXISTS idx_research_query ON research_results(query);
            ",
        )?;

        Ok(Self {
            kv,
            db: Arc::new(Mutex::new(db)),
        })
    }

    // ── redb KV operations ──────────────────────────────────────────

    /// Store a value in the KV store.
    pub fn kv_put(&self, key: &str, value: &[u8]) -> Result<()> {
        let write_txn = self.kv.begin_write()?;
        {
            let mut table = write_txn.open_table(KV_TABLE)?;
            table.insert(key, value)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get a value from the KV store.
    pub fn kv_get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let read_txn = self.kv.begin_read()?;
        let table = read_txn.open_table(KV_TABLE)?;
        Ok(table.get(key)?.map(|v| v.value().to_vec()))
    }

    /// Delete a value from the KV store.
    pub fn kv_delete(&self, key: &str) -> Result<bool> {
        let write_txn = self.kv.begin_write()?;
        let existed = {
            let mut table = write_txn.open_table(KV_TABLE)?;
            table.remove(key)?.is_some()
        };
        write_txn.commit()?;
        Ok(existed)
    }

    // ── Cell state operations ───────────────────────────────────────

    /// Store Cell checkpoint state.
    pub fn store_cell_state(&self, cell_id: &str, state: &[u8]) -> Result<()> {
        let write_txn = self.kv.begin_write()?;
        {
            let mut table = write_txn.open_table(CELL_STATE_TABLE)?;
            table.insert(cell_id, state)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Load Cell checkpoint state.
    pub fn load_cell_state(&self, cell_id: &str) -> Result<Option<Vec<u8>>> {
        let read_txn = self.kv.begin_read()?;
        let table = read_txn.open_table(CELL_STATE_TABLE)?;
        Ok(table.get(cell_id)?.map(|v| v.value().to_vec()))
    }
}
