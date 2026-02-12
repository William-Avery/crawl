//! Guest-side SDK for crawl Cell (plugin) authors.
//!
//! This crate provides WIT bindings and helper utilities for writing
//! crawl plugins that compile to wasm32-wasip2.
//!
//! Each plugin crate should use this crate's `generate_bindings!` macro
//! or call `wit_bindgen::generate!` directly with `path: "../../wit"`.

// Generate bindings from WIT files, with public export macro.
wit_bindgen::generate!({
    world: "cell",
    path: "../../wit",
    pub_export_macro: true,
});

// Re-export generated types for plugin authors.
pub use exports::crawl::plugin::plugin_api;
pub use crawl::plugin::host_tools;
pub use crawl::plugin::llm_api;

/// Helper: read a file via the host tool and return as string.
pub fn read_file_string(path: &str, max_bytes: u32) -> Result<String, String> {
    let bytes = host_tools::read_file(path, max_bytes)?;
    String::from_utf8(bytes).map_err(|e| format!("invalid UTF-8: {e}"))
}

/// Helper: list processes via the host tool.
pub fn list_processes() -> Result<Vec<host_tools::ProcessInfo>, String> {
    host_tools::list_processes()
}

/// Helper: query the LLM with a simple prompt.
pub fn llm_query(prompt: &str, max_tokens: u32) -> Result<llm_api::LlmResponse, String> {
    llm_api::query(&llm_api::LlmRequest {
        prompt: prompt.to_string(),
        max_tokens,
        temperature: None,
    })
}

/// Helper: emit a journal event.
pub fn emit_event(kind: &str, payload: &serde_json::Value) -> Result<(), String> {
    let payload_str = serde_json::to_string(payload).map_err(|e| e.to_string())?;
    host_tools::emit_event(kind, &payload_str)
}

/// Helper: store a value in semantic memory.
pub fn memory_store(content: &str, metadata: &serde_json::Value) -> Result<String, String> {
    let metadata_str = serde_json::to_string(metadata).map_err(|e| e.to_string())?;
    host_tools::memory_store(content, &metadata_str)
}

/// Helper: search semantic memory.
pub fn memory_search(query: &str, top_k: u32) -> Result<Vec<host_tools::MemoryResult>, String> {
    host_tools::memory_search(query, top_k)
}
