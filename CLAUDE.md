# CLAUDE.md

This file is for Claude Code — it provides context about this repository for AI-assisted development.

---

## Project: Policy-Driven Local Brain + Cells System

**Status:** Design phase — no code yet. This file is the authoritative design reference.

**Language:** Rust (Brain daemon), WASM (Cells/plugins)

---

## System Purpose

A single-machine adaptive architecture that observes the host, identifies unknowns, monitors health, procures diagnostic samples, builds/updates skills in sandbox, recommends actions, researches under strict policy, and evolves safely via controlled updates.

Properties: **local-first, offline-capable, policy-bound, budget-controlled, human-in-the-loop for irreversible actions, non-financial, non-authentication-interacting**.

---

## Architecture

### Brain (Supervisor Daemon — Rust)

Long-running daemon responsible for: task scheduling, timer enforcement, budget enforcement, capability gating, policy enforcement, memory management, research routing, skill proposal pipeline, version update pipeline, query API, tool-calling validation, monitoring & metrics, optional inter-brain metadata exchange.

**Invariants:**
- Never performs raw OS operations directly
- Only exposes controlled Tools
- Signs and enforces immutable runtime policy
- Treats all external content as tainted
- Fully observable and explainable

### Cells (WASM Plugins)

Constrained workers compiled to WASM.

**Constraints:** No direct OS access. Operate only via host-provided tools. Receive capability subsets from Brain. Cannot escalate permissions. Must obey time/resource budgets. Must checkpoint if incomplete.

**Requirements:** Return structured results. Journal all write operations. Never ignore timers. Never extend their own budgets.

---

## Core Task Verbs

Every Cell task must declare exactly one verb:

| Verb | Purpose | Key Rules |
|------|---------|-----------|
| **IDENTIFY** | Determine what an entity is (process, file, service, dependency, error pattern) | Returns hypothesis + evidence + confidence + suggested follow-up |
| **MONITOR** | Observe state over time (resources, logs, file drift, network listeners) | Returns snapshot + trend data + baseline updates + anomaly signals |
| **PROCURE** | Collect minimal diagnostic evidence | Minimal data only, redaction enforced, agent-owned storage, hash + provenance recorded |
| **MAINTAIN** | Keep agent ecosystem healthy (cache pruning, index rebuild, model rotation, GC, storage optimization) | Returns maintenance report + what changed + space reclaimed |
| **RESEARCH** | Deep-dive knowledge gathering from external sources | Tiered: memory → web (allowlisted) → LLM synthesis. All web content tainted. Returns findings + synthesis + confidence + sources |
| **BUILD** | Create new artifacts in sandbox (scripts, parsers, plugins) | Sandbox only, must include tests, no host execution. Returns artifact proposal + test results + risk classification + rollback plan |
| **UPDATE** | Propose modifications to skills or Brain code | Git diff only, sandbox CI, atomic deployment, rollback retained, user/policy approval required |

---

## Timer & Budget Model

Every Brain-to-Cell task includes:
- `time_budget_ms` OR `deadline_at`
- `max_tool_calls`
- `max_bytes_read` / `max_bytes_written`
- `max_network_calls`
- `risk_tier`

**Cells must:** stop on hard timeout, return `CHECKPOINT` if incomplete, never extend budgets.

**Checkpoint returns:** completed steps, findings, current stage, continuation plan, remaining estimate. Continuation fields: `continuation_id`, `cursor`, `state_blob_ref`. Checkpoint is normal behavior.

---

## Policy System

### Feature Flags (Brain-level)

```
network = false
cli_exec = true
ui_automation = false
credentials_access = false
self_update = true
peer_sync = false
```

**Disabled features do not exist** — they are not merely restricted, they are absent.

### Capability Enforcement
- Cells receive a subset of enabled features
- Cannot escalate or modify policy
- Brain copies inherit identical signed policy

---

## Surface Taxonomy

**Local OS:** Filesystem (scoped), process inventory, logs, hardware sensors. *Disabled by default:* UI automation, credential stores, system config mutation.

**Execution:** CLI execution (allowlisted), build/test sandbox.

**Network:** Allowlist-first, GET-only HTTP, no form submission, no authentication flows, no financial site interaction.

---

## Safety: Blocked Categories

Permanently blocked — no purchase/payment surface exists:
- Banking, brokerage, crypto exchanges
- Payment processors, tax portals
- Login/account flows

---

## Research Loop (Tiered)

1. **Tier 1** — Local memory
2. **Tier 2** — Offline references
3. **Tier 3** — Web (allowlisted only)
4. **Tier 4** — External model (minimal redacted context)
5. **Tier 5** — Books (last resort)

**Book policy:** Paid books always require explicit user approval, cannot be auto-purchased. Free material allowed earlier but must be justified. If gap resolves without book, mark `not_needed`.

---

## Prompt Injection Defense

All external content is **TAINTED**. Tainted content cannot: invoke tools, modify policy, expand capabilities, trigger purchases.

Architecture roles: **Researcher** (tainted), **Planner** (trusted), **Verifier** (trusted).

---

## Self-Update Pipeline

1. UPDATE verb produces git diff
2. Sandbox build + tests
3. Static + behavioral checks
4. Approval gate
5. Atomic version switch
6. Rollback retained
7. GC legacy versions

---

## Memory System

Brain manages persistent memory (referenced in Brain responsibilities: "Memory management"). Research results flow through tiered lookup. Cells checkpoint state for continuation.

## Query API

Brain exposes a query API — the system is queryable, tool-callable, and monitorable by external consumers.

## Observability

Brain provides monitoring & metrics. All Cell operations are journaled. All tool calls are validated and logged. The system is fully observable and explainable.
