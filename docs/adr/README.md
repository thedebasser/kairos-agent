# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) documenting the major technical choices made during Kairos Agent development.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-agent-orchestrator-separation.md) | Agent–Orchestrator Separation | Accepted | 2026-02-15 |
| [002](002-config-based-simulation.md) | Config-Based Simulation (Not Raw Code) | Accepted | 2026-02-20 |
| [003](003-local-first-llm-routing.md) | Local-First LLM Routing | Accepted | 2026-02-22 |
| [004](004-pluggable-tracing-sinks.md) | Pluggable Tracing Sinks | Accepted | 2026-03-01 |
| [005](005-step-level-caching.md) | Step-Level Input-Hash Caching | Accepted | 2026-03-01 |
| [006](006-prompt-templates-as-files.md) | Prompt Templates as Versioned Files | Accepted | 2026-02-18 |
| [007](007-two-tier-validation.md) | Two-Tier Validation Strategy | Accepted | 2026-02-20 |
| [008](008-pipeline-adapter-pattern.md) | Pipeline Adapter Pattern | Accepted | 2026-02-25 |

## Format

Each ADR follows a lightweight format:
- **Status** — Proposed, Accepted, Deprecated, Superseded
- **Context** — What problem we were solving
- **Decision** — What we chose
- **Consequences** — Tradeoffs and implications
