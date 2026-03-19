-- Migration 004: Add thinking_content to agent_runs (Finding 1.3)
-- Stores LLM chain-of-thought reasoning so it can be reviewed without Langfuse.

ALTER TABLE agent_runs
    ADD COLUMN IF NOT EXISTS thinking_content TEXT;

COMMENT ON COLUMN agent_runs.thinking_content IS
    'Extended thinking / chain-of-thought from LLM calls (Finding 1.3)';
