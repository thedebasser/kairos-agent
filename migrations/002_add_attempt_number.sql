-- Kairos Agent — Add attempt_number to agent_runs
-- Migration: 002_add_attempt_number.sql
-- Finding 1.2: Track retry attempts per agent run.

ALTER TABLE agent_runs
    ADD COLUMN IF NOT EXISTS attempt_number INTEGER DEFAULT 1;

COMMENT ON COLUMN agent_runs.attempt_number IS '1-based retry counter for this step execution';
