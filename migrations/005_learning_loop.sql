-- Migration 005: Learning loop enhancements (AI Architecture Review §1–§6)
--
-- 1. Add new columns to training_examples:
--    - category: for per-category querying
--    - reasoning: design reasoning from the LLM
--    - thinking_content: extended thinking / chain-of-thought
--    - verified: final gate — examples NOT injected into prompts until True
--    - iteration_count: how many iterations it took to pass validation
--
-- 2. Add knowledge JSONB to category_stats for accumulated category learning
--
-- 3. Add index for fast verified-example lookups

-- training_examples: new columns
ALTER TABLE training_examples
    ADD COLUMN IF NOT EXISTS category VARCHAR(100),
    ADD COLUMN IF NOT EXISTS reasoning TEXT,
    ADD COLUMN IF NOT EXISTS thinking_content TEXT,
    ADD COLUMN IF NOT EXISTS verified BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS iteration_count INTEGER NOT NULL DEFAULT 1;

-- Index for querying verified examples by pipeline + category
CREATE INDEX IF NOT EXISTS idx_training_verified
    ON training_examples (pipeline, category, verified);

COMMENT ON COLUMN training_examples.verified IS
    'Final gate — examples are NOT used in prompts until verified=True by an operator';

COMMENT ON COLUMN training_examples.reasoning IS
    'LLM design reasoning explaining parameter choices';

COMMENT ON COLUMN training_examples.thinking_content IS
    'Extended thinking / chain-of-thought from the LLM generation step';

COMMENT ON COLUMN training_examples.iteration_count IS
    'Number of validation iterations before the simulation passed';

-- category_stats: accumulated knowledge JSONB
ALTER TABLE category_stats
    ADD COLUMN IF NOT EXISTS knowledge JSONB;

COMMENT ON COLUMN category_stats.knowledge IS
    'Accumulated per-category knowledge (parameter ranges, failure modes, etc.)';
