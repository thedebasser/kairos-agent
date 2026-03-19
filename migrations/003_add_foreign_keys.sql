-- Kairos Agent — Add foreign key constraints
-- Migration: 003_add_foreign_keys.sql
-- Finding 2.4: Orphaned records accumulate without FK constraints.
-- Run AFTER 002_add_attempt_number.sql.

-- video_ideas → pipeline_runs
ALTER TABLE video_ideas
    DROP CONSTRAINT IF EXISTS fk_ideas_pipeline_run,
    ADD CONSTRAINT fk_ideas_pipeline_run
        FOREIGN KEY (pipeline_run_id)
        REFERENCES pipeline_runs(pipeline_run_id)
        ON DELETE CASCADE;

-- simulations → pipeline_runs
ALTER TABLE simulations
    DROP CONSTRAINT IF EXISTS fk_simulations_pipeline_run,
    ADD CONSTRAINT fk_simulations_pipeline_run
        FOREIGN KEY (pipeline_run_id)
        REFERENCES pipeline_runs(pipeline_run_id)
        ON DELETE CASCADE;

-- simulations → video_ideas
ALTER TABLE simulations
    DROP CONSTRAINT IF EXISTS fk_simulations_idea,
    ADD CONSTRAINT fk_simulations_idea
        FOREIGN KEY (idea_id)
        REFERENCES video_ideas(idea_id)
        ON DELETE CASCADE;

-- outputs → pipeline_runs
ALTER TABLE outputs
    DROP CONSTRAINT IF EXISTS fk_outputs_pipeline_run,
    ADD CONSTRAINT fk_outputs_pipeline_run
        FOREIGN KEY (pipeline_run_id)
        REFERENCES pipeline_runs(pipeline_run_id)
        ON DELETE CASCADE;

-- outputs → simulations
ALTER TABLE outputs
    DROP CONSTRAINT IF EXISTS fk_outputs_simulation,
    ADD CONSTRAINT fk_outputs_simulation
        FOREIGN KEY (simulation_id)
        REFERENCES simulations(simulation_id)
        ON DELETE CASCADE;

-- agent_runs → pipeline_runs
ALTER TABLE agent_runs
    DROP CONSTRAINT IF EXISTS fk_agent_runs_pipeline_run,
    ADD CONSTRAINT fk_agent_runs_pipeline_run
        FOREIGN KEY (pipeline_run_id)
        REFERENCES pipeline_runs(pipeline_run_id)
        ON DELETE CASCADE;

-- publish_queue → outputs
ALTER TABLE publish_queue
    DROP CONSTRAINT IF EXISTS fk_publish_queue_output,
    ADD CONSTRAINT fk_publish_queue_output
        FOREIGN KEY (output_id)
        REFERENCES outputs(output_id)
        ON DELETE CASCADE;

-- publish_log → outputs
ALTER TABLE publish_log
    DROP CONSTRAINT IF EXISTS fk_publish_log_output,
    ADD CONSTRAINT fk_publish_log_output
        FOREIGN KEY (output_id)
        REFERENCES outputs(output_id)
        ON DELETE CASCADE;

-- publish_log → publish_queue
ALTER TABLE publish_log
    DROP CONSTRAINT IF EXISTS fk_publish_log_queue,
    ADD CONSTRAINT fk_publish_log_queue
        FOREIGN KEY (queue_id)
        REFERENCES publish_queue(queue_id)
        ON DELETE CASCADE;
