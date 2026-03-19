/** Shared TypeScript interfaces matching the FastAPI response schemas. */

export interface RunSummary {
  pipeline_run_id: string;
  pipeline: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  total_cost_usd: number | null;
  total_duration_ms: number | null;
  concept_title: string | null;
}

export interface StepSummary {
  step: string;
  step_number: number;
  attempt: number;
  status: string;
  duration_ms: number;
}

export interface RunDetail {
  pipeline_run_id: string;
  pipeline: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  total_cost_usd: number | null;
  total_duration_ms: number | null;
  total_llm_calls: number | null;
  concept_title: string | null;
  final_video_path: string | null;
  errors: string[];
  steps: StepSummary[];
}

export interface RunListResponse {
  runs: RunSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface TraceEvent {
  event_type: string;
  event_id: string;
  run_id: string;
  timestamp: string;
  [key: string]: unknown;
}

export interface EventListResponse {
  run_id: string;
  events: TraceEvent[];
  count: number;
}

export interface PipelineStartResponse {
  pipeline_run_id: string;
  pipeline: string;
  status: string;
  message: string;
}

export interface WsMessage {
  type: 'event' | 'complete' | 'pong' | 'heartbeat';
  data?: TraceEvent;
  reason?: string;
}
