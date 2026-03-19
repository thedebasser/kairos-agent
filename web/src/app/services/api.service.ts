import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  EventListResponse,
  PipelineStartResponse,
  RunDetail,
  RunListResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  /** List runs with optional filters. */
  listRuns(params?: {
    limit?: number;
    offset?: number;
    pipeline?: string;
    status?: string;
  }): Observable<RunListResponse> {
    let httpParams = new HttpParams();
    if (params?.limit) httpParams = httpParams.set('limit', params.limit);
    if (params?.offset) httpParams = httpParams.set('offset', params.offset);
    if (params?.pipeline) httpParams = httpParams.set('pipeline', params.pipeline);
    if (params?.status) httpParams = httpParams.set('status', params.status);

    return this.http.get<RunListResponse>(`${this.baseUrl}/runs`, {
      params: httpParams,
    });
  }

  /** Get full detail for a single run. */
  getRun(runId: string): Observable<RunDetail> {
    return this.http.get<RunDetail>(`${this.baseUrl}/runs/${runId}`);
  }

  /** Get all events for a run. */
  getRunEvents(runId: string): Observable<EventListResponse> {
    return this.http.get<EventListResponse>(
      `${this.baseUrl}/runs/${runId}/events`
    );
  }

  /** Start a new pipeline run. */
  startPipeline(pipeline: string = 'physics'): Observable<PipelineStartResponse> {
    return this.http.post<PipelineStartResponse>(
      `${this.baseUrl}/pipeline/start`,
      { pipeline }
    );
  }

  /** Get pipeline availability status. */
  getPipelineStatus(): Observable<{ available_pipelines: string[]; status: string }> {
    return this.http.get<{ available_pipelines: string[]; status: string }>(
      `${this.baseUrl}/pipeline/status`
    );
  }
}
