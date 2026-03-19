import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';
import { ApiService } from '../../services/api.service';
import { WebSocketService } from '../../services/websocket.service';
import { RunDetail, StepSummary, TraceEvent } from '../../models/api.models';

@Component({
  selector: 'app-run-detail',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './run-detail.component.html',
  styleUrls: ['./run-detail.component.scss'],
})
export class RunDetailComponent implements OnInit, OnDestroy {
  run: RunDetail | null = null;
  events: TraceEvent[] = [];
  loading = true;
  error: string | null = null;
  liveConnected = false;

  private destroy$ = new Subject<void>();

  constructor(
    private route: ActivatedRoute,
    private api: ApiService,
    private ws: WebSocketService
  ) {}

  ngOnInit(): void {
    const runId = this.route.snapshot.paramMap.get('id')!;
    this.loadRun(runId);
    this.loadEvents(runId);

    // Connect live WebSocket for running pipelines
    this.ws
      .connectToRun(runId)
      .pipe(takeUntil(this.destroy$))
      .subscribe((event) => {
        this.events.push(event);
        this.liveConnected = true;
      });

    this.ws.runComplete.pipe(takeUntil(this.destroy$)).subscribe(() => {
      // Reload the run to get final status
      this.loadRun(runId);
      this.liveConnected = false;
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.ws.disconnect();
  }

  private loadRun(runId: string): void {
    this.api.getRun(runId).subscribe({
      next: (run) => {
        this.run = run;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.detail ?? 'Failed to load run';
        this.loading = false;
      },
    });
  }

  private loadEvents(runId: string): void {
    this.api.getRunEvents(runId).subscribe({
      next: (res) => (this.events = res.events),
      error: () => {}, // Events are optional
    });
  }

  statusClass(status: string): string {
    switch (status?.toLowerCase()) {
      case 'running':
        return 'badge-running';
      case 'success':
      case 'completed':
        return 'badge-success';
      case 'error':
      case 'failed':
        return 'badge-error';
      default:
        return 'badge-default';
    }
  }

  stepStatusIcon(status: string): string {
    switch (status?.toLowerCase()) {
      case 'success':
      case 'completed':
        return '✓';
      case 'error':
      case 'failed':
        return '✗';
      case 'running':
        return '⟳';
      default:
        return '·';
    }
  }

  formatDuration(ms: number | null): string {
    if (ms == null) return '—';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  formatCost(cost: number | null): string {
    if (cost == null) return '—';
    return `$${cost.toFixed(4)}`;
  }

  formatTimestamp(ts: string): string {
    return new Date(ts).toLocaleTimeString();
  }

  shortId(id: string): string {
    return id?.slice(0, 8) ?? '';
  }

  trackEvent(_index: number, ev: TraceEvent): string {
    return ev.event_id;
  }
}
