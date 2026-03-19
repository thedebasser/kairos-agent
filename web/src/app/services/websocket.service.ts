import { Injectable, OnDestroy } from '@angular/core';
import { Observable, Subject } from 'rxjs';
import { environment } from '../../environments/environment';
import { TraceEvent, WsMessage } from '../models/api.models';

/**
 * Manages WebSocket connections for live event streaming.
 *
 * Usage:
 *   const events$ = wsService.connectToRun(runId);
 *   events$.subscribe(event => { ... });
 *   wsService.disconnect();
 */
@Injectable({ providedIn: 'root' })
export class WebSocketService implements OnDestroy {
  private ws: WebSocket | null = null;
  private events$ = new Subject<TraceEvent>();
  private complete$ = new Subject<void>();
  private pingInterval: ReturnType<typeof setInterval> | null = null;

  /** Observable of live trace events. */
  get events(): Observable<TraceEvent> {
    return this.events$.asObservable();
  }

  /** Emits when the run completes. */
  get runComplete(): Observable<void> {
    return this.complete$.asObservable();
  }

  /** Connect to a specific run's event stream. */
  connectToRun(runId: string): Observable<TraceEvent> {
    this.disconnect();
    const url = `${environment.wsUrl}/ws/runs/${runId}`;
    this.initSocket(url);
    return this.events$;
  }

  /** Connect to the global event broadcast. */
  connectGlobal(): Observable<TraceEvent> {
    this.disconnect();
    const url = `${environment.wsUrl}/ws/events`;
    this.initSocket(url);
    return this.events$;
  }

  /** Disconnect and clean up. */
  disconnect(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  ngOnDestroy(): void {
    this.disconnect();
    this.events$.complete();
    this.complete$.complete();
  }

  private initSocket(url: string): void {
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      // Start keepalive pings every 25s
      this.pingInterval = setInterval(() => {
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 25_000);
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        if (msg.type === 'event' && msg.data) {
          this.events$.next(msg.data);
        } else if (msg.type === 'complete') {
          this.complete$.next();
        }
        // pong and heartbeat are silently consumed
      } catch {
        // Ignore malformed messages
      }
    };

    this.ws.onerror = () => {
      // Error handling — could add retry logic here
    };

    this.ws.onclose = () => {
      if (this.pingInterval) {
        clearInterval(this.pingInterval);
        this.pingInterval = null;
      }
    };
  }
}
