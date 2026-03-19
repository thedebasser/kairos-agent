import {
  Component,
  OnInit,
  OnDestroy,
  ElementRef,
  ViewChild,
  AfterViewChecked,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, takeUntil } from 'rxjs';
import { WebSocketService } from '../../services/websocket.service';
import { TraceEvent } from '../../models/api.models';

@Component({
  selector: 'app-event-feed',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './event-feed.component.html',
  styleUrls: ['./event-feed.component.scss'],
})
export class EventFeedComponent implements OnInit, OnDestroy, AfterViewChecked {
  events: TraceEvent[] = [];
  connected = false;
  private destroy$ = new Subject<void>();
  private shouldScroll = true;

  @ViewChild('feedContainer') feedContainer!: ElementRef<HTMLDivElement>;

  constructor(private ws: WebSocketService) {}

  ngOnInit(): void {
    this.ws
      .connectGlobal()
      .pipe(takeUntil(this.destroy$))
      .subscribe((event) => {
        this.events.push(event);
        // Cap at 500 events to avoid memory issues
        if (this.events.length > 500) {
          this.events = this.events.slice(-250);
        }
        this.shouldScroll = true;
        this.connected = true;
      });
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll && this.feedContainer) {
      const el = this.feedContainer.nativeElement;
      el.scrollTop = el.scrollHeight;
      this.shouldScroll = false;
    }
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.ws.disconnect();
  }

  formatTime(ts: string): string {
    return new Date(ts).toLocaleTimeString();
  }

  eventColor(type: string): string {
    if (type.includes('error') || type.includes('fail')) return 'ev-error';
    if (type.includes('complete') || type.includes('success')) return 'ev-success';
    if (type.includes('start') || type.includes('begin')) return 'ev-start';
    return 'ev-default';
  }

  trackEvent(_index: number, ev: TraceEvent): string {
    return ev.event_id;
  }

  clearEvents(): void {
    this.events = [];
  }
}
