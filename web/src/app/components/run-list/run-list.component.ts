import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { RunSummary } from '../../models/api.models';

@Component({
  selector: 'app-run-list',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './run-list.component.html',
  styleUrl: './run-list.component.scss',
})
export class RunListComponent implements OnInit {
  runs: RunSummary[] = [];
  total = 0;
  page = 0;
  pageSize = 20;
  loading = true;
  error = '';

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadRuns();
  }

  loadRuns(): void {
    this.loading = true;
    this.error = '';
    this.api
      .listRuns({ limit: this.pageSize, offset: this.page * this.pageSize })
      .subscribe({
        next: (res) => {
          this.runs = res.runs;
          this.total = res.total;
          this.loading = false;
        },
        error: (err) => {
          this.error = 'Failed to load runs. Is the API running?';
          this.loading = false;
          console.error(err);
        },
      });
  }

  get totalPages(): number {
    return Math.ceil(this.total / this.pageSize);
  }

  nextPage(): void {
    if (this.page < this.totalPages - 1) {
      this.page++;
      this.loadRuns();
    }
  }

  prevPage(): void {
    if (this.page > 0) {
      this.page--;
      this.loadRuns();
    }
  }

  formatDate(iso: string | null): string {
    if (!iso) return '--';
    return new Date(iso).toLocaleString();
  }

  formatDuration(ms: number | null): string {
    if (!ms) return '--';
    if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
    return `${ms}ms`;
  }

  formatCost(usd: number | null): string {
    if (usd === null || usd === undefined) return '--';
    return `$${usd.toFixed(4)}`;
  }

  statusClass(status: string): string {
    switch (status) {
      case 'running':
        return 'badge-running';
      case 'success':
      case 'approved':
      case 'published':
        return 'badge-success';
      case 'failed':
        return 'badge-error';
      default:
        return 'badge-default';
    }
  }
}
