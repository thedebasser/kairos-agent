import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-pipeline-control',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './pipeline-control.component.html',
  styleUrls: ['./pipeline-control.component.scss'],
})
export class PipelineControlComponent implements OnInit {
  availablePipelines: string[] = [];
  selectedPipeline = 'physics';
  starting = false;
  error: string | null = null;
  success: string | null = null;

  constructor(private api: ApiService, private router: Router) {}

  ngOnInit(): void {
    this.api.getPipelineStatus().subscribe({
      next: (res) => (this.availablePipelines = res.available_pipelines),
      error: () => (this.availablePipelines = ['physics', 'domino']),
    });
  }

  startPipeline(): void {
    this.starting = true;
    this.error = null;
    this.success = null;

    this.api.startPipeline(this.selectedPipeline).subscribe({
      next: (res) => {
        this.success = `Started ${res.pipeline} run: ${res.pipeline_run_id.slice(0, 8)}…`;
        this.starting = false;
        // Navigate to the new run after a short delay
        setTimeout(() => {
          this.router.navigate(['/runs', res.pipeline_run_id]);
        }, 1500);
      },
      error: (err) => {
        this.error = err.error?.detail ?? 'Failed to start pipeline';
        this.starting = false;
      },
    });
  }
}
