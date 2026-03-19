import { Component } from '@angular/core';
import { RouterOutlet, RouterLink } from '@angular/router';
import { PipelineControlComponent } from './components/pipeline-control/pipeline-control.component';
import { EventFeedComponent } from './components/event-feed/event-feed.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, PipelineControlComponent, EventFeedComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'Kairos Dashboard';
}
