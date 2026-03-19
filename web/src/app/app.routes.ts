import { Routes } from '@angular/router';
import { RunListComponent } from './components/run-list/run-list.component';
import { RunDetailComponent } from './components/run-detail/run-detail.component';

export const routes: Routes = [
  { path: '', component: RunListComponent },
  { path: 'runs/:id', component: RunDetailComponent },
  { path: '**', redirectTo: '' },
];
