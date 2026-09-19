export interface ValidationSummary {
  row_count: number;
  column_count: number;
  duplicate_row_count: number;
  total_missing: number;
}

export interface HealthStatusCount {
  status: "NORMAL" | "WATCH" | "ALERT" | "CRITICAL";
  count: number;
  percentage: number;
}

export interface OverviewResponse {
  total_observations: number;
  phase1_healthy_count: number;
  phase2_count: number;
  ucl: number;
  alert_rate_percent: number;
  observed_failure_rate_percent: number;
  health_status_distribution: HealthStatusCount[];
  validation: ValidationSummary;
}