export interface T2GroupComparison {
  group: "Failure" | "Non-failure";
  count: number;
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  q25: number;
  q75: number;
  alerts: number;
  alert_percentage: number;
}

export interface T2EvaluationResponse {
  true_positives: number;
  false_positives: number;
  true_negatives: number;
  false_negatives: number;
  precision: number;
  recall: number;
  specificity: number;
  f1_score: number;
  false_positive_rate: number;
  alert_rate: number;
  actual_failure_rate: number;
  failure_capture_rate: number;
  group_comparison: T2GroupComparison[];
}