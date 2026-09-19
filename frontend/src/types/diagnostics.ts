export interface CovarianceDiagnostics {
  condition_number: number;
  min_eigenvalue: number;
  max_eigenvalue: number;
  determinant: number;
  is_positive_definite: boolean;
  is_numerically_suitable: boolean;
}

export interface Phase1T2Distribution {
  count: number;
  mean: number;
  median: number;
  std: number;
  minimum: number;
  maximum: number;
  p90: number;
  p95: number;
  p99: number;
  n_above_ucl: number;
  pct_above_ucl: number;
}

export interface NormalityDiagnostic {
  qq_correlation: number;
  test_name: string;
  limitations: string;
}

export interface TopT2Observation {
  UDI: number;
  Hotelling_T2: number;
}

export interface DiagnosticsResponse {
  phase1_covariance: CovarianceDiagnostics;
  phase1_t2_distribution: Phase1T2Distribution;
  normality_diagnostic: NormalityDiagnostic;
  top10_phase1_t2: TopT2Observation[];
}