export interface ManovaDescriptiveStat {
  "Machine failure": string;
  Sensor: string;
  count: number;
  mean: number;
  std: number;
  median: number;
}

export interface ManovaTest {
  test_name: string;
  value: number;
  f_value: number;
  num_df: number;
  den_df: number;
  p_value: number;
}

export interface ManovaResponse {
  group_sizes: Record<string, number>;
  descriptive_stats: ManovaDescriptiveStat[];
  mean_differences: Record<string, number>;
  multivariate_tests: ManovaTest[];
  limitations: string[];
}