export type HealthStatus = "NORMAL" | "WATCH" | "ALERT" | "CRITICAL";

export interface ContributingSensor {
  sensor: string;
  z_score: number;
  direction: "HIGH" | "LOW";
}

export interface MachineHealthResponse {
  udi: number;
  t2_value: number;
  ucl: number;
  t2_ratio: number;
  health_status: HealthStatus;
  priority: string;
  sensor_values: Record<string, number>;
  standardized_sensors: Record<string, number>;
  contributing_sensors: ContributingSensor[];
  primary_recommendation: string;
  additional_recommendations: string[];
  evidence: string[];
  limitations: string[];
  failure_probability: number | null;
  failure_probability_note: string;
}