import type { HealthStatus } from "./machineHealth";

export interface SPCPoint {
  udi: number;
  t2: number;
  alert: boolean;
  health_status: HealthStatus;
}

export interface SPCResponse {
  ucl: number;
  watch_threshold: number;
  alert_threshold: number;
  points: SPCPoint[];
}