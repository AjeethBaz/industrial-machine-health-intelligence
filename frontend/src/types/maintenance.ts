import type {
  ContributingSensor,
  HealthStatus,
} from "./machineHealth";

export interface MaintenanceResponse {
  udi: number;
  health_status: HealthStatus;
  priority: string;
  contributing_sensors: ContributingSensor[];
  primary_recommendation: string;
  additional_recommendations: string[];
  evidence: string[];
  limitations: string[];
}