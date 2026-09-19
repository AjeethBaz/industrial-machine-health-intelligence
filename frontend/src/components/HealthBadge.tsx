import type { HealthStatus } from "../types/machineHealth";

interface HealthBadgeProps {
  status: HealthStatus;
}

export default function HealthBadge({ status }: HealthBadgeProps) {
  return (
    <span className={`status-badge status-${status.toLowerCase()}`}>
      <span className="status-dot" />
      {status}
    </span>
  );
}