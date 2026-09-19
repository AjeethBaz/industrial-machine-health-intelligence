import type { HealthStatus } from "../types/machineHealth";

interface StatusBadgeProps {
  status: HealthStatus;
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  return (
    <span className={`status-badge status-${status.toLowerCase()}`}>
      <span className="status-dot" />
      {status}
    </span>
  );
}