import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { fetchMachineHealth } from "../api/client";
import type {
  ContributingSensor,
  MachineHealthResponse,
} from "../types/machineHealth";
import MetricCard from "../components/MetricCard";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

function formatNumber(value: number, digits = 4): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function sensorLabel(sensor: string): string {
  return sensor
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function severityClass(status: string): string {
  return `status-${status.toLowerCase()}`;
}

function statusMessage(status: string): string {
  switch (status) {
    case "CRITICAL":
      return "The current sensor pattern is far outside the healthy baseline. Prioritize a maintenance investigation.";
    case "ALERT":
      return "The current sensor pattern is meaningfully different from the healthy baseline. Plan a maintenance review.";
    case "WATCH":
      return "The machine shows an early deviation from the healthy baseline. Increase monitoring and inspect during the next check.";
    default:
      return "The machine sensor pattern is within the established healthy monitoring limits.";
  }
}

function sensorTone(sensor: ContributingSensor): string {
  const magnitude = Math.abs(sensor.z_score);

  if (magnitude >= 3) {
    return "contributor-high";
  }

  return "contributor-low";
}

export default function MachineHealth() {
  const { udi: udiParam } = useParams<{ udi: string }>();
  const udi = Number(udiParam);

  const [machine, setMachine] = useState<MachineHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadMachineHealth() {
      if (!Number.isInteger(udi) || udi <= 0) {
        if (active) {
          setError("The machine UDI in the URL is invalid.");
          setLoading(false);
        }
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const response = await fetchMachineHealth(udi);

        if (active) {
          setMachine(response);
        }
      } catch (requestError: unknown) {
        if (active) {
          const axiosError = requestError as {
            response?: {
              status?: number;
              data?: {
                detail?: string;
              };
            };
            message?: string;
          };

          if (axiosError.response?.status === 404) {
            setError(
              `Machine UDI ${udi} was not found in the Phase-II monitoring dataset.`
            );
          } else {
            setError(
              axiosError.response?.data?.detail ??
                axiosError.message ??
                "Unable to load machine health information."
            );
          }
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadMachineHealth();

    return () => {
      active = false;
    };
  }, [udi]);

  const sortedSensors = useMemo(() => {
    if (!machine?.contributing_sensors) {
      return [];
    }

    return [...machine.contributing_sensors].sort(
      (first, second) =>
        Math.abs(second.z_score) - Math.abs(first.z_score)
    );
  }, [machine]);

  if (loading) {
    return <LoadingState message={`Loading machine UDI ${udi} health data...`} />;
  }

  if (error || !machine) {
    return (
      <ErrorState
        message={error ?? "Machine health information is unavailable."}
      />
    );
  }

  const statusClass = severityClass(machine.health_status);
  const ratio = Number(machine.t2_ratio);
  const ratioLabel = Number.isFinite(ratio) ? `${ratio.toFixed(2)}×` : "—";

  return (
    <div className="machine-health-page">
      <div className="content-page-title">
        Machine Health — UDI {machine.udi}
      </div>

      <div className="content-page-subtitle">
        Phase-II multivariate monitoring result compared against the healthy
        Phase-I baseline.
      </div>

      <section className={`machine-health-hero ${statusClass}`}>
        <div>
          <div className="machine-health-hero-label">
            Current health classification
          </div>

          <div className="machine-health-status-row">
            <span className={`status-badge ${statusClass}`}>
              <span className="status-dot" />
              {machine.health_status}
            </span>

            <span className="machine-health-priority">
              Maintenance priority: {machine.priority}
            </span>
          </div>

          <p>{statusMessage(machine.health_status)}</p>
        </div>

        <div className="machine-health-action">
          <span>Recommended action</span>
          <strong>{machine.primary_recommendation}</strong>
        </div>
      </section>

      <div className="metric-grid machine-kpi-grid">
        <MetricCard
          label="Hotelling’s T²"
          value={formatNumber(machine.t2_value)}
          caption="Current multivariate statistic"
          tone={
            machine.health_status === "CRITICAL"
              ? "critical"
              : machine.health_status === "ALERT"
                ? "alert"
                : machine.health_status === "WATCH"
                  ? "watch"
                  : "normal"
          }
        />

        <MetricCard
          label="Established UCL"
          value={formatNumber(machine.ucl)}
          caption="Upper Control Limit from Phase-I"
          tone="blue"
        />

        <MetricCard
          label="T² / UCL Ratio"
          value={ratioLabel}
          caption={
            ratio > 1
              ? "Above the healthy monitoring boundary"
              : "Within the healthy monitoring boundary"
          }
          tone={
            ratio > 3
              ? "critical"
              : ratio > 1.5
                ? "alert"
                : ratio > 1
                  ? "watch"
                  : "normal"
          }
        />
      </div>

      <div className="analytics-grid">
        <SectionCard
          title="What the monitor detected"
          caption="These are statistical monitoring findings, not confirmed mechanical root causes."
        >
          {machine.evidence?.length ? (
            <ul className="bullet-list">
              {machine.evidence.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <div className="empty-inline">
              No additional evidence description was returned for this machine.
            </div>
          )}
        </SectionCard>

        <SectionCard
          title="Recommended maintenance steps"
          caption="Use these steps as an investigation checklist and apply site safety procedures."
        >
          {machine.additional_recommendations?.length ? (
            <ol className="maintenance-action-list">
              {machine.additional_recommendations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ol>
          ) : (
            <div className="empty-inline">
              {machine.primary_recommendation}
            </div>
          )}
        </SectionCard>
      </div>

      <SectionCard
        title="Contributing sensor deviations"
        caption="Sensor values are standardized relative to the healthy Phase-I baseline. Larger absolute z-scores indicate stronger deviations."
      >
        {sortedSensors.length ? (
          <div className="contributor-grid">
            {sortedSensors.map((sensor) => (
              <article
                className={`contributor-card ${sensorTone(sensor)}`}
                key={`${sensor.sensor}-${sensor.direction}`}
              >
                <div className="contributor-label">
                  {sensorLabel(sensor.sensor)}
                </div>

                <div className="contributor-value">
                  {sensor.direction === "HIGH" ? "+" : ""}
                  {formatNumber(sensor.z_score, 3)}
                </div>

                <div className="contributor-caption">
                  Unusually {sensor.direction.toLowerCase()} relative to the
                  healthy baseline
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-inline">
            No individual sensor deviations were returned by the monitoring
            service.
          </div>
        )}
      </SectionCard>

      <SectionCard
        title="Statistical limitations"
        caption="Important context for safe interpretation."
      >
        <div className="warning-panel">
          <ul className="bullet-list muted-list">
            {(machine.limitations?.length
              ? machine.limitations
              : [
                  "Hotelling’s T² measures multivariate abnormality relative to the healthy baseline.",
                  "An abnormal score does not independently confirm a physical fault or establish a cause.",
                  "Physical inspection and domain-specific measurements are required to confirm a failure mechanism.",
                ]
            ).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>

          <p>
            {machine.failure_probability_note ||
              "Failure probability is not available because T² is an abnormality statistic, not a calibrated probability of failure."}
          </p>
        </div>
      </SectionCard>
    </div>
  );
}