import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchSPC } from "../api/client";
import type { HealthStatus } from "../types/machineHealth";
import type { SPCResponse } from "../types/spc";
import MetricCard from "../components/MetricCard";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

type FilterType = "ALL" | "ALERTS" | HealthStatus;

function formatValue(value: number): string {
  return value.toFixed(4);
}

export default function SPCMonitoring() {
  const navigate = useNavigate();
  const [spc, setSpc] = useState<SPCResponse | null>(null);
  const [filter, setFilter] = useState<FilterType>("ALL");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadSPC() {
      try {
        const data = await fetchSPC();
        if (active) {
          setSpc(data);
        }
      } catch (requestError: unknown) {
        if (active) {
          const axiosError = requestError as {
            response?: { data?: { detail?: string } };
            message?: string;
          };

          setError(
            axiosError.response?.data?.detail ??
              axiosError.message ??
              "Unable to load SPC monitoring data."
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadSPC();

    return () => {
      active = false;
    };
  }, []);

  const displayedPoints = useMemo(() => {
    if (!spc) {
      return [];
    }

    if (filter === "ALL") {
      return spc.points;
    }

    if (filter === "ALERTS") {
      return spc.points.filter((point) => point.alert);
    }

    return spc.points.filter((point) => point.health_status === filter);
  }, [spc, filter]);

  if (loading) {
    return <LoadingState message="Loading Hotelling's T² control chart..." />;
  }

  if (error || !spc) {
    return <ErrorState message={error ?? "SPC data is unavailable."} />;
  }

  const totalAlerts = spc.points.filter((point) => point.alert).length;

  return (
    <div>
      <div className="content-page-title">SPC Monitoring</div>
      <div className="content-page-subtitle">
        Phase-II Hotelling&apos;s T² monitoring relative to the healthy
        Phase-I baseline.
      </div>

      <div className="metric-grid machine-kpi-grid">
        <MetricCard
          label="Phase-II Observations"
          value={spc.points.length.toLocaleString()}
          caption="Monitoring sequence"
        />
        <MetricCard
          label="T² Alerts"
          value={totalAlerts.toLocaleString()}
          caption="Observations above established UCL"
          tone="alert"
        />
        <MetricCard
          label="T² UCL"
          value={formatValue(spc.ucl)}
          caption="Theoretical Phase-II control limit"
        />
        <MetricCard
          label="Critical Threshold"
          value={formatValue(spc.alert_threshold)}
          caption="Operational critical severity boundary"
          tone="critical"
        />
      </div>

      <SectionCard
        title="Hotelling's T² Control Chart"
        caption="Click a red alert point to open its Machine Health page. Thresholds and T² values are supplied by FastAPI."
      >
        <div className="filter-row">
          {(["ALL", "ALERTS", "NORMAL", "WATCH", "ALERT", "CRITICAL"] as FilterType[]).map(
            (option) => (
              <button
                key={option}
                type="button"
                className={`filter-button${filter === option ? " active" : ""}`}
                onClick={() => setFilter(option)}
              >
                {option === "ALL" ? "All" : option}
              </button>
            )
          )}
        </div>

        <div className="chart-box">
          <ResponsiveContainer width="100%" height={480}>
            <LineChart
              data={displayedPoints}
              margin={{ top: 18, right: 22, left: 4, bottom: 18 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(148,163,184,0.18)"
              />
              <XAxis
                dataKey="udi"
                type="number"
                domain={["dataMin", "dataMax"]}
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                label={{
                  value: "UDI / Observation Sequence",
                  position: "insideBottom",
                  offset: -6,
                  fill: "#94a3b8",
                }}
              />
              <YAxis
                dataKey="t2"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                label={{
                  value: "Hotelling's T²",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#94a3b8",
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#11161d",
                  border: "1px solid #334155",
                  borderRadius: 8,
                  color: "#e6ebf0",
                }}
                labelFormatter={(label) => `UDI: ${label}`}
                formatter={(value: number, _name, item) => {
                  const point = item.payload as {
                    health_status: string;
                    alert: boolean;
                  };

                  return [
                    `${formatValue(value)} | ${point.health_status} | ${
                      point.alert ? "T² Alert" : "No Alert"
                    }`,
                    "T²",
                  ];
                }}
              />
              <ReferenceLine
                y={spc.watch_threshold}
                stroke="#f1c40f"
                strokeDasharray="6 6"
                label={{
                  value: "Watch",
                  fill: "#f1c40f",
                  fontSize: 11,
                }}
              />
              <ReferenceLine
                y={spc.ucl}
                stroke="#e6ebf0"
                strokeDasharray="8 5"
                label={{
                  value: "UCL",
                  fill: "#e6ebf0",
                  fontSize: 11,
                }}
              />
              <ReferenceLine
                y={spc.alert_threshold}
                stroke="#e74c3c"
                strokeDasharray="6 6"
                label={{
                  value: "Critical",
                  fill: "#e74c3c",
                  fontSize: 11,
                }}
              />
              <Line
                type="monotone"
                dataKey="t2"
                stroke="#60a5fa"
                strokeWidth={1.4}
                dot={false}
                activeDot={{ r: 5 }}
              />
              <Scatter
                data={displayedPoints.filter((point) => point.alert)}
                dataKey="t2"
                fill="#e74c3c"
                onClick={(point) => {
                  const clickedPoint = point as unknown as { udi?: number };

                  if (clickedPoint.udi) {
                    navigate(`/machine/${clickedPoint.udi}`);
                  }
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </SectionCard>

      <SectionCard title="Statistical Interpretation">
        <div className="info-panel">
          <strong>Hotelling&apos;s T² is a multivariate abnormality measure.</strong>
          <p>
            A T² alert identifies an observation whose combined sensor profile
            is unusual relative to the Phase-I healthy baseline. It does not
            represent a calibrated failure probability and does not confirm a
            physical machine fault.
          </p>
        </div>
      </SectionCard>
    </div>
  );
}