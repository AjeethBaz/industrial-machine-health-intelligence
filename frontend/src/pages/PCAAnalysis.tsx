import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchPCA } from "../api/client";
import type { PCAResponse } from "../types/pca";
import MetricCard from "../components/MetricCard";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

const COLORS = ["#3b82f6", "#22d3ee", "#2ecc71", "#f1c40f", "#e67e22"];

function percent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function loadingColor(value: number): string {
  const intensity = Math.min(Math.abs(value), 1);

  if (value >= 0) {
    return `rgba(59, 130, 246, ${0.15 + intensity * 0.72})`;
  }

  return `rgba(231, 76, 60, ${0.15 + intensity * 0.72})`;
}

function formatNumber(value: unknown, digits = 4): string {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : "—";
}

export default function PCAAnalysis() {
  const [pca, setPca] = useState<PCAResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadPCA() {
      try {
        const data = await fetchPCA();

        if (active) {
          setPca(data);
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
              "Unable to load PCA analysis."
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadPCA();

    return () => {
      active = false;
    };
  }, []);

  const explainedVariance = useMemo(() => {
    if (!pca) {
      return [];
    }

    const raw = (pca as PCAResponse & {
      explained_variance?: unknown;
      explained_variance_ratio?: unknown;
    }).explained_variance ?? (
      pca as PCAResponse & { explained_variance_ratio?: unknown }
    ).explained_variance_ratio;

    if (!Array.isArray(raw)) {
      return [];
    }

    return raw
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value));
  }, [pca]);

  const components = useMemo(() => {
    if (!pca) {
      return [];
    }

    const apiComponents = (pca as PCAResponse & {
      components?: unknown;
      component_names?: unknown;
    }).components;

    if (Array.isArray(apiComponents) && apiComponents.length > 0) {
      return apiComponents.map((component, index) =>
        typeof component === "string" ? component : `PC${index + 1}`
      );
    }

    return explainedVariance.map((_, index) => `PC${index + 1}`);
  }, [pca, explainedVariance]);

  const cumulative = useMemo(() => {
    if (!pca) {
      return [];
    }

    const apiCumulative = (pca as PCAResponse & { cumulative?: unknown })
      .cumulative;

    if (Array.isArray(apiCumulative) && apiCumulative.length > 0) {
      return apiCumulative
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value));
    }

    let total = 0;

    return explainedVariance.map((value) => {
      total += value;
      return total;
    });
  }, [pca, explainedVariance]);

  const componentsFor90 = useMemo(() => {
    if (!pca) {
      return 0;
    }

    const apiValue = Number(
      (pca as PCAResponse & {
        components_required_for_90_percent?: unknown;
      }).components_required_for_90_percent
    );

    if (Number.isInteger(apiValue) && apiValue > 0) {
      return apiValue;
    }

    const calculatedIndex = cumulative.findIndex((value) => value >= 0.9);

    return calculatedIndex === -1 ? cumulative.length : calculatedIndex + 1;
  }, [pca, cumulative]);

  const chartData = useMemo(() => {
    return explainedVariance.map((explained, index) => ({
      component: components[index] ?? `PC${index + 1}`,
      explained: explained * 100,
      cumulative: (cumulative[index] ?? 0) * 100,
    }));
  }, [components, cumulative, explainedVariance]);

  if (loading) {
    return <LoadingState message="Loading Phase-I PCA results..." />;
  }

  if (error || !pca) {
    return <ErrorState message={error ?? "PCA results are unavailable."} />;
  }

  if (explainedVariance.length === 0) {
    return (
      <ErrorState message="PCA results were returned, but the explained-variance array is missing or invalid." />
    );
  }

  const pcaLoadings = (
    pca as PCAResponse & {
      loadings?: Record<string, Record<string, number> | number[]>;
    }
  ).loadings ?? {};

  const cumulativeAt90 =
    cumulative[Math.max(0, componentsFor90 - 1)] ?? cumulative.at(-1) ?? 0;

  return (
    <div>
      <div className="content-page-title">PCA Analysis</div>

      <div className="content-page-subtitle">
        Principal Component Analysis fitted only on standardized healthy
        Phase-I sensor observations.
      </div>

      <div className="metric-grid machine-kpi-grid">
        <MetricCard
          label="Principal Components"
          value={String(components.length)}
          caption="Sensor-space variance structure"
        />

        <MetricCard
          label="PC1 Explained Variance"
          value={percent(explainedVariance[0] ?? 0)}
          caption="Largest variance component"
          tone="blue"
        />

        <MetricCard
          label="Components for ≥90%"
          value={`${componentsFor90} PCs`}
          caption={`Cumulative: ${percent(cumulativeAt90)}`}
          tone="normal"
        />
      </div>

      <div className="analytics-grid">
        <SectionCard
          title="Explained Variance"
          caption="The amount of healthy Phase-I variation represented by each principal component."
        >
          <div className="chart-box">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(148,163,184,0.18)"
                  vertical={false}
                />

                <XAxis
                  dataKey="component"
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                />

                <YAxis
                  tickFormatter={(value: number) => `${value}%`}
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                />

                <Tooltip
                  contentStyle={{
                    backgroundColor: "#11161d",
                    border: "1px solid #334155",
                    borderRadius: 8,
                  }}
                  formatter={(value: number) => [
                    `${Number(value).toFixed(2)}%`,
                    "Explained Variance",
                  ]}
                />

                <Bar dataKey="explained" radius={[5, 5, 0, 0]}>
                  {chartData.map((item, index) => (
                    <Cell
                      key={item.component}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>

        <SectionCard
          title="Cumulative Explained Variance"
          caption="How much total healthy operating variation is retained as components are added."
        >
          <div className="chart-box">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(148,163,184,0.18)"
                />

                <XAxis
                  dataKey="component"
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                />

                <YAxis
                  domain={[0, 100]}
                  tickFormatter={(value: number) => `${value}%`}
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                />

                <Tooltip
                  contentStyle={{
                    backgroundColor: "#11161d",
                    border: "1px solid #334155",
                    borderRadius: 8,
                  }}
                  formatter={(value: number) => [
                    `${Number(value).toFixed(2)}%`,
                    "Cumulative Variance",
                  ]}
                />

                <ReferenceLine
                  y={90}
                  stroke="#f1c40f"
                  strokeDasharray="6 6"
                  label={{
                    value: "90%",
                    fill: "#f1c40f",
                    fontSize: 11,
                  }}
                />

                <Line
                  type="monotone"
                  dataKey="cumulative"
                  stroke="#22d3ee"
                  strokeWidth={3}
                  dot={{ r: 4, fill: "#22d3ee" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="PCA Loading Matrix"
        caption="Positive and negative loadings show how strongly each sensor aligns with each component. PCA describes data patterns; it does not prove a cause of failure."
      >
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Sensor</th>
                {components.map((component) => (
                  <th key={component}>{component}</th>
                ))}
              </tr>
            </thead>

            <tbody>
              {Object.entries(pcaLoadings).map(([sensor, sensorLoadings]) => (
                <tr key={sensor}>
                  <td>{sensor}</td>

                  {components.map((component, index) => {
                    const rawValue = Array.isArray(sensorLoadings)
                      ? sensorLoadings[index]
                      : sensorLoadings?.[component];

                    const value = Number(rawValue);
                    const validValue = Number.isFinite(value);

                    return (
                      <td
                        key={component}
                        className="loading-cell"
                        style={{
                          backgroundColor: validValue
                            ? loadingColor(value)
                            : "rgba(148, 163, 184, 0.06)",
                          color:
                            validValue && Math.abs(value) >= 0.6
                              ? "#ffffff"
                              : "#e6ebf0",
                        }}
                      >
                        {validValue ? formatNumber(value) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="How to read this page">
        <div className="info-panel">
          <ul className="bullet-list">
            <li>
              PC1 captures the single largest pattern of variation in the
              healthy baseline data.
            </li>
            <li>
              Fewer components are easier to visualize, but enough components
              should be retained to represent the desired amount of variation.
            </li>
            <li>
              Loading values indicate sensor association with a component, not
              direct causal responsibility for a fault.
            </li>
          </ul>
        </div>
      </SectionCard>
    </div>
  );
}