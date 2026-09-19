import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchOverview } from "../api/client";
import type { OverviewResponse } from "../types/overview";
import KPICard from "../components/KPICard";

const STATUS_COLORS: Record<string, string> = {
  NORMAL: "#2ecc71",
  WATCH: "#f1c40f",
  ALERT: "#e67e22",
  CRITICAL: "#e74c3c",
};

function formatNumber(value: number): string {
  return value.toLocaleString();
}

function formatPercent(value: number): string {
  return `${value.toFixed(2)}%`;
}

function formatDecimal(value: number, digits = 4): string {
  return value.toFixed(digits);
}

export default function Dashboard() {
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    fetchOverview()
      .then((data) => {
        if (isMounted) {
          setOverview(data);
          setError(null);
        }
      })
      .catch((err) => {
        if (isMounted) {
          const message =
            err?.response?.data?.detail ||
            err?.message ||
            "Failed to reach the backend API.";
          setError(message);
        }
      })
      .finally(() => {
        if (isMounted) {
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  if (loading) {
    return <div className="state-message">Loading overview from FastAPI backend...</div>;
  }

  if (error || !overview) {
    return (
      <div className="state-message error">
        Could not load dashboard data: {error ?? "Unknown error"}. Confirm the
        FastAPI backend is running at http://127.0.0.1:8000.
      </div>
    );
  }

  const chartData = overview.health_status_distribution.map((item) => ({
    status: item.status,
    count: item.count,
    percentage: item.percentage,
  }));

  return (
    <div>
      <div className="kpi-grid">
        <KPICard
          label="Total Observations"
          value={formatNumber(overview.total_observations)}
          caption="Full AI4I 2020 dataset"
        />
        <KPICard
          label="Phase-I Healthy Baseline"
          value={formatNumber(overview.phase1_healthy_count)}
          caption="Machine failure == 0, first 70% by UDI"
        />
        <KPICard
          label="Phase-II Observations"
          value={formatNumber(overview.phase2_count)}
          caption="Remaining 30% by UDI"
        />
        <KPICard
          label="Hotelling's T² UCL"
          value={formatDecimal(overview.ucl)}
          caption={`p=5, n=${formatNumber(overview.phase1_healthy_count)}, alpha=0.05`}
        />
        <KPICard
          label="T² Alert Rate (Phase-II)"
          value={formatPercent(overview.alert_rate_percent)}
          caption="Multivariate abnormality signal, not a failure rate"
        />
        <KPICard
          label="Observed Failure Rate"
          value={formatPercent(overview.observed_failure_rate_percent)}
          caption="Machine failure label, full dataset"
        />
      </div>

      <div className="section">
        <div className="section-title">Health Status Distribution (Phase-II)</div>
        <div className="section-caption">
          Classification derived from Hotelling's T² relative to the established UCL.
          NORMAL, WATCH, ALERT, and CRITICAL are operational severity bands, not
          calibrated failure probabilities.
        </div>
        <div className="card">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#232b36" vertical={false} />
              <XAxis dataKey="status" stroke="#94a3b3" fontSize={12} />
              <YAxis stroke="#94a3b3" fontSize={12} allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#161d26",
                  border: "1px solid #232b36",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(value: number, _name, props) => [
                  `${value} (${props.payload.percentage.toFixed(2)}%)`,
                  "Observations",
                ]}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {chartData.map((entry) => (
                  <Cell key={entry.status} fill={STATUS_COLORS[entry.status] ?? "#7f8c8d"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="section">
        <div className="section-title">Dataset Validation</div>
        <div className="section-caption">
          Reported by src/data_loader.py via GET /api/overview.
        </div>
        <div className="card">
          <table>
            <tbody>
              <tr>
                <td style={{ color: "var(--color-text-secondary)", padding: "6px 0" }}>Row count</td>
                <td style={{ textAlign: "right", fontFamily: "var(--font-mono)" }}>
                  {formatNumber(overview.validation.row_count)}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--color-text-secondary)", padding: "6px 0" }}>Column count</td>
                <td style={{ textAlign: "right", fontFamily: "var(--font-mono)" }}>
                  {overview.validation.column_count}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--color-text-secondary)", padding: "6px 0" }}>Duplicate rows</td>
                <td style={{ textAlign: "right", fontFamily: "var(--font-mono)" }}>
                  {overview.validation.duplicate_row_count}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--color-text-secondary)", padding: "6px 0" }}>Total missing values</td>
                <td style={{ textAlign: "right", fontFamily: "var(--font-mono)" }}>
                  {overview.validation.total_missing}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}