import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchDiagnostics } from "../api/client";
import type { DiagnosticsResponse } from "../types/diagnostics";
import MetricCard from "../components/MetricCard";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

function value(valueToFormat: number, digits = 4): string {
  return valueToFormat.toFixed(digits);
}

export default function Diagnostics() {
  const [diagnostics, setDiagnostics] = useState<DiagnosticsResponse | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadDiagnostics() {
      try {
        const data = await fetchDiagnostics();

        if (active) {
          setDiagnostics(data);
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
              "Unable to load statistical diagnostics."
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadDiagnostics();

    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return <LoadingState message="Loading statistical diagnostics..." />;
  }

  if (error || !diagnostics) {
    return (
      <ErrorState message={error ?? "Statistical diagnostics are unavailable."} />
    );
  }

  const covariance = diagnostics.phase1_covariance;
  const t2 = diagnostics.phase1_t2_distribution;
  const topObservations = diagnostics.top10_phase1_t2.map((row) => ({
    udi: String(row.UDI),
    t2: row.Hotelling_T2,
  }));

  return (
    <div>
      <div className="content-page-title">Statistical Diagnostics</div>
      <div className="content-page-subtitle">
        Numerical covariance diagnostics, Phase-I T² distribution summaries,
        and practical multivariate normality evidence.
      </div>

      <SectionCard
        title="Phase-I Covariance Diagnostics"
        caption="Computed on standardized healthy Phase-I observations. The covariance matrix is assessed but not automatically modified."
      >
        <div className="metric-grid">
          <MetricCard
            label="Condition Number"
            value={value(covariance.condition_number)}
            caption="Numerical stability indicator"
            tone={covariance.is_numerically_suitable ? "normal" : "alert"}
          />
          <MetricCard
            label="Minimum Eigenvalue"
            value={value(covariance.min_eigenvalue, 6)}
            caption="Smallest covariance eigenvalue"
          />
          <MetricCard
            label="Maximum Eigenvalue"
            value={value(covariance.max_eigenvalue, 6)}
            caption="Largest covariance eigenvalue"
          />
          <MetricCard
            label="Determinant"
            value={value(covariance.determinant, 6)}
            caption="Covariance matrix determinant"
          />
          <MetricCard
            label="Positive Definite"
            value={covariance.is_positive_definite ? "YES" : "NO"}
            caption="Eigenvalue diagnostic"
            tone={covariance.is_positive_definite ? "normal" : "critical"}
          />
          <MetricCard
            label="Numerically Suitable"
            value={covariance.is_numerically_suitable ? "YES" : "NO"}
            caption="For existing Hotelling's T² calculation"
            tone={covariance.is_numerically_suitable ? "normal" : "critical"}
          />
        </div>
      </SectionCard>

      <SectionCard
        title="Phase-I T² Distribution"
        caption="Descriptive summary of healthy baseline observations. Percentiles do not replace the theoretical Phase-II UCL."
      >
        <div className="metric-grid">
          <MetricCard
            label="Count"
            value={t2.count.toLocaleString()}
            caption="Healthy Phase-I observations"
          />
          <MetricCard label="Mean" value={value(t2.mean)} caption="Average T²" />
          <MetricCard
            label="Median"
            value={value(t2.median)}
            caption="Central T²"
          />
          <MetricCard
            label="Standard Deviation"
            value={value(t2.std)}
            caption="T² dispersion"
          />
          <MetricCard
            label="Minimum"
            value={value(t2.minimum)}
            caption="Lowest Phase-I T²"
          />
          <MetricCard
            label="Maximum"
            value={value(t2.maximum)}
            caption="Largest Phase-I T²"
            tone="alert"
          />
          <MetricCard label="P90" value={value(t2.p90)} caption="90th percentile" />
          <MetricCard label="P95" value={value(t2.p95)} caption="95th percentile" />
          <MetricCard
            label="P99"
            value={value(t2.p99)}
            caption="99th percentile"
            tone="watch"
          />
          <MetricCard
            label="Above UCL"
            value={t2.n_above_ucl.toLocaleString()}
            caption="Existing UCL comparison"
            tone="alert"
          />
          <MetricCard
            label="Above UCL %"
            value={`${t2.pct_above_ucl.toFixed(2)}%`}
            caption="No empirical UCL replacement"
          />
        </div>
      </SectionCard>

      <div className="analytics-grid">
        <SectionCard
          title="Multivariate Normality Diagnostic"
          caption="Mahalanobis-distance versus chi-square Q-Q correspondence."
        >
          <div className="normality-score">
            <span>Q-Q Correlation</span>
            <strong>
              {value(diagnostics.normality_diagnostic.qq_correlation, 6)}
            </strong>
          </div>

          <div className="warning-panel">
            <strong>Descriptive diagnostic only</strong>
            <p>
              The Q-Q correlation is a descriptive diagnostic, not a formal
              multivariate normality test.
            </p>
          </div>

          <p className="diagnostic-paragraph">
            <strong>{diagnostics.normality_diagnostic.test_name}</strong>
            <br />
            {diagnostics.normality_diagnostic.limitations}
          </p>
        </SectionCard>

        <SectionCard
          title="Top Phase-I T² Observations"
          caption="Highest baseline T² values shown for diagnostic review only. No observations are removed."
        >
          <div className="chart-box">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={topObservations}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(148,163,184,0.18)"
                  vertical={false}
                />
                <XAxis
                  dataKey="udi"
                  tick={{ fill: "#94a3b8", fontSize: 10 }}
                />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#11161d",
                    border: "1px solid #334155",
                    borderRadius: 8,
                  }}
                  formatter={(chartValue: number) => [
                    value(chartValue),
                    "Hotelling's T²",
                  ]}
                />
                <Bar dataKey="t2" fill="#e67e22" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>
      </div>

      <SectionCard title="Top 10 Phase-I T² Observations">
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>UDI</th>
                <th>Hotelling&apos;s T²</th>
              </tr>
            </thead>
            <tbody>
              {diagnostics.top10_phase1_t2.map((row, index) => (
                <tr key={row.UDI}>
                  <td>{index + 1}</td>
                  <td>{row.UDI}</td>
                  <td className="value-mono">{value(row.Hotelling_T2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}