import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchMANOVA } from "../api/client";
import type { ManovaResponse } from "../types/manova";
import MetricCard from "../components/MetricCard";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

function formatValue(value: number, digits = 4): string {
  return value.toFixed(digits);
}

function formatPValue(value: number): string {
  if (value < 0.001) {
    return "p < 0.001";
  }

  return `p = ${value.toFixed(4)}`;
}

export default function ManovaAnalysis() {
  const [manova, setManova] = useState<ManovaResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadMANOVA() {
      try {
        const data = await fetchMANOVA();

        if (active) {
          setManova(data);
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
              "Unable to load MANOVA results."
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadMANOVA();

    return () => {
      active = false;
    };
  }, []);

  const meanDifferenceData = useMemo(() => {
    if (!manova) {
      return [];
    }

    return Object.entries(manova.mean_differences).map(
      ([sensor, difference]) => ({
        sensor,
        difference,
      })
    );
  }, [manova]);

  if (loading) {
    return <LoadingState message="Loading MANOVA group comparison..." />;
  }

  if (error || !manova) {
    return <ErrorState message={error ?? "MANOVA data is unavailable."} />;
  }

  const nonFailureCount = manova.group_sizes["Non-failure"] ?? 0;
  const failureCount = manova.group_sizes["Failure"] ?? 0;
  const imbalance =
    failureCount > 0 ? `${(nonFailureCount / failureCount).toFixed(1)} : 1` : "N/A";

  return (
    <div>
      <div className="content-page-title">MANOVA</div>
      <div className="content-page-subtitle">
        Multivariate comparison of raw sensor mean vectors between observed
        failure and non-failure groups.
      </div>

      <div className="metric-grid machine-kpi-grid">
        <MetricCard
          label="Non-failure Group"
          value={nonFailureCount.toLocaleString()}
          caption="Machine failure = 0"
          tone="normal"
        />
        <MetricCard
          label="Failure Group"
          value={failureCount.toLocaleString()}
          caption="Machine failure = 1"
          tone="alert"
        />
        <MetricCard
          label="Group Imbalance"
          value={imbalance}
          caption="Non-failure : failure"
          tone="watch"
        />
      </div>

      <SectionCard
        title="Multivariate Test Statistics"
        caption="Raw p-values are returned by FastAPI. The frontend only applies conventional display formatting."
      >
        <div className="test-grid">
          {manova.multivariate_tests.map((test) => (
            <article className="test-card" key={test.test_name}>
              <div className="test-name">{test.test_name}</div>
              <div className="test-value">{formatValue(test.value, 6)}</div>
              <div className="test-details">
                F = {formatValue(test.f_value, 4)}
                <br />
                df = {formatValue(test.num_df, 0)},{" "}
                {formatValue(test.den_df, 0)}
              </div>
              <div className="test-pvalue">{formatPValue(test.p_value)}</div>
            </article>
          ))}
        </div>
      </SectionCard>

      <SectionCard
        title="Group Mean Differences"
        caption="Raw sensor mean difference: failure group minus non-failure group."
      >
        <div className="chart-box">
          <ResponsiveContainer width="100%" height={360}>
            <BarChart
              data={meanDifferenceData}
              layout="vertical"
              margin={{ top: 10, right: 20, left: 180, bottom: 10 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(148,163,184,0.18)"
              />
              <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="sensor"
                width={170}
                tick={{ fill: "#cbd5e1", fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#11161d",
                  border: "1px solid #334155",
                  borderRadius: 8,
                }}
                formatter={(value: number) => [
                  formatValue(value, 4),
                  "Failure − Non-failure",
                ]}
              />
              <Bar
                dataKey="difference"
                fill="#60a5fa"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Sensor</th>
                <th>Failure − Non-failure Mean Difference</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(manova.mean_differences).map(
                ([sensor, difference]) => (
                  <tr key={sensor}>
                    <td>{sensor}</td>
                    <td className={difference >= 0 ? "value-high" : "value-low"}>
                      {difference >= 0 ? "+" : ""}
                      {formatValue(difference, 4)}
                    </td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard
        title="Sensor Descriptive Statistics"
        caption="Raw-unit statistics by observed Machine failure group."
      >
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Group</th>
                <th>Sensor</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Median</th>
              </tr>
            </thead>
            <tbody>
              {manova.descriptive_stats.map((row, index) => (
                <tr
                  key={`${row["Machine failure"]}-${row.Sensor}-${index}`}
                >
                  <td>{row["Machine failure"]}</td>
                  <td>{row.Sensor}</td>
                  <td>{row.count.toLocaleString()}</td>
                  <td className="value-mono">{formatValue(row.mean, 4)}</td>
                  <td className="value-mono">{formatValue(row.std, 4)}</td>
                  <td className="value-mono">{formatValue(row.median, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Interpretation and Limitations">
        <div className="info-panel">
          <strong>Non-causal interpretation</strong>
          <p>
            A statistically significant MANOVA result supports the conclusion
            that the joint sensor mean vector differs between observed failure
            and non-failure groups. It does not demonstrate that the sensor
            variables cause machine failure.
          </p>
        </div>

        <ul className="bullet-list muted-list">
          {manova.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </SectionCard>
    </div>
  );
}