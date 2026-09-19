import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchT2Evaluation } from "../api/client";
import type { T2EvaluationResponse } from "../types/evaluation";
import MetricCard from "../components/MetricCard";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

function percent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function decimal(value: number): string {
  return value.toFixed(4);
}

export default function T2Evaluation() {
  const [evaluation, setEvaluation] = useState<T2EvaluationResponse | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadEvaluation() {
      try {
        const data = await fetchT2Evaluation();

        if (active) {
          setEvaluation(data);
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
              "Unable to load T² evaluation."
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadEvaluation();

    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return <LoadingState message="Loading T² monitoring evaluation..." />;
  }

  if (error || !evaluation) {
    return (
      <ErrorState message={error ?? "T² evaluation results are unavailable."} />
    );
  }

  const distributionChart = evaluation.group_comparison.map((group) => ({
    group: group.group,
    mean: group.mean,
    median: group.median,
    q75: group.q75,
  }));

  return (
    <div>
      <div className="content-page-title">T² Evaluation</div>
      <div className="content-page-subtitle">
        Evaluation of Phase-II Hotelling&apos;s T² monitoring alerts against
        the observed Machine failure outcome.
      </div>

      <div className="metric-grid machine-kpi-grid">
        <MetricCard
          label="Precision"
          value={percent(evaluation.precision)}
          caption="Observed failures among T² alerts"
          tone="alert"
        />
        <MetricCard
          label="Recall"
          value={percent(evaluation.recall)}
          caption="Observed failures captured"
          tone="normal"
        />
        <MetricCard
          label="Specificity"
          value={percent(evaluation.specificity)}
          caption="Non-failure observations without alert"
          tone="blue"
        />
        <MetricCard
          label="F1 Score"
          value={percent(evaluation.f1_score)}
          caption="Precision-recall balance"
        />
        <MetricCard
          label="False Positive Rate"
          value={percent(evaluation.false_positive_rate)}
          caption="Non-failure alerts"
          tone="watch"
        />
        <MetricCard
          label="Alert Rate"
          value={percent(evaluation.alert_rate)}
          caption="Phase-II T² alerts"
          tone="alert"
        />
        <MetricCard
          label="Observed Failure Rate"
          value={percent(evaluation.actual_failure_rate)}
          caption="Machine failure outcome label"
        />
        <MetricCard
          label="Failure Capture Rate"
          value={percent(evaluation.failure_capture_rate)}
          caption="Equivalent to recall"
          tone="normal"
        />
      </div>

      <div className="analytics-grid">
        <SectionCard
          title="Confusion Matrix"
          caption="Comparison of existing T² alert flags against observed Machine failure labels."
        >
          <div className="confusion-grid">
            <div className="confusion-heading">Observed Outcome / Monitoring Signal</div>
            <div className="confusion-heading">T² Alert</div>
            <div className="confusion-heading">No T² Alert</div>

            <div className="confusion-label">Observed Failure</div>
            <div className="confusion-cell confusion-tp">
              <strong>{evaluation.true_positives}</strong>
              <span>True Positive</span>
            </div>
            <div className="confusion-cell confusion-fn">
              <strong>{evaluation.false_negatives}</strong>
              <span>False Negative</span>
            </div>

            <div className="confusion-label">Observed Non-failure</div>
            <div className="confusion-cell confusion-fp">
              <strong>{evaluation.false_positives}</strong>
              <span>False Positive</span>
            </div>
            <div className="confusion-cell confusion-tn">
              <strong>{evaluation.true_negatives}</strong>
              <span>True Negative</span>
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Monitoring Interpretation">
          <div className="info-panel">
            <strong>T² is an abnormality-monitoring statistic.</strong>
            <p>
              It is not inherently a trained failure classifier. A T² alert
              indicates an unusual multivariate sensor state relative to the
              healthy baseline; it does not automatically mean the machine
              will fail.
            </p>
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="T² Distribution by Observed Failure Status"
        caption="Failure and non-failure groups are shown for post-hoc evaluation of monitoring behavior."
      >
        <div className="chart-box">
          <ResponsiveContainer width="100%" height={330}>
            <BarChart data={distributionChart}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(148,163,184,0.18)"
                vertical={false}
              />
              <XAxis
                dataKey="group"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
              />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#11161d",
                  border: "1px solid #334155",
                  borderRadius: 8,
                }}
                formatter={(value: number, name: string) => [
                  decimal(value),
                  name,
                ]}
              />
              <Legend />
              <Bar
                dataKey="mean"
                fill="#3b82f6"
                name="Mean T²"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="median"
                fill="#22d3ee"
                name="Median T²"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="q75"
                fill="#f1c40f"
                name="Q75 T²"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Group</th>
                <th>Count</th>
                <th>Mean T²</th>
                <th>Median</th>
                <th>Std</th>
                <th>Q25</th>
                <th>Q75</th>
                <th>Alerts</th>
                <th>Alert %</th>
              </tr>
            </thead>
            <tbody>
              {evaluation.group_comparison.map((group) => (
                <tr key={group.group}>
                  <td>{group.group}</td>
                  <td>{group.count.toLocaleString()}</td>
                  <td className="value-mono">{decimal(group.mean)}</td>
                  <td className="value-mono">{decimal(group.median)}</td>
                  <td className="value-mono">{decimal(group.std)}</td>
                  <td className="value-mono">{decimal(group.q25)}</td>
                  <td className="value-mono">{decimal(group.q75)}</td>
                  <td>{group.alerts.toLocaleString()}</td>
                  <td className="value-mono">
                    {group.alert_percentage.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}