interface MetricCardProps {
  label: string;
  value: string;
  caption?: string;
  tone?: "default" | "normal" | "watch" | "alert" | "critical" | "blue";
}

export default function MetricCard({
  label,
  value,
  caption,
  tone = "default",
}: MetricCardProps) {
  return (
    <article className={`metric-card metric-card-${tone}`}>
      <div className="metric-card-label">{label}</div>
      <div className="metric-card-value">{value}</div>
      {caption ? (
        <div className="metric-card-caption">{caption}</div>
      ) : null}
    </article>
  );
}