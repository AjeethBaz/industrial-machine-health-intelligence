interface KPICardProps {
  label: string;
  value: string;
  caption?: string;
}

export default function KPICard({ label, value, caption }: KPICardProps) {
  return (
    <div className="kpi-card">
      <div className="kpi-card-label">{label}</div>
      <div className="kpi-card-value">{value}</div>
      {caption && <div className="kpi-card-caption">{caption}</div>}
    </div>
  );
}