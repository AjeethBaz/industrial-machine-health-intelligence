import { NavLink } from "react-router-dom";

interface NavItem {
  label: string;
  path: string;
}

const NAV_ITEMS: NavItem[] = [
  { label: "Dashboard", path: "/" },
  { label: "Machine Health", path: "/machine/7012" },
  { label: "SPC Monitoring", path: "/spc" },
  { label: "PCA Analysis", path: "/pca" },
  { label: "T² Evaluation", path: "/evaluation" },
  { label: "MANOVA", path: "/manova" },
  { label: "Diagnostics", path: "/diagnostics" },
  { label: "Maintenance", path: "/maintenance" },
  { label: "AI Maintenance Assistant", path: "/assistant" },
];

export default function Sidebar() {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-title">
          Machine Health Intelligence
        </div>
        <div className="sidebar-brand-subtitle">
          Multivariate SPC for Predictive Maintenance
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === "/"}
            className={({ isActive }) =>
              `sidebar-nav-item${isActive ? " active" : ""}`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}