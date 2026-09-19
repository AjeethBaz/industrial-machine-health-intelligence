import type { ReactNode } from "react";
import Sidebar from "./Sidebar";
import Header from "./Header";

interface LayoutProps {
  children: ReactNode;
  onMachineSelect: (udi: number) => void;
}

export default function Layout({
  children,
  onMachineSelect,
}: LayoutProps) {
  return (
    <div className="app-shell">
      <Sidebar />
      <Header onMachineSelect={onMachineSelect} />
      <main className="app-main">{children}</main>
    </div>
  );
}