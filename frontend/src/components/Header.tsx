import MachineSelector from "./MachineSelector";

interface HeaderProps {
  onMachineSelect: (udi: number) => void;
}

export default function Header({ onMachineSelect }: HeaderProps) {
  return (
    <header className="app-header">
      <div>
        <div className="page-title">
          Industrial Machine Health Intelligence Platform
        </div>
        <div className="page-subtitle">
          Multivariate Statistical Process Control for Predictive Maintenance
        </div>
      </div>

      <MachineSelector onSelect={onMachineSelect} />
    </header>
  );
}