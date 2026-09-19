import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useNavigate,
} from "react-router-dom";

import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import MachineHealth from "./pages/MachineHealth";
import SPCMonitoring from "./pages/SPCMonitoring";
import PCAAnalysis from "./pages/PCAAnalysis";
import T2Evaluation from "./pages/T2Evaluation";
import ManovaAnalysis from "./pages/ManovaAnalysis";
import Diagnostics from "./pages/Diagnostics";
import Maintenance from "./pages/Maintenance";
import AIAssistant from "./pages/AIAssistant";

function AppRoutes() {
  const navigate = useNavigate();

  function handleMachineSelect(udi: number) {
    navigate(`/machine/${udi}`);
  }

  return (
    <Layout onMachineSelect={handleMachineSelect}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/machine/:udi" element={<MachineHealth />} />
        <Route path="/spc" element={<SPCMonitoring />} />
        <Route path="/pca" element={<PCAAnalysis />} />
        <Route path="/evaluation" element={<T2Evaluation />} />
        <Route path="/manova" element={<ManovaAnalysis />} />
        <Route path="/diagnostics" element={<Diagnostics />} />
        <Route path="/maintenance" element={<Maintenance />} />
        <Route path="/assistant" element={<AIAssistant />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}