import axios from "axios";
import type { OverviewResponse } from "../types/overview";
import type { MachineHealthResponse } from "../types/machineHealth";
import type { SPCResponse } from "../types/spc";
import type { PCAResponse } from "../types/pca";
import type { T2EvaluationResponse } from "../types/evaluation";
import type { ManovaResponse } from "../types/manova";
import type { DiagnosticsResponse } from "../types/diagnostics";
import type { MaintenanceResponse } from "../types/maintenance";
import type { AssistantResponse } from "../types/assistant";

export const API_BASE_URL = "http://127.0.0.1:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export async function fetchOverview(): Promise<OverviewResponse> {
  const response = await apiClient.get<OverviewResponse>("/api/overview");
  return response.data;
}

export async function fetchMachineHealth(
  udi: number
): Promise<MachineHealthResponse> {
  const response = await apiClient.get<MachineHealthResponse>(
    `/api/machines/${udi}`
  );
  return response.data;
}

export async function fetchSPC(): Promise<SPCResponse> {
  const response = await apiClient.get<SPCResponse>("/api/spc");
  return response.data;
}

export async function fetchPCA(): Promise<PCAResponse> {
  const response = await apiClient.get<PCAResponse>("/api/pca");
  return response.data;
}

export async function fetchT2Evaluation(): Promise<T2EvaluationResponse> {
  const response = await apiClient.get<T2EvaluationResponse>(
    "/api/t2-evaluation"
  );
  return response.data;
}

export async function fetchMANOVA(): Promise<ManovaResponse> {
  const response = await apiClient.get<ManovaResponse>("/api/manova");
  return response.data;
}

export async function fetchDiagnostics(): Promise<DiagnosticsResponse> {
  const response = await apiClient.get<DiagnosticsResponse>(
    "/api/diagnostics"
  );
  return response.data;
}

export async function fetchMaintenance(
  udi: number
): Promise<MaintenanceResponse> {
  const response = await apiClient.get<MaintenanceResponse>(
    `/api/maintenance/${udi}`
  );
  return response.data;
}

export async function sendAssistantMessage(
  udi: number,
  question?: string
): Promise<AssistantResponse> {
  const response = await apiClient.post<AssistantResponse>(
    "/api/assistant",
    {
      udi,
      ...(question?.trim() ? { question: question.trim() } : {}),
    }
  );

  return response.data;
}

export default apiClient;