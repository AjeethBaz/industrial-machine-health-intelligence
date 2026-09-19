export interface AssistantResponse {
  udi: number;
  response: string;
  failure_probability: number | null;
  failure_probability_note: string;
}