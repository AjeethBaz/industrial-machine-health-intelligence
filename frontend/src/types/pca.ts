export interface PCAResponse {
  components: string[];
  explained_variance: number[];
  cumulative: number[];
  loadings: Record<string, Record<string, number>>;
  components_required_for_90_percent: number;
}