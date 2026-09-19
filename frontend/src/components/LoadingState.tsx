interface LoadingStateProps {
  message?: string;
}

export default function LoadingState({
  message = "Loading data from the FastAPI backend...",
}: LoadingStateProps) {
  return (
    <div className="page-state page-state-loading">
      <div className="loading-spinner" />
      <p>{message}</p>
    </div>
  );
}