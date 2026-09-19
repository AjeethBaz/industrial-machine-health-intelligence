interface ErrorStateProps {
  message: string;
}

export default function ErrorState({ message }: ErrorStateProps) {
  return (
    <div className="page-state page-state-error">
      <h2>Unable to load this analysis</h2>
      <p>{message}</p>
      <p className="page-state-hint">
        Confirm that FastAPI is running at
        {" "}
        <code>http://127.0.0.1:8000</code>.
      </p>
    </div>
  );
}