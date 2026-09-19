import type { ReactNode } from "react";
import LoadingState from "./LoadingState";
import ErrorState from "./ErrorState";

interface PageStateProps {
  loading: boolean;
  error: string | null;
  loadingMessage?: string;
  children: ReactNode;
}

export default function PageState({
  loading,
  error,
  loadingMessage,
  children,
}: PageStateProps) {
  if (loading) {
    return <LoadingState message={loadingMessage} />;
  }

  if (error) {
    return <ErrorState message={error} />;
  }

  return <>{children}</>;
}