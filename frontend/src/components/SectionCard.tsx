import type { ReactNode } from "react";

interface SectionCardProps {
  title?: string;
  caption?: string;
  children: ReactNode;
  className?: string;
}

export default function SectionCard({
  title,
  caption,
  children,
  className = "",
}: SectionCardProps) {
  return (
    <section className={`section-card ${className}`}>
      {title ? <h2 className="section-card-title">{title}</h2> : null}
      {caption ? (
        <p className="section-card-caption">{caption}</p>
      ) : null}
      {children}
    </section>
  );
}