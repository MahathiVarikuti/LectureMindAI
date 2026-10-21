export interface Source {
  chunk_id: string;
  document_id: string | null;
  filename: string | null;
  page: number | null;
  text: string;
  score: number;
}

interface SourceCardProps {
  source: Source;
  index: number;
}

export default function SourceCard({ source, index }: SourceCardProps) {
  return (
    <div className="source-card">
      <div className="source-card-header">
        <h4>
          [{index}] {source.filename ?? "Unknown source"}
          {source.page ? ` · page ${source.page}` : ""}
        </h4>
        <span className="source-score">
          {(source.score * 100).toFixed(0)}% match
        </span>
      </div>
      <p>
        {source.text.length > 300 ? `${source.text.slice(0, 300)}...` : source.text}
      </p>
    </div>
  );
}
