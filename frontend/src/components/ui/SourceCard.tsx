import type { Source } from "@/lib/api";

export function formatTime(seconds: number | null | undefined): string {
  const total = Math.max(0, Math.floor(seconds ?? 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** Page for PDFs, timestamp (range) for audio/video, frame time for video frames. */
export function citationLabel(source: Source): string {
  if (source.modality === "video_frame") return `video frame at ${formatTime(source.timestamp)}`;
  if (source.timestamp_start !== null && source.timestamp_start !== undefined) {
    const end = source.timestamp_end !== null && source.timestamp_end !== undefined ? `–${formatTime(source.timestamp_end)}` : "";
    return `${formatTime(source.timestamp_start)}${end}`;
  }
  return source.page ? `page ${source.page}` : "";
}

interface SourceCardProps {
  source: Source;
  index: number;
}

export default function SourceCard({ source, index }: SourceCardProps) {
  const label = citationLabel(source);
  return (
    <div className="source-card">
      <div className="source-card-header">
        <h4>
          [{index}] {source.filename ?? "Unknown source"}
          {label ? ` · ${label}` : ""}
        </h4>
        <span className="source-score">{(source.score * 100).toFixed(0)}% match</span>
      </div>
      {source.text && (
        <p>{source.text.length > 300 ? `${source.text.slice(0, 300)}...` : source.text}</p>
      )}
    </div>
  );
}
