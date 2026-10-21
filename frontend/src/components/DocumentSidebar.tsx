"use client";

import { DocumentInfo, deleteDocument } from "@/lib/api";

interface Props {
  sessionId: string;
  documents: DocumentInfo[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onDeleted: () => void;
}

function formatModality(m: string) {
  const map: Record<string, string> = {
    pdf: "PDF",
    audio: "Audio",
    video: "Video",
    video_transcript: "Video",
  };
  return map[m] || m;
}

export default function DocumentSidebar({
  sessionId,
  documents,
  selected,
  onToggle,
  onDeleted,
}: Props) {
  const handleDelete = async (docId: string) => {
    if (!confirm("Delete this document and all its indexed content?")) return;
    try {
      await deleteDocument(sessionId, docId);
      onDeleted();
    } catch {
      alert("Failed to delete document");
    }
  };

  return (
    <div className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white/85 p-4 shadow-[0_12px_30px_rgba(15,23,42,0.06)] backdrop-blur-sm sm:p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
            Library
          </p>
          <h2 className="mt-1 text-lg font-semibold text-slate-900">
            Documents <span className="text-slate-400">({documents.length})</span>
          </h2>
        </div>
        {selected.size > 0 && (
          <span className="rounded-full bg-blue-50 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-blue-700">
            {selected.size} selected
          </span>
        )}
      </div>

      {documents.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-8 text-center">
          <p className="text-sm font-medium text-slate-700">No documents yet</p>
          <p className="mt-1 text-xs text-slate-500">Upload a PDF, audio file, or video to get started.</p>
        </div>
      ) : (
        <ul className="space-y-2 overflow-y-auto pr-1">
          {documents.map((doc) => (
            <li
              key={doc.document_id}
              className={`flex items-start gap-3 rounded-2xl border p-3 transition ${
                selected.has(doc.document_id)
                  ? "border-blue-200 bg-blue-50/80"
                  : "border-slate-200 bg-slate-50/60 hover:border-slate-300 hover:bg-slate-100/80"
              }`}
            >
              <input
                type="checkbox"
                checked={selected.has(doc.document_id)}
                onChange={() => onToggle(doc.document_id)}
                className="mt-1 h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                title="Include in query scope"
              />

              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-800" title={doc.filename}>
                  {doc.filename}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {formatModality(doc.modality)} · {doc.chunk_count} chunks
                </p>
              </div>

              <button
                onClick={() => handleDelete(doc.document_id)}
                className="shrink-0 rounded-lg border border-red-200 bg-red-50 px-2 py-1 text-[10px] font-medium text-red-600 transition hover:bg-red-100"
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}

      {selected.size > 0 && (
        <p className="mt-4 rounded-xl border border-blue-100 bg-blue-50 px-3 py-2 text-xs text-blue-700">
          Query scoped to {selected.size} document{selected.size > 1 ? "s" : ""}
        </p>
      )}
    </div>
  );
}
