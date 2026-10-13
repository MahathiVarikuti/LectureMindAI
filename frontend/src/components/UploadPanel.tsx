"use client";

import { useRef, useState } from "react";
import { uploadFile, uploadUrl } from "@/lib/api";

interface Props {
  sessionId: string;
  onUploaded: () => void;
}

export default function UploadPanel({ sessionId, onUploaded }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const result = await uploadFile(sessionId, file);
      setMessage(result.message);
      onUploaded();
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail || "Upload failed";
      setError(String(msg));
    } finally {
      setLoading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleUrl = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const result = await uploadUrl(sessionId, url.trim());
      setMessage(result.message);
      setUrl("");
      onUploaded();
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail || "URL ingestion failed";
      setError(String(msg));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/85 p-4 shadow-[0_12px_30px_rgba(15,23,42,0.06)] backdrop-blur-sm sm:p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
            Upload
          </p>
          <h2 className="mt-1 text-lg font-semibold text-slate-900">Lecture material</h2>
        </div>
        <div className="rounded-full border border-blue-100 bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700">
          PDF / Audio / Video
        </div>
      </div>

      <label className="group block cursor-pointer rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-4 transition hover:border-blue-400 hover:bg-blue-50/50">
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.mp3,.wav,.m4a,.ogg,.flac,.mp4,.webm,.mkv,.avi,.mov"
          onChange={handleFile}
          disabled={loading}
          className="hidden"
        />
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-sm font-medium text-slate-800">Select a file</p>
            <p className="mt-1 text-xs text-slate-500">PDFs, audio, or video</p>
          </div>
          <div className="rounded-xl bg-white px-3 py-2 text-xs font-semibold text-blue-700 shadow-sm ring-1 ring-slate-200 transition group-hover:ring-blue-200">
            Browse
          </div>
        </div>
      </label>

      <div className="mt-4 space-y-3">
        <div className="flex items-center gap-2">
          <div className="h-px flex-1 bg-slate-200" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
            or
          </span>
          <div className="h-px flex-1 bg-slate-200" />
        </div>

        <div className="flex gap-2">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Paste a media URL"
            disabled={loading}
            className="flex-1 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-700 placeholder:text-slate-400 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
          />
          <button
            onClick={handleUrl}
            disabled={loading || !url.trim()}
            className="rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            Add
          </button>
        </div>
      </div>

      <p className="mt-4 text-xs leading-5 text-amber-700">
        Only upload your own recordings or openly-licensed content. This tool is not a general media downloader.
      </p>

      {loading && (
        <div className="mt-4 flex items-center gap-2 rounded-xl border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-700">
          <span className="h-2 w-2 animate-pulse rounded-full bg-blue-600" />
          Processing upload...
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {message && (
        <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
          {message}
        </div>
      )}
    </div>
  );
}
