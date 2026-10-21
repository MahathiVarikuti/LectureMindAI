"use client";

import { FormEvent, useRef, useState } from "react";
import { errorMessage, uploadFile, uploadUrl } from "@/lib/api";

const ACCEPT = ".pdf,.mp3,.wav,.m4a,.ogg,.flac,.aac,.mp4,.webm,.mkv,.avi,.mov";

interface UploadPanelProps {
  onUploaded: () => void;
}

export default function UploadPanel({ onUploaded }: UploadPanelProps) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [url, setUrl] = useState("");
  const [frames, setFrames] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const run = async (label: string, job: () => Promise<{ message?: string; duplicate?: boolean }>) => {
    setBusy(true);
    setStatus({ kind: "ok", text: label });
    try {
      const result = await job();
      setStatus({ kind: "ok", text: result.duplicate ? "Already uploaded." : result.message ?? "Done." });
      onUploaded();
    } catch (error) {
      setStatus({ kind: "error", text: errorMessage(error, "Upload failed.") });
    } finally {
      setBusy(false);
    }
  };

  const onFile = (file: File | undefined) => {
    if (!file) return;
    void run(`Processing ${file.name} — audio/video can take a few minutes…`, () => uploadFile(file));
    if (fileInput.current) fileInput.current.value = "";
  };

  const onUrl = (event: FormEvent) => {
    event.preventDefault();
    if (!url.trim()) return;
    void run("Downloading and processing…", () => uploadUrl(url.trim(), frames)).then(() => setUrl(""));
  };

  return (
    <div className="upload-panel">
      <label className={`new-chat-button upload-button ${busy ? "busy" : ""}`}>
        {busy ? "Processing…" : "＋ Upload PDF, audio or video"}
        <input ref={fileInput} type="file" accept={ACCEPT} hidden disabled={busy}
               onChange={(e) => onFile(e.target.files?.[0])} />
      </label>
      <form onSubmit={onUrl} className="url-form">
        <input type="url" value={url} onChange={(e) => setUrl(e.target.value)} disabled={busy}
               placeholder="…or paste an audio/video URL" aria-label="Media URL" />
        <button type="submit" disabled={busy || !url.trim()}>Add</button>
        <label className="url-frames">
          <input type="checkbox" checked={frames} onChange={(e) => setFrames(e.target.checked)} disabled={busy} />
          also index video frames (slower)
        </label>
      </form>
      <p className="upload-disclaimer">
        URL import is for your own recordings or openly-licensed content only.
      </p>
      {status && <p className={status.kind === "error" ? "upload-status error" : "upload-status"} role="status">{status.text}</p>}
    </div>
  );
}
