"use client";

import { useRef, useState } from "react";
import { errorMessage, uploadFile } from "@/lib/api";

const ACCEPT = ".pdf,.mp3,.wav,.m4a,.ogg,.flac,.aac,.mp4,.webm,.mkv,.avi,.mov";

interface UploadPanelProps {
  onUploaded: () => void;
}

export default function UploadPanel({ onUploaded }: UploadPanelProps) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
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

  return (
    <div className="upload-panel">
      <label className={`new-chat-button upload-button ${busy ? "busy" : ""}`}>
        {busy ? "Processing…" : "＋ Upload PDF, audio or video"}
        <input ref={fileInput} type="file" accept={ACCEPT} hidden disabled={busy}
               onChange={(e) => onFile(e.target.files?.[0])} />
      </label>
      {status && <p className={status.kind === "error" ? "upload-status error" : "upload-status"} role="status">{status.text}</p>}
    </div>
  );
}
