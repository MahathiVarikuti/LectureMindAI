import axios from "axios";
import { getSessionId } from "@/lib/session";

// Configure with NEXT_PUBLIC_API_URL (see .env.example); never hardcode a deployment URL.
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
});

export default api;

export type Modality = "pdf" | "audio" | "video";

export interface DocumentInfo {
  document_id: string;
  filename: string;
  modality: Modality;
  chunk_count: number;
  uploaded_at: string;
}

export interface Source {
  chunk_id: string;
  document_id: string | null;
  filename: string | null;
  modality: string | null; // pdf | audio | video_transcript | video_frame
  page: number | null;
  timestamp_start: number | null;
  timestamp_end: number | null;
  timestamp: number | null;
  text: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  rewritten_query: string;
  abstained: boolean;
  flagged: boolean;
  sources: Source[];
  used_documents: string[];
  context: string;
}

export interface StoredMessage {
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

/** Turn an axios/FastAPI error into a message safe to show the user. */
export function errorMessage(error: unknown, fallback: string): string {
  const err = error as { response?: { status?: number; data?: { detail?: unknown } }; message?: string };
  const detail = err.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (err.response?.status === 429) return "Too many requests. Please wait a moment and try again.";
  if (err.response && !err.response.status) return fallback;
  return err.message === "Network Error" ? "Cannot reach the server. Is the backend running?" : fallback;
}

const session = () => ({ session_id: getSessionId() });

export const listDocuments = () =>
  api.get<DocumentInfo[]>("/documents", { params: session() }).then((r) => r.data);

export const deleteDocument = (id: string) =>
  api.delete(`/documents/${id}`, { params: session() });

export const uploadFile = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", getSessionId());
  // Transcribing media on a CPU can take minutes: no client-side timeout.
  return api.post("/upload", form, { timeout: 0 }).then((r) => r.data);
};

export const uploadUrl = (url: string, includeFrames: boolean) =>
  api.post("/upload-url", { url, session_id: getSessionId(), include_frames: includeFrames }, { timeout: 0 })
    .then((r) => r.data);

export const askQuestion = (query: string, selectedDocuments: string[]) =>
  api.post<QueryResponse>("/query", { query, session_id: getSessionId(), selected_documents: selectedDocuments },
    { timeout: 180000 }).then((r) => r.data);

export const getHistory = () =>
  api.get<{ messages: StoredMessage[] }>("/history", { params: session() }).then((r) => r.data.messages);

export const clearHistory = () => api.delete("/history", { params: session() });
