import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  timeout: 300000,
});

export interface DocumentInfo {
  document_id: string;
  filename: string;
  modality: string;
  chunk_count: number;
  uploaded_at: string;
}

export interface SourceCitation {
  document_id: string;
  filename: string;
  modality: string;
  page?: number | null;
  timestamp_start?: number | null;
  timestamp_end?: number | null;
  timestamp?: number | null;
  snippet: string;
}

export interface QueryResponse {
  answer: string;
  sources: SourceCitation[];
  used_documents: string[];
  context?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: SourceCitation[];
}

export async function uploadFile(sessionId: string, file: File) {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", sessionId);
  const { data } = await api.post("/upload", form);
  return data;
}

export async function uploadUrl(sessionId: string, url: string) {
  const { data } = await api.post("/upload-url", { url, session_id: sessionId });
  return data;
}

export async function queryDocuments(
  sessionId: string,
  query: string,
  selectedDocuments?: string[]
) {
  const { data } = await api.post<QueryResponse>("/query", {
    query,
    session_id: sessionId,
    selected_documents: selectedDocuments?.length ? selectedDocuments : undefined,
  });
  return data;
}

export async function listDocuments(sessionId: string) {
  const { data } = await api.get<DocumentInfo[]>("/documents", {
    params: { session_id: sessionId },
  });
  return data;
}

export async function deleteDocument(sessionId: string, documentId: string) {
  const { data } = await api.delete(`/documents/${documentId}`, {
    params: { session_id: sessionId },
  });
  return data;
}

export async function getChatHistory(sessionId: string) {
  const { data } = await api.get<ChatMessage[]>("/messages", {
    params: { session_id: sessionId },
  });
  return data;
}

export async function healthCheck() {
  const { data } = await api.get("/health");
  return data;
}
