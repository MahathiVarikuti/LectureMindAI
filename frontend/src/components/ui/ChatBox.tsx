"use client";

import api from "@/lib/api";
import { FormEvent, useEffect, useMemo, useState } from "react";
import SourceCard, { Source } from "@/components/ui/SourceCard";
import MessageBubble from "@/components/ui/MessageBubble";

interface Grounding {
  groundedness: number;
  unsupported: string[];
}

interface Timings {
  retrieval?: number;
  generation?: number;
  grounding?: number;
  total?: number;
}

interface QueryResponse {
  answer: string;
  rewritten_query: string;
  abstained: boolean;
  abstain_reason: string | null;
  top_score: number;
  flagged: boolean;
  grounding: Grounding | null;
  sources: Source[];
  timings: Timings;
}

interface DocumentInfo {
  document_id: string;
  filename: string;
  chunk_count: number;
}

const SESSION_KEY = "lecturemindai_session_id";

function getSessionId() {
  const existing = window.localStorage.getItem(SESSION_KEY);
  if (existing) return existing;
  const id = window.crypto.randomUUID();
  window.localStorage.setItem(SESSION_KEY, id);
  return id;
}

export default function ChatBox() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const selectedNames = useMemo(
    () => documents.filter((doc) => selected.includes(doc.document_id)).map((doc) => doc.filename),
    [documents, selected],
  );

  const [uploading, setUploading] = useState(false);

  const loadDocuments = () =>
    api
      .get("/documents")
      .then((res) => setDocuments(res.data as DocumentInfo[]))
      .catch((error) => console.error("Error loading documents:", error));

  useEffect(() => {
    loadDocuments();
  }, []);

  const uploadPdf = async (file: File | undefined) => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      setUploading(true);
      const response = await api.post("/upload", formData);
      if (response.data.duplicate) alert("This file was already uploaded.");
      await loadDocuments();
    } catch (error) {
      console.error("Error uploading file:", error);
      const detail = (error as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      alert(detail || "Failed to upload file.");
    } finally {
      setUploading(false);
    }
  };

  const toggleDocument = (id: string) =>
    setSelected((prev) => (prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]));

  const askQuestion = async (event?: FormEvent) => {
    event?.preventDefault();
    if (!query.trim()) return;
    try {
      setLoading(true);
      const response = await api.post("/query", {
        query,
        session_id: getSessionId(),
        selected_documents: selected,
      });
      setResult(response.data as QueryResponse);
    } catch (error) {
      console.error("Error asking question:", error);
      alert("Failed to get answer.");
    } finally {
      setLoading(false);
    }
  };

  const newChat = async () => {
    const old = window.localStorage.getItem(SESSION_KEY);
    if (old) await api.delete(`/session/${old}`).catch(() => undefined);
    window.localStorage.removeItem(SESSION_KEY);
    setResult(null);
    setQuery("");
  };

  return (
    <div className="workspace">
      <aside className="workspace-sidebar">
        <div className="brand-mark"><span>LM</span><div><strong>LectureMind</strong><small>your study companion</small></div></div>
        <div className="sidebar-section">
          <div className="sidebar-label">Knowledge base <span>{documents.length}</span></div>
          <p className="sidebar-hint">Choose documents to narrow retrieval. No selection searches everything.</p>
          <label className="new-chat-button" style={{ cursor: uploading ? "wait" : "pointer", textAlign: "center" }}>
            {uploading ? "Uploading…" : "＋ Upload PDF"}
            <input
              type="file"
              accept="application/pdf,.pdf"
              hidden
              disabled={uploading}
              onChange={(e) => {
                uploadPdf(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </label>
          <div className="document-list">
            {documents.length === 0 ? <div className="empty-docs">Upload notes to begin.</div> : documents.map((doc) => (
              <label key={doc.document_id} className={`document-row ${selected.includes(doc.document_id) ? "selected" : ""}`}>
                <input type="checkbox" checked={selected.includes(doc.document_id)} onChange={() => toggleDocument(doc.document_id)} />
                <span className="document-icon">PDF</span>
                <span className="document-name" title={doc.filename}>{doc.filename}</span>
                <span className="document-count">{doc.chunk_count}</span>
              </label>
            ))}
          </div>
        </div>
        <button onClick={newChat} className="new-chat-button">＋ New conversation</button>
        <div className="sidebar-footer">Retrieval: hybrid search + reranking<br />Answers are grounded in your uploaded notes.</div>
      </aside>

      <main className="chat-pane">
        <header className="chat-header">
          <div><span className="eyebrow">Study workspace</span><h1>Ask your notes</h1><p>{selectedNames.length ? `Searching ${selectedNames.join(", ")}` : "Search across your entire knowledge base"}</p></div>
          <div className="status-pill"><span /> API connected</div>
        </header>

        <div className="conversation">
          {!result && !loading && <div className="welcome"><div className="welcome-icon">✦</div><h2>What would you like to learn?</h2><p>Ask a precise question, compare concepts, or request a summary from your uploaded material.</p><div className="prompt-grid">{["Summarize the key ideas", "Explain this in simple terms", "What are the required qualifications?"].map((prompt) => <button key={prompt} onClick={() => setQuery(prompt)}>{prompt}<span>→</span></button>)}</div></div>}
          {result && <div className="answer-card"><div className="answer-heading"><div><span className="answer-label">Answer</span>{result.rewritten_query && result.rewritten_query !== query && <p className="rewritten">Searched as: {result.rewritten_query}</p>}</div><span className={result.flagged ? "confidence warning" : "confidence"}>{result.flagged ? "Review answer" : "Grounded answer"}</span></div><MessageBubble content={result.answer} role="assistant" />
            {result.flagged && result.grounding && <div className="grounding-warning"><strong>Some claims need review</strong><span>{Math.round(result.grounding.groundedness * 100)}% of answer claims matched retrieved notes.</span>{result.grounding.unsupported.map((sentence) => <p key={sentence}>“{sentence}”</p>)}</div>}
            <div className="metrics"><div><span>Top relevance</span><strong>{Math.round(result.top_score * 100)}%</strong><small>reranker score</small></div><div><span>Groundedness</span><strong>{result.grounding ? `${Math.round(result.grounding.groundedness * 100)}%` : "—"}</strong><small>claim support</small></div><div><span>Sources</span><strong>{result.sources.length}</strong><small>retrieved chunks</small></div><div><span>Response time</span><strong>{result.timings.total ? `${result.timings.total.toFixed(1)}s` : "—"}</strong><small>end to end</small></div></div>
          </div>}
          {loading && <div className="loading-card"><span className="loading-dot" /><span>Reading your notes and checking relevant passages…</span></div>}
          {result?.sources.length ? <section className="sources-section"><div className="section-heading"><div><span className="eyebrow">Evidence</span><h2>Retrieved passages</h2></div><span>{result.sources.length} sources</span></div><div className="source-grid">{result.sources.map((source, index) => <SourceCard key={source.chunk_id} source={source} index={index + 1} />)}</div></section> : null}
        </div>

        <form className="composer" onSubmit={askQuestion}><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void askQuestion(); } }} placeholder="Ask anything about your notes…" rows={1} disabled={loading} /><button type="submit" disabled={loading || !query.trim()} aria-label="Ask question">{loading ? "…" : "↑"}</button><span className="composer-hint">Enter to ask · Shift + Enter for new line</span></form>
      </main>
    </div>
  );
}
