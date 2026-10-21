"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  askQuestion, clearHistory, deleteDocument, DocumentInfo, errorMessage, getHistory, listDocuments, Source,
} from "@/lib/api";
import MessageBubble from "@/components/ui/MessageBubble";
import SourceCard from "@/components/ui/SourceCard";
import UploadPanel from "@/components/ui/UploadPanel";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  abstained?: boolean;
  error?: boolean;
}

const MODALITY_LABEL: Record<string, string> = { pdf: "PDF", audio: "AUD", video: "VID" };

export default function ChatBox() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  const selectedNames = useMemo(
    () => documents.filter((doc) => selected.includes(doc.document_id)).map((doc) => doc.filename),
    [documents, selected],
  );

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setSelected((prev) => prev.filter((id) => docs.some((d) => d.document_id === id)));
    } catch (error) {
      console.error("Error loading documents:", error);
    }
  }, []);

  // Restore documents and the persisted conversation (SQLite on the backend) after a reload.
  useEffect(() => {
    void refreshDocuments();
    getHistory()
      .then((stored) => setMessages(stored.map((m) => ({ role: m.role, content: m.content }))))
      .catch((error) => console.error("Error loading history:", error));
  }, [refreshDocuments]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const toggleDocument = (id: string) =>
    setSelected((prev) => (prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]));

  const removeDocument = async (doc: DocumentInfo) => {
    if (!window.confirm(`Delete "${doc.filename}" and its indexed content?`)) return;
    try {
      await deleteDocument(doc.document_id);
      await refreshDocuments();
    } catch (error) {
      alert(errorMessage(error, "Could not delete the document."));
    }
  };

  const ask = async (event?: FormEvent) => {
    event?.preventDefault();
    const text = query.trim();
    if (!text || loading) return;
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setQuery("");
    setLoading(true);
    try {
      const response = await askQuestion(text, selected);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: response.answer, sources: response.sources, abstained: response.abstained },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: errorMessage(error, "Failed to get an answer."), error: true },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const newChat = async () => {
    await clearHistory().catch(() => undefined);
    setMessages([]);
    setQuery("");
  };

  return (
    <div className="workspace">
      <aside className="workspace-sidebar">
        <div className="brand-mark"><span>LM</span><div><strong>LectureMind</strong><small>your study companion</small></div></div>
        <div className="sidebar-section">
          <div className="sidebar-label">Knowledge base <span>{documents.length}</span></div>
          <UploadPanel onUploaded={refreshDocuments} />
          <p className="sidebar-hint">Choose documents to narrow retrieval. No selection searches everything.</p>
          <div className="document-list">
            {documents.length === 0 ? <div className="empty-docs">Upload notes to begin.</div> : documents.map((doc) => (
              <div key={doc.document_id} className={`document-row ${selected.includes(doc.document_id) ? "selected" : ""}`}>
                <label className="document-pick">
                  <input type="checkbox" checked={selected.includes(doc.document_id)} onChange={() => toggleDocument(doc.document_id)} />
                  <span className="document-icon">{MODALITY_LABEL[doc.modality] ?? "DOC"}</span>
                  <span className="document-name" title={doc.filename}>{doc.filename}</span>
                  <span className="document-count">{doc.chunk_count}</span>
                </label>
                <button className="doc-delete" onClick={() => void removeDocument(doc)} aria-label={`Delete ${doc.filename}`} title="Delete">×</button>
              </div>
            ))}
          </div>
        </div>
        <button onClick={() => void newChat()} className="new-chat-button">＋ New conversation</button>
        <div className="sidebar-footer">Answers are grounded in your uploaded material.<br />Your session lives in this browser only.</div>
      </aside>

      <main className="chat-pane">
        <header className="chat-header">
          <div><span className="eyebrow">Study workspace</span><h1>Ask your notes</h1><p>{selectedNames.length ? `Searching ${selectedNames.join(", ")}` : "Search across your entire knowledge base"}</p></div>
        </header>

        <div className="conversation">
          {messages.length === 0 && !loading && (
            <div className="welcome"><div className="welcome-icon">✦</div><h2>What would you like to learn?</h2><p>Upload a PDF, lecture recording or video, then ask a precise question or request a summary.</p></div>
          )}
          {messages.map((message, i) => message.role === "user" ? (
            <div key={i} className="user-row"><div className="user-bubble">{message.content}</div></div>
          ) : (
            <div key={i} className="answer-card">
              <div className="answer-heading">
                <span className="answer-label">Answer</span>
                {message.abstained && <span className="confidence warning">Not in your material</span>}
              </div>
              <MessageBubble content={message.content} role="assistant" />
              {message.error && <p className="upload-status error">The request failed.</p>}
              {message.sources && message.sources.length > 0 && (
                <div className="source-grid">
                  {message.sources.map((source, index) => <SourceCard key={source.chunk_id} source={source} index={index + 1} />)}
                </div>
              )}
            </div>
          ))}
          {loading && <div className="loading-card"><span className="loading-dot" /><span>Reading your material…</span></div>}
          <div ref={endRef} />
        </div>

        <form className="composer" onSubmit={ask}><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(); } }} placeholder="Ask anything about your material…" rows={1} disabled={loading} /><button type="submit" disabled={loading || !query.trim()} aria-label="Ask question">{loading ? "…" : "↑"}</button><span className="composer-hint">Enter to ask · Shift + Enter for new line</span></form>
      </main>
    </div>
  );
}
