"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  ChatMessage,
  QueryResponse,
  SourceCitation,
  getChatHistory,
  queryDocuments,
} from "@/lib/api";

interface Props {
  sessionId: string;
  selectedDocuments: Set<string>;
}

function formatCitation(src: SourceCitation): string {
  if (src.page != null) return `${src.filename}, p.${src.page}`;
  if (src.timestamp_start != null) {
    const end =
      src.timestamp_end != null
        ? `–${src.timestamp_end.toFixed(0)}s`
        : "";
    return `${src.filename}, ${src.timestamp_start.toFixed(0)}s${end}`;
  }
  if (src.timestamp != null) return `${src.filename}, ${src.timestamp.toFixed(0)}s`;
  return src.filename;
}

export default function ChatInterface({ sessionId, selectedDocuments }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messageListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getChatHistory(sessionId)
      .then(setMessages)
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    const list = messageListRef.current;
    if (!list) return;
    list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setLoading(true);
    try {
      const selected = selectedDocuments.size
        ? Array.from(selectedDocuments)
        : undefined;
      const res: QueryResponse = await queryDocuments(sessionId, q, selected);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, sources: res.sources },
      ]);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail || "Query failed";
      setError(String(msg));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full min-h-[620px] flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white/85 shadow-[0_18px_40px_rgba(15,23,42,0.08)] backdrop-blur-sm">
      <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50/80 px-4 py-3 sm:px-5">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
            Assistant
          </p>
          <h2 className="text-base font-semibold text-slate-900">Conversation</h2>
        </div>
        <div className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[10px] font-medium text-slate-600">
          {selectedDocuments.size ? `${selectedDocuments.size} documents scoped` : "All documents"}
        </div>
      </div>

      <div
        ref={messageListRef}
        className="flex-1 space-y-4 overflow-y-auto bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.06),transparent_30%)] p-4 sm:p-5"
      >
        {messages.length === 0 && (
          <div className="flex h-full min-h-[420px] items-center justify-center">
            <div className="max-w-md rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-6 py-8 text-center">
              <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-100 text-xl text-blue-700">
                ✦
              </div>
              <p className="text-lg font-semibold text-slate-800">Ready to answer your questions</p>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Upload lecture material and ask about the content. The assistant will ground every answer in your uploaded sources.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[88%] rounded-2xl px-4 py-3 shadow-sm sm:max-w-[80%] ${
                msg.role === "user"
                  ? "bg-slate-900 text-white"
                  : "border border-slate-200 bg-white text-slate-800"
              }`}
            >
              {msg.role === "assistant" ? (
                <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 text-slate-800">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              ) : (
                <p className="whitespace-pre-wrap text-sm leading-6">{msg.content}</p>
              )}

              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 border-t border-slate-200 pt-3">
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Sources
                  </p>
                  <ul className="space-y-2 text-xs text-slate-600">
                    {msg.sources.map((s, j) => (
                      <li key={j} className="rounded-xl bg-slate-50 px-2.5 py-2 ring-1 ring-slate-200">
                        <div className="font-medium text-slate-700">{formatCitation(s)}</div>
                        {s.snippet && (
                          <div className="mt-1 line-clamp-2 text-slate-500">{s.snippet.slice(0, 140)}...</div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <span className="inline-flex gap-1">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-blue-500 [animation-delay:-0.2s]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-blue-500 [animation-delay:-0.1s]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-blue-500" />
                </span>
                Thinking...
              </div>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="border-t border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
      )}

      <div className="sticky bottom-0 border-t border-slate-200 bg-slate-50/80 p-4 sm:p-5">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
            placeholder="Ask a question about your lectures..."
            disabled={loading}
            className="flex-1 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-slate-100"
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className="rounded-2xl bg-gradient-to-r from-blue-600 to-indigo-600 px-5 py-3 text-sm font-medium text-white shadow-lg shadow-blue-600/20 transition hover:brightness-105 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300 disabled:shadow-none"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
