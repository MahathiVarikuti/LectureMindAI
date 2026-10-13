"use client";

import { useCallback, useEffect, useState } from "react";
import { DocumentInfo, listDocuments } from "@/lib/api";
import { getSessionId } from "@/lib/session";
import ChatInterface from "@/components/ChatInterface";
import DocumentSidebar from "@/components/DocumentSidebar";
import UploadPanel from "@/components/UploadPanel";

export default function HomePage() {
  const [sessionId, setSessionId] = useState("");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    const timer = window.setTimeout(() => setSessionId(getSessionId()), 0);
    return () => window.clearTimeout(timer);
  }, []);

  const refreshDocs = useCallback(async () => {
    if (!sessionId) return;
    try {
      const docs = await listDocuments(sessionId);
      setDocuments(docs);
    } catch {
      /* backend may not be running yet */
    }
  }, [sessionId]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshDocs();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [refreshDocs]);

  const toggleDoc = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  if (!sessionId) return null;

  return (
    <div className="min-h-screen overflow-hidden text-slate-900">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <header className="mb-6 rounded-2xl border border-slate-200/80 bg-white/80 px-6 py-5 shadow-[0_10px_35px_rgba(15,23,42,0.05)] backdrop-blur-sm">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-blue-600">
                LectureMindAI
              </p>
              <h1 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                Ask questions about your lecture materials
              </h1>
            </div>
            <div className="rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-xs font-medium text-blue-700">
              Multimodal RAG • Session {sessionId.slice(0, 8)}
            </div>
          </div>
        </header>

        <main className="grid grid-cols-1 gap-6 xl:grid-cols-[360px_minmax(0,1fr)] xl:items-stretch">
          <aside className="space-y-6">
            <UploadPanel sessionId={sessionId} onUploaded={refreshDocs} />
            <DocumentSidebar
              sessionId={sessionId}
              documents={documents}
              selected={selected}
              onToggle={toggleDoc}
              onDeleted={refreshDocs}
            />
          </aside>

          <section className="min-w-0 xl:h-[calc(100vh-12rem)]">
            <ChatInterface sessionId={sessionId} selectedDocuments={selected} />
          </section>
        </main>
      </div>
    </div>
  );
}
