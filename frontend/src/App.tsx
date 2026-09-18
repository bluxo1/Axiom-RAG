import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ChatResponse,
  type Citation,
  type DocumentSummary,
  chat,
  deleteDocument,
  listDocuments,
  uploadDocument,
} from "./api";
import { useHealth } from "./useHealth";

interface Turn {
  question: string;
  response: ChatResponse;
}

// Phase 1 chat UI: an upload box, the ingested-document list, and a message
// thread. Confidence badges and citation verification arrive in Phases 2-3;
// citations here are the retrieved sources, shown as cards under each answer.
export default function App(): JSX.Element {
  const health = useHealth();
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshDocuments = useCallback(async () => {
    try {
      const { documents: docs } = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not load documents");
    }
  }, []);

  useEffect(() => {
    void refreshDocuments();
  }, [refreshDocuments]);

  const onUpload = async (file: File): Promise<void> => {
    setError(null);
    setBusy(true);
    try {
      await uploadDocument(file);
      await refreshDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "upload failed");
    } finally {
      setBusy(false);
    }
  };

  const onDelete = async (docId: string): Promise<void> => {
    try {
      await deleteDocument(docId);
      await refreshDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "delete failed");
    }
  };

  const onAsk = async (): Promise<void> => {
    const trimmed = question.trim();
    if (!trimmed || busy) {
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const response = await chat(trimmed, sessionId);
      setSessionId(response.session_id);
      setTurns((prior) => [...prior, { question: trimmed, response }]);
      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "chat failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4 flex items-baseline justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Axiom</h1>
          <p className="text-xs text-slate-400">Start from what you can prove.</p>
        </div>
        <HealthDot health={health} />
      </header>

      <main className="mx-auto max-w-3xl px-6 py-6 space-y-6">
        <DocumentsPanel
          documents={documents}
          busy={busy}
          onUpload={onUpload}
          onDelete={onDelete}
        />

        {error && (
          <p className="rounded border border-rose-800 bg-rose-950/50 px-3 py-2 text-sm text-rose-300">
            {error}
          </p>
        )}

        <Thread turns={turns} />

        <div className="flex gap-2">
          <input
            className="flex-1 rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-slate-500"
            placeholder={
              documents.length === 0 ? "Upload a document first…" : "Ask a question…"
            }
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                void onAsk();
              }
            }}
            disabled={busy}
          />
          <button
            className="rounded bg-slate-100 px-4 py-2 text-sm font-medium text-slate-900 disabled:opacity-40"
            onClick={() => void onAsk()}
            disabled={busy || !question.trim()}
          >
            Ask
          </button>
        </div>
      </main>
    </div>
  );
}

function HealthDot({ health }: { health: ReturnType<typeof useHealth> }): JSX.Element {
  const color =
    health.status === "ok"
      ? "text-emerald-400"
      : health.status === "error"
        ? "text-rose-400"
        : "text-slate-500";
  const label =
    health.status === "ok"
      ? `backend ${health.version}`
      : health.status === "error"
        ? "backend unreachable"
        : "connecting…";
  return <span className={`text-xs ${color}`}>● {label}</span>;
}

function DocumentsPanel({
  documents,
  busy,
  onUpload,
  onDelete,
}: {
  documents: DocumentSummary[];
  busy: boolean;
  onUpload: (file: File) => void;
  onDelete: (docId: string) => void;
}): JSX.Element {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-slate-300">Documents</h2>
        <button
          className="rounded border border-slate-700 px-3 py-1 text-xs text-slate-200 disabled:opacity-40"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
        >
          Upload PDF / TXT / MD
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md,.markdown"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onUpload(file);
            }
            event.target.value = "";
          }}
        />
      </div>
      {documents.length === 0 ? (
        <p className="mt-3 text-xs text-slate-500">No documents yet.</p>
      ) : (
        <ul className="mt-3 space-y-1">
          {documents.map((doc) => (
            <li
              key={doc.doc_id}
              className="flex items-center justify-between text-sm text-slate-300"
            >
              <span className="truncate">
                {doc.name}{" "}
                <span className="text-xs text-slate-500">
                  ({doc.chunks} chunks · {doc.status})
                </span>
              </span>
              <button
                className="ml-2 text-xs text-slate-500 hover:text-rose-400"
                onClick={() => onDelete(doc.doc_id)}
              >
                remove
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Thread({ turns }: { turns: Turn[] }): JSX.Element {
  if (turns.length === 0) {
    return <p className="text-sm text-slate-500">Ask a question to get a grounded answer.</p>;
  }
  return (
    <div className="space-y-5">
      {turns.map((turn, index) => (
        <div key={index} className="space-y-2">
          <p className="text-sm font-medium text-slate-200">{turn.question}</p>
          <p className="whitespace-pre-wrap text-sm text-slate-300">{turn.response.answer}</p>
          {turn.response.insufficient_evidence && (
            <p className="text-xs text-amber-400">No supporting evidence in the documents.</p>
          )}
          <div className="space-y-1">
            {turn.response.citations.map((citation) => (
              <CitationCard key={citation.id} citation={citation} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function CitationCard({ citation }: { citation: Citation }): JSX.Element {
  const where = citation.page === null ? citation.doc : `${citation.doc}, p.${citation.page}`;
  return (
    <div className="rounded border border-slate-800 bg-slate-900/60 px-3 py-2 text-xs">
      <div className="flex items-center justify-between text-slate-400">
        <span>
          [{citation.id}] {where}
        </span>
        <span>score {citation.score.toFixed(2)}</span>
      </div>
      <p className="mt-1 line-clamp-3 text-slate-400">{citation.quote}</p>
    </div>
  );
}
