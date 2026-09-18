import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ChatResponse,
  type Citation,
  type Confidence,
  type DocumentSummary,
  chatStream,
  deleteDocument,
  listDocuments,
  uploadDocument,
} from "./api";
import { useHealth } from "./useHealth";

interface Turn {
  question: string;
  response: ChatResponse;
}

// Chat UI: an upload box, the ingested-document list, and a message thread.
// Each answer shows a confidence badge (green grounded / yellow low-confidence /
// blue web-fallback), an optional warning line, and verified citation cards —
// web citations link out to their live source (Design.md §4).
export default function App(): JSX.Element {
  const health = useHealth();
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Live streaming state for the in-flight turn: a status line, then the
  // verified answer text as it arrives (Design.md §1.2 streaming contract).
  const [pending, setPending] = useState<{ question: string; status: string; text: string } | null>(
    null,
  );

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
    setQuestion("");
    setPending({ question: trimmed, status: "grounding", text: "" });
    try {
      const response = await chatStream(trimmed, sessionId, {
        onStatus: (state) =>
          setPending((prior) => (prior ? { ...prior, status: state } : prior)),
        onToken: (text) =>
          setPending((prior) => (prior ? { ...prior, text: prior.text + text } : prior)),
      });
      setSessionId(response.session_id);
      setTurns((prior) => [...prior, { question: trimmed, response }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "chat failed");
      setQuestion(trimmed); // restore the unsent question so it isn't lost.
    } finally {
      setPending(null);
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

        <Thread turns={turns} pending={pending} />

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

function Thread({
  turns,
  pending,
}: {
  turns: Turn[];
  pending: { question: string; status: string; text: string } | null;
}): JSX.Element {
  if (turns.length === 0 && !pending) {
    return <p className="text-sm text-slate-500">Ask a question to get a grounded answer.</p>;
  }
  return (
    <div className="space-y-5">
      {turns.map((turn, index) => (
        <div key={index} className="space-y-2">
          <p className="text-sm font-medium text-slate-200">{turn.question}</p>
          <div className="flex flex-wrap items-center gap-2">
            <ConfidenceBadge
              confidence={turn.response.confidence}
              insufficientEvidence={turn.response.insufficient_evidence}
            />
          </div>
          <p className="whitespace-pre-wrap text-sm text-slate-300">{turn.response.answer}</p>
          {turn.response.warning && (
            <p className="text-xs text-amber-400">⚠ {turn.response.warning}</p>
          )}
          {turn.response.insufficient_evidence && (
            <p className="text-xs text-slate-400">No supporting evidence in the documents.</p>
          )}
          <div className="space-y-1">
            {turn.response.citations.map((citation) => (
              <CitationCard key={citation.id} citation={citation} />
            ))}
          </div>
        </div>
      ))}
      {pending && <PendingTurn pending={pending} />}
    </div>
  );
}

// The in-flight turn: the question, a status pill while the grounding pipeline
// runs, then the verified answer text as tokens stream in. No confidence badge
// yet — that lands with the `done` event once the answer is a finished Turn.
function PendingTurn({
  pending,
}: {
  pending: { question: string; status: string; text: string };
}): JSX.Element {
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium text-slate-200">{pending.question}</p>
      {pending.text ? (
        <p className="whitespace-pre-wrap text-sm text-slate-300">
          {pending.text}
          <span className="ml-0.5 inline-block h-3 w-1 animate-pulse bg-slate-500 align-middle" />
        </p>
      ) : (
        <p className="text-xs text-slate-500">{labelForStatus(pending.status)}</p>
      )}
    </div>
  );
}

function labelForStatus(status: string): string {
  return status === "grounding" ? "Grounding — verifying citations…" : `${status}…`;
}

// GREEN grounded / YELLOW low-confidence / BLUE web-fallback (Design.md §4).
// A refusal (no evidence) gets a neutral slate badge, not a confidence claim.
function ConfidenceBadge({
  confidence,
  insufficientEvidence,
}: {
  confidence: Confidence | null | undefined;
  insufficientEvidence: boolean;
}): JSX.Element {
  if (insufficientEvidence || !confidence) {
    return <Badge className="border-slate-600 bg-slate-800 text-slate-300" label="No answer" />;
  }
  const pct = `${Math.round(confidence.score * 100)}%`;
  if (confidence.level === "web") {
    return (
      <Badge
        className="border-sky-700 bg-sky-950/60 text-sky-300"
        label={`Web sources · ${pct}`}
      />
    );
  }
  if (confidence.level === "low") {
    return (
      <Badge
        className="border-amber-700 bg-amber-950/60 text-amber-300"
        label={`Low confidence · ${pct}`}
      />
    );
  }
  return (
    <Badge
      className="border-emerald-700 bg-emerald-950/60 text-emerald-300"
      label={`Grounded · ${pct}`}
    />
  );
}

function Badge({ className, label }: { className: string; label: string }): JSX.Element {
  return (
    <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${className}`}>
      {label}
    </span>
  );
}

function CitationCard({ citation }: { citation: Citation }): JSX.Element {
  const where = citation.page === null ? citation.doc : `${citation.doc}, p.${citation.page}`;
  return (
    <div className="rounded border border-slate-800 bg-slate-900/60 px-3 py-2 text-xs">
      <div className="flex items-center justify-between text-slate-400">
        <span>
          [{citation.id}]{" "}
          {citation.url ? (
            <a
              href={citation.url}
              target="_blank"
              rel="noreferrer"
              className="text-sky-400 hover:underline"
            >
              {where} ↗
            </a>
          ) : (
            where
          )}
        </span>
        <span>score {citation.score.toFixed(2)}</span>
      </div>
      <p className="mt-1 line-clamp-3 text-slate-400">{citation.quote}</p>
    </div>
  );
}
