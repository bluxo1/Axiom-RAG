// Typed client for the Axiom API (Design.md §1). One place that knows the wire
// shapes and unwraps the structured error envelope into a thrown Error.
import { API_BASE_URL } from "./config";

export interface DocumentSummary {
  doc_id: string;
  name: string;
  chunks: number;
  status: string;
}

export interface Citation {
  id: string;
  chunk_id: string;
  doc: string;
  page: number | null;
  quote: string;
  score: number;
  source: string;
  url?: string | null;
}

export interface Confidence {
  score: number;
  level: "high" | "low" | "web";
  breakdown: { retrieval: number; faithfulness: number; coverage: number };
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  insufficient_evidence: boolean;
  session_id: string;
  confidence?: Confidence | null;
  flagged?: boolean;
  fallback_used?: boolean;
  warning?: string | null;
}

interface ErrorEnvelope {
  error?: { code?: string; message?: string };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    let message = `request failed (${response.status})`;
    try {
      const body = (await response.json()) as ErrorEnvelope;
      if (body.error?.message) {
        message = body.error.message;
      }
    } catch {
      // Non-JSON error body: keep the status-based message.
    }
    throw new Error(message);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function listDocuments(): Promise<{ documents: DocumentSummary[] }> {
  return request("/documents");
}

export function uploadDocument(file: File): Promise<DocumentSummary> {
  const form = new FormData();
  form.append("file", file);
  return request("/documents", { method: "POST", body: form });
}

export function deleteDocument(docId: string): Promise<void> {
  return request(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
}

export function chat(question: string, sessionId: string | null): Promise<ChatResponse> {
  return request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
}

export interface StreamHandlers {
  onStatus?: (state: string) => void;
  onToken?: (text: string) => void;
}

// Consume POST /chat/stream (SSE). The status event arrives first, then the
// verified answer token-by-token, then a `done` event with the full payload —
// which is what this resolves with. Tokens are already verified server-side
// (the grounding gate runs before any token is sent), so callers can render
// them live without risking an unverified claim on screen.
export async function chatStream(
  question: string,
  sessionId: string | null,
  handlers: StreamHandlers = {},
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  if (!response.ok || !response.body) {
    let message = `request failed (${response.status})`;
    try {
      const body = (await response.json()) as ErrorEnvelope;
      if (body.error?.message) {
        message = body.error.message;
      }
    } catch {
      // Non-JSON error body: keep the status-based message.
    }
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done: ChatResponse | null = null;

  for (;;) {
    const { value, done: finished } = await reader.read();
    if (finished) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    // Frames are separated by a blank line; process every complete one.
    let split: number;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const parsed = parseFrame(frame);
      if (!parsed) {
        continue;
      }
      if (parsed.event === "status") {
        handlers.onStatus?.(String((parsed.data as { state?: string }).state ?? ""));
      } else if (parsed.event === "token") {
        handlers.onToken?.(String((parsed.data as { text?: string }).text ?? ""));
      } else if (parsed.event === "done") {
        done = parsed.data as ChatResponse;
      } else if (parsed.event === "error") {
        const err = parsed.data as { message?: string };
        throw new Error(err.message ?? "stream failed");
      }
    }
  }

  if (done === null) {
    throw new Error("stream ended before a complete answer");
  }
  return done;
}

function parseFrame(frame: string): { event: string; data: unknown } | null {
  let event = "";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event: ")) {
      event = line.slice("event: ".length);
    } else if (line.startsWith("data: ")) {
      data = line.slice("data: ".length);
    }
  }
  if (!event || !data) {
    return null;
  }
  return { event, data: JSON.parse(data) };
}
