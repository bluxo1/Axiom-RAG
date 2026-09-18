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
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  insufficient_evidence: boolean;
  session_id: string;
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
