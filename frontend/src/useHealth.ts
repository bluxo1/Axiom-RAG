import { useEffect, useState } from "react";
import { API_BASE_URL } from "./config";

export type HealthState =
  | { status: "loading" }
  | { status: "ok"; service: string; version: string; environment: string }
  | { status: "error"; message: string };

interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
}

// Pings the backend liveness probe once on mount. This is the end-to-end proof
// the Phase 0 exit criterion asks for: a pinned-toolchain frontend build that
// can reach `/health` on the backend.
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${API_BASE_URL}/health`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`backend returned ${response.status}`);
        }
        const body = (await response.json()) as HealthResponse;
        setState({
          status: "ok",
          service: body.service,
          version: body.version,
          environment: body.environment,
        });
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        const message = error instanceof Error ? error.message : "unknown error";
        setState({ status: "error", message });
      });

    return () => controller.abort();
  }, []);

  return state;
}
