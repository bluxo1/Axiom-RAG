import { useHealth } from "./useHealth";
import { API_BASE_URL } from "./config";

// Phase 0 shell only: a wordmark and a live backend status line. The chat UI
// (upload box + message thread) lands in Phase 1; citation cards and the
// confidence badge come in Phases 2-3.
export default function App(): JSX.Element {
  const health = useHealth();

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-6">
      <div className="w-full max-w-md space-y-6 text-center">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Axiom</h1>
          <p className="mt-1 text-sm text-slate-400">Start from what you can prove.</p>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900 p-4 text-left">
          <p className="text-xs uppercase tracking-wide text-slate-500">Backend</p>
          <p className="mt-1 break-all text-xs text-slate-500">{API_BASE_URL}/health</p>
          <div className="mt-3">
            <HealthLine health={health} />
          </div>
        </div>
      </div>
    </main>
  );
}

function HealthLine({ health }: { health: ReturnType<typeof useHealth> }): JSX.Element {
  switch (health.status) {
    case "loading":
      return <p className="text-sm text-slate-400">Checking…</p>;
    case "ok":
      return (
        <p className="text-sm text-emerald-400">
          ● {health.service} {health.version} ({health.environment})
        </p>
      );
    case "error":
      return <p className="text-sm text-rose-400">● unreachable — {health.message}</p>;
  }
}
