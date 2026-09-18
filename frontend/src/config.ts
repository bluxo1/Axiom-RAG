// The backend base URL is injected at build time by Vite from
// `VITE_API_BASE_URL` (documented in `.env.example`). It defaults to the local
// dev backend so a fresh clone runs without an env file.
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
