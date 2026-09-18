// The backend base URL is injected at build time by Vite from
// `VITE_API_BASE_URL` (documented in `.env.example`). It defaults to the local
// dev backend so a fresh clone runs without an env file.
// Includes the version prefix so a keyless clone reaches the v1 API directly.
// Matches VITE_API_BASE_URL in .env.example.
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
