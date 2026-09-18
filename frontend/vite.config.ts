import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies `/api` to the backend so the browser talks to one
// origin and CORS stays out of the local loop. The proxy target is the backend
// container/host; in the browser bundle the base URL comes from
// `VITE_API_BASE_URL` (see `src/config.ts`).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
