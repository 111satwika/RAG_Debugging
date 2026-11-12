import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the UI runs on :5173 and forwards /api calls to the FastAPI server on :8765.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8765" } },
});
