import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = "http://127.0.0.1:8200";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      "/api": backend,
      "/metrics": backend,
      "/ws": { target: backend.replace("http", "ws"), ws: true },
    },
  },
});
