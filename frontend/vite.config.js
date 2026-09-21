import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ingest": "http://localhost:3000",
      "/chat": "http://localhost:3000",
    },
  },
});
