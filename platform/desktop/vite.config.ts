import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// OAIY Desktop's React UI is tiny — just a status panel + service list.
//
// The dev port must match `devUrl` in tauri.conf.json, so it is pinned rather
// than negotiated. 17973 sits next to the Rust backend's 17972, giving the whole
// product one contiguous block.
//
// It is deliberately NOT Tauri's default 1420. That default is shared by every
// Tauri and Vite scaffold on the machine, so `tauri dev` races whatever else is
// running — and loses in a confusing way, because the failure surfaces as
// "beforeDevCommand terminated with a non-zero status code" rather than "another
// project owns your dev port". It cost an unrelated project's dev server once
// already. Same lesson as the API's move off llama.cpp's :8080.
//
// `strictPort` is load-bearing: without it Vite silently picks the next free
// port while `devUrl` keeps pointing at this one, so the Tauri window loads
// whatever *is* on 17973 — possibly another app entirely — or a blank page.
// Failing to bind is the correct outcome.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 17973,
    strictPort: true,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
