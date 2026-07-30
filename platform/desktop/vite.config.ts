import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The companion's React UI is tiny — just a status panel + service list.
// Vite dev port is fixed to 1420 (Tauri default) so `tauri dev` knows
// where to find it; the bundled localhost API is on a SEPARATE port
// (17972) and is served by the Rust backend, not Vite.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
