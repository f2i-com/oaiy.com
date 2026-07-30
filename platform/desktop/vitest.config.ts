import { defineConfig } from 'vitest/config';

/**
 * Component tests for the OAIY Desktop UI.
 *
 * Deliberately separate from `vite.config.ts` (which vitest would otherwise pick
 * up): that file pins the dev-server port for `tauri dev` and loads the React
 * plugin, neither of which a test run wants. Vitest prefers this file
 * automatically, and esbuild reads `jsx: "react-jsx"` from tsconfig, so TSX
 * compiles with no plugin chain.
 *
 * `jsdom` by default because every test here mounts a component — the sibling
 * FormLogic web app defaults to `node` and opts in per file, but this app has no
 * non-DOM tests to keep fast.
 */
export default defineConfig({
  test: {
    environment: 'jsdom',
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    setupFiles: ['./src/test-setup.ts'],
  },
});
