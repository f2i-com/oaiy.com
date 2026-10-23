import { configDefaults, defineConfig } from 'vitest/config';

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
    // A window that names no system, on every computer: jsdom's own puts the test computer's (`win32`, `linux`) in its
    // user agent, which the dashboard reads as Windows or Linux (platform.ts), so the tests that expect no system's
    // examples and notes failed on Windows. A test about one system stubs that system's own.
    environmentOptions: { jsdom: { userAgent: 'Mozilla/5.0 (unknown OS) AppleWebKit/537.36 (KHTML, like Gecko) jsdom' } },
    // The staging script's checks live beside it in scripts/, as plain JS: the
    // TypeScript build's `include` is `src` only, and an import of an .mjs from
    // there would need a declaration the build does not have.
    include: ['src/**/*.{test,spec}.{ts,tsx}', 'scripts/**/*.test.mjs'],
    // oaiyctl, the release check and the smoke-test token helper use node:test (oaiyctl also a real POSIX
    // shell), rather than Vitest/jsdom. npm test runs them separately, preserving both runners' failures.
    exclude: [
      ...configDefaults.exclude,
      'scripts/oaiyctl.test.mjs',
      'scripts/check-release.test.mjs',
      'scripts/smoke-token.test.mjs',
    ],
    setupFiles: ['./src/test-setup.ts'],
  },
});
