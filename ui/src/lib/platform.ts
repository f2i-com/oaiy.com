/**
 * Build-target detection.
 *
 * The web build loads `src/tauri-shim/core.ts`, which installs
 * `window.__OAIY_WEB_SHIM__ = true` (and shims `invoke` so unmapped native
 * commands resolve to null). A real desktop (Tauri) build uses the actual
 * `@tauri-apps` APIs and never loads the shim, so the flag stays unset.
 *
 * Use this to gate features that need native capabilities the browser can't
 * provide — e.g. unzipping/verifying `.oaiy` packages or scanning the
 * filesystem for installable packages. Importing a `.json` flow still works
 * in the browser and should NOT be gated.
 */
export function isWebBuild(): boolean {
  return (
    typeof window !== 'undefined' &&
    (window as Window & { __OAIY_WEB_SHIM__?: boolean }).__OAIY_WEB_SHIM__ === true
  );
}
