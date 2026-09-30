/**
 * Build-target detection.
 *
 * The web build loads `src/tauri-shim/core.ts`, which installs
 * `window.__OAIY_WEB_SHIM__ = true` (and shims `invoke` so unmapped native
 * commands resolve to null).
 *
 * That is NOT a sign of a browser tab: OAIY's own window shows this same
 * build (platform/desktop serves platform/ui/dist to it), so the flag is set
 * there too. To gate something on where the editor is, ask what it can do
 * here (lib/caps.ts, shared/capabilities): a tab is told from OAIY's window
 * by the desktop the window is given, not by the shim. Nothing calls this
 * function for that reason: gating the Packages section on it would have
 * taken the section out of OAIY's window.
 */
export function isWebBuild(): boolean {
  return (
    typeof window !== 'undefined' &&
    (window as Window & { __OAIY_WEB_SHIM__?: boolean }).__OAIY_WEB_SHIM__ === true
  );
}
