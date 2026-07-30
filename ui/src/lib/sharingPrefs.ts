/**
 * Sharing/backend preferences — small localStorage-backed wrapper used
 * by the Settings panel toggle + the backend dispatcher.
 *
 * Three things live here:
 *   1. The boolean "is backend sharing enabled?" — independent of the
 *      build-time `VITE_API_BASE`. The env var is the *default*; the
 *      user can flip it off (forcing standalone mode) or on (if a
 *      backend URL is configured) at any time from Settings.
 *   2. The per-flow password used for AES-GCM encryption (see
 *      flowCrypto.ts). Stored under `oaiy.flowPassword.<hash_edit>` so
 *      the same browser remembers passwords across reloads without
 *      mixing them between flows.
 *   3. The "auto-approve runs from this hash" boolean — when on, the
 *      browser dispatcher skips the per-run approval toast. Off by
 *      default; users have to explicitly opt in.
 */

const TOGGLE_KEY      = 'oaiy.backendEnabled';
const PASSWORD_PREFIX = 'oaiy.flowPassword.';
const AUTOAPPROVE_PREFIX = 'oaiy.autoApproveRuns.';

/** The compile-time backend base URL. Empty string = no backend in this build. */
const BUILD_API_BASE: string = (
  (import.meta as unknown as { env?: Record<string, string | undefined> }).env
    ?.VITE_API_BASE ?? ''
).replace(/\/+$/, '');

export function backendBaseUrl(): string {
  return BUILD_API_BASE;
}

/**
 * Whether the user wants backend sharing on right now.
 *
 * Resolution: explicit user choice (localStorage) wins; default is ON
 * when a URL is configured at build time, OFF otherwise. So a build
 * without `VITE_API_BASE` runs fully standalone unless the user
 * actively flips the toggle (which is rejected by the UI because
 * there's nothing to talk to).
 */
export function isSharingEnabled(): boolean {
  try {
    const stored = localStorage.getItem(TOGGLE_KEY);
    if (stored === 'true') return true;
    if (stored === 'false') return false;
  } catch { /* ignore */ }
  return BUILD_API_BASE !== '';
}

export function setSharingEnabled(enabled: boolean): void {
  try {
    localStorage.setItem(TOGGLE_KEY, enabled ? 'true' : 'false');
    // Tell any open subscribers (e.g. the dispatcher hook) so they can
    // start/stop without a page reload.
    window.dispatchEvent(new CustomEvent('oaiy:sharing-changed', { detail: { enabled } }));
  } catch { /* ignore */ }
}

export function onSharingChanged(handler: (enabled: boolean) => void): () => void {
  const wrapped = (e: Event) => {
    const enabled = (e as CustomEvent<{ enabled: boolean }>).detail?.enabled ?? false;
    handler(enabled);
  };
  window.addEventListener('oaiy:sharing-changed', wrapped);
  return () => window.removeEventListener('oaiy:sharing-changed', wrapped);
}

// ---------------------------------------------------------------------------
// Per-flow password (used for AES-GCM encryption — see flowCrypto.ts)
// ---------------------------------------------------------------------------
//
// SECURITY — owner side vs recipient side (deliberately different policies):
//
//   * RECIPIENT side (openSharedFlow.ts) holds a shared-link password in
//     memory for the session ONLY, never localStorage. A recipient is just
//     viewing someone else's link on a machine that isn't theirs; persisting
//     it would leave a stranger's secret on disk for any later same-origin XSS.
//
//   * OWNER side (here) DOES persist, because it is load-bearing for
//     *unattended remote dispatch*: the owner's own browser must be able to
//     decrypt incoming run inputs / encrypt results after a page reload
//     without re-prompting, otherwise the "let an AI trigger runs while you're
//     away" feature breaks on every refresh. This is the owner's own machine
//     and their own flow, so the trust model differs. The residual risk is
//     that a same-origin XSS on the owner's machine could read it — acceptable
//     for an unattended-automation opt-in, but documented here so the two
//     modules don't read as contradictory.

/**
 * Look up the password the user set for this flow's edit hash. Returns
 * null if no password (i.e. flow is stored plaintext).
 */
export function getFlowPassword(hashEdit: string): string | null {
  try { return localStorage.getItem(PASSWORD_PREFIX + hashEdit); }
  catch { return null; }
}

export function setFlowPassword(hashEdit: string, password: string | null): void {
  try {
    if (password === null || password === '') {
      localStorage.removeItem(PASSWORD_PREFIX + hashEdit);
    } else {
      localStorage.setItem(PASSWORD_PREFIX + hashEdit, password);
    }
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Per-flow auto-approve runs toggle
// ---------------------------------------------------------------------------

export function isAutoApproveRuns(hashEdit: string): boolean {
  try { return localStorage.getItem(AUTOAPPROVE_PREFIX + hashEdit) === 'true'; }
  catch { return false; }
}

export function setAutoApproveRuns(hashEdit: string, on: boolean): void {
  try {
    if (on) localStorage.setItem(AUTOAPPROVE_PREFIX + hashEdit, 'true');
    else localStorage.removeItem(AUTOAPPROVE_PREFIX + hashEdit);
  } catch { /* ignore */ }
}
