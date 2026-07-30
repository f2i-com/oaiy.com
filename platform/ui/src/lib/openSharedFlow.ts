/**
 * Open a shared flow from a URL query parameter (`?flow=<hash>`).
 *
 * Three states the user can land in:
 *   1. No `?flow` param → null returned, normal app boot proceeds.
 *   2. Plaintext flow → fetched + returned, no prompt.
 *   3. Encrypted flow → asks for a password (via `promptPassword`
 *      callback, default `window.prompt`). On success, remembers the
 *      password under sharingPrefs so future loads of the same link in
 *      the same browser skip the prompt.
 *
 * The caller decides what to do with the returned `FlowSnapshot` —
 * typically merge it into the in-browser project state and switch
 * focus to its first flow.
 */

import {
  readFlow,
  PasswordRequiredError,
  type FlowSnapshot,
  type ReadFlow,
} from './backendDispatcher';
import { decryptEnvelope } from './flowCrypto';

// SECURITY: shared-link passwords are held in memory for this session only —
// never persisted to localStorage. Plaintext-at-rest under a public, guessable
// key would let any later same-origin XSS read every saved flow password, so we
// trade the cross-reload convenience for not storing the secret. Re-prompts
// after a page reload. (Owner-side keys for unattended run dispatch live
// elsewhere and are unaffected.)
const sessionFlowPasswords = new Map<string, string>();

// SECURITY: a shared flow is UNTRUSTED. Its `settings` blob must not be allowed to
// silently override where the RECIPIENT's AI/HTTP traffic goes, or to disable their
// local-network protections — otherwise a malicious link redirects the victim's model
// calls (prompts + API keys) to an attacker, or turns off their SSRF guard. We strip
// these keys on import so the flow runs against the recipient's own environment; the
// flow's nodes fall back to the recipient's endpoints/defaults. Non-routing defaults
// (provider/model names, theme, etc.) are harmless and kept.
const UNSAFE_IMPORTED_SETTING_KEYS = [
  'defaultAIEndpoint',
  'ollamaEndpoint',
  'lmstudioEndpoint',
  'defaultImageEndpoint',
  'defaultVideoEndpoint',
  'allowAllLocalNetwork',
  'localNetworkWhitelist',
] as const;

function sanitizeUntrustedFlowSettings(flow: FlowSnapshot): FlowSnapshot {
  const settings = (flow as { settings?: unknown }).settings;
  if (!settings || typeof settings !== 'object') return flow;
  const cleaned: Record<string, unknown> = { ...(settings as Record<string, unknown>) };
  const stripped: string[] = [];
  for (const k of UNSAFE_IMPORTED_SETTING_KEYS) {
    if (k in cleaned) {
      delete cleaned[k];
      stripped.push(k);
    }
  }
  if (stripped.length === 0) return flow;
  console.warn(
    `[openSharedFlow] dropped endpoint/network overrides from a shared flow's settings ` +
      `(this flow runs against your own environment): ${stripped.join(', ')}`,
  );
  return { ...(flow as object), settings: cleaned } as FlowSnapshot;
}

export interface OpenSharedFlowResult {
  /** The hash from the URL (view or edit). */
  hash: string;
  /** True when the URL hash matched the edit hash on the backend. */
  isEditable: boolean;
  /** The decrypted (or plaintext) flow snapshot. */
  flow: FlowSnapshot;
  /** Server-side metadata — title, created/updated timestamps, etc. */
  meta: ReadFlow;
}

export interface OpenSharedFlowOptions {
  /**
   * Optional async password prompt — receives the previous attempt's
   * failure message (or null on first attempt) and returns either the
   * entered password or null to cancel. Default uses `window.prompt`.
   */
  promptPassword?: (lastError: string | null) => Promise<string | null>;
}

/** Read `?flow=<hash>` off the current URL. Returns null when absent. */
export function shareHashFromUrl(url: URL = new URL(window.location.href)): string | null {
  const hash = url.searchParams.get('flow');
  return hash && hash.length > 0 ? hash : null;
}

// Non-optional concrete handle for the inline `??` fallback. The
// option-type version (OpenSharedFlowOptions['promptPassword']) widens
// to undefined, which TS can't narrow inside the function body.
const defaultPromptPassword = async (lastError: string | null): Promise<string | null> => {
  const msg = lastError
    ? `Decryption failed (${lastError}). Try again — or cancel to abandon the load.`
    : 'This flow is encrypted. Enter the password to open it.';
  // window.prompt returns null on cancel. Trim incidental whitespace.
  const result = window.prompt(msg, '');
  return result === null ? null : result.trim();
};

/**
 * Resolve the shared-link hash on the URL into a flow snapshot. Returns
 * null when there's no `?flow` param (normal cold boot). Throws when
 * the user cancels the password prompt.
 */
export async function openSharedFlowFromUrl(
  opts: OpenSharedFlowOptions = {},
): Promise<OpenSharedFlowResult | null> {
  const hash = shareHashFromUrl();
  if (!hash) return null;

  // Both branches return Promise<string | null>. DEFAULT_PROMPT's type
  // is widened by the OpenSharedFlowOptions optional, so we extract a
  // concrete non-optional handle for the rest of the function.
  const promptPassword: (lastError: string | null) => Promise<string | null> =
    opts.promptPassword ?? defaultPromptPassword;
  // Remembered password from a previous successful open — try first
  // so the user doesn't get re-prompted for their own flows.
  let password: string | null = sessionFlowPasswords.get(hash) ?? null;

  try {
    // Plaintext path (or password-already-correct path).
    const meta = await readFlow(hash, password ? { password } : {});
    return {
      hash,
      isEditable: meta.hash_edit === hash,
      flow: sanitizeUntrustedFlowSettings(meta.flow_json),
      meta,
    };
  } catch (err) {
    if (!(err instanceof PasswordRequiredError)) throw err;

    // Encrypted — prompt loop until success or user cancels.
    let lastError: string | null = null;
    // Up to 5 attempts so a typo doesn't hose the user, but we don't
    // spin forever if they cancel.
    for (let i = 0; i < 5; i++) {
      const entered = await promptPassword(lastError);
      if (entered === null) {
        throw new Error('password prompt cancelled');
      }
      try {
        const plain = await decryptEnvelope(err.envelope, entered);
        // Remember in memory for this session only (never persisted to disk).
        sessionFlowPasswords.set(hash, entered);
        const safe = sanitizeUntrustedFlowSettings(plain as FlowSnapshot);
        return {
          hash,
          isEditable: err.meta.hash_edit === hash,
          flow: safe,
          meta: { ...err.meta, flow_json: safe },
        };
      } catch (decErr) {
        lastError = (decErr as Error).message ?? String(decErr);
      }
    }
    throw new Error('too many password attempts');
  }
}
