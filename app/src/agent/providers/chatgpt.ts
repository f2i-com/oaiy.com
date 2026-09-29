/**
 * ChatGPT through OAIY Desktop's Codex connector (`/api/ai/providers/
 * openai-codex-agent/v1`): signed out, it answers 428 with the code
 * `codex_not_authenticated`, before a reply or, rarely, inside a stream that
 * had begun. Either way the person is told where to sign in, and nothing
 * else is tried in its place.
 */

/** What the person is told when OAIY is not signed in to ChatGPT. */
export const CHATGPT_SIGN_IN =
  'The Agent runs on ChatGPT, and OAIY is not signed in to ChatGPT. Sign in to ChatGPT in OAIY (the dashboard\'s Settings → Agent, or its setup wizard), then ask again.';

/** The error code OAIY's Codex connector gives when it is signed out. */
export const NOT_SIGNED_IN = 'codex_not_authenticated';

/** Whether a failure's body (or a stream's error code) says OAIY is not signed in to ChatGPT. */
export function signInNeeded(said: string): boolean {
  return said.includes(NOT_SIGNED_IN);
}
