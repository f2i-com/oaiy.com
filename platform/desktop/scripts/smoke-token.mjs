// The token that smoke-server.mjs gives the packaged server: drawn until the server's own `check` takes it.
//
// The server refuses an OAIY_SERVER_TOKEN that has a common pattern (`0000`, `1234`, a repeat, a word), and a random
// token has one now and then by chance: 43 base64url characters are refused about one time in 10,000. A release smoke
// test that took the first draw would fail the release that often for a reason that is nobody's mistake, so it draws
// again. `check` is the server's own judgement, which is the rule the release ships and not a copy of it.
import { randomBytes } from 'node:crypto';

/**
 * @param {(candidate: string) => { status: number | null, stderr?: string, error?: Error }} check
 *   asks the server whether it takes the candidate (`oaiy-server check` with OAIY_SERVER_TOKEN set to it)
 * @param {{ draws?: number, draw?: () => string }} [options]
 * @returns {string} the first candidate the server took
 */
export function drawAcceptedToken(check, { draws = 25, draw = () => randomBytes(32).toString('base64url') } = {}) {
  let said = '';
  for (let i = 0; i < draws; i++) {
    const candidate = draw();
    const r = check(candidate);
    if (r.status === 0) return candidate;
    // What the server said, never the token it said it of.
    said = (r.stderr || r.error?.message || `exit ${r.status}`).replaceAll(candidate, '<token>');
  }
  throw new Error(`the server's check refused ${draws} random tokens in a row: ${said}`);
}
