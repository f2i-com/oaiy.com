/**
 * What counts as "this computer or its network" for E5 (design 8: a fresh load of an app makes ZERO requests to loopback or LAN
 * ranges), and the recorder that holds a page to it.
 *
 * The harness's own recorder (harness.mjs `newContext`) refuses the product's own ports at loopback and stops there: right for
 * keeping a test off the owner's desktop, too narrow for the claim of E5, which is about ANY address a public page has no
 * business with: the loopback range, the private ranges, link-local, the carrier-grade range Tailscale uses, and names that are
 * only ever local. `watchLocal` records every request of a browser context that goes to one of them (E5 shows it sees a page's and a
 * dedicated worker's; Playwright reports a service worker's on the context in Chromium too, which no case here demonstrates), and
 * REFUSES it before it leaves, so a test that looks for a probe cannot reach anything.
 *
 * The harness's own hosts (`agent.web.localhost:PORT` ...) resolve to loopback but are the sites under test, not probes: they
 * are excluded by name, and only those.
 */
import { loadTs } from '../support/load.mjs';

// The classifier is the apps' own (shared/capabilities/local.ts): what a test counts as a probe is what a guard in the apps refuses.
const { isLocalHostname } = await loadTs('shared/capabilities/local.ts');
export { isLocalHostname };

/**
 * Hold a browser context to that. Every request the context makes (any page or worker) is looked at once, and one to
 * this computer or its network that is not a site under test is REFUSED and recorded.
 *
 * @param {import('playwright').BrowserContext} context
 * @param {{ sites: string[], refuseAfterMs?: number, answer?: (request: { url: string, method: string }) => ({ status?: number, delay?: number, body: unknown } | null) }} options
 *   `sites`: the `host:port` of every site under test (`agent.web.localhost:PORT` ...), which are local addresses and not probes.
 *   `refuseAfterMs`: how long a request is held before it is refused (default none), for a test that looks at what a page says while
 *   its request is out. `answer`: a stand-in for what would be at that address (OAIY Desktop): given a request it returns a JSON
 *   answer (after `delay` ms, if it says so), and the request is fulfilled with it in the browser and goes no further, or null to refuse
 *   it as usual. Either way the request is recorded.
 * @returns {{ attempts: string[], details: { url: string, method: string, headers: Record<string,string>, at: number }[] }} `attempts`
 *   lists every request to a local address that was made, in order, with its method: `GET http://127.0.0.1:17972/api/health`.
 *   `details` are the same, with the headers the page set (an Authorization) and when it was made.
 */
export async function watchLocal(context, { sites, refuseAfterMs = 0, answer = null }) {
  const own = new Set(sites.map((s) => s.toLowerCase()));
  const attempts = [];
  /** The same requests with what a test may want to read: when (ms since this context was watched) and the headers the page set. */
  const details = [];
  const started = Date.now();
  const isProbe = (url) => {
    const u = new URL(url);
    if (!/^(https?|wss?):$/.test(u.protocol)) return false;
    return !own.has(u.host.toLowerCase()) && isLocalHostname(u.hostname);
  };
  // What the browser was asked for, counted before anything answers: a refused request is still one that was made.
  context.on('request', (request) => {
    if (!isProbe(request.url())) return;
    attempts.push(`${request.method()} ${request.url()}`);
    details.push({ url: request.url(), method: request.method(), headers: request.headers(), at: Date.now() - started });
  });
  await context.route(
    (url) => isProbe(url.href),
    async (route) => {
      if (refuseAfterMs > 0) await new Promise((resolve) => setTimeout(resolve, refuseAfterMs));
      const said = answer?.({ url: route.request().url(), method: route.request().method() });
      if (said) {
        // A slow answer (`delay`, in ms): what a page does while an OAIY that is on its way has not answered.
        if (said.delay > 0) await new Promise((resolve) => setTimeout(resolve, said.delay));
        // The page is on another origin: what answers it has to allow that, as OAIY Desktop does.
        await route.fulfill({ status: said.status ?? 200, contentType: 'application/json', headers: { 'access-control-allow-origin': '*' }, body: JSON.stringify(said.body) }).catch(() => {});
        return;
      }
      await route.abort('blockedbyclient').catch(() => {});
    },
  );
  return { attempts, details };
}
