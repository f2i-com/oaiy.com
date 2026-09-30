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

/** IPv4 `a.b.c.d` as four numbers, or null. */
function ipv4(host) {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (!m) return null;
  const parts = m.slice(1).map(Number);
  return parts.every((n) => n <= 255) ? parts : null;
}

/** An IPv6 literal's first 16-bit group, or null when it is not one. IPv4-mapped forms are answered by their IPv4 address. */
function ipv6(host) {
  const bare = host.startsWith('[') && host.endsWith(']') ? host.slice(1, -1) : host;
  if (!bare.includes(':')) return null;
  const mapped = /^::ffff:(\d+\.\d+\.\d+\.\d+)$/i.exec(bare);
  if (mapped) return { v4: mapped[1] };
  const mappedHex = /^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/i.exec(bare);
  if (mappedHex) {
    const hi = parseInt(mappedHex[1], 16);
    const lo = parseInt(mappedHex[2], 16);
    return { v4: `${hi >> 8}.${hi & 255}.${lo >> 8}.${lo & 255}` };
  }
  if (bare === '::1' || bare === '::') return { first: 0, loopback: true };
  const first = parseInt(bare.split(':')[0] || '0', 16);
  return Number.isNaN(first) ? null : { first };
}

/** Whether `hostname` (as a URL gives it: `127.0.0.1`, `[::1]`, `printer.local`) is this computer or its network. */
export function isLocalHostname(hostname) {
  const host = String(hostname).toLowerCase().replace(/\.$/, '');
  if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal') || host.endsWith('.lan') || host.endsWith('.home.arpa')) return true;
  const six = ipv6(host);
  if (six) {
    if (six.v4) return isLocalHostname(six.v4);
    if (six.loopback) return true;
    // fc00::/7 (unique local), fe80::/10 (link-local)
    return (six.first & 0xfe00) === 0xfc00 || (six.first & 0xffc0) === 0xfe80;
  }
  const four = ipv4(host);
  if (!four) return false;
  const [a, b] = four;
  return (
    a === 0 || // "this network", and 0.0.0.0 which a browser sends to this computer
    a === 127 || // loopback
    a === 10 ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    (a === 169 && b === 254) || // link-local
    (a === 100 && b >= 64 && b <= 127) // carrier-grade NAT: Tailscale
  );
}

/**
 * Hold a browser context to it. Every request the context makes (any page, worker or service worker) is looked at once, and one to
 * this computer or its network that is not a site under test is REFUSED and recorded.
 *
 * @param {import('playwright').BrowserContext} context
 * @param {{ sites: string[], refuseAfterMs?: number, answer?: (request: { url: string, method: string }) => ({ status?: number, body: unknown } | null) }} options
 *   `sites`: the `host:port` of every site under test (`agent.web.localhost:PORT` ...), which are local addresses and not probes.
 *   `refuseAfterMs`: how long a request is held before it is refused (default none), for a test that looks at what a page says while
 *   its request is out. `answer`: a stand-in for what would be at that address (OAIY Desktop): given a request it returns a JSON
 *   answer, and the request is fulfilled with it in the browser and goes no further, or null to refuse it as usual. Either way the
 *   request is recorded.
 * @returns {{ attempts: string[] }} `attempts` lists every request to a local address that was made, in order, with its method:
 *   `GET http://127.0.0.1:17972/api/health`.
 */
export async function watchLocal(context, { sites, refuseAfterMs = 0, answer = null }) {
  const own = new Set(sites.map((s) => s.toLowerCase()));
  const attempts = [];
  const isProbe = (url) => {
    const u = new URL(url);
    if (!/^(https?|wss?):$/.test(u.protocol)) return false;
    return !own.has(u.host.toLowerCase()) && isLocalHostname(u.hostname);
  };
  // What the browser was asked for, counted before anything answers: a refused request is still one that was made.
  context.on('request', (request) => {
    if (isProbe(request.url())) attempts.push(`${request.method()} ${request.url()}`);
  });
  await context.route(
    (url) => isProbe(url.href),
    async (route) => {
      if (refuseAfterMs > 0) await new Promise((resolve) => setTimeout(resolve, refuseAfterMs));
      const said = answer?.({ url: route.request().url(), method: route.request().method() });
      if (said) {
        // The page is on another origin: what answers it has to allow that, as OAIY Desktop does.
        await route.fulfill({ status: said.status ?? 200, contentType: 'application/json', headers: { 'access-control-allow-origin': '*' }, body: JSON.stringify(said.body) }).catch(() => {});
        return;
      }
      await route.abort('blockedbyclient').catch(() => {});
    },
  );
  return { attempts };
}
