/**
 * What counts as "this computer or its network" for E5 (design 8: a fresh load of an app makes ZERO requests to loopback or LAN
 * ranges), and the recorder that holds a page to it.
 *
 * The harness's own recorder (harness.mjs `newContext`) refuses the product's own ports at loopback and stops there: right for
 * keeping a test off the owner's desktop, too narrow for the claim of E5, which is about ANY address a public page has no
 * business with: the loopback range, the private ranges, link-local, the carrier-grade range Tailscale uses, names that are
 * only ever local, and the IPv6 forms that carry an IPv4 address. `watchLocal` records every request of a browser context that goes to
 * one of them (E5 shows it sees a page's and a dedicated worker's; Playwright reports a service worker's on the context in Chromium too,
 * which no case here demonstrates), and REFUSES it before it leaves, so a test that looks for a probe cannot reach anything.
 *
 * The harness's own hosts (`agent.web.localhost:PORT` ...) resolve to loopback but are the sites under test, not probes: they
 * are excluded by name, and only those.
 *
 * The list is this file's OWN, not the apps' (shared/capabilities/local.ts, which the flow editor's guard on media addresses uses): a
 * recorder that took its list from the code it holds to account would see only what that code already knows, and a range the guard left
 * out would pass unseen. Written another way (a table of CIDR blocks over 128-bit numbers, IPv4 as IPv4-mapped), and made to agree with
 * the apps' by tests/unit/local-ranges.test.mjs, so a gap in either shows there as a difference.
 */
import net from 'node:net';

const MAPPED = 0xffffn << 32n;
const bits32 = 0xffffffffn;

/** An IPv4 dotted quad as an IPv4-mapped 128-bit number. */
const fromV4 = (text) => MAPPED | text.split('.').reduce((n, part) => (n << 8n) | BigInt(part), 0n);

/** An IPv6 literal as a 128-bit number. The URL parser puts every literal in one form, so what is one is its call and not this file's. */
function fromV6(text) {
  const normal = new URL(`http://[${text}]/`).hostname.slice(1, -1);
  const [head, tail = ''] = normal.split('::');
  const side = (part) => (part === '' ? [] : part.split(':'));
  const groups = [...side(head), ...new Array(8 - side(head).length - side(tail).length).fill('0'), ...side(tail)];
  return groups.reduce((n, group) => (n << 16n) | BigInt(parseInt(group, 16)), 0n);
}

/** [block, ...]: the blocks of this computer and its network. IPv4 blocks are in IPv4 form, IPv6 blocks in IPv6 form. */
const BLOCKS = [
  '0.0.0.0/8', // "this network"
  '10.0.0.0/8',
  '100.64.0.0/10', // carrier-grade NAT: Tailscale
  '127.0.0.0/8',
  '169.254.0.0/16', // link-local, a cloud's metadata address
  '172.16.0.0/12',
  '192.168.0.0/16',
  '::/128',
  '::1/128',
  'fc00::/7', // unique local
  'fe80::/10', // link-local
  '64:ff9b:1::/48', // the local-use NAT64 prefix: a network's own
].map((text) => {
  const [base, length] = text.split('/');
  const v4 = !base.includes(':');
  const prefix = BigInt(Number(length) + (v4 ? 96 : 0));
  const shift = 128n - prefix;
  return { shift, start: (v4 ? fromV4(base) : fromV6(base)) >> shift };
});

const inBlocks = (address) => BLOCKS.some(({ shift, start }) => address >> shift === start);

/** The IPv4 addresses (as mapped numbers) that an IPv6 address carries: NAT64, 6to4 and Teredo. */
function carried(address) {
  const found = [];
  if (address >> 32n === 0x0064ff9bn << 64n) found.push(MAPPED | (address & bits32)); // 64:ff9b::/96
  if (address >> 112n === 0x2002n) found.push(MAPPED | ((address >> 80n) & bits32)); // 2002::/16, the address is bits 16 to 47
  if (address >> 96n === 0x20010000n) found.push(MAPPED | (~address & bits32)); // 2001::/32, the client's address kept inverted
  return found;
}

const NAMES = /(^|\.)(localhost|local|internal|lan)$|\.home\.arpa$/;

/** Whether `hostname` (as a URL gives it: `127.0.0.1`, `[::1]`, `printer.local`) is this computer or its network. */
export function isLocalHostname(hostname) {
  let host = String(hostname).toLowerCase();
  if (host.endsWith('.')) host = host.slice(0, -1);
  if (host === '') return false;
  if (NAMES.test(host)) return true;
  const bare = host.startsWith('[') && host.endsWith(']') ? host.slice(1, -1) : host;
  if (net.isIPv6(bare)) {
    const address = fromV6(bare);
    return inBlocks(address) || carried(address).some(inBlocks);
  }
  if (net.isIPv4(bare)) return inBlocks(fromV4(bare));
  // One label, no dots: resolved by this network's own search domain, hosts file or mDNS.
  return !host.includes('.') && !host.includes(':');
}

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
