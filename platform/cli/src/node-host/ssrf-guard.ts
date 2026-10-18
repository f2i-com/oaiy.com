/**
 * SSRF guard for the Node host's outbound HTTP (`http_request`, browser navigate).
 *
 * The desktop Rust host applies DNS-pinning + private-IP blocking to its
 * `http_request`; the Node host historically went straight to `fetch()`. Under
 * `oaiy worker` the URL is attacker-controlled (it comes from queue inputs), so an
 * unguarded fetch is server-side request forgery to loopback/RFC1918/link-local
 * targets (e.g. http://169.254.169.254/ metadata, http://127.0.0.1:<admin>). This
 * module blocks those: it resolves the host, rejects any non-public address, and
 * re-validates every redirect hop.
 *
 * DNS-rebinding (TOCTOU) is closed too: ssrfSafeFetch resolves + validates the host, then PINS the
 * connection to that exact IP via an undici Agent (connect.lookup), so fetch can't re-resolve to a
 * private address between our check and the connect. The hostname still drives TLS SNI/cert + the
 * Host header — only A/AAAA resolution is pinned. (assertHostPublic, used by the browser guard where
 * Chromium does its own DNS, validates only — pinning isn't possible there.)
 */
import { fetch as undiciFetch, Agent } from 'undici';
import { lookup } from 'node:dns/promises';
import { isIP } from 'node:net';

/** True if an IP literal is loopback/private/link-local/reserved (not internet-routable). */
export function isBlockedIp(ip: string): boolean {
  const fam = isIP(ip);
  if (fam === 4) return isBlockedIpv4(ip);
  if (fam === 6) return isBlockedIpv6(ip);
  return true; // not a parseable IP → block, fail closed
}

function isBlockedIpv4(ip: string): boolean {
  const parts = ip.split('.').map((p) => Number(p));
  if (parts.length !== 4 || parts.some((p) => !Number.isInteger(p) || p < 0 || p > 255)) {
    return true;
  }
  const [a, b] = parts;
  return (
    a === 0 || // 0.0.0.0/8 "this network"
    a === 10 || // 10/8 private
    a === 127 || // loopback
    (a === 100 && b >= 64 && b <= 127) || // 100.64/10 CGNAT
    (a === 169 && b === 254) || // link-local incl. 169.254.169.254 cloud metadata
    (a === 172 && b >= 16 && b <= 31) || // 172.16/12 private
    (a === 192 && b === 0) || // 192.0.0/24 + 192.0.2/24 (special-use/test)
    (a === 192 && b === 168) || // 192.168/16 private
    (a === 198 && (b === 18 || b === 19)) || // 198.18/15 benchmarking
    a >= 224 // 224/4 multicast + 240/4 reserved + 255.255.255.255
  );
}

/** An IPv6 address as its eight 16-bit groups (a trailing dotted IPv4 part read as the last two), or null. */
function ipv6Groups(ip: string): number[] | null {
  let text = ip;
  const dotted = /(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(text);
  if (dotted) {
    const p = dotted.slice(1).map(Number);
    if (p.some((n) => n > 255)) return null;
    text = `${text.slice(0, dotted.index)}${((p[0] << 8) | p[1]).toString(16)}:${((p[2] << 8) | p[3]).toString(16)}`;
  }
  const halves = text.split('::');
  if (halves.length > 2) return null;
  const part = (h: string) => (h === '' ? [] : h.split(':').map((x) => (/^[0-9a-f]{1,4}$/.test(x) ? parseInt(x, 16) : NaN)));
  const head = part(halves[0]);
  const rest = halves.length === 2 ? part(halves[1]) : [];
  const fill = halves.length === 2 ? 8 - head.length - rest.length : 0;
  if (fill < 0) return null;
  const groups = [...head, ...new Array(fill).fill(0), ...rest];
  return groups.length === 8 && groups.every((x) => Number.isInteger(x)) ? groups : null;
}

function isBlockedIpv6(ip: string): boolean {
  const lower = ip.toLowerCase().replace(/^\[/, '').replace(/\]$/, '').replace(/%.*$/, '');
  const g = ipv6Groups(lower);
  if (!g) return true; // not one we can read → block, fail closed
  const v4 = (hi: number, lo: number) => `${hi >> 8}.${hi & 255}.${lo >> 8}.${lo & 255}`;
  const zero = (from: number, to: number) => g.slice(from, to).every((x) => x === 0);
  if (zero(0, 8) || (zero(0, 7) && g[7] === 1)) return true; // unspecified / loopback
  // An IPv4 address inside one, which reaches that address: mapped (::ffff:a.b.c.d, which the URL parser writes as
  // ::ffff:7f00:1, so the dotted form alone was not enough), the old compatible form (::a.b.c.d), NAT64
  // (64:ff9b::a.b.c.d) and 6to4 (2002:aabb:ccdd::).
  if (zero(0, 5) && g[5] === 0xffff) return isBlockedIpv4(v4(g[6], g[7]));
  if (zero(0, 6)) return isBlockedIpv4(v4(g[6], g[7]));
  if (g[0] === 0x64 && g[1] === 0xff9b && zero(2, 6)) return isBlockedIpv4(v4(g[6], g[7]));
  if (g[0] === 0x64 && g[1] === 0xff9b && g[2] === 1) return true; // 64:ff9b:1::/48 local NAT64
  if (g[0] === 0x2002) return isBlockedIpv4(v4(g[1], g[2]));
  if ((g[0] & 0xfe00) === 0xfc00) return true; // fc00::/7 unique-local
  if ((g[0] & 0xffc0) === 0xfe80) return true; // fe80::/10 link-local
  if ((g[0] & 0xffc0) === 0xfec0) return true; // fec0::/10 site-local (deprecated)
  if ((g[0] & 0xff00) === 0xff00) return true; // ff00::/8 multicast
  if (g[0] === 0x2001 && g[1] === 0x0db8) return true; // 2001:db8::/32 documentation
  if (g[0] === 0x0100 && zero(1, 4)) return true; // 100::/64 discard
  return false;
}

/**
 * Resolve + validate a host and return the public IP to PIN the connection to (so a DNS rebind
 * can't swing a later resolution to a private address). An IP literal has no DNS step to rebind →
 * returns null. Throws if the host, or ANY resolved address, is non-public.
 */
async function resolvePinnedAddress(
  host: string,
): Promise<{ address: string; family: number } | null> {
  if (isIP(host)) {
    if (isBlockedIp(host)) throw new Error(`blocked request to non-public address: ${host}`);
    return null; // literal — fetch connects straight to it, no DNS step to rebind
  }
  let addrs: Array<{ address: string; family: number }>;
  try {
    addrs = await lookup(host, { all: true });
  } catch {
    throw new Error(`could not resolve host: ${host}`);
  }
  if (addrs.length === 0) throw new Error(`could not resolve host: ${host}`);
  for (const a of addrs) {
    if (isBlockedIp(a.address)) {
      throw new Error(`blocked request to non-public address: ${host} -> ${a.address}`);
    }
  }
  // Every resolved address validated public → pin the first (any is safe).
  return { address: addrs[0].address, family: addrs[0].family };
}

/** Resolve `host` and throw if it (or any resolved address) is non-public. Validation only — used
 *  by the browser guard, where the connection can't be pinned (Chromium does its own DNS). */
export async function assertHostPublic(host: string): Promise<void> {
  await resolvePinnedAddress(host);
}

/** Parse + scheme-check a URL, then assert its host is public. Returns the parsed URL. */
export async function assertUrlAllowed(raw: string): Promise<URL> {
  let u: URL;
  try {
    u = new URL(raw);
  } catch {
    throw new Error(`invalid URL: ${raw}`);
  }
  if (u.protocol !== 'http:' && u.protocol !== 'https:') {
    throw new Error(`blocked non-http(s) URL scheme: ${u.protocol}`);
  }
  await assertHostPublic(u.hostname);
  return u;
}

/** An undici Agent that connects to the PRE-VALIDATED IP — the kernel skips a second DNS lookup
 *  that a rebind could swing private — while the hostname still drives TLS SNI/cert + the Host
 *  header. */
function pinnedAgent(pin: { address: string; family: number }): Agent {
  return new Agent({
    connect: {
      // undici's connect lookup uses the all-ADDRESSES callback form — cb(err, [{address, family}]) —
      // NOT cb(err, address, family); the latter makes undici read an undefined address and fail.
      lookup(_hostname, _opts, cb) {
        cb(null, [{ address: pin.address, family: pin.family }]);
      },
    },
  });
}

/**
 * Fetch with SSRF protection: scheme-check + resolve-and-validate every hop, PIN the connection to
 * the validated IP (closing the DNS-rebinding TOCTOU), and cap the redirect chain. The caller
 * handles the body. Uses undici's own fetch so it stays self-consistent with the pinned Agent.
 */
export async function ssrfSafeFetch(
  raw: string,
  init: RequestInit,
  maxRedirects = 10,
): Promise<Response> {
  let url = raw;
  // Because we drive the redirect loop ourselves (`redirect: 'manual'`, so each
  // hop can be IP-validated), we also inherit two responsibilities undici's own
  // follow logic handles and a naive re-issue loses:
  //
  //  1. Credentials must not cross an origin. Under `oaiy worker` the URL comes
  //     from queue inputs and `init.headers` is whatever the flow set — which for
  //     a service call is an `Authorization: Bearer <apiKeyConstant>`. Replaying
  //     that on a redirect hands the user's API key to whatever host answers
  //     `302 Location: https://attacker.example/`, and the SSRF check waves it
  //     through because an attacker-owned host is exactly "public".
  //  2. 301/302/303 must become a GET with no body; only 307/308 preserve them.
  //
  // `carried` is mutated per hop; `init` is never reused directly after hop 0.
  const carried: RequestInit = { ...init };
  // `Headers` is a Node global here; the DOM `HeadersInit` type is not in this
  // package's lib, hence the cast rather than an annotation.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const carriedHeaders = new Headers((init.headers as any) ?? undefined);
  let originOf = (v: string): string => {
    try {
      return new URL(v).origin;
    } catch {
      return '';
    }
  };
  let currentOrigin = originOf(raw);

  for (let hop = 0; hop <= maxRedirects; hop++) {
    // Scheme-check + resolve-and-validate ONCE per hop (not via assertUrlAllowed, so we don't
    // resolve twice — the second resolve is exactly the window a rebind would exploit).
    let u: URL;
    try {
      u = new URL(url);
    } catch {
      throw new Error(`invalid URL: ${url}`);
    }
    if (u.protocol !== 'http:' && u.protocol !== 'https:') {
      throw new Error(`blocked non-http(s) URL scheme: ${u.protocol}`);
    }
    const pin = await resolvePinnedAddress(u.hostname);
    const agent = pin ? pinnedAgent(pin) : undefined;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const resp = await undiciFetch(url, {
      ...(carried as any),
      headers: carriedHeaders,
      redirect: 'manual',
      ...(agent ? { dispatcher: agent } : {}),
    });
    if (resp.status >= 300 && resp.status < 400) {
      const loc = resp.headers.get('location');
      if (loc) {
        // Release this hop's connection before following the redirect.
        try {
          await resp.body?.cancel();
        } catch {
          /* ignore */
        }
        agent?.destroy().catch(() => {});
        const next = new URL(loc, url).toString();

        // Cross-origin hop: drop everything that authenticates the caller.
        const nextOrigin = originOf(next);
        if (nextOrigin !== currentOrigin) {
          carriedHeaders.delete('authorization');
          carriedHeaders.delete('cookie');
          carriedHeaders.delete('proxy-authorization');
          // and the keys services take in headers of their own (x-api-key, api-key, x-goog-api-key, xi-api-key,
          // x-auth-token, …): carried on, the key went to whatever host the redirect named
          const names: string[] = [];
          carriedHeaders.forEach((_value, name) => names.push(name));
          for (const name of names) {
            if (/(^|-)(api-?key|key|token|secret|auth|authorization|signature|session)$/i.test(name)) carriedHeaders.delete(name);
          }
          currentOrigin = nextOrigin;
        }

        // Per the Fetch spec, 301/302/303 rewrite the request to a bodyless GET.
        // 307/308 exist precisely to preserve the method and body.
        if (resp.status === 301 || resp.status === 302 || resp.status === 303) {
          carried.method = 'GET';
          carried.body = undefined;
          carriedHeaders.delete('content-type');
          carriedHeaders.delete('content-length');
        }

        url = next;
        continue;
      }
    }
    // undici's Response is the spec Response at runtime; the type is nominally distinct.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return resp as any as Response;
  }
  throw new Error(`too many redirects (> ${maxRedirects})`);
}
