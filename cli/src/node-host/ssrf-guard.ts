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

function isBlockedIpv6(ip: string): boolean {
  const lower = ip.toLowerCase().replace(/^\[/, '').replace(/\]$/, '');
  if (lower === '::1' || lower === '::') return true; // loopback / unspecified
  const mapped = lower.match(/^::ffff:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$/);
  if (mapped) return isBlockedIpv4(mapped[1]); // IPv4-mapped ::ffff:a.b.c.d
  if (/^f[cd]/.test(lower)) return true; // fc00::/7 unique-local
  if (/^fe[89ab]/.test(lower)) return true; // fe80::/10 link-local
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
      ...(init as any),
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
        url = new URL(loc, url).toString();
        continue;
      }
    }
    // undici's Response is the spec Response at runtime; the type is nominally distinct.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return resp as any as Response;
  }
  throw new Error(`too many redirects (> ${maxRedirects})`);
}
