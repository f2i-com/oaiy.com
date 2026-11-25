/**
 * Network Utilities Module
 *
 * Pure utility functions for network-related operations.
 * Used for local network detection and URL parsing.
 */

/**
 * Check if a URL is a local/private network address.
 * Detects localhost variants and private IP ranges (RFC 1918).
 *
 * @param url - The URL to check
 * @returns true if the URL points to a local/private network address
 *
 * @example
 * ```typescript
 * isLocalNetworkUrl('http://localhost:3000') // true
 * isLocalNetworkUrl('http://192.168.1.100:8080') // true
 * isLocalNetworkUrl('https://example.com') // false
 * ```
 */
function isPrivateIpv4(ip: string): boolean {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(ip);
  if (!m) return false;
  const a = Number(m[1]);
  const b = Number(m[2]);
  const c = Number(m[3]);
  const d = Number(m[4]);
  if (a > 255 || b > 255 || c > 255 || d > 255) return false;
  // 127.0.0.0/8 loopback
  if (a === 127) return true;
  // 0.0.0.0/8 unspecified ("this host")
  if (a === 0) return true;
  // 10.0.0.0/8 RFC 1918
  if (a === 10) return true;
  // 172.16.0.0/12 RFC 1918
  if (a === 172 && b >= 16 && b <= 31) return true;
  // 192.168.0.0/16 RFC 1918
  if (a === 192 && b === 168) return true;
  // 169.254.0.0/16 link-local / cloud metadata
  if (a === 169 && b === 254) return true;
  // 100.64.0.0/10 shared address space (CGNAT / Tailscale, RFC 6598)
  if (a === 100 && b >= 64 && b <= 127) return true;
  return false;
}

function parseIpv6Groups(raw: string): number[] | null {
  // Strip optional zone id (%eth0 / %25eth0)
  let addr = raw.split('%')[0].toLowerCase();
  // Convert embedded dotted-quad IPv4 suffix (e.g. ::ffff:127.0.0.1) into two hex groups
  const v4Match = /:(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(addr);
  if (v4Match) {
    const o1 = Number(v4Match[1]);
    const o2 = Number(v4Match[2]);
    const o3 = Number(v4Match[3]);
    const o4 = Number(v4Match[4]);
    if (o1 > 255 || o2 > 255 || o3 > 255 || o4 > 255) return null;
    const hi = ((o1 << 8) | o2).toString(16);
    const lo = ((o3 << 8) | o4).toString(16);
    addr = `${addr.slice(0, v4Match.index + 1)}${hi}:${lo}`;
  }
  const halves = addr.split('::');
  if (halves.length > 2) return null;
  const parsePart = (part: string): number[] | null => {
    if (!part) return [];
    const tokens = part.split(':');
    const out: number[] = [];
    for (const t of tokens) {
      if (!/^[0-9a-f]{1,4}$/.test(t)) return null;
      out.push(parseInt(t, 16));
    }
    return out;
  };
  const left = parsePart(halves[0]);
  if (!left) return null;
  if (halves.length === 1) {
    return left.length === 8 ? left : null;
  }
  const right = parsePart(halves[1]);
  if (!right) return null;
  const missing = 8 - left.length - right.length;
  if (missing < 0) return null;
  return [...left, ...new Array<number>(missing).fill(0), ...right];
}

export function isLocalNetworkUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    // Strip trailing DNS root dot(s) so "localhost." and "foo.localhost." cannot bypass
    const hostname = parsed.hostname.toLowerCase().replace(/\.+$/, '');
    if (!hostname) return false;

    // localhost and *.localhost (RFC 6761)
    if (hostname === 'localhost' || hostname.endsWith('.localhost')) {
      return true;
    }

    // IPv4 local/private/CGNAT/loopback/unspecified ranges
    if (isPrivateIpv4(hostname)) {
      return true;
    }

    // IPv6 literals (URL parser keeps brackets, e.g. '[::1]', '[::ffff:7f00:1]')
    if ((hostname.startsWith('[') && hostname.endsWith(']')) || hostname.includes(':')) {
      const inner = hostname.startsWith('[') && hostname.endsWith(']')
        ? hostname.slice(1, -1)
        : hostname;
      const groups = parseIpv6Groups(inner);
      if (groups) {
        const allZeroPrefix6 =
          groups[0] === 0 &&
          groups[1] === 0 &&
          groups[2] === 0 &&
          groups[3] === 0 &&
          groups[4] === 0 &&
          groups[5] === 0;
        // :: (unspecified) and ::1 (loopback)
        if (allZeroPrefix6 && groups[6] === 0 && (groups[7] === 0 || groups[7] === 1)) {
          return true;
        }
        // fc00::/7 unique-local
        if ((groups[0] & 0xfe00) === 0xfc00) {
          return true;
        }
        // fe80::/10 link-local
        if ((groups[0] & 0xffc0) === 0xfe80) {
          return true;
        }
        // ::ffff:0:0/96 IPv4-mapped IPv6 (both dotted and hex normalized forms)
        // and deprecated ::/96 IPv4-compatible IPv6
        const isIpv4Mapped =
          groups[0] === 0 &&
          groups[1] === 0 &&
          groups[2] === 0 &&
          groups[3] === 0 &&
          groups[4] === 0 &&
          groups[5] === 0xffff;
        if (isIpv4Mapped || allZeroPrefix6) {
          const ipv4 = `${groups[6] >> 8}.${groups[6] & 0xff}.${groups[7] >> 8}.${groups[7] & 0xff}`;
          if (isPrivateIpv4(ipv4)) return true;
        }
      }
    }

    return false;
  } catch {
    return false;
  }
}

/**
 * Extract the host:port string from a URL for whitelist matching.
 * Uses default ports (80 for HTTP, 443 for HTTPS) when not specified.
 *
 * @param url - The URL to parse
 * @returns The host:port string, or empty string if parsing fails
 *
 * @example
 * ```typescript
 * getHostPort('http://localhost:3000') // 'localhost:3000'
 * getHostPort('https://example.com') // 'example.com:443'
 * getHostPort('http://192.168.1.1') // '192.168.1.1:80'
 * ```
 */
export function getHostPort(url: string): string {
  try {
    const parsed = new URL(url);
    const port = parsed.port || (parsed.protocol === 'https:' ? '443' : '80');
    const hostname = parsed.hostname.toLowerCase().replace(/\.+$/, '');
    return `${hostname}:${port}`;
  } catch {
    return '';
  }
}

/**
 * Check if a URL's host:port matches a whitelist entry.
 * Supports exact matches, host-only matches, and wildcard port matches.
 *
 * @param hostPort - The host:port string to check (e.g., "localhost:3000")
 * @param whitelistEntry - The whitelist entry to match against
 * @returns true if the entry matches
 *
 * @example
 * ```typescript
 * matchesWhitelistEntry('localhost:3000', 'localhost:3000') // true (exact)
 * matchesWhitelistEntry('localhost:3000', 'localhost') // true (host only)
 * matchesWhitelistEntry('localhost:3000', 'localhost:*') // true (wildcard port)
 * matchesWhitelistEntry('localhost:3000', 'localhost:8080') // false
 * ```
 */
export function matchesWhitelistEntry(hostPort: string, whitelistEntry: string): boolean {
  if (!whitelistEntry) return false;
  // Exact match
  if (whitelistEntry === hostPort) return true;

  // Extract host safely for both IPv4/domain ("host:port") and bracketed IPv6 ("[::1]:8080")
  let host: string;
  if (hostPort.startsWith('[')) {
    const closeBracket = hostPort.indexOf(']');
    host = closeBracket >= 0 ? hostPort.slice(0, closeBracket + 1) : hostPort;
  } else {
    host = hostPort.split(':')[0];
  }

  // Match without port (allow any port on that host)
  if (whitelistEntry === host) return true;
  if (host.startsWith('[') && host.endsWith(']') && whitelistEntry === host.slice(1, -1)) {
    return true;
  }

  // Match with wildcard port
  if (whitelistEntry.endsWith(':*')) {
    const entryHost = whitelistEntry.slice(0, -2);
    if (entryHost === host) return true;
    if (host.startsWith('[') && host.endsWith(']') && entryHost === host.slice(1, -1)) {
      return true;
    }
  }

  return false;
}

/**
 * Check if a URL is in a whitelist.
 *
 * @param url - The URL to check
 * @param whitelist - Array of whitelist entries (host, host:port, or host:*)
 * @param allowAll - If true, bypass whitelist check and allow all
 * @returns true if the URL is whitelisted
 */
export function isUrlWhitelisted(
  url: string,
  whitelist: string[],
  allowAll: boolean = false
): boolean {
  // Global override allows all
  if (allowAll) {
    return true;
  }

  const hostPort = getHostPort(url);
  if (!hostPort) return false;

  return whitelist.some(entry => matchesWhitelistEntry(hostPort, entry));
}
