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
export function isLocalNetworkUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.toLowerCase();

    // Check for localhost variants
    // Note: URL parser preserves brackets for IPv6, so [::1] stays as [::1]
    if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1' || hostname === '[::1]') {
      return true;
    }

    // 0.0.0.0 (unspecified IPv4 — routes to localhost on some OSes)
    if (hostname === '0.0.0.0') {
      return true;
    }
    // IPv6 literals: the URL parser keeps the brackets, e.g. '[fd00::1]'. Match the
    // genuinely-local ranges the Rust http_request enforcer also classes as private,
    // so a local IPv6 service triggers the consent prompt + allow_private_networks flag
    // (the Rust layer stays the hard boundary with DNS pinning + fail-closed default).
    if (hostname.startsWith('[') && hostname.endsWith(']')) {
      const v6 = hostname.slice(1, -1); // already lower-cased
      if (v6 === '::1' || v6 === '::') return true; // loopback / unspecified
      if (/^f[cd]/.test(v6)) return true; // fc00::/7 unique-local
      if (/^fe[89ab]/.test(v6)) return true; // fe80::/10 link-local
    }

    // Check for private IP ranges
    // 10.0.0.0/8
    if (/^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(hostname)) {
      return true;
    }
    // 172.16.0.0/12
    if (/^172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}$/.test(hostname)) {
      return true;
    }
    // 192.168.0.0/16
    if (/^192\.168\.\d{1,3}\.\d{1,3}$/.test(hostname)) {
      return true;
    }
    // Link-local 169.254.0.0/16
    if (/^169\.254\.\d{1,3}\.\d{1,3}$/.test(hostname)) {
      return true;
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
    return `${parsed.hostname}:${port}`;
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
  // Exact match
  if (whitelistEntry === hostPort) return true;

  // Match without port (allow any port on that host)
  const host = hostPort.split(':')[0];
  if (whitelistEntry === host) return true;

  // Match with wildcard port
  if (whitelistEntry.endsWith(':*') && whitelistEntry.slice(0, -2) === host) return true;

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
