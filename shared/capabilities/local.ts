/**
 * What counts as "this computer or its network": the addresses a public page has no business asking (design 3.5 rule R1), for the
 * apps' own guards and for the browser tests that hold a page to them (web/tests/e2e/local-ranges.mjs uses this same function, so what
 * a test counts as a probe is what a guard refuses).
 *
 * Loopback (127/8, `::1`, `localhost` and every `*.localhost` name, which a browser resolves to this computer), the private ranges
 * (10/8, 172.16/12, 192.168/16, fc00::/7), link-local (169.254/16, fe80::/10), carrier-grade NAT (100.64/10, which Tailscale uses),
 * `0.0.0.0` and "this network" (0/8), an IPv4 address written as IPv6, and names that are only ever local (`.local`, `.internal`,
 * `.lan`, `.home.arpa`).
 */

/** IPv4 `a.b.c.d` as four numbers, or null. */
function ipv4(host: string): number[] | null {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (!m) return null;
  const parts = m.slice(1).map(Number);
  return parts.every((n) => n <= 255) ? parts : null;
}

/** An IPv6 literal's first 16-bit group, or null when it is not one. IPv4-mapped forms are answered by their IPv4 address. */
function ipv6(host: string): { v4?: string; first?: number; loopback?: boolean } | null {
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
export function isLocalHostname(hostname: string): boolean {
  const host = String(hostname).toLowerCase().replace(/\.$/, '');
  if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal') || host.endsWith('.lan') || host.endsWith('.home.arpa')) return true;
  const six = ipv6(host);
  if (six) {
    if (six.v4) return isLocalHostname(six.v4);
    if (six.loopback) return true;
    const first = six.first ?? 0;
    // fc00::/7 (unique local), fe80::/10 (link-local)
    return (first & 0xfe00) === 0xfc00 || (first & 0xffc0) === 0xfe80;
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
