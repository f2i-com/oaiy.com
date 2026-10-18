/**
 * What counts as "this computer or its network": the addresses a public page has no business asking (design 3.5 rule R1), for the
 * apps' own guards. (The browser tests hold a page to the same list with their OWN implementation, web/tests/e2e/local-ranges.mjs, and a
 * unit test makes the two agree: a range left out of one shows as a difference, not as a page passing unseen.)
 *
 * Loopback (127/8, `::1`, `localhost` and every `*.localhost` name, which a browser resolves to this computer), the private ranges
 * (10/8, 172.16/12, 192.168/16, fc00::/7), link-local (169.254/16, fe80::/10), carrier-grade NAT (100.64/10, which Tailscale uses),
 * `0.0.0.0` and "this network" (0/8), an IPv4 address written as IPv6 (`::ffff:a.b.c.d`) or carried in one (NAT64 `64:ff9b::/96`,
 * 6to4 `2002::/16`, Teredo `2001::/32`: local when the address they carry is), the local-use NAT64 prefix `64:ff9b:1::/48`, names that
 * are only ever local (`.local`, `.internal`, `.lan`, `.home.arpa`) and a name of a single label (`printer`, `nas`), which only a
 * search domain, a hosts file or mDNS on this network can resolve.
 *
 * What it cannot see: it reads the address as written. A public name that redirects to this network, and a public name that resolves to
 * a private address (`192.168.1.5.nip.io`), look public here, as do the other names a person's own DNS may point inside. The browser's
 * own question about connecting to devices on the network (Local Network Access) still gates them; see README.md.
 *
 * It is the one rule for addresses in shared/: shared/providers/errors.ts asks it too, in three ways that mean three things.
 *   - isLocalHostname: this computer or its network, as wide as it can be told (a page's guard, and where a browser may ask the person).
 *   - isPrivateNetworkHostname: where traffic stays on this computer or this network, so a key may go there over plain http. The same,
 *     less what may cross the internet or is no one's own network: carrier-grade NAT (100.64/10, a carrier's), 0/8 and `::` (no host
 *     to reach), an IPv4 address carried in IPv6 by a translator or a tunnel (NAT64, 6to4, Teredo go through a relay) and a name of a
 *     single label (whatever a search domain makes of it).
 *   - isLoopbackHostname: this computer alone (`localhost`, `*.localhost`, 127/8, `::1`), which a browser counts as secure over plain http.
 */

/** IPv4 `a.b.c.d` as four numbers, or null. */
function ipv4(host: string): number[] | null {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (!m) return null;
  const parts = m.slice(1).map(Number);
  return parts.every((n) => n <= 255) ? parts : null;
}

/** Whether an IPv4 address (four numbers) is this computer or its network. */
function ipv4IsLocal([a, b]: number[]): boolean {
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

/** An IPv6 literal (with or without its brackets, a dotted IPv4 tail allowed) as eight 16-bit groups, or null when it is not one. */
function ipv6Groups(host: string): number[] | null {
  let bare = host.startsWith('[') && host.endsWith(']') ? host.slice(1, -1) : host;
  if (!bare.includes(':')) return null;
  bare = bare.split('%')[0]; // a zone
  // A dotted IPv4 tail is two groups.
  const tail = /^(.*:)(\d+\.\d+\.\d+\.\d+)$/.exec(bare);
  if (tail) {
    const four = ipv4(tail[2]);
    if (!four) return null;
    bare = `${tail[1]}${((four[0] << 8) | four[1]).toString(16)}:${((four[2] << 8) | four[3]).toString(16)}`;
  }
  const halves = bare.split('::');
  if (halves.length > 2) return null;
  const side = (text: string): number[] | null => {
    if (text === '') return [];
    const groups = text.split(':');
    return groups.every((g) => /^[0-9a-f]{1,4}$/i.test(g)) ? groups.map((g) => parseInt(g, 16)) : null;
  };
  const head = side(halves[0]);
  const rest = halves.length === 2 ? side(halves[1]) : [];
  if (!head || !rest) return null;
  if (halves.length === 1) return head.length === 8 ? head : null;
  const missing = 8 - head.length - rest.length;
  return missing >= 1 ? [...head, ...new Array<number>(missing).fill(0), ...rest] : null;
}

const dotted = (hi: number, lo: number) => [hi >> 8, hi & 255, lo >> 8, lo & 255];

/** Whether an IPv6 address (eight groups) is this computer or its network, or carries an IPv4 address that is. */
function ipv6IsLocal(g: number[]): boolean {
  const zero = (from: number, to: number) => g.slice(from, to).every((x) => x === 0);
  if (zero(0, 7)) return g[7] <= 1; // `::` and `::1`
  if (zero(0, 5) && g[5] === 0xffff) return ipv4IsLocal(dotted(g[6], g[7])); // IPv4-mapped
  if (g[0] === 0x64 && g[1] === 0xff9b) {
    if (zero(2, 6)) return ipv4IsLocal(dotted(g[6], g[7])); // NAT64, the well-known prefix: the IPv4 address it reaches
    if (g[2] === 1) return true; // 64:ff9b:1::/48, the local-use NAT64 prefix (RFC 8215): a network's own
  }
  if (g[0] === 0x2002) return ipv4IsLocal(dotted(g[1], g[2])); // 6to4: the IPv4 address it carries
  if (g[0] === 0x2001 && g[1] === 0) return ipv4IsLocal(dotted(~g[6] & 0xffff, ~g[7] & 0xffff)); // Teredo: the client's address, kept inverted
  return ipv6Private(g);
}

/** fc00::/7 (unique local) and fe80::/10 (link-local). */
function ipv6Private(g: number[]): boolean {
  return (g[0] & 0xfe00) === 0xfc00 || (g[0] & 0xffc0) === 0xfe80;
}

/** Whether `hostname` (as a URL gives it: `127.0.0.1`, `[::1]`, `printer.local`) is this computer or its network. */
export function isLocalHostname(hostname: string): boolean {
  const host = String(hostname).toLowerCase().replace(/\.$/, '');
  if (host === '') return false;
  if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal') || host.endsWith('.lan') || host.endsWith('.home.arpa')) return true;
  if (host.includes(':') || host.startsWith('[')) {
    const groups = ipv6Groups(host);
    return groups ? ipv6IsLocal(groups) : false;
  }
  const four = ipv4(host);
  if (four) return ipv4IsLocal(four);
  // A name of one label is resolved by this network (a search domain, a hosts file, mDNS), not by the public DNS.
  return !host.includes('.');
}

/** A hostname as the three rules read it: lowercase, without a final dot. */
function plain(hostname: string): string {
  return String(hostname).toLowerCase().replace(/\.$/, '');
}

/**
 * Whether `hostname` is one whose traffic stays on this computer or this network (see the top of this file for what is left out
 * of isLocalHostname, and why): a key sent there over plain http is not sent across the internet. A name is read by isLocalHostname's
 * own list of names, a name of one label aside; `192.168.1.5.evil.example` is a name, not an address.
 */
export function isPrivateNetworkHostname(hostname: string): boolean {
  const host = plain(hostname);
  if (host === '') return false;
  if (host.includes(':') || host.startsWith('[')) {
    const g = ipv6Groups(host);
    if (!g) return false;
    if (g.slice(0, 7).every((x) => x === 0)) return g[7] === 1; // `::1`, not `::`
    if (g.slice(0, 5).every((x) => x === 0) && g[5] === 0xffff) return ipv4StaysHere(dotted(g[6], g[7])); // IPv4 written as IPv6
    return ipv6Private(g);
  }
  const four = ipv4(host);
  if (four) return ipv4StaysHere(four);
  return isLoopbackHostname(host) || (host.includes('.') && isLocalHostname(host));
}

/** isLocalHostname's IPv4 ranges but 0/8 (no host) and 100.64/10 (carrier-grade NAT, its only range in 100/8). */
function ipv4StaysHere(four: number[]): boolean {
  return ipv4IsLocal(four) && four[0] !== 0 && four[0] !== 100;
}

/** Whether `hostname` is this computer alone: `localhost` and `*.localhost`, 127/8 and `::1`, which a browser counts as secure. */
export function isLoopbackHostname(hostname: string): boolean {
  const host = plain(hostname);
  if (host === 'localhost' || host.endsWith('.localhost')) return true;
  if (host.includes(':') || host.startsWith('[')) {
    const g = ipv6Groups(host);
    return g !== null && g.slice(0, 7).every((x) => x === 0) && g[7] === 1;
  }
  return ipv4(host)?.[0] === 127;
}
