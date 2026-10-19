/**
 * The network gate: one switch every outbound request made on the model's
 * behalf answers to — the sandbox's `fetch`/`curl`/`wget`, and the agent's
 * `web_fetch`. The user drives it with `/internet` (`/net`): open,
 * allowlist-only or blocked, plus per-host allow and deny lists. Requests to
 * the configured AI provider are not tool traffic and are never gated.
 *
 * Ported from coder-cli's `tool_core::net_gate`.
 */

export type NetMode = 'open' | 'allowlist' | 'blocked';

export interface NetGateSettings {
  mode: NetMode;
  allow: string[];
  deny: string[];
}

export interface NetGateStatus {
  settings: NetGateSettings;
  allowed: number;
  blocked: number;
  recentBlocked: Array<{ host: string; via: string }>;
}

const RECENT_BLOCKED = 12;

export function parseMode(value: string): NetMode | null {
  switch (value.trim().toLowerCase()) {
    case 'open': case 'on': case 'true': case 'yes': return 'open';
    case 'allowlist': case 'allow-list': case 'restricted': case 'limited': case 'only': return 'allowlist';
    case 'blocked': case 'off': case 'false': case 'no': case 'block': case 'closed': return 'blocked';
    default: return null;
  }
}

/** `https://*.Example.com:443/x` -> `example.com`. */
export function normalizeHost(raw: string): string {
  let host = raw.trim().toLowerCase();
  const scheme = host.indexOf('://');
  if (scheme >= 0) host = host.slice(scheme + 3);
  const end = host.search(/[/?#]/);
  if (end >= 0) host = host.slice(0, end);
  const at = host.lastIndexOf('@');
  if (at >= 0) host = host.slice(at + 1);
  if (host.startsWith('[')) {
    const close = host.indexOf(']');
    if (close > 0) host = host.slice(1, close);
  } else if ((host.match(/:/g) ?? []).length === 1) {
    host = host.slice(0, host.indexOf(':'));
  }
  return host.replace(/^\*\./, '').replace(/\.$/, '');
}

/** A label of a host name: letters, digits, hyphens (not at either end) and, as some real hosts have, underscores. */
const LABEL = /^(?!-)[a-z0-9_-]{1,63}(?<!-)$/;

/**
 * `raw` as the gate's lists keep a host (normalizeHost), or null when it is not one: a host name (`example.com`, `localhost`, one
 * written in another script, kept as the browser writes it), an IPv4 address or an IPv6 one. `*.example.com` is `example.com`,
 * which covers its subdomains anyway; a name with a comma, a space, a `*` left in it or nothing at all is refused.
 */
export function gateHost(raw: string): string | null {
  let host = normalizeHost(raw);
  if (!host || host.length > 253) return null;
  if (host.includes(':')) return /^[0-9a-f:.]+$/.test(host) && (host.match(/:/g) ?? []).length >= 2 ? host : null;
  // (a name in another script, as the browser sends it: bücher.example is xn--bcher-kva.example)
  if (/[^\x00-\x7f]/.test(host)) {
    try {
      host = new URL(`http://${host}/`).hostname;
    } catch {
      return null;
    }
  }
  const labels = host.split('.');
  if (!labels.every((l) => LABEL.test(l))) return null;
  // (no top-level domain is all digits: a name that ends in one is an IPv4 address, four numbers each up to 255)
  if (/^\d+$/.test(labels[labels.length - 1])) return labels.length === 4 && labels.every((l) => /^\d{1,3}$/.test(l) && Number(l) <= 255) ? host : null;
  return host;
}

function hostMatches(host: string, entry: string): boolean {
  return entry !== '' && (host === entry || host.endsWith(`.${entry}`));
}

export class NetGate {
  private settings: NetGateSettings;
  private allowedCount = 0;
  private blockedCount = 0;
  private recent: Array<{ host: string; via: string }> = [];
  private listeners = new Set<() => void>();

  constructor(settings?: Partial<NetGateSettings>) {
    this.settings = {
      mode: settings?.mode ?? 'open',
      allow: [...new Set((settings?.allow ?? []).map(normalizeHost).filter(Boolean))].sort(),
      deny: [...new Set((settings?.deny ?? []).map(normalizeHost).filter(Boolean))].sort(),
    };
  }

  onChange(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private changed(): void {
    for (const listener of this.listeners) listener();
  }

  getSettings(): NetGateSettings {
    return { mode: this.settings.mode, allow: [...this.settings.allow], deny: [...this.settings.deny] };
  }

  get mode(): NetMode {
    return this.settings.mode;
  }

  setMode(mode: NetMode): void {
    this.settings.mode = mode;
    this.changed();
  }

  replace(settings: NetGateSettings): void {
    this.settings = new NetGate(settings).getSettings();
    this.changed();
  }

  /** Put a host on the allow list (and off the deny list): the host as kept, or null when `raw` is not one (gateHost). */
  allowHost(raw: string): string | null {
    const host = gateHost(raw);
    if (!host) return null;
    this.settings.deny = this.settings.deny.filter((h) => h !== host);
    if (!this.settings.allow.includes(host)) this.settings.allow = [...this.settings.allow, host].sort();
    this.changed();
    return host;
  }

  /** Put a host on the deny list (and off the allow list): the host as kept, or null when `raw` is not one. */
  denyHost(raw: string): string | null {
    const host = gateHost(raw);
    if (!host) return null;
    this.settings.allow = this.settings.allow.filter((h) => h !== host);
    if (!this.settings.deny.includes(host)) this.settings.deny = [...this.settings.deny, host].sort();
    this.changed();
    return host;
  }

  forgetHost(raw: string): boolean {
    const host = normalizeHost(raw);
    const before = this.settings.allow.length + this.settings.deny.length;
    this.settings.allow = this.settings.allow.filter((h) => h !== host);
    this.settings.deny = this.settings.deny.filter((h) => h !== host);
    this.changed();
    return this.settings.allow.length + this.settings.deny.length !== before;
  }

  /** Would a connection to `host` be allowed? No side effects. */
  permits(raw: string): { ok: true } | { ok: false; reason: string } {
    const host = normalizeHost(raw);
    if (!host) return { ok: false, reason: 'the network gate needs a host name' };
    if (this.settings.deny.some((e) => hostMatches(host, e))) {
      return { ok: false, reason: `\`${host}\` is on the network gate's deny list (/internet allow ${host} to lift it)` };
    }
    switch (this.settings.mode) {
      case 'open':
        return { ok: true };
      case 'blocked':
        return { ok: false, reason: `the network gate is closed (\`/internet off\`); \`${host}\` was not contacted. The user can reopen it with \`/internet on\` or \`/internet allow ${host}\`` };
      case 'allowlist':
        return this.settings.allow.some((e) => hostMatches(host, e))
          ? { ok: true }
          : { ok: false, reason: `\`${host}\` is not on the network gate's allow list; the user can add it with \`/internet allow ${host}\`` };
    }
  }

  /** Decide for a real request and record the outcome. */
  check(raw: string, via: string): { ok: true } | { ok: false; reason: string } {
    const decision = this.permits(raw);
    if (decision.ok) {
      this.allowedCount++;
    } else {
      this.blockedCount++;
      this.recent.push({ host: normalizeHost(raw), via });
      if (this.recent.length > RECENT_BLOCKED) this.recent.shift();
    }
    this.changed();
    return decision;
  }

  status(): NetGateStatus {
    return { settings: this.getSettings(), allowed: this.allowedCount, blocked: this.blockedCount, recentBlocked: [...this.recent] };
  }
}
