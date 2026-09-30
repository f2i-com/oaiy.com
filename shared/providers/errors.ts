/*
 * Errors a person can act on: what went wrong when a provider was called, in words, with what to do about it.
 *
 * The first half (`ConnectionErrorKind` to `kindForStatus`) is the Agent's own, moved here unchanged: adapted from softn.com
 * (apps/softn-studio/src/lib/providerConnection.ts), Copyright f2i-com, licensed under the Apache License, Version 2.0.
 * app/src/agent/providers/providerConnection.ts re-exports it.
 *
 * The second half is the providers origin's: the three ways a call can fail that the browser reports as one
 * "Failed to fetch" (design 4.2, step 4), told apart, and the redaction that keeps a provider's own error text from
 * carrying a key back to a page.
 */
import { CLOUD_BASE, LOCAL_SERVERS } from './endpoints';
import type { LocalServerKind, ProviderType } from './types';

export type ConnectionErrorKind =
  | 'auth'
  | 'forbidden'
  | 'rate-limited'
  | 'not-found'
  | 'server'
  | 'http'
  | 'network'
  | 'mixed-content'
  | 'invalid-response'
  | 'timeout'
  | 'cancelled'
  | 'no-model';

export class ProviderConnectionError extends Error {
  readonly kind: ConnectionErrorKind;
  readonly status?: number;

  constructor(kind: ConnectionErrorKind, message: string, status?: number) {
    super(message);
    this.name = 'ProviderConnectionError';
    this.kind = kind;
    this.status = status;
  }
}

export interface ErrorContext {
  type: ProviderType;
  serverKind?: LocalServerKind;
  /** The address that was called. */
  url: string;
  /** This page's origin, for the CORS advice. */
  pageOrigin?: string;
  /** The provider's own words, when it sent any. */
  detail?: string;
  /**
   * Say the address as its origin only. For a message an app will read: the base path of an address can hold an account or a tenant id,
   * and the list an app is given shows it no more than the host.
   */
  omitAddressPath?: boolean;
}

function isLocalType(type: ProviderType): boolean {
  return type === 'local' || type === 'custom';
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

function originOf(url: string): string {
  try {
    return new URL(url).origin;
  } catch {
    return url;
  }
}

/** Whether a hostname is this computer — the one plain-http target an https page may call. */
export function isLoopbackHost(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, '').toLowerCase();
  return host === 'localhost' || host.endsWith('.localhost') || host === '::1' || /^127(\.\d{1,3}){3}$/.test(host);
}

/**
 * Whether a hostname is one whose traffic stays on this computer or this network, so that a key sent to it over plain http is not sent across
 * the internet: this computer (`localhost`, 127/8, ::1), the private ranges (10/8, 172.16/12, 192.168/16), link-local addresses
 * (169.254/16, fe80::/10), IPv6 unique-local addresses (fc00::/7), and a name ending `.local` (mDNS). Nothing else: not 100.64/10 (which
 * is shared address space, not a person's own network), not a single-label name, and not a name that only STARTS with an address
 * (`192.168.1.5.evil.example`). `hostname` is what `new URL(...).hostname` says, which is where an address written as a number, in hex or
 * in octal (`http://3232235781/`) has already been read as the dotted address it is.
 */
export function isPrivateNetworkHost(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (isLoopbackHost(host) || host.endsWith('.local')) return true;
  const v4 = (a: number, b: number): boolean => a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 169 && b === 254) || a === 127;
  const dotted = /^(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/.exec(host);
  if (dotted) return v4(Number(dotted[1]), Number(dotted[2]));
  if (!host.includes(':')) return false;
  // An IPv4 address written into an IPv6 one (::ffff:192.168.1.5, which a URL reads as ::ffff:c0a8:105) is the IPv4 address.
  const mapped = /^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/.exec(host);
  if (mapped) {
    const high = parseInt(mapped[1], 16);
    return v4(high >> 8, high & 0xff);
  }
  return /^fe[89ab][0-9a-f]:/.test(host) || /^f[cd][0-9a-f]{2}:/.test(host);
}

/** Whether the address is one on this computer or this network (where a browser may ask for a permission the person can give). */
export function isLocalAddress(url: string): boolean {
  try {
    const host = new URL(url).hostname.replace(/^\[|\]$/g, '');
    if (isLoopbackHost(host) || host.endsWith('.local')) return true;
    const m = /^(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/.exec(host);
    if (!m) return false;
    const [a, b] = [Number(m[1]), Number(m[2])];
    return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 169 && b === 254) || (a === 100 && b >= 64 && b <= 127);
  } catch {
    return false;
  }
}

/**
 * Whether the browser will block this call as mixed content: an https page
 * may not call plain http, except on this computer (localhost, 127.0.0.1,
 * [::1]), which browsers treat as secure.
 */
export function isBlockedMixedContent(target: string, pageProtocol: string): boolean {
  if (pageProtocol !== 'https:') return false;
  try {
    const url = new URL(target);
    return url.protocol === 'http:' && !isLoopbackHost(url.hostname);
  } catch {
    return false;
  }
}

const PROVIDER_LABEL: Record<ProviderType, string> = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  local: 'The local server',
  custom: 'The server',
};

/** What went wrong, in words, with what to do about it. */
export function describeConnectionError(kind: ConnectionErrorKind, context: ErrorContext, status?: number): string {
  const who = PROVIDER_LABEL[context.type];
  const detail = context.detail ? ` The provider said: “${context.detail}”` : '';
  const local = isLocalType(context.type);
  switch (kind) {
    case 'auth':
      return local
        ? `${who} asked for a key and did not accept the one given (401). Enter the key it expects, or leave the key empty if it does not use one.${detail}`
        : `${who} did not accept this API key (401). Check that the whole key was pasted and that it has not been revoked.${detail}`;
    case 'forbidden':
      return local
        ? `${who} refused the request (403). Check its access settings.${detail}`
        : `${who} accepted the key but will not let it do this (403). The key may be restricted, or the account may not have access yet — check the key’s permissions and the account’s billing in the ${who} console.${detail}`;
    case 'rate-limited':
      return local
        ? `${who} is too busy to answer (429). Wait a moment and test again.${detail}`
        : `${who} is refusing requests for now (429). On a new account this usually means no credit or billing is set up; otherwise it is a rate limit. Check billing and usage in the ${who} console, then test again.${detail}`;
    case 'not-found':
      return `Nothing answered at ${context.omitAddressPath ? originOf(context.url) : context.url} (404). Check the address: it should be the server’s base, such as ${local ? LOCAL_SERVERS[context.serverKind ?? 'ollama'].baseUrl : CLOUD_BASE[context.type] ?? 'https://…/v1'}, without /chat/completions at the end.${detail}`;
    case 'server':
      return `${who} had an error of its own (${status ?? 'server error'}). Try again in a moment.${detail}`;
    case 'http':
      return `${who} answered with an error (${status ?? 'unknown status'}).${detail}`;
    case 'mixed-content':
      return `This page is served over HTTPS, and browsers block it from calling a plain http:// address that is not on this computer (${hostOf(context.url)}). For a server on this computer use http://localhost or http://127.0.0.1; for one elsewhere, put it behind HTTPS.`;
    case 'network': {
      if (!local) return `Studio could not reach ${hostOf(context.url)}. Check the internet connection, and whether an extension, proxy or firewall is blocking it.`;
      const page = context.pageOrigin ?? 'this site';
      const fix = context.serverKind === 'lmstudio'
        ? 'In LM Studio, open the Developer tab, start the server, and switch on “Enable CORS”.'
        : context.serverKind === 'oaiy'
          ? `Start OAIY, and add ${page} to gateway.cors_origins in its config (or set an API key there and enter it here).`
          : context.serverKind === 'other'
          ? `Allow requests from ${page} in the server’s CORS settings.`
          : `For Ollama, quit it and start it again with OLLAMA_ORIGINS set to include ${page} (for example OLLAMA_ORIGINS="${page}" ollama serve). LM Studio has an “Enable CORS” switch in its server settings.`;
      return `Studio could not reach ${originOf(context.url)}. Either the server is not running there, or it is blocking requests from this page (CORS). ${fix}`;
    }
    case 'invalid-response':
      return `${who} answered, but not with a model list. Check that the address points at an OpenAI-compatible API${local ? ' (usually ending in /v1)' : ''}.`;
    case 'timeout':
      return `${who} did not answer in time. Check that it is running and reachable, then test again.`;
    case 'cancelled':
      return 'The check was cancelled.';
    case 'no-model':
      return 'Choose a model first.';
  }
}

/** The error kind for an HTTP status. */
export function kindForStatus(status: number): ConnectionErrorKind {
  if (status === 401) return 'auth';
  if (status === 403) return 'forbidden';
  if (status === 429 || status === 402) return 'rate-limited';
  if (status === 404) return 'not-found';
  if (status >= 500) return 'server';
  return 'http';
}

// --- What the browser reports as one "Failed to fetch" -----------------------------

/**
 * Why a call that never got an answer failed, as far as a page can tell (design 4.2, step 4). The browser reports a server
 * that is down, a CORS refusal, a blocked local-network permission and mixed content as the same TypeError, so the
 * providers origin asks once more, with `mode: 'no-cors'`: an opaque answer means the server is up and refused CORS; no
 * answer means it is down, or blocked by the local-network permission.
 */
export type NetworkFailure = 'cors' | 'unreadable' | 'unreachable' | 'mixed-content';

export interface NetworkFailureContext {
  /** The address that was called. */
  url: string;
  /** The origin of the page that called it: the one to allow. */
  pageOrigin: string;
  serverKind?: LocalServerKind;
  /** Whether the address is on this computer or this network, for the sentence about permission. */
  local: boolean;
}

/** The sentence for a failure, with the one origin to allow and, where there is one, the exact line to write. */
export function describeNetworkFailure(failure: NetworkFailure, context: NetworkFailureContext): string {
  const target = originOf(context.url);
  const page = context.pageOrigin;
  switch (failure) {
    case 'mixed-content':
      return `This page is served over HTTPS, and browsers block it from calling a plain http:// address that is not on this computer (${hostOf(context.url)}). For a server on this computer use http://localhost or http://127.0.0.1; for one elsewhere, put it behind HTTPS.`;
    case 'cors': {
      const line =
        context.serverKind === 'ollama'
          ? `Start Ollama with OLLAMA_ORIGINS="${page}" (for example OLLAMA_ORIGINS="${page}" ollama serve).`
          : context.serverKind === 'lmstudio'
            ? 'In LM Studio, open the Developer tab and switch on “Enable CORS”, then allow this page.'
            : context.serverKind === 'oaiy'
              ? `Add ${page} to gateway.cors_origins in OAIY’s config.`
              : `Allow requests from ${page} in the server’s CORS settings (for llama.cpp: --cors-origins ${page} --cors-headers Authorization,Content-Type).`;
      return `${target} is running, but it does not allow requests from ${page}. Allow ${page}: ${line}`;
    }
    case 'unreadable':
      // A service on the internet that is up and answered in a way the browser will not let a page read. Some providers do this to a
      // wrong or revoked key on a chat request (OpenAI's 401 has no CORS headers), so the first thing to look at is the key.
      return `${target} answered, but the browser could not read the answer. Some providers answer this way when the key is wrong or has been revoked: check the key on the Providers page, then test again.`;
    case 'unreachable':
      return `Nothing answered at ${target}. Either the server is not running there, or the browser blocked the call${
        context.local ? ' (a page may need your permission to reach devices on your local network: look for the prompt or the address-bar icon)' : ''
      }. Check the address, and that the server is running.`;
  }
}

/**
 * The text with a secret, and anything that looks like part of it, taken out. A provider's own error can quote the key it
 * refused (OpenAI's 401 shows `sk-proj-…abcd`). The whole secret goes, and so does any run of four or more characters that the
 * secret also contains: that is what a mask leaves.
 *
 * It is ONLY for text the holder shows on its own pages, whose DOM no app can read. It is NOT what stands between an app and the key:
 * output that depends on the key is a channel (an app that chooses what a provider echoes reads, from what is removed, which pieces the
 * key holds, and a key under eight characters is not scrubbed at all), so an app is given none of a provider's error text and the
 * question does not arise (web/providers/src/fixed.ts).
 */
export function redactSecret(text: string, secret: string): string {
  if (!secret || secret.length < 8) return text;
  const WINDOW = 4;
  const windows = new Set<string>();
  for (let i = 0; i + WINDOW <= secret.length; i++) windows.add(secret.slice(i, i + WINDOW));
  let result = '';
  let hidden = false;
  for (let i = 0; i < text.length; ) {
    const covered = i + WINDOW <= text.length && windows.has(text.slice(i, i + WINDOW));
    if (covered) {
      // Take the whole run that overlaps the secret's windows.
      let end = i + WINDOW;
      while (end < text.length && windows.has(text.slice(end - WINDOW + 1, end + 1))) end++;
      if (!hidden) result += '…';
      hidden = true;
      i = end;
      continue;
    }
    hidden = false;
    result += text[i];
    i++;
  }
  return result;
}
