/**
 * Slash commands typed into the chat: `/internet` (`/net`) drives the network
 * gate, `/clear` starts a new conversation, `/help` lists them.
 */
import { parseMode, type NetGate } from './gate/netgate';

export const INTERNET_USAGE =
  'usage: /internet [status] | on | off | allowlist [host…] | allow <host…> | deny <host…> | forget <host…> | reset   (/net for short)';

export function internetStatus(gate: NetGate): string {
  const status = gate.status();
  const s = status.settings;
  const mode = {
    open: 'on (open: tools may reach any host not on the deny list)',
    allowlist: 'allowlist (tools may reach only the allowed hosts)',
    blocked: 'off (blocked: no tool traffic leaves)',
  }[s.mode];
  const lines = [
    `internet=${mode}`,
    `  allow: ${s.allow.length ? s.allow.join(', ') : '(none)'}`,
    `  deny:  ${s.deny.length ? s.deny.join(', ') : '(none)'}`,
    `  requests this session: ${status.allowed} allowed, ${status.blocked} blocked`,
  ];
  if (status.recentBlocked.length) {
    lines.push(`  recently blocked: ${status.recentBlocked.slice(-6).reverse().map((b) => `${b.host} (${b.via})`).join(', ')}`);
  }
  lines.push('  covers: fetch/curl/wget in the sandbox and web_fetch. Requests to your AI provider are not gated. Local-network hosts (localhost, 192.168.x.x, …) need an explicit allow.');
  return lines.join('\n');
}

/** Run `/internet …`; returns the text to show. */
export function internetCommand(gate: NetGate, args: string[]): string {
  const action = args[0]?.toLowerCase();
  const hosts = args.slice(1);
  let note = '';
  if (!action || action === 'status' || action === '?') return `${internetStatus(gate)}\n${INTERNET_USAGE}`;
  switch (action) {
    case 'reset':
      gate.replace({ mode: 'open', allow: [], deny: [] });
      break;
    case 'allow': case 'add': case 'permit': {
      if (!hosts.length) return 'usage: /internet allow <host> [host…]';
      const tried = hosts.map((h) => ({ given: h, host: gate.allowHost(h) }));
      const added = tried.flatMap((t) => (t.host ? [t.host] : []));
      const bad = tried.filter((t) => !t.host).map((t) => t.given);
      // (nothing allowed, nothing changed: the gate is not opened for a mistyped host)
      if (!added.length) return `Not a host name: ${hosts.join(', ')}. Nothing was changed.`;
      if (gate.mode === 'blocked') {
        gate.setMode('allowlist');
        note = ' (the gate was closed; it now lets through only the allowed hosts)';
      }
      note = `allowed ${added.join(', ')}${note}${bad.length ? ` (not host names, left out: ${bad.join(', ')})` : ''}`;
      break;
    }
    case 'deny': case 'ban': {
      if (!hosts.length) return 'usage: /internet deny <host> [host…]';
      const tried = hosts.map((h) => ({ given: h, host: gate.denyHost(h) }));
      const denied = tried.flatMap((t) => (t.host ? [t.host] : []));
      const bad = tried.filter((t) => !t.host).map((t) => t.given);
      if (!denied.length) return `Not a host name: ${hosts.join(', ')}. Nothing was changed.`;
      note = `denied ${denied.join(', ')}${bad.length ? ` (not host names, left out: ${bad.join(', ')})` : ''}`;
      break;
    }
    case 'forget': case 'remove': case 'rm': {
      if (!hosts.length) return 'usage: /internet forget <host> [host…]';
      const forgot = hosts.filter((h) => gate.forgetHost(h));
      const unknown = hosts.filter((h) => !forgot.includes(h));
      note = forgot.length ? `forgot ${forgot.join(', ')}${unknown.length ? ` (not on either list: ${unknown.join(', ')})` : ''}` : `${unknown.join(', ')} ${unknown.length === 1 ? 'is' : 'are'} not on the allow or deny list: nothing was changed`;
      break;
    }
    default: {
      const mode = parseMode(action);
      if (!mode) return INTERNET_USAGE;
      gate.setMode(mode);
      if (mode === 'allowlist') for (const h of hosts) gate.allowHost(h);
    }
  }
  return note ? `${note}\n${internetStatus(gate)}` : internetStatus(gate);
}

export const HELP = `Commands:
  /internet … (/net)   the network gate: on | off | allowlist | allow <host> | deny <host> | forget <host> | reset | status
  /clear               start a new conversation (the project's files stay)
  /help                this list

Everything runs in this tab: files live in the browser, code runs on the Zipp VM in a Web Worker,
and nothing leaves except requests to your AI provider and what the network gate lets through.`;

/** What a phone conversation's box takes: words for its agent, and /clear. */
export const PHONE_HELP = "Here, what you write goes to this conversation's agent (the receptionist on a call, or the agent of their texts). /clear starts the conversation again once its call has ended.";
