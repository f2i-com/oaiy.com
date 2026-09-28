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
      const added = hosts.map((h) => gate.allowHost(h)).filter(Boolean);
      if (gate.mode === 'blocked') {
        gate.setMode('allowlist');
        note = ' (the gate was closed; it now lets through only the allowed hosts)';
      }
      note = `allowed ${added.join(', ')}${note}`;
      break;
    }
    case 'deny': case 'ban': {
      if (!hosts.length) return 'usage: /internet deny <host> [host…]';
      note = `denied ${hosts.map((h) => gate.denyHost(h)).filter(Boolean).join(', ')}`;
      break;
    }
    case 'forget': case 'remove': case 'rm': {
      if (!hosts.length) return 'usage: /internet forget <host> [host…]';
      for (const h of hosts) gate.forgetHost(h);
      note = `forgot ${hosts.join(', ')}`;
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
