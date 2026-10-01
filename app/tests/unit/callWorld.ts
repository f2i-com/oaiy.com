// What the call-nudge tests share: the local engine behind a fake fetch (who asks, when, and streams that break or hang),
// agents on it, and a phone's worth of Sessions and Outreach with the desktop stood in for (no call is ever placed).
import { vi } from 'vitest';
import { Agent, WARM_TOKENS, type AgentEvent } from '../../src/agent/agent';
import type { Turn } from '../../src/agent/protocol';
import type { ProviderConfig } from '../../src/agent/providers/types';
import type { Desktop } from '../../src/desktop/bridge';
import { NetGate } from '../../src/gate/netgate';
import { Outreach, type Campaign, type DoNotContact } from '../../src/outreach';
import { outreachTools } from '../../src/outreachTools';
import { PhoneLine } from '../../src/phoneLine';
import { Sessions } from '../../src/sessions';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import type { CallerNote, SessionInfo } from '../../src/vfs/projects';
import { Vfs } from '../../src/vfs/vfs';

export const ENGINE: ProviderConfig = { id: 'oaiy', type: 'local', serverKind: 'oaiy', name: 'OAIY', apiKey: '', baseUrl: 'http://127.0.0.1:8080', modelId: 'Qwen3.8-Flash-Next', followEngine: true };
export const GREEN = { business: 'Green Lawns', receptionist: 'Aokie' };
export const JANE = '+61412345678';
export const TOM = '+61498765432';

export interface Answer {
  text?: string;
  calls?: Array<{ name: string; input: Record<string, unknown> }>;
  /** Wait this long before the first word. */
  delayMs?: number;
  /** Send one chunk, then break the stream. */
  breakAfterChunk?: boolean;
  /** Send nothing, ever (until the request is aborted). */
  hang?: boolean;
}

/** One request as the server saw it: who asked, when it came and when it was answered (ms from the engine's start). */
export interface Seen {
  who: string;
  at: number;
  end?: number;
}

/** Who made a request: a call's agent (or its warm), the runner (it has start_outreach), or an agent that says its name (ROLE:name). */
export function who(body: Record<string, unknown>): string {
  const messages = body.messages as Array<{ role: string; content: unknown }>;
  const role = /ROLE:(\w+)/.exec(String(messages[0].content))?.[1];
  if (role) return role;
  const tools = ((body.tools as Array<{ function: { name: string } }> | undefined) ?? []).map((t) => t.function.name);
  if (tools.includes('end_call')) return body.max_tokens === WARM_TOKENS ? 'warm' : 'call';
  return tools.includes('start_outreach') ? 'runner' : 'other';
}

/**
 * The local engine behind a fake fetch: each request answered by `answer`. `log` says who asked, in the order they came;
 * `seen` also when (and when each was answered).
 */
export function engine(answer: (body: Record<string, unknown>, who: string) => Answer, log: string[] = [], seen: Seen[] = []): Array<Record<string, unknown>> {
  const sent: Array<Record<string, unknown>> = [];
  const t0 = Date.now();
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as Record<string, unknown>;
    sent.push(body);
    const asker = who(body);
    log.push(asker);
    const entry: Seen = { who: asker, at: Date.now() - t0 };
    seen.push(entry);
    const a = answer(body, asker);
    const events: unknown[] = [];
    for (const piece of a.text?.match(/.{1,7}/gs) ?? []) events.push({ choices: [{ index: 0, delta: { content: piece } }] });
    for (const [i, call] of (a.calls ?? []).entries()) {
      events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${sent.length}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] });
    }
    events.push({ choices: [{ index: 0, delta: {}, finish_reason: a.calls?.length ? 'tool_calls' : 'stop' }] }, { choices: [], usage: { prompt_tokens: 10, completion_tokens: 5 } });
    const chunks = events.map((e) => new TextEncoder().encode(`data: ${JSON.stringify(e)}\n\n`));
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        init.signal?.addEventListener('abort', () => {
          try {
            controller.error(new DOMException('aborted', 'AbortError'));
          } catch {
            /* already closed */
          }
        });
        if (a.hang) return;
        const go = () => {
          try {
            if (a.breakAfterChunk) {
              controller.enqueue(chunks[0]);
              setTimeout(() => {
                try {
                  controller.error(new TypeError('network error'));
                } catch {
                  /* closed */
                }
              }, 5);
              return;
            }
            for (const c of chunks) controller.enqueue(c);
            controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
            controller.close();
            entry.end = Date.now() - t0;
          } catch {
            /* let go by the app */
          }
        };
        if (a.delayMs) setTimeout(go, a.delayMs);
        else go();
      },
    });
    return new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  });
  return sent;
}

export const until = async (ok: () => boolean) => {
  for (let i = 0; i < 300 && !ok(); i++) await new Promise((r) => setTimeout(r, 5));
};
export const later = (ms = 40) => new Promise((r) => setTimeout(r, ms));
export const count = (log: string[], name: string) => log.filter((l) => l === name).length;

/** An agent on the local engine; `role` says its name to the fake engine. */
export function agent(role: string, extra: Partial<ConstructorParameters<typeof Agent>[0]> = {}) {
  const events: AgentEvent[] = [];
  const a = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ENGINE, projectSummary: () => '', instructions: `ROLE:${role}`, ...extra });
  return { agent: a, events, emit: (e: AgentEvent) => events.push(e) };
}

/** A phone's worth of conversations and outreach on the local engine; the desktop is a stand-in. */
export function world(opts: { answer?: boolean; phone?: { holdsCalls: boolean } } = {}) {
  let index: SessionInfo[] = [];
  const chats = new Map<string, Turn[]>();
  let callers: CallerNote[] = [];
  const project = {
    loadSessions: async () => index,
    saveSessions: async (list: SessionInfo[]) => void (index = list),
    loadSessionChat: async (id: string) => chats.get(id) ?? [],
    saveSessionChat: async (id: string, turns: Turn[]) => void chats.set(id, turns),
    loadCallers: async () => callers,
    saveCallers: async (list: CallerNote[]) => void (callers = list),
  };
  const said: string[] = [];
  let dialN = 0;
  const desktop = {
    say: async (_callId: string, text: string) => void said.push(text),
    finishCall: async () => ({ ok: true, output: {} }),
    callTool: async () => ({ ok: true, output: { recorded: true } }),
    command: async (_c: string, command: string) => {
      if (command === 'call.dial') return { callId: `call_out_${++dialN}`, operationId: `op_${dialN}`, dialsToday: dialN, maxDailyDials: 20 };
      return { accepted: true };
    },
    calendarFree: async () => 'Fri 2 Oct: open 8-5.',
  };
  const messages: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, answer: opts.answer ?? false, calls: true, instructions: '', callInstructions: '' };
  const sessions = new Sessions(
    project as never,
    (extra) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ENGINE, projectSummary: () => '', ...extra }),
    () => messages,
    () => desktop as unknown as Desktop,
    { changed: () => {}, event: () => {} },
    () => '',
  );
  sessions.identity = () => GREEN;
  const saved = new Map<string, Campaign>();
  let dnc: DoNotContact[] = [];
  const outreach = new Outreach({
    store: { loadOutreach: async () => [...saved.values()], saveOutreach: async (c: Campaign) => void saved.set(c.id, JSON.parse(JSON.stringify(c)) as Campaign), loadDoNotContact: async () => dnc, saveDoNotContact: async (l: DoNotContact[]) => void (dnc = l) },
    files: () => new Vfs(),
    desktop: () => desktop as unknown as Desktop,
    phone: () => ({ holdsCalls: opts.phone ? opts.phone.holdsCalls : true, holdsTexts: true, connected: true }),
    line: new PhoneLine(),
    callbacks: () => null,
    screening: async () => null,
    callsToOaiy: async () => true,
    rules: async () => ({ quietStart: 0, quietEnd: 0, maxDailyDials: 20, outboundEnabled: true }),
    sessions: () => sessions.forOutreach(),
    post: () => true,
    report: () => {},
    identity: () => GREEN,
    now: () => new Date(2026, 8, 29, 14, 0).getTime(),
  });
  sessions.outreach = outreach;
  return { sessions, outreach, said };
}

export const CAMPAIGN = {
  kind: 'call',
  name: 'Hedge price answer',
  objective: 'Tell Jane the price for hedge trimming is $25 an hour, and ask whether she is happy to go ahead.',
  openingLine: "Hi {first_name}, it's {receptionist} from {business} about your quote. Have you got a minute?",
  collect: [{ key: 'happy', question: 'Happy to go ahead?', type: 'yes_no' }],
  people: [{ name: 'Jane Smith', number: '0412 345 678', fields: {} }],
};

/** The Front desk's runner, wired as main.ts wires it (whether its call is going on), with the outreach tools it places a call with. */
export function runnerOf(w: ReturnType<typeof world>, extra: Partial<ConstructorParameters<typeof Agent>[0]> = {}) {
  const tools = outreachTools({
    engine: () => w.outreach,
    origin: () => ({ kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' }),
    ready: async () => '',
    screening: async () => null,
    approve: async () => true,
  });
  return agent('runner', { sessionTools: tools, waitingOnCall: () => w.outreach.callLiveFor('front-desk'), ...extra });
}

/** The same runner as main makes it: not told whether its call is going on. */
export function runnerOnMain(w: ReturnType<typeof world>) {
  return runnerOf(w, { waitingOnCall: undefined });
}

export async function settled(sessions: Sessions): Promise<void> {
  for (let i = 0; i < 300 && sessions.busy; i++) await new Promise((r) => setTimeout(r, 10));
  await Promise.all(sessions.list.map((s) => s.speech?.done));
}
