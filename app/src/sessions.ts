/**
 * The project's conversations besides its own chat: one per person who texts
 * or calls the phone (through Aokie and OAIY Desktop). Each has its own agent,
 * with the person's instructions and a way to answer: a text-message tool, or,
 * on a call, its words spoken as it writes them. They take turns: one works at
 * a time, in the order their messages came (a local model answers one request
 * at a time anyway; a caller goes first), and a message for a conversation that
 * is working reaches it at its next step.
 */
import { Agent, type AgentEvent, type AgentOptions, type SessionTool } from './agent/agent';
import { TOOLS } from './agent/tools';
import type { Turn } from './agent/protocol';
import type { Desktop, DesktopEvent } from './desktop/bridge';
import { textCalendarTools } from './desktop/calendarTools';
import type { MessageSettings } from './settings';
import type { OpenProject, SessionInfo } from './vfs/projects';

/** A number that marks a pretend conversation: its replies are never sent. */
export const TEST_NUMBER = 'test';

export interface Session extends SessionInfo {
  agent: Agent;
  /** The run in progress. */
  running: Promise<void> | null;
  controller: AbortController | null;
  /** Messages for its next run (the text messages, or the person's own). */
  waiting: string[];
  /** The live call it is on (a call conversation). */
  callId?: string;
  /** The receptionist brief the phone sent with the call. */
  brief?: string;
  /** What the agent writes, spoken on the call. */
  speech?: Speech;
  /** Its agent is using a tool now (a caller speaking over it does not stop the tool). */
  inTool?: boolean;
  /** A flow's tasks waiting for their answers, each by its prompt (one task a run; a message of the person's answers none). */
  answers?: Array<{ prompt: string; settle: (reply: string, error?: string) => void }>;
}

/** How the app makes an agent for this project, with a conversation's own instructions and tools. */
export type MakeAgent = (extra: Pick<AgentOptions, 'instructions' | 'sessionTools' | 'tools' | 'reasoning' | 'conversation'>) => Agent;

/**
 * What an agent on a call may use besides the call's own tools: a short list,
 * so its prompt stays small and the caller is answered at once (reading and
 * noting things in the project, and the web).
 */
/**
 * What a call's or a text thread's agent may do beyond its own tools (the
 * reply, the calendar, the lookup): read the front desk's files. The person on
 * the other end is a stranger to this computer, so nothing that writes, runs
 * code, fetches from the web or makes pictures and sounds.
 */
export const KNOWLEDGE_TOOLS = new Set(['read_file', 'list_files', 'search_file', 'grep', 'glob', 'file_info', 'view_image']);
/** Where the front desk keeps what the person gives its agents to go by. */
const REFERENCE = 'Your person\'s reference files are the front desk\'s: /brief.md (your direction), /knowledge (what the business wants you to know) and /uploads (documents and pictures they gave you). Read them when a question is one they cover.';
export const CALL_TOOLS = KNOWLEDGE_TOOLS;

export interface SessionHooks {
  /** The list changed: a new conversation, an unread message, one started or finished working. */
  changed: () => void;
  /** A text message came into a conversation (before its agent reads it). */
  arrived?: (session: Session, text: string) => void;
  /** What a conversation's agent is doing (drawn when that conversation is shown). */
  event: (session: Session, event: AgentEvent) => void;
}

const digits = (number: string) => number.replace(/[^\d+]/g, '');
/** The same phone number, with or without its country code ("+61491570006", "0491570006"): the last nine digits agree. */
export function sameNumber(a: string, b: string): boolean {
  const [x, y] = [a.replace(/\D/g, ''), b.replace(/\D/g, '')];
  return x.length >= 8 && y.length >= 8 && x.slice(-9) === y.slice(-9);
}

/** A text message as the conversation's agent reads it. */
export function textMessage(title: string, number: string, body: string): string {
  const who = title && title !== number ? `${title} (${number})` : number;
  return `Text message from ${who}:\n${body}`;
}

/** What a text-message conversation is for, in its agent's instructions. */
export function smsInstructions(title: string, number: string, instructions: string, test: boolean): string {
  const who = title && title !== number ? `${title} (${number})` : number;
  return [
    `This conversation is a text-message thread with ${who}, on the phone of the person you work for.${test ? ' It is a test: your replies are shown, not sent.' : ''}`,
    'Their messages arrive as "Text message from …". Answer them with send_text_message: short plain text (no markdown), in the language they write in. Only what you send with it reaches them; anything else you write is seen only by the person you work for.',
    'A message without that label comes from the person you work for, who may be watching: do what they say (they may tell you what to reply, or ask you to do something first).',
    `${REFERENCE} Use your flows made tools when a message needs one. You cannot change files, browse the web or run code here. There is no need to reply to a message that needs no answer (a thank-you, an emoji).`,
    'To book them in: find a time with calendar_free_times, agree a day and time with them, then request_appointment. It is a request that staff confirm (they are texted when it is): never say it is booked.',
    `The instructions of the person you work for, for text messages:\n${instructions.trim() || '(none)'}`,
  ].join('\n');
}

/**
 * A call's agent writes its reply; this speaks it as it comes, a sentence at a
 * time, in order. Hushed (the caller spoke over it), the rest of that reply is
 * dropped.
 */
export class Speech {
  private buffer = '';
  private chain: Promise<void> = Promise.resolve();
  private hushed = false;
  /** What was said on this call, so a sentence is not said twice. */
  private said = new Set<string>();
  /** Something has been said in this reply. */
  private spoke = false;

  constructor(private readonly say: (text: string) => Promise<void>, private readonly failed: (error: string) => void = () => {}) {}

  /** A new reply: speak again. */
  begin(): void {
    this.hushed = false;
    this.buffer = '';
    this.spoke = false;
  }

  /** A tool is taking a while: say a short line, unless this reply has said something already. */
  hold(line: string): void {
    if (this.hushed || this.spoke) return;
    this.speak(line);
  }

  /** A new call: nothing has been said on it yet. */
  newCall(): void {
    this.said.clear();
  }

  push(delta: string): void {
    if (this.hushed) return;
    this.buffer += delta;
    for (;;) {
      // A sentence ends at . ! ? … (then a space) or a line break; a long run of words at a comma.
      const end = /[.!?\u2026]+["')\]]?(?=\s)|\n+/.exec(this.buffer);
      let at = end ? end.index + end[0].length : -1;
      if (at < 0 && this.buffer.length > 220) {
        const comma = this.buffer.lastIndexOf(', ', 220);
        at = comma > 40 ? comma + 1 : this.buffer.lastIndexOf(' ', 220);
      }
      if (at <= 0) return;
      this.speak(this.buffer.slice(0, at));
      this.buffer = this.buffer.slice(at);
    }
  }

  /** The reply ended: speak what is left. */
  flush(): void {
    if (!this.hushed) this.speak(this.buffer);
    this.buffer = '';
  }

  hush(): void {
    this.hushed = true;
    this.buffer = '';
  }

  /** Everything queued has been sent to be spoken. */
  get done(): Promise<void> {
    return this.chain;
  }

  private speak(text: string): void {
    const clean = spoken(text);
    if (!clean) return;
    // A sentence of a few words already said on this call is not said again (a model repeats itself;
    // a short "Sure!" or "Okay." may come again).
    const key = clean.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
    if (key.split(' ').length >= 4) {
      if (this.said.has(key)) return;
      this.said.add(key);
    }
    this.spoke = true;
    this.chain = this.chain.then(() => (this.hushed ? undefined : this.say(clean))).catch((e: unknown) => this.failed((e as Error).message));
  }
}

/** Text as it can be said: no markdown, links as their words, no emoji. */
export function spoken(text: string): string {
  return text
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[*_`#>|~]+/g, '')
    .replace(/\p{Extended_Pictographic}/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/** What a call conversation is for, in its agent's instructions. */
export function callInstructions(title: string, number: string, brief: string, instructions: string): string {
  const who = title && title !== number ? `${title} (${number})` : number || 'the caller';
  return [
    `This conversation is a live phone call with ${who}, on the phone of the person you work for. Everything you write is spoken aloud to the caller as you write it, so write only what you would say: one or two short sentences, plain words, no markdown, lists, emoji or links. Then stop, and let them answer.`,
    'Their words arrive as "Caller: …", transcribed from speech (allow for a misheard word). A message without that label comes from the person you work for, who may be watching: do what they say.',
    'Your call tools: request_appointment (a booking request for staff to confirm; never say it is booked or confirmed), lookup_business_data (a question about the business\'s records or calendar), end_call (a short goodbye, then the call ends; use it when the caller is done). Your other tools work too.',
    REFERENCE,
    'To look something up (a tool, a file), do it in the same reply as a few words: say "Let me check." and make the call at once, then answer from what it returned. Never say you will check without doing it: the caller hears you and waits. If the caller speaks while you check, their words reach you with the result: answer both.',
    'Say only what you know: from these instructions, the brief, or what a tool returned. Never make up availability, times, prices or bookings. If you cannot check, say so, and offer to take their preferred time as a request for staff to confirm.',
    'Never repeat something you have already said on this call. When the caller says goodbye or is done, call end_call with a short goodbye, and write nothing else.',
    brief.trim() ? `The receptionist brief:\n${brief.trim()}` : '',
    `The instructions of the person you work for, for calls:\n${instructions.trim() || '(none)'}`,
  ].filter(Boolean).join('\n');
}

/** How long a tool may run on a call before a short line is said, and the line. */
const HOLD_AFTER_MS = 2_000;
const HOLD_LINE = 'One moment, let me check.';

/** What the call's agent is told when the business's records cannot be checked. */
export const LOOKUP_UNAVAILABLE =
  "The business's records could not be checked just now (no business lookup is set up on OAIY Desktop, or it failed). Do not guess times, availability, prices or bookings: tell the caller you can't check right now, and offer to take their preferred time as a request for staff to confirm.";

/**
 * What a new call keeps of the caller's earlier calls: the last few things said, without the
 * automatic notes, tool calls, or any line the agent said more than once. A model copies what it
 * said before, so a repeated line would be said on every call.
 */
export function earlierWords(turns: Turn[], keep = 6): Turn[] {
  const words: Array<{ role: 'user' | 'assistant'; text: string }> = [];
  for (const t of turns) {
    if (t.role === 'user' && !t.automatic && t.text.startsWith('Caller:')) words.push({ role: 'user', text: t.text });
    else if (t.role === 'assistant' && !t.calls.length && t.text.trim()) words.push({ role: 'assistant', text: t.text });
  }
  const times = new Map<string, number>();
  for (const t of words) if (t.role === 'assistant') times.set(t.text.trim(), (times.get(t.text.trim()) ?? 0) + 1);
  return words
    .filter((t) => t.role === 'user' || times.get(t.text.trim()) === 1)
    .slice(-keep)
    .map((t): Turn => (t.role === 'user' ? { role: 'user', text: t.text } : { role: 'assistant', text: t.text, calls: [] }));
}

/** A conversation's words, as the runner reads them: who said what, and what was done. */
export function conversationText(turns: Turn[], last = 40): string {
  const lines: string[] = [];
  for (const t of turns) {
    if (t.role === 'user') lines.push(t.text.startsWith('[OAIY]') ? `(${t.text.replace(/^\[OAIY\]\s*/, '')})` : t.text);
    else if (t.role === 'assistant') {
      if (t.text.trim()) lines.push(`Agent: ${t.text.trim()}`);
      for (const c of t.calls) lines.push(`(used ${c.name}${c.name === 'send_text_message' && typeof c.input.body === 'string' ? `: "${c.input.body}"` : ''})`);
    }
  }
  const shown = lines.slice(-last).join('\n');
  return shown.length > 8000 ? `…${shown.slice(-8000)}` : shown;
}

/**
 * The runner's view of its sub-agents (the front desk's main agent): the
 * phone's conversations, and what was said and done in one.
 */
export function phoneConversationsTool(sessions: () => Sessions | null): SessionTool {
  return {
    spec: {
      name: 'phone_conversations',
      description:
        "The phone's conversations, each answered by a sub-agent of yours: calls, text-message threads and flows' tasks. With no id: the list, newest first. With an id: what was said and done in it (its last `last` lines, 40 by default).",
      parameters: { type: 'object', properties: { id: { type: 'string', description: "A conversation's id, from the list" }, last: { type: 'number' } } },
    },
    run: async (input) => {
      const all = sessions()?.list ?? [];
      const id = typeof input.id === 'string' ? input.id.trim() : '';
      if (id) {
        const s = all.find((x) => x.id === id);
        if (!s) return `No conversation ${id}. Call phone_conversations with no id for the list.`;
        const last = typeof input.last === 'number' && input.last > 0 ? Math.min(200, Math.floor(input.last)) : 40;
        return `${s.kind === 'call' ? 'Calls' : s.kind === 'task' ? "Tasks from the flow" : 'Text messages'} with ${s.title}${s.title !== s.key ? ` (${s.key})` : ''}${s.callId ? ', on a call now' : ''}:\n${conversationText(s.agent.turns, last) || '(nothing yet)'}`;
      }
      if (!all.length) return 'No calls, texts or flow tasks yet.';
      const when = (ms: number) => new Date(ms).toLocaleString('en-AU', { weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });
      return all
        .map((s) => `${s.id}: ${s.kind === 'call' ? 'call' : s.kind === 'task' ? 'flow task' : 'texts'} with ${s.title}${s.title !== s.key ? ` (${s.key})` : ''}, last ${when(s.lastAt)}${s.running ? ', working now' : ''}${s.callId ? ', on a call now' : ''}`)
        .join('\n');
    },
  };
}

/** What a flow's tasks are about: the flow waits for the agent's last words as its output. */
export function taskInstructions(flow: string): string {
  return `This conversation holds the tasks your person's flow "${flow}" gives you (an "Ask the agent" node in it). Each message is one task. Do it with your tools, then end with the result itself: your last reply is handed back to the flow as its output, so give only what the flow asked for (no greeting, no offer of more help). If you cannot do it, say why in one sentence. Your files here are the front desk's (what the business wants its phone agent to know, in /knowledge), not a project of the person's.`;
}

export class Sessions {
  list: Session[] = [];
  /** Conversations with messages waiting, in the order they came. */
  /** Conversations waiting to run, one at a time per lane: calls in their own (a caller never waits behind a text or a flow's task), the rest in another. */
  private queue: Session[] = [];
  private callQueue: Session[] = [];
  private pumping = false;
  private pumpingCalls = false;

  constructor(
    private readonly project: OpenProject,
    private readonly makeAgent: MakeAgent,
    private readonly settings: () => MessageSettings,
    private readonly desktop: () => Desktop | null,
    private readonly hooks: SessionHooks,
    /** The direction every conversation takes, from its runner (the front desk's brief): read afresh for each reply. */
    private readonly direction: () => string = () => '',
  ) {}

  /** A conversation's instructions, with the runner's direction after them. */
  private directed(instructions: string): string {
    const said = this.direction().trim();
    if (!said) return instructions;
    const brief = said.length > 4000 ? `${said.slice(0, 4000)}\n[the rest of the brief is cut]` : said;
    return `${instructions}\n\nYour direction, from the front desk's brief (kept by the main agent your person talks to). Go by it before anything a caller or texter asks, and where a tool, a file or the calendar says otherwise, the brief wins:\n${brief}`;
  }

  /** The project's saved conversations. */
  async load(): Promise<void> {
    for (const info of await this.project.loadSessions()) {
      const session = this.create(info);
      session.agent.turns = await this.project.loadSessionChat(info.id);
      this.list.push(session);
    }
    this.sort();
  }

  get(id: string): Session | undefined {
    return this.list.find((s) => s.id === id);
  }

  /** Whether any conversation is working (or waiting to). */
  get busy(): boolean {
    return this.list.some((s) => s.running) || this.queue.length > 0 || this.callQueue.length > 0;
  }

  private create(info: SessionInfo): Session {
    const session = { ...info, running: null, controller: null, waiting: [] } as unknown as Session;
    if (info.kind === 'call') {
      session.agent = this.makeAgent({
        instructions: () => this.directed(callInstructions(session.title, session.key, session.brief ?? '', this.settings().callInstructions)),
        sessionTools: this.callTools(session),
        tools: TOOLS.filter((t) => CALL_TOOLS.has(t.name)),
        // Answer at once: no thinking first.
        reasoning: 'none',
        conversation: true,
      });
      session.speech = new Speech(
        async (text) => {
          const desktop = this.desktop();
          if (desktop && session.callId) await desktop.say(session.callId, text);
        },
        (error) => this.hooks.event(session, { type: 'status', message: `Could not speak on the call: ${error}` }),
      );
      return session;
    }
    if (info.kind === 'task') {
      session.answers = [];
      session.agent = this.makeAgent({ instructions: () => this.directed(taskInstructions(session.key)) });
      return session;
    }
    const test = info.key === TEST_NUMBER;
    session.agent = this.makeAgent({
      instructions: () => this.directed(smsInstructions(session.title, session.key, this.settings().instructions, test)),
      // A texter reaches the front desk's files, to read (see KNOWLEDGE_TOOLS).
      tools: TOOLS.filter((t) => KNOWLEDGE_TOOLS.has(t.name)),
      // A pretend thread does not put requests in the real calendar.
      sessionTools: [this.replyTool(session, test), ...(test ? [] : textCalendarTools(this.desktop, session.key, () => session.title))],
      conversation: true,
    });
    return session;
  }

  /** A call's own tools: through the desktop to Aokie. */
  private callTools(session: Session): SessionTool[] {
    const live = () => {
      const desktop = this.desktop();
      if (!desktop) throw new Error('OAIY Desktop is not connected');
      if (!session.callId) throw new Error('the call has ended');
      return { desktop, callId: session.callId };
    };
    const outcome = (r: { ok: boolean; output: unknown }) => JSON.stringify(r.output ?? {}, null, 1) + (r.ok ? '' : '\n(The phone refused it.)');
    return [
      {
        spec: {
          name: 'end_call',
          description: 'Say a short goodbye, then hang up. Use it when the caller is done, not before.',
          parameters: { type: 'object', properties: { goodbye: { type: 'string', description: 'The goodbye, one short sentence' } } },
        },
        run: async (input) => {
          const { desktop, callId } = live();
          session.speech?.hush();
          const r = await desktop.finishCall(callId, String(input.goodbye ?? ''));
          return r.ok ? 'The goodbye is being said, then the call ends. Write nothing more.' : `Could not end the call: ${outcome(r)}`;
        },
      },
      {
        spec: {
          name: 'request_appointment',
          description: 'Record an appointment REQUEST for staff to confirm (never a confirmed booking), once the caller has clearly asked for a slot and agreed to it. agreementPhrase is the caller\'s own words agreeing, as they said them.',
          parameters: {
            type: 'object',
            required: ['callerName', 'service', 'date', 'time', 'agreementPhrase'],
            properties: {
              callerName: { type: 'string', description: 'The name the caller gave on this call' },
              service: { type: 'string' },
              date: { type: 'string', description: 'YYYY-MM-DD' },
              time: { type: 'string', description: 'HH:MM, 24-hour' },
              agreementPhrase: { type: 'string', description: 'The caller\'s words agreeing to this slot, as said' },
            },
          },
        },
        run: async (input) => {
          const { desktop, callId } = live();
          return outcome(await desktop.callTool(callId, 'request_appointment', input));
        },
      },
      {
        spec: {
          name: 'lookup_business_data',
          description: 'Ask the business\'s records a question (an existing appointment, availability, a customer\'s details): answered by the business lookup flow.',
          parameters: { type: 'object', required: ['question'], properties: { question: { type: 'string' } } },
        },
        run: async (input) => {
          const { desktop, callId } = live();
          const r = await desktop.callTool(callId, 'lookup_business_data', { question: String(input.question ?? '') });
          // No records to ask (no business-lookup flow on the desktop, or it failed): say so plainly, so nothing is made up.
          if (JSON.stringify(r.output ?? '').includes('LOOKUP UNAVAILABLE')) return LOOKUP_UNAVAILABLE;
          return outcome(r);
        },
      },
    ];
  }

  /**
   * An event from the desktop's calls: a call begins (its conversation, with
   * the caller's history), the caller speaks (the agent answers, before
   * anyone else waiting), the caller speaks over it (the rest of that reply is
   * dropped), and the call ends.
   */
  async callEvent(event: Record<string, unknown>): Promise<Session | null> {
    const callId = typeof event.callId === 'string' ? event.callId : '';
    const type = String(event.type ?? '');
    if (!callId) return null;
    let session = this.list.find((s) => s.callId === callId) ?? null;
    if (type === 'call.started' || (!session && type === 'call.caller')) {
      const from = String(event.from ?? '') || callId;
      session = await this.conversationWith(from, String(event.name ?? ''), 'call');
      session.callId = callId;
      if (typeof event.instructions === 'string') session.brief = event.instructions;
      session.lastAt = Date.now();
      session.unread++;
      // A new call starts afresh, with only the last few words of the earlier ones.
      if (type === 'call.started' && !session.running) session.agent.turns = earlierWords(session.agent.turns);
      if (type === 'call.started') session.speech?.newCall();
      const greeting = typeof event.greeting === 'string' && event.greeting.trim() ? ` You greeted them: "${event.greeting.trim()}"` : '';
      session.agent.turns.push({ role: 'user', text: `[OAIY] 📞 A call from ${session.title}${session.title !== session.key ? ` (${session.key})` : ''} began.${greeting}`, automatic: true });
      await this.save(session);
      await this.saveIndex();
      this.hooks.changed();
      // The model reads the call's prompt while the greeting plays: its first answer comes sooner.
      if (type === 'call.started' && !session.running) void session.agent.warm();
      if (type === 'call.started') return session;
    }
    if (!session) return null;
    switch (type) {
      case 'call.caller': {
        const text = `Caller: ${String(event.text ?? '').trim()}`;
        session.lastAt = Date.now();
        this.hooks.arrived?.(session, text);
        this.deliver(session, text, true);
        break;
      }
      case 'call.interrupted':
        // The words stop. A tool at work goes on: what the caller says reaches the agent with its result.
        session.speech?.hush();
        if (!session.inTool) session.controller?.abort();
        break;
      case 'call.ended':
        session.speech?.hush();
        this.stop(session);
        session.callId = undefined;
        session.agent.turns.push({ role: 'user', text: '[OAIY] 📞 The call ended.', automatic: true });
        await this.save(session);
        this.hooks.changed();
        break;
    }
    return session;
  }

  /** send_text_message: through the desktop to Aokie's sms.send (a test conversation's replies are only shown). */
  private replyTool(session: Session, test: boolean): SessionTool {
    return {
      spec: {
        name: 'send_text_message',
        description: `Send a text message (SMS) to ${session.title || session.key} from the phone: your reply in this conversation. Short plain text, no markdown.`,
        parameters: { type: 'object', required: ['body'], properties: { body: { type: 'string', description: 'The message, as they will read it' } } },
      },
      run: async (input, signal) => {
        const body = String(input.body ?? '').trim();
        if (!body) throw new Error('body is empty: write the message to send');
        if (body.length > 1600) throw new Error(`the message is ${body.length} characters: keep it under 1600 (a text message is read on a phone)`);
        if (test) return `Not sent (a test conversation): "${body}"`;
        const desktop = this.desktop();
        if (!desktop) throw new Error('OAIY Desktop is not connected, so the phone cannot send it. Tell the person you work for.');
        const key = `oaiy:sms:${session.id}:${crypto.randomUUID()}`;
        const sent = (await desktop.command('aokie', 'sms.send', { to: session.key, body }, key, signal)) as Record<string, unknown> | null;
        const id = sent && typeof sent.messageId === 'string' ? ` (message ${sent.messageId})` : '';
        return `Sent to ${session.title || session.key}${id}.`;
      },
    };
  }

  /** The conversation with `number` (texts or calls), made when it is the first from them. */
  async conversationWith(number: string, name: string, kind: 'sms' | 'call' = 'sms'): Promise<Session> {
    const key = number === TEST_NUMBER ? TEST_NUMBER : digits(number) || number;
    // No name given (a call's caller id has none): the name another conversation with the same number has.
    const known = name || this.list.find((s) => s.title !== s.key && sameNumber(s.key, key))?.title || '';
    const existing = this.list.find((s) => s.kind === kind && s.key === key);
    if (existing) {
      if (known && existing.title === existing.key) existing.title = known;
      return existing;
    }
    const session = this.create({ id: `${kind}-${key.replace(/\W/g, '') || key}`, kind, key, title: known || key, lastAt: Date.now(), unread: 0 });
    this.list.push(session);
    await this.saveIndex();
    return session;
  }

  /** An event from the desktop: a text message becomes (or continues) a conversation. */
  async desktopEvent(event: DesktopEvent): Promise<Session | null> {
    if (event.name !== 'aokie.sms.received') return null;
    const from = String(event.data.from ?? '');
    const body = String(event.data.body ?? '');
    if (!from || !body) return null;
    // A text the phone delivers again (it does, when it reconnects) is not a new one.
    const handle = String(event.data.handle ?? '');
    if (handle) {
      const existing = this.list.find((s) => s.kind === 'sms' && s.key === digits(from));
      if (existing?.handles?.includes(handle)) return null;
    }
    const session = await this.textArrived(from, String(event.data.name ?? ''), body);
    if (handle) {
      session.handles = [...(session.handles ?? []), handle].slice(-100);
      await this.saveIndex();
    }
    return session;
  }

  /** A text message from `number`: shown in its conversation, and answered when answering is on. */
  async textArrived(number: string, name: string, body: string): Promise<Session> {
    const session = await this.conversationWith(number, name);
    session.lastAt = Date.now();
    session.unread++;
    const text = textMessage(session.title, session.key, body);
    this.hooks.arrived?.(session, text);
    // A pretend text is always answered: trying the agent is what it is for.
    if (this.settings().answer || session.key === TEST_NUMBER) this.deliver(session, text);
    else {
      // Kept, not answered: it is there when the person looks, or answers it themselves.
      session.agent.turns.push({ role: 'user', text });
      await this.save(session);
    }
    this.sort();
    await this.saveIndex();
    this.hooks.changed();
    return session;
  }

  /**
   * Answering was turned on: the texts that came while it was off, and still
   * wait for an answer (the last message of their conversation, from the last
   * hour), are answered now. Older ones are left: a late reply to an old text
   * surprises more than it helps.
   */
  answerWaiting(within = 60 * 60_000): number {
    let n = 0;
    for (const session of this.list) {
      if (session.running || this.queue.includes(session) || Date.now() - session.lastAt > within) continue;
      const last = session.agent.turns.at(-1);
      if (last?.role !== 'user' || !last.text.startsWith('Text message from ')) continue;
      session.agent.turns.pop();
      this.deliver(session, last.text);
      n++;
    }
    return n;
  }

  /**
   * A flow's task: done in that flow's conversation, one task a run, and
   * answered with what the agent said last.
   */
  task(from: string, text: string): Promise<string> {
    const key = from.trim() || 'a flow';
    let session = this.list.find((s) => s.kind === 'task' && s.key === key);
    if (!session) {
      session = this.create({ id: `task-${key.replace(/\W+/g, '-').toLowerCase().slice(0, 40) || 'flow'}-${Date.now().toString(36)}`, kind: 'task', key, title: key, lastAt: Date.now(), unread: 0 });
      this.list.push(session);
      void this.saveIndex();
    }
    const s = session;
    s.answers ??= [];
    s.lastAt = Date.now();
    s.unread++;
    this.sort();
    this.hooks.changed();
    // Queued at once, so tasks run in the order they came.
    const prompt = `[OAIY] Your flow "${key}" asks: ${text}`;
    return new Promise<string>((resolve, reject) => {
      s.answers!.push({ prompt, settle: (reply, error) => (error ? reject(new Error(error)) : resolve(reply)) });
      this.deliver(s, prompt);
    });
  }

  /** The person's own message in a conversation: it goes first. */
  say(session: Session, text: string): void {
    this.deliver(session, text, true);
  }

  /** A message for a conversation: to its running agent, or its next run. */
  private deliver(session: Session, text: string, first = false): void {
    // A flow's task is its own run (its answer is what that run says), never added to another.
    if (session.kind !== 'task' && session.running && session.agent.interject(text)) return;
    session.waiting.push(text);
    const lane = session.kind === 'call' ? this.callQueue : this.queue;
    if (!lane.includes(session)) {
      if (first) lane.unshift(session);
      else lane.push(session);
    }
    void this.pump(session.kind === 'call');
  }

  /** Run a lane's waiting conversations, one at a time. */
  private async pump(calls = false): Promise<void> {
    if (calls ? this.pumpingCalls : this.pumping) return;
    if (calls) this.pumpingCalls = true;
    else this.pumping = true;
    const lane = calls ? this.callQueue : this.queue;
    try {
      while (lane.length) {
        const session = lane.shift()!;
        const prompt = session.kind === 'task' ? session.waiting.shift() ?? '' : session.waiting.splice(0).join('\n\n');
        if (session.kind === 'task' && session.waiting.length) lane.push(session);
        if (prompt) await this.run(session, prompt);
      }
    } finally {
      if (calls) this.pumpingCalls = false;
      else this.pumping = false;
    }
  }

  private async run(session: Session, prompt: string): Promise<void> {
    const controller = new AbortController();
    session.controller = controller;
    let finish!: () => void;
    session.running = new Promise<void>((resolve) => (finish = resolve));
    this.hooks.changed();
    session.speech?.begin();
    // What the run says last (a flow's task is answered with it), or why it failed.
    let said = '';
    let failed = '';
    // On a call: a tool that takes a while gets a short line said, so the caller is not left in silence.
    let holding: ReturnType<typeof setTimeout> | null = null;
    const stopHolding = () => {
      if (holding) clearTimeout(holding);
      holding = null;
    };
    try {
      await session.agent.run(prompt, (event) => {
        if (event.type === 'done') said = event.text;
        if (event.type === 'error') failed = event.message;
        if (event.type === 'tool_call') {
          session.inTool = true;
          stopHolding();
          if (session.speech) holding = setTimeout(() => session.speech?.hold(HOLD_LINE), HOLD_AFTER_MS);
        }
        if (event.type === 'tool_result') {
          session.inTool = false;
          stopHolding();
          // What it says next is heard, even if the caller spoke over the words before the tool.
          session.speech?.begin();
        }
        if (session.speech && event.type === 'text') session.speech.push(event.delta);
        // A reply ends (a tool is called, or the model's turn is over): what it said is complete.
        if (session.speech && (event.type === 'tool_call' || event.type === 'usage')) session.speech.flush();
        this.hooks.event(session, event);
      }, controller.signal);
      session.speech?.flush();
    } catch (error) {
      failed = (error as Error).message;
      this.hooks.event(session, { type: 'error', message: (error as Error).message });
    } finally {
      stopHolding();
      session.inTool = false;
      // The task this run was (none, when it was a message of the person's).
      const task = session.answers?.findIndex((a) => a.prompt === prompt) ?? -1;
      if (task >= 0) session.answers!.splice(task, 1)[0].settle(said.trim(), said.trim() ? undefined : failed || 'the agent finished without an answer');
      session.running = null;
      session.controller = null;
      finish();
      session.lastAt = Date.now();
      await this.save(session);
      await this.saveIndex();
      // Messages that came as it finished: its next turn.
      const unread = session.agent.takeUnread();
      if (unread.length) this.deliver(session, unread.join('\n\n'));
      this.hooks.changed();
    }
  }

  stop(session: Session): void {
    session.controller?.abort();
    session.waiting = [];
    this.queue = this.queue.filter((s) => s !== session);
    this.callQueue = this.callQueue.filter((s) => s !== session);
  }

  stopAll(): void {
    for (const s of this.list) this.stop(s);
  }

  /** The person looked at it. */
  seen(session: Session): void {
    if (!session.unread) return;
    session.unread = 0;
    void this.saveIndex();
    this.hooks.changed();
  }

  async remove(session: Session): Promise<void> {
    this.stop(session);
    this.list = this.list.filter((s) => s !== session);
    await this.project.saveSessionChat(session.id, []);
    await this.saveIndex();
    this.hooks.changed();
  }

  async save(session: Session): Promise<void> {
    await this.project.saveSessionChat(session.id, session.agent.savedTurns());
  }

  private sort(): void {
    this.list.sort((a, b) => b.lastAt - a.lastAt);
  }

  private async saveIndex(): Promise<void> {
    await this.project.saveSessions(this.list.map(({ id, kind, key, title, lastAt, unread, handles }) => ({ id, kind, key, title, lastAt, unread, ...(handles?.length ? { handles } : {}) })));
  }
}

/**
 * Follows the desktop's events: every two seconds, what came after the last
 * one seen. The first look only notes where the stream is, so events from
 * before this page opened are never acted on again.
 */
export class DesktopEvents {
  private since: number | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  /** The last problem reaching the desktop (cleared when it answers). */
  problem = '';

  constructor(
    private readonly desktop: () => Desktop | null,
    private readonly handle: (event: DesktopEvent) => void | Promise<void>,
    private readonly status: (problem: string) => void = () => {},
    private readonly every = 2000,
  ) {}

  start(): void {
    this.stopped = false;
    void this.tick();
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  /** Look now (a test, or right after pairing). */
  async tick(): Promise<void> {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const desktop = this.desktop();
    let wait = this.every;
    if (desktop) {
      try {
        if (this.since === null) {
          // The first look: where the stream is now, paging to its end (the desktop keeps the last 500).
          let since = 0;
          for (;;) {
            const page = await desktop.events(since);
            if (!page.events.length || page.next <= since) break;
            since = page.next;
          }
          this.since = since;
        } else {
          const { events, next } = await desktop.events(this.since);
          this.since = next;
          for (const e of events) await this.handle(e);
        }
        if (this.problem) {
          this.problem = '';
          this.status('');
        }
      } catch (error) {
        const message = (error as Error).message || 'OAIY Desktop did not answer';
        if (message !== this.problem) {
          this.problem = message;
          this.status(message);
        }
        wait = this.every * 5;
      }
    }
    if (!this.stopped) this.timer = setTimeout(() => void this.tick(), wait);
  }
}

/** The turns of a conversation, for drawing it. */
export function sessionTurns(session: Session): Turn[] {
  return session.agent.turns;
}
