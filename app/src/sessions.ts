/**
 * The project's conversations besides its own chat: one per person who texts
 * the phone (through Aokie and OAIY Desktop). Each has its own agent, with the
 * person's instructions for text messages and a tool to reply. They take turns:
 * one works at a time, in the order their messages came (a local model answers
 * one request at a time anyway), and a message for a conversation that is
 * working reaches it at its next step.
 */
import { Agent, type AgentEvent, type AgentOptions, type SessionTool } from './agent/agent';
import type { Turn } from './agent/protocol';
import type { Desktop, DesktopEvent } from './desktop/bridge';
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
}

/** How the app makes an agent for this project, with a conversation's own instructions and tools. */
export type MakeAgent = (extra: Pick<AgentOptions, 'instructions' | 'sessionTools'>) => Agent;

export interface SessionHooks {
  /** The list changed: a new conversation, an unread message, one started or finished working. */
  changed: () => void;
  /** A text message came into a conversation (before its agent reads it). */
  arrived?: (session: Session, text: string) => void;
  /** What a conversation's agent is doing (drawn when that conversation is shown). */
  event: (session: Session, event: AgentEvent) => void;
}

const digits = (number: string) => number.replace(/[^\d+]/g, '');

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
    'Use your other tools (the project\'s files, the web, flows) when a message needs it. There is no need to reply to a message that needs no answer (a thank-you, an emoji).',
    `The instructions of the person you work for, for text messages:\n${instructions.trim() || '(none)'}`,
  ].join('\n');
}

export class Sessions {
  list: Session[] = [];
  /** Conversations with messages waiting, in the order they came. */
  private queue: Session[] = [];
  private pumping = false;

  constructor(
    private readonly project: OpenProject,
    private readonly makeAgent: MakeAgent,
    private readonly settings: () => MessageSettings,
    private readonly desktop: () => Desktop | null,
    private readonly hooks: SessionHooks,
  ) {}

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
    return this.list.some((s) => s.running) || this.queue.length > 0;
  }

  private create(info: SessionInfo): Session {
    const session = { ...info, running: null, controller: null, waiting: [] } as unknown as Session;
    const test = info.key === TEST_NUMBER;
    session.agent = this.makeAgent({
      instructions: () => smsInstructions(session.title, session.key, this.settings().instructions, test),
      sessionTools: [this.replyTool(session, test)],
    });
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

  /** The conversation with `number`, made when it is the first message from them. */
  async conversationWith(number: string, name: string): Promise<Session> {
    const key = number === TEST_NUMBER ? TEST_NUMBER : digits(number);
    const existing = this.list.find((s) => s.kind === 'sms' && s.key === key);
    if (existing) {
      if (name && existing.title === existing.key) existing.title = name;
      return existing;
    }
    const session = this.create({ id: `sms-${key.replace(/\D/g, '') || key}`, kind: 'sms', key, title: name || key, lastAt: Date.now(), unread: 0 });
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
    return this.textArrived(from, String(event.data.name ?? ''), body);
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

  /** The person's own message in a conversation: it goes first. */
  say(session: Session, text: string): void {
    this.deliver(session, text, true);
  }

  /** A message for a conversation: to its running agent, or its next run. */
  private deliver(session: Session, text: string, first = false): void {
    if (session.running && session.agent.interject(text)) return;
    session.waiting.push(text);
    if (!this.queue.includes(session)) {
      if (first) this.queue.unshift(session);
      else this.queue.push(session);
    }
    void this.pump();
  }

  /** Run the waiting conversations, one at a time. */
  private async pump(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    try {
      while (this.queue.length) {
        const session = this.queue.shift()!;
        const prompt = session.waiting.splice(0).join('\n\n');
        if (prompt) await this.run(session, prompt);
      }
    } finally {
      this.pumping = false;
    }
  }

  private async run(session: Session, prompt: string): Promise<void> {
    const controller = new AbortController();
    session.controller = controller;
    let finish!: () => void;
    session.running = new Promise<void>((resolve) => (finish = resolve));
    this.hooks.changed();
    try {
      await session.agent.run(prompt, (event) => this.hooks.event(session, event), controller.signal);
    } catch (error) {
      this.hooks.event(session, { type: 'error', message: (error as Error).message });
    } finally {
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
    await this.project.saveSessions(this.list.map(({ id, kind, key, title, lastAt, unread }) => ({ id, kind, key, title, lastAt, unread })));
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
