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
import type { CallerNote, OpenProject, SessionInfo } from './vfs/projects';

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
  /** This call's agent was reminded that a booking it promised was not requested (once a call). */
  bookingNudged?: boolean;
  /** This call's sentences as they played (ms from the call's start): what the caller heard, and when. */
  played?: Array<{ text: string; startMs: number; endMs: number }>;
  /** When the caller last cut this call's agent off (ms from the call's start). */
  cutAtMs?: number;
  /** Acknowledgements ("mm-hmm") said over the agent: it talked on, and reads them with the caller's next words. */
  aside?: string[];
  /** The caller is speaking now (their words are not in yet). */
  callerSpeaking?: boolean;
  speakingTimer?: ReturnType<typeof setTimeout>;
  /** Answers that came in while the caller spoke: given to the agent with their words. */
  held?: string[];
  /** A flow's tasks waiting for their answers, each by its prompt (one task a run; a message of the person's answers none). */
  answers?: Array<{ prompt: string; settle: (reply: string, error?: string) => void }>;
}

/** The most facts kept about one person (the oldest go first). */
const MAX_FACTS = 30;
/** The most turns a conversation keeps (a caller's calls add up): the oldest calls go first. */
const MAX_KEPT_TURNS = 600;

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
  /** A run ended (finished, failed or cut off): what comes next is a new reply. */
  finished?: (session: Session) => void;
  /** What is known about a person changed (the phone greets them by name). */
  named?: (note: CallerNote) => void;
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

/** Today, as a person says it ("Monday 28 September 2026"): so "next Tuesday" can be worked out. The date only: a prompt that changed each minute would miss the model's cache. */
export function today(now = new Date()): string {
  return now.toLocaleDateString('en-AU', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }).replace(',', '');
}

/** When something happened, short ("Mon 28 Sep, 11:05 pm"). */
export function whenSaid(ms: number): string {
  return new Date(ms).toLocaleString('en-AU', { weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });
}

/** What a call's or a text thread's agent is told about the person, from their note. */
export function knownText(note: CallerNote | undefined): string {
  const tools = 'Save their name when they tell you it, and anything worth knowing next time, with remember; earlier_conversations finds what was said in their earlier calls and texts.';
  if (!note || (!note.name && !note.facts.length)) return `Nothing is saved about them yet. ${tools}`;
  return [
    'What you know about them, saved across their calls and texts (by you, or by the main agent your person talks to):',
    `Name: ${note.name || 'not known yet'}`,
    ...note.facts.map((f) => `- ${f}`),
    tools,
  ].join('\n');
}

/** What a text-message conversation is for, in its agent's instructions. */
export function smsInstructions(title: string, number: string, instructions: string, test: boolean, known = '', now = new Date()): string {
  const who = title && title !== number ? `${title} (${number})` : number;
  return [
    `This conversation is a text-message thread with ${who}, on the phone of the person you work for.${test ? ' It is a test: your replies are shown, not sent.' : ''} Today is ${today(now)}.`,
    'Their messages arrive as "Text message from …". Answer them with send_text_message: short plain text (no markdown), in the language they write in. Only what you send with it reaches them; anything else you write is seen only by the person you work for.',
    'A message without that label comes from the person you work for, who may be watching: do what they say (they may tell you what to reply, or ask you to do something first).',
    `${REFERENCE} Use your flows made tools when a message needs one. You cannot change files, browse the web or run code here. There is no need to reply to a message that needs no answer (a thank-you, an emoji).`,
    'To book them in: find a time with calendar_free_times, agree a day and time with them, then request_appointment. It is a request that staff confirm (they are texted when it is): never say it is booked.',
    'A message "[OAIY] A note from the runner" is your person\'s direction, passed on by the main agent they talk to: go by it, without quoting it.',
    known,
    `The instructions of the person you work for, for text messages:\n${instructions.trim() || '(none)'}`,
  ].filter(Boolean).join('\n');
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
  /** This reply's sentences held back as said before: said after all when the reply has nothing new (the caller is never met with silence). */
  private repeated: string[] = [];
  /** What this reply has said so far (what the caller heard of it, when they spoke over it). */
  reply: string[] = [];
  /** A filler word ("Sure!") may start this reply: not after a tool, and only once. */
  private fillerOk = true;

  constructor(private readonly say: (text: string) => Promise<void>, private readonly failed: (error: string) => void = () => {}) {}

  /** A new reply: speak again. */
  begin(afterTool = false): void {
    this.hushed = false;
    this.buffer = '';
    this.spoke = false;
    this.repeated = [];
    this.reply = [];
    this.fillerOk = !afterTool;
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
    if (!this.hushed) {
      this.speak(this.buffer);
      // Everything it said was said before: better said again than nothing at all.
      if (!this.spoke) for (const line of this.repeated) this.enqueue(line);
    }
    this.buffer = '';
    this.repeated = [];
  }

  hush(): void {
    this.hushed = true;
    this.buffer = '';
  }

  /** Sentences the caller never heard (cut off before they played): when they come again, they are said. */
  unsay(lines: string[]): void {
    for (const line of lines) this.said.delete(sayKey(line));
  }

  /** Everything queued has been sent to be spoken. */
  get done(): Promise<void> {
    return this.chain;
  }

  private speak(text: string): void {
    const clean = spoken(text);
    if (!clean) return;
    // A filler word is said only to open a reply: a run of them ("Sure thing! Happy to!") is noise.
    if (isFiller(clean)) {
      const ok = this.fillerOk;
      this.fillerOk = false;
      if (!ok) return;
    } else this.fillerOk = false;
    // A sentence of a few words already said on this call is not said again (a model repeats itself;
    // a short "Sure!" or "Okay." may come again).
    const key = sayKey(clean);
    if (key.split(' ').length >= 4) {
      if (this.said.has(key)) {
        this.repeated.push(clean);
        return;
      }
      this.said.add(key);
    }
    this.enqueue(clean);
  }

  private enqueue(clean: string): void {
    this.spoke = true;
    this.reply.push(clean);
    this.chain = this.chain.then(() => (this.hushed ? undefined : this.say(clean))).catch((e: unknown) => this.failed((e as Error).message));
  }
}

/** A sentence as it is compared with what was said before: its words, in lower case. */
function sayKey(sentence: string): string {
  return sentence.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
}

/** A sentence that is only a filler word or two: "Sure!", "Great!", "Happy to!", "Of course!". */
export function isFiller(sentence: string): boolean {
  return /^(sure( thing)?|great|good|okay|ok|of course|happy to( help)?|you bet|absolutely|certainly|perfect|no problem|alright|all right|awesome|wonderful|excellent|got it|right|lovely|no worries)[!.,]*$/i.test(sentence.trim());
}

/** A reply's sentences, split where Speech splits them. */
function sentences(text: string): string[] {
  return text.split(/(?<=[.!?\u2026]["')\]]?)\s+|\n+/).map((x) => x.trim()).filter(Boolean);
}

/**
 * A call's replies as the caller heard them, for the model to read back: a
 * sentence said before on the call is left out (Speech did not say it again),
 * and so is a filler word anywhere but a reply's start. What a model reads of
 * itself it copies, so a line left in would come back again and again.
 */
export function tidyReplies(turns: Turn[]): void {
  const said = new Set<string>();
  for (let i = 0; i < turns.length; i++) {
    const t = turns[i];
    if (t.role !== 'assistant' || !t.text.trim()) continue;
    const opens = turns[i - 1]?.role === 'user';
    const kept: string[] = [];
    for (const [n, line] of sentences(t.text).entries()) {
      const clean = spoken(line);
      const key = sayKey(clean);
      if (isFiller(clean) && !(opens && n === 0)) continue;
      if (key.split(' ').length >= 4) {
        if (said.has(key)) continue;
        said.add(key);
      }
      kept.push(clean);
    }
    // All of it was said before: it was said again (see Speech.flush), so it stays.
    if (kept.length) t.text = kept.join(' ');
  }
}

/** Text as it can be said: no markdown, links as their words, no emoji, no quotation marks (a model may quote its own reply). */
export function spoken(text: string): string {
  return text
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[*_`#>|~"\u201c\u201d]+/g, '')
    .replace(/\p{Extended_Pictographic}/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/** What a call conversation is for, in its agent's instructions. */
export function callInstructions(brief: string, instructions: string): string {
  return [
    `This conversation is a live phone call, on the phone of the person you work for: who is calling, today's date and what you know about them are in the note that starts the call. Everything you write is spoken aloud to the caller as you write it, so write only what you would say: one or two short sentences, plain words, no markdown, lists, emoji, links or quotation marks. Start with what matters, not a filler word. Then stop, and let them answer.`,
    'Their words arrive as "Caller [0:42]: …", transcribed from speech (allow for a misheard word), with when they said them (minutes and seconds into the call). "over you" means they spoke while you were talking: a short "mm-hmm" or "yeah" does not stop you (you see it with their next words); more than that stops you, and you see what you were saying. If they have not finished (they stopped mid-sentence, or said "um, let me think"), write nothing at all: an empty reply keeps listening. A message "[OAIY] A note from the runner" is your person\'s direction, passed on by the main agent they talk to: go by it, without reading it out. Any other message without the "Caller:" label comes from the person you work for, who may be watching: do what they say.',
    'Your call tools: request_appointment (a booking request for staff to confirm; never say it is booked or confirmed), lookup_business_data (a question about the business\'s records or calendar), end_call (a short goodbye, then the call ends; use it when the caller is done; a brief that says finish_call means end_call). Your other tools work too.',
    REFERENCE,
    'To look something up, do it in the same reply as a few words: say "Let me check." and make the call at once. Never say you will check without doing it: the caller hears you and waits. lookup_business_data answers later, in a message of its own ("[OAIY] The answer to your lookup …"): keep the conversation going meanwhile (answer anything else they say, without guessing the answer), and tell them the answer when it comes. Other tools (a file, remember) answer at once.',
    'Say only what you know: from these instructions, the brief, or what a tool returned. Never make up availability, times, prices or bookings, and never say a time is free or agree to one unless a tool said it is. If you cannot check, say so, and offer to take their preferred time as a request for staff to confirm.',
    'To take a booking request: once you have the service, the day and time they want and their name, call request_appointment in that same reply, and only then tell them it is requested. Saying you have noted it without calling request_appointment records nothing.',
    'Never repeat something you have already said on this call. When the caller says goodbye or is done, call end_call with a short goodbye, and write nothing else.',
    'When you know their name, use it now and then, as a receptionist who remembers them would.',
    brief.trim() ? `The receptionist brief:\n${brief.trim()}` : '',
    `The instructions of the person you work for, for calls:\n${instructions.trim() || '(none)'}`,
  ].filter(Boolean).join('\n');
}

/**
 * Whether a reply tells the caller a booking is (or will be) requested or noted:
 * "I'll request Tuesday at one", "I've noted that down", "I'll put you in for…".
 */
export function promisesBooking(text: string): boolean {
  const t = text.trim();
  // Asking, or offering on a condition ("tell me a time and I'll take it as a request"), is not a promise.
  if (!t || /\?\s*$/.test(t) || /\b(if|tell me|let me know|would you like|what day|what time|which day|when would)\b/i.test(t)) return false;
  return /\b(i['’]ll|i will|i['’]ve|i have|i['’]m going to)\s+(just\s+)?(request|book|note|pencil|put (you|it|that|this) (in|down)|lock)/i.test(t);
}

/** Whether the caller has said when they want to come: a day, a date or a time. */
export function namesATime(turns: Turn[]): boolean {
  return turns.some((t) => t.role === 'user' && /(^|\n)Caller\b/.test(t.text) && /\b(mon|tues|wednes|thurs|fri|satur|sun)day\b|\b(tomorrow|today|tonight|morning|afternoon|evening|noon|midday|next week)\b|\b\d{1,2}(:\d\d)?\s*(a\.?m\.?|p\.?m\.?)|\bo'?clock\b|\b\d{1,2}(st|nd|rd|th)\b/i.test(t.text));
}

/** What a call's agent is told when it promised a booking without requesting it. */
const BOOKING_NUDGE = "[OAIY] You told the caller their booking is requested, but request_appointment was not called, so nothing is recorded. Call it now (agreementPhrase: the caller's own words agreeing to the day and time), then tell them in a few words that it is requested.";

/** How long a tool may run on a call before a short line is said, and the line. */
const HOLD_AFTER_MS = 2_000;
const HOLD_LINE = 'One moment, let me check.';

/** What the call's agent is told when it asks the business's records: the answer comes later. */
export const LOOKUP_ASKED =
  'Asked. The answer comes to you in a message of its own, usually within a few seconds. Until then keep the conversation going: tell the caller you are checking, or answer anything else they ask, and do not guess the answer.';

/** A time on a call, from its start ("0:42", "12:05"). */
export function callClock(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/**
 * A caller's words as the call's agent reads them: when they said them, and
 * how that fell against its own speech (what it was saying when they spoke
 * over it, or cut in).
 */
export function callerLine(text: string, when: { startMs?: unknown; over?: unknown; cut?: unknown }, youSaid = ''): string {
  const at = typeof when.startMs === 'number' ? callClock(when.startMs) : '';
  const said = youSaid ? ` as you said "${youSaid.length > 90 ? `${youSaid.slice(0, 90)}…` : youSaid}"` : '';
  const how = when.cut === true ? `cutting in${said}` : when.over === true ? `over you${said}` : '';
  const label = [at, how].filter(Boolean).join(', ');
  return `Caller${label ? ` [${label}]` : ''}: ${text}`;
}

/** What the call's agent is told when the business's records cannot be checked. */
export const LOOKUP_UNAVAILABLE =
  "The business's records could not be checked just now (no business lookup is set up on OAIY Desktop, or it failed). Do not guess times, availability, prices or bookings: tell the caller you can't check right now, and offer to take their preferred time as a request for staff to confirm.";

/** The note that starts a call: who, when, and what is known about them (the model's view of the call starts there). */
export function callStartNote(who: string, greeting: string, known: string, now = new Date(), returning?: number, outbound?: { purpose?: string }): string {
  const rang = returning !== undefined || !!outbound;
  const opened = greeting.trim() ? ` You ${rang ? 'opened with' : 'greeted them'}: "${greeting.trim()}"` : '';
  const why = outbound?.purpose?.trim() ? ` Why you rang: ${outbound.purpose.trim()}` : '';
  const began = returning !== undefined
    ? `You rang ${who} back, returning their missed call from ${whenSaid(returning)}; they answered ${whenSaid(now.getTime())}.`
    : outbound
      ? `You rang ${who}; they answered ${whenSaid(now.getTime())}.${why}`
      : `A call from ${who} began, ${whenSaid(now.getTime())}.`;
  return `[OAIY] 📞 ${began}${opened}\nToday is ${today(now)}.\n${known}`;
}

/** A call's first turn: the note that it began (the model's view of the call starts there). */
export function isCallStart(turn: Turn): boolean {
  return turn.role === 'user' && !!turn.automatic && /^\[OAIY\] 📞 (A call from|You rang)/.test(turn.text);
}

/**
 * A conversation's turns as its parts: each call on its own (from the note that
 * it began), with that note as its title; a text thread is one part.
 */
export function conversationParts(turns: Turn[]): Array<{ title: string; lines: string[] }> {
  const parts: Array<{ title: string; turns: Turn[] }> = [];
  for (const t of turns) {
    if (isCallStart(t) || !parts.length) parts.push({ title: isCallStart(t) ? (t as { text: string }).text.replace(/^\[OAIY\]\s*/, '').split('. ')[0] : '', turns: [] });
    parts[parts.length - 1].turns.push(t);
  }
  return parts.map((p) => ({ title: p.title, lines: conversationLines(p.turns) })).filter((p) => p.lines.length);
}

/** A conversation's words, a line each: who said what, and what was done. */
export function conversationLines(turns: Turn[]): string[] {
  const lines: string[] = [];
  for (const t of turns) {
    if (t.role === 'user') lines.push(t.text.startsWith('[OAIY]') ? `(${t.text.replace(/^\[OAIY\]\s*/, '').split('\n')[0]})` : t.text.replace(/^\[The user sent this while you[^\]]*\]\n\n/, ''));
    else if (t.role === 'assistant') {
      if (t.text.trim()) lines.push(`Agent: ${t.text.trim()}`);
      for (const c of t.calls) lines.push(`(used ${c.name}${c.name === 'send_text_message' && typeof c.input.body === 'string' ? `: "${c.input.body}"` : ''})`);
    }
  }
  return lines;
}

/** A conversation's words, as the runner reads them: who said what, and what was done. */
export function conversationText(turns: Turn[], last = 40): string {
  const shown = conversationLines(turns).slice(-last).join('\n');
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
      return all
        .map((s) => `${s.id}: ${s.kind === 'call' ? 'calls' : s.kind === 'task' ? 'flow task' : 'texts'} with ${s.title}${s.title !== s.key ? ` (${s.key})` : ''}, last ${whenSaid(s.lastAt)}${s.running ? ', working now' : ''}${s.callId ? ', on a call now' : ''}`)
        .join('\n');
    },
  };
}

/** One person's note, as the runner reads it. */
function noteText(c: CallerNote): string {
  return `${c.number}: ${c.name || '(no name yet)'}${c.facts.length ? `\n${c.facts.map((f) => `  - ${f}`).join('\n')}` : ''}`;
}

/**
 * The runner's view of what its sub-agents know about each person who calls
 * or texts, and a way to change it: what it knows reaches their next reply.
 */
export function callerNotesTool(sessions: () => Sessions | null): SessionTool {
  return {
    spec: {
      name: 'caller_notes',
      description:
        "What the phone's agents know about each person who calls or texts: one note a person (their name, and short facts): each of their calls' agents reads it as the call starts, their text thread's agent before every reply. With no number: everyone's. With a number: theirs. To change it, give name, add (one fact), remove (takes out the facts that contain these words) or facts (all of them, replacing the rest).",
      parameters: {
        type: 'object',
        properties: {
          number: { type: 'string', description: 'Their phone number' },
          name: { type: 'string' },
          add: { type: 'string', description: 'A fact to add, in a few words' },
          remove: { type: 'string' },
          facts: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    run: async (input) => {
      const all = sessions();
      if (!all) throw new Error('the phone is not set up here (OAIY Desktop has not connected yet)');
      const number = typeof input.number === 'string' ? input.number.trim() : '';
      if (!number) return all.callers.length ? all.callers.map(noteText).join('\n') : 'Nothing is saved about anyone yet.';
      const change = {
        ...(typeof input.name === 'string' ? { name: input.name } : {}),
        ...(typeof input.add === 'string' && input.add.trim() ? { add: input.add } : {}),
        ...(typeof input.remove === 'string' && input.remove.trim() ? { remove: input.remove } : {}),
        ...(Array.isArray(input.facts) ? { facts: input.facts.map(String) } : {}),
      };
      if (!Object.keys(change).length) {
        const note = all.callerNote(number);
        return note ? noteText(note) : `Nothing is saved about ${number}.`;
      }
      return `Saved. ${noteText(await all.noteCaller(number, change))}`;
    },
  };
}

/**
 * The runner passes something on to one of the phone's conversations: read
 * with its next reply, or acted on at once.
 */
export function tellAgentTool(sessions: () => Sessions | null): SessionTool {
  return {
    spec: {
      name: 'tell_agent',
      description:
        "Pass a note to one of the phone's conversations (its id from phone_conversations): a call happening now, a text thread, or a flow's tasks. Its agent reads it before its next reply, as your person's direction. With now, it acts on it at once (on a call it may speak to the caller; in a text thread it may text them). For every conversation, change /brief.md instead; for one person's next calls and texts, caller_notes.",
      parameters: {
        type: 'object',
        required: ['id', 'note'],
        properties: { id: { type: 'string' }, note: { type: 'string', description: 'What it should know or do' }, now: { type: 'boolean' } },
      },
    },
    run: async (input) => {
      const all = sessions();
      const id = typeof input.id === 'string' ? input.id.trim() : '';
      const session = all?.get(id);
      if (!all || !session) return `No conversation ${id}. Call phone_conversations for the list.`;
      const note = typeof input.note === 'string' ? input.note.trim() : '';
      if (!note) throw new Error('note is empty: write what it should know or do');
      return all.pass(session, note, input.now === true);
    },
  };
}

/** What a flow's tasks are about: the flow waits for the agent's last words as its output. */
export function taskInstructions(flow: string): string {
  return `This conversation holds the tasks your person's flow "${flow}" gives you (an "Ask the agent" node in it). Each message is one task. Do it with your tools, then end with the result itself: your last reply is handed back to the flow as its output, so give only what the flow asked for (no greeting, no offer of more help). If you cannot do it, say why in one sentence. Your files here are the front desk's (what the business wants its phone agent to know, in /knowledge), not a project of the person's.`;
}

export class Sessions {
  list: Session[] = [];
  /** What the phone's agents know about the people who call and text: one note a person. */
  callers: CallerNote[] = [];
  /** The missed call being returned to `number` now, when the call to them is a call back. */
  callingBack: (number: string) => { missedAt: number } | undefined = () => undefined;
  /** Conversations with messages waiting, in the order they came. */
  /** Conversations waiting to run, one at a time per lane: calls in their own (a caller never waits behind a text or a flow's task), the rest in another. */
  private queue: Session[] = [];
  private callQueue: Session[] = [];
  private pumping = false;
  private pumpingCalls = false;
  /**
   * The last calls that ended, by id, with their conversation (null: a call this
   * page did not follow). The desktop may still send a caller's last words
   * after the end (they were being transcribed as the caller hung up): they are
   * kept, and the call is not taken up again.
   */
  private ended = new Map<string, Session | null>();

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
    this.callers = await this.project.loadCallers();
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
        instructions: () => this.directed(callInstructions(session.brief ?? '', this.settings().callInstructions)),
        sessionTools: [...this.callTools(session), ...this.personTools(session)],
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
      instructions: () => this.directed(smsInstructions(session.title, session.key, this.settings().instructions, test, knownText(this.callerNote(session.key)))),
      // A texter reaches the front desk's files, to read (see KNOWLEDGE_TOOLS).
      tools: TOOLS.filter((t) => KNOWLEDGE_TOOLS.has(t.name)),
      // A pretend thread does not put requests in the real calendar.
      sessionTools: [this.replyTool(session, test), ...this.personTools(session), ...(test ? [] : textCalendarTools(this.desktop, session.key, () => session.title))],
      conversation: true,
    });
    return session;
  }

  /** What is known about the person at `number` (a call's and a text thread's numbers agree by their last nine digits). */
  callerNote(number: string): CallerNote | undefined {
    return this.callers.find((c) => c.number === number || sameNumber(c.number, number));
  }

  /**
   * Change what is known about the person at `number`: their name, a fact added
   * or taken out, or all the facts at once. Their conversations take the name.
   */
  async noteCaller(number: string, change: { name?: string; add?: string; remove?: string; facts?: string[] }): Promise<CallerNote> {
    const clean = (f: string) => f.replace(/\s+/g, ' ').trim().slice(0, 200);
    let note = this.callerNote(number);
    if (!note) {
      note = { number: number === TEST_NUMBER ? number : digits(number) || number, facts: [], updatedAt: Date.now() };
      this.callers.push(note);
    }
    if (typeof change.name === 'string') note.name = clean(change.name).slice(0, 80) || undefined;
    if (change.facts) note.facts = change.facts.map(clean).filter(Boolean);
    const add = clean(change.add ?? '');
    if (add && !note.facts.some((f) => f.toLowerCase() === add.toLowerCase())) note.facts.push(add);
    const remove = (change.remove ?? '').trim().toLowerCase();
    if (remove) note.facts = note.facts.filter((f) => !f.toLowerCase().includes(remove));
    note.facts = note.facts.slice(-MAX_FACTS);
    note.updatedAt = Date.now();
    // Their conversations take the name; cleared, they go back to the number.
    const renamed = typeof change.name === 'string';
    for (const s of this.list) if (s.kind !== 'task' && (s.key === note.number || sameNumber(s.key, note.number)) && (note.name || renamed)) s.title = note.name || s.key;
    await this.project.saveCallers(this.callers);
    await this.saveIndex();
    this.hooks.named?.(note);
    this.hooks.changed();
    return note;
  }

  /** The person's earlier calls and texts (theirs only, never anyone else's), as their agent reads them. */
  earlierWith(session: Session, words = ''): string {
    const parts: Array<{ title: string; lines: string[] }> = [];
    for (const s of this.list) {
      if (s.kind === 'task' || !(s === session || s.key === session.key || sameNumber(s.key, session.key))) continue;
      // What this conversation's agent reads already is left out.
      const end = s === session ? s.agent.turns.length - s.agent.view().length : s.agent.turns.length;
      for (const part of conversationParts(s.agent.turns.slice(0, end))) parts.push({ title: s.kind === 'call' ? part.title || 'A call' : 'Their text messages', lines: part.lines });
    }
    if (!parts.length) return 'Nothing earlier: this is the first time they have been in touch (or what came before was removed).';
    const cut = (text: string) => (text.length > 6000 ? `…${text.slice(-6000)}` : text);
    const wanted = words.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((w) => w.length >= 3);
    if (wanted.length) {
      const found: string[] = [];
      for (const part of parts) {
        const hits = part.lines.filter((l) => wanted.some((w) => l.toLowerCase().includes(w)));
        if (hits.length) found.push(`${part.title}:`, ...hits.map((l) => `  ${l}`));
      }
      return found.length ? cut(found.slice(-60).join('\n')) : `Nothing earlier with them mentions "${words.trim()}".`;
    }
    return cut(parts.slice(-3).map((p) => `${p.title}:\n${p.lines.slice(-20).map((l) => `  ${l}`).join('\n')}`).join('\n\n'));
  }

  /** A call's and a text thread's tools for the person on the other end: what is known about them, and their earlier conversations. */
  private personTools(session: Session): SessionTool[] {
    return [
      {
        spec: {
          name: 'remember',
          description: 'Save something about the person you are talking with, for their next call or text: their name when they tell you it, or one short fact worth knowing next time (what they usually book, a preference, where the job is). Only about them, and only what they said or what happened.',
          parameters: { type: 'object', properties: { name: { type: 'string', description: 'Their name, as they said it' }, fact: { type: 'string', description: 'One short fact, in a few words' } } },
        },
        run: async (input) => {
          const name = typeof input.name === 'string' ? input.name.trim() : '';
          const fact = typeof input.fact === 'string' ? input.fact.trim() : '';
          if (!name && !fact) throw new Error('give their name or a fact to save');
          await this.noteCaller(session.key, { ...(name ? { name } : {}), ...(fact ? { add: fact } : {}) });
          return 'Saved.';
        },
      },
      {
        spec: {
          name: 'earlier_conversations',
          description: 'Your earlier calls and text messages with this person (only theirs): what was said and done. With words: the lines that mention any of them. Without: how the last ones went.',
          parameters: { type: 'object', properties: { words: { type: 'string', description: 'Words to look for, e.g. "mowing Tuesday"' } } },
        },
        run: async (input) => this.earlierWith(session, typeof input.words === 'string' ? input.words : ''),
      },
    ];
  }

  /** A note from the runner for one conversation: read with its next reply, or (now) acted on at once. */
  async pass(session: Session, note: string, now = false): Promise<string> {
    if (session.kind === 'call' && !session.callId) return `${session.title} is not on a call now. For their next call, save it with caller_notes.`;
    const text = `[OAIY] A note from the runner (the main agent your person talks to): ${note}`;
    if (now || session.running) {
      this.deliver(session, text, true);
      return now ? 'Passed on: it acts on it now.' : 'Passed on: it reads it at its next step.';
    }
    session.agent.turns.push({ role: 'user', text, automatic: true });
    await this.save(session);
    this.hooks.changed();
    return 'Passed on: it reads it before its next reply.';
  }

  /** An answer for a call's agent (a lookup's): at once, or with the caller's words when they are speaking. */
  private answered(session: Session, callId: string, text: string): void {
    if (session.callId !== callId) return;
    if (session.callerSpeaking) (session.held ??= []).push(text);
    else this.deliver(session, text, true);
  }

  /** What this call's agent was saying at `ms` (from the call's start), as it played. */
  private sayingAt(session: Session, ms: unknown): string {
    if (typeof ms !== 'number') return '';
    return session.played?.find((p) => p.startMs <= ms && ms <= p.endMs + 300)?.text ?? '';
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
        // It answers later, in a message of its own: the agent talks with the caller meanwhile.
        run: async (input) => {
          const { desktop, callId } = live();
          const question = String(input.question ?? '');
          void desktop
            .callTool(callId, 'lookup_business_data', { question })
            // No records to ask (no business-lookup flow on the desktop, or it failed): say so plainly, so nothing is made up.
            .then((r) => (JSON.stringify(r.output ?? '').includes('LOOKUP UNAVAILABLE') ? LOOKUP_UNAVAILABLE : outcome(r)), (e: unknown) => `The lookup failed (${(e as Error).message}). ${LOOKUP_UNAVAILABLE}`)
            .then((answer) => this.answered(session, callId, `[OAIY] The answer to your lookup "${question}":\n${answer}`));
          return LOOKUP_ASKED;
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
    // The desktop's live calls, sent as its event stream opens: a call not among them has ended
    // (its end came while the stream was down, as when the desktop restarted).
    if (type === 'hello' && Array.isArray(event.calls)) {
      const live = new Set(event.calls.map(String));
      for (const s of this.list) if (s.callId && !live.has(s.callId)) await this.endCall(s);
      return null;
    }
    if (!callId) return null;
    if (this.ended.has(callId)) {
      // Words that come after the end are the caller's last: kept in their conversation, not answered.
      if (type !== 'call.started') {
        if (type === 'call.caller') await this.heardAfterEnd(this.ended.get(callId) ?? null, event);
        return null;
      }
      // The same call started again (its stream came back): it goes on.
      this.ended.delete(callId);
    }
    let session = this.list.find((s) => s.callId === callId) ?? null;
    if (type === 'call.started' || (!session && type === 'call.caller')) {
      // A hidden number: a conversation of its own for this call (never shared with another hidden caller), named as such.
      const hidden = !String(event.from ?? '').trim();
      const from = hidden ? callId : String(event.from);
      session = await this.conversationWith(from, hidden ? 'Hidden number' : String(event.name ?? ''), 'call');
      session.callId = callId;
      if (typeof event.instructions === 'string') session.brief = event.instructions;
      session.lastAt = Date.now();
      session.unread++;
      if (type === 'call.started') session.speech?.newCall();
      // A new call starts afresh: its agent reads from the note that it began. The earlier calls
      // stay, for the chat and for earlier_conversations; what is known about the caller is in its instructions.
      const fresh = type === 'call.started' && !session.running;
      if (fresh) session.agent.turns = keptTurns(session.agent.turns);
      if (type === 'call.started') {
        session.bookingNudged = false;
        session.played = [];
        session.aside = [];
        session.held = [];
        session.cutAtMs = undefined;
        session.callerSpeaking = false;
      }
      const who = `${session.title}${session.title !== session.key ? ` (${session.key})` : ''}`;
      // A call the phone placed (Aokie says so): the agent rang them, and why.
      const outbound = event.direction === 'outbound' ? { purpose: typeof event.purpose === 'string' ? event.purpose : '' } : undefined;
      const note = callStartNote(who, typeof event.greeting === 'string' ? event.greeting : '', knownText(this.callerNote(session.key)), new Date(), this.callingBack(from)?.missedAt, outbound);
      session.agent.turns.push({ role: 'user', text: note, automatic: true, ...(fresh ? { fresh: true } : {}) });
      await this.save(session);
      await this.saveIndex();
      this.hooks.changed();
      // The model reads the call's prompt while the greeting plays: its first answer comes sooner.
      if (type === 'call.started' && !session.running) void session.agent.warm();
      if (type === 'call.started') return session;
    }
    if (!session) {
      // A call this page did not follow: its end is noted all the same, so its last words do not open it here.
      if (type === 'call.ended') this.noteEnded(callId, null);
      return null;
    }
    switch (type) {
      case 'call.speech_started':
        // The caller is speaking: an answer that comes now waits for their words (or a few seconds, if it was a noise).
        session.callerSpeaking = true;
        clearTimeout(session.speakingTimer);
        session.speakingTimer = setTimeout(() => this.heardAll(session!), 6_000);
        break;
      case 'call.said':
        if (typeof event.text === 'string' && typeof event.startMs === 'number' && typeof event.endMs === 'number') {
          (session.played ??= []).push({ text: event.text, startMs: event.startMs, endMs: event.endMs });
          if (session.played.length > 400) session.played.splice(0, 100);
        }
        break;
      case 'call.caller': {
        const words = String(event.text ?? '').trim();
        if (!words) break;
        const line = callerLine(words, event, event.over === true || event.cut === true ? this.sayingAt(session, event.startMs) : '');
        session.lastAt = Date.now();
        this.hooks.arrived?.(session, line);
        clearTimeout(session.speakingTimer);
        session.callerSpeaking = false;
        // "Mm-hmm" over the agent: it talks on, and reads it with the caller's next words.
        if (event.backchannel === true) {
          (session.aside ??= []).push(line);
          this.heardAll(session);
          break;
        }
        this.deliver(session, [...(session.aside ?? []).splice(0), line, ...(session.held ?? []).splice(0)].join('\n'), true);
        break;
      }
      case 'call.interrupted':
        // The words stop. A tool at work goes on: what the caller says reaches the agent with its result.
        if (typeof event.atMs === 'number') session.cutAtMs = event.atMs;
        session.speech?.hush();
        if (!session.inTool) session.controller?.abort();
        // A reply is written faster than it is spoken: one whose run has ended may still have been playing.
        if (!session.running) await this.cutAfterRun(session);
        break;
      case 'call.ended':
        await this.endCall(session);
        break;
    }
    return session;
  }

  /** The call `session` is on has ended: it stops speaking and working, and its conversation says so. */
  private async endCall(session: Session): Promise<void> {
    if (session.callId) this.noteEnded(session.callId, session);
    clearTimeout(session.speakingTimer);
    session.callerSpeaking = false;
    session.speech?.hush();
    this.stop(session);
    session.callId = undefined;
    session.agent.turns.push({ role: 'user', text: '[OAIY] 📞 The call ended.', automatic: true });
    await this.save(session);
    this.hooks.changed();
  }

  /**
   * The caller cut in on a reply whose run had ended: it stays in the
   * conversation as far as they heard it (the sentences that had begun
   * playing), and the rest may be said again.
   */
  private async cutAfterRun(session: Session): Promise<void> {
    const cutAt = session.cutAtMs;
    session.cutAtMs = undefined;
    const last = session.agent.turns.at(-1);
    const reply = session.speech?.reply ?? [];
    // Nothing known of when it played (no call.said): kept whole, as a run cut off keeps it.
    if (cutAt === undefined || last?.role !== 'assistant' || last.calls.length || !reply.length || !session.played?.length) return;
    const heard = reply.filter((line) => session.played!.some((p) => p.text === line && p.startMs < cutAt));
    if (heard.length === reply.length) return;
    session.speech?.unsay(reply.filter((line) => !heard.includes(line)));
    last.text = heard.length ? `${heard.join(' ')}…` : '';
    await this.save(session);
  }

  private noteEnded(callId: string, session: Session | null): void {
    this.ended.set(callId, session);
    // The last few are enough: late words come within seconds of the end.
    if (this.ended.size > 20) this.ended.delete(this.ended.keys().next().value!);
  }

  /** The caller's words, heard after their call ended: kept in their conversation, not answered (no one is there to hear it). */
  private async heardAfterEnd(session: Session | null, event: Record<string, unknown>): Promise<void> {
    const words = String(event.text ?? '').trim();
    // Not into a run, or a call that has begun since: those have their own words.
    if (!session || !words || session.running || session.callId) return;
    const line = callerLine(words, event);
    this.hooks.arrived?.(session, line);
    session.agent.turns.push({ role: 'user', text: line });
    await this.save(session);
    this.hooks.changed();
  }

  /** The caller is not speaking (their words came, or it was a noise): answers held for them go to the agent. */
  private heardAll(session: Session): void {
    clearTimeout(session.speakingTimer);
    session.callerSpeaking = false;
    const held = (session.held ?? []).splice(0);
    if (held.length && session.callId) this.deliver(session, held.join('\n'), true);
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
    // The same number, written with or without its country code, is the same conversation.
    const existing = this.list.find((s) => s.kind === kind && (s.key === key || sameNumber(s.key, key)));
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
    // The call this run answers, if it is a call's: once that call has ended, what is left for the run is not answered.
    const onCall = session.callId;
    // What the run says last (a flow's task is answered with it), or why it failed.
    let said = '';
    let failed = '';
    // On a call: a tool that takes a while gets a short line said, so the caller is not left in silence.
    let holding: ReturnType<typeof setTimeout> | null = null;
    const stopHolding = () => {
      if (holding) clearTimeout(holding);
      holding = null;
    };
    // end_call has run: nothing more is said in this run, whatever tool answers after it.
    let goodbye = false;
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
          // What it says next is heard, even if the caller spoke over the words before the tool;
          // but nothing after the goodbye (end_call), even when another tool of the same reply answers after it: the call is ending.
          if (event.result.name === 'end_call') goodbye = true;
          if (!goodbye) session.speech?.begin(true);
        }
        if (session.speech && event.type === 'text') session.speech.push(event.delta);
        // A reply ends (a tool is called, or the model's turn is over): what it said is complete.
        if (session.speech && (event.type === 'tool_call' || event.type === 'usage')) session.speech.flush();
        this.hooks.event(session, event);
      }, controller.signal);
      session.speech?.flush();
      // Cut off (the caller spoke over it): what they heard of the reply is kept, so the agent knows it was said.
      const cutAt = session.cutAtMs;
      const playedBy = (line: string) => cutAt === undefined || !session.played?.length || session.played.some((p) => p.text === line && p.startMs < cutAt);
      const heard = (session.speech?.reply ?? []).filter(playedBy).join(' ');
      // What was cut off before it played was never heard: it may be said again.
      if (controller.signal.aborted && cutAt !== undefined) session.speech?.unsay((session.speech?.reply ?? []).filter((line) => !playedBy(line)));
      session.cutAtMs = undefined;
      if (controller.signal.aborted && heard && session.agent.turns.at(-1)?.role !== 'assistant') session.agent.turns.push({ role: 'assistant', text: `${heard}…`, calls: [] });
      if (session.speech) tidyReplies(session.agent.view());
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
      this.hooks.finished?.(session);
      // A booking promised but not requested: the agent is told once, and requests it.
      if (session.callId && !controller.signal.aborted && !session.bookingNudged && promisesBooking(said) && namesATime(session.agent.view()) && !requested(session)) {
        session.bookingNudged = true;
        unread.push(BOOKING_NUDGE);
      }
      if (unread.length && onCall && session.callId !== onCall) {
        // Its call has ended: what came for it (the caller's last words) is kept, and no one is answered.
        // A call begun since has its own words.
        if (!session.callId) {
          session.agent.turns.push({ role: 'user', text: unread.join('\n\n') });
          await this.save(session);
        }
      } else if (unread.length) this.deliver(session, unread.join('\n\n'));
      this.hooks.changed();
    }
  }

  stop(session: Session): void {
    session.controller?.abort();
    session.waiting = [];
    // Taken out in place: a lane being pumped goes on with the same array, and what is added to it later is run.
    for (const lane of [this.queue, this.callQueue]) {
      const at = lane.indexOf(session);
      if (at >= 0) lane.splice(at, 1);
    }
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

/** A conversation's turns, the oldest calls left out once there are too many (from a call's start, so a call is kept whole). */
function keptTurns(turns: Turn[]): Turn[] {
  if (turns.length <= MAX_KEPT_TURNS) return turns;
  const from = turns.findIndex((t, i) => i >= turns.length - MAX_KEPT_TURNS && isCallStart(t));
  return turns.slice(from > 0 ? from : turns.length - MAX_KEPT_TURNS);
}

/** Whether this call's agent has requested an appointment. */
function requested(session: Session): boolean {
  return session.agent.view().some((t) => t.role === 'assistant' && t.calls.some((c) => c.name === 'request_appointment'));
}
