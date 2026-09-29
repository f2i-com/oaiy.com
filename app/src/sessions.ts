/**
 * The project's conversations besides its own chat: one per person who calls
 * or texts the phone (through Aokie and OAIY Desktop), however the phone
 * writes their number, and one per flow that gives the agent tasks. A
 * person's calls and texts are one conversation (a thread, threads.ts), each
 * way with its own agent (a lane): their texts' agent, with the person's
 * instructions and a text-message tool; and their calls' agent, fresh each
 * call, whose words are spoken as it writes them. Lanes take turns: one works
 * at a time, in the order their messages came (a local model answers one
 * request at a time anyway; a caller goes first, in a lane of their own), and
 * a message for a lane that is working reaches it at its next step.
 */
import { Agent, type AgentEvent, type AgentOptions, type SessionTool } from './agent/agent';
import { TOOLS } from './agent/tools';
import type { Turn } from './agent/protocol';
import type { Contact, Desktop, DesktopEvent } from './desktop/bridge';
import { callCalendarTools, textCalendarTools } from './desktop/calendarTools';
import { isHidden, localCountry, phoneKey, samePerson } from './phoneNumbers';
import type { MessageSettings } from './settings';
import { MAX_FACTS, regroup, threadId, threadOrder, type Way } from './threads';
import type { CallerNote, OpenProject, SessionInfo } from './vfs/projects';
import { OUTREACH_AFTER_CALL, type OutreachLink, type OutreachSessions } from './outreach';
import { CONTACT_FRESH_MS, TEXT_CONTACT_WAIT_MS, contactKey, mirrorContact, moveFacts, refused, sameFact, unionFacts, type ContactsApi } from './contacts';
import { NO_IDENTITY, identityInstructions, type Identity } from './identity';

/** What the conversations ask of outreach (outreach.ts): who a call or a text thread is about, and a text that asks to stop. */
export interface OutreachHooks {
  forCall(callId: string, number: string): OutreachLink | undefined;
  /** Someone on a list ringing in (or rung back) now: the link their call will have, found without changing anything. */
  forRing?(number: string): OutreachLink | undefined;
  forText(number: string): OutreachLink | undefined;
  /** A call of it ended: whether its agent should be asked for the result now. */
  callEnded(link: OutreachLink, lines: string[]): boolean;
  /** A text that asks to stop, from someone texted for it: handled (kept, not answered). */
  stopWord(number: string, body: string): boolean;
}

/** A number that marks a pretend conversation: its replies are never sent. */
export const TEST_NUMBER = 'test';

/** One lane of a conversation: the agent that answers a person's calls, or their texts, or a flow's tasks. */
export interface Session extends SessionInfo {
  /** The conversation it is part of (a person's calls and texts share one). */
  thread: string;
  /** When the first of the messages waiting for its next run came (its turn is kept with that time). */
  waitingSince?: number;
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
  /** When this call began, on this computer's clock (the desktop times the call's events from then). */
  clockZero?: number;
  /** When the caller's words the next run answers ended (this computer's clock): a hold word may follow them. Unset for words that may have none (said over the greeting). */
  heardAt?: number;
  /** The hold word waiting to be said in this run (see HOLD_WORD_AFTER_MS). */
  holdWord?: ReturnType<typeof setTimeout>;
  /** Answers that came in while the caller spoke: given to the agent with their words. */
  held?: string[];
  /** What the agent wrote and the desktop dropped unsaid, as the caller spoke first (call.dropped): given to it with their words. */
  unsaid?: string[];
  /** The words of the reply that calls end_call, held back to be its goodbye (so no second goodbye is said). */
  parting?: string;
  /** The outreach this call or text thread is part of: the person we rang (or who rang in from a list), with its objective and record_result. */
  outreach?: OutreachLink;
  /** This outreach call's agent was told once to record the result before ending the call. */
  endNudged?: boolean;
  /** A flow's tasks waiting for their answers, each by its prompt (one task a run; a message of the person's answers none). */
  answers?: Array<{ prompt: string; settle: (reply: string, error?: string) => void }>;
}

/**
 * One conversation, as it is listed and shown: a person's calls and texts
 * (their lanes), or a flow's tasks.
 */
export interface Thread {
  id: string;
  kind: 'person' | 'task';
  /** Who: their number (E.164), a hidden caller's call, the pretend conversation's "test", or the flow's name. */
  key: string;
  title: string;
  /** When it last heard or said something. */
  lastAt: number;
  unread: number;
  lanes: Session[];
  call?: Session;
  sms?: Session;
  task?: Session;
  /** The ways they have been in touch, and the latest. */
  ways: Way[];
  lastWay?: Way;
  /** A lane of it is working now. */
  running: boolean;
  /** Its call going on now. */
  live?: Session;
  /** A caller who hid their number: calls only, never texted. */
  hidden: boolean;
}

/** The most turns a conversation keeps (a caller's calls add up): the oldest calls go first. */
const MAX_KEPT_TURNS = 600;
/** How far back the note that starts a call looks for the person's last contact. */
const RECENT_MS = 2 * 24 * 60 * 60_000;
/** A call heard ringing in later than this after it began ringing is not warmed (see warmCall): it has been answered or missed. */
const RING_FRESH_MS = 20_000;

/** A call to read ahead (Sessions.warmCall): who, the call's id as it rings in, a dial's purpose and opening line, and its outreach. */
export interface WarmCall {
  number: string;
  name?: string;
  callId?: string;
  outbound?: { purpose?: string; opening?: string };
  link?: OutreachLink;
}

/** How the app makes an agent for this project, with a conversation's own instructions and tools, for a conversation of `kind`. */
export type MakeAgent = (extra: Pick<AgentOptions, 'instructions' | 'sessionTools' | 'tools' | 'reasoning' | 'conversation'>, kind: SessionInfo['kind']) => Agent;

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

/**
 * The same phone number as the phone itself matches numbers (Aokie's blocked
 * list): the last nine digits agree. A person is matched by samePerson
 * (phoneNumbers.ts), which reads the number whole.
 */
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

/**
 * What a call's or a text thread's agent is told about the person, from their
 * note (their contact on OAIY Desktop, as last read): their name, the
 * business's own notes about them first, then what was remembered. The
 * business's notes win over anything remembered, and a name the business gave
 * them is the one to use.
 */
export function knownText(note: CallerNote | undefined): string {
  const tools = 'Save their name when they tell you it, and anything worth knowing next time, with remember; earlier_conversations finds what was said in their earlier calls and texts.';
  const notes = note?.notes?.trim() ?? '';
  const owned = note?.ownerFacts ?? [];
  if (!note || (!note.name && !note.facts.length && !notes && !owned.length)) return `Nothing is saved about them yet. ${tools}`;
  const business = !!notes || owned.length > 0;
  return [
    'What you know about them, saved across their calls and texts:',
    `Name: ${note.name || 'not known yet'}${note.name && note.nameBy === 'owner' ? ' (the name the business has them by: use it)' : ''}`,
    ...(notes ? [`Notes from the business: ${notes}`] : business ? ['Notes from the business:'] : []),
    ...owned.map((f) => `- ${f}`),
    ...(note.facts.length ? ['What was remembered about them (by you, or by the main agent your person talks to):', ...note.facts.map((f) => `- ${f}`)] : []),
    business ? 'The notes from the business come from your person: where anything remembered says otherwise, the notes win.' : '',
    tools,
  ].filter(Boolean).join('\n');
}

/** What a text-message conversation is for, in its agent's instructions. */
export function smsInstructions(title: string, number: string, instructions: string, test: boolean, known = '', now = new Date(), identity: Identity = NO_IDENTITY): string {
  const who = title && title !== number ? `${title} (${number})` : number;
  return [
    `This conversation is a text-message thread with ${who}, on the phone of the person you work for.${test ? ' It is a test: your replies are shown, not sent.' : ''} Today is ${today(now)}.`,
    identityInstructions(identity),
    'Their messages arrive as "Text message from …". Answer them with send_text_message: short plain text (no markdown), in the language they write in. Only what you send with it reaches them; anything else you write is seen only by the person you work for.',
    'A message without that label comes from the person you work for, who may be watching: do what they say (they may tell you what to reply, or ask you to do something first).',
    `${REFERENCE} Use your flows made tools when a message needs one. You cannot change files, browse the web or run code here. There is no need to reply to a message that needs no answer (a thank-you, an emoji).`,
    'To book them in: find a time with calendar_free_times, agree a day and time with them, then request_appointment. It is a request that staff confirm (they are texted when it is): never say it is booked.',
    "To cancel one of theirs: cancel_appointment. Tell them its result plainly (a confirmed booking is passed to staff, who confirm the cancellation), and don't check the calendar again.",
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
  /** A hold word has been said in this turn (see holdWord). */
  private acknowledged = false;

  /** `say(text, hold)`: `hold` for a hold word, which the desktop says only while the caller is quiet. */
  constructor(private readonly say: (text: string, hold: boolean) => Promise<void>, private readonly failed: (error: string) => void = () => {}) {}

  /** A new reply: speak again (`afterTool`: the same turn goes on, after a tool). */
  begin(afterTool = false): void {
    this.hushed = false;
    this.buffer = '';
    this.spoke = false;
    this.repeated = [];
    this.reply = [];
    this.fillerOk = !afterTool;
    if (!afterTool) this.acknowledged = false;
  }

  /** A tool is taking a while: say a short line, unless this reply has said something already. */
  hold(line: string): void {
    if (this.hushed || this.spoke) return;
    this.speak(line);
  }

  /**
   * The reply is slow to come: a short acknowledgement ("Okay —") now, once a
   * turn, and only before the reply has said anything. It is not the reply's
   * own words: a tool's "one moment" may still follow it, and a filler word
   * that opens the reply after it is not said ("Okay — Sure!" is one too many).
   */
  holdWord(word: string): boolean {
    const clean = spoken(word);
    if (!clean || this.hushed || this.spoke || this.acknowledged) return false;
    this.acknowledged = true;
    this.fillerOk = false;
    this.enqueue(clean, false);
    return true;
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
      // Said once the next words have begun: the reply's last sentence waits for its end, so a
      // goodbye written before end_call is still here to be the goodbye (see take).
      if (!/\S/.test(this.buffer.slice(at))) return;
      this.speak(this.buffer.slice(0, at));
      this.buffer = this.buffer.slice(at);
    }
  }

  /**
   * The reply's words not said yet, taken instead of said: the reply ends the
   * call, and they are its goodbye (said by the phone, which hangs up after).
   */
  take(): string {
    const rest = this.hushed ? '' : spoken(this.buffer);
    this.buffer = '';
    this.repeated = [];
    // Words are coming (as the goodbye): no "one moment" line over a tool of this reply.
    if (rest) this.spoke = true;
    return rest;
  }

  /** Say `text` now (words held back for a goodbye that did not happen). */
  sayNow(text: string): void {
    const clean = spoken(text);
    if (clean && !this.hushed) this.enqueue(clean);
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

  /** Queue `clean` to be said; `words`: it is the reply's own (a hold word is not). */
  private enqueue(clean: string, words = true): void {
    if (words) {
      this.spoke = true;
      this.reply.push(clean);
    }
    this.chain = this.chain.then(() => (this.hushed ? undefined : this.say(clean, !words))).catch((e: unknown) => this.failed((e as Error).message));
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

/**
 * What a call conversation is for, in its agent's instructions. They are the
 * same for every call, and the whole call long, so the engine keeps them read
 * (its prompt cache): what is this call's own (who, when, what is known about
 * them, the receptionist brief the phone sent, an outreach's part) is in the
 * note that starts the call (callStartNote).
 */
export function callInstructions(instructions: string, calendar = false, identity: Identity = NO_IDENTITY): string {
  return [
    `This conversation is a live phone call, on the phone of the person you work for: who is calling, today's date, what you know about them and the receptionist brief are in the note that starts the call (and, on a call of your person's outreach, why you rang). Everything you write is spoken aloud to the caller as you write it, so write only what you would say: one or two short sentences, plain words, no markdown, lists, emoji, links or quotation marks. Start with what matters, not a filler word. Then stop, and let them answer.`,
    identityInstructions(identity),
    'Their words arrive as "Caller [0:42]: …", transcribed from speech (allow for a misheard word), with when they said them (minutes and seconds into the call). "over you" means they spoke while you were talking: a short "mm-hmm" or "yeah" does not stop you (you see it with their next words); more than that stops you, and you see what you were saying. If they have not finished (they stopped mid-sentence, or said "um, let me think"), write nothing at all: an empty reply keeps listening. A message "[OAIY] A note from the runner" is your person\'s direction, passed on by the main agent they talk to: go by it, without reading it out. Any other message without the "Caller:" label comes from the person you work for, who may be watching: do what they say.',
    `Your call tools: request_appointment (a booking request for staff to confirm; never say it is booked or confirmed), ${calendar ? "calendar_free_times (what is free, answered at once: a line a day with its hours, the times booked, the free ranges and when a service can start; use it for any question of when they can come), cancel_appointment (cancels the caller's own appointment on a day: a request at once, a confirmed booking passed to staff, who confirm the cancellation; after using it, say the result plainly and don't check the calendar again), " : ''}lookup_business_data (a question about the business's records or calendar), end_call (a short goodbye, then the call ends; use it when the caller is done; a brief that says finish_call means end_call). Your other tools work too.`,
    'Answer directly when you can: use a tool only when what the caller said needs one. Confirming a booking you already know about (from the note that starts the call) needs no availability check.',
    REFERENCE,
    `To look something up, do it in the same reply as a few words: say "Let me check." and make the call at once. Never say you will check without doing it: the caller hears you and waits. lookup_business_data answers later, in a message of its own ("[OAIY] The answer to your lookup …"): keep the conversation going meanwhile (answer anything else they say, without guessing the answer), and tell them the answer when it comes. Other tools (${calendar ? 'calendar_free_times, ' : ''}a file, remember) answer at once.`,
    'Say only what you know: from these instructions, the brief, or what a tool returned. Never make up availability, times, prices or bookings, and never say a time is free or agree to one unless a tool said it is. If you cannot check, say so, and offer to take their preferred time as a request for staff to confirm.',
    'To take a booking request: once you have the service, the day and time they want and their name, call request_appointment in that same reply, and only then tell them it is requested. Saying you have noted it without calling request_appointment records nothing.',
    'Never repeat something you have already said on this call. When the caller says goodbye or is done, call end_call with a short goodbye, and write nothing else: its goodbye is the one thing said (words written in that reply are said as the goodbye instead), and nothing written after end_call is ever said.',
    'When you know their name, use it now and then, as a receptionist who remembers them would.',
    `The instructions of the person you work for, for calls:\n${instructions.trim() || '(none)'}`,
  ].filter(Boolean).join('\n');
}

/**
 * What a call's note says that is this call's own, after who and when: the
 * receptionist brief the phone sent with it, and an outreach's part (the
 * person on the list, why you rang, what to find out).
 */
export function callOwnInstructions(brief: string, outreach = ''): string {
  return [brief.trim() ? `The receptionist brief:\n${brief.trim()}` : '', outreach.trim()].filter(Boolean).join('\n');
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

/**
 * The hold word: when the model has said nothing this long after the caller's
 * words ended, a short acknowledgement is said, once a turn, so the caller is
 * not met with silence while the reply is written. Never over the greeting, a
 * tool's "one moment", or a goodbye (end_call). Tune it here.
 */
export const HOLD_WORD_AFTER_MS = 1_500;
/** The acknowledgements, taken in turn. */
export const HOLD_WORDS = ['Okay —', 'Sure,', 'Mm, right.'];

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

/**
 * What the call's agent is told of a reply it wrote that was never said: the
 * caller spoke first, and it was dropped (their words follow it).
 */
export function unsaidNote(lines: string[]): string {
  const draft = lines.join(' ');
  return `[OAIY] Not said: the caller spoke before you could say "${draft.length > 400 ? `${draft.slice(0, 400)}…` : draft}", so it was dropped, and they did not hear it. Answer what they said now; say any of it only if it still matters.`;
}

/** What the call's agent is told when the business's records cannot be checked. */
export const LOOKUP_UNAVAILABLE =
  "The business's records could not be checked just now (no business lookup is set up on OAIY Desktop, or it failed). Do not guess times, availability, prices or bookings: tell the caller you can't check right now, and offer to take their preferred time as a request for staff to confirm.";

/**
 * The note that starts a call: who, when, what is known about them, and their
 * last contact when it was recent (the model's view of the call starts there);
 * then what else is this call's own (`own`: see callOwnInstructions).
 */
export function callStartNote(who: string, greeting: string, known: string, now = new Date(), returning?: number, outbound?: { purpose?: string }, recent = '', own = ''): string {
  const rang = returning !== undefined || !!outbound;
  const opened = greeting.trim() ? ` You ${rang ? 'opened with' : 'greeted them'}: "${greeting.trim()}"` : '';
  const why = outbound?.purpose?.trim() ? ` Why you rang: ${outbound.purpose.trim()}` : '';
  const began = returning !== undefined
    ? `You rang ${who} back, returning their missed call from ${whenSaid(returning)}; they answered ${whenSaid(now.getTime())}.`
    : outbound
      ? `You rang ${who}; they answered ${whenSaid(now.getTime())}.${why}`
      : `A call from ${who} began, ${whenSaid(now.getTime())}.`;
  return `[OAIY] 📞 ${began}${opened}\nToday is ${today(now)}.\n${known}${recent ? `\n${recent}` : ''}${own.trim() ? `\n${own.trim()}` : ''}`;
}

/** Words for a short summary: on one line, and not too long. */
const clipLine = (text: string, max = 160) => {
  const one = text.replace(/\s+/g, ' ').trim();
  return one.length > max ? `${one.slice(0, max - 1)}…` : one;
};

/**
 * The person's last contact, in a few lines, for the note that starts their
 * call: what they texted and were texted, said and were told, and a booking
 * asked for, the latest `max` of them, when it was within the last two days.
 * Empty when there was none (their earlier calls and texts are a tool call
 * away: earlier_conversations).
 */
export function recentContact(turns: readonly Turn[], lastAt: number, now = Date.now(), max = 6): string {
  if (!turns.length || !lastAt || now - lastAt > RECENT_MS) return '';
  const lines: string[] = [];
  for (const t of turns) {
    if (t.role === 'user') {
      if (t.automatic && isCallStart(t)) {
        lines.push(`(${t.text.replace(/^\[OAIY\] 📞\s*/, '').split('\n')[0].split('. ')[0].replace(/\.$/, '')})`);
        continue;
      }
      if (t.automatic || t.text.startsWith('[OAIY]')) continue;
      const texts = [...t.text.matchAll(/^Text message from [^\n]*:\n([\s\S]*?)(?=\n\nText message from |$)/gm)].map((m) => m[1]);
      if (texts.length) for (const body of texts) lines.push(`They texted: "${clipLine(body)}"`);
      else for (const m of t.text.matchAll(/^Caller(?: \[[^\]\n]*\])?: (.+)$/gm)) lines.push(`They said: "${clipLine(m[1])}"`);
    } else if (t.role === 'assistant') {
      if (t.text.trim() && t.via === 'call') lines.push(`You said: "${clipLine(t.text)}"`);
      for (const c of t.calls) {
        if (c.name === 'send_text_message' && typeof c.input.body === 'string') lines.push(`You texted: "${clipLine(c.input.body)}"`);
        if (c.name === 'request_appointment') lines.push(`(You asked for an appointment: ${[c.input.service, c.input.date, c.input.time].filter(Boolean).join(', ')})`);
      }
    }
  }
  const last = lines.slice(-max);
  if (!last.length) return '';
  return [`Their last contact, ${whenSaid(lastAt)} (earlier_conversations has more):`, ...last.map((l) => `- ${l}`)].join('\n');
}

/** A call's first turn: the note that it began (the model's view of the call starts there). */
export function isCallStart(turn: Turn): boolean {
  return turn.role === 'user' && !!turn.automatic && /^\[OAIY\] 📞 (A call from|You rang)/.test(turn.text);
}

/**
 * A conversation's turns as its parts: each call on its own (from the note that
 * it began), with that note as its title; each run of texts between them one
 * part ("Their text messages").
 */
export function conversationParts(turns: Turn[]): Array<{ title: string; lines: string[] }> {
  const parts: Array<{ title: string; way?: Way; turns: Turn[] }> = [];
  for (const t of turns) {
    const current = parts[parts.length - 1];
    const start = isCallStart(t);
    if (start || !current || (t.via && current.way && t.via !== current.way)) {
      const title = start ? (t as { text: string }).text.replace(/^\[OAIY\]\s*/, '').split('. ')[0] : t.via === 'sms' ? `Their text messages${typeof t.at === 'number' ? `, from ${whenSaid(t.at)}` : ''}` : t.via === 'call' ? 'The call, going on' : '';
      parts.push({ title, way: t.via ?? (start ? 'call' : undefined), turns: [] });
    }
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
        "The phone's conversations, each answered by sub-agents of yours: one per person (their calls and their texts together) and one per flow that gives you tasks. With no id: the list, newest first. With an id: what was said and done in it, in order (its last `last` lines, 40 by default).",
      parameters: { type: 'object', properties: { id: { type: 'string', description: "A conversation's id, from the list" }, last: { type: 'number' } } },
    },
    run: async (input) => {
      const all = sessions();
      const threads = all?.threads() ?? [];
      const id = typeof input.id === 'string' ? input.id.trim() : '';
      const who = (t: Thread) => `${t.title}${t.title !== t.key && t.kind === 'person' && !t.hidden ? ` (${t.key})` : ''}`;
      const what = (t: Thread) => (t.kind === 'task' ? 'flow tasks' : t.ways.length === 2 ? 'calls and texts' : t.ways[0] === 'call' ? 'calls' : 'texts');
      if (id) {
        const t = all?.thread(id);
        if (!t) return `No conversation ${id}. Call phone_conversations with no id for the list.`;
        const last = typeof input.last === 'number' && input.last > 0 ? Math.min(200, Math.floor(input.last)) : 40;
        const label = t.kind === 'task' ? 'Tasks from the flow' : what(t).replace(/^./, (c) => c.toUpperCase());
        return `${label} with ${who(t)}${t.live ? ', on a call now' : ''}:\n${conversationText(all!.turnsOf(t.id), last) || '(nothing yet)'}`;
      }
      if (!threads.length) return 'No calls, texts or flow tasks yet.';
      return threads.map((t) => `${t.id}: ${what(t)} with ${who(t)}, last ${whenSaid(t.lastAt)}${t.running ? ', working now' : ''}${t.live ? ', on a call now' : ''}`).join('\n');
    },
  };
}

/**
 * How a fact about a customer is written: the business reads them on the
 * dashboard's Contacts ("What the receptionist remembered"), so plainly, about
 * the customer, with none of the agents' own words for things.
 */
export function factStyle(business = ''): string {
  return `Write each fact as a short plain note about the customer, e.g. "Wants to keep the Fri 2 Oct 1 pm booking". Refer to the business by its name (${business ? `"${business}"` : 'e.g. "Green Lawns"'}) or as "the owner", never "your person", and use no internal wording (no agents, runner, tools, notes or instructions).`;
}

/** One person's note, as the runner reads it: their name (and who gave it), the business's notes, and what was remembered. */
function noteText(c: CallerNote): string {
  const lines = [
    ...(c.notes?.trim() ? [`  Notes from the business: ${c.notes.trim()}`] : []),
    ...(c.ownerFacts ?? []).map((f) => `  - ${f} (the business's)`),
    ...c.facts.map((f) => `  - ${f}`),
  ];
  return `${c.number}: ${c.name || '(no name yet)'}${c.name && c.nameBy === 'owner' ? ' (named by your person in Contacts)' : ''}${lines.length ? `\n${lines.join('\n')}` : ''}`;
}

/**
 * The runner's view of what its sub-agents know about each person who calls
 * or texts (their contacts on OAIY Desktop, read afresh), and a way to change
 * it: what it knows reaches their next reply.
 */
export function callerNotesTool(sessions: () => Sessions | null): SessionTool {
  return {
    spec: {
      name: 'caller_notes',
      description:
        `What the phone's agents know about each person who calls or texts, from their contact on OAIY Desktop (the dashboard's Contacts): their name, your person's own notes about them, and short facts remembered on their calls and texts. Each of their calls' agents reads it as the call starts, their text thread's agent before every reply. With no number: everyone's who has been in touch. With a number: theirs. To change what was remembered, give name, add (one fact), remove (takes out the remembered facts that contain these words) or facts (all of them, replacing the rest). A name or notes your person set in Contacts are theirs: they stay (your person changes them there). The facts show in Contacts, under "What the receptionist remembered": ${factStyle()}`,
      parameters: {
        type: 'object',
        properties: {
          number: { type: 'string', description: 'Their phone number' },
          name: { type: 'string' },
          add: { type: 'string', description: 'A fact to add: a short plain note about the customer, e.g. "Wants to keep the Fri 2 Oct 1 pm booking"' },
          remove: { type: 'string' },
          facts: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    run: async (input) => {
      const all = sessions();
      if (!all) throw new Error('the phone is not set up here (OAIY Desktop has not connected yet)');
      const number = typeof input.number === 'string' ? input.number.trim() : '';
      if (!number) {
        const offline = (await all.refreshContacts()) ? '' : '\n(OAIY Desktop could not be reached: this is what was known when it last could.)';
        return all.callers.length ? `${all.callers.map(noteText).join('\n')}${offline}` : `Nothing is saved about anyone yet.${offline}`;
      }
      const change = {
        ...(typeof input.name === 'string' ? { name: input.name } : {}),
        ...(typeof input.add === 'string' && input.add.trim() ? { add: input.add } : {}),
        ...(typeof input.remove === 'string' && input.remove.trim() ? { remove: input.remove } : {}),
        ...(Array.isArray(input.facts) ? { facts: input.facts.map(String) } : {}),
      };
      if (!Object.keys(change).length) {
        const read = await all.readContact(number);
        const note = all.callerNote(number);
        const offline = read === null ? '\n(OAIY Desktop could not be reached: this is what was known when it last could.)' : '';
        return note ? `${noteText(note)}${offline}` : `Nothing is saved about ${number}.${offline}`;
      }
      const { note, desk } = await all.noteCaller(number, change);
      return `Saved. ${noteText(note)}${desk ? `\n${desk}` : ''}`;
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
        "Pass a note to one of the phone's conversations (its id from phone_conversations): a person's (while they are on a call, their call's agent has it; otherwise their texts' agent), or a flow's tasks. Its agent reads it before its next reply, as your person's direction. With now, it acts on it at once (on a call it may speak to the caller; otherwise it may text them). For every conversation, change /brief.md instead; for one person's next calls and texts, caller_notes.",
      parameters: {
        type: 'object',
        required: ['id', 'note'],
        properties: { id: { type: 'string' }, note: { type: 'string', description: 'What it should know or do' }, now: { type: 'boolean' } },
      },
    },
    run: async (input) => {
      const all = sessions();
      const id = typeof input.id === 'string' ? input.id.trim() : '';
      const thread = all?.thread(id);
      if (!all || !thread) return `No conversation ${id}. Call phone_conversations for the list.`;
      const note = typeof input.note === 'string' ? input.note.trim() : '';
      if (!note) throw new Error('note is empty: write what it should know or do');
      return all.pass(thread, note, input.now === true);
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
  /** Whether the desktop's calendar is on (a plugin provides it): a text thread's calendar tools only then. */
  calendarOn: () => boolean = () => true;
  /** Who answers, and for whom (identity.ts): every call and text thread says those names. */
  identity: () => Identity = () => NO_IDENTITY;
  /** Whether this page answers the phone's calls now (it holds their lease): a call ringing in is warmed only then. */
  answersCalls: () => boolean = () => true;
  /** The warm going on for a call not begun yet (see warmCall). */
  private warming: { key: string; callId?: string; controller: AbortController } | null = null;
  /** Hold words said so far: the next is the next in HOLD_WORDS. */
  private holdWords = 0;
  /** How long after the caller's words a hold word waits (HOLD_WORD_AFTER_MS; a test makes it short). */
  holdWordAfterMs = HOLD_WORD_AFTER_MS;
  /** Outreach (the runner's lists of people to call or text), while it runs on this page. */
  outreach: OutreachHooks | null = null;
  /**
   * OAIY Desktop's contacts (contacts.ts): what is known about a person is
   * read from their contact, and what is remembered is written there. Null:
   * the Front desk's own notes only.
   */
  contacts: ContactsApi | null = null;
  /** The facts kept here before contacts were moved to the desktop: from then on a person's facts are their contact's. */
  private moved = false;
  /** When each person's contact was last read (by contact key), and the reads going on. */
  private contactRead = new Map<string, number>();
  private contactReading = new Map<string, Promise<boolean | null>>();
  /** Facts being moved or sent to the desktop now. */
  private syncing: Promise<void> | null = null;
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
  /** Each conversation's turns in the order they happened, as last kept (the order of what was there before). */
  private orders = new Map<string, { turns: Turn[]; has: Set<Turn> }>();
  /** Each conversation's file being written: one write at a time, each with all its lanes' turns as they are then. */
  private writing = new Map<string, Promise<void>>();

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

  /**
   * The project's saved conversations. Those kept before a person's calls and
   * texts were one (or under a number now read as someone else's) are merged
   * first (threads.ts, `regroup`), once: the files as they were are copied to
   * `.backup-<day>/` in the front desk's storage before anything is changed.
   */
  async load(): Promise<void> {
    const infos = await this.project.loadSessions();
    const chats = new Map<string, Turn[]>();
    for (const info of infos) {
      const file = info.thread ?? info.id;
      if (!chats.has(file)) chats.set(file, await this.project.loadSessionChat(file));
    }
    const stored = regroup({ infos, chats, callers: await this.project.loadCallers() }, localCountry());
    if (stored.changed) {
      const day = new Date();
      await this.project.backupSessions(`.backup-${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, '0')}-${String(day.getDate()).padStart(2, '0')}`);
      for (const thread of stored.rewritten) await this.project.saveSessionChat(thread, stored.chats.get(thread) ?? []);
      await this.project.saveCallers(stored.callers);
      await this.project.saveSessions(stored.infos);
      for (const file of stored.stale) await this.project.removeSessionChat(file);
    }
    this.callers = stored.callers;
    for (const info of stored.infos) {
      const session = this.create(info as SessionInfo & { thread: string });
      const turns = stored.chats.get(session.thread) ?? [];
      session.agent.turns = session.kind === 'task' ? turns : turns.filter((t) => t.via === session.kind);
      this.list.push(session);
    }
    for (const [thread, turns] of stored.chats) this.orders.set(thread, { turns, has: new Set(turns) });
    this.sort();
  }

  /** A lane, by its id. */
  get(id: string): Session | undefined {
    return this.list.find((s) => s.id === id);
  }

  /** The conversations, newest first: a person's calls and texts as one, and each flow's tasks. */
  threads(): Thread[] {
    const byId = new Map<string, Session[]>();
    for (const s of this.list) byId.set(s.thread, [...(byId.get(s.thread) ?? []), s]);
    return [...byId.values()].map((lanes) => this.threadOf(lanes)).sort((a, b) => b.lastAt - a.lastAt);
  }

  /** A conversation, by its id (or one of its lanes'). */
  thread(id: string): Thread | undefined {
    const lanes = this.list.filter((s) => s.thread === id);
    if (lanes.length) return this.threadOf(lanes);
    const lane = this.get(id);
    return lane ? this.thread(lane.thread) : undefined;
  }

  private threadOf(lanes: Session[]): Thread {
    const call = lanes.find((s) => s.kind === 'call');
    const sms = lanes.find((s) => s.kind === 'sms');
    const task = lanes.find((s) => s.kind === 'task');
    const newest = [...lanes].sort((a, b) => b.lastAt - a.lastAt);
    const named = newest.find((s) => s.title && s.title !== s.key);
    const ways: Way[] = [...(call ? ['call' as const] : []), ...(sms ? ['sms' as const] : [])];
    const last = newest.find((s) => s.kind !== 'task');
    return {
      id: lanes[0].thread,
      kind: task ? 'task' : 'person',
      key: (sms ?? call ?? task)!.key,
      title: named?.title ?? newest[0].title,
      lastAt: newest[0].lastAt,
      unread: lanes.reduce((n, s) => n + s.unread, 0),
      lanes,
      call,
      sms,
      task,
      ways,
      ...(last ? { lastWay: last.kind as Way } : {}),
      running: lanes.some((s) => !!s.running),
      ...(call?.callId ? { live: call } : {}),
      hidden: !!call?.hidden && !sms,
    };
  }

  /**
   * A conversation's turns, all its lanes', in the order they happened: as
   * they were kept, with what its lanes have said since after (see
   * threadOrder). What is new gets the time it happened.
   */
  turnsOf(thread: string): Turn[] {
    const lanes = this.list.filter((s) => s.thread === thread);
    if (lanes.length === 1 && lanes[0].kind === 'task') return lanes[0].agent.turns;
    for (const lane of lanes) this.stamp(lane);
    const turns = threadOrder(this.orders.get(thread)?.turns ?? [], lanes.map((s) => ({ via: s.kind as Way, turns: s.agent.turns })));
    this.orders.set(thread, { turns, has: new Set(turns) });
    return turns;
  }

  /**
   * The time a lane's new turns happened (`at`, never earlier than the turn
   * before it): those at its end, `at` (now, or when the message they answer
   * came). Turns kept from before times were kept have none, and get none.
   */
  private stamp(lane: Session, at = Date.now()): void {
    const turns = lane.agent.turns;
    const kept = this.orders.get(lane.thread)?.has;
    let i = turns.length - 1;
    while (i >= 0 && typeof turns[i].at !== 'number' && !kept?.has(turns[i])) i--;
    const floor = i >= 0 ? turns[i].at ?? 0 : 0;
    for (let k = i + 1; k < turns.length; k++) {
      turns[k].at = Math.max(floor, at);
      if (lane.kind !== 'task') turns[k].via ??= lane.kind;
    }
  }

  /** Whether any conversation is working (or waiting to). */
  get busy(): boolean {
    return this.list.some((s) => s.running) || this.queue.length > 0 || this.callQueue.length > 0;
  }

  private create(info: SessionInfo): Session {
    const session = { ...info, running: null, controller: null, waiting: [] } as unknown as Session;
    if (info.kind === 'call') {
      // The calendar's free times, answered at once (one tool a conversation: its repeat check lasts from call to call).
      const calendar = callCalendarTools(this.desktop, () => (session.hidden ? '' : session.key));
      const person = this.personTools(session);
      session.agent = this.makeAgent({
        // The same for every call (the engine keeps them read): the brief the phone sent, and an outreach's
        // part (one we placed, or someone on a list who rang in), are in the note that starts the call.
        instructions: () => this.directed(callInstructions(this.settings().callInstructions, this.calendarOn(), this.identity())),
        // The calendar's tool only while there is a calendar (a plugin provides it); record_result on an outreach call.
        sessionTools: () => [...this.callTools(session), ...person, ...(this.calendarOn() ? calendar : []), ...(session.outreach ? [session.outreach.resultTool()] : [])],
        tools: TOOLS.filter((t) => CALL_TOOLS.has(t.name)),
        // Answer at once: no thinking first.
        reasoning: 'none',
        conversation: true,
      }, 'call');
      session.speech = new Speech(
        async (text, hold) => {
          const desktop = this.desktop();
          if (desktop && session.callId) await desktop.say(session.callId, text, hold);
        },
        (error) => this.hooks.event(session, { type: 'status', message: `Could not speak on the call: ${error}` }),
      );
      return session;
    }
    if (info.kind === 'task') {
      session.answers = [];
      session.agent = this.makeAgent({ instructions: () => this.directed(taskInstructions(session.key)) }, 'task');
      return session;
    }
    const test = info.key === TEST_NUMBER;
    const own = [this.replyTool(session, test), ...this.personTools(session)];
    // A pretend thread does not put requests in the real calendar.
    const calendar = test ? [] : textCalendarTools(this.desktop, session.key, () => session.title);
    // Someone texted for an outreach: its objective and record_result, while their replies are its (and a while after).
    const link = () => this.outreach?.forText(session.key);
    session.agent = this.makeAgent({
      instructions: () => this.directed([smsInstructions(session.title, session.key, this.settings().instructions, test, knownText(this.callerNote(session.key)), new Date(), this.identity()), link()?.instructions()].filter(Boolean).join('\n')),
      // A texter reaches the front desk's files, to read (see KNOWLEDGE_TOOLS).
      tools: TOOLS.filter((t) => KNOWLEDGE_TOOLS.has(t.name)),
      // The calendar's tools only while there is a calendar (a plugin provides it).
      sessionTools: () => {
        const outreach = link();
        return [...own, ...(this.calendarOn() ? calendar : []), ...(outreach ? [outreach.resultTool()] : [])];
      },
      conversation: true,
    }, 'sms');
    return session;
  }

  /** What is known about the person at `number`, however it is written ("0491 570 006", "+61491570006"). */
  callerNote(number: string): CallerNote | undefined {
    return this.callers.find((c) => c.number === number || (number !== TEST_NUMBER && c.number !== TEST_NUMBER && samePerson(c.number, number)));
  }

  /**
   * Change what is known about the person at `number`: their name, a fact
   * remembered or taken out, or all the remembered facts at once. Kept here
   * and in their contact on the desktop; a name the person gave them in
   * Contacts stays theirs. Their conversations take the name. `desk` says
   * what the desktop did not take ('' when it took it all, or there is no
   * desktop to ask).
   */
  async noteCaller(number: string, change: { name?: string; add?: string; remove?: string; facts?: string[] }): Promise<{ note: CallerNote; desk: string }> {
    const clean = (f: string) => f.replace(/\s+/g, ' ').trim().slice(0, 200);
    let note = this.callerNote(number);
    if (!note) {
      note = { number: number === TEST_NUMBER ? number : phoneKey(number) || number.trim(), facts: [], updatedAt: Date.now() };
      this.callers.push(note);
    }
    const said: string[] = [];
    if (typeof change.name === 'string') {
      const name = clean(change.name).slice(0, 80) || undefined;
      if (note.nameBy === 'owner' && note.name) {
        if (name !== note.name) said.push(`Their name stays ${note.name}: your person gave it to them in Contacts (they change it there).`);
      } else {
        note.name = name;
        if (name) note.nameBy = 'agent';
        else delete note.nameBy;
      }
    }
    const before = [...note.facts];
    if (change.facts) note.facts = change.facts.map(clean).filter(Boolean);
    const add = clean(change.add ?? '');
    if (add && !note.facts.some((f) => sameFact(f, add))) note.facts.push(add);
    const remove = (change.remove ?? '').trim().toLowerCase();
    if (remove) note.facts = note.facts.filter((f) => !f.toLowerCase().includes(remove));
    note.facts = note.facts.slice(-MAX_FACTS);
    note.updatedAt = Date.now();
    this.titled(note, typeof change.name === 'string');
    await this.project.saveCallers(this.callers);
    await this.saveIndex();
    // The phone greets them by the name (only when one was given: a note with none would clear the desktop's).
    if (typeof change.name === 'string') this.hooks.named?.(note);
    this.hooks.changed();
    const desk = await this.sendFacts(note, before);
    return { note, desk: [...said, desk].filter(Boolean).join(' ') };
  }

  /** The person's conversations take the name in their note; one cleared (`renamed`), they go back to the number. */
  private titled(note: CallerNote, renamed = false): void {
    for (const s of this.list) if (s.kind !== 'task' && !s.hidden && (s.key === note.number || (s.key !== TEST_NUMBER && samePerson(s.key, note.number))) && (note.name || renamed)) s.title = note.name || s.key;
  }

  /**
   * The facts remembered (and taken out) since `before`, written to the
   * person's contact on the desktop: those it refuses are kept here only; with
   * the desktop out of reach, what was remembered waits in the note (`unsent`)
   * for it. Says what the desktop did not take ('' when it took it all).
   */
  private async sendFacts(note: CallerNote, before: readonly string[]): Promise<string> {
    const added = note.facts.filter((f) => !before.some((b) => sameFact(b, f)));
    const removed = before.filter((b) => !note.facts.some((f) => sameFact(b, f)));
    if (!this.contacts || !contactKey(note.number) || (!added.length && !removed.length)) return '';
    const said: string[] = [];
    let contact: Contact | null | undefined;
    const waiting = [...added];
    try {
      while (waiting.length) {
        const fact = waiting[0];
        try {
          contact = (await this.contacts.addFact(note.number, fact)).contact ?? contact;
        } catch (error) {
          if (!refused(error)) throw error;
          said.push(`"${fact}" is kept here only: OAIY Desktop's contacts refused it (${(error as Error).message}).`);
        }
        waiting.shift();
      }
      if (removed.length) {
        const now = await this.contacts.get(note.number);
        contact ??= now;
        // From the last down: a fact forgotten does not move the ones before it. The person's own facts stay theirs.
        for (let i = (now?.facts.length ?? 0) - 1; i >= 0; i--) {
          const fact = now!.facts[i];
          if (fact.by === 'agent' && removed.some((r) => sameFact(r, fact.text))) contact = (await this.contacts.forgetFact(note.number, i, fact.text)) ?? contact;
        }
      }
    } catch {
      // Out of reach: what was remembered is sent when the desktop can be reached (syncContacts).
      if (waiting.length) note.unsent = unionFacts(note.unsent ?? [], waiting);
      await this.project.saveCallers(this.callers);
      return [...said, `OAIY Desktop could not be reached, so ${removed.length && !waiting.length ? 'what was taken out is taken out here only' : 'it is kept here and sent to their contact when the desktop can be reached'}.`].join(' ');
    }
    if (contact !== undefined) {
      this.contactRead.set(contactKey(note.number), Date.now());
      await this.mirror(note.number, contact);
    }
    return said.join(' ');
  }

  /** Read the person's contact again when the last read is older than CONTACT_FRESH_MS: that read (null when none is needed, or none can be made). */
  private freshen(number: string): Promise<boolean | null> | null {
    const key = contactKey(number);
    if (!this.contacts || !key || number === TEST_NUMBER) return null;
    if (this.contactReading.has(key)) return this.contactReading.get(key)!;
    const at = this.contactRead.get(key);
    if (at !== undefined && Date.now() - at < CONTACT_FRESH_MS) return null;
    return this.readContact(number);
  }

  /**
   * Read the person's contact from the desktop and keep it in their note: true
   * when what is known about them changed, false when it did not (or there are
   * no contacts to read), null when the desktop could not be reached (what was
   * known stays).
   */
  readContact(number: string): Promise<boolean | null> {
    const key = contactKey(number);
    const contacts = this.contacts;
    if (!contacts || !key || number === TEST_NUMBER) return Promise.resolve(false);
    const going = this.contactReading.get(key);
    if (going) return going;
    const reading = (async () => {
      try {
        const contact = await contacts.get(number);
        this.contactRead.set(key, Date.now());
        return await this.mirror(number, contact);
      } catch {
        return null;
      } finally {
        this.contactReading.delete(key);
      }
    })();
    this.contactReading.set(key, reading);
    return reading;
  }

  /**
   * Every contact read at once (as the desktop connects, and for the runner's
   * caller_notes): each person who has a note or a conversation here gets
   * theirs. False when the desktop could not be reached.
   */
  async refreshContacts(): Promise<boolean> {
    if (!this.contacts) return true;
    let all: Contact[];
    try {
      all = await this.contacts.list();
    } catch {
      return false;
    }
    const byKey = new Map(all.map((c) => [c.key, c]));
    const people = new Map<string, string>();
    for (const n of this.callers) if (contactKey(n.number) && n.number !== TEST_NUMBER) people.set(contactKey(n.number), n.number);
    for (const s of this.list) if (s.kind !== 'task' && !s.hidden && s.key !== TEST_NUMBER && contactKey(s.key)) people.set(contactKey(s.key), people.get(contactKey(s.key)) ?? s.key);
    const now = Date.now();
    let changed = false;
    for (const [key, number] of people) {
      this.contactRead.set(key, now);
      if (await this.mirror(number, byKey.get(key) ?? null, false)) changed = true;
    }
    if (changed) {
      await this.project.saveCallers(this.callers);
      await this.saveIndex();
      this.hooks.changed();
    }
    return true;
  }

  /** Keep a contact as it was read in the person's note: saved, and their conversations named, when it changed (`save` false: the caller saves). */
  private async mirror(number: string, contact: Contact | null, save = true): Promise<boolean> {
    const note = this.callerNote(number);
    const next = mirrorContact(note, contact, phoneKey(number) || number, this.moved);
    if (!next.changed || !next.note) return false;
    const was = note?.name;
    if (note) {
      for (const k of Object.keys(note) as Array<keyof CallerNote>) if (!(k in next.note)) delete note[k];
      Object.assign(note, next.note);
    } else this.callers.push(next.note);
    this.titled(note ?? next.note, was !== undefined && next.note.name !== was);
    if (save) {
      await this.project.saveCallers(this.callers);
      await this.saveIndex();
      this.hooks.changed();
    }
    return true;
  }

  /**
   * The Front desk's facts moved to the desktop's contacts, once (callers.json
   * as it was is kept in the mark, `contacts-moved.json`); then any remembered
   * while the desktop was out of reach sent. Tried again next time when the
   * desktop cannot be reached; what it already has is not added twice.
   */
  syncContacts(): Promise<void> {
    this.syncing ??= this.sync().finally(() => (this.syncing = null));
    return this.syncing;
  }

  private async sync(): Promise<void> {
    const contacts = this.contacts;
    if (!contacts) return;
    const add = async (number: string, text: string) => (await contacts.addFact(number, text)).added;
    if (!this.moved) {
      if (await this.project.loadContactsMoved()) this.moved = true;
      else {
        const before = structuredClone(this.callers);
        const moved = await moveFacts(this.callers, add);
        if (!moved.done) return;
        for (const note of this.callers) delete note.unsent;
        await this.project.saveContactsMoved({ at: Date.now(), sent: moved.sent, there: moved.there, skipped: moved.skipped, callers: before });
        await this.project.saveCallers(this.callers);
        this.moved = true;
        return;
      }
    }
    // Remembered while the desktop was out of reach.
    let changed = false;
    for (const note of this.callers.filter((n) => n.unsent?.length && contactKey(n.number))) {
      while (note.unsent?.length) {
        try {
          await add(note.number, note.unsent[0]);
        } catch (error) {
          if (!refused(error)) {
            if (changed) await this.project.saveCallers(this.callers);
            return;
          }
        }
        note.unsent.shift();
        changed = true;
      }
      delete note.unsent;
    }
    if (changed) await this.project.saveCallers(this.callers);
  }

  /**
   * The person's earlier calls and texts (theirs only, never anyone else's), in
   * the order they happened, as their agent reads them: what that agent reads
   * already is left out.
   */
  earlierWith(session: Session, words = ''): string {
    const reads = new Set(session.agent.view());
    const parts = conversationParts(this.turnsOf(session.thread).filter((t) => !reads.has(t))).map((p) => ({ ...p, title: p.title || 'Earlier' }));
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
    const business = () => this.identity().business;
    return [
      {
        // Read each time: the business's name as the desktop says it now (the same for every call, so the prompt stays the same).
        get spec() {
          return {
            name: 'remember',
            description: `Save something about the person you are talking with, for their next call or text: their name when they tell you it, or one short fact worth knowing next time (what they usually book, a preference, where the job is). Only about them, and only what they said or what happened. The business reads these facts in its Contacts: ${factStyle(business())}`,
            parameters: { type: 'object', properties: { name: { type: 'string', description: 'Their name, as they said it' }, fact: { type: 'string', description: 'One short fact: a plain note about them, e.g. "Wants to keep the Fri 2 Oct 1 pm booking"' } } },
          };
        },
        run: async (input) => {
          const name = typeof input.name === 'string' ? input.name.trim() : '';
          const fact = typeof input.fact === 'string' ? input.fact.trim() : '';
          if (!name && !fact) throw new Error('give their name or a fact to save');
          // Kept in their contact on the desktop (a fact as the receptionist's), and here; the name as the phone greets them.
          const { note } = await this.noteCaller(session.key, { ...(name ? { name } : {}), ...(fact ? { add: fact } : {}) });
          return name && note.nameBy === 'owner' && note.name !== name ? `Saved. The business has them as ${note.name}: call them that.` : 'Saved.';
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

  /**
   * A note from the runner for one conversation: read with its next reply, or
   * (now) acted on at once. A person's goes to their call's agent while they
   * are on a call, and otherwise to their texts'.
   */
  async pass(target: Thread | Session, note: string, now = false): Promise<string> {
    const session = 'lanes' in target ? target.live ?? target.sms ?? target.task ?? target.call! : target;
    if (session.kind === 'call' && !session.callId) return `${session.title} is not on a call now. For their next call, save it with caller_notes.`;
    const text = `[OAIY] A note from the runner (the main agent your person talks to): ${note}`;
    if (now || session.running) {
      this.deliver(session, text, true);
      return now ? 'Passed on: it acts on it now.' : 'Passed on: it reads it at its next step.';
    }
    session.agent.turns.push({ role: 'user', text, automatic: true, at: Date.now() });
    await this.save(session);
    this.hooks.changed();
    return 'Passed on: it reads it before its next reply.';
  }

  /**
   * The lane the person's own message in a conversation goes to: the call
   * going on now; else their texts' agent (made now if they only ever rang);
   * a hidden caller's, their calls'; a flow's, its tasks'.
   */
  async laneFor(thread: Thread): Promise<Session> {
    if (thread.live) return thread.live;
    if (thread.task) return thread.task;
    if (thread.sms) return thread.sms;
    if (thread.hidden || !thread.call) return thread.call ?? thread.lanes[0];
    return this.conversationWith(thread.key, thread.title !== thread.key ? thread.title : '', 'sms');
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
          description: 'Say a short goodbye, then hang up. Use it when the caller is done, not before. The goodbye is the only thing said: write no other words in that reply (words you do write are said as the goodbye instead of this one), and nothing after it.',
          parameters: {
            type: 'object',
            properties: {
              goodbye: { type: 'string', description: 'The goodbye, one short sentence' },
              // An outreach call that reached a voicemail with no message to leave: hang up without a word.
              ...(session.outreach && !session.outreach.inbound ? { silent: { type: 'boolean', description: 'Voicemail with no message only (after record_result voicemail): hang up saying nothing' } } : {}),
            },
          },
        },
        run: async (input) => {
          const { desktop, callId } = live();
          // Words this reply wrote before end_call are its goodbye: said once, then the phone hangs up
          // after them. The goodbye given is said only when the reply wrote nothing.
          const parting = session.parting ?? '';
          session.parting = undefined;
          const given = String(input.goodbye ?? '').trim();
          const goodbye = parting || given;
          const link = session.outreach && !session.outreach.inbound ? session.outreach : undefined;
          if (link) {
            // An outreach call ends with its result recorded: asked once, then it may end all the same.
            if (!link.recorded() && !session.endNudged) {
              session.endNudged = true;
              // The call goes on: the words held for the goodbye are said now.
              if (parting) session.speech?.sayNow(parting);
              return 'Not yet: record_result first (the outcome and what they said), then end_call.';
            }
            if (input.silent === true) {
              if (!link.voicemailRecorded()) throw new Error('silent is only for a voicemail with no message: record_result with outcome voicemail first, or say a short goodbye');
              link.hangingUp();
              session.speech?.hush();
              await desktop.command('aokie', 'call.hangup', { callId: link.phoneCallId() ?? callId }, `oaiy:outreach-hangup:${callId}`);
              return 'Hung up without a word (a voicemail, no message). Write nothing more.';
            }
            // You rang them: a goodbye of your own, not the phone's "Thanks for calling".
            if (!goodbye) throw new Error('goodbye is needed: a short thank-you (you rang them, so not "thanks for calling")');
          }
          // What was queued to be said goes first; nothing after the goodbye.
          if (session.speech) await Promise.race([session.speech.done, new Promise((r) => setTimeout(r, 3000))]);
          session.speech?.hush();
          const r = await desktop.finishCall(callId, goodbye);
          if (!r.ok) return `Could not end the call: ${outcome(r)}`;
          return parting && given && parting !== given
            ? `The words you wrote ("${parting}") are the goodbye, so "${given}" was not said (one goodbye, not two). The call ends once they have played. Write nothing more.`
            : 'The goodbye is being said, then the call ends. Write nothing more.';
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
   * The note that starts a call with the person of lane `s` (callStartNote),
   * as it reads now: their contact as last read, then the brief the phone
   * sent and the outreach's part (the lane's own).
   */
  private startNote(s: Session, call: { greeting: string; hidden: boolean; began: Date; missedAt?: number; outbound?: { purpose?: string }; recent: string }): string {
    const who = `${s.title}${s.title !== s.key ? ` (${s.key})` : ''}`;
    return callStartNote(who, call.greeting, knownText(call.hidden ? undefined : this.callerNote(s.key)), call.began, call.missedAt, call.outbound, call.recent, callOwnInstructions(s.brief ?? '', s.outreach?.instructions() ?? ''));
  }

  /**
   * Have the model read a call's prompt before the call is answered: as it
   * rings in (`aokie.call.incoming`), or as a dial goes out (an outreach's, a
   * call back's). The call's instructions and tools are the same then as its
   * first reply's will be (all that is this call's own is in the note that
   * starts it), so by the caller's first words the engine holds them, and
   * reads only the note and their words. Never waited for: a lane of its own,
   * never listed or kept, whose warm is let go when another call warms, the
   * call ends before it began, or it is not needed (a call is going on here:
   * its turns come first). Nothing on a model that keeps no prompt (ChatGPT's
   * live-call route): see Agent.warm.
   */
  warmCall(call: WarmCall): void {
    try {
      this.warmFor(call);
    } catch {
      // Only a head start: one that cannot be made never stops a dial or the phone's events.
      this.stopWarming();
    }
  }

  private warmFor(call: WarmCall): void {
    // A hidden number is often turned away by the phone's screening: not worth the engine's time.
    if (isHidden(call.number)) return;
    const key = phoneKey(call.number) || call.number.trim();
    if (!key || this.list.some((s) => s.kind === 'call' && (s.callId || (s.running && this.isPerson(s, key))))) return;
    if (this.warming && this.warming.key === key && !this.warming.controller.signal.aborted) return;
    this.stopWarming();
    const controller = new AbortController();
    const warming = { key, ...(call.callId ? { callId: call.callId } : {}), controller };
    this.warming = warming;
    const note = this.callerNote(key);
    const known = this.list.find((s) => s.kind !== 'task' && this.isPerson(s, key) && s.title !== s.key)?.title;
    const probe = this.create({ id: `warm-${key.replace(/\W/g, '')}`, kind: 'call', key, title: (note?.nameBy === 'owner' ? note.name : '') || call.name || known || note?.name || key, lastAt: Date.now(), unread: 0, thread: `warm:${key}` } as SessionInfo);
    // The outreach it is part of: one of ours being dialled, or someone on a list ringing in (or rung back).
    probe.outreach = call.link ?? this.outreach?.forRing?.(key);
    const link = probe.outreach && !probe.outreach.inbound ? probe.outreach : undefined;
    const outbound = call.outbound || link ? { purpose: link?.objective ?? call.outbound?.purpose ?? '' } : undefined;
    const missedAt = this.callingBack(key)?.missedAt;
    probe.agent.turns = [{ role: 'user', text: this.startNote(probe, { greeting: call.outbound?.opening ?? '', hidden: false, began: new Date(), missedAt, outbound, recent: '' }), automatic: true, fresh: true }];
    const done = () => {
      if (this.warming === warming) this.warming = null;
    };
    void probe.agent.warm(controller.signal).then(done, done);
  }

  /** The warm of a call not begun yet (warmCall) is let go. */
  private stopWarming(): void {
    this.warming?.controller.abort();
    this.warming = null;
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
      // A hidden number ("", "Private", "Withheld"): a conversation of its own for this call (never shared with another hidden caller), named as such.
      const hidden = isHidden(String(event.from ?? ''));
      const from = hidden ? callId : String(event.from);
      // A call of an outreach: one we placed (by its id, kept across a reload), or someone on a list who rang in. Its person's name names the conversation.
      const outreach = !hidden ? this.outreach?.forCall(callId, from) : undefined;
      session = await this.conversationWith(from, hidden ? 'Hidden number' : String(event.name ?? '') || outreach?.person || '', 'call', hidden);
      // Their last contact before this call (their texts, their last call), from what the conversation holds now.
      const history = this.turnsOf(session.thread);
      const lastContact = Math.max(0, ...history.map((t) => t.at ?? 0), ...this.list.filter((s) => s.thread === session!.thread && s !== session && s.agent.turns.length).map((s) => s.lastAt), session.agent.turns.length ? session.lastAt : 0);
      const recent = hidden ? '' : recentContact(history, lastContact);
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
        session.unsaid = undefined;
        session.cutAtMs = undefined;
        session.callerSpeaking = false;
        session.endNudged = false;
        // The desktop starts the call's clock as it says the call began: its times (a caller's words' end) are from now.
        session.clockZero = Date.now();
      } else session.clockZero = undefined;
      session.heardAt = undefined;
      // (A call taken up by the caller's words, after a reload, keeps its outreach too; any other has none.)
      session.outreach = outreach;
      // A call the phone placed (Aokie says so): the agent rang them, and why (an outreach's objective).
      const link = session.outreach && !session.outreach.inbound ? session.outreach : undefined;
      const outbound = event.direction === 'outbound' || link ? { purpose: link?.objective ?? (typeof event.purpose === 'string' ? event.purpose : '') } : undefined;
      const s = session;
      const greeting = typeof event.greeting === 'string' ? event.greeting : '';
      const began = new Date();
      const missedAt = this.callingBack(from)?.missedAt;
      // Who, and what is known about them: their contact as last read, never waited for (the call goes on
      // at once), and read again meanwhile when that was a while ago. The brief and an outreach's part too.
      const noteNow = () => this.startNote(s, { greeting, hidden, began, missedAt, outbound, recent });
      const reading = hidden ? null : this.freshen(s.key);
      const start = { role: 'user' as const, text: noteNow(), automatic: true, ...(fresh ? { fresh: true } : {}), at: Date.now() };
      session.agent.turns.push(start);
      await this.save(session);
      await this.saveIndex();
      this.hooks.changed();
      // The contact read afresh before the agent has read the note (it does with the caller's first words): the note says what it brought.
      if (reading) {
        void reading.then((changed) => {
          if (!changed || s.callId !== callId || s.running || s.agent.turns.at(-1) !== start) return;
          start.text = noteNow();
          void this.save(s).catch(() => {});
        });
      }
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
        // No hold word over them.
        clearTimeout(session.holdWord);
        session.holdWord = undefined;
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
        // When their words ended, for the hold word: none for words said before the greeting had played (or over it).
        const greeting = session.played?.[0];
        const afterGreeting = !!greeting && (typeof event.startMs !== 'number' || event.startMs >= greeting.endMs);
        const ended = typeof event.endMs === 'number' && session.clockZero !== undefined ? Math.min(Date.now(), session.clockZero + event.endMs) : Date.now();
        session.heardAt = afterGreeting ? ended : undefined;
        // What the agent was about to say when they spoke (dropped unsaid): its own draft, read before their words.
        const unsaid = session.unsaid?.length ? unsaidNote(session.unsaid) : '';
        session.unsaid = undefined;
        this.deliver(session, [...(session.aside ?? []).splice(0), unsaid, line, ...(session.held ?? []).splice(0)].filter(Boolean).join('\n'), true);
        break;
      }
      case 'call.dropped': {
        // The desktop held what the agent wrote while the caller spoke, and dropped it when their words came: it was
        // never said. The agent stops (as at a cut), and reads it as its own unsaid draft, with their words.
        // (A hold word dropped with it was not the agent's own words.)
        const sentences = Array.isArray(event.sentences) ? event.sentences.map(String).filter((line) => line && !HOLD_WORDS.some((w) => spoken(w) === line)) : [];
        const played = session.played ?? [];
        // Sentences of the reply not yet sent to the desktop are as unsaid.
        const unsent = (session.speech?.reply ?? []).filter((line) => !sentences.includes(line) && !played.some((p) => p.text === line));
        const draft = [...sentences, ...unsent];
        if (draft.length) session.unsaid = [...(session.unsaid ?? []), ...draft];
        if (typeof event.atMs === 'number') session.cutAtMs = event.atMs;
        session.speech?.hush();
        // Never heard: said again, when the agent says it again.
        session.speech?.unsay(draft);
        if (!session.inTool) session.controller?.abort();
        if (!session.running) await this.cutAfterRun(session);
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
      case 'call.resumed': {
        // The caller only acknowledged the reply they cut off ("yeah, sure"): the desktop says the rest itself, from
        // the first line they did not hear whole. It is that reply going on, not a new turn (their words come as a
        // backchannel): its record has those lines again, the one cut off midway once.
        const lines = Array.isArray(event.sentences) ? event.sentences.map(String) : [];
        const last = session.agent.turns.at(-1);
        if (session.running || !lines.length) break;
        if (last?.role === 'assistant' && !last.calls.length) {
          let heard = last.text.replace(/…$/, '').trim();
          for (const line of [...lines].reverse()) if (heard.endsWith(line)) heard = heard.slice(0, -line.length).trim();
          last.text = [heard, ...lines].filter(Boolean).join(' ');
        } else if (last?.role === 'user') session.agent.turns.push({ role: 'assistant', text: lines.join(' '), calls: [], at: Date.now() });
        await this.save(session);
        break;
      }
      case 'call.ended':
        await this.endCall(session, typeof event.reason === 'string' ? event.reason : '');
        break;
    }
    return session;
  }

  /** The call `session` is on has ended: it stops speaking and working, and its conversation says so (and why, when it failed). */
  private async endCall(session: Session, reason = ''): Promise<void> {
    if (session.callId) this.noteEnded(session.callId, session);
    clearTimeout(session.speakingTimer);
    session.callerSpeaking = false;
    session.unsaid = undefined;
    session.speech?.hush();
    this.stop(session);
    session.callId = undefined;
    // A call that failed (the phone's voice link broke, say) says why: the person sees it in the call's record.
    const failed = /fail|error|lost/i.test(reason) ? reason.trim() : '';
    session.agent.turns.push({ role: 'user', text: failed ? `[OAIY] 📞 The call ended: ${failed}.` : '[OAIY] 📞 The call ended.', automatic: true, at: Date.now() });
    await this.save(session);
    // An outreach call that ended before its result was recorded: its agent records it now, from what was said (nothing is spoken).
    const link = session.outreach;
    if (link && this.outreach?.callEnded(link, conversationLines(session.agent.view()).slice(-8))) this.queueRun(session, OUTREACH_AFTER_CALL);
    this.hooks.changed();
  }

  /** A run for a lane after the one it has now (never read into that one: a call's run that was stopped as it ended reads nothing more). */
  private queueRun(session: Session, text: string): void {
    session.waiting.push(text);
    session.waitingSince ??= Date.now();
    const lane = session.kind === 'call' ? this.callQueue : this.queue;
    if (!lane.includes(session)) lane.push(session);
    void this.pump(session.kind === 'call');
  }

  // ---- what outreach (outreach.ts) asks of the conversations ----

  /** The phone's conversations as outreach sees them. */
  forOutreach(): OutreachSessions {
    return {
      openText: async (number, name, note) => {
        const session = await this.conversationWith(number, name, 'sms');
        session.agent.turns.push({ role: 'user', text: note, automatic: true, at: Date.now() });
        session.lastAt = Date.now();
        this.sort();
        await this.save(session);
        await this.saveIndex();
        this.hooks.changed();
        return session.thread;
      },
      liveCall: (callId) => this.list.some((s) => s.callId === callId),
      heardSince: (number, at) => {
        const lane = this.list.find((s) => s.kind !== 'task' && this.isPerson(s, phoneKey(number) || number));
        if (!lane) return [];
        return this.turnsOf(lane.thread)
          .filter((t) => t.role === 'user' && (t.at ?? 0) >= at)
          .flatMap((t) => (t as { text: string }).text.split('\n').filter((l) => /^Caller\b/.test(l)));
      },
      askForResult: (number, link) => {
        void this.conversationWith(number, '', 'call').then((session) => {
          if (session.callId) return;
          session.outreach = link;
          // A call this page did not follow: its note (which has the outreach's part) may not be in the conversation.
          this.queueRun(session, `${OUTREACH_AFTER_CALL}\n\n${link.instructions()}`);
        });
      },
      // A dial of an outreach: its prompt read while the phone rings.
      warmCall: (number, name, link) => this.warmCall({ number, name, link, outbound: { purpose: link.objective } }),
    };
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
    session.agent.turns.push({ role: 'user', text: line, at: Date.now() });
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

  /** Whether a lane is the person at `key` (never a hidden caller's, and the pretend conversation only itself). */
  private isPerson(s: Session, key: string): boolean {
    if (s.kind === 'task' || s.hidden) return false;
    if (s.key === key) return true;
    return s.key !== TEST_NUMBER && key !== TEST_NUMBER && samePerson(s.key, key);
  }

  /**
   * The lane of `kind` (their texts, or their calls) in the conversation with
   * the person at `number`, however it is written: made when it is the first
   * of its kind from them, in their conversation when they have one. A hidden
   * caller's call has one of its own.
   */
  async conversationWith(number: string, name: string, kind: 'sms' | 'call' = 'sms', hidden = false): Promise<Session> {
    const key = hidden || number === TEST_NUMBER ? number : phoneKey(number) || number.trim();
    const person = (s: Session) => !hidden && this.isPerson(s, key);
    // The name the person gave them in Contacts first; else the one given (a call's caller id may have
    // none), the name their conversation has, or the one the agents learned.
    const note = hidden ? undefined : this.callerNote(key);
    const known = (note?.nameBy === 'owner' ? note.name : '') || name || this.list.find((s) => person(s) && s.title !== s.key)?.title || note?.name || '';
    const existing = this.list.find((s) => s.kind === kind && person(s));
    if (existing) {
      if (known && existing.title === existing.key) existing.title = known;
      return existing;
    }
    // Their other way in: its conversation, and its key (the number as first kept).
    const sibling = this.list.find(person);
    const session = this.create({
      id: `${kind}-${key.replace(/\W/g, '') || key}`,
      kind,
      key: sibling?.key ?? key,
      title: known || sibling?.title || key,
      lastAt: Date.now(),
      unread: 0,
      thread: sibling?.thread ?? threadId(key),
      ...(hidden ? { hidden: true } : {}),
    });
    this.list.push(session);
    // The other lane takes the name, when this one brings it.
    if (sibling && known && sibling.title === sibling.key) sibling.title = known;
    await this.saveIndex();
    return session;
  }

  /**
   * An event from the desktop: a call ringing in has its prompt read before it
   * is answered (and let go if it ends unanswered); a text message becomes (or
   * continues) a conversation.
   */
  async desktopEvent(event: DesktopEvent, now = Date.now()): Promise<Session | null> {
    const call = String(event.data.callId ?? '') || event.correlationId;
    if (event.name === 'aokie.call.incoming') {
      // (Not one heard of late, as when the desktop was out of reach a while: it has been answered or missed by now.)
      const at = Date.parse(event.occurredAt);
      if (this.answersCalls() && !(now - at > RING_FRESH_MS)) this.warmCall({ number: String(event.data.from ?? ''), name: String(event.data.name ?? ''), ...(call ? { callId: call } : {}) });
      return null;
    }
    if (event.name === 'aokie.call.ended') {
      if (call && this.warming?.callId === call) this.stopWarming();
      return null;
    }
    if (event.name !== 'aokie.sms.received') return null;
    const from = String(event.data.from ?? '');
    const body = String(event.data.body ?? '');
    if (!from || !body) return null;
    // A text the phone delivers again (it does, when it reconnects) is not a new one.
    const handle = String(event.data.handle ?? '');
    if (handle) {
      const existing = this.list.find((s) => s.kind === 'sms' && this.isPerson(s, phoneKey(from) || from));
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
    // STOP from someone texted for an outreach is read by code, first: they are not contacted again, and nothing is sent back.
    const stopped = this.outreach?.stopWord(session.key, body) ?? false;
    // A pretend text is always answered: trying the agent is what it is for. Someone texted for an
    // outreach is answered even while answering is off (only them: the outreach's objective is theirs).
    if (!stopped && (this.settings().answer || session.key === TEST_NUMBER || !!this.outreach?.forText(session.key))) this.deliver(session, text);
    else {
      // Kept, not answered: it is there when the person looks, or answers it themselves.
      session.agent.turns.push({ role: 'user', text, at: Date.now() });
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
      // Answered now, and kept as it came.
      if (typeof last.at === 'number') session.waitingSince ??= last.at;
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
      const id = `task-${key.replace(/\W+/g, '-').toLowerCase().slice(0, 40) || 'flow'}-${Date.now().toString(36)}`;
      session = this.create({ id, kind: 'task', key, title: key, lastAt: Date.now(), unread: 0, thread: id });
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

  /** The person's own message in a conversation (to the lane `laneFor` gives), or in one lane: it goes first. */
  async say(target: Thread | Session, text: string): Promise<Session> {
    const session = 'lanes' in target ? await this.laneFor(target) : target;
    this.deliver(session, text, true);
    return session;
  }

  /** A message for a lane: to its running agent, or its next run. */
  private deliver(session: Session, text: string, first = false): void {
    // A flow's task is its own run (its answer is what that run says), never added to another.
    if (session.kind !== 'task' && session.running && session.agent.interject(text)) {
      // Read by the run going on: no hold word of a run of its own.
      session.heardAt = undefined;
      return;
    }
    session.waiting.push(text);
    session.waitingSince ??= Date.now();
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
    // When the caller's words this run answers ended: a hold word follows them if nothing is said soon.
    const heardAt = session.heardAt;
    session.heardAt = undefined;
    this.hooks.changed();
    // A text thread's agent reads the person's contact as it is now: a moment's wait at most (a text can
    // wait that long; a call's agent never waits, see callEvent).
    if (session.kind === 'sms') {
      const reading = this.freshen(session.key);
      if (reading) await Promise.race([reading, new Promise((r) => setTimeout(r, TEXT_CONTACT_WAIT_MS))]);
    }
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
    // The reply being written calls end_call (seen as its calls begin): its words are held back to be the goodbye.
    let ending = false;
    let draft = '';
    session.parting = undefined;
    // The hold word (HOLD_WORD_AFTER_MS): said once, when the model has begun nothing (no words, no tool) that
    // long after the caller's words ended, and never with a goodbye coming, over the caller, or after the call.
    let replying = false;
    const stopHoldWord = () => {
      clearTimeout(session.holdWord);
      session.holdWord = undefined;
    };
    if (session.speech && onCall && heardAt !== undefined) {
      stopHoldWord();
      session.holdWord = setTimeout(() => {
        session.holdWord = undefined;
        if (replying || goodbye || ending || session.parting || session.callerSpeaking || session.callId !== onCall || controller.signal.aborted) return;
        if (session.speech?.holdWord(HOLD_WORDS[this.holdWords % HOLD_WORDS.length])) this.holdWords++;
      }, Math.max(0, heardAt + this.holdWordAfterMs - Date.now()));
    }
    // When the messages it answers came: the time its first turn is kept with (the rest, as they come).
    let arrived = session.waitingSince;
    session.waitingSince = session.waiting.length ? Date.now() : undefined;
    try {
      await session.agent.run(prompt, (event) => {
        this.stamp(session, arrived);
        arrived = undefined;
        if (event.type === 'done') said = event.text;
        if (event.type === 'error') failed = event.message;
        // The reply has begun (its words, or a tool, whose own line covers a wait): no hold word now.
        if ((event.type === 'text' && event.delta.trim()) || event.type === 'tool_start' || event.type === 'tool_draft' || event.type === 'tool_call') {
          replying = true;
          stopHoldWord();
        }
        if (event.type === 'tool_start' && event.name === 'end_call') ending = true;
        // A model that writes its calls as text (OAIY's own format): the name is in the draft.
        if (event.type === 'tool_draft' && session.speech) {
          draft = event.start ? event.text : draft + event.text;
          if (/<function=end_call>|"name"\s*:\s*"end_call"/.test(draft)) ending = true;
        }
        if (event.type === 'tool_call') {
          session.inTool = true;
          stopHolding();
          // Not over a reply that ends the call: its goodbye is coming.
          if (session.speech && !session.parting) holding = setTimeout(() => session.speech?.hold(HOLD_LINE), HOLD_AFTER_MS);
        }
        if (event.type === 'tool_result') {
          session.inTool = false;
          stopHolding();
          // What it says next is heard, even if the caller spoke over the words before the tool;
          // but nothing after the goodbye (end_call), even when another tool of the same reply answers after it: the call is ending.
          if (event.result.name === 'end_call' && !event.result.isError && !/^Not yet/.test(event.result.content)) goodbye = true;
          if (!goodbye) session.speech?.begin(true);
        }
        if (session.speech && event.type === 'text') session.speech.push(event.delta);
        // A reply ends (a tool is called, or the model's turn is over): what it said is complete. A reply
        // that ends the call keeps its last words for end_call: they are the goodbye, said once.
        if (session.speech && event.type === 'usage') {
          if (!ending) session.speech.flush();
          else {
            const rest = session.speech.take();
            if (rest) session.parting = rest;
          }
          ending = false;
          draft = '';
        } else if (session.speech && event.type === 'tool_call') session.speech.flush();
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
      stopHoldWord();
      session.inTool = false;
      session.parting = undefined;
      // The task this run was (none, when it was a message of the person's).
      const task = session.answers?.findIndex((a) => a.prompt === prompt) ?? -1;
      if (task >= 0) session.answers!.splice(task, 1)[0].settle(said.trim(), said.trim() ? undefined : failed || 'the agent finished without an answer');
      session.running = null;
      session.controller = null;
      finish();
      session.lastAt = Date.now();
      // Messages that came as it finished: its next turn.
      const unread = session.agent.takeUnread();
      // Its call has ended: what came for it (the caller's last words) is kept, and no one is answered
      // (kept before the save, which has it). A call begun since has its own words.
      const callOver = !!onCall && session.callId !== onCall;
      if (callOver && unread.length && !session.callId) session.agent.turns.push({ role: 'user', text: unread.join('\n\n'), at: Date.now() });
      await this.save(session);
      await this.saveIndex();
      this.hooks.finished?.(session);
      // A booking promised but not requested: the agent is told once, and requests it.
      if (session.callId && !controller.signal.aborted && !session.bookingNudged && promisesBooking(said) && namesATime(session.agent.view()) && !requested(session)) {
        session.bookingNudged = true;
        unread.push(BOOKING_NUDGE);
      }
      if (!callOver && unread.length) this.deliver(session, unread.join('\n\n'));
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

  /** The person looked at it (a conversation, all its lanes). */
  seen(target: Thread | Session): void {
    const lanes = 'lanes' in target ? target.lanes : [target];
    if (!lanes.some((s) => s.unread)) return;
    for (const s of lanes) s.unread = 0;
    void this.saveIndex();
    this.hooks.changed();
  }

  /** A conversation removed: all its lanes, and its turns. */
  async remove(target: Thread | Session): Promise<void> {
    const thread = 'lanes' in target ? target.id : target.thread;
    const lanes = this.list.filter((s) => s.thread === thread);
    for (const s of lanes) this.stop(s);
    this.list = this.list.filter((s) => s.thread !== thread);
    this.orders.delete(thread);
    await this.project.saveSessionChat(thread, []);
    await this.saveIndex();
    this.hooks.changed();
  }

  /**
   * Keep a lane's conversation: all its lanes' turns in their order, in its
   * file. One write at a time for a conversation, each with the turns as they
   * are when it is written (two lanes of one person can finish at once).
   */
  async save(session: Session): Promise<void> {
    const thread = session.thread;
    const write = (this.writing.get(thread) ?? Promise.resolve()).then(async () => {
      const lanes = this.list.filter((s) => s.thread === thread);
      if (!lanes.length) return;
      // As each agent keeps its own: pictures its tools showed long ago are left out.
      const saved = new Map<Turn, Turn>();
      for (const lane of lanes) {
        const kept = lane.agent.savedTurns();
        lane.agent.turns.forEach((t, i) => saved.set(t, kept[i] ?? t));
      }
      await this.project.saveSessionChat(thread, this.turnsOf(thread).map((t) => saved.get(t) ?? t));
    });
    const settled = write.catch(() => {});
    this.writing.set(thread, settled);
    await write;
  }

  private sort(): void {
    this.list.sort((a, b) => b.lastAt - a.lastAt);
  }

  private async saveIndex(): Promise<void> {
    await this.project.saveSessions(this.list.map(({ id, kind, key, title, lastAt, unread, handles, thread, hidden }) => ({ id, kind, key, title, lastAt, unread, ...(handles?.length ? { handles } : {}), thread, ...(hidden ? { hidden } : {}) })));
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
    /**
     * The events the first look passes over (those from before this page
     * opened, or came while it reloaded), a page at a time: not acted on as
     * new, but what was under way (an outreach call's end) can be settled.
     */
    private readonly backlog: (events: DesktopEvent[]) => void | Promise<void> = () => {},
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
            try {
              await this.backlog(page.events);
            } catch {
              /* what could not be settled from it is settled later (outreach looks at a lost call itself) */
            }
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
