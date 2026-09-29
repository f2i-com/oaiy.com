/**
 * How a conversation's words read in the chat, worked out from the text the
 * agent reads (never changed: it is what the model sees and what is saved).
 * A call's words arrive as `Caller [0:42, over you as you said "…"]: …`, a
 * text as `Text message from Lance (+61…):\n…`, a flow's task as
 * `[OAIY] Your flow "…" asks: …`; the chat shows who said it, and when, and
 * leaves out the label. Anything it does not recognise is shown as it is.
 */
import { displayNumber } from '../../phoneNumbers';

/** A caller's words on a call. */
export interface CallerPart {
  kind: 'caller';
  text: string;
  /** When they began, in ms from the call's start (when the label says). */
  atMs?: number;
  /** They spoke while the agent talked. */
  over?: boolean;
  /** They cut the agent off. */
  cut?: boolean;
  /** An acknowledgement ("mm-hmm") said over the agent, which it talked on over: inferred from the words. */
  backchannel?: boolean;
  /** What the agent was saying when they spoke over it or cut in. */
  during?: string;
}

/** A text message from the other end of a text thread. */
export interface TextPart {
  kind: 'text';
  /** Their name, when the phone knows it. */
  name?: string;
  number: string;
  text: string;
}

/** A note from OAIY itself: a lookup's answer, the runner's direction. */
export interface NotePart {
  kind: 'note';
  text: string;
}

/** A flow's task for the agent. */
export interface FlowPart {
  kind: 'flow';
  flow: string;
  text: string;
}

/** Words that are none of those: the person's own, shown as they are. */
export interface PlainPart {
  kind: 'plain';
  text: string;
}

export type Part = CallerPart | TextPart | NotePart | FlowPart | PlainPart;

/** A call's start, from the note that opens it. */
export interface CallStart {
  /** Who: the name the phone knows, or the number. */
  name: string;
  number?: string;
  /** They rang (in), or the agent rang them (out, or back). */
  direction: 'in' | 'out' | 'back';
  /** When it began, as the note says it ("Tue 29 Sep, 10:17 am"). */
  when: string;
  /** That time, when it can be read. */
  at?: Date;
  /** What the phone said first. */
  greeting?: string;
}

/** Up to three acknowledgement words, as CALLS.md lists them: heard, and talked on over. */
const ACK = /^(?:(?:mm+-?hmm+|mhm+|uh-?huh|hmm+|yeah|yep|yes|okay|ok|right|sure|i see|got it|alright|cool)[\s,.!]*){1,3}$/i;

/** Whether words are only an acknowledgement ("mm-hmm", "yeah, okay"). */
export function isAcknowledgement(text: string): boolean {
  const words = text.trim();
  return !!words && words.split(/\s+/).length <= 4 && ACK.test(words);
}

/** A time on a call, from its start: "0:42", "12:05". */
export function clock(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** A clock ("0:42", "1:02:05") as ms, or undefined. */
function clockMs(text: string): number | undefined {
  const m = /^(?:(\d+):)?(\d{1,3}):(\d{2})$/.exec(text.trim());
  if (!m) return undefined;
  return ((Number(m[1] ?? 0) * 60 + Number(m[2])) * 60 + Number(m[3])) * 1000;
}

const CALLER = /^Caller(?: \[([^\]\n]*)\])?: ?(.*)$/;

/** One `Caller […]: words` line, or null when the line is not one. */
export function parseCallerLine(line: string): CallerPart | null {
  const m = CALLER.exec(line);
  if (!m) return null;
  const part: CallerPart = { kind: 'caller', text: m[2] };
  const label = m[1] ?? '';
  if (label) {
    const [first, ...rest] = label.split(', ');
    const at = clockMs(first);
    if (at !== undefined) part.atMs = at;
    const how = (at === undefined ? [first, ...rest] : rest).join(', ');
    if (/^cutting in\b/.test(how)) part.cut = true;
    else if (/^over you\b/.test(how)) part.over = true;
    const said = /as you said "([\s\S]*)"$/.exec(how);
    if (said) part.during = said[1];
  }
  if (part.over && isAcknowledgement(part.text)) part.backchannel = true;
  return part;
}

/** A call's words: each `Caller …:` line its own part, `[OAIY]` lines notes, anything else as it is. */
export function parseCallTurn(text: string): Part[] {
  const parts: Part[] = [];
  // A blank line ends a part: what follows it is something else (the person's own words, sent with the caller's).
  let gap = false;
  for (const line of text.split('\n')) {
    const caller = parseCallerLine(line);
    if (caller || line.startsWith('[OAIY]')) {
      parts.push(caller ?? { kind: 'note', text: line.replace(/^\[OAIY\]\s*/, '') });
      gap = false;
      continue;
    }
    if (!line.trim()) {
      gap = true;
      continue;
    }
    const last = parts.at(-1);
    // A line straight after another goes on from it (a lookup's answer runs over several lines).
    if (last && !gap) last.text += `\n${line}`;
    else parts.push({ kind: 'plain', text: line });
    gap = false;
  }
  return parts.filter((p) => p.kind === 'caller' || p.text.trim());
}

/** "Lance (+61491570006)" as a name and a number; "+61400333444" as a number. */
export function splitWho(who: string): { name?: string; number: string } {
  const m = /^(.*\S) \(([^()]+)\)$/.exec(who.trim());
  if (m) return { name: m[1], number: m[2] };
  return { number: who.trim() };
}

const TEXT_HEAD = /^Text message from (.+):$/;

/**
 * A text thread's words: each `Text message from …:` and the lines after it one
 * part; what comes before the first (the person's own words) as it is.
 */
export function parseTextTurn(text: string): Part[] {
  const parts: Part[] = [];
  for (const line of text.split('\n')) {
    const head = TEXT_HEAD.exec(line);
    if (head) {
      const { name, number } = splitWho(head[1]);
      parts.push({ kind: 'text', ...(name ? { name } : {}), number, text: '' });
      continue;
    }
    const last = parts.at(-1);
    if (last) last.text = last.text ? `${last.text}\n${line}` : line;
    else parts.push({ kind: 'plain', text: line });
  }
  for (const p of parts) p.text = p.text.replace(/^\n+|\n+$/g, '');
  return parts.filter((p) => p.kind === 'text' || p.text.trim());
}

/** `[OAIY] Your flow "X" asks: …` as the flow and its task, or null. */
export function parseFlowAsk(text: string): FlowPart | null {
  const m = /^\[OAIY\] Your flow "([^"]*)" asks: ([\s\S]*)$/.exec(text);
  return m ? { kind: 'flow', flow: m[1], text: m[2] } : null;
}

/** The note that opens a call (`[OAIY] 📞 A call from … began, …`), read; null when it is not one. */
export function parseCallStart(text: string, now = new Date()): CallStart | null {
  const first = text.split('\n')[0];
  const m = /^\[OAIY\] 📞 (?:A call from (.+?) began, (.+?)\.|You rang (.+?) back, returning their missed call from .+?; they answered (.+?)\.|You rang (.+?); they answered (.+?)\.)(?= |$)/.exec(first);
  if (!m) return null;
  const direction = m[1] ? 'in' : m[3] ? 'back' : 'out';
  const who = splitWho(m[1] ?? m[3] ?? m[5]);
  const when = (m[2] ?? m[4] ?? m[6]).trim();
  const greeting = /You (?:greeted them|opened with): "([\s\S]*)"$/.exec(first)?.[1];
  const at = readWhen(when, now);
  return { name: who.name ?? who.number, ...(who.name ? { number: who.number } : {}), direction, when, ...(at ? { at } : {}), ...(greeting ? { greeting } : {}) };
}

/** What outreach (outreach.ts) writes into a conversation, read back to draw it. */
export type OutreachNote =
  /** A line after each person (and a campaign paused), one or several. */
  | { kind: 'lines'; lines: Array<{ name: string; text: string }> }
  /** The report at the end: the counts, the answers, and what to do now. */
  | { kind: 'report'; name: string; head: string; lines: string[]; answers: string; afterwards: string }
  /** A person's text thread: the outreach text they were sent. */
  | { kind: 'texted'; name: string; when: string; body: string };

const OUTREACH_LINE = /^\[OAIY\] Outreach "([^"\n]+)" (?:· (.+)|is paused: (.+))$/;

/** An outreach note (see OutreachNote), or null when the text is not one. */
export function parseOutreachNote(text: string): OutreachNote | null {
  const t = text.replace(/^\[The user sent this while you[^\]]*\]\n\n/, '').trim();
  const texted = /^\[OAIY\] Outreach "([^"\n]+)": you texted them \(([^)]*)\): "([\s\S]*)"\. Their replies come here\.$/.exec(t);
  if (texted) return { kind: 'texted', name: texted[1], when: texted[2], body: texted[3] };
  const finished = /^\[OAIY\] Outreach "([^"\n]+)" is finished/.exec(t);
  if (finished) {
    const [head, ...rest] = t.split('\n');
    const fence = /```text\n([\s\S]*?)\n```/.exec(t);
    const before = rest.join('\n').split('```text')[0].split('\n').map((l) => l.trim()).filter((l) => l && !/^Their answers, as recorded/.test(l));
    const after = fence ? t.slice(t.indexOf(fence[0]) + fence[0].length).trim() : '';
    return { kind: 'report', name: finished[1], head: head.replace(/^\[OAIY\]\s*/, ''), lines: before, answers: fence?.[1] ?? '', afterwards: after };
  }
  const rows = t.split('\n').filter((l) => l.trim());
  if (!rows.length) return null;
  const lines: Array<{ name: string; text: string }> = [];
  for (const row of rows) {
    const m = OUTREACH_LINE.exec(row.trim());
    if (!m) return null;
    lines.push({ name: m[1], text: m[2] ?? `Paused: ${m[3]}` });
  }
  return { kind: 'lines', lines };
}

/** `[OAIY] 📞 The call ended.` (or `…ended: why.`): the reason, '' for none; null when it is not one. */
export function parseCallEnd(text: string): string | null {
  const m = /^\[OAIY\] 📞 The call ended(?:: ([\s\S]*?))?\.?$/.exec(text.trim());
  return m ? (m[1] ?? '') : null;
}

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

/**
 * A time as `whenSaid` writes it ("Tue 29 Sep, 10:17 am", or "Tue, 29 Sept,
 * 10:17 am" as some browsers write en-AU): no year, so the latest such day not
 * after tomorrow. Undefined when it cannot be read.
 */
export function readWhen(text: string, now = new Date()): Date | undefined {
  const m = /(\d{1,2})\s+([A-Za-z]{3,})\.?,?\s+(\d{1,2}):(\d{2})\s*([ap])\.?\s*m\.?/i.exec(text);
  if (!m) return undefined;
  const month = MONTHS.indexOf(m[2].slice(0, 3).toLowerCase());
  if (month < 0) return undefined;
  const hour = (Number(m[3]) % 12) + (m[5].toLowerCase() === 'p' ? 12 : 0);
  for (const year of [now.getFullYear(), now.getFullYear() - 1]) {
    const date = new Date(year, month, Number(m[1]), hour, Number(m[4]));
    if (date.getTime() <= now.getTime() + 24 * 60 * 60_000) return date;
  }
  return undefined;
}

/** A day as a person says it: "Today", "Yesterday", "Monday", or "Mon 14 Sep" (with the year when it is not this one). */
export function dayLabel(date: Date, now = new Date()): string {
  const day = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(now) - day(date)) / (24 * 60 * 60_000));
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days > 1 && days < 7) return date.toLocaleDateString(undefined, { weekday: 'long' });
  return date.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', ...(date.getFullYear() !== now.getFullYear() ? { year: 'numeric' } : {}) });
}

/** A time of day, short ("10:17 am"). */
export function timeLabel(date: Date): string {
  return date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
}

/** How long ago, short: "now", "5m", "3h", "Mon", "14 Sep". */
export function ago(at: number, now = Date.now()): string {
  const minutes = Math.round((now - at) / 60_000);
  if (minutes < 1) return 'now';
  if (minutes < 60) return `${minutes}m`;
  if (minutes < 24 * 60) return `${Math.round(minutes / 60)}h`;
  const date = new Date(at);
  if (minutes < 6 * 24 * 60) return date.toLocaleDateString(undefined, { weekday: 'short' });
  return date.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

/**
 * A phone number, spaced as it is read: one of the country for local numbers
 * as it is dialled there ("0491 570 006"), any other with its country
 * ("+44 20 7946 0958"); anything else as it is.
 */
export function formatNumber(number: string, country?: string): string {
  return displayNumber(number, country);
}

/**
 * A phone conversation's words, read by what they are rather than by which
 * agent has them (a person's calls and texts are one conversation): texts
 * (`Text message from …:`), a note from OAIY on its own, or a call's lines
 * (the caller's, OAIY's notes, the person's own).
 */
export function parsePhoneTurn(text: string): Part[] {
  const lines = text.split('\n');
  if (lines.some((l) => TEXT_HEAD.test(l))) return parseTextTurn(text);
  if (text.startsWith('[OAIY] ') && !lines.some((l) => CALLER.test(l))) return [{ kind: 'note', text: text.replace(/^\[OAIY\]\s*/, '') }];
  return parseCallTurn(text);
}

/** Which way a turn came: a call (its start, a caller's words) or a text; null for anything else. */
export function wayOf(turn: { role: string; text?: string; automatic?: boolean; via?: string }): 'call' | 'sms' | null {
  if (turn.role !== 'user' || typeof turn.text !== 'string') return null;
  if (turn.automatic && /^\[OAIY\] 📞 (?:A call from|You rang)/.test(turn.text)) return 'call';
  const lines = turn.text.split('\n');
  if (lines.some((l) => TEXT_HEAD.test(l))) return 'sms';
  if (lines.some((l) => CALLER.test(l))) return 'call';
  return null;
}

/** What kind of conversation some turns are, from their words: a call's, a text thread's, a flow's tasks, or the person's own. */
export function conversationKind(turns: ReadonlyArray<{ role: string; text?: string; automatic?: boolean }>): 'call' | 'sms' | 'task' | 'own' {
  for (const t of turns) {
    if (t.role !== 'user' || typeof t.text !== 'string') continue;
    if (t.automatic && /^\[OAIY\] 📞 (?:A call from|You rang)/.test(t.text)) return 'call';
    if (/^Text message from .+:$/m.test(t.text.split('\n')[0] ?? '')) return 'sms';
    if (/^Caller(?: \[[^\]\n]*\])?: /.test(t.text)) return 'call';
    if (/^\[OAIY\] Your flow "[^"]*" asks: /.test(t.text)) return 'task';
  }
  return 'own';
}

/** One to two letters for an avatar: "Lance" → "L", "Priya Shah" → "PS"; a number or nothing → ''. */
export function initials(name: string): string {
  const words = name.replace(/[^\p{L}\p{N}\s'-]/gu, ' ').trim().split(/\s+/).filter((w) => /^\p{L}/u.test(w));
  if (!words.length) return '';
  return (words[0][0] + (words.length > 1 ? words[words.length - 1][0] : '')).toUpperCase();
}
