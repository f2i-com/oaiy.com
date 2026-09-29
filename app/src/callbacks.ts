/**
 * Missed calls, called back. A call to the phone that was missed (it rang
 * out, or came while the receptionist was on another call) is kept, and once
 * the receptionist is free the number is rung back through Aokie (call.dial);
 * the front desk's agent takes the call as it takes any other. Only numbers
 * the person's filter allows are called, never a blocked or a private one. A
 * caller who gets through again, or texts, needs no call back. Aokie keeps its
 * own guardrails (outbound calling switched on, quiet hours, a daily cap): a
 * refusal waits and tries again later.
 */
import type { Desktop, DesktopEvent } from './desktop/bridge';
import { isHidden, samePerson, toE164 } from './phoneNumbers';
import type { MessageSettings } from './settings';
import type { OpenProject } from './vfs/projects';

export interface Callback {
  /** Their number, as the phone gave it. */
  number: string;
  /** When their call was missed (ms). */
  missedAt: number;
  /** Rings placed so far, and when the next may be (ms). */
  tries: number;
  nextAt: number;
  state: 'waiting' | 'calling' | 'done' | 'dropped';
  /** How it ended, or why the last try did not reach them. */
  note?: string;
  /** They hung up waiting in the queue (the line was busy), not a call that rang out: the call back says sorry first. */
  queued?: boolean;
}

/** Aokie's call screening, as the person set it: which calls are answered at all. */
export interface Screening {
  /** Only callers whose number matches are answered (empty: everyone). */
  acceptPattern: string;
  /** Numbers never answered: one a line (Aokie matches the last nine digits). */
  blockedNumbers: string;
  /** Callers who hide their number are not answered. */
  rejectPrivate: boolean;
}

/** Australian numbers, as a caller id shows them: +61…, 61… or 0… (Aokie's acceptPattern). */
export const AU_PATTERN = '^\\s*(\\+?61|\\(?0[1-9])';

/** Which numbers are called back: those the receptionist answers, Australian ones, or any. Never a blocked or private one. */
export type CallBackFilter = 'answered' | 'au' | 'any';

/** Rings to a number before it is left (a missed call is not chased all day). */
const MAX_TRIES = 2;
/** A missed call waits this long first: they may well ring again themselves. */
const FIRST_WAIT_MS = 90_000;
/** Between tries, and after a refusal that says nothing about when. */
const RETRY_MS = 20 * 60_000;
/** A dial with no end heard of it is taken as over after this long. */
const CALLING_MS = 10 * 60_000;
/** Missed calls older than this are not called back. */
const TOO_OLD_MS = 24 * 60 * 60_000;

/**
 * Whether a blocked list (one number a line, or with commas) has `number`: as
 * Aokie matches it (the last nine digits, of six or more), or as the same
 * person however either is written (phoneNumbers.ts). Either one blocks it.
 */
export function isBlocked(number: string, blockedNumbers: string): boolean {
  const suffix = (n: string) => n.replace(/\D/g, '').slice(-9);
  const entries = blockedNumbers.split(/[,;\n]/).map((b) => b.trim()).filter(Boolean);
  return entries.some((b) => (suffix(b).length >= 6 && suffix(b) === suffix(number)) || samePerson(b, number));
}

/** Whether `number` is one the filter calls back (given Aokie's screening). Never a hidden or blocked one. */
export function callsBack(number: string, filter: CallBackFilter, screening: Screening | null): boolean {
  const digits = number.replace(/\D/g, '');
  if (digits.length < 6 || isHidden(number)) return false;
  if (isBlocked(number, screening?.blockedNumbers ?? '')) return false;
  const matches = (pattern: string) => {
    try {
      return new RegExp(pattern).test(number.trim());
    } catch {
      return true;
    }
  };
  // Australian: as Aokie's pattern reads the caller id, or an Australian number however it is written ("0011 61…").
  if (filter === 'au') return matches(AU_PATTERN) || !!toE164(number, 'AU')?.startsWith('+61');
  if (filter === 'answered' && screening?.acceptPattern.trim()) return matches(screening.acceptPattern.trim());
  return true;
}

/** What Aokie's refusal of a dial means for the next try: when it may be. */
export function retryAfter(error: string, now = Date.now()): number {
  if (/quiet hours/i.test(error)) return now + 30 * 60_000;
  if (/daily dial cap/i.test(error)) {
    const tomorrow = new Date(now);
    tomorrow.setHours(24, 5, 0, 0);
    return tomorrow.getTime();
  }
  return now + RETRY_MS;
}

export class Callbacks {
  list: Callback[] = [];
  private timer: ReturnType<typeof setInterval> | null = null;
  private ticking = false;

  constructor(
    private readonly project: OpenProject,
    private readonly settings: () => MessageSettings,
    private readonly desktop: () => Desktop | null,
    /** The receptionist is free: no call going on. */
    private readonly free: () => boolean,
    /** Aokie's screening now (read when a call back is due). */
    private readonly screening: () => Promise<Screening | null>,
    private readonly changed: () => void = () => {},
    /**
     * Whether Aokie sends its calls to OAIY (true), to its own voice (false), or cannot
     * say (null). On its own voice, FormLogic's follow-ups ring missed calls back:
     * OAIY does not as well, so no one is rung twice.
     */
    private readonly callsToOaiy: () => Promise<boolean | null> = async () => true,
    /** A number never to be rung (they asked not to be called: the do-not-contact list). */
    private readonly skip: (number: string) => boolean = () => false,
    /** A call back is being dialled: the model reads its call's prompt while the phone rings (never waited for). */
    private readonly warm: (number: string, purpose: string) => void = () => {},
  ) {}

  /** A call back is ringing now. */
  ringing(): boolean {
    return this.list.some((c) => c.state === 'calling');
  }

  /** A call back is due now (it goes before an outreach list: a missed caller first). */
  due(now = Date.now()): boolean {
    return this.settings().callBack && this.list.some((c) => c.state === 'waiting' && c.nextAt <= now && now - c.missedAt <= TOO_OLD_MS && !this.skip(c.number));
  }

  async load(): Promise<void> {
    this.list = await this.project.loadCallbacks();
  }

  start(every = 15_000): void {
    this.stop();
    this.timer = setInterval(() => void this.tick(), every);
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  /** The ones still to be made (waiting, or ringing now). */
  get open(): Callback[] {
    return this.list.filter((c) => c.state === 'waiting' || c.state === 'calling');
  }

  /** The call back ringing `number` now, if one is. */
  calling(number: string): Callback | undefined {
    return this.list.find((c) => c.state === 'calling' && samePerson(c.number, number));
  }

  /** An event from the desktop: a missed call is kept; a call that got through, or a text, settles it; a call back's end is noted. */
  async event(event: DesktopEvent): Promise<void> {
    const d = event.data;
    if (event.name === 'aokie.call.ended') {
      const number = String(d.from ?? d.callerPhone ?? '').trim();
      if (!number) return;
      const outcome = String(d.outcome ?? '');
      if (d.direction === 'outbound') await this.rang(number, outcome);
      else if (outcome === 'missed') await this.missed(number);
      // Left waiting in the queue while the line was busy: rung back like a missed call, with a sorry.
      // (One who hung up on hold, after talking, is texted an apology by Aokie's flows, not rung.)
      else if (outcome === 'abandoned_in_queue') await this.missed(number, Date.now(), true);
      else if (outcome === 'completed') await this.settle(number, 'They rang again and were answered.');
    } else if (event.name === 'aokie.sms.received') {
      const number = String(d.from ?? '').trim();
      if (number) await this.settle(number, 'They texted: the text conversation has them.');
    }
  }

  /** A call from `number` was missed: it is called back once the receptionist is free. */
  async missed(number: string, at = Date.now(), queued = false): Promise<void> {
    const open = this.open.find((c) => samePerson(c.number, number));
    if (open) {
      open.missedAt = at;
      if (queued) open.queued = true;
      if (open.state === 'waiting') open.nextAt = Math.min(open.nextAt, at + FIRST_WAIT_MS);
    } else this.list.push({ number, missedAt: at, tries: 0, nextAt: at + FIRST_WAIT_MS, state: 'waiting', ...(queued ? { queued: true } : {}) });
    await this.save();
  }

  /** They got through another way: no call back. */
  async settle(number: string, why: string): Promise<void> {
    const open = this.open.filter((c) => c.state === 'waiting' && samePerson(c.number, number));
    if (!open.length) return;
    for (const c of open) Object.assign(c, { state: 'done', note: why });
    await this.save();
  }

  /** A call back to `number` ended: answered, it is done; not, it is tried again later (or left). */
  async rang(number: string, outcome: string): Promise<void> {
    const c = this.calling(number);
    if (!c) return;
    if (outcome === 'completed') Object.assign(c, { state: 'done', note: 'Called back.' });
    else if (c.tries >= MAX_TRIES) Object.assign(c, { state: 'dropped', note: `No answer after ${c.tries} tries.` });
    else Object.assign(c, { state: 'waiting', nextAt: Date.now() + RETRY_MS, note: 'No answer: trying again later.' });
    await this.save();
  }

  /** Ring the next one back, when calling back is on and the receptionist is free. */
  async tick(now = Date.now()): Promise<void> {
    if (this.ticking) return;
    this.ticking = true;
    try {
      const settings = this.settings();
      const desktop = this.desktop();
      // A dial whose end was never heard (the phone went away): taken as not answered.
      for (const c of this.list) if (c.state === 'calling' && now - c.nextAt > CALLING_MS) await this.rang(c.number, 'lost');
      if (!settings.callBack || !desktop || !this.free() || this.list.some((c) => c.state === 'calling')) return;
      for (const c of this.list) if (c.state === 'waiting' && now - c.missedAt > TOO_OLD_MS) Object.assign(c, { state: 'dropped', note: 'Too long ago to call back.' });
      const next = this.list.filter((c) => c.state === 'waiting' && c.nextAt <= now).sort((a, b) => a.missedAt - b.missedAt)[0];
      if (!next) return;
      if (this.skip(next.number)) {
        Object.assign(next, { state: 'dropped', note: 'They asked not to be called.' });
        await this.save();
        return;
      }
      if (!callsBack(next.number, settings.callBackFilter, await this.screening())) {
        Object.assign(next, { state: 'dropped', note: 'Not one of the numbers you call back.' });
        await this.save();
        return;
      }
      const route = await this.callsToOaiy();
      if (route === false) {
        Object.assign(next, { state: 'dropped', note: "Calls go to Aokie's own voice, so its follow-ups ring missed calls back." });
        await this.save();
        return;
      }
      // Cannot say now (the phone away): asked again at the next look.
      if (route === null) return;
      const when = new Date(next.missedAt).toLocaleString('en-AU', { weekday: 'short', hour: 'numeric', minute: '2-digit' });
      const purpose = next.queued
        ? `Returning their call from ${when}: they hung up waiting while the line was busy. Say sorry for the wait, find out what they needed, and help them as on any call.`
        : `Returning their missed call from ${when}: find out what they needed, and help them as on any call.`;
      this.warm(next.number, purpose);
      try {
        await desktop.command('aokie', 'call.dial', {
          number: next.number,
          openingLine: next.queued ? QUEUE_CALL_BACK_LINE : settings.callBackLine.trim() || DEFAULT_CALL_BACK_LINE,
          purpose,
        }, `oaiy:callback:${next.number}:${next.missedAt}:${next.tries + 1}`);
        Object.assign(next, { state: 'calling', tries: next.tries + 1, nextAt: now, note: undefined });
      } catch (error) {
        const message = (error as Error).message;
        Object.assign(next, { nextAt: retryAfter(message, now), note: `The phone did not call: ${message}` });
      }
      await this.save();
    } finally {
      this.ticking = false;
    }
  }

  private async save(): Promise<void> {
    // Kept: the open ones, and the last few settled (to show).
    const settled = this.list.filter((c) => c.state === 'done' || c.state === 'dropped').slice(-20);
    this.list = [...this.list.filter((c) => c.state === 'waiting' || c.state === 'calling'), ...settled];
    await this.project.saveCallbacks(this.list);
    this.changed();
  }
}

/** What the receptionist says first when they answer a call back. */
export const DEFAULT_CALL_BACK_LINE = "Hi, it's the receptionist, returning your call from earlier. How can I help?";
/** The same, to someone who hung up waiting in the queue. */
export const QUEUE_CALL_BACK_LINE = "Hi, it's the receptionist, returning your call. Sorry you were kept waiting earlier. How can I help?";
