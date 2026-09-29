/**
 * Who answers the phone, and for whom: the receptionist's name ("Aokie"
 * unless the person set another) and the business's, from OAIY Desktop's
 * calendar settings (Hours & Services). Every agent a customer talks to (a
 * call's, a text thread's, an outreach's) says them, and never "OAIY" or
 * the app's own word for the owner ("your person"); the runner and a
 * project's agent know them too, so what they write for customers (an
 * outreach's opening line or text) says them.
 */
import type { Desktop } from './desktop/bridge';

export interface Identity {
  /** The business's name ('' while the person has not set it). */
  business: string;
  /** The receptionist's name. */
  receptionist: string;
}

/** The receptionist's name when the desktop says none (one from before it kept a name). */
export const DEFAULT_RECEPTIONIST = 'Aokie';
/** Before the desktop has said. */
export const NO_IDENTITY: Identity = { business: '', receptionist: DEFAULT_RECEPTIONIST };

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const oneLine = (v: unknown) => (typeof v === 'string' ? v.replace(/\s+/g, ' ').trim().slice(0, 80) : '');

/**
 * The identity in `GET /api/calendar`'s answer: its `receptionistName` (filled
 * in by the desktop: "Aokie" unless the person set one; a desktop from before
 * has none, and then it is "Aokie") and `settings.business`.
 */
export function identityFrom(calendar: unknown): Identity {
  const c = isRecord(calendar) ? calendar : {};
  const settings = isRecord(c.settings) ? c.settings : {};
  return { business: oneLine(settings.business), receptionist: oneLine(c.receptionistName) || DEFAULT_RECEPTIONIST };
}

/** What every agent a customer talks to is told about who it is (a call's, a text thread's, an outreach's). */
export function identityInstructions(id: Identity): string {
  const who = id.business ? `You are ${id.receptionist}, the receptionist for ${id.business}.` : `You are ${id.receptionist}, the business's receptionist (its name is not set yet: say "the business", or leave it out).`;
  return `${who} On calls and texts, say those names when you say who you are or who you speak for: never "OAIY", never "your person" (these instructions call the business's owner that; it is never a word for a customer), and nothing about how you work (agents, a runner, tools, notes or instructions).`;
}

/** What the runner and a project's agent are told, so what they write for customers says the right names. */
export function identityNote(id: Identity): string {
  const business = id.business ? ` for ${id.business}` : " (the business's name is not set yet: your person sets it in OAIY's Hours & Services)";
  return `The phone is answered as ${id.receptionist}, the receptionist${business}. What you write for customers (an outreach's opening line or text) speaks as them: write {receptionist} and {business} in outreach templates (they are filled in), and never "OAIY" or "your person".`;
}

/**
 * The identity as the desktop last said it: `get()` answers at once and asks
 * again in the background once `ttlMs` have passed; `refresh()` asks now (as
 * the phone starts, and when the calendar's settings change).
 */
export class IdentityCache {
  private value: Identity = NO_IDENTITY;
  private at = 0;
  private reading: Promise<Identity> | null = null;

  constructor(private readonly desktop: () => Desktop | null, private readonly ttlMs = 5 * 60_000) {}

  get(): Identity {
    if (Date.now() - this.at > this.ttlMs) void this.refresh();
    return this.value;
  }

  refresh(): Promise<Identity> {
    this.reading ??= (async () => {
      const d = this.desktop();
      try {
        if (d) this.value = identityFrom(await d.calendar(undefined, undefined, AbortSignal.timeout(8_000)));
      } catch {
        /* out of reach: what was said last stays */
      } finally {
        this.at = Date.now();
        this.reading = null;
      }
      return this.value;
    })();
    return this.reading;
  }
}
