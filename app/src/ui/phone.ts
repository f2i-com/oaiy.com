/**
 * The Phone dialog: pairing this page with OAIY Desktop (which runs Aokie, the
 * phone bridge), whether the agent answers text messages, and the person's
 * instructions for them. A pretend text tries it without the phone. While no
 * plugin on the desktop provides the phone, only the pairing shows, as the
 * OAIY Desktop dialog.
 */
import { DESKTOP_ORIGIN, desktopHealth, pairingStatus, startPairing } from '../desktop/bridge';
import { AU_PATTERN, DEFAULT_CALL_BACK_LINE, type Callback, type CallBackFilter, type Screening } from '../callbacks';
import { COUNTRIES, countryOf, detectCountry, displayNumber } from '../phoneNumbers';
import type { DesktopSettings, MessageSettings } from '../settings';
import { h } from './dom';
import { askText, modal } from './modal';

export interface PhoneDialog {
  desktop: DesktopSettings | null;
  /** The page is in OAIY's own window: the desktop is given, nothing to pair. */
  given?: boolean;
  /** A plugin on the desktop provides the phone (false: only the pairing shows). */
  phone?: boolean;
  /** Why this page does not answer texts although answering is on (another page does). */
  elsewhere?: string;
  messages: MessageSettings;
  /** A pairing was approved (or forgotten: null). */
  paired: (desktop: DesktopSettings | null) => void;
  /** Pretend a text message arrived, to try the agent without the phone. */
  test: (body: string) => void;
  /** Aokie's call screening (null: the phone is not reachable), and how to change it. */
  screening?: { load: () => Promise<Screening | null>; save: (screening: Screening) => Promise<void> };
  /** Missed calls waiting to be rung back, and the last few settled. */
  callbacks?: Callback[];
}

export async function editPhone(options: PhoneDialog): Promise<MessageSettings | null> {
  let desktop = options.desktop;
  const phone = options.phone !== false;
  const status = h('p.muted', options.given ? (phone ? "This is OAIY's own window: texts and calls to the phone come here." : "This is OAIY's own window: it is OAIY Desktop's.") : `Looking for OAIY Desktop at ${desktop?.origin ?? DESKTOP_ORIGIN}. This page reaches out to your computer only from here, so your browser may ask whether it may connect to your network: that is this.`);
  const code = h('div.pair-code');
  code.hidden = true;
  let pairing: AbortController | null = null;

  const pair = h('button', { type: 'button', onclick: () => void runPairing() }, 'Pair with OAIY Desktop');
  const forget = h('button', { type: 'button', onclick: () => {
    pairing?.abort();
    desktop = null;
    options.paired(null);
    void refresh();
  } }, 'Forget the pairing');

  async function refresh(): Promise<void> {
    const origin = desktop?.origin ?? DESKTOP_ORIGIN;
    const health = await desktopHealth(origin, AbortSignal.timeout(3000));
    pair.hidden = !health || !!desktop;
    forget.hidden = !desktop || !!options.given;
    if (!phone) {
      // No phone: pairing brings the desktop's flows, calendar and speech to this page.
      status.textContent = options.given
        ? health ? `This is OAIY's own window (OAIY Desktop ${health.version}).` : 'OAIY Desktop is not answering.'
        : !health
        ? `OAIY Desktop is not running at ${origin}. Start it to use its flows and speech from this page.`
        : desktop
          ? `Paired with OAIY Desktop ${health.version} at ${origin}.`
          : `OAIY Desktop ${health.version} is running at ${origin}. Pair with it to use its flows and speech from this page.`;
      return;
    }
    status.textContent = options.given
      ? health ? `This is OAIY's own window: texts and calls to the phone come here (OAIY Desktop ${health.version}).` : 'OAIY Desktop is not answering.'
      : !health
      ? `OAIY Desktop is not running at ${origin}. Start it: it runs Aokie, which connects the phone.`
      : desktop
        ? `Paired with OAIY Desktop ${health.version} at ${origin}. Text messages to the phone come here.`
        : `OAIY Desktop ${health.version} is running at ${origin}. Pair with it so text messages to the phone come here.`;
  }

  async function runPairing(): Promise<void> {
    pairing?.abort();
    const abort = (pairing = new AbortController());
    pair.disabled = true;
    try {
      const request = await startPairing(DESKTOP_ORIGIN, `OAIY on ${location.host}`);
      code.textContent = request.code;
      code.hidden = false;
      status.textContent = 'In OAIY Desktop, approve the pairing request that shows this code:';
      const until = Date.now() + 5 * 60_000;
      while (!abort.signal.aborted && Date.now() < until) {
        await new Promise((r) => setTimeout(r, 1500));
        const now = await pairingStatus(DESKTOP_ORIGIN, request.pairingId);
        if (now.status === 'approved' && now.token) {
          desktop = { origin: DESKTOP_ORIGIN, token: now.token };
          options.paired(desktop);
          code.hidden = true;
          await refresh();
          return;
        }
        if (now.status === 'denied' || now.status === 'expired') {
          code.hidden = true;
          status.textContent = now.status === 'denied' ? 'The pairing was denied in OAIY Desktop.' : 'The pairing request expired: try again.';
          return;
        }
      }
    } catch (error) {
      status.textContent = `Could not pair: ${(error as Error).message}`;
    } finally {
      pair.disabled = false;
    }
  }

  const answer = h('input', { type: 'checkbox', checked: options.messages.answer }) as HTMLInputElement;
  const instructions = h('textarea', { placeholder: 'How the agent answers text messages: what it may say, what it should never promise, when to leave it to you…' }) as HTMLTextAreaElement;
  instructions.value = options.messages.instructions;
  // The country a number written without its own is read for: a person's calls and texts are then one conversation.
  const automatic = countryOf(detectCountry());
  const country = h('select', {},
    h('option', { value: '' }, `Automatic: ${automatic.name} (+${automatic.dial})`),
    ...[...COUNTRIES].sort((a, b) => a.name.localeCompare(b.name)).map((c) => h('option', { value: c.code }, `${c.name} (+${c.dial})`)),
  ) as HTMLSelectElement;
  country.value = options.messages.country ?? '';
  const countryNote = h('p.muted.phone-note');
  const samples: Record<string, string> = { AU: '+61491570006', NZ: '+64211234567', GB: '+447700900123', IE: '+353851234567', US: '+14155550132', CA: '+14165550132' };
  const showCountry = () => {
    const c = countryOf(country.value || automatic.code);
    const sample = samples[c.code] ? `, like ${displayNumber(samples[c.code], c)},` : '';
    countryNote.textContent = `A number written without its country code${sample} is read as ${c.name}'s (+${c.dial}), so each person's calls and texts are one conversation however the phone writes their number.`;
  };
  country.addEventListener('change', showCountry);
  showCountry();
  const calls = h('input', { type: 'checkbox', checked: options.messages.calls }) as HTMLInputElement;
  const callInstructions = h('textarea', { placeholder: 'Added to the receptionist brief Aokie sends with each call: what the agent may say on the phone, when to take a message…' }) as HTMLTextAreaElement;
  callInstructions.value = options.messages.callInstructions;
  // Who is answered: Aokie's screening, read from the phone (its own settings).
  const accept = h('select', {},
    h('option', { value: 'any' }, 'Any number'),
    h('option', { value: 'au' }, 'Australian numbers only'),
    h('option', { value: 'pattern' }, 'Numbers matching a pattern…'),
  ) as HTMLSelectElement;
  const pattern = h('input', { placeholder: 'A regular expression the caller id must match, e.g. ^\\+?61' }) as HTMLInputElement;
  const blocked = h('textarea', { placeholder: 'One number a line: never answered (any format: 0400 000 000, +61400000000)' }) as HTMLTextAreaElement;
  const rejectPrivate = h('input', { type: 'checkbox' }) as HTMLInputElement;
  const screenNote = h('p.muted', 'Reading the phone\'s call screening…');
  const screenFields = h('div.phone-screening', { hidden: true },
    h('label', 'Answer calls from', accept),
    pattern,
    h('label', 'Blocked numbers', blocked),
    h('label.row', rejectPrivate, ' Do not answer callers who hide their number'),
  );
  const showPattern = () => (pattern.hidden = accept.value !== 'pattern');
  accept.addEventListener('change', showPattern);
  let screeningLoaded: Screening | null = null;
  if (options.screening && phone) {
    void options.screening.load().then((s) => {
      screeningLoaded = s;
      if (!s) {
        screenNote.textContent = 'The phone is not reachable now: who is answered is set here once OAIY Desktop and Aokie are running.';
        return;
      }
      accept.value = !s.acceptPattern.trim() ? 'any' : s.acceptPattern.trim() === AU_PATTERN ? 'au' : 'pattern';
      pattern.value = accept.value === 'pattern' ? s.acceptPattern : '';
      blocked.value = s.blockedNumbers.split(/[,;\n]/).map((n) => n.trim()).filter(Boolean).join('\n');
      rejectPrivate.checked = s.rejectPrivate;
      showPattern();
      screenNote.hidden = true;
      screenFields.hidden = false;
    }, (e: unknown) => (screenNote.textContent = `Could not read the phone's call screening: ${(e as Error).message}`));
  } else screenNote.textContent = 'Who is answered is set here once OAIY Desktop is connected.';

  // Missed calls rung back.
  const callBack = h('input', { type: 'checkbox', checked: options.messages.callBack }) as HTMLInputElement;
  const callBackFilter = h('select', {},
    h('option', { value: 'answered' }, 'The numbers it answers'),
    h('option', { value: 'au' }, 'Australian numbers only'),
    h('option', { value: 'any' }, 'Any number'),
  ) as HTMLSelectElement;
  callBackFilter.value = options.messages.callBackFilter;
  const callBackLine = h('input', { value: options.messages.callBackLine, placeholder: DEFAULT_CALL_BACK_LINE }) as HTMLInputElement;
  const waiting = (options.callbacks ?? []).filter((c) => c.state === 'waiting' || c.state === 'calling');
  const settled = (options.callbacks ?? []).filter((c) => c.state === 'done' || c.state === 'dropped').slice(-5).reverse();
  const when = (ms: number) => new Date(ms).toLocaleString(undefined, { weekday: 'short', hour: 'numeric', minute: '2-digit' });
  const callbackList = h('ul.phone-callbacks',
    ...waiting.map((c) => h('li', `${c.number}: missed ${when(c.missedAt)}, ${c.state === 'calling' ? 'ringing them now' : c.tries ? `tried ${c.tries}×, next ${when(c.nextAt)}` : 'to call back'}${c.note ? ` (${c.note})` : ''}`)),
    ...settled.map((c) => h('li.muted', `${c.number}: missed ${when(c.missedAt)}. ${c.note ?? ''}`)),
  );
  callbackList.hidden = !waiting.length && !settled.length;

  const test = h('button', { type: 'button', title: 'Pretend a text message arrived: the agent answers it in a test conversation, and nothing is sent', onclick: async () => {
    const body = await askText({ title: 'A pretend text message', message: 'The agent answers it in a test conversation. Nothing is sent to a phone.', label: 'The message', value: 'Hi, are you open this Saturday?', ok: 'Send it to the agent' });
    if (body) options.test(body);
  } }, 'Try a pretend text…');

  void refresh();
  if (!phone) {
    // No plugin provides the phone: the dialog is OAIY Desktop's pairing, and nothing else.
    await modal<MessageSettings>({
      title: 'OAIY Desktop',
      message: 'Pair this page with OAIY Desktop to use its flows and speech here. Calls and texts come once a plugin on it provides the phone (the AI Receptionist).',
      body: [h('div.phone-form', status, code, h('div.row', pair, forget))],
      ok: { label: 'Done', value: () => null as unknown as MessageSettings },
      cancel: 'Close',
    });
    (pairing as AbortController | null)?.abort();
    return null;
  }
  const result = await modal<MessageSettings>({
    title: 'Phone',
    message: 'Calls and text messages come through Aokie, which OAIY Desktop runs with the Bluetooth adapter and your phone.',
    wide: true,
    body: [h('div.phone-form',
      status,
      code,
      h('div.row', pair, forget),
      h('label.row', answer, ' Answer text messages as they arrive (every text to your phone, sent from your number)'),
      ...(options.elsewhere ? [h('p.muted', options.elsewhere)] : []),
      h('label', 'Your instructions for text messages', instructions),
      h('div.row', test),
      h('label.row', calls, ' Answer phone calls (the agent talks with the caller, and you see the call here)'),
      h('label', 'Your instructions for calls', callInstructions),
      h('label', 'Country for local numbers', country),
      countryNote,
      h('h3.phone-heading', 'Who is answered'),
      screenNote,
      screenFields,
      h('h3.phone-heading', 'Missed calls'),
      h('label.row', callBack, ' Call back missed calls when the receptionist is free (turns on outbound calling in Aokie, within its quiet hours and daily limit)'),
      h('label', 'Call back', callBackFilter),
      h('label', 'What it says first when they answer', callBackLine),
      callbackList,
    )],
    ok: { label: 'Save', value: () => ({
      answer: answer.checked,
      instructions: instructions.value.trim(),
      calls: calls.checked,
      callInstructions: callInstructions.value.trim(),
      callBack: callBack.checked,
      callBackFilter: callBackFilter.value as CallBackFilter,
      callBackLine: callBackLine.value.trim(),
      country: country.value,
    }) },
    cancel: 'Close',
  });
  (pairing as AbortController | null)?.abort();
  // Who is answered: saved to the phone when it was read from it (and changed).
  if (result && options.screening && screeningLoaded) {
    const next: Screening = {
      acceptPattern: accept.value === 'au' ? AU_PATTERN : accept.value === 'pattern' ? pattern.value.trim() : '',
      blockedNumbers: blocked.value.split(/\n/).map((n) => n.trim()).filter(Boolean).join(', '),
      rejectPrivate: rejectPrivate.checked,
    };
    const was = screeningLoaded as Screening;
    if (next.acceptPattern !== was.acceptPattern || next.blockedNumbers !== was.blockedNumbers.split(/[,;\n]/).map((n) => n.trim()).filter(Boolean).join(', ') || next.rejectPrivate !== was.rejectPrivate) await options.screening.save(next);
  }
  return result;
}
