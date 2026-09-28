/**
 * The Phone dialog: pairing this page with OAIY Desktop (which runs Aokie, the
 * phone bridge), whether the agent answers text messages, and the person's
 * instructions for them. A pretend text tries it without the phone.
 */
import { DESKTOP_ORIGIN, desktopHealth, pairingStatus, startPairing } from '../desktop/bridge';
import type { DesktopSettings, MessageSettings } from '../settings';
import { h } from './dom';
import { askText, modal } from './modal';

export interface PhoneDialog {
  desktop: DesktopSettings | null;
  /** The page is in OAIY's own window: the desktop is given, nothing to pair. */
  given?: boolean;
  /** Why this page does not answer texts although answering is on (another page does). */
  elsewhere?: string;
  messages: MessageSettings;
  /** A pairing was approved (or forgotten: null). */
  paired: (desktop: DesktopSettings | null) => void;
  /** Pretend a text message arrived, to try the agent without the phone. */
  test: (body: string) => void;
}

export async function editPhone(options: PhoneDialog): Promise<MessageSettings | null> {
  let desktop = options.desktop;
  const status = h('p.muted', 'Looking for OAIY Desktop…');
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
    status.textContent = options.given
      ? health ? `This is OAIY's own window: text messages to the phone come here (OAIY Desktop ${health.version}).` : 'OAIY Desktop is not answering.'
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
  const test = h('button', { type: 'button', title: 'Pretend a text message arrived: the agent answers it in a test conversation, and nothing is sent', onclick: async () => {
    const body = await askText({ title: 'A pretend text message', message: 'The agent answers it in a test conversation. Nothing is sent to a phone.', label: 'The message', value: 'Hi, are you open this Saturday?', ok: 'Send it to the agent' });
    if (body) options.test(body);
  } }, 'Try a pretend text…');

  void refresh();
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
    )],
    ok: { label: 'Save', value: () => ({ answer: answer.checked, instructions: instructions.value.trim() }) },
    cancel: 'Close',
  });
  (pairing as AbortController | null)?.abort();
  return result;
}
