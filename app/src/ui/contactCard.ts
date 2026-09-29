/**
 * A person's contact, read only, in a small card under their conversation's
 * "Contact" button: their name (and who gave it), the person's notes for the
 * receptionist, and what the receptionist remembered. In a browser tab; OAIY's
 * own window opens the dashboard's Contacts page on them instead, where they
 * are changed.
 */
import { clear, h } from './dom';
import { icon } from './icons';
import { formatNumber } from './chat/transcript';

export interface ContactCardData {
  name: string;
  number: string;
  /** Who named them: the person (in Contacts) or the receptionist. */
  nameBy?: 'owner' | 'agent' | null;
  notes: string;
  /** Facts the person wrote themselves (read with their notes). */
  ownerFacts: string[];
  /** What the receptionist remembered. */
  facts: string[];
  /** Read from OAIY Desktop now, or the Front desk's copy (the desktop out of reach, or not paired). */
  source: 'desktop' | 'copy';
  /** They have no contact on the desktop yet. */
  none?: boolean;
  /** Why the dashboard did not open on them, when this card stands in for it. */
  why?: string;
}

/** The card open now, and how to close it. */
let open: { card: HTMLElement; anchor: HTMLElement; close: () => void } | null = null;

/** The card's contents. */
export function contactCardBody(data: ContactCardData): HTMLElement[] {
  const by = data.name && data.nameBy === 'owner' ? 'named by you' : data.name && data.nameBy === 'agent' ? 'named by the receptionist' : '';
  const out: HTMLElement[] = [
    h('div.contact-head', h('span.contact-avatar', { 'aria-hidden': 'true' }, icon('user')), h('div.contact-who', h('strong.contact-name', data.name || formatNumber(data.number)), h('span.contact-number', [data.name ? formatNumber(data.number) : '', by].filter(Boolean).join(' · ')))),
  ];
  if (data.why) out.push(h('p.contact-note', data.why));
  if (data.none) out.push(h('p.contact-empty', 'No contact for them yet: they get one when they call or text, or when you add them in Contacts.'));
  const notes = data.notes.trim();
  out.push(
    h(
      'section.contact-section',
      h('h4', 'Notes for the receptionist'),
      ...(notes ? [h('p.contact-notes', notes)] : []),
      ...(data.ownerFacts.length ? [h('ul.contact-facts', ...data.ownerFacts.map((f) => h('li', f)))] : []),
      ...(!notes && !data.ownerFacts.length ? [h('p.contact-empty', 'None yet. The receptionist reads these on every call and text with them.')] : []),
    ),
    h(
      'section.contact-section',
      h('h4', 'What the receptionist remembered'),
      data.facts.length ? h('ul.contact-facts', ...data.facts.map((f) => h('li', f))) : h('p.contact-empty', 'Nothing yet.'),
    ),
  );
  if (data.source === 'copy') out.push(h('p.contact-note', 'OAIY Desktop could not be reached: this is what the Front desk last knew.'));
  out.push(h('p.contact-foot', "Change them in OAIY's dashboard: AI Receptionist → Contacts."));
  return out;
}

/**
 * Show a contact's card under `anchor` (in `host`, which it is laid out in),
 * with what `load` reads; the same anchor again closes it, as do Escape and a
 * click elsewhere.
 */
export function showContactCard(host: HTMLElement, anchor: HTMLElement, load: () => Promise<ContactCardData>): void {
  if (open) {
    const same = open.anchor === anchor;
    open.close();
    if (same) return;
  }
  const card = h('div.contact-pop', { role: 'dialog', 'aria-label': 'Contact', tabindex: '-1' }, h('p.contact-empty', 'Reading the contact…'));
  const away = (e: Event) => {
    if (!card.contains(e.target as Node) && !anchor.contains(e.target as Node)) close();
  };
  const key = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      close();
      anchor.focus();
    }
  };
  const close = () => {
    card.remove();
    document.removeEventListener('pointerdown', away, true);
    document.removeEventListener('keydown', key, true);
    anchor.setAttribute('aria-expanded', 'false');
    if (open?.card === card) open = null;
  };
  open = { card, anchor, close };
  host.append(card);
  anchor.setAttribute('aria-expanded', 'true');
  document.addEventListener('pointerdown', away, true);
  document.addEventListener('keydown', key, true);
  void load().then(
    (data) => {
      if (!card.isConnected) return;
      clear(card);
      card.append(...contactCardBody(data));
    },
    (error: unknown) => {
      if (!card.isConnected) return;
      clear(card);
      card.append(h('p.contact-note', `The contact could not be read: ${(error as Error).message}`));
    },
  );
}

/** Close the card, if one is open (the conversation shown changed). */
export function closeContactCard(): void {
  open?.close();
}
