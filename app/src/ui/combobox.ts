/**
 * A searchable picker: a button that shows the current choice, and a popover
 * with a search field (focused as it opens) over a grouped list. It follows
 * WAI-ARIA's combobox pattern: the field is the combobox, the list its
 * listbox, and the option in play is its active descendant, so focus stays in
 * the field while Up and Down, Home and End, Page Up and Down move through the
 * list, Enter chooses and Escape closes. Typing on the closed button opens it
 * with what was typed. Used for the conversations and the projects.
 */
import { clear, h } from './dom';
import { icon } from './icons';

export interface ComboItem {
  id: string;
  /** Its name. */
  label: string;
  /** A line under the name. */
  detail?: string;
  /** Beside the name, at the end (a time). */
  meta?: string;
  /** The group it is listed under. */
  group?: string;
  /** More words it is found by (a phone number, a status). */
  keywords?: string;
  /** Its icon, drawn afresh each time. */
  icon?: () => Node;
  /** A count on it (unread messages). */
  badge?: number;
  /** A dot that pulses (working, or on a call). */
  pulse?: 'working' | 'live';
  /** A class for the row (its kind). */
  kind?: string;
  /** Its tooltip. */
  title?: string;
  /** A way to remove it, shown on hover or focus. */
  remove?: { label: string; run: () => void };
}

/** The words of a search, lower case. */
function terms(query: string): string[] {
  return query.toLowerCase().split(/\s+/).filter(Boolean);
}

/** Digits only (a number is found however it is spaced: "0491 570", "+61491570"). */
const digitsOf = (text: string) => text.replace(/\D/g, '');

/**
 * The items a search finds, best first: every word of it must be in the name,
 * the line under it or its keywords (a word of digits matches a number however
 * it is written). A name that starts with the search comes first, then a name
 * with a word that does, then the rest, each in the order given.
 */
export function filterItems<T extends Pick<ComboItem, 'label' | 'detail' | 'keywords' | 'group'>>(items: T[], query: string): T[] {
  const words = terms(query);
  if (!words.length) return items.slice();
  const scored: Array<{ item: T; score: number; index: number }> = [];
  items.forEach((item, index) => {
    const label = item.label.toLowerCase();
    const hay = [item.label, item.detail, item.keywords, item.group].filter(Boolean).join(' ').toLowerCase();
    const hayDigits = digitsOf(hay);
    const found = words.every((w) => hay.includes(w) || (/^\+?[\d\s-]{3,}$/.test(w) && digitsOf(w).length >= 3 && hayDigits.includes(digitsOf(w))));
    if (!found) return;
    const first = words[0];
    const score = label.startsWith(query.trim().toLowerCase()) ? 0 : label.split(/[\s()+-]+/).some((part) => part.startsWith(first)) ? 1 : label.includes(first) ? 2 : 3;
    scored.push({ item, score, index });
  });
  return scored.sort((a, b) => a.score - b.score || a.index - b.index).map((s) => s.item);
}

/**
 * Items under their groups, the groups in `order` (then any others, as they
 * come). While searching, the best match leads: its group goes first.
 */
export function groupItems<T extends Pick<ComboItem, 'group'>>(items: T[], order: string[] = [], bestFirst = false): Array<{ group: string; items: T[] }> {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const g = item.group ?? '';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g)!.push(item);
  }
  const names = [...groups.keys()];
  const rank = (g: string) => (order.includes(g) ? order.indexOf(g) : order.length + names.indexOf(g));
  const lead = bestFirst && items.length ? (items[0].group ?? '') : null;
  names.sort((a, b) => (a === lead ? -1 : b === lead ? 1 : rank(a) - rank(b)));
  return names.map((group) => ({ group, items: groups.get(group)! }));
}

/** The keys that move through the list. */
export type MoveKey = 'ArrowDown' | 'ArrowUp' | 'Home' | 'End' | 'PageDown' | 'PageUp';

/** Where the active option goes for a key, over `count` options (-1: none). Up and Down wrap; the page keys go `page` at a time and stop at the ends. */
export function moveActive(current: number, key: MoveKey, count: number, page = 8): number {
  if (count <= 0) return -1;
  switch (key) {
    case 'ArrowDown': return current < 0 ? 0 : (current + 1) % count;
    case 'ArrowUp': return current < 0 ? count - 1 : (current - 1 + count) % count;
    case 'Home': return 0;
    case 'End': return count - 1;
    case 'PageDown': return Math.min(count - 1, Math.max(0, current) + page);
    case 'PageUp': return Math.max(0, (current < 0 ? count - 1 : current) - page);
  }
}

/** What a key in the search field does: move, choose, close, remove the active option, or nothing (it types). */
export function keyAction(e: Pick<KeyboardEvent, 'key' | 'altKey' | 'ctrlKey' | 'metaKey' | 'shiftKey' | 'isComposing'>, caretAtEnd: boolean): { move: MoveKey } | 'choose' | 'close' | 'remove' | null {
  if (e.isComposing) return null;
  switch (e.key) {
    case 'ArrowDown': case 'ArrowUp': case 'PageDown': case 'PageUp':
      return e.altKey && e.key === 'ArrowUp' ? 'close' : { move: e.key };
    case 'Home': case 'End':
      // Shift+Home/End still selects the search's text.
      return e.shiftKey ? null : { move: e.key };
    case 'Enter': return 'choose';
    case 'Escape': return 'close';
    case 'Delete': return caretAtEnd && !e.shiftKey ? 'remove' : null;
    default: return null;
  }
}

let seq = 0;

export interface ComboOptions {
  /** The popover's name, for screen readers ("Conversations"). */
  label: string;
  placeholder: string;
  /** What the button shows for the current item (null when there is none). */
  button: (current: ComboItem | null, items: ComboItem[]) => Array<Node | string>;
  choose: (id: string) => void;
  /** The groups' order. */
  groups?: string[];
  /** A class for the whole picker. */
  className?: string;
  /** The button's tooltip. */
  title?: string;
}

export class Combobox {
  readonly element: HTMLElement;
  private readonly id = `combo-${++seq}`;
  private readonly button: HTMLButtonElement;
  private readonly pop: HTMLElement;
  private readonly search: HTMLInputElement;
  private readonly list: HTMLElement;
  private readonly empty: HTMLElement;
  private items: ComboItem[] = [];
  private current: string | null = null;
  /** The options shown now, in list order. */
  private shown: ComboItem[] = [];
  private active = -1;
  private isOpen = false;

  constructor(private readonly options: ComboOptions) {
    this.button = h('button.combo-button', {
      type: 'button',
      'aria-haspopup': 'listbox',
      'aria-expanded': 'false',
      'aria-controls': `${this.id}-pop`,
      title: options.title,
      onclick: () => (this.isOpen ? this.close() : this.open()),
      onkeydown: (e: KeyboardEvent) => this.buttonKey(e),
    }) as HTMLButtonElement;
    this.search = h('input.combo-search', {
      type: 'text',
      role: 'combobox',
      'aria-expanded': 'true',
      'aria-controls': `${this.id}-list`,
      'aria-autocomplete': 'list',
      'aria-label': `Search ${options.label.toLowerCase()}`,
      placeholder: options.placeholder,
      autocomplete: 'off',
      spellcheck: false,
      oninput: () => this.render(true),
      onkeydown: (e: KeyboardEvent) => this.searchKey(e),
    }) as HTMLInputElement;
    this.list = h('div.combo-list', { role: 'listbox', id: `${this.id}-list`, 'aria-label': options.label });
    this.empty = h('div.combo-empty', { role: 'status' });
    this.pop = h('div.combo-pop', { id: `${this.id}-pop`, hidden: true }, h('div.combo-field', icon('search'), this.search), this.list, this.empty);
    this.element = h('div.combo', { class: options.className ?? '' }, this.button, this.pop);
    // A click elsewhere closes it (not a click in a dialog it opened, such as removing a conversation's confirm).
    document.addEventListener('pointerdown', (e) => {
      if (this.isOpen && !this.element.contains(e.target as Node) && !(e.target as Element | null)?.closest?.('dialog')) this.close(false);
    });
    this.element.addEventListener('focusout', (e) => {
      const to = e.relatedTarget as Node | null;
      if (this.isOpen && to && !this.element.contains(to) && !(to as Element).closest?.('dialog')) this.close(false);
    });
    this.list.addEventListener('pointermove', (e) => {
      const row = (e.target as Element).closest<HTMLElement>('.combo-option');
      const i = row ? this.shown.findIndex((item) => this.optionId(item) === row.id) : -1;
      if (i >= 0 && i !== this.active) this.setActive(i, false);
    });
    // Keep the field focused when an option is pressed.
    this.list.addEventListener('mousedown', (e) => e.preventDefault());
  }

  /** The items and the current one. An open picker keeps its search and the option in play. */
  set(items: ComboItem[], current: string | null): void {
    this.items = items;
    this.current = current;
    clear(this.button);
    this.button.append(...this.options.button(items.find((i) => i.id === current) ?? null, items), icon('chevron-down', 'combo-caret'));
    if (this.isOpen) this.render(false);
  }

  get opened(): boolean {
    return this.isOpen;
  }

  open(query = ''): void {
    this.isOpen = true;
    this.pop.hidden = false;
    this.element.classList.add('open');
    this.button.setAttribute('aria-expanded', 'true');
    this.search.value = query;
    this.active = -1;
    this.render(true);
    this.search.focus();
    if (query) this.search.setSelectionRange(query.length, query.length);
  }

  close(focusButton = true): void {
    if (!this.isOpen) return;
    this.isOpen = false;
    this.pop.hidden = true;
    this.element.classList.remove('open');
    this.button.setAttribute('aria-expanded', 'false');
    this.search.removeAttribute('aria-activedescendant');
    if (focusButton) this.button.focus();
  }

  private optionId(item: ComboItem): string {
    return `${this.id}-o-${item.id.replace(/[^\w-]/g, '_')}`;
  }

  /** Draw the list for the search now; `fresh` puts the active option on the best match (or the current item). */
  private render(fresh: boolean): void {
    const query = this.search.value;
    const activeId = this.shown[this.active]?.id;
    const found = filterItems(this.items, query);
    const groups = groupItems(found, this.options.groups, !!query.trim());
    this.shown = groups.flatMap((g) => g.items);
    clear(this.list);
    for (const { group, items } of groups) {
      const labelId = `${this.id}-g-${group.replace(/\W/g, '_') || 'none'}`;
      this.list.append(h(
        'div.combo-group',
        { role: 'group', ...(group ? { 'aria-labelledby': labelId } : {}) },
        ...(group ? [h('div.combo-group-label', { id: labelId }, h('span', group), h('span.combo-count', { 'aria-label': `${items.length} ${items.length === 1 ? 'item' : 'items'}` }, String(items.length)))] : []),
        ...items.map((item) => this.option(item)),
      ));
    }
    this.empty.hidden = this.shown.length > 0;
    this.empty.textContent = this.shown.length ? '' : query.trim() ? `No matches for "${query.trim()}"` : 'Nothing here yet';
    let next = fresh ? (query.trim() ? 0 : this.shown.findIndex((i) => i.id === this.current)) : this.shown.findIndex((i) => i.id === activeId);
    if (next < 0 && this.shown.length) next = 0;
    this.setActive(next, fresh);
  }

  private option(item: ComboItem): HTMLElement {
    const isCurrent = item.id === this.current;
    const remove = item.remove;
    return h(
      'div.combo-option',
      {
        id: this.optionId(item),
        role: 'option',
        'aria-selected': 'false',
        class: [isCurrent ? 'current' : '', item.pulse ?? ''].filter(Boolean).join(' '),
        title: item.title ?? '',
        'data-kind': item.kind ?? '',
        'data-id': item.id,
        onclick: (e: MouseEvent) => {
          if ((e.target as Element).closest('.combo-remove')) return;
          this.pick(item);
        },
      },
      h('span.combo-icon', { 'aria-hidden': 'true' }, item.icon?.() ?? ''),
      h(
        'span.combo-text',
        h('span.combo-label', h('span.combo-name', item.label), ...(item.pulse ? [h('span.combo-pulse', { 'aria-label': item.pulse === 'live' ? 'on a call now' : 'working' })] : [])),
        ...(item.detail ? [h('span.combo-detail', item.detail)] : []),
      ),
      h(
        'span.combo-end',
        ...(item.meta ? [h('span.combo-meta', item.meta)] : []),
        ...(item.badge ? [h('span.combo-badge', { 'aria-label': `${item.badge} unread` }, String(item.badge))] : []),
        ...(isCurrent ? [h('span.combo-check', { 'aria-label': '(showing)' }, icon('check'))] : []),
      ),
      ...(remove
        ? [h('button.combo-remove', { type: 'button', tabindex: -1, title: remove.label, 'aria-label': `${remove.label}: ${item.label}`, onclick: (e: Event) => {
            e.stopPropagation();
            this.close(false);
            remove.run();
          } }, icon('x'))]
        : []),
    );
  }

  private setActive(index: number, scroll: boolean): void {
    this.active = index;
    let activeEl: HTMLElement | null = null;
    for (const [i, item] of this.shown.entries()) {
      const el = this.list.querySelector<HTMLElement>(`#${CSS.escape(this.optionId(item))}`);
      if (!el) continue;
      el.classList.toggle('active', i === index);
      el.setAttribute('aria-selected', String(i === index));
      if (i === index) activeEl = el;
    }
    if (activeEl) {
      this.search.setAttribute('aria-activedescendant', activeEl.id);
      if (scroll) activeEl.scrollIntoView({ block: 'nearest' });
    } else this.search.removeAttribute('aria-activedescendant');
  }

  private pick(item: ComboItem): void {
    this.close();
    if (item.id !== this.current) this.options.choose(item.id);
  }

  private searchKey(e: KeyboardEvent): void {
    const atEnd = this.search.selectionStart === this.search.value.length && this.search.selectionEnd === this.search.value.length;
    const action = keyAction(e, atEnd);
    if (!action) return;
    e.preventDefault();
    if (action === 'close') {
      this.close();
      return;
    }
    if (action === 'choose') {
      const item = this.shown[this.active];
      if (item) this.pick(item);
      return;
    }
    if (action === 'remove') {
      const item = this.shown[this.active];
      if (item?.remove) {
        this.close(false);
        item.remove.run();
      }
      return;
    }
    this.setActive(moveActive(this.active, action.move, this.shown.length), true);
  }

  /** On the closed button: arrows open it; a letter opens it with the search begun. */
  private buttonKey(e: KeyboardEvent): void {
    if (this.isOpen) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      this.open();
    } else if (e.key.length === 1 && e.key !== ' ' && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      this.open(e.key);
    }
  }
}
