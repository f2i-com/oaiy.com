/**
 * The conversations to switch between, in a searchable picker: the button
 * shows the one on screen (who, and their number or what it is), and the list
 * holds them all by kind (yours, calls, texts, flows' tasks), each with when
 * it last heard or said something, its unread count, and a way to remove a
 * finished one.
 */
import { Combobox, type ComboItem } from './combobox';
import { h } from './dom';
import { icon } from './icons';
import { ago, formatNumber, initials } from './chat/transcript';

/** One conversation, as the app lists it. The fields after `close` are optional: without them it is read from the label. */
export interface ConversationTab {
  id: string | null;
  label: string;
  title?: string;
  status?: string;
  unread: number;
  working: boolean;
  close?: () => void;
  /** What it is: the person's own (in a project, "Set up OAIY", or the Front desk's runner), a call, a text thread or a flow's tasks. */
  kind?: ConversationKind;
  /** Who it is with (a name, or the number), or the flow's name. */
  name?: string;
  /** Their number (a flow's tasks: the flow's name). */
  key?: string;
  /** When it last heard or said something (ms). */
  lastAt?: number;
  /** On a call now. */
  live?: boolean;
}

export type ConversationKind = 'project' | 'runner' | 'setup' | 'call' | 'sms' | 'task';

/** The id the person's own conversation (null) goes by in the list. */
const OWN = '\u0000own';
const GROUPS = ['Yours', 'Calls', 'Texts', 'Flow tasks'];
const TEST = 'test';

/** A tab's kind: given, or read from its label's mark (a label from before kinds were given). */
export function tabKind(tab: ConversationTab): ConversationKind {
  if (tab.kind) return tab.kind;
  if (tab.id === null) return tab.label.includes('🧭') ? 'runner' : 'project';
  if (tab.label.includes('📞')) return 'call';
  if (tab.label.includes('🔀')) return 'task';
  return 'sms';
}

/** A tab's name: given, or its label without its mark. */
export function tabName(tab: ConversationTab): string {
  return tab.name ?? (tab.label.replace(/^[^\p{L}\p{N}+(]+/u, '').trim() || tab.label);
}

const KIND_ICON: Record<ConversationKind, string> = { project: 'sparkle', runner: 'compass', setup: 'settings', call: 'phone', sms: 'message', task: 'flow' };

/** A conversation's avatar: the person's initials, or its kind's icon. */
export function conversationAvatar(kind: ConversationKind, name: string, live = false): HTMLElement {
  const letters = kind === 'call' || kind === 'sms' ? initials(name) : '';
  return h('span.convo-avatar', { class: `kind-${kind}${live ? ' live' : ''}`, 'aria-hidden': 'true' }, letters ? h('span.convo-initials', letters) : icon(KIND_ICON[kind]), ...(letters ? [h('span.convo-kind', icon(KIND_ICON[kind]))] : []));
}

/** The line under a conversation's name: its number and kind, or what it is. */
export function conversationDetail(tab: ConversationTab, forButton = false): string {
  const kind = tabKind(tab);
  if (tab.live) return 'On a call now';
  if (tab.working && !forButton) return 'Working…';
  const name = tabName(tab);
  const key = tab.key ?? '';
  if (kind === 'sms' && key === TEST) return 'Test conversation: replies are shown, not sent';
  if ((kind === 'call' || kind === 'sms') && key) {
    const number = key !== name && /\d{4,}/.test(key) ? formatNumber(key) : '';
    const what = kind === 'call' ? 'Calls' : 'Text messages';
    return number ? `${number} · ${what}` : what;
  }
  if (kind === 'task') return 'Tasks from a flow';
  return tab.status ?? '';
}

export class SessionPicker {
  private tabs: ConversationTab[] = [];
  private select: (id: string | null) => void = () => {};
  private readonly combo = new Combobox({
    label: 'Conversations',
    placeholder: 'Search by name, number or kind…',
    groups: GROUPS,
    className: 'session-picker',
    title: "All the conversations: yours, the calls, the texts and the flows' tasks",
    choose: (id) => this.select(id === OWN ? null : id),
    button: (current) => this.buttonContent(current),
  });
  readonly element = this.combo.element;

  /** The conversations, the one shown, and what choosing one does. */
  set(tabs: ConversationTab[], active: string | null, select: (id: string | null) => void): void {
    this.tabs = tabs;
    this.select = select;
    const items: ComboItem[] = tabs.map((tab) => {
      const kind = tabKind(tab);
      const name = kind === 'sms' && tab.key === TEST ? 'Test' : tabName(tab);
      const group = kind === 'project' || kind === 'runner' || kind === 'setup' ? 'Yours' : kind === 'call' ? 'Calls' : kind === 'sms' ? 'Texts' : 'Flow tasks';
      return {
        id: tab.id ?? OWN,
        label: name,
        detail: conversationDetail(tab),
        meta: tab.lastAt ? ago(tab.lastAt) : '',
        group,
        kind,
        keywords: [tab.key, tab.key ? formatNumber(tab.key) : '', tab.status, tab.title, kind === 'call' ? 'call phone' : kind === 'sms' ? 'text sms message' : kind === 'task' ? 'flow task' : 'mine own'].filter(Boolean).join(' '),
        icon: () => conversationAvatar(kind, name, !!tab.live),
        badge: tab.unread || undefined,
        pulse: tab.live ? 'live' : tab.working ? 'working' : undefined,
        title: tab.title ?? name,
        remove: tab.close ? { label: 'Remove this conversation', run: tab.close } : undefined,
      };
    });
    this.combo.set(items, active ?? OWN);
  }

  /** The button: the conversation shown, who with, and how many others (with their unread count). */
  private buttonContent(current: ComboItem | null): Array<Node | string> {
    const tab = this.tabs.find((t) => (t.id ?? OWN) === current?.id) ?? this.tabs[0];
    if (!tab) return [h('span.convo-name', 'Conversations')];
    const kind = tabKind(tab);
    const name = current?.label ?? tabName(tab);
    const others = this.tabs.filter((t) => t !== tab);
    const unread = others.reduce((n, t) => n + t.unread, 0);
    return [
      conversationAvatar(kind, name, !!tab.live),
      h(
        'span.convo-text',
        h('span.convo-name', { class: tab.working || tab.live ? 'working' : '' }, name, ...(tab.live ? [h('span.convo-live', 'Live')] : [])),
        h('span.convo-detail', conversationDetail(tab, true)),
      ),
      h('span.convo-more', `${others.length} other${others.length === 1 ? '' : 's'}`),
      ...(unread ? [h('span.convo-unread', { 'aria-label': `${unread} unread elsewhere` }, String(unread))] : []),
    ];
  }
}
