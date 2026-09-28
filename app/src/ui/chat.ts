/**
 * The chat pane: the conversation with the agent, streamed as it arrives,
 * with a card per tool call; and the prompt box, which also takes slash
 * commands.
 */
import type { AgentEvent } from '../agent/agent';
import type { Attachment, ToolCall, ToolResult, Turn } from '../agent/protocol';
import { planAfter, planChanges, readPlan, readTasks, settlePlan, type Plan } from '../agent/tools';
import { formatTokens } from '../agent/context';
import { clear, h } from './dom';
import { mediaElement, mediaKind, type Media } from './media';
import { renderMarkdown } from './markdown';

/** How long a flag waits for a comment before it goes without one. */
const FLAG_WAIT_SECONDS = 10;
/**
 * Streamed text (the reply, thinking, a tool call being written) arrives in
 * hundreds of pieces a second; redrawing a long text for each froze the page
 * and could garble the display. Each box redraws at most this often, with all
 * that has arrived by then.
 */
const STREAM_REDRAW_MS = 80;

/** The prompt a picture, clip or sound is made from, shown on its card without opening it. */
function mediaPrompt(call: ToolCall): string {
  if (!/^generate_(image|video|speech|music)$/.test(call.name)) return '';
  const i = call.input;
  const text = [i.prompt, call.name === 'generate_speech' ? i.input : null, i.lyrics].filter((v): v is string => typeof v === 'string' && !!v.trim());
  return text.join('\n\n');
}

/** What the model was sent for a step, as text to read: the system prompt, the tools, then the conversation. */
function promptText(e: { system: string; turns: Turn[]; tools: string[] }): string {
  const parts = [`━━ SYSTEM PROMPT ━━\n${e.system}`, `━━ TOOLS ━━\n${e.tools.join(', ')}`];
  for (const t of e.turns) {
    if (t.role === 'user') parts.push(`━━ ${t.automatic ? 'BOT.COMPUTER' : 'USER'} ━━\n${t.text}${t.images?.length ? `\n[${t.images.length} image(s)]` : ''}`);
    else if (t.role === 'assistant') parts.push(`━━ MODEL ━━\n${[t.text, ...t.calls.map((c) => `→ ${c.name} ${JSON.stringify(c.input, null, 2)}`)].filter(Boolean).join('\n')}`);
    else parts.push(...t.results.map((r) => `━━ RESULT of ${r.name}${r.isError ? ' (error)' : ''} ━━\n${r.content}${r.images?.length ? `\n[${r.images.length} image(s)]` : ''}`));
  }
  return parts.join('\n\n');
}

function summarizeCall(call: ToolCall): string {
  const i = call.input;
  const s = (k: string) => (typeof i[k] === 'string' ? String(i[k]) : '');
  switch (call.name) {
    case 'read_file': case 'write_file': case 'append_file': case 'edit_file': case 'delete_file': case 'list_files': return s('path') || '/';
    case 'review_frame': return `${s('path')}${i.accept === true ? ' (accept)' : ''}`;
    case 'grep': return `${s('pattern')}${s('path') ? ` in ${s('path')}` : ''}`;
    case 'glob': return s('pattern');
    case 'sandbox_shell': return s('command').split('\n')[0];
    case 'code_run': return `${s('language') || 'javascript'}${s('file') ? ` ${s('file')}` : ''}`;
    case 'web_fetch': return s('url');
    case 'delegate': return Array.isArray(i.tasks) ? `${i.tasks.length} task${i.tasks.length === 1 ? '' : 's'}` : '';
    case 'present_file': case 'view_image': case 'file_info': case 'search_file': return s('path');
    case 'softn_import': return s('path');
    case 'softn_check': case 'softn_inspect': return s('app');
    case 'softn_interact': return Array.isArray(i.actions) ? i.actions.map((a) => Object.entries(a as Record<string, unknown>).filter(([k]) => k !== 'nth').map(([k, v]) => (k === 'value' ? `"${v}"` : `${k} ${typeof v === 'string' ? `"${v}"` : v}`)).join(' ')).join(', ') : '';
    case 'softn_docs': return s('search') ? `search: ${s('search')}` : s('topic') || s('section') || 'map';
    case 'softn_components': return Array.isArray(i.names) ? i.names.join(', ') : s('names');
    case 'softn_examples': return [s('name'), s('file'), s('install_to') && `→ ${s('install_to')}`].filter(Boolean).join(' ') || 'list';
    default: return '';
  }
}

/** Steps after the one in progress that the checklist shows before folding the rest into "+N more". */
const PLAN_UPCOMING = 4;

/** How many turns of a saved conversation are drawn at a time (newest first; older ones as the log is scrolled down to them). */
const REPLAY_PAGE = 60;

/** Where the last page of `turns` starts: `size` turns back, moved back to the request they belong to. */
function pageStart(turns: Turn[], size: number): number {
  let from = Math.max(0, turns.length - size);
  while (from > 0 && turns[from]?.role !== 'user') from--;
  return from;
}

export class ChatPane {
  readonly element = h('section.chat');
  private readonly log = h('div.chat-log', { role: 'log', 'aria-live': 'polite', 'aria-relevant': 'additions' });
  private readonly input = h('textarea.chat-input', { rows: 3, placeholder: 'Ask OAIY…  (/help for commands)', title: 'Enter sends; Shift+Enter starts a new line' });
  private readonly send = h('button.primary', 'Send');
  private readonly attachButton = h('button.attach', { title: 'Attach files or images (or drop them here, or paste an image)', 'aria-label': 'Attach files' }, '📎');
  private readonly picker = h('input', { type: 'file', multiple: true, style: 'display:none' });
  private readonly pending = h('div.attachments');
  private files: File[] = [];
  private readonly status = h('div.chat-status', { role: 'status', 'aria-live': 'polite' });
  /** How full the model's context is. */
  private readonly meterFill = h('span.context-fill');
  private readonly meterText = h('span.context-text');
  private readonly meter = h('span.context-meter', { title: 'How much of the model\'s context the conversation uses. Older turns are summarized before it fills up.' }, h('span.context-bar', this.meterFill), this.meterText);
  /** The agent's checklist for the current request, pinned above the log. */
  private readonly planBox = h('section.plan', { 'aria-live': 'polite' });
  /** The project's conversations: its own chat, and one per person who texts the phone. */
  /**
   * The conversations: the one shown, in a line of its own, and all of them in
   * a list that opens below it (one to a row, scrolling down when there are
   * many: never a strip to scroll sideways).
   */
  private readonly sessionTabs = h('nav.session-switch', { 'aria-label': 'Conversations', hidden: true });
  private sessionsOpen = false;
  private closeSessions: () => void = () => {};
  /** The person's choice to show or hide the steps; null follows the run (hidden once finished). */
  private planOpen: boolean | null = null;
  /** Every step shown, not only the one in progress and the next few (the finished ones fold into one line). */
  private planAll = false;
  /** What the agent is doing now, shown under the step in progress. */
  private readonly planActivity = h('span.plan-activity');
  private current: { box: HTMLElement; text: string; body: HTMLElement } | null = null;
  private thinking: { box: HTMLElement; text: string } | null = null;
  /** A tool call being written, shown as it streams until the call is whole. */
  private draft: { box: HTMLElement; body: HTMLElement; raw: string; json: boolean } | null = null;
  private cards = new Map<string, HTMLElement>();
  /** Sub-agent task rows, by task id (the delegate call's id, #, the task's index). */
  private taskRows = new Map<string, { row: HTMLElement; state: HTMLElement; activity: HTMLElement; report: HTMLElement }>();
  private busy = false;
  /** Blob URLs behind the media in the log, revoked when it is cleared. */
  private media: Media[] = [];

  constructor(
    private readonly handlers: {
      submit: (text: string, files: File[]) => void;
      stop: () => void;
      /** A project file's bytes, for showing it (null when it is gone). */
      file: (path: string) => Uint8Array | null;
      /** Open a project file (or, for an unpacked .softn, its app) in the workspace. */
      open: (path: string, app?: string) => void;
      /** The person flagged a picture the agent made, with what is wrong ('' when they did not say). */
      flag?: (path: string, comment: string) => void;
    },
  ) {
    this.planBox.hidden = true;
    this.meter.hidden = true;
    this.log.addEventListener('scroll', () => {
      this.stick = this.log.scrollTop < 60;
      this.toCurrent.hidden = this.log.scrollTop < 300;
      this.maybeLoadOlder();
    }, { passive: true });
    // The conversations' list closes on a click elsewhere, or Escape.
    document.addEventListener('pointerdown', (e) => {
      if (this.sessionsOpen && !this.sessionTabs.contains(e.target as Node)) this.closeSessions();
    });
    this.sessionTabs.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.sessionsOpen) {
        this.closeSessions();
        (this.sessionTabs.querySelector('.session-current') as HTMLElement | null)?.focus();
      }
    });
    this.element.append(h('div.pane-title', 'Agent', this.meter), this.sessionTabs, this.planBox, h('div.chat-log-wrap', this.log, this.toCurrent), this.status, this.pending, h('div.chat-compose', this.attachButton, this.input, this.send), this.picker);
    this.attachButton.addEventListener('click', () => this.picker.click());
    this.picker.addEventListener('change', () => {
      if (this.picker.files) this.addFiles([...this.picker.files]);
      this.picker.value = '';
    });
    this.element.addEventListener('dragover', (e) => {
      if (e.dataTransfer?.types.includes('Files')) {
        e.preventDefault();
        this.element.classList.add('dropping');
      }
    });
    this.element.addEventListener('dragleave', (e) => {
      if (e.target === this.element || !this.element.contains(e.relatedTarget as Node)) this.element.classList.remove('dropping');
    });
    this.element.addEventListener('drop', (e) => {
      this.element.classList.remove('dropping');
      if (!e.dataTransfer?.files.length) return;
      e.preventDefault();
      this.addFiles([...e.dataTransfer.files]);
    });
    this.input.addEventListener('paste', (e) => {
      const pasted = [...(e.clipboardData?.files ?? [])];
      if (!pasted.length) return;
      e.preventDefault();
      // A pasted screenshot arrives as "image.png": give it a useful name.
      this.addFiles(pasted.map((f, i) => (f.name === 'image.png' ? new File([f], `pasted-${new Date().toISOString().replace(/[:.]/g, '-')}${i ? `-${i}` : ''}.png`, { type: f.type }) : f)));
    });
    // While the agent works, a written message is sent to it; with nothing written, the button stops it.
    this.send.addEventListener('click', () => (this.busy && !this.hasMessage() ? this.handlers.stop() : this.submit()));
    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        this.submit();
      }
    });
    this.input.addEventListener('input', () => this.updateSend());
  }

  addFiles(files: File[]): void {
    this.files.push(...files);
    this.renderPending();
    this.input.focus();
  }

  private pendingUrls: string[] = [];
  private renderPending(): void {
    this.updateSend();
    for (const url of this.pendingUrls.splice(0)) URL.revokeObjectURL(url);
    clear(this.pending);
    this.files.forEach((file, i) => {
      const url = file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
      if (url) this.pendingUrls.push(url);
      const thumb = url ? h('img', { src: url, alt: '' }) : h('span.file-icon', '📄');
      this.pending.append(
        h(
          'span.attachment',
          { title: `${file.name} (${file.size.toLocaleString()} bytes)` },
          thumb,
          h('span.attachment-name', file.name),
          h('button.icon', { title: 'Remove', onclick: () => { this.files.splice(i, 1); this.renderPending(); } }, '✕'),
        ),
      );
    });
  }

  private submit(): void {
    const text = this.input.value.trim();
    if (!text && !this.files.length) return;
    // Slash commands are commands, even with files waiting.
    if (text.startsWith('/') && this.files.length) {
      this.input.value = '';
      this.handlers.submit(text, []);
      return;
    }
    const files = this.files;
    this.files = [];
    this.renderPending();
    this.input.value = '';
    this.updateSend();
    this.handlers.submit(text, files);
  }

  focus(): void {
    this.input.focus();
  }

  setBusy(busy: boolean): void {
    this.busy = busy;
    // A new run: the last run's last tool is not what the agent does now.
    if (busy) this.setActivity('');
    if (this.currentPlan) this.showPlan(this.currentPlan, busy);
    this.input.placeholder = busy ? 'Message the agent while it works…  (it reads it at its next step)' : 'Ask OAIY…  (/help for commands)';
    this.updateSend();
    if (!busy) this.setStatus('');
  }

  private hasMessage(): boolean {
    return !!this.input.value.trim() || this.files.length > 0;
  }

  /** Stop while the agent works with nothing written; Send otherwise. */
  private updateSend(): void {
    const stop = this.busy && !this.hasMessage();
    this.send.textContent = stop ? 'Stop' : 'Send';
    this.send.classList.toggle('danger', stop);
  }

  setStatus(text: string): void {
    this.status.textContent = text;
  }

  /**
   * The log reads newest first: the current step is at the top. It follows
   * new output only while the person is at the top: scrolled down to read
   * older messages, they stay put (the browser keeps what they read in place
   * as new entries arrive above it).
   */
  private stick = true;
  private scrollQueued = false;
  private scroll(): void {
    if (!this.stick || this.scrollQueued || this.sink) return;
    this.scrollQueued = true;
    requestAnimationFrame(() => {
      this.scrollQueued = false;
      if (this.stick) this.log.scrollTop = 0;
    });
  }

  /** Back to the top, where the agent's current step is. */
  private readonly toCurrent = h('button.to-current', { hidden: true, title: 'Back to the newest messages and what the agent is doing now', onclick: () => this.toTop() }, '↑ Current');

  /** Straight to the top (the newest), following it from there. */
  private toTop(): void {
    this.stick = true;
    const still = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.log.scrollTo({ top: 0, behavior: still ? 'auto' : 'smooth' });
    this.toCurrent.hidden = true;
  }

  /** Where new entries go: the top of the log, or (drawing older turns) a holder of their own. */
  private sink: HTMLElement | null = null;
  private add(entry: HTMLElement): void {
    (this.sink ?? this.log).prepend(entry);
  }

  /**
   * The conversations to switch between: the project's chat (id null) and the
   * text-message threads, each with its unread count and whether it is working.
   * Hidden while the project has only its own chat.
   */
  setSessions(tabs: Array<{ id: string | null; label: string; title?: string; status?: string; unread: number; working: boolean; close?: () => void }>, active: string | null, select: (id: string | null) => void): void {
    clear(this.sessionTabs);
    this.sessionTabs.hidden = tabs.length < 2;
    if (!tabs.length) return;
    const current = tabs.find((t) => t.id === active) ?? tabs[0];
    const others = tabs.filter((t) => t !== current);
    const unread = others.reduce((n, t) => n + t.unread, 0);
    const list = h('div.session-list', { role: 'listbox', 'aria-label': 'Conversations' });
    const head = h(
      'button.session-current',
      { 'aria-haspopup': 'listbox', title: 'All the conversations: yours, the calls, the texts and the flows\' tasks', onclick: () => toggle(!this.sessionsOpen) },
      h('span.session-label', { class: current.working ? 'working' : '' }, current.label),
      h('span.session-more', `${others.length} other${others.length === 1 ? '' : 's'}`),
      ...(unread ? [h('span.session-unread', { 'aria-label': `${unread} unread elsewhere` }, String(unread))] : []),
      h('span.session-caret', { 'aria-hidden': 'true' }, '▾'),
    );
    const toggle = (open: boolean) => {
      this.sessionsOpen = open;
      list.hidden = !open;
      head.setAttribute('aria-expanded', String(open));
      this.sessionTabs.classList.toggle('open', open);
    };
    const choose = (id: string | null) => {
      toggle(false);
      select(id);
    };
    for (const tab of tabs) {
      list.append(h(
        'div.session-tab.session-row',
        {
          class: `${tab.id === active ? 'active' : ''} ${tab.working ? 'working' : ''}`,
          role: 'option',
          tabindex: 0,
          title: tab.title ?? tab.label,
          'aria-selected': String(tab.id === active),
          onclick: () => choose(tab.id),
          onkeydown: (e: KeyboardEvent) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              choose(tab.id);
            }
          },
        },
        h('span.session-text', h('span.session-label', tab.label), ...(tab.status ? [h('small.session-status', tab.status)] : [])),
        ...(tab.unread ? [h('span.session-unread', { 'aria-label': `${tab.unread} unread` }, String(tab.unread))] : []),
        ...(tab.close
          ? [h('span.session-close', { role: 'button', title: 'Remove this conversation', 'aria-label': 'Remove this conversation', onclick: (e: Event) => { e.stopPropagation(); tab.close!(); } }, '×')]
          : []),
      ));
    }
    this.sessionTabs.append(head, list);
    this.closeSessions = () => toggle(false);
    toggle(this.sessionsOpen);
  }

  /** How full the context is: `used` tokens of `window`. */
  showContext(used: number, window: number): void {
    const share = Math.min(1, used / Math.max(1, window));
    this.meter.hidden = false;
    this.meterFill.style.width = `${Math.round(share * 100)}%`;
    this.meter.dataset.level = share > 0.9 ? 'high' : share > 0.7 ? 'mid' : 'low';
    this.meterText.textContent = `${formatTokens(used)} / ${formatTokens(window)}`;
  }

  /** A summary that replaced older turns for the model: a note that opens to show it. */
  private summaryNote(text: string, heading: string): void {
    this.current = null;
    const body = text.replace(/^\[(?:OAIY|bot\.computer)\][^\n]*\n(<project>[\s\S]*?<\/project>\n\n)?/, '');
    this.add(h('details.msg.compacted', h('summary', h('span', '⇣'), h('span', ` ${heading}`)), h('pre', body)));
    this.scroll();
  }

  /**
   * Show the checklist: the goal, progress, and the steps. While it runs, the
   * finished steps fold into one line and only the next few are listed, so the
   * list stays short and the log below keeps its room; "show all" lists every
   * step.
   */
  showPlan(plan: Plan | null, running = this.busy): void {
    clear(this.planBox);
    if (!plan) {
      this.planBox.hidden = true;
      return;
    }
    const done = plan.items.filter((i) => i.status === 'done').length;
    const total = plan.items.length;
    const finished = done === total;
    const open = this.planOpen ?? !(finished && !running);
    this.planBox.hidden = false;
    this.planBox.classList.toggle('finished', finished);
    this.planBox.classList.toggle('collapsed', !open);
    const head = h(
      'button.plan-head',
      { title: open ? 'Hide the steps' : 'Show the steps', 'aria-expanded': String(open), onclick: () => {
        this.planOpen = !open;
        this.showPlan(plan, running);
      } },
      h('span.plan-title', finished ? '✓ Done' : running ? 'Working on' : 'Plan'),
      h('span.plan-goal', plan.goal || plan.items.find((i) => i.status === 'active')?.text || ''),
      h('span.plan-count', `${done}/${total}`),
    );
    const bar = h('div.plan-bar', h('span', { style: `width:${Math.round((done / total) * 100)}%` }));
    const toggle = (label: string, title: string) => h('li.plan-fold', h('button', { title, onclick: () => {
      this.planAll = !this.planAll;
      this.showPlan(plan, running);
    } }, label));
    const current = plan.items.findIndex((i) => i.status === 'active');
    const last = current < 0 ? -1 : current + PLAN_UPCOMING;
    const all = this.planAll || finished;
    const rows: HTMLElement[] = [];
    let hiddenLater = 0;
    plan.items.forEach((item, i) => {
      if (!all && item.status === 'done') return;
      if (!all && current >= 0 && i > last && item.status !== 'active') {
        hiddenLater++;
        return;
      }
      const active = item.status === 'active';
      rows.push(h(
        `li.plan-item.${item.status}`,
        { class: active && running ? 'running' : '' },
        h('span.plan-mark', { 'aria-label': item.status }, item.status === 'done' ? '✓' : ''),
        h('span.plan-body', h('span.plan-text', `${i + 1}. ${item.text}`), ...(active && running ? [this.planActivity] : [])),
      ));
    });
    if (!all && done) rows.unshift(toggle(`✓ ${done} step${done > 1 ? 's' : ''} done`, 'Show every step'));
    if (!all && hiddenLater) rows.push(toggle(`+${hiddenLater} more`, 'Show every step'));
    if (this.planAll && !finished) rows.push(toggle('Show less', 'Show only the step in progress and the next ones'));
    const list = h('ol.plan-items', ...rows);
    list.hidden = !open;
    this.planBox.append(head, bar, list);
  }

  /** What the agent is doing now, under the step in progress. */
  private setActivity(text: string): void {
    this.planActivity.textContent = text;
  }

  /**
   * Marks in the log where the plan moved: a new plan, a step started or
   * finished, the steps changed. The log reads newest first, so each step's
   * work sits above the mark where it started.
   */
  private planMarks(before: Plan | null, after: Plan): void {
    const fresh = !before || (!!after.goal && !!before.goal && after.goal !== before.goal);
    const { done, started } = planChanges(before, after);
    const n = after.items.length;
    const mark = (cls: string, icon: string, text: string) => {
      this.current = null;
      this.add(h(`div.msg.step${cls}`, h('span.step-icon', icon), h('span.step-text', text)));
    };
    if (fresh) mark('.plan-made', '☰', `Plan: ${after.goal || `${n} steps`}${after.goal ? ` (${n} steps)` : ''}`);
    else if (after.items.map((i) => i.text).join('\n') !== before.items.map((i) => i.text).join('\n')) mark('.plan-changed', '✎', `Plan changed: now ${n} steps`);
    for (const i of done) mark('.done', '✓', `Step ${i + 1} of ${n} done: ${after.items[i].text}`);
    for (const i of started) mark('.started', '▶', `Step ${i + 1} of ${n}: ${after.items[i].text}`);
    if (done.length || started.length || fresh) this.scroll();
  }

  private currentPlan: Plan | null = null;

  clearLog(): void {
    this.meter.hidden = true;
    this.hiddenTurns = [];
    this.taskRows.clear();
    this.currentPlan = null;
    this.planAll = false;
    this.setActivity('');
    this.showPlan(null);
    for (const m of this.media.splice(0)) m.dispose();
    clear(this.log);
    this.older.remove();
    this.stick = true;
    this.cards.clear();
    this.current = null;
    this.thinking = null;
  }

  /**
   * A project file shown in the log: an image as a thumbnail, audio and
   * video with a player, anything else as a chip; each opens the file.
   */
  private fileView(path: string, name = path.split('/').pop() ?? path, app?: string, flaggable = false): HTMLElement {
    const open = () => this.handlers.open(path, app);
    const bytes = mediaKind(path) ? this.handlers.file(path) : null;
    const media = bytes ? mediaElement(path, bytes, { compact: true, onOpen: open }) : null;
    if (media) {
      this.media.push(media);
      // A picture the agent made can be flagged: the agent makes it again.
      if (flaggable && this.handlers.flag && mediaKind(path) === 'image') {
        const wrap = h('span.flaggable', media.element);
        wrap.append(h('button.flag-button', { type: 'button', title: 'Flag this picture: say what is wrong (or let the agent find it), and it is made again', 'aria-label': `Flag ${path}`, onclick: () => this.flagBox(path, wrap) }, '⚑'));
        return wrap;
      }
      return media.element;
    }
    const icon = app || /\.softn$/i.test(name) ? '📦' : mediaKind(path) === 'image' ? '🖼' : '📄';
    return h('button.attachment', { title: app ? `Unpacked into ${app}/: click to preview it` : `${path}: click to open`, onclick: open }, h('span.file-icon', icon), h('span.attachment-name', name));
  }

  /**
   * The box under a flagged picture: a quick comment on what is wrong. Left
   * untouched for a few seconds, the flag goes without one, and the agent
   * looks for what is wrong itself.
   */
  private flagBox(path: string, wrap: HTMLElement): void {
    const holder = wrap.closest('.msg') ?? wrap;
    const open = holder.nextElementSibling;
    if (open instanceof HTMLElement && open.dataset.flag === path) {
      open.querySelector('textarea')?.focus();
      return;
    }
    const comment = h('textarea.flag-comment', { rows: 2, placeholder: 'What is wrong with it? e.g. three arms; his jacket should be red (optional)' });
    const note = h('span.flag-note');
    let left = FLAG_WAIT_SECONDS;
    let timer: ReturnType<typeof setInterval> | null = null;
    const stopCountdown = () => {
      if (!timer) return;
      clearInterval(timer);
      timer = null;
      note.textContent = 'Enter flags it with your comment';
    };
    const done = (text: string | null) => {
      stopCountdown();
      box.remove();
      if (text === null) return;
      wrap.classList.add('flagged');
      wrap.title = text ? `Flagged: ${text}` : 'Flagged: the agent looks for what is wrong';
      this.handlers.flag?.(path, text);
    };
    const box = h(
      'div.msg.flag-box',
      { 'data-flag': path },
      h('div.flag-title', `⚑ Flag ${path.split('/').pop() ?? path}`),
      comment,
      h('div.flag-actions', note, h('button', { type: 'button', onclick: () => done(null) }, 'Cancel'), h('button.primary', { type: 'button', onclick: () => done(comment.value.trim()) }, 'Flag')),
    );
    const tick = () => {
      note.textContent = `Flags it without a comment in ${left} s: the agent then looks for what is wrong`;
    };
    comment.addEventListener('input', stopCountdown);
    comment.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        done(comment.value.trim());
      } else if (e.key === 'Escape') {
        done(null);
      }
    });
    tick();
    timer = setInterval(() => {
      left--;
      if (left <= 0) done(comment.value.trim());
      else tick();
    }, 1000);
    holder.after(box);
    comment.focus();
    this.scroll();
  }

  /**
   * Words from the other end of a call or text thread while its agent answers:
   * drawn where they came, and the reply being written goes on in its own box
   * below them (it began first), not split around them.
   */
  heard(text: string): void {
    this.add(h('div.msg.user', h('div.msg-body', text || ' ')));
    if (!this.sink) this.toTop();
  }

  /** A run ended (finished, or cut off): what the agent writes next is a new reply. */
  endReply(): void {
    this.current = null;
    this.thinking = null;
  }

  /** The person's message: a new request, or (`during`) one sent while the agent works, which keeps its plan. */
  user(text: string, attachments: Attachment[] = [], during = false, note = 'sent while the agent works: it reads it at its next step'): void {
    this.current = null;
    // Sending a message goes back to the top, where the reply comes.
    if (during) {
      const box = h('div.msg.user.during', h('div.msg-body', text || ' '), h('div.msg-note', note));
      if (attachments.length) box.append(h('div.msg-attachments', ...attachments.map((a) => this.fileView(a.path, a.name, a.app))));
      this.add(box);
      if (!this.sink) this.toTop();
      return;
    }
    // A new request gets its own plan.
    this.currentPlan = null;
    this.planOpen = null;
    this.showPlan(null);
    const box = h('div.msg.user', h('div.msg-body', text || (attachments.length ? '' : ' ')));
    if (attachments.length) box.append(h('div.msg-attachments', ...attachments.map((a) => this.fileView(a.path, a.name, a.app))));
    this.add(box);
    if (!this.sink) this.toTop();
  }

  /** The model's thinking stays shown once the reply moves on: it says why the model did what it did. */
  private doneThinking(): void {
    if (this.thinking) this.thinking.box.querySelector('summary')!.textContent = '💭 What the model thought';
    this.thinking = null;
  }

  /** A box with the model's thinking, open to read. */
  private thoughtBox(text: string, streaming = false): HTMLElement {
    const box = h('details.thinking', { open: true }, h('summary', streaming ? '💭 The model is thinking…' : '💭 What the model thought'), h('pre', text));
    this.add(box);
    return box;
  }

  /** What the model was sent for this step: folded, and written out only when opened (it is long). */
  private promptView(e: { system: string; turns: Turn[]; tools: string[] }): void {
    this.current = null;
    const chars = e.system.length + e.turns.reduce((n, t) => n + (t.role === 'user' ? t.text.length : t.role === 'assistant' ? t.text.length + JSON.stringify(t.calls).length : t.results.reduce((m, r) => m + r.content.length, 0)), 0);
    const pre = h('pre');
    const box = h('details.prompt-view', h('summary', `📝 The prompt the model was sent: ${e.turns.length} message${e.turns.length === 1 ? '' : 's'}, about ${formatTokens(Math.round(chars / 4))} tokens (click to read)`), pre);
    box.addEventListener('toggle', () => {
      if ((box as HTMLDetailsElement).open && !pre.textContent) pre.textContent = promptText(e);
    });
    this.add(box);
    this.scroll();
  }

  /**
   * A tool call as it is written: OAIY's raw call (its tags turned into a
   * readable layout), or JSON arguments with their strings unescaped. A script
   * written by append_file reads as it is written.
   */
  private toolDraft(text: string, start: boolean, json: boolean): void {
    // OAIY sends the finished call as JSON too: the raw draft already shows it.
    if (json && this.draft && !this.draft.json) return;
    if (!this.draft || start) {
      this.draft?.box.remove();
      this.current = null;
      const body = h('pre.tool-result.draft-body');
      const box = h('details.tool.draft', { open: true }, h('summary', h('span.tool-name', 'writing…')), body);
      box.classList.add('pending');
      this.add(box);
      this.draft = { box, body, raw: '', json };
    }
    this.draft.raw += text;
    const draft = this.draft;
    this.redraw(draft.box, () => {
      const raw = draft.raw;
      const name = draft.json ? '' : /<function=([\w.-]+)>/.exec(raw)?.[1] ?? '';
      if (name) draft.box.querySelector('.tool-name')!.textContent = `writing ${name}…`;
      draft.body.textContent = draft.json
        ? raw.replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\t/g, '\t')
        : raw
            .replace(/<\/?tool_call>\s*/g, '')
            .replace(/<function=[\w.-]+>\s*/g, '')
            .replace(/<parameter=([\w.-]+)>\n?/g, '$1:\n')
            .replace(/\n?<\/(parameter|function)>/g, '\n');
      // A long draft scrolls in its box: the newest lines stay in view.
      draft.body.scrollTop = draft.body.scrollHeight;
    });
  }

  /** A message from the app itself (commands, errors, notices). */
  system(text: string, kind: 'info' | 'error' = 'info'): void {
    this.current = null;
    this.add(h('div.msg.system', { class: kind }, h('pre', text)));
    this.scroll();
  }

  /** Run `draw` soon (see STREAM_REDRAW_MS), once for however many calls for the same box came before it ran (each box is drawn: a replayed conversation has many). */
  private redraws = new Map<HTMLElement, () => void>();
  private redraw(key: HTMLElement, draw: () => void): void {
    const queued = this.redraws.has(key);
    this.redraws.set(key, draw);
    if (queued) return;
    setTimeout(() => {
      const latest = this.redraws.get(key);
      this.redraws.delete(key);
      requestAnimationFrame(() => {
        latest?.();
        this.scroll();
      });
    }, STREAM_REDRAW_MS);
  }

  private assistantText(delta: string): void {
    if (!this.current) {
      const body = h('div.msg-body');
      const box = h('div.msg.assistant', body);
      this.add(box);
      this.current = { box, text: '', body };
    }
    this.current.text += delta;
    // A long reply streams in many pieces: drawn a few times a second, not for each.
    const target = this.current;
    this.redraw(target.box, () => {
      target.body.innerHTML = renderMarkdown(target.text);
    });
  }

  private toolCard(call: ToolCall): void {
    this.current = null;
    this.thinking = null;
    // The checklist shows the plan; a card per update would only repeat it.
    if (call.name === 'update_plan') return;
    if (call.name === 'delegate') {
      this.delegateCard(call);
      return;
    }
    const result = h('pre.tool-result', 'running…');
    const card = h(
      'details.tool',
      h('summary', h('span.tool-name', call.name), ' ', h('span.tool-arg', summarizeCall(call)), ...(mediaPrompt(call) ? [h('span.tool-prompt', { title: mediaPrompt(call) }, mediaPrompt(call))] : [])),
      h('pre.tool-input', JSON.stringify(call.input, null, 2)),
      result,
    );
    card.classList.add('pending');
    this.cards.set(call.id, card);
    this.add(card);
    this.scroll();
  }

  /** A delegate call: its tasks, each with its state, what it is doing, and its report. */
  private delegateCard(call: ToolCall): void {
    let tasks: Array<{ title: string }> = [];
    try {
      tasks = readTasks(call.input);
    } catch {
      /* the result says what was wrong */
    }
    const list = h('ol.agent-tasks');
    const card = h(
      'details.tool.delegate',
      { open: true },
      h('summary', h('span.tool-name', 'agents'), ' ', h('span.tool-arg', `${tasks.length} task${tasks.length === 1 ? '' : 's'} for sub-agents`)),
      list,
      h('pre.tool-result', 'waiting for the tasks…'),
    );
    card.classList.add('pending');
    tasks.forEach((task, i) => this.taskRow(`${call.id}#${i}`, task.title, list));
    this.cards.set(call.id, card);
    this.add(card);
    this.scroll();
  }

  private taskRow(id: string, title: string, list?: HTMLElement): { row: HTMLElement; state: HTMLElement; activity: HTMLElement; report: HTMLElement } {
    const existing = this.taskRows.get(id);
    if (existing) return existing;
    const state = h('span.task-state', 'queued');
    const activity = h('span.task-activity');
    const report = h('pre.task-report');
    const row = h('li.agent-task.queued', h('div.task-head', h('span.task-mark'), h('span.task-title', title), state), activity, h('details.task-more', h('summary', 'report'), report));
    const card = this.cards.get(id.split('#')[0]);
    let target = list ?? card?.querySelector('.agent-tasks');
    // A sub-agent under another tool's card (the review of a picture): a list of its own just below the card, seen without opening it.
    if (!target && card) {
      const next = card.previousElementSibling;
      target = next instanceof HTMLElement && next.matches('ol.agent-tasks.under') ? next : null;
      if (!target) {
        target = h('ol.agent-tasks.under');
        card.before(target);
      }
    }
    target?.append(row);
    const entry = { row, state, activity, report };
    this.taskRows.set(id, entry);
    return entry;
  }

  private agentTask(e: Extract<AgentEvent, { type: 'agent_task' }>): void {
    const entry = this.taskRow(e.id, e.title);
    entry.row.className = `agent-task ${e.state}`;
    entry.state.textContent = e.state === 'running' ? 'working' : e.state === 'failed' ? 'not finished' : e.state;
    if (e.activity) entry.activity.textContent = e.activity;
    if (e.result !== undefined) {
      entry.report.textContent = e.result || '(no report)';
      entry.activity.textContent = e.state === 'done' ? 'finished' : entry.activity.textContent;
    }
    if (e.state === 'running') this.setStatus(`sub-agent: ${e.title}: ${e.activity ?? ''}`);
  }

  /** The automatic check after the agent changed an app: a card like a tool's. */
  private checkCard(e: Extract<AgentEvent, { type: 'check' }>): void {
    if (e.state === 'running') {
      this.current = null;
      this.thinking = null;
      this.setStatus(`checking ${e.root || 'the app'}…`);
      const card = h('details.tool.auto', h('summary', h('span.tool-name', 'automatic check'), ' ', h('span.tool-arg', e.root ? `${e.root}/` : '/')), h('pre.tool-result', 'checking the files and rendering the app…'));
      card.classList.add('pending');
      this.cards.set(e.id, card);
      this.add(card);
      this.scroll();
      return;
    }
    const card = this.cards.get(e.id);
    if (!card) return;
    card.classList.remove('pending');
    card.classList.add(e.state === 'ok' ? 'ok' : 'failed');
    const pre = card.querySelector('.tool-result');
    if (pre) pre.textContent = e.text ?? '';
    // A failed check is worth seeing without a click: the agent fixes it next.
    if (e.state === 'failed') (card as HTMLDetailsElement).open = true;
    this.scroll();
  }

  private toolResult(result: ToolResult): void {
    const card = this.cards.get(result.id);
    if (!card) return;
    card.classList.remove('pending');
    card.classList.add(result.isError ? 'failed' : 'ok');
    const pre = card.querySelector('.tool-result');
    if (pre) pre.textContent = result.content;
    // A delegate card replayed from a saved chat: its tasks' outcomes come from the report.
    if (result.name === 'delegate') {
      for (const m of result.content.matchAll(/### Task (\d+): .*? \((done|not finished)\)\n([\s\S]*?)(?=\n\n### Task |\n\nApps still failing|\n\nCheck the results|$)/g)) {
        const entry = this.taskRows.get(`${result.id}#${Number(m[1]) - 1}`);
        if (!entry || !entry.row.classList.contains('queued')) continue;
        entry.row.className = `agent-task ${m[2] === 'done' ? 'done' : 'failed'}`;
        entry.state.textContent = m[2];
        entry.report.textContent = m[3];
      }
      (card as HTMLDetailsElement).open = false;
      const arg = card.querySelector('summary .tool-arg');
      if (arg) arg.textContent = result.content.split('\n')[0];
    }
    for (const image of result.images ?? []) {
      card.append(h('img.tool-image', { src: `data:${image.mediaType};base64,${image.data}`, alt: image.label ?? 'image shown to the model' }));
    }
    // A file the agent hands over is shown outside the (collapsed) card.
    if (result.files?.length) {
      // Pictures the agent made can be flagged; files it only shows (uploads, extracted frames) cannot.
      const shown = h('div.msg.presented', ...result.files.map((path) => this.fileView(path, undefined, undefined, result.name === 'generate_image')));
      card.before(shown);
    }
    this.scroll();
  }

  event(e: AgentEvent): void {
    switch (e.type) {
      case 'text':
        this.doneThinking();
        this.assistantText(e.delta);
        break;
      case 'thinking':
        if (!this.thinking) {
          const box = this.thoughtBox('', true);
          this.thinking = { box, text: '' };
        }
        this.thinking.text += e.delta;
        {
          const thinking = this.thinking;
          this.redraw(thinking.box, () => {
            const pre = thinking.box.querySelector('pre')!;
            pre.textContent = thinking.text;
            // The newest thoughts stay in view as they stream.
            pre.scrollTop = pre.scrollHeight;
          });
        }
        break;
      case 'prompt':
        this.promptView(e);
        break;
      case 'tool_draft':
        this.doneThinking();
        this.toolDraft(e.text, e.start, e.index !== undefined);
        break;
      case 'tool_start':
        this.setStatus(`writing a ${e.name} call…`);
        break;
      case 'tool_call':
        this.doneThinking();
        this.draft?.box.remove();
        this.draft = null;
        this.setStatus(`running ${e.call.name}…`);
        if (e.call.name !== 'update_plan') this.setActivity(`${e.call.name} ${summarizeCall(e.call)}`.trim());
        this.toolCard(e.call);
        break;
      case 'tool_result':
        this.setStatus('thinking…');
        this.toolResult(e.result);
        break;
      case 'status':
        this.setStatus(e.message);
        break;
      case 'usage':
        break;
      case 'check':
        this.checkCard(e);
        break;
      case 'plan':
        if (this.currentPlan?.goal !== e.plan.goal) {
          this.planOpen = null;
          this.planAll = false;
        }
        this.planMarks(this.currentPlan, e.plan);
        this.currentPlan = e.plan;
        this.showPlan(e.plan, true);
        break;
      case 'context':
        this.showContext(e.used, e.window);
        break;
      case 'agent_task':
        this.agentTask(e);
        break;
      case 'compact':
        this.current = null;
        this.add(
          h('div.msg.nudge.compacted', h('span', '⇣'), h('span', e.how === 'summary'
            ? ` Context compacted: ${e.turns} earlier turns summarized for the model (${formatTokens(e.before)} → ${formatTokens(e.after)} tokens). The chat keeps everything.`
            : ` Context compacted: the model could not write a summary, so ${e.turns} earlier turns were reduced to their requests and changes (${formatTokens(e.before)} → ${formatTokens(e.after)} tokens).`)),
        );
        this.scroll();
        break;
      case 'nudge':
        this.current = null;
        this.add(h('div.msg.nudge', h('span', '↻'), h('span', ` Not finished yet, so the agent carries on: ${e.message}`)));
        this.scroll();
        break;
      case 'done':
        this.current = null;
        this.setActivity('');
        break;
      case 'error':
        this.system(e.message, 'error');
        break;
    }
  }

  /** Turns of a long saved conversation not drawn yet: drawn as the person scrolls down to them. */
  private hiddenTurns: Turn[] = [];
  /** The end of the log while older turns are left: scrolling near it draws them. */
  private readonly older = h('button.show-earlier', { onclick: () => this.loadOlder() });

  /**
   * Show a saved conversation, newest first. A long one opens at its latest
   * turns, and older ones are drawn a page at a time as the person scrolls
   * down to them: drawing hundreds of tool cards at once is slow.
   */
  replay(turns: Turn[], latest = REPLAY_PAGE): void {
    this.clearLog();
    const from = pageStart(turns, latest);
    this.hiddenTurns = turns.slice(0, from);
    // The plan the turns drawn change (one step at a time, maybe) is the one the older turns left.
    this.currentPlan = planAfter(this.hiddenTurns);
    this.drawTurns(turns.slice(from));
    this.showOlder();
    this.log.scrollTop = 0;
  }

  /** The note at the end of the log: how much older conversation there is. */
  private showOlder(): void {
    if (!this.hiddenTurns.length) {
      this.older.remove();
      return;
    }
    this.older.textContent = `↓ ${this.hiddenTurns.length} older turns: scroll down, or click, to show them`;
    this.log.append(this.older);
    // A short page does not fill the log: nothing to scroll, so draw on.
    requestAnimationFrame(() => this.maybeLoadOlder());
  }

  private maybeLoadOlder(): void {
    if (this.hiddenTurns.length && this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight < 400) this.loadOlder();
  }

  /** Draw the next page of older turns at the end of the log, leaving what is live (the current reply, the plan) as it is. */
  private loadOlder(): void {
    if (!this.hiddenTurns.length || this.sink) return;
    const from = pageStart(this.hiddenTurns, REPLAY_PAGE);
    const page = this.hiddenTurns.slice(from);
    this.hiddenTurns = this.hiddenTurns.slice(0, from);
    const live = { current: this.current, thinking: this.thinking, draft: this.draft, plan: this.currentPlan, planOpen: this.planOpen };
    // The page's marks follow the plan as it was then, not as it is now.
    this.currentPlan = planAfter(this.hiddenTurns);
    this.sink = h('div');
    try {
      this.drawTurns(page, false);
    } finally {
      this.older.before(...this.sink.children);
      this.sink = null;
      ({ current: this.current, thinking: this.thinking, draft: this.draft, plan: this.currentPlan, planOpen: this.planOpen } = live);
      this.showPlan(this.currentPlan);
      this.showOlder();
    }
  }

  /** Draw turns in the order they came (each goes on top of the ones before). */
  private drawTurns(turns: Turn[], showPlan = true): void {
    for (const turn of turns) {
      if (turn.role === 'user' && turn.summary) {
        this.summaryNote(turn.text, 'Earlier conversation summarized for the model (click to read the summary)');
        continue;
      }
      if (turn.role === 'user' && turn.automatic) {
        this.current = null;
        this.add(h('div.msg.nudge', h('span', '↻'), h('span', ` ${turn.text.replace(/^\[(?:OAIY|bot\.computer)\] /, '').split('\n')[0]}`)));
        continue;
      }
      if (turn.role === 'user') {
        // A message sent while the agent worked carries a note for the model: show it as the person wrote it.
        const during = /^\[The user sent this while you[^\]]*\]\n\n/.exec(turn.text);
        const text = (during ? turn.text.slice(during[0].length) : turn.text).replace(/^<project>[\s\S]*?<\/project>\n\n/, '');
        const attached = /\n\n\[Attached and saved in the project: ([\s\S]*)\]$/.exec(text);
        // Chats saved before attachments were kept on the turn: read them from the note.
        const fallback = attached ? attached[1].split('; ').map((n) => n.split(/ \(|: /)[0]).filter((p) => /^uploads\/[^\s]+$/.test(p)).map((path) => ({ name: path.split('/').pop()!, path })) : [];
        this.user(attached ? text.slice(0, attached.index) : text, turn.attachments ?? fallback, !!during);
      }
      else if (turn.role === 'assistant') {
        if (turn.thinking) this.thoughtBox(turn.thinking);
        if (turn.text) this.assistantText(turn.text);
        this.current = null;
        for (const call of turn.calls) {
          this.toolCard(call);
          if (call.name === 'update_plan') {
            try {
              const next = settlePlan(readPlan(call.input, this.currentPlan));
              this.planMarks(this.currentPlan, next);
              this.currentPlan = next;
            } catch {
              /* a plan the tool refused */
            }
          }
        }
      } else for (const r of turn.results) this.toolResult(r);
    }
    if (showPlan) this.showPlan(this.currentPlan, false);
  }
}
