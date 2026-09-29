/**
 * The chat pane: the conversation with the agent, streamed as it arrives,
 * with a row per tool call; and the prompt box, which also takes slash
 * commands.
 *
 * The log reads oldest first, the newest just above the message box, and
 * keeps to the bottom while the reader is there (chat/scroll.ts). What each
 * person said is grouped under their
 * name (a run: one speaker, one header), a call's and a text thread's words
 * are shown without the labels the agent reads them with (transcript.ts), and
 * tool calls made one after another fold into one "Used N tools" row.
 */
import type { AgentEvent } from '../agent/agent';
import type { Attachment, ToolCall, ToolResult, Turn } from '../agent/protocol';
import { planAfter, planChanges, readPlan, readTasks, settlePlan, type Plan } from '../agent/tools';
import { formatTokens } from '../agent/context';
import { clear, h } from './dom';
import { icon } from './icons';
import { mediaElement, mediaKind, type Media } from './media';
import { renderMarkdown } from './markdown';
import { SessionPicker, tabKind, tabName, type ConversationTab } from './sessionPicker';
import { argLines, clip, duration, mediaPrompt, prettyResult, summarizeCall, toolIcon, toolLabel } from './chat/tools';
import {
  clock,
  conversationKind,
  dayLabel,
  formatNumber,
  initials,
  parseCallEnd,
  parseCallStart,
  parseCallTurn,
  parseFlowAsk,
  parseTextTurn,
  timeLabel,
  type CallStart,
  type Part,
} from './chat/transcript';
import { Follow, anchorIndex, anchoredScrollTop } from './chat/scroll';

/** How long a flag waits for a comment before it goes without one. */
const FLAG_WAIT_SECONDS = 10;
/**
 * Streamed text (the reply, thinking, a tool call being written) arrives in
 * hundreds of pieces a second; redrawing a long text for each froze the page
 * and could garble the display. Each box redraws at most this often, with all
 * that has arrived by then.
 */
const STREAM_REDRAW_MS = 80;

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

/** Steps after the one in progress that the checklist shows before folding the rest into "+N more". */
const PLAN_UPCOMING = 4;

/** How many turns of a saved conversation are drawn at a time (the latest first; older ones as the log is scrolled up to them). */
const REPLAY_PAGE = 60;

/** Where the last page of `turns` starts: `size` turns back, moved back to the request they belong to. */
function pageStart(turns: Turn[], size: number): number {
  let from = Math.max(0, turns.length - size);
  while (from > 0 && turns[from]?.role !== 'user') from--;
  return from;
}

/** Who said something: the person, the agent, the caller, the one texting, or a flow. */
type Speaker = 'you' | 'agent' | 'caller' | 'texter' | 'flow';

/** Suggestions for an empty conversation: they fill the message box, and are sent only when the person sends them. */
const SUGGESTIONS: Record<'runner' | 'project', { title: string; text: string; ideas: string[] }> = {
  runner: {
    title: "Your phone's front desk",
    text: "Tell me what callers and texters should hear: I keep the brief, the knowledge files and each caller's notes up to date, and I can look over what the phone's agents said.",
    ideas: [
      'What did callers and texters ask about today?',
      "This week, tell callers we're booked until Friday.",
      "Don't quote prices for jobs that take more than a day.",
      'Add our services and prices to the knowledge files.',
    ],
  },
  project: {
    title: 'What shall we make?',
    text: 'Ask for code, a web page or a SoftN app: I write the files here, run them on the Zipp VM, and show you the result.',
    ideas: ['Explain what this project does.', 'Build a one-page site with a contact form.', 'Start a SoftN app that keeps a task list.', 'Find the TODOs and fix the first one.'],
  },
};

export class ChatPane {
  readonly element = h('section.chat');
  private readonly log = h('div.chat-log', { role: 'log', 'aria-live': 'polite', 'aria-relevant': 'additions' });
  private readonly input = h('textarea.chat-input', { rows: 1, placeholder: 'Ask OAIY…', 'aria-label': 'Message the agent', title: 'Enter sends; Shift+Enter starts a new line' }) as HTMLTextAreaElement;
  private readonly send = h('button.send-button', { type: 'button' }) as HTMLButtonElement;
  private readonly attachButton = h('button.attach', { type: 'button', title: 'Attach files or images (or drop them here, or paste an image)', 'aria-label': 'Attach files' }, icon('paperclip'));
  private readonly picker = h('input', { type: 'file', multiple: true, style: 'display:none' });
  private readonly pending = h('div.attachments');
  private files: File[] = [];
  private readonly status = h('div.chat-status', { role: 'status', 'aria-live': 'polite' });
  private readonly hint = h('span.composer-hint');
  /** How full the model's context is. */
  private readonly meterFill = h('span.context-fill');
  private readonly meterText = h('span.context-text');
  private readonly meter = h('span.context-meter', { title: 'How much of the model\'s context the conversation uses. Older turns are summarized before it fills up.' }, h('span.context-bar', this.meterFill), this.meterText);
  /** The agent's checklist for the current request, pinned above the log. */
  private readonly planBox = h('section.plan', { 'aria-live': 'polite' });
  /**
   * The conversations: the project's own, and the phone's (each call and text
   * thread, and flows' tasks), in a picker that searches them.
   */
  private readonly sessionTabs = h('nav.session-switch', { 'aria-label': 'Conversations', hidden: true });
  private readonly sessionPicker = new SessionPicker();
  /** A call going on now, in the conversation shown: how long it has run. */
  private readonly liveBar = h('div.call-live', { role: 'status', hidden: true });
  private liveTimer: ReturnType<typeof setInterval> | null = null;
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
  /** What the conversation shown is, read from its words: a call's, a text thread's, a flow's tasks, or the person's own. */
  private kind: 'call' | 'sms' | 'task' | 'own' = 'own';
  /** The person's own conversation is the Front desk's runner (else a project's). */
  private runner = false;
  /** The conversation shown is a call going on now. */
  private live = false;
  /** Who is on the call (from the note that began it). */
  private callName = 'Caller';
  /** When the call shown began (ms), as near as its note and the caller's times tell. */
  private callStartAt: number | null = null;
  /** Drawing a saved conversation (no times are measured). */
  private replaying = false;
  /** The empty conversation's welcome. */
  private readonly hero = h('div.chat-empty');
  /** What the log holds, oldest first; the agent at work (`status`) is always last. */
  private readonly feed = h('div.chat-feed');
  /** Whether the log keeps to the bottom, and how much came while it did not. */
  private readonly follow = new Follow();
  private readonly jumpCount = h('span.jump-count');

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
    this.log.append(this.feed);
    this.feed.append(this.status);
    this.sessionTabs.append(this.sessionPicker.element);
    this.log.addEventListener('scroll', () => {
      this.follow.scrolled(this.log);
      this.showJump();
      this.maybeLoadOlder();
    }, { passive: true });
    // Whatever makes the log taller or its window shorter (a reply streaming, a row opened, a picture
    // loaded, the message box growing): at the bottom, it stays there.
    if (typeof ResizeObserver === 'function') {
      const watch = new ResizeObserver(() => this.pin());
      watch.observe(this.feed);
      watch.observe(this.log);
    }
    // A code block's Copy button.
    this.log.addEventListener('click', (e) => {
      const button = (e.target as Element).closest?.('.code-copy');
      if (button instanceof HTMLElement) void this.copyCode(button);
    });
    const composer = h(
      'div.composer',
      this.pending,
      this.input,
      h('div.composer-bar', this.attachButton, this.hint, this.send),
    );
    composer.addEventListener('click', (e) => {
      if (e.target === composer) this.input.focus();
    });
    this.element.append(
      h('div.pane-title.chat-head', h('span.pane-kicker', 'Agent'), this.meter),
      this.sessionTabs,
      this.liveBar,
      this.planBox,
      h('div.chat-log-wrap', this.log, this.toLatest),
      h('div.chat-compose', composer),
      this.picker,
    );
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
    this.input.addEventListener('input', () => {
      this.updateSend();
      this.grow();
    });
    this.updateSend();
    this.updateEmpty();
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
      const thumb = url ? h('img', { src: url, alt: '' }) : h('span.file-icon', icon('file'));
      this.pending.append(
        h(
          'span.attachment',
          { title: `${file.name} (${file.size.toLocaleString()} bytes)` },
          thumb,
          h('span.attachment-name', file.name),
          h('button.icon', { type: 'button', title: 'Remove', 'aria-label': `Remove ${file.name}`, onclick: () => { this.files.splice(i, 1); this.renderPending(); } }, icon('x')),
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
      this.grow();
      this.handlers.submit(text, []);
      return;
    }
    const files = this.files;
    this.files = [];
    this.renderPending();
    this.input.value = '';
    this.grow();
    this.updateSend();
    this.handlers.submit(text, files);
  }

  focus(): void {
    this.input.focus();
  }

  /** The message box grows with what is written, up to a limit, then scrolls. */
  private grow(): void {
    this.input.style.height = 'auto';
    const max = Math.max(96, Math.min(240, Math.round(window.innerHeight * 0.32)));
    this.input.style.height = `${Math.min(max, this.input.scrollHeight + 2)}px`;
    this.input.style.overflowY = this.input.scrollHeight + 2 > max ? 'auto' : 'hidden';
  }

  setBusy(busy: boolean): void {
    this.busy = busy;
    // A new run: the last run's last tool is not what the agent does now.
    if (busy) this.setActivity('');
    if (this.currentPlan) this.showPlan(this.currentPlan, busy);
    this.input.placeholder = busy ? 'Message the agent while it works…' : this.placeholder();
    this.element.classList.toggle('busy', busy);
    this.updateSend();
    if (!busy) this.setStatus('');
  }

  /** What the message box asks for, in the conversation shown. */
  private placeholder(): string {
    if (this.kind === 'call') return this.live ? 'Tell the receptionist something: it reads it before its next reply…' : "Ask about this call, or tell its agent something…";
    if (this.kind === 'sms') return 'Tell the agent what to text…';
    return this.runner ? 'Tell the runner what the phone should know…' : 'Ask OAIY…';
  }

  private hasMessage(): boolean {
    return !!this.input.value.trim() || this.files.length > 0;
  }

  /** Stop while the agent works with nothing written; Send otherwise. */
  private updateSend(): void {
    const stop = this.busy && !this.hasMessage();
    clear(this.send);
    this.send.append(icon(stop ? 'stop' : 'arrow-up'));
    this.send.dataset.mode = stop ? 'stop' : 'send';
    this.send.classList.toggle('danger', stop);
    this.send.title = stop ? 'Stop the agent' : this.busy ? 'Send: the agent reads it at its next step' : 'Send (Enter)';
    this.send.setAttribute('aria-label', stop ? 'Stop' : 'Send');
    this.send.disabled = !stop && !this.hasMessage();
    this.hint.textContent = stop ? 'The agent is working: the square button stops it' : this.busy ? 'Enter sends it: the agent reads it at its next step' : 'Enter to send · Shift+Enter for a new line · /help';
  }

  setStatus(text: string): void {
    this.status.textContent = text;
  }

  /**
   * The log reads oldest first, the newest at the bottom. It keeps to the
   * bottom while the reader is there (or near it); scrolled up to read older
   * messages, they stay put, and "Latest" says how much is new below.
   */
  private scrollQueued = false;
  private scroll(): void {
    if (!this.follow.stick || this.scrollQueued || this.sink) return;
    this.scrollQueued = true;
    requestAnimationFrame(() => {
      this.scrollQueued = false;
      this.pin();
    });
  }

  /** At the bottom now, when the log follows. */
  private pin(): void {
    if (this.follow.stick && !this.sink) this.log.scrollTop = this.log.scrollHeight;
    this.showJump();
  }

  /** Down to the latest, and what the agent is doing now. */
  private readonly toLatest = h('button.to-latest', { type: 'button', hidden: true, title: 'Down to the newest messages and what the agent is doing now', onclick: () => this.toBottom(true) }, icon('arrow-down'), h('span', 'Latest'), this.jumpCount);

  /** "Latest", while the reader is away from the bottom (with how many messages came meanwhile). */
  private showJump(): void {
    this.toLatest.hidden = !this.follow.offerJump(this.log);
    this.jumpCount.textContent = this.follow.unseen ? `${this.follow.unseen} new` : '';
  }

  /** Straight to the bottom (the newest), following it from there. */
  private toBottom(smooth = false): void {
    const glide = smooth && !(typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches);
    this.log.scrollTo({ top: this.log.scrollHeight, behavior: glide ? 'smooth' : 'auto' });
    this.follow.jump(glide ? undefined : this.log.scrollTop);
    this.showJump();
  }

  /** Where new entries go: the end of the log, or (drawing older turns) a holder of their own. */
  private sink: HTMLElement | null = null;

  /** Put `node` at the end of `box` (in the log: before the agent-at-work line, which stays last). */
  private place(box: HTMLElement, node: HTMLElement): void {
    if (box === this.feed) this.feed.insertBefore(node, this.status);
    else box.append(node);
  }

  /** The last entry of `box` (the log's own lines, and the welcome, are not entries). */
  private lastEntry(box: HTMLElement): Element | null {
    let last = box === this.feed ? this.status.previousElementSibling : box.lastElementChild;
    while (last && (last === this.hero || last === this.older)) last = last.previousElementSibling;
    return last;
  }

  /**
   * Put an entry at the end: under its speaker's header when the last entry is
   * theirs already (a run), in a new run otherwise; an entry of no one's (a
   * note from the app, a day) on its own.
   */
  private add(entry: HTMLElement, speaker: Speaker | null = null, who = ''): void {
    const box = this.sink ?? this.feed;
    if (!this.sink && !this.replaying) {
      this.follow.arrived(entry.matches('.msg.user, .msg.incoming, .msg.assistant, .msg.system, .sms-out'));
      this.showJump();
    }
    if (!speaker) {
      this.place(box, entry);
      return;
    }
    if (!this.sink) this.hero.remove();
    let run = this.endRun(box, speaker, who);
    if (!run) {
      run = this.makeRun(speaker, who);
      this.place(box, run);
    }
    run.lastElementChild!.append(entry);
  }

  /** The run at the end of `box`, when it is `speaker`'s (and, for someone on the phone, the same someone). */
  private endRun(box: HTMLElement, speaker: Speaker, who = ''): HTMLElement | null {
    const last = this.lastEntry(box);
    return last instanceof HTMLElement && last.matches('section.run') && last.dataset.speaker === speaker && (last.dataset.who ?? '') === who ? last : null;
  }

  /** A speaker's run: their avatar and name over what they said. */
  private makeRun(speaker: Speaker, who: string): HTMLElement {
    const name = speaker === 'you' ? 'You' : speaker === 'agent' ? this.agentName() : who || 'Caller';
    const avatar = h('span.avatar', { class: `avatar-${speaker}`, 'aria-hidden': 'true' });
    const letters = speaker === 'caller' || speaker === 'texter' ? initials(who) : '';
    avatar.append(letters ? document.createTextNode(letters) : icon(speaker === 'you' ? 'user' : speaker === 'agent' ? this.agentIcon() : speaker === 'flow' ? 'flow' : speaker === 'caller' ? 'phone' : 'message'));
    return h(
      'section.run',
      { 'data-speaker': speaker, 'data-who': who },
      h('div.run-head', avatar, h('span.run-name', { 'data-role': speaker }, name)),
      h('div.run-body'),
    );
  }

  /** What the agent is called in the conversation shown. */
  private agentName(): string {
    if (this.kind === 'call' || this.kind === 'sms') return 'Receptionist';
    if (this.kind === 'own' && this.runner) return 'Runner';
    return 'Agent';
  }

  /** Take an entry out, and its run with it when that leaves the run empty. */
  private removeEntry(entry: Element): void {
    const body = entry.parentElement;
    entry.remove();
    if (body?.matches('.run-body') && !body.childElementCount) body.parentElement?.remove();
  }

  /**
   * The conversations to switch between: the project's own (id null), the
   * calls, the text-message threads and the flows' tasks, each with its unread
   * count and whether it is working. Hidden while there is only the project's own.
   */
  setSessions(tabs: ConversationTab[], active: string | null, select: (id: string | null) => void): void {
    this.sessionTabs.hidden = tabs.length < 2;
    this.sessionPicker.set(tabs, active, select);
    const shown = tabs.find((t) => t.id === active) ?? tabs[0];
    const own = tabs.find((t) => t.id === null);
    const runner = !!own && tabKind(own) === 'runner';
    if (runner !== this.runner) {
      this.runner = runner;
      this.applyNames();
      if (this.hero.isConnected) this.drawHero();
      if (!this.busy) this.input.placeholder = this.placeholder();
    }
    this.showLive(!!shown?.live, shown ? tabName(shown) : '');
  }

  /** Name the agent's runs as the conversation calls it (the runner's is known only once the conversations are listed). */
  private applyNames(): void {
    for (const el of this.log.querySelectorAll<HTMLElement>('.run-name[data-role="agent"]')) el.textContent = this.agentName();
    for (const el of this.log.querySelectorAll<HTMLElement>('.avatar-agent')) el.replaceChildren(icon(this.agentIcon()));
  }

  /** The agent's avatar: the runner's compass, or a spark. */
  private agentIcon(): string {
    return this.runner && this.kind === 'own' ? 'compass' : 'sparkle';
  }

  /** The strip over a call going on now: a pulsing dot, who, and how long. */
  private showLive(live: boolean, name: string): void {
    if (live !== this.live && !this.busy) {
      this.live = live;
      this.input.placeholder = this.placeholder();
    }
    this.live = live;
    this.liveBar.hidden = !live;
    if (this.liveTimer) clearInterval(this.liveTimer);
    this.liveTimer = null;
    if (!live) return;
    const tick = () => {
      clear(this.liveBar);
      const time = this.callStartAt ? clock(Date.now() - this.callStartAt) : '';
      this.liveBar.append(h('span.live-dot', { 'aria-hidden': 'true' }), h('span.live-label', 'Live call'), h('span.live-name', this.callName !== 'Caller' ? this.callName : name), h('span.live-time', { title: 'How long the call has run' }, time));
    };
    tick();
    this.liveTimer = setInterval(tick, 1000);
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
    this.add(h('details.msg.compacted', h('summary', icon('layers'), h('span', heading)), h('pre', body)));
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
      { type: 'button', title: open ? 'Hide the steps' : 'Show the steps', 'aria-expanded': String(open), onclick: () => {
        this.planOpen = !open;
        this.showPlan(plan, running);
      } },
      h('span.plan-title', finished ? '✓ Done' : running ? 'Working on' : 'Plan'),
      h('span.plan-goal', plan.goal || plan.items.find((i) => i.status === 'active')?.text || ''),
      h('span.plan-count', `${done}/${total}`),
      icon('chevron-down', 'plan-caret'),
    );
    const bar = h('div.plan-bar', h('span', { style: `width:${Math.round((done / total) * 100)}%` }));
    const toggle = (label: string, title: string) => h('li.plan-fold', h('button', { type: 'button', title, onclick: () => {
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
   * finished, the steps changed. Each step's work follows the mark where it
   * started.
   */
  private planMarks(before: Plan | null, after: Plan): void {
    const fresh = !before || (!!after.goal && !!before.goal && after.goal !== before.goal);
    const { done, started } = planChanges(before, after);
    const n = after.items.length;
    const mark = (cls: string, glyph: string, text: string) => {
      this.current = null;
      this.add(h(`div.msg.step${cls}`, h('span.step-icon', glyph), h('span.step-text', { title: text }, text)), 'agent');
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
    clear(this.feed);
    // What the agent is doing now shows at the end of the log, where its reply comes.
    this.feed.append(this.status);
    this.older.remove();
    this.follow.jump();
    this.showJump();
    this.cards.clear();
    this.current = null;
    this.thinking = null;
    this.updateEmpty();
  }

  /** The empty conversation's welcome: what it is for, and a few things to ask (they fill the box; nothing is sent). */
  private updateEmpty(): void {
    if (this.feed.querySelector('section.run, details.tool, .msg.user')) {
      this.hero.remove();
      return;
    }
    this.drawHero();
    // First in the log: notes from the app come after it, nearer the message box.
    if (!this.hero.isConnected) this.feed.prepend(this.hero);
  }

  private drawHero(): void {
    clear(this.hero);
    if (this.kind !== 'own') {
      const [glyph, words] = this.kind === 'call' ? ['phone', 'No words on this call yet.'] : this.kind === 'sms' ? ['message', 'No texts in this thread yet.'] : ['flow', 'No tasks from this flow yet.'];
      this.hero.className = 'chat-empty quiet';
      this.hero.append(h('span.chat-empty-icon', icon(glyph)), h('p', words));
      return;
    }
    const s = SUGGESTIONS[this.runner ? 'runner' : 'project'];
    this.hero.className = 'chat-empty';
    this.hero.append(
      h('span.chat-empty-icon', icon(this.runner ? 'compass' : 'sparkle')),
      h('h2', s.title),
      h('p', s.text),
      h('div.chat-suggestions', { role: 'group', 'aria-label': 'Suggestions' }, ...s.ideas.map((idea) => h('button.suggestion', { type: 'button', title: 'Put this in the message box (it is not sent until you send it)', onclick: () => this.suggest(idea) }, idea))),
    );
  }

  /** A suggestion goes into the message box, to change or send. */
  private suggest(text: string): void {
    this.input.value = text;
    this.grow();
    this.updateSend();
    this.input.focus();
    this.input.setSelectionRange(text.length, text.length);
  }

  private async copyCode(button: HTMLElement): Promise<void> {
    const code = button.closest('.code-block')?.querySelector('pre')?.textContent ?? '';
    let ok = false;
    try {
      await navigator.clipboard.writeText(code);
      ok = true;
    } catch {
      // No clipboard access: copy what is selected the old way.
      const area = h('textarea', { style: 'position:fixed;opacity:0' }, code) as HTMLTextAreaElement;
      document.body.append(area);
      area.select();
      ok = document.execCommand('copy');
      area.remove();
    }
    button.textContent = ok ? 'Copied' : 'Copy failed';
    button.classList.toggle('copied', ok);
    setTimeout(() => {
      button.textContent = 'Copy';
      button.classList.remove('copied');
    }, 1600);
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
    const glyph = app || /\.softn$/i.test(name) ? 'package' : mediaKind(path) === 'image' ? 'image' : 'file';
    return h('button.attachment', { type: 'button', title: app ? `Unpacked into ${app}/: click to preview it` : `${path}: click to open`, onclick: open }, h('span.file-icon', icon(glyph)), h('span.attachment-name', name));
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
   * drawn where they came (at the end), and the reply being written goes on in
   * its own box above them (it began first), not split around them. The log
   * follows them as it follows anything new: at the bottom, it keeps there.
   */
  heard(text: string): void {
    this.words(text, []);
    this.scroll();
  }

  /** A run ended (finished, or cut off): what the agent writes next is a new reply. */
  endReply(): void {
    this.current = null;
    this.thinking = null;
  }

  /** The person's message: a new request, or (`during`) one sent while the agent works, which keeps its plan. */
  user(text: string, attachments: Attachment[] = [], during = false, note = 'sent while the agent works: it reads it at its next step'): void {
    this.current = null;
    if (!during) {
      // A new request gets its own plan.
      this.currentPlan = null;
      this.planOpen = null;
      this.showPlan(null);
    }
    const own = this.words(text, attachments, during, note);
    if (this.sink || this.replaying) return;
    // The person's own message takes them to the bottom, where the reply comes; a caller's or a
    // texter's words (shown here when no reply is being written) are followed like anything new.
    if (own) this.toBottom();
    else this.scroll();
  }

  /**
   * A message's words, each part under who said it: on a call, the caller's
   * lines (with when, and how they fell against the agent's speech); in a text
   * thread, each text; a flow's task; notes from OAIY; and the person's own.
   */
  private words(text: string, attachments: Attachment[], during = false, note = ''): boolean {
    let parts: Part[];
    if (this.kind === 'call') parts = parseCallTurn(text);
    else if (this.kind === 'sms') parts = /^\[OAIY\] /.test(text) ? [{ kind: 'note', text: text.replace(/^\[OAIY\]\s*/, '') }] : parseTextTurn(text);
    else if (this.kind === 'task') parts = [parseFlowAsk(text) ?? (/^\[OAIY\] /.test(text) ? { kind: 'note', text: text.replace(/^\[OAIY\]\s*/, '') } : { kind: 'plain', text })];
    else parts = [{ kind: 'plain', text }];
    let own = false;
    for (const part of parts) {
      switch (part.kind) {
        case 'caller': {
          // A live call: the caller's time tells when the call began (the note that began it says only its minute).
          if (!this.replaying && part.atMs !== undefined) this.callStartAt = Date.now() - part.atMs;
          const tags: HTMLElement[] = [];
          const said = part.during ? `, as the agent said "${part.during}"` : '';
          if (part.backchannel) tags.push(h('span.msg-tag.backchannel', { title: `An acknowledgement said over the agent${said}: it talked on` }, 'backchannel'));
          else if (part.cut) tags.push(h('span.msg-tag.cut', { title: `Cut the agent off${said}` }, 'cut in'));
          else if (part.over) tags.push(h('span.msg-tag.over', { title: `Said over the agent${said}` }, 'talked over'));
          const meta = [...(part.atMs !== undefined ? [h('span.msg-time', { title: 'Into the call' }, clock(part.atMs))] : []), ...tags];
          const bubble = h('div.msg.incoming.caller', { class: part.backchannel ? 'backchannel' : '' }, h('div.msg-body', part.text || ' '), ...(meta.length ? [h('div.msg-meta', ...meta)] : []));
          this.add(bubble, 'caller', this.callName);
          break;
        }
        case 'text':
          this.add(h('div.msg.incoming.texter', h('div.msg-body', part.text || ' ')), 'texter', part.name ?? formatNumber(part.number));
          break;
        case 'flow':
          this.add(h('div.msg.incoming.flow-ask', h('div.msg-body', part.text || ' ')), 'flow', part.flow);
          break;
        case 'note':
          this.note(part.text);
          break;
        case 'plain':
          own = true;
          this.ownBubble(part.text, attachments, during, note);
          break;
      }
    }
    if (!own && (attachments.length || !parts.length)) {
      this.ownBubble(text && parts.length ? '' : text, attachments, during, note);
      own = true;
    }
    return own;
  }

  /** The person's own words, and what they attached. */
  private ownBubble(text: string, attachments: Attachment[], during: boolean, note: string): void {
    const box = h('div.msg.user', { class: during ? 'during' : '' }, h('div.msg-body', text || (attachments.length ? '' : ' ')));
    if (!text && attachments.length) box.classList.add('only-files');
    if (during) box.append(h('div.msg-note', note));
    if (attachments.length) box.append(h('div.msg-attachments', ...attachments.map((a) => this.fileView(a.path, a.name, a.app))));
    this.add(box, 'you');
  }

  /** A note from OAIY in a conversation (a lookup's answer, the runner's direction): quiet, and set apart. */
  private note(text: string): void {
    this.current = null;
    const [first, ...rest] = text.split('\n');
    this.add(rest.length
      ? h('details.msg.oaiy-note', h('summary', icon('info'), h('span', first)), h('pre', rest.join('\n')))
      : h('div.msg.oaiy-note', icon('info'), h('span', first)));
  }

  /** The model's thinking stays shown once the reply moves on: it says why the model did what it did. */
  private doneThinking(): void {
    const label = this.thinking?.box.querySelector('.thinking-label');
    if (label) label.textContent = 'What the model thought';
    this.thinking?.box.classList.remove('streaming');
    this.thinking = null;
  }

  /** A box with the model's thinking, open to read. */
  private thoughtBox(text: string, streaming = false): HTMLElement {
    const box = h('details.thinking', { open: true, class: streaming ? 'streaming' : '' }, h('summary', icon('sparkle'), h('span.thinking-label', streaming ? 'Thinking…' : 'What the model thought')), h('pre', text));
    this.add(box, 'agent');
    return box;
  }

  /** What the model was sent for this step: folded, and written out only when opened (it is long). */
  private promptView(e: { system: string; turns: Turn[]; tools: string[] }): void {
    this.current = null;
    const chars = e.system.length + e.turns.reduce((n, t) => n + (t.role === 'user' ? t.text.length : t.role === 'assistant' ? t.text.length + JSON.stringify(t.calls).length : t.results.reduce((m, r) => m + r.content.length, 0)), 0);
    const pre = h('pre');
    const box = h(
      'details.prompt-view',
      { title: 'What the model was sent for this step (click to read it)' },
      h('summary', icon('file-text'), h('span', `Prompt · ${e.turns.length} message${e.turns.length === 1 ? '' : 's'} · about ${formatTokens(Math.round(chars / 4))} tokens`)),
      pre,
    );
    box.addEventListener('toggle', () => {
      if ((box as HTMLDetailsElement).open && !pre.textContent) pre.textContent = promptText(e);
    });
    this.add(box, 'agent');
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
      if (this.draft) this.removeEntry(this.draft.box);
      this.current = null;
      const body = h('pre.tool-result.draft-body');
      const box = h('details.tool.draft', { open: true }, h('summary.tool-row', h('span.tool-status', { 'aria-hidden': 'true' }), icon('pencil', 'tool-icon'), h('span.tool-name', 'Writing a tool call…')), body);
      box.classList.add('pending');
      this.add(box, 'agent');
      this.draft = { box, body, raw: '', json };
    }
    this.draft.raw += text;
    const draft = this.draft;
    this.redraw(draft.box, () => {
      const raw = draft.raw;
      const name = draft.json ? '' : /<function=([\w.-]+)>/.exec(raw)?.[1] ?? '';
      if (name) draft.box.querySelector('.tool-name')!.textContent = `Writing ${toolLabel(name).toLowerCase()}…`;
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

  /** A message from the app itself (commands, errors, notices): a short one as a quiet line, a long one as a card. */
  system(text: string, kind: 'info' | 'error' = 'info'): void {
    this.current = null;
    const short = !text.includes('\n') && text.length <= 110;
    this.add(h('div.msg.system', { class: `${kind} ${short ? 'line' : 'card'}${text.includes('\n') ? ' lines' : ''}` }, icon(kind === 'error' ? 'alert' : 'info'), h('pre', text)));
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
      this.add(box, 'agent');
      this.current = { box, text: '', body };
    }
    this.current.text += delta;
    const target = this.current;
    // A saved reply is drawn whole, at once: the log lands at its end, and older turns drawn in
    // above what the reader sees keep it where it is, only if nothing grows after.
    if (this.replaying) {
      target.body.innerHTML = renderMarkdown(target.text);
      return;
    }
    // A long reply streams in many pieces: drawn a few times a second, not for each.
    this.redraw(target.box, () => {
      target.body.innerHTML = renderMarkdown(target.text);
    });
  }

  /** A tool call's row: what it did in plain words, on what, and (opened) its arguments and result. */
  private toolRow(call: ToolCall, extra = ''): HTMLDetailsElement {
    const prompt = mediaPrompt(call);
    const args = h('div.tool-args');
    const card = h(
      'details.tool',
      { class: extra, 'data-tool': call.name },
      h(
        'summary.tool-row',
        { title: `${call.name}${summarizeCall(call) ? `: ${summarizeCall(call)}` : ''}` },
        h('span.tool-status', { role: 'img', 'aria-label': 'running' }),
        icon(toolIcon(call.name), 'tool-icon'),
        h('span.tool-name', toolLabel(call.name)),
        h('span.tool-arg', summarizeCall(call)),
        h('span.tool-time'),
        ...(prompt ? [h('span.tool-prompt', { title: prompt }, prompt)] : []),
      ),
      h(
        'div.tool-detail',
        h('div.tool-section', h('div.tool-section-head', h('span', 'Input'), h('code.tool-raw', call.name)), args),
        h('div.tool-section.tool-output', h('div.tool-section-head', h('span', 'Result')), h('pre.tool-result', 'running…')),
      ),
    ) as HTMLDetailsElement;
    // The arguments (a whole file, sometimes) are laid out when the row is first opened.
    card.addEventListener('toggle', () => {
      if (!card.open || args.childElementCount) return;
      const lines = argLines(call.input);
      if (!lines.length) args.append(h('span.tool-empty', 'No arguments'));
      for (const { key, value, block } of lines) {
        const { text, more } = clip(value, 40, 4000);
        const shown = block ? h('pre.tool-arg-value', text) : h('span.tool-arg-value', text);
        const row = h('div.tool-arg-row', { class: block ? 'block' : '' }, h('span.tool-arg-key', key), shown);
        if (more > 0) row.append(h('button.tool-more', { type: 'button', onclick: (e: Event) => {
          shown.textContent = value;
          (e.currentTarget as HTMLElement).remove();
        } }, `Show all (${more.toLocaleString()} more characters)`));
        args.append(row);
      }
    });
    if (!this.replaying) card.dataset.started = String(performance.now());
    card.classList.add('pending');
    return card;
  }

  /** A tool's result in its row: laid out, cut to a readable length with the rest a click away. */
  private setResult(card: HTMLElement, content: string): void {
    const pre = card.querySelector('.tool-result');
    if (!pre) return;
    const full = prettyResult(content);
    const { text, more } = clip(full);
    pre.textContent = text || '(nothing)';
    pre.parentElement?.querySelector(':scope > .tool-more')?.remove();
    if (more > 0) pre.after(h('button.tool-more', { type: 'button', onclick: (e: Event) => {
      pre.textContent = full;
      (e.currentTarget as HTMLElement).remove();
    } }, `Show all (${more.toLocaleString()} more characters)`));
  }

  /** A row's state: running, done or failed (and how long it took, when it ran here). */
  private settle(card: HTMLElement, failed: boolean): void {
    card.classList.remove('pending');
    card.classList.add(failed ? 'failed' : 'ok');
    card.querySelector('.tool-status')?.setAttribute('aria-label', failed ? 'failed' : 'done');
    const started = Number(card.dataset.started);
    const time = card.querySelector('.tool-time');
    if (started && time) time.textContent = duration(performance.now() - started);
    const group = card.closest<HTMLElement>('details.tool-group');
    if (group) this.updateGroup(group);
  }

  /** Whether a row folds in with the calls around it (the sub-agents', checks, drafts and texts stand alone). */
  private groupable(el: Element): boolean {
    return el.matches('details.tool') && !el.matches('.delegate, .auto, .draft, .sms-out');
  }

  /**
   * A tool's row, at the end: folded in with the calls just before it ("Used 3
   * tools"), the prompts sent for those steps with them, or on its own when it
   * is the first.
   */
  private addTool(card: HTMLElement): void {
    const box = this.sink ?? this.feed;
    const run = this.endRun(box, 'agent');
    const body = run?.lastElementChild as HTMLElement | undefined;
    if (!body) {
      this.add(card, 'agent');
      return;
    }
    // The prompts sent since the last row (oldest first), and what came before them.
    const prompts: HTMLElement[] = [];
    let before = body.lastElementChild as HTMLElement | null;
    while (before?.matches('.prompt-view')) {
      prompts.unshift(before);
      before = before.previousElementSibling as HTMLElement | null;
    }
    let group: HTMLElement | null = null;
    if (before?.matches('details.tool-group')) group = before;
    else if (before && this.groupable(before)) {
      group = this.toolGroup();
      // The lone row, and the prompt sent for its step, go into the group.
      const prompt = before.previousElementSibling;
      const lone = prompt?.matches('.prompt-view') ? [prompt as HTMLElement, before] : [before];
      lone[0].before(group);
      group.lastElementChild!.append(...lone);
    }
    if (!group) {
      this.add(card, 'agent');
      return;
    }
    group.lastElementChild!.append(...prompts, card);
    this.updateGroup(group);
  }

  private toolGroup(): HTMLElement {
    return h(
      'details.tool-group',
      h('summary.tool-group-head', h('span.tool-status', { role: 'img', 'aria-label': 'running' }), icon('layers', 'tool-icon'), h('span.tool-group-title'), h('span.tool-group-names')),
      h('div.tool-group-list'),
    );
  }

  /** A group's line: how many tools, which, and whether one is running or failed. */
  private updateGroup(group: HTMLElement): void {
    const cards = [...group.lastElementChild!.children].filter((c): c is HTMLElement => c.matches('details.tool'));
    const running = cards.find((c) => c.classList.contains('pending'));
    const failed = cards.filter((c) => c.classList.contains('failed')).length;
    group.classList.toggle('pending', !!running);
    group.classList.toggle('failed', !running && failed > 0);
    group.classList.toggle('ok', !running && !failed);
    group.querySelector('.tool-status')?.setAttribute('aria-label', running ? 'running' : failed ? `${failed} failed` : 'done');
    const title = group.querySelector('.tool-group-title')!;
    const names = group.querySelector('.tool-group-names')!;
    if (running) {
      title.textContent = `Using ${cards.length} tools`;
      names.textContent = `${running.querySelector('.tool-name')?.textContent ?? ''} ${running.querySelector('.tool-arg')?.textContent ?? ''}`.trim();
    } else {
      title.textContent = `Used ${cards.length} tools`;
      const labels = [...new Set(cards.map((c) => c.querySelector('.tool-name')?.textContent ?? ''))].filter(Boolean);
      names.textContent = `${labels.slice(0, 3).join(' · ')}${labels.length > 3 ? ` · +${labels.length - 3}` : ''}${failed ? ` · ${failed} failed` : ''}`;
    }
  }

  /** What a row sits in: its group, when it is in one (what is shown beside it goes beside the group, where it is seen). */
  private anchorOf(card: HTMLElement): HTMLElement {
    return card.closest<HTMLElement>('details.tool-group') ?? card;
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
    // A text the agent sends in a text thread reads as a text.
    if (call.name === 'send_text_message' && this.kind === 'sms') {
      const card = this.toolRow(call, 'sms-out');
      const summary = card.querySelector('summary')!;
      clear(summary);
      summary.append(h('span.sms-body', typeof call.input.body === 'string' ? call.input.body : ''), h('span.sms-state', h('span.tool-status', { role: 'img', 'aria-label': 'sending' }), h('span.sms-state-text', 'Sending…')));
      this.cards.set(call.id, card);
      this.add(card, 'agent');
      this.scroll();
      return;
    }
    const card = this.toolRow(call);
    this.cards.set(call.id, card);
    this.addTool(card);
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
      { open: true, 'data-tool': 'delegate' },
      h('summary.tool-row', h('span.tool-status', { role: 'img', 'aria-label': 'running' }), icon('users', 'tool-icon'), h('span.tool-name', 'Sub-agents'), h('span.tool-arg', `${tasks.length} task${tasks.length === 1 ? '' : 's'}`)),
      list,
      h('pre.tool-result', 'waiting for the tasks…'),
    );
    card.classList.add('pending');
    tasks.forEach((task, i) => this.taskRow(`${call.id}#${i}`, task.title, list));
    this.cards.set(call.id, card);
    this.add(card, 'agent');
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
    // A sub-agent under another tool's row (the review of a picture): a list of its own just after the row (or its group), seen without opening it.
    if (!target && card) {
      const anchor = this.anchorOf(card);
      let next = anchor.nextElementSibling;
      while (next?.matches('.msg.presented')) next = next.nextElementSibling;
      target = next instanceof HTMLElement && next.matches('ol.agent-tasks.under') ? next : null;
      if (!target) {
        target = h('ol.agent-tasks.under');
        anchor.after(target);
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

  /** The automatic check after the agent changed an app: a row like a tool's. */
  private checkCard(e: Extract<AgentEvent, { type: 'check' }>): void {
    if (e.state === 'running') {
      this.current = null;
      this.thinking = null;
      this.setStatus(`checking ${e.root || 'the app'}…`);
      const card = h(
        'details.tool.auto',
        { 'data-tool': 'check' },
        h('summary.tool-row', h('span.tool-status', { role: 'img', 'aria-label': 'running' }), icon('app', 'tool-icon'), h('span.tool-name', 'Automatic check'), h('span.tool-arg', e.root ? `${e.root}/` : '/'), h('span.tool-time')),
        h('div.tool-detail', h('div.tool-section.tool-output', h('pre.tool-result', 'checking the files and rendering the app…'))),
      );
      card.classList.add('pending');
      if (!this.replaying) card.dataset.started = String(performance.now());
      this.cards.set(e.id, card);
      this.add(card, 'agent');
      this.scroll();
      return;
    }
    const card = this.cards.get(e.id);
    if (!card) return;
    this.settle(card, e.state !== 'ok');
    this.setResult(card, e.text ?? '');
    // A failed check is worth seeing without a click: the agent fixes it next.
    if (e.state === 'failed') (card as HTMLDetailsElement).open = true;
    this.scroll();
  }

  private toolResult(result: ToolResult): void {
    const card = this.cards.get(result.id);
    if (!card) return;
    this.settle(card, result.isError);
    this.setResult(card, result.content);
    if (card.classList.contains('sms-out')) {
      const test = /^Not sent \(a test conversation\)/.test(result.content);
      card.querySelector('.sms-state-text')!.textContent = result.isError ? 'Not sent' : test ? 'Shown, not sent (a test)' : 'Sent';
      card.querySelector('.sms-state .tool-status')?.setAttribute('aria-label', result.isError ? 'not sent' : 'sent');
    }
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
    const output = card.querySelector('.tool-output') ?? card;
    for (const image of result.images ?? []) {
      output.append(h('img.tool-image', { src: `data:${image.mediaType};base64,${image.data}`, alt: image.label ?? 'image shown to the model' }));
    }
    // A file the agent hands over is shown outside the (folded) row.
    if (result.files?.length) {
      // Pictures the agent made can be flagged; files it only shows (uploads, extracted frames) cannot.
      const shown = h('div.msg.presented', ...result.files.map((path) => this.fileView(path, undefined, undefined, result.name === 'generate_image')));
      this.anchorOf(card).after(shown);
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
        if (this.draft) this.removeEntry(this.draft.box);
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
          h('div.msg.nudge.compacted', icon('layers'), h('span', e.how === 'summary'
            ? `Context compacted: ${e.turns} earlier turns summarized for the model (${formatTokens(e.before)} → ${formatTokens(e.after)} tokens). The chat keeps everything.`
            : `Context compacted: the model could not write a summary, so ${e.turns} earlier turns were reduced to their requests and changes (${formatTokens(e.before)} → ${formatTokens(e.after)} tokens).`)),
        );
        this.scroll();
        break;
      case 'nudge':
        this.current = null;
        this.add(h('div.msg.nudge', icon('refresh'), h('span', `Not finished yet, so the agent carries on: ${e.message}`)), 'agent');
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

  /** Turns of a long saved conversation not drawn yet: drawn as the person scrolls up to them. */
  private hiddenTurns: Turn[] = [];
  /** The start of the log while older turns are left: scrolling near it draws them. */
  private readonly older = h('button.show-earlier', { type: 'button', onclick: () => this.loadOlder() });

  /**
   * Show a saved conversation, oldest first, at its end (the latest). A long
   * one opens at its latest turns, and older ones are drawn a page at a time
   * as the person scrolls up to them: drawing hundreds of tool cards at once
   * is slow.
   */
  replay(turns: Turn[], latest = REPLAY_PAGE): void {
    // What the conversation is, from its words (the list of conversations says it only after).
    this.kind = conversationKind(turns);
    this.callName = 'Caller';
    this.callStartAt = null;
    for (const t of turns) {
      const start = t.role === 'user' && t.automatic ? parseCallStart(t.text) : null;
      if (start) {
        this.callName = start.name;
        this.callStartAt = start.at?.getTime() ?? null;
      }
    }
    if (!this.busy) this.input.placeholder = this.placeholder();
    this.clearLog();
    const from = pageStart(turns, latest);
    this.hiddenTurns = turns.slice(0, from);
    // The plan the turns drawn change (one step at a time, maybe) is the one the older turns left.
    this.currentPlan = planAfter(this.hiddenTurns);
    this.drawTurns(turns.slice(from));
    this.showOlder();
    this.updateEmpty();
    // A conversation opens at its end, following what comes.
    this.toBottom();
  }

  /** The note at the start of the log: how much older conversation there is. */
  private showOlder(): void {
    if (!this.hiddenTurns.length) {
      this.older.remove();
      return;
    }
    this.older.textContent = `${this.hiddenTurns.length} older turns: scroll up, or click, to show them`;
    if (this.feed.firstElementChild !== this.older) this.feed.prepend(this.older);
    // A short page does not fill the log: nothing to scroll, so draw on.
    requestAnimationFrame(() => this.maybeLoadOlder());
  }

  private maybeLoadOlder(): void {
    if (this.hiddenTurns.length && this.log.scrollTop < 400) this.loadOlder();
  }

  /**
   * What the reader sees at the top of the log, and where: it is held there
   * while older turns go in above it.
   */
  private topAnchor(): { el: Element; top: number } | null {
    const view = this.log.getBoundingClientRect().top;
    // What is inside the runs (the older page's last run may go on into the first one shown, above what is read in it).
    const entries = [...this.feed.children].filter((el) => el !== this.older).flatMap((el) => (el.matches('section.run') ? [...(el.lastElementChild?.children ?? [])] : [el]));
    const spans = entries.map((el) => {
      const r = el.getBoundingClientRect();
      return { top: r.top - view, bottom: r.bottom - view };
    });
    const i = anchorIndex(spans);
    return i < 0 ? null : { el: entries[i], top: spans[i].top };
  }

  /** Draw the next page of older turns at the start of the log, leaving what is live (the current reply, the plan) as it is, and what the reader sees where it was. */
  private loadOlder(): void {
    if (!this.hiddenTurns.length || this.sink) return;
    const from = pageStart(this.hiddenTurns, REPLAY_PAGE);
    const page = this.hiddenTurns.slice(from);
    this.hiddenTurns = this.hiddenTurns.slice(0, from);
    const live = { current: this.current, thinking: this.thinking, draft: this.draft, plan: this.currentPlan, planOpen: this.planOpen, callName: this.callName };
    const anchor = this.topAnchor();
    // The page's marks follow the plan as it was then, not as it is now.
    this.currentPlan = planAfter(this.hiddenTurns);
    this.sink = h('div');
    try {
      this.drawTurns(page, false);
    } finally {
      // Where the pages meet, one speaker's run goes on under the header it has.
      let first = this.older.isConnected ? this.older.nextElementSibling : this.feed.firstElementChild;
      while (first && (first === this.hero || first === this.status)) first = first.nextElementSibling;
      const last = this.sink.lastElementChild;
      if (first instanceof HTMLElement && last instanceof HTMLElement && first.matches('section.run') && last.matches('section.run') && first.dataset.speaker === last.dataset.speaker && first.dataset.who === last.dataset.who) {
        first.lastElementChild!.prepend(...last.lastElementChild!.children);
        last.remove();
      }
      if (this.older.isConnected) this.older.after(...this.sink.children);
      else this.feed.prepend(...this.sink.children);
      this.sink = null;
      ({ current: this.current, thinking: this.thinking, draft: this.draft, plan: this.currentPlan, planOpen: this.planOpen, callName: this.callName } = live);
      this.showPlan(this.currentPlan);
      this.showOlder();
      // What the reader was looking at stays where it was (at the bottom, the log keeps to it).
      if (this.follow.stick) this.pin();
      else if (anchor?.el.isConnected) {
        const view = this.log.getBoundingClientRect().top;
        this.log.scrollTop = anchoredScrollTop(this.log.scrollTop, anchor.top, anchor.el.getBoundingClientRect().top - view);
      }
    }
  }

  /** Where a call began: the day and time, who, and what the phone said first. */
  private callStart(start: CallStart): void {
    this.current = null;
    this.callName = start.name;
    const what = start.direction === 'in' ? `Call from ${start.name}` : start.direction === 'back' ? `Called ${start.name} back` : `Called ${start.name}`;
    const when = start.at ? `${dayLabel(start.at)} · ${timeLabel(start.at)}` : start.when;
    this.add(h('div.day-divider', { title: start.number ? `${start.name} (${formatNumber(start.number)})` : start.name }, h('span.day-divider-text', icon('phone'), h('strong', when), h('span', what))));
    if (start.greeting) {
      this.add(h('div.msg.assistant.greeting', h('div.msg-body', start.greeting), h('div.msg-meta', h('span.msg-time', '0:00'), h('span.msg-tag', { title: 'What the phone said as it answered' }, 'greeting'))), 'agent');
    }
  }

  /** Draw turns in the order they came (each after the ones before). */
  private drawTurns(turns: Turn[], showPlan = true): void {
    this.replaying = true;
    try {
      for (const turn of turns) this.drawTurn(turn);
    } finally {
      this.replaying = false;
    }
    if (showPlan) this.showPlan(this.currentPlan, false);
  }

  private drawTurn(turn: Turn): void {
    if (turn.role === 'user' && turn.summary) {
      this.summaryNote(turn.text, 'Earlier conversation summarized for the model (click to read the summary)');
      return;
    }
    if (turn.role === 'user' && turn.automatic) {
      this.current = null;
      const start = parseCallStart(turn.text);
      if (start) {
        this.callStart(start);
        return;
      }
      const ended = parseCallEnd(turn.text);
      if (ended !== null) {
        this.add(h('div.call-end', h('span.call-end-pill', icon('phone-off'), h('span', ended ? `The call ended: ${ended}` : 'The call ended'))));
        return;
      }
      this.add(h('div.msg.nudge', icon('refresh'), h('span', turn.text.replace(/^\[(?:OAIY|bot\.computer)\] /, '').split('\n')[0])), 'agent');
      return;
    }
    if (turn.role === 'user') {
      // A message sent while the agent worked carries a note for the model: show it as the person wrote it.
      const during = /^\[The user sent this while you[^\]]*\]\n\n/.exec(turn.text);
      const text = (during ? turn.text.slice(during[0].length) : turn.text).replace(/^<project>[\s\S]*?<\/project>\n\n/, '');
      const attached = /\n\n\[Attached and saved in the project: ([\s\S]*)\]$/.exec(text);
      // Chats saved before attachments were kept on the turn: read them from the note.
      const fallback = attached ? attached[1].split('; ').map((n) => n.split(/ \(|: /)[0]).filter((p) => /^uploads\/[^\s]+$/.test(p)).map((path) => ({ name: path.split('/').pop()!, path })) : [];
      this.user(attached ? text.slice(0, attached.index) : text, turn.attachments ?? fallback, !!during);
    } else if (turn.role === 'assistant') {
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
}
