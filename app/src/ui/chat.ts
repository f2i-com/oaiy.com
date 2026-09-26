/**
 * The chat pane: the conversation with the agent, streamed as it arrives,
 * with a card per tool call; and the prompt box, which also takes slash
 * commands.
 */
import type { AgentEvent } from '../agent/agent';
import type { Attachment, ToolCall, ToolResult, Turn } from '../agent/protocol';
import { readPlan, readTasks, type Plan } from '../agent/tools';
import { formatTokens } from '../agent/context';
import { clear, h } from './dom';
import { mediaElement, mediaKind, type Media } from './media';
import { renderMarkdown } from './markdown';

function summarizeCall(call: ToolCall): string {
  const i = call.input;
  const s = (k: string) => (typeof i[k] === 'string' ? String(i[k]) : '');
  switch (call.name) {
    case 'read_file': case 'write_file': case 'edit_file': case 'delete_file': case 'list_files': return s('path') || '/';
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

export class ChatPane {
  readonly element = h('section.chat');
  private readonly log = h('div.chat-log', { role: 'log', 'aria-live': 'polite', 'aria-relevant': 'additions' });
  private readonly input = h('textarea.chat-input', { rows: 3, placeholder: 'Ask bot.computer…  (/help for commands)', title: 'Enter sends; Shift+Enter starts a new line' });
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
  /** The person's choice to show or hide the steps; null follows the run (hidden once finished). */
  private planOpen: boolean | null = null;
  private current: { box: HTMLElement; text: string; body: HTMLElement } | null = null;
  private thinking: { box: HTMLElement; text: string } | null = null;
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
    },
  ) {
    this.planBox.hidden = true;
    this.meter.hidden = true;
    this.log.addEventListener('scroll', () => {
      this.stick = this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight < 60;
    }, { passive: true });
    this.element.append(h('div.pane-title', 'Agent', this.meter), this.planBox, this.log, this.status, this.pending, h('div.chat-compose', this.attachButton, this.input, this.send), this.picker);
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
    this.send.addEventListener('click', () => (this.busy ? this.handlers.stop() : this.submit()));
    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        if (!this.busy) this.submit();
      }
    });
  }

  addFiles(files: File[]): void {
    this.files.push(...files);
    this.renderPending();
    this.input.focus();
  }

  private pendingUrls: string[] = [];
  private renderPending(): void {
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
    this.handlers.submit(text, files);
  }

  focus(): void {
    this.input.focus();
  }

  setBusy(busy: boolean): void {
    this.busy = busy;
    if (this.currentPlan) this.showPlan(this.currentPlan, busy);
    this.send.textContent = busy ? 'Stop' : 'Send';
    this.send.classList.toggle('danger', busy);
    if (!busy) this.setStatus('');
  }

  setStatus(text: string): void {
    this.status.textContent = text;
  }

  /** Follow new output only while the person is at the bottom: scrolled up to read, they stay put. */
  private stick = true;
  private scrollQueued = false;
  private scroll(): void {
    if (!this.stick || this.scrollQueued) return;
    this.scrollQueued = true;
    requestAnimationFrame(() => {
      this.scrollQueued = false;
      if (this.stick) this.log.scrollTop = this.log.scrollHeight;
    });
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
    const body = text.replace(/^\[bot\.computer\][^\n]*\n(<project>[\s\S]*?<\/project>\n\n)?/, '');
    this.log.append(h('details.msg.compacted', h('summary', h('span', '⇣'), h('span', ` ${heading}`)), h('pre', body)));
    this.scroll();
  }

  /** Show the checklist: the goal, progress, and each step's state. */
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
    const list = h(
      'ol.plan-items',
      ...plan.items.map((item) =>
        h(`li.plan-item.${item.status}`, { class: item.status === 'active' && running ? 'running' : '' }, h('span.plan-mark', { 'aria-label': item.status }, item.status === 'done' ? '✓' : item.status === 'active' ? '' : ''), h('span.plan-text', item.text)),
      ),
    );
    list.hidden = !open;
    this.planBox.append(head, bar, list);
  }

  private currentPlan: Plan | null = null;

  clearLog(): void {
    this.meter.hidden = true;
    this.hiddenTurns = [];
    this.taskRows.clear();
    this.currentPlan = null;
    this.showPlan(null);
    for (const m of this.media.splice(0)) m.dispose();
    clear(this.log);
    this.cards.clear();
    this.current = null;
    this.thinking = null;
  }

  /**
   * A project file shown in the log: an image as a thumbnail, audio and
   * video with a player, anything else as a chip; each opens the file.
   */
  private fileView(path: string, name = path.split('/').pop() ?? path, app?: string): HTMLElement {
    const open = () => this.handlers.open(path, app);
    const bytes = mediaKind(path) ? this.handlers.file(path) : null;
    const media = bytes ? mediaElement(path, bytes, { compact: true, onOpen: open }) : null;
    if (media) {
      this.media.push(media);
      return media.element;
    }
    const icon = app || /\.softn$/i.test(name) ? '📦' : mediaKind(path) === 'image' ? '🖼' : '📄';
    return h('button.attachment', { title: app ? `Unpacked into ${app}/: click to preview it` : `${path}: click to open`, onclick: open }, h('span.file-icon', icon), h('span.attachment-name', name));
  }

  user(text: string, attachments: Attachment[] = []): void {
    this.current = null;
    this.stick = true;
    // A new request gets its own plan.
    this.currentPlan = null;
    this.planOpen = null;
    this.showPlan(null);
    const box = h('div.msg.user', h('div.msg-body', text || (attachments.length ? '' : ' ')));
    if (attachments.length) box.append(h('div.msg-attachments', ...attachments.map((a) => this.fileView(a.path, a.name, a.app))));
    this.log.append(box);
    this.scroll();
  }

  /** A message from the app itself (commands, errors, notices). */
  system(text: string, kind: 'info' | 'error' = 'info'): void {
    this.current = null;
    this.log.append(h('div.msg.system', { class: kind }, h('pre', text)));
    this.scroll();
  }

  private renderQueued = false;
  private assistantText(delta: string): void {
    if (!this.current) {
      const body = h('div.msg-body');
      const box = h('div.msg.assistant', body);
      this.log.append(box);
      this.current = { box, text: '', body };
    }
    this.current.text += delta;
    // A long reply streams in many pieces: draw it at most once a frame.
    if (this.renderQueued) return;
    this.renderQueued = true;
    const target = this.current;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      target.body.innerHTML = renderMarkdown(target.text);
      this.scroll();
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
      h('summary', h('span.tool-name', call.name), ' ', h('span.tool-arg', summarizeCall(call))),
      h('pre.tool-input', JSON.stringify(call.input, null, 2)),
      result,
    );
    card.classList.add('pending');
    this.cards.set(call.id, card);
    this.log.append(card);
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
    this.log.append(card);
    this.scroll();
  }

  private taskRow(id: string, title: string, list?: HTMLElement): { row: HTMLElement; state: HTMLElement; activity: HTMLElement; report: HTMLElement } {
    const existing = this.taskRows.get(id);
    if (existing) return existing;
    const state = h('span.task-state', 'queued');
    const activity = h('span.task-activity');
    const report = h('pre.task-report');
    const row = h('li.agent-task.queued', h('div.task-head', h('span.task-mark'), h('span.task-title', title), state), activity, h('details.task-more', h('summary', 'report'), report));
    const target = list ?? this.cards.get(id.split('#')[0])?.querySelector('.agent-tasks');
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
      this.log.append(card);
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
      const shown = h('div.msg.presented', ...result.files.map((path) => this.fileView(path)));
      card.after(shown);
    }
    this.scroll();
  }

  event(e: AgentEvent): void {
    switch (e.type) {
      case 'text':
        this.thinking = null;
        this.assistantText(e.delta);
        break;
      case 'thinking':
        if (!this.thinking) {
          const box = h('details.thinking', h('summary', 'thinking…'), h('pre'));
          this.log.append(box);
          this.thinking = { box, text: '' };
        }
        this.thinking.text += e.delta;
        this.thinking.box.querySelector('pre')!.textContent = this.thinking.text;
        break;
      case 'tool_start':
        this.setStatus(`writing a ${e.name} call…`);
        break;
      case 'tool_call':
        this.setStatus(`running ${e.call.name}…`);
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
        if (this.currentPlan?.goal !== e.plan.goal) this.planOpen = null;
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
        this.log.append(
          h('div.msg.nudge.compacted', h('span', '⇣'), h('span', e.how === 'summary'
            ? ` Context compacted: ${e.turns} earlier turns summarized for the model (${formatTokens(e.before)} → ${formatTokens(e.after)} tokens). The chat keeps everything.`
            : ` Context compacted: the model could not write a summary, so ${e.turns} earlier turns were reduced to their requests and changes (${formatTokens(e.before)} → ${formatTokens(e.after)} tokens).`)),
        );
        this.scroll();
        break;
      case 'nudge':
        this.current = null;
        this.log.append(h('div.msg.nudge', h('span', '↻'), h('span', ` Not finished yet, so the agent carries on: ${e.message}`)));
        this.scroll();
        break;
      case 'done':
        this.current = null;
        break;
      case 'error':
        this.system(e.message, 'error');
        break;
    }
  }

  /** Turns of a long saved conversation not drawn yet ("Show earlier"). */
  private hiddenTurns: Turn[] = [];

  /**
   * Show a saved conversation. A long one opens at its latest turns, with a
   * button for the rest: drawing hundreds of tool cards at once is slow.
   */
  replay(turns: Turn[], latest = 120): void {
    this.clearLog();
    let from = Math.max(0, turns.length - latest);
    while (from > 0 && turns[from]?.role !== 'user') from--;
    if (from > 0) {
      this.hiddenTurns = turns.slice(0, from);
      const more = h('button.show-earlier', { onclick: () => {
        const all = [...this.hiddenTurns, ...turns.slice(from)];
        const top = this.log.scrollHeight - this.log.scrollTop;
        this.replay(all, Infinity);
        this.stick = false;
        this.log.scrollTop = this.log.scrollHeight - top;
      } }, `Show ${from} earlier turns`);
      this.log.append(more);
    }
    this.drawTurns(turns.slice(from));
  }

  private drawTurns(turns: Turn[]): void {
    for (const turn of turns) {
      if (turn.role === 'user' && turn.summary) {
        this.summaryNote(turn.text, 'Earlier conversation summarized for the model (click to read the summary)');
        continue;
      }
      if (turn.role === 'user' && turn.automatic) {
        this.current = null;
        this.log.append(h('div.msg.nudge', h('span', '↻'), h('span', ` ${turn.text.replace(/^\[bot\.computer\] /, '').split('\n')[0]}`)));
        continue;
      }
      if (turn.role === 'user') {
        const text = turn.text.replace(/^<project>[\s\S]*?<\/project>\n\n/, '');
        const attached = /\n\n\[Attached and saved in the project: ([\s\S]*)\]$/.exec(text);
        // Chats saved before attachments were kept on the turn: read them from the note.
        const fallback = attached ? attached[1].split('; ').map((n) => n.split(/ \(|: /)[0]).filter((p) => /^uploads\/[^\s]+$/.test(p)).map((path) => ({ name: path.split('/').pop()!, path })) : [];
        this.user(attached ? text.slice(0, attached.index) : text, turn.attachments ?? fallback);
      }
      else if (turn.role === 'assistant') {
        if (turn.text) this.assistantText(turn.text);
        this.current = null;
        for (const call of turn.calls) {
          this.toolCard(call);
          if (call.name === 'update_plan') {
            try {
              this.currentPlan = readPlan(call.input);
            } catch {
              /* a plan the tool refused */
            }
          }
        }
      } else for (const r of turn.results) this.toolResult(r);
    }
    this.showPlan(this.currentPlan, false);
  }
}
