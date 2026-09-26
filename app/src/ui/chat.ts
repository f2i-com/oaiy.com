/**
 * The chat pane: the conversation with the agent, streamed as it arrives,
 * with a card per tool call; and the prompt box, which also takes slash
 * commands.
 */
import type { AgentEvent } from '../agent/agent';
import type { Attachment, ToolCall, ToolResult, Turn } from '../agent/protocol';
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
    case 'present_file': case 'view_image': case 'file_info': case 'search_file': return s('path');
    case 'softn_import': return s('path');
    case 'softn_check': case 'softn_docs': return s('app');
    default: return '';
  }
}

export class ChatPane {
  readonly element = h('section.chat');
  private readonly log = h('div.chat-log');
  private readonly input = h('textarea.chat-input', { rows: 3, placeholder: 'Ask bot.computer…  (/help for commands)', title: 'Enter sends; Shift+Enter starts a new line' });
  private readonly send = h('button.primary', 'Send');
  private readonly attachButton = h('button.attach', { title: 'Attach files or images (or drop them here, or paste an image)', 'aria-label': 'Attach files' }, '📎');
  private readonly picker = h('input', { type: 'file', multiple: true, style: 'display:none' });
  private readonly pending = h('div.attachments');
  private files: File[] = [];
  private readonly status = h('div.chat-status');
  private current: { box: HTMLElement; text: string; body: HTMLElement } | null = null;
  private thinking: { box: HTMLElement; text: string } | null = null;
  private cards = new Map<string, HTMLElement>();
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
    this.element.append(h('div.pane-title', 'Agent'), this.log, this.status, this.pending, h('div.chat-compose', this.attachButton, this.input, this.send), this.picker);
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

  private renderPending(): void {
    clear(this.pending);
    this.files.forEach((file, i) => {
      const thumb = file.type.startsWith('image/') ? h('img', { src: URL.createObjectURL(file), alt: '' }) : h('span.file-icon', '📄');
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
    this.send.textContent = busy ? 'Stop' : 'Send';
    this.send.classList.toggle('danger', busy);
    if (!busy) this.setStatus('');
  }

  setStatus(text: string): void {
    this.status.textContent = text;
  }

  private scroll(): void {
    this.log.scrollTop = this.log.scrollHeight;
  }

  clearLog(): void {
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

  private assistantText(delta: string): void {
    if (!this.current) {
      const body = h('div.msg-body');
      const box = h('div.msg.assistant', body);
      this.log.append(box);
      this.current = { box, text: '', body };
    }
    this.current.text += delta;
    this.current.body.innerHTML = renderMarkdown(this.current.text);
    this.scroll();
  }

  private toolCard(call: ToolCall): void {
    this.current = null;
    this.thinking = null;
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

  private toolResult(result: ToolResult): void {
    const card = this.cards.get(result.id);
    if (!card) return;
    card.classList.remove('pending');
    card.classList.add(result.isError ? 'failed' : 'ok');
    const pre = card.querySelector('.tool-result');
    if (pre) pre.textContent = result.content;
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
      case 'done':
        this.current = null;
        break;
      case 'error':
        this.system(e.message, 'error');
        break;
    }
  }

  /** Show a saved conversation. */
  replay(turns: Turn[]): void {
    this.clearLog();
    for (const turn of turns) {
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
        for (const call of turn.calls) this.toolCard(call);
      } else for (const r of turn.results) this.toolResult(r);
    }
  }
}
