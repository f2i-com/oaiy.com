/**
 * The chat pane: the conversation with the agent, streamed as it arrives,
 * with a card per tool call; and the prompt box, which also takes slash
 * commands.
 */
import type { AgentEvent } from '../agent/agent';
import type { ToolCall, ToolResult, Turn } from '../agent/protocol';
import { clear, h } from './dom';
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
    default: return '';
  }
}

export class ChatPane {
  readonly element = h('section.chat');
  private readonly log = h('div.chat-log');
  private readonly input = h('textarea.chat-input', { rows: 3, placeholder: 'Ask bot.computer…  (/help for commands)', title: 'Enter sends; Shift+Enter starts a new line' });
  private readonly send = h('button.primary', 'Send');
  private readonly status = h('div.chat-status');
  private current: { box: HTMLElement; text: string; body: HTMLElement } | null = null;
  private thinking: { box: HTMLElement; text: string } | null = null;
  private cards = new Map<string, HTMLElement>();
  private busy = false;

  constructor(private readonly handlers: { submit: (text: string) => void; stop: () => void }) {
    this.element.append(h('div.pane-title', 'Agent'), this.log, this.status, h('div.chat-compose', this.input, this.send));
    this.send.addEventListener('click', () => (this.busy ? this.handlers.stop() : this.submit()));
    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        if (!this.busy) this.submit();
      }
    });
  }

  private submit(): void {
    const text = this.input.value.trim();
    if (!text) return;
    this.input.value = '';
    this.handlers.submit(text);
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
    clear(this.log);
    this.cards.clear();
    this.current = null;
    this.thinking = null;
  }

  user(text: string): void {
    this.current = null;
    this.log.append(h('div.msg.user', h('div.msg-body', text)));
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
      if (turn.role === 'user') this.user(turn.text.replace(/^<project>[\s\S]*?<\/project>\n\n/, ''));
      else if (turn.role === 'assistant') {
        if (turn.text) this.assistantText(turn.text);
        this.current = null;
        for (const call of turn.calls) this.toolCard(call);
      } else for (const r of turn.results) this.toolResult(r);
    }
  }
}
