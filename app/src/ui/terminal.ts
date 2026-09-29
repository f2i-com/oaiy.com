/**
 * The terminal pane: the same emulated shell the agent uses, for the person.
 * Commands run on the Zipp VM against the project, through the network gate.
 * It folds down to its header (remembered in this browser), giving the editor
 * the room.
 */
import type { NetGate } from '../gate/netgate';
import { SandboxHost, summarize } from '../sandbox/host';
import { runInSandbox } from '../sandbox/runner';
import type { Vfs } from '../vfs/vfs';
import { h } from './dom';
import { icon } from './icons';

/** Where this browser remembers the terminal folded. */
const FOLDED = 'oaiy.terminal.folded';

function remembered(): boolean {
  try {
    return localStorage.getItem(FOLDED) === '1';
  } catch {
    return false;
  }
}

function remember(folded: boolean): void {
  try {
    localStorage.setItem(FOLDED, folded ? '1' : '0');
  } catch {
    // storage can be unavailable: it is only a convenience
  }
}

export class TerminalPane {
  readonly element = h('section.terminal');
  private readonly out = h('pre.term-out');
  private readonly cwdLabel = h('span.term-cwd', '/');
  private readonly prompt = h('span.term-prompt', { 'aria-hidden': 'true' }, this.cwdLabel, h('span.term-sigil', '$'));
  private readonly input = h('input.term-input', { spellcheck: false, autocomplete: 'off', 'aria-label': 'Terminal command', placeholder: 'ls, grep -rn TODO ., python script.py, help…' });
  private readonly toggle = h('button.term-toggle', { type: 'button' }) as HTMLButtonElement;
  private readonly where = h('span.term-where');
  private cwd = '/';
  private env: Record<string, string> = {};
  private history: string[] = [];
  private cursor = 0;
  private running = false;
  private queue: string[] = [];
  /** Stops the command that is running (Ctrl+C). */
  private stop: AbortController | null = null;

  constructor(private vfs: Vfs, private readonly gate: NetGate) {
    const head = h(
      'div.pane-title.term-head',
      { title: 'An emulated shell on the Zipp VM, confined to the project' },
      h('span.pane-kicker', 'Terminal'),
      h('span.term-about', 'emulated shell on the Zipp VM, confined to the project'),
      this.where,
      this.toggle,
    );
    // The header folds it too (not its button's own click, which does it already).
    head.addEventListener('click', (e) => {
      if (!(e.target as Element).closest('button')) this.fold(!this.element.classList.contains('folded'));
    });
    this.toggle.addEventListener('click', (e) => {
      // Its icon changes as it folds: the header must not take the same click again.
      e.stopPropagation();
      this.fold(!this.element.classList.contains('folded'));
    });
    this.element.append(head, this.out, h('div.term-line', this.prompt, this.input));
    this.fold(remembered(), false);
    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') void this.run(this.input.value);
      else if (e.key === 'ArrowUp' && this.history.length) {
        this.cursor = Math.max(0, this.cursor - 1);
        this.input.value = this.history[this.cursor] ?? '';
        e.preventDefault();
      } else if (e.key === 'ArrowDown') {
        this.cursor = Math.min(this.history.length, this.cursor + 1);
        this.input.value = this.history[this.cursor] ?? '';
        e.preventDefault();
      } else if (e.key === 'l' && e.ctrlKey) {
        this.out.textContent = '';
        e.preventDefault();
      } else if (e.key === 'c' && e.ctrlKey && !this.input.value.slice(this.input.selectionStart ?? 0, this.input.selectionEnd ?? 0) && this.running) {
        this.queue = [];
        this.stop?.abort();
        this.write('^C\n', 'term-err');
        e.preventDefault();
      }
    });
    this.write('Type `help` to see what the shell can do.\n');
  }

  /** Fold the terminal down to its header, or open it (and remember which, for this browser). */
  private fold(folded: boolean, save = true): void {
    this.element.classList.toggle('folded', folded);
    this.toggle.setAttribute('aria-expanded', String(!folded));
    this.toggle.title = folded ? 'Show the terminal' : 'Hide the terminal';
    this.toggle.setAttribute('aria-label', this.toggle.title);
    this.toggle.replaceChildren(icon(folded ? 'chevron-up' : 'chevron-down'));
    if (save) remember(folded);
    if (!folded && save) this.input.focus();
  }

  setVfs(vfs: Vfs): void {
    this.vfs = vfs;
    this.cwd = '/';
    this.env = {};
    this.queue = [];
    this.showPrompt();
    this.out.textContent = '';
  }

  private showPrompt(): void {
    this.cwdLabel.textContent = this.cwd;
    this.where.textContent = this.cwd === '/' ? '' : this.cwd;
  }

  private write(text: string, cls?: string): void {
    const span = h('span', text);
    if (cls) span.className = cls;
    this.out.append(span);
    this.out.scrollTop = this.out.scrollHeight;
  }

  private async run(command: string): Promise<void> {
    // A command entered while another runs goes next, in order.
    if (this.running) {
      if (command.trim()) this.queue.push(command);
      this.input.value = '';
      return;
    }
    this.input.value = '';
    if (!command.trim()) return;
    this.history.push(command);
    this.cursor = this.history.length;
    if (this.out.textContent && !this.out.textContent.endsWith('\n')) this.write('\n');
    this.write(`${this.cwd} $ ${command}\n`, 'term-cmd');
    if (command.trim() === 'clear') {
      this.out.textContent = '';
      return;
    }
    this.running = true;
    this.prompt.classList.add('busy');
    this.stop = new AbortController();
    try {
      const host = new SandboxHost(this.vfs, this.gate, 'terminal', 60_000);
      host.signal = this.stop.signal;
      const cwd = this.vfs.stat(this.cwd)?.type === 'dir' ? this.cwd : '/';
      const outcome = await runInSandbox({ lang: 'shell', source: command, cwd, env: this.env, limits: { maxSteps: 2_000_000_000 } }, host, { timeoutMs: 60_000, signal: this.stop.signal });
      const shell = outcome.result?.shell;
      if (shell && !outcome.result?.error) {
        this.cwd = shell.cwd;
        this.env = shell.env;
        if (shell.stdout) this.write(shell.stdout);
        if (shell.stderr) this.write(shell.stderr, 'term-err');
      } else {
        const { stdout, stderr } = summarize(outcome);
        if (stdout) this.write(stdout);
        this.write(stderr || 'the command did not finish\n', 'term-err');
      }
    } catch (error) {
      this.write(`${(error as Error).message}\n`, 'term-err');
    } finally {
      this.running = false;
      this.prompt.classList.remove('busy');
      this.showPrompt();
      this.stop = null;
      // Back to the prompt only when the person was typing in the terminal, not somewhere else meanwhile.
      if (document.activeElement === document.body || this.element.contains(document.activeElement)) this.input.focus();
    }
    const next = this.queue.shift();
    if (next !== undefined) await this.run(next);
  }
}
