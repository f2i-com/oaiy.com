/**
 * The terminal pane: the same emulated shell the agent uses, for the person.
 * Commands run on the Zipp VM against the project, through the network gate.
 */
import type { NetGate } from '../gate/netgate';
import { SandboxHost, summarize } from '../sandbox/host';
import { runInSandbox } from '../sandbox/runner';
import type { Vfs } from '../vfs/vfs';
import { h } from './dom';

export class TerminalPane {
  readonly element = h('section.terminal');
  private readonly out = h('pre.term-out');
  private readonly prompt = h('span.term-prompt', '/ $');
  private readonly input = h('input.term-input', { spellcheck: false, autocomplete: 'off', placeholder: 'ls, grep -rn TODO ., python script.py, help…' });
  private cwd = '/';
  private env: Record<string, string> = {};
  private history: string[] = [];
  private cursor = 0;
  private running = false;
  private queue: string[] = [];
  /** Stops the command that is running (Ctrl+C). */
  private stop: AbortController | null = null;

  constructor(private vfs: Vfs, private readonly gate: NetGate) {
    this.element.append(
      h('div.pane-title', 'Terminal ', h('span.muted', '— emulated shell on the Zipp VM, confined to the project')),
      this.out,
      h('div.term-line', this.prompt, this.input),
    );
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

  setVfs(vfs: Vfs): void {
    this.vfs = vfs;
    this.cwd = '/';
    this.env = {};
    this.queue = [];
    this.prompt.textContent = '/ $';
    this.out.textContent = '';
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
      this.prompt.textContent = `${this.cwd} $`;
      this.stop = null;
      // Back to the prompt only when the person was typing in the terminal, not somewhere else meanwhile.
      if (document.activeElement === document.body || this.element.contains(document.activeElement)) this.input.focus();
    }
    const next = this.queue.shift();
    if (next !== undefined) await this.run(next);
  }
}
