import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { NetGate } from '../../src/gate/netgate';
import { runInSandbox } from '../../src/sandbox/runner';
import { TerminalPane, inOrder } from '../../src/ui/terminal';
import { Vfs } from '../../src/vfs/vfs';

// DOM rendering uses EventTargets and sandbox I/O is mocked. TerminalPane
// installs its real listeners and runs its real input, history and queue logic.
const { ElementStub } = vi.hoisted(() => {
  class ElementStub extends EventTarget {
    className = '';
    title = '';
    value = '';
    scrollTop = 0;
    scrollHeight = 0;
    children: Array<ElementStub | string> = [];
    classList = {
      contains: (name: string) => this.className.split(' ').includes(name),
      toggle: (name: string, force: boolean) => {
        const classes = new Set(this.className.split(' ').filter(Boolean));
        if (force) classes.add(name);
        else classes.delete(name);
        this.className = [...classes].join(' ');
        return force;
      },
      add: (name: string) => { this.classList.toggle(name, true); },
      remove: (name: string) => { this.classList.toggle(name, false); },
    };
    get textContent(): string {
      return this.children.map((child) => typeof child === 'string' ? child : child.textContent).join('');
    }
    set textContent(text: string) { this.children = [text]; }
    append(...children: Array<ElementStub | string>): void { this.children.push(...children); }
    replaceChildren(...children: Array<ElementStub | string>): void { this.children = children; }
    setAttribute(): void {}
    focus(): void {}
    contains(child: unknown): boolean {
      return this === child || this.children.some((entry) => entry instanceof ElementStub && entry.contains(child));
    }
    find(className: string): ElementStub | undefined {
      if (this.classList.contains(className)) return this;
      for (const child of this.children) {
        if (child instanceof ElementStub) {
          const match = child.find(className);
          if (match) return match;
        }
      }
    }
  }
  return { ElementStub };
});

vi.mock('../../src/ui/dom', () => ({
  h: (tag: string, props?: unknown, ...children: Array<InstanceType<typeof ElementStub> | string>) => {
    const element = new ElementStub();
    element.className = tag.split('.').slice(1).join(' ');
    if (typeof props === 'string' || props instanceof ElementStub) children.unshift(props);
    element.append(...children);
    return element;
  },
}));
vi.mock('../../src/ui/icons', () => ({ icon: () => new ElementStub() }));
vi.mock('../../src/sandbox/host', () => ({
  SandboxHost: class {},
  summarize: () => ({ stdout: '', stderr: '' }),
}));
vi.mock('../../src/sandbox/runner', () => ({ runInSandbox: vi.fn() }));

type KeyOptions = Pick<KeyboardEventInit, 'code' | 'location' | 'keyCode' | 'isComposing' | 'ctrlKey'>;

// Browser KeyboardEvent fields are inherited getters, not enumerable own
// properties. Use that shape in the Node test environment (which has no DOM).
class KeyboardEventStub extends Event {
  constructor(private readonly keyValue: string, private readonly options: KeyOptions = {}) {
    super('keydown', { cancelable: true });
  }
  get key(): string { return this.keyValue; }
  get code(): string { return this.options.code ?? ''; }
  get location(): number { return this.options.location ?? 0; }
  get keyCode(): number { return this.options.keyCode ?? 0; }
  get isComposing(): boolean { return this.options.isComposing ?? false; }
  get ctrlKey(): boolean { return this.options.ctrlKey ?? false; }
}

function legacyEnter(): Event {
  const event = new Event('keydown', { cancelable: true });
  Object.defineProperties(event, { key: { value: 'Enter' }, keyCode: { value: 229 } });
  return event; // Older event shapes can omit isComposing entirely.
}

const outcome = { result: { shell: { cwd: '/', env: {}, stdout: '', stderr: '' } } } as Awaited<ReturnType<typeof runInSandbox>>;

function terminal() {
  const pane = new TerminalPane(new Vfs(), {} as NetGate);
  const element = pane.element as unknown as InstanceType<typeof ElementStub>;
  const input = element.find('term-input')!;
  const output = element.find('term-out')!;
  // Observe the real command boundary without substituting the key handler.
  const run = vi.spyOn(pane as unknown as { run(command: string): Promise<void> }, 'run');
  return { input, output, run };
}

beforeEach(() => {
  vi.mocked(runInSandbox).mockReset().mockResolvedValue(outcome);
  const body = new ElementStub();
  vi.stubGlobal('document', { body, activeElement: body });
  vi.stubGlobal('localStorage', { getItem: () => null });
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const composingEnters = [
  { name: 'a composing Enter with prototype getters', event: () => new KeyboardEventStub('Enter', { isComposing: true, keyCode: 13 }) },
  { name: 'a composing NumpadEnter', event: () => new KeyboardEventStub('Enter', { isComposing: true, code: 'NumpadEnter', location: 3 }) },
  { name: 'legacy keyCode 229 with isComposing false', event: () => new KeyboardEventStub('Enter', { keyCode: 229, isComposing: false }) },
  { name: 'legacy keyCode 229 without isComposing', event: legacyEnter },
];

describe('terminal composition keys', () => {
  it.each(composingEnters)('leaves the draft and output intact for $name', ({ event }) => {
    const { input, output, run } = terminal();
    input.value = 'echo 日本語';
    const before = output.textContent;
    const key = event();
    expect(Object.hasOwn(key, 'isComposing')).toBe(false);

    input.dispatchEvent(key);

    expect(run).not.toHaveBeenCalled();
    expect(runInSandbox).not.toHaveBeenCalled();
    expect(input.value).toBe('echo 日本語');
    expect(output.textContent).toBe(before);
    expect(key.defaultPrevented).toBe(false);
  });

  it.each(composingEnters)('does not queue a draft behind a running command for $name', async ({ event }) => {
    let finish!: (result: typeof outcome) => void;
    vi.mocked(runInSandbox).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    const { input, run } = terminal();
    input.value = 'echo active';
    input.dispatchEvent(new KeyboardEventStub('Enter'));
    const active = run.mock.results[0].value as Promise<void>;
    input.value = 'echo 日本語';

    input.dispatchEvent(event());
    const draft = input.value;
    finish(outcome);
    await active;

    expect(run).toHaveBeenCalledTimes(1);
    expect(runInSandbox).toHaveBeenCalledTimes(1);
    expect(draft).toBe('echo 日本語');
    expect(input.value).toBe('echo 日本語');
  });

  it('runs only on the first ordinary Enter after composition ends', async () => {
    const { input, run } = terminal();
    input.value = 'echo 日本語';
    input.dispatchEvent(new Event('compositionstart'));
    input.dispatchEvent(new KeyboardEventStub('Enter', { isComposing: true }));
    expect(run).not.toHaveBeenCalled();
    input.dispatchEvent(new Event('compositionend'));
    expect(run).not.toHaveBeenCalled();
    expect(input.value).toBe('echo 日本語');

    input.dispatchEvent(new KeyboardEventStub('Enter'));
    await run.mock.results[0].value;

    expect(run).toHaveBeenCalledExactlyOnceWith('echo 日本語');
    expect(runInSandbox).toHaveBeenCalledTimes(1);
    expect(input.value).toBe('');
  });

  it('leaves Process keys to the input method', () => {
    const { input, run } = terminal();
    input.value = 'echo 日本語';
    const key = new KeyboardEventStub('Process');
    input.dispatchEvent(key);
    expect(run).not.toHaveBeenCalled();
    expect(input.value).toBe('echo 日本語');
    expect(key.defaultPrevented).toBe(false);
  });

  it.each([
    { key: 'ArrowUp', options: { isComposing: true } },
    { key: 'ArrowDown', options: { isComposing: true } },
    { key: 'ArrowUp', options: { keyCode: 229, isComposing: false } },
    { key: 'ArrowDown', options: { keyCode: 229, isComposing: false } },
  ])('preserves the draft and history cursor for $key during composition ($options)', async ({ key, options }) => {
    const { input, run } = terminal();
    for (const command of ['echo first', 'echo second']) {
      input.value = command;
      input.dispatchEvent(new KeyboardEventStub('Enter'));
      await run.mock.results.at(-1)!.value;
    }
    input.value = 'echo 日本語';
    const composingArrow = new KeyboardEventStub(key, options);
    input.dispatchEvent(composingArrow);
    expect(input.value).toBe('echo 日本語');
    expect(composingArrow.defaultPrevented).toBe(false);

    const up = new KeyboardEventStub('ArrowUp');
    input.dispatchEvent(up);
    expect(input.value).toBe('echo second');
    expect(up.defaultPrevented).toBe(true);
    const down = new KeyboardEventStub('ArrowDown');
    input.dispatchEvent(down);
    expect(input.value).toBe('');
    expect(down.defaultPrevented).toBe(true);
    expect(runInSandbox).toHaveBeenCalledTimes(2);
  });
});

describe('terminal ordinary Enter compatibility', () => {
  it.each([
    { name: 'ordinary Enter with browser defaults', options: {} },
    { name: 'ordinary Enter with keyCode 13', options: { keyCode: 13 } },
    { name: 'NumpadEnter', options: { code: 'NumpadEnter', location: 3 } },
  ])('executes exactly once and preserves event defaults for $name', async ({ options }) => {
    const { input, run } = terminal();
    input.value = 'echo ready';
    const enter = new KeyboardEventStub('Enter', options);
    input.dispatchEvent(enter);
    await run.mock.results[0].value;

    expect(run).toHaveBeenCalledExactlyOnceWith('echo ready');
    expect(runInSandbox).toHaveBeenCalledTimes(1);
    expect(vi.mocked(runInSandbox).mock.calls[0][0]).toMatchObject({ source: 'echo ready', lang: 'shell' });
    expect(input.value).toBe('');
    expect(enter.defaultPrevented).toBe(false);
  });
});

describe('terminal output', () => {
  it('puts stdout and stderr in the order the shell wrote them', () => {
    expect(inOrder({ stdout: 'one\nrc=1\n', stderr: 'cat: x: No such file or directory\n', order: [[1, 4], [2, 34], [1, 5]] })).toEqual([
      { text: 'one\n', err: false },
      { text: 'cat: x: No such file or directory\n', err: true },
      { text: 'rc=1\n', err: false },
    ]);
    // An older shell's report (no order) is its stdout, then its stderr; what an order cut short does not reach comes after.
    expect(inOrder({ stdout: 'a\n', stderr: 'b\n' })).toEqual([{ text: 'a\n', err: false }, { text: 'b\n', err: true }]);
    expect(inOrder({ stdout: 'abc', stderr: 'xyz', order: [[2, 1], [1, 1]] })).toEqual([
      { text: 'x', err: true },
      { text: 'abc', err: false },
      { text: 'yz', err: true },
    ]);
  });

  it('draws a command\'s output in that order, its errors marked', async () => {
    vi.mocked(runInSandbox).mockResolvedValueOnce({ result: { shell: { cwd: '/', env: {}, stdout: 'one\nrc=1\n', stderr: 'oops\n', order: [[1, 4], [2, 5], [1, 5]] } } } as unknown as typeof outcome);
    const { input, output, run } = terminal();
    input.value = 'echo one; cat x; echo rc=$?';
    input.dispatchEvent(new KeyboardEventStub('Enter'));
    await run.mock.results[0].value;
    const drawn = output.children.slice(-3) as Array<InstanceType<typeof ElementStub>>;
    expect(drawn.map((piece) => [piece.textContent, piece.className])).toEqual([['one\n', ''], ['oops\n', 'term-err'], ['rc=1\n', '']]);
  });

  it('says how to stop a running command, and its Stop button stops it as Ctrl+C does', async () => {
    let signal: AbortSignal | undefined;
    let finish!: (result: typeof outcome) => void;
    vi.mocked(runInSandbox).mockImplementationOnce((_request, _host, options) => {
      signal = options?.signal;
      return new Promise((resolve) => { finish = resolve; });
    });
    const pane = new TerminalPane(new Vfs(), {} as NetGate);
    const element = pane.element as unknown as InstanceType<typeof ElementStub>;
    const input = element.find('term-input')!;
    const stop = element.find('term-stop')!;
    const output = element.find('term-out')!;
    input.value = 'python -c "while True: pass"';
    input.dispatchEvent(new KeyboardEventStub('Enter'));
    await Promise.resolve();
    expect((stop as unknown as { hidden: boolean }).hidden).toBe(false);
    expect((input as unknown as { placeholder: string }).placeholder).toMatch(/Ctrl\+C stops it/);

    stop.dispatchEvent(new Event('click'));
    expect(signal?.aborted).toBe(true);
    expect(output.textContent).toContain('^C');
    finish(outcome);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect((stop as unknown as { hidden: boolean }).hidden).toBe(true);
    expect((input as unknown as { placeholder: string }).placeholder).toMatch(/^ls, grep/);
  });
});
