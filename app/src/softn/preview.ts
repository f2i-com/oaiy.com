/**
 * The live SoftN preview: the app rendered by SoftN's hosted runtime (from a
 * softn.com release, installed into public/softn/ by scripts/fetch-softn.mjs)
 * in an opaque-origin, sandboxed iframe with its own strict CSP. The frame
 * gets the app's files and the Zipp engine's bytes over postMessage, and
 * reports load and render errors back on a MessagePort.
 *
 * bot.computer's bridge (scripts/softn-bridge/bot-bridge.js, loaded in the
 * frame before the runtime) reports what goes wrong while the app runs, and
 * describes or operates the page for the agent (inspect, act).
 *
 * The runtime takes one app per page load, so a change reloads the frame
 * (after edits settle); the app's in-memory state starts over, as it does in
 * SoftN Studio.
 */
import { zippBase } from '../sandbox/runner';
import type { Vfs } from '../vfs/vfs';
import { clear, h } from '../ui/dom';
import { appFiles, appLabel, findApps, isSoftnApp, logicSyntax } from './softn';

export interface PreviewResult {
  ok: boolean;
  errors: string[];
  /** console.warn from the app while it loaded (the runtime warns about missing handlers). */
  warnings?: string[];
  /** When the render these errors belong to started. */
  at: number;
}

/** Something the running app reported through the bridge. */
export interface Problem {
  level: 'error' | 'warning';
  message: string;
  at: number;
}

/** What the page shows after inspect or act. */
export interface PageReport {
  ok: boolean;
  page: string;
  done?: string[];
  error?: string;
  /** Problems the app reported while this ran. */
  problems: Problem[];
}

export type PreviewAction = { click: string; nth?: number } | { fill: string; value: string; nth?: number } | { select: string; value: string; nth?: number } | { key: string } | { wait: number };

let engineBytes: Promise<ArrayBuffer> | null = null;
function zippBytes(): Promise<ArrayBuffer> {
  engineBytes ??= fetch(`${zippBase()}zipp_wasm_bg.wasm`).then((r) => {
    if (!r.ok) throw new Error(`the Zipp engine could not be loaded (${r.status})`);
    return r.arrayBuffer();
  });
  engineBytes.catch(() => (engineBytes = null));
  return engineBytes;
}

let runtimeCheck: Promise<boolean> | null = null;
/** Is SoftN's hosted runtime installed alongside this app? */
export function softnRuntimeAvailable(): Promise<boolean> {
  runtimeCheck ??= fetch(`${import.meta.env.BASE_URL}softn/runtime-manifest.json`, { cache: 'no-store' })
    .then((r) => r.ok && (r.headers.get('content-type') ?? '').includes('json'))
    .catch(() => false);
  return runtimeCheck;
}

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

const decoder = new TextDecoder('utf-8', { fatal: true });
/** How long a render has to report an error before it counts as clean. */
const SETTLE_MS = 2500;

export class SoftnPreview {
  readonly element = h('section.preview');
  private readonly status = h('span.preview-status', '');
  private readonly picker = h('select.preview-app', { title: 'Which SoftN app to show' });
  /** The app folder shown ('' is the project root). */
  app = '';
  private readonly frameHost = h('div.preview-frame');

  private timer: ReturnType<typeof setTimeout> | null = null;
  private stale = true;
  private visible = false;
  private generation = 0;
  private current: PreviewResult = { ok: false, errors: [], at: 0 };
  private waiters: Array<(r: PreviewResult) => void> = [];
  private settleTimer: ReturnType<typeof setTimeout> | null = null;
  private frame: HTMLIFrameElement | null = null;
  /** Problems the running app reported since it was last rendered. */
  problems: Problem[] = [];
  /** The render in progress collects the bridge's errors here. */
  private collecting: { errors: string[]; warnings: string[]; settle: () => void } | null = null;
  private requests = new Map<number, (result: Omit<PageReport, 'problems'>) => void>();
  private nextRequest = 1;
  private readonly banner = h('div.preview-problem');
  private readonly bannerText = h('span.preview-problem-text');
  private readonly fixButton = h('button.primary', { title: 'Send these errors to the agent and ask it to fix the app' }, 'Fix with agent');
  /** Set by the page: ask the agent to fix the app in `root`. Returns false when the agent is busy. */
  onFix: ((root: string, problems: Problem[]) => boolean) | null = null;

  constructor(private vfs: Vfs, private readonly projectId: () => string) {
    const reload = h('button', { title: 'Render the app again', onclick: () => this.render() }, '↻ Reload');
    this.picker.addEventListener('change', () => this.setApp(this.picker.value));
    this.banner.append(
      h('span.preview-problem-icon', '⚠'),
      this.bannerText,
      this.fixButton,
      h('button', { title: 'Hide until the next error', onclick: () => (this.banner.hidden = true) }, 'Dismiss'),
    );
    this.banner.hidden = true;
    this.fixButton.addEventListener('click', () => {
      const problems = this.problems.filter((p) => p.level === 'error');
      if (!problems.length || !this.onFix) return;
      if (this.onFix(this.app, problems)) this.banner.hidden = true;
    });
    window.addEventListener('message', (event) => this.fromBridge(event));
    this.element.append(h('div.pane-title', 'App preview ', this.picker, this.status, reload), this.banner, this.frameHost);
    this.picker.hidden = true;
    this.showMessage('Open or start a SoftN app (/softn new) to see it here.');
  }

  /** The SoftN apps in the project, for the picker; keeps the choice if it still exists. */
  refreshApps(): string[] {
    const apps = findApps(this.vfs);
    if (!apps.includes(this.app)) this.app = apps[0] ?? '';
    clear(this.picker);
    for (const root of apps) this.picker.append(h('option', { value: root, selected: root === this.app }, root ? `${root}/` : '/ (project root)'));
    this.picker.hidden = apps.length < 2;
    return apps;
  }

  /** Show the app in `root`. */
  setApp(root: string): void {
    this.app = root;
    this.refreshApps();
    this.stale = true;
    if (this.visible) this.render();
  }

  setVfs(vfs: Vfs): void {
    this.vfs = vfs;
    this.app = '';
    this.refreshApps();
    this.stale = true;
    this.current = { ok: false, errors: [], at: 0 };
    if (this.visible) this.render();
  }

  setVisible(visible: boolean): void {
    this.visible = visible;
    if (visible && this.stale) this.render();
  }

  /** The project changed: re-render after edits settle (the agent writes in bursts). */
  changed(path: string | null): void {
    if (path === null || /(^|\/)manifest\.json$/i.test(path)) this.refreshApps();
    if (path !== null && this.app && !path.startsWith(`${this.app}/`)) return;
    if (path !== null && !/(^|\/)(manifest\.json|permission\.json)$|\.(ui|logic|py|xdb)$|(^|\/)assets\//i.test(path)) return;
    this.stale = true;
    if (!this.visible) return;
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.render(), 700);
  }

  private showMessage(text: string): void {

    clear(this.frameHost);
    this.frameHost.append(h('div.empty', text));
  }

  private setStatus(text: string, kind: 'ok' | 'busy' | 'error' = 'busy'): void {
    this.status.textContent = text;
    this.status.dataset.kind = kind;
  }

  private fromBridge(event: MessageEvent): void {
    const data = event.data as { __botComputer?: boolean; type?: string; level?: string; message?: string; at?: number; id?: number; result?: Omit<PageReport, 'problems'> };
    if (!this.frame || event.source !== this.frame.contentWindow || data?.__botComputer !== true) return;
    if (data.type === 'bot:problem' && typeof data.message === 'string') {
      const problem: Problem = { level: data.level === 'warning' ? 'warning' : 'error', message: data.message.slice(0, 2000), at: Date.now() };
      this.problems.push(problem);
      if (this.problems.length > 100) this.problems.shift();
      if (this.collecting) {
        (problem.level === 'error' ? this.collecting.errors : this.collecting.warnings).push(problem.message);
        if (problem.level === 'error') this.collecting.settle();
      }
      if (problem.level === 'error') this.showProblems();
    } else if (data.type === 'bot:reply' && typeof data.id === 'number') {
      this.requests.get(data.id)?.(data.result ?? { ok: false, page: '', error: 'no reply' });
      this.requests.delete(data.id);
    }
  }

  private showProblems(): void {
    const errors = this.problems.filter((p) => p.level === 'error');
    if (!errors.length) {
      this.banner.hidden = true;
      return;
    }
    const first = errors[errors.length - 1].message.split('\n')[0];
    this.bannerText.textContent = errors.length > 1 ? `${errors.length} errors in the app. Latest: ${first}` : `The app reported an error: ${first}`;
    this.bannerText.title = errors.map((p) => p.message).join('\n\n');
    this.fixButton.hidden = !this.onFix;
    this.banner.hidden = false;
    this.setStatus(`error: ${first}`, 'error');
  }

  private request(message: Record<string, unknown>, timeoutMs = 20_000): Promise<Omit<PageReport, 'problems'>> {
    const frame = this.frame?.contentWindow;
    if (!frame) return Promise.resolve({ ok: false, page: '', error: 'the preview is not showing an app' });
    const id = this.nextRequest++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.requests.delete(id);
        resolve({ ok: false, page: '', error: 'the preview did not answer (is bot.computer\'s bridge installed? run npm run fetch:softn -- --ensure)' });
      }, timeoutMs);
      this.requests.set(id, (result) => {
        clearTimeout(timer);
        resolve(result);
      });
      frame.postMessage({ __botComputer: true, id, ...message }, '*');
    });
  }

  /** Render `root` unless it is already live and current. */
  private async ready(root?: string): Promise<PreviewResult> {
    if ((root !== undefined && root !== this.app) || this.stale || !this.frame || this.current.at === 0) return this.checkNow(root);
    return this.current;
  }

  /** Describe what the app shows, as text. */
  inspect(root?: string): Promise<PageReport> {
    return this.exclusive(() => this.inspectNow(root));
  }

  private async inspectNow(root?: string): Promise<PageReport> {
    const rendered = await this.ready(root);
    if (!this.frame) return { ok: false, page: '', error: rendered.errors.join('; ') || 'nothing is rendered', problems: [] };
    const since = Date.now();
    const result = await this.request({ type: 'bot:inspect' });
    return { ...result, problems: this.problems.filter((p) => p.at >= since) };
  }

  /** Click, fill, choose and press keys in the app, then describe it. */
  act(root: string | undefined, actions: PreviewAction[]): Promise<PageReport> {
    return this.exclusive(() => this.actNow(root, actions));
  }

  private async actNow(root: string | undefined, actions: PreviewAction[]): Promise<PageReport> {
    const rendered = await this.ready(root);
    if (!this.frame) return { ok: false, page: '', error: rendered.errors.join('; ') || 'nothing is rendered', problems: [] };
    const since = Date.now();
    const result = await this.request({ type: 'bot:act', actions }, 60_000);
    // Errors a click causes can arrive a moment after it.
    await new Promise((r) => setTimeout(r, 200));
    return { ...result, problems: this.problems.filter((p) => p.at >= since) };
  }

  private settle(result: Omit<PreviewResult, 'at'>): void {
    this.current = { ...result, at: this.current.at };
    for (const w of this.waiters.splice(0)) w(this.current);
  }

  /**
   * One check, inspect or act at a time: agents working in parallel would
   * otherwise switch the preview to their own app under each other.
   */
  private lock: Promise<unknown> = Promise.resolve();
  private exclusive<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.lock.then(fn, fn);
    this.lock = run.catch(() => {});
    return run;
  }

  /** Render now and wait for the outcome (for the agent's softn_check). */
  check(root?: string): Promise<PreviewResult> {
    return this.exclusive(() => this.checkNow(root));
  }

  private async checkNow(root?: string): Promise<PreviewResult> {
    if (root !== undefined && root !== this.app) {
      this.app = root;
      this.refreshApps();
    }
    if (!(await softnRuntimeAvailable())) return { ok: false, errors: ['the SoftN preview runtime is not installed (npm run fetch:softn); only the file checks ran'], at: Date.now() };
    const done = new Promise<PreviewResult>((resolve) => this.waiters.push(resolve));
    this.render();
    return done;
  }

  async render(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const generation = ++this.generation;
    this.stale = false;
    this.current = { ok: false, errors: [], at: Date.now() };
    this.problems = [];
    this.collecting = null;
    this.banner.hidden = true;
    this.frame = null;
    if (this.settleTimer) clearTimeout(this.settleTimer);
    if (!isSoftnApp(this.vfs, this.app)) {
      this.setStatus('');
      this.showMessage(this.app ? `${appLabel(this.app)} is not a SoftN app any more.` : 'No SoftN app here yet: an app is a folder whose manifest.json has a "main" .ui page. Try /softn new, import a .softn, or ask the agent to build one.');
      this.settle({ ok: false, errors: [`${appLabel(this.app)} is not a SoftN app: manifest.json with a "main" .ui page is missing`] });
      return;
    }
    const root = this.app;
    if (!(await softnRuntimeAvailable())) {
      this.setStatus('');
      this.showMessage('The SoftN preview runtime is not installed. Run `npm run fetch:softn` and reload bot.computer. You can still build, check and export .softn apps.');
      this.settle({ ok: false, errors: ['the SoftN preview runtime is not installed'] });
      return;
    }
    this.setStatus('rendering…');
    // The runtime shows a logic syntax error inside the frame without
    // reporting it, so the logic is also compiled here, on Zipp.
    const syntax = logicSyntax(this.vfs, root).catch(() => []);
    const client: Record<string, string> = {};
    const assets: Record<string, string> = {};
    for (const [path, data] of appFiles(this.vfs, root)) {
      if (path.startsWith('assets/')) {
        assets[path] = toBase64(data);
        continue;
      }
      try {
        client[path] = decoder.decode(data);
      } catch {
        assets[path] = toBase64(data);
      }
    }
    let bytes: ArrayBuffer;
    try {
      bytes = await zippBytes();
    } catch (error) {
      this.setStatus('failed', 'error');
      this.settle({ ok: false, errors: [(error as Error).message] });
      return;
    }
    if (generation !== this.generation) return;

    const frame = h('iframe', { title: 'SoftN app preview' }) as HTMLIFrameElement;
    // Opaque origin: the app cannot reach bot.computer's storage, files or keys.
    frame.setAttribute('sandbox', 'allow-scripts allow-forms allow-modals');
    // Lets the page embed the frame under its own COEP even where the host
    // cannot add headers to the frame's response.
    frame.setAttribute('credentialless', '');
    frame.setAttribute('allow', '');
    frame.src = `${import.meta.env.BASE_URL}softn/index.html?v=${generation}`;
    const errors: string[] = [];
    const warnings: string[] = [];
    this.frame = frame;
    this.collecting = {
      errors,
      warnings,
      settle: () => {
        if (generation !== this.generation) return;
        this.setStatus(`error: ${errors[errors.length - 1]}`, 'error');
        this.settle({ ok: false, errors: [...errors], warnings: [...warnings] });
      },
    };
    const onMessage = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.data?.type !== 'formlogic:ready') return;
      window.removeEventListener('message', onMessage);
      const channel = new MessageChannel();
      channel.port1.onmessage = (e: MessageEvent) => {
        if (generation !== this.generation) return;
        const data = e.data as { type?: string; reason?: string; id?: unknown };
        if (data?.type === 'error') {
          errors.push(data.reason || 'the app failed to load');
          this.setStatus(`error: ${errors[errors.length - 1]}`, 'error');
          this.settle({ ok: false, errors: [...errors], warnings: [...warnings] });
        } else if (data?.type === 'call') {
          channel.port1.postMessage({ id: data.id, result: { error: 'This preview has no backend: server/ actions run only when the app is deployed.' } });
        }
      };
      frame.contentWindow!.postMessage(
        { type: 'formlogic:init', client, assets, appId: `bot.computer-${this.projectId()}-${root || 'root'}`, dark: matchMedia('(prefers-color-scheme: dark)').matches, engine: 'zipp-web-python', zippWasm: bytes.slice(0) },
        '*',
        [channel.port2],
      );
      this.settleTimer = setTimeout(async () => {
        const problems = (await syntax).map((f) => `${f.file}: ${f.message}`);
        if (generation !== this.generation || errors.length) return;
        if (problems.length) {
          errors.push(...problems);
          this.setStatus(`error: ${problems[0]}`, 'error');
          this.settle({ ok: false, errors: [...errors], warnings: [...warnings] });
          return;
        }
        this.setStatus(`live · ${new Date().toLocaleTimeString()}`, 'ok');
        this.settle({ ok: true, errors: [], warnings: [...warnings] });
      }, SETTLE_MS);
    };
    window.addEventListener('message', onMessage);
    setTimeout(() => {
      if (generation !== this.generation || this.status.dataset.kind !== 'busy') return;
      window.removeEventListener('message', onMessage);
      this.setStatus('the preview runtime did not start', 'error');
      this.settle({ ok: false, errors: ['the SoftN preview runtime did not start within 20 s'] });
    }, 20_000);
    clear(this.frameHost);

    this.frameHost.append(frame);
  }
}
