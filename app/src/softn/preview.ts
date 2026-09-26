/**
 * The live SoftN preview: the app rendered by SoftN's hosted runtime (from a
 * softn.com release, installed into public/softn/ by scripts/fetch-softn.mjs)
 * in an opaque-origin, sandboxed iframe with its own strict CSP. The frame
 * gets the app's files and the Zipp engine's bytes over postMessage, and
 * reports load and render errors back on a MessagePort.
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
  /** When the render these errors belong to started. */
  at: number;
}

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

  constructor(private vfs: Vfs, private readonly projectId: () => string) {
    const reload = h('button', { title: 'Render the app again', onclick: () => this.render() }, '↻ Reload');
    this.picker.addEventListener('change', () => this.setApp(this.picker.value));
    this.element.append(h('div.pane-title', 'App preview ', this.picker, this.status, reload), this.frameHost);
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

  private settle(result: Omit<PreviewResult, 'at'>): void {
    this.current = { ...result, at: this.current.at };
    for (const w of this.waiters.splice(0)) w(this.current);
  }

  /** Render now and wait for the outcome (for the agent's softn_check). */
  async check(root?: string): Promise<PreviewResult> {
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
    frame.setAttribute('sandbox', 'allow-scripts allow-forms allow-modals allow-popups');
    // Lets the page embed the frame under its own COEP even where the host
    // cannot add headers to the frame's response.
    frame.setAttribute('credentialless', '');
    frame.setAttribute('allow', '');
    frame.src = `${import.meta.env.BASE_URL}softn/index.html?v=${generation}`;
    const errors: string[] = [];
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
          this.settle({ ok: false, errors: [...errors] });
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
          this.settle({ ok: false, errors: [...errors] });
          return;
        }
        this.setStatus(`live · ${new Date().toLocaleTimeString()}`, 'ok');
        this.settle({ ok: true, errors: [] });
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
