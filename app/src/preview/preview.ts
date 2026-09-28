/**
 * The live preview: a SoftN app or a web page of the project, in an
 * opaque-origin, sandboxed iframe with its own strict CSP.
 *
 * - A SoftN app is rendered by SoftN's hosted runtime (from a softn.com
 *   release, installed into public/softn/ by scripts/fetch-softn.mjs). The
 *   frame gets the app's files and the Zipp engine's bytes over postMessage,
 *   and reports load and render errors back on a MessagePort.
 * - A web page (any .html file of the project, with its CSS and JavaScript) is
 *   built by public/webpage/page-host.js from the project's files, and its
 *   JavaScript runs on the Zipp VM against the frame's real DOM, through the
 *   facade in zippDom.guest.js.
 *
 * bot.computer's bridge (scripts/softn-bridge/bot-bridge.js, in both frames)
 * reports what goes wrong while the page runs, and describes, operates and
 * screenshots it for the agent. The frame can be given a viewport size
 * (phone, tablet, …) for responsive design; it is then scaled to fit the pane,
 * while the page inside lays out at the size chosen.
 *
 * A frame takes one app or page per load, so a change reloads it (after edits
 * settle); the page's in-memory state starts over, as it does in SoftN Studio.
 */
import { zippBase } from '../sandbox/runner';
import type { Vfs } from '../vfs/vfs';
import { clear, h } from '../ui/dom';
import { appFiles, appLabel, findApps, isSoftnApp, logicSyntax } from '../softn/softn';
import { findPages, pageFiles, pageLabel } from './page';
import FACADE from './zippDom.guest.js?raw';

export type PreviewTarget = { kind: 'app'; root: string } | { kind: 'page'; path: string };

export interface PreviewResult {
  ok: boolean;
  errors: string[];
  /** console.warn from the app while it loaded (the runtime warns about missing handlers); for a page, files it could not load. */
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

export interface Shot {
  png: Uint8Array;
  width: number;
  height: number;
  /** The page's full height, and how far it was scrolled. */
  pageHeight: number;
  scrollY: number;
  /** A full-page shot that stopped short of a very long page. */
  cut: boolean;
}

export interface Viewport {
  width: number;
  height: number;
}

/** The viewport's size now: chosen, or the pane's when it fits the pane. */
export interface ViewportInfo extends Viewport {
  fit: boolean;
  /** How much the frame is shrunk to fit the pane (1 = not at all). */
  scale: number;
  preset?: string;
}

export const VIEWPORTS: Record<string, Viewport & { label: string }> = {
  phone: { width: 390, height: 844, label: 'Phone' },
  tablet: { width: 820, height: 1180, label: 'Tablet' },
  laptop: { width: 1366, height: 768, label: 'Laptop' },
  desktop: { width: 1920, height: 1080, label: 'Desktop' },
};
export const MIN_VIEWPORT = 200;
export const MAX_VIEWPORT = 3840;

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

let glueSource: Promise<string> | null = null;
/** The engine's JavaScript bindings, which the page frame imports itself (it has an opaque origin). */
function zippGlue(): Promise<string> {
  glueSource ??= fetch(`${zippBase()}zipp_wasm.js`).then((r) => {
    if (!r.ok) throw new Error(`the Zipp engine could not be loaded (${r.status})`);
    return r.text();
  });
  glueSource.catch(() => (glueSource = null));
  return glueSource;
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
/** A request's answer when the frame was re-rendered under it. */
const RELOADED = 'the preview was re-rendered while this ran';

export const targetLabel = (target: PreviewTarget): string => (target.kind === 'app' ? appLabel(target.root) : pageLabel(target.path));
const sameTarget = (a: PreviewTarget | null, b: PreviewTarget | null): boolean =>
  !!a && !!b && a.kind === b.kind && (a.kind === 'app' ? a.root === (b as { root: string }).root : a.path === (b as { path: string }).path);
const targetValue = (t: PreviewTarget): string => (t.kind === 'app' ? `app:${t.root}` : `page:${t.path}`);

type BridgeReply = Omit<PageReport, 'problems'> & { shot?: Omit<Shot, 'png'> & { png: ArrayBuffer } };

export class Preview {
  readonly element = h('section.preview');
  private readonly status = h('span.preview-status', '');
  private readonly picker = h('select.preview-app', { title: 'What to show: a SoftN app or a web page of the project' });
  private readonly sizePicker = h('select.preview-size', { title: 'The size of the screen the page is shown on' });
  private readonly sizeWidth = h('input.preview-size-input', { type: 'number', min: String(MIN_VIEWPORT), max: String(MAX_VIEWPORT), title: 'Width in CSS pixels' }) as HTMLInputElement;
  private readonly sizeHeight = h('input.preview-size-input', { type: 'number', min: String(MIN_VIEWPORT), max: String(MAX_VIEWPORT), title: 'Height in CSS pixels' }) as HTMLInputElement;
  private readonly sizeCustom = h('span.preview-size-custom', this.sizeWidth, '×', this.sizeHeight);
  private readonly sizeNote = h('span.preview-size-note', '');
  /** What is shown: an app folder ('' is the project root) or a page. */
  target: PreviewTarget | null = null;
  private readonly frameHost = h('div.preview-frame');
  private readonly stage = h('div.preview-stage');
  private viewportSize: (Viewport & { preset?: string }) | null = null;
  private scale = 1;
  /** When the frame last changed size: the page's resize handlers run a moment later. */
  private resizedAt = 0;

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
  private requests = new Map<number, (result: BridgeReply) => void>();
  private nextRequest = 1;
  /** Whether the frame shown now has said its bridge is there. */
  private bridgeReady = false;
  /** Checks, inspections, actions or screenshots running (edits wait to re-render until they are done). */
  private busy = 0;
  /** A page the frame asked to open while a request was running: opened once it has answered. */
  private pendingPage: string | null = null;
  private readonly banner = h('div.preview-problem');
  private readonly bannerText = h('span.preview-problem-text');
  private readonly fixButton = h('button.primary', { title: 'Send these errors to the agent and ask it to fix it' }, 'Fix with agent');
  /** Each page's localStorage, kept while bot.computer is open (the frame starts afresh on every render). */
  private readonly storage = new Map<string, Record<string, string>>();
  /** Set by the page: ask the agent to fix what is shown. Returns false when the agent is busy. */
  onFix: ((target: PreviewTarget, problems: Problem[]) => boolean) | null = null;

  constructor(private vfs: Vfs, private readonly projectId: () => string) {
    const reload = h('button', { title: 'Render it again', onclick: () => this.render() }, '↻ Reload');
    this.picker.addEventListener('change', () => {
      const [kind, ...rest] = this.picker.value.split(':');
      const value = rest.join(':');
      if (kind === 'page') this.setPage(value);
      else this.setApp(value);
    });
    this.sizePicker.append(
      h('option', { value: 'fit' }, 'Fit pane'),
      ...Object.entries(VIEWPORTS).map(([key, v]) => h('option', { value: key }, `${v.label} ${v.width}×${v.height}`)),
      h('option', { value: 'custom' }, 'Custom…'),
    );
    this.sizePicker.addEventListener('change', () => {
      const value = this.sizePicker.value;
      if (value === 'fit') this.setViewport(null);
      else if (value === 'custom') this.setViewport({ width: this.viewportSize?.width ?? 1024, height: this.viewportSize?.height ?? 768 });
      else this.setViewport(VIEWPORTS[value]);
    });
    const custom = () => {
      const width = Number(this.sizeWidth.value);
      const height = Number(this.sizeHeight.value);
      if (width >= MIN_VIEWPORT && height >= MIN_VIEWPORT) this.setViewport({ width, height });
    };
    this.sizeWidth.addEventListener('change', custom);
    this.sizeHeight.addEventListener('change', custom);
    this.banner.append(
      h('span.preview-problem-icon', '⚠'),
      this.bannerText,
      this.fixButton,
      h('button', { title: 'Hide until the next error', onclick: () => (this.banner.hidden = true) }, 'Dismiss'),
    );
    this.banner.hidden = true;
    this.fixButton.addEventListener('click', () => {
      const problems = this.problems.filter((p) => p.level === 'error');
      if (!problems.length || !this.onFix || !this.target) return;
      if (this.onFix(this.target, problems)) this.banner.hidden = true;
    });
    window.addEventListener('message', (event) => this.fromBridge(event));
    new ResizeObserver(() => this.layout()).observe(this.frameHost);
    this.frameHost.append(this.stage);
    this.element.append(h('div.pane-title', 'Preview ', this.picker, this.sizePicker, this.sizeCustom, this.sizeNote, this.status, reload), this.banner, this.frameHost);
    this.picker.hidden = true;
    this.sizeCustom.hidden = true;
    this.showMessage('Open a web page (an .html file) or a SoftN app (/softn new) to see it here.');
  }

  /** The app folder shown ('' when it is the project root, or when a page is shown). */
  get app(): string {
    return this.target?.kind === 'app' ? this.target.root : '';
  }

  /** The apps and pages of the project, for the picker; keeps the choice if it still exists. */
  refreshApps(): string[] {
    const apps = findApps(this.vfs);
    const pages = findPages(this.vfs);
    const exists = (t: PreviewTarget | null) => !!t && (t.kind === 'app' ? apps.includes(t.root) : pages.includes(t.path));
    if (!exists(this.target)) this.target = apps.length ? { kind: 'app', root: apps[0] } : pages.length ? { kind: 'page', path: pages[0] } : this.target?.kind === 'app' ? this.target : null;
    clear(this.picker);
    const chosen = this.target ? targetValue(this.target) : '';
    const option = (t: PreviewTarget, text: string) => h('option', { value: targetValue(t), selected: targetValue(t) === chosen }, text);
    if (apps.length) this.picker.append(h('optgroup', { label: 'SoftN apps' }, ...apps.map((root) => option({ kind: 'app', root }, root ? `${root}/` : '/ (project root)'))));
    if (pages.length) this.picker.append(h('optgroup', { label: 'Web pages' }, ...pages.map((path) => option({ kind: 'page', path }, `/${path}`))));
    this.picker.hidden = apps.length + pages.length < 2;
    return apps;
  }

  /** Show the app in `root`. */
  setApp(root: string): void {
    this.show({ kind: 'app', root });
  }

  /** Show the web page at `path`. */
  setPage(path: string): void {
    this.show({ kind: 'page', path });
  }

  show(target: PreviewTarget): void {
    this.target = target;
    this.refreshApps();
    this.stale = true;
    if (this.visible) this.render();
  }

  setVfs(vfs: Vfs): void {
    this.vfs = vfs;
    this.target = null;
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
    if (path === null || /(^|\/)manifest\.json$/i.test(path) || /\.html?$/i.test(path)) this.refreshApps();
    const target = this.target;
    if (path !== null && target) {
      const key = path.replace(/^\/+/, '');
      if (target.kind === 'app') {
        if (target.root && !key.startsWith(`${target.root}/`)) return;
        if (!/(^|\/)(manifest\.json|permission\.json)$|\.(ui|logic|py|xdb)$|(^|\/)assets\//i.test(key)) return;
      } else {
        const folder = target.path.includes('/') ? `${target.path.slice(0, target.path.lastIndexOf('/'))}/` : '';
        if ((folder && !key.startsWith(folder)) || key.split('/').some((s) => s.startsWith('.'))) return;
      }
    }
    this.stale = true;
    if (!this.visible) return;
    this.scheduleRender(700);
  }

  /** Re-render after edits settle, but not under a check, inspection, action or screenshot that is running. */
  private scheduleRender(ms: number): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      if (this.busy) this.scheduleRender(300);
      else if (this.stale) this.render();
    }, ms);
  }

  // --- the viewport --------------------------------------------------------------

  /** Show the page at a screen size (null: fill the pane). The page lays out at that size; the frame is scaled to fit. */
  setViewport(size: (Viewport & { preset?: string }) | null): ViewportInfo {
    if (size) {
      const width = Math.round(Math.min(MAX_VIEWPORT, Math.max(MIN_VIEWPORT, size.width)));
      const height = Math.round(Math.min(MAX_VIEWPORT, Math.max(MIN_VIEWPORT, size.height)));
      const preset = size.preset ?? Object.keys(VIEWPORTS).find((k) => VIEWPORTS[k].width === width && VIEWPORTS[k].height === height);
      this.viewportSize = { width, height, preset };
    } else {
      this.viewportSize = null;
    }
    this.sizePicker.value = !this.viewportSize ? 'fit' : (this.viewportSize.preset ?? 'custom');
    this.sizeCustom.hidden = this.sizePicker.value !== 'custom';
    if (this.viewportSize) {
      this.sizeWidth.value = String(this.viewportSize.width);
      this.sizeHeight.value = String(this.viewportSize.height);
    }
    this.layout();
    return this.viewport();
  }

  viewport(): ViewportInfo {
    if (this.viewportSize) return { ...this.viewportSize, fit: false, scale: this.scale };
    const rect = this.frameHost.getBoundingClientRect();
    return { width: Math.round(rect.width), height: Math.round(rect.height), fit: true, scale: 1 };
  }

  private layout(): void {
    const frame = this.frame;
    const size = this.viewportSize;
    this.resizedAt = Date.now();
    this.stage.classList.toggle('sized', !!size);
    if (!size) {
      this.scale = 1;
      this.stage.style.width = this.stage.style.height = '';
      if (frame) frame.style.width = frame.style.height = frame.style.transform = '';
      this.sizeNote.textContent = '';
      return;
    }
    const room = this.frameHost.getBoundingClientRect();
    const scale = room.width > 0 && room.height > 0 ? Math.min(1, (room.width - 16) / size.width, (room.height - 16) / size.height) : 1;
    this.scale = Math.max(0.05, scale);
    this.stage.style.width = `${Math.floor(size.width * this.scale)}px`;
    this.stage.style.height = `${Math.floor(size.height * this.scale)}px`;
    if (frame) {
      frame.style.width = `${size.width}px`;
      frame.style.height = `${size.height}px`;
      frame.style.transform = this.scale < 1 ? `scale(${this.scale})` : '';
    }
    this.sizeNote.textContent = this.scale < 0.995 ? `${Math.round(this.scale * 100)}%` : '';
  }

  // --- what the frame says ----------------------------------------------------------

  private showMessage(text: string): void {
    clear(this.stage);
    this.stage.append(h('div.empty', text));
  }

  private setStatus(text: string, kind: 'ok' | 'busy' | 'error' = 'busy'): void {
    this.status.textContent = text;
    this.status.dataset.kind = kind;
  }

  private fromBridge(event: MessageEvent): void {
    const data = event.data as { __botComputer?: boolean; type?: string; level?: string; message?: string; at?: number; id?: number; result?: BridgeReply; path?: unknown; store?: unknown; kind?: string };
    if (!this.frame || event.source !== this.frame.contentWindow || data?.__botComputer !== true) return;
    if (data.type === 'bot:bridge') {
      this.bridgeReady = true;
    } else if (data.type === 'bot:problem' && typeof data.message === 'string') {
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
    } else if (data.type === 'bot:navigate' && typeof data.path === 'string' && this.target?.kind === 'page') {
      // A link or location change to another page of the project.
      const path = data.path.replace(/^\/+/, '');
      const pages = findPages(this.vfs);
      const page = pages.includes(path) ? path : pages.find((p) => p === `${path}/index.html` || p === `${path}index.html`);
      // A link clicked by the agent: its request answers from this page first.
      if (page && this.requests.size) this.pendingPage = page;
      else if (page) this.setPage(page);
      else this.setStatus(`no page /${path} in the project`, 'error');
    } else if (data.type === 'bot:storage' && this.target?.kind === 'page' && data.store && typeof data.store === 'object') {
      this.storage.set(this.target.path, data.store as Record<string, string>);
    } else if (data.type === 'bot:dialog' && typeof data.message === 'string') {
      this.setStatus(`${data.kind ?? 'alert'}: ${data.message.slice(0, 120)}`, 'ok');
    }
  }

  private showProblems(): void {
    const errors = this.problems.filter((p) => p.level === 'error');
    if (!errors.length) {
      this.banner.hidden = true;
      return;
    }
    const first = errors[errors.length - 1].message.split('\n')[0];
    const what = this.target?.kind === 'page' ? 'page' : 'app';
    this.bannerText.textContent = errors.length > 1 ? `${errors.length} errors in the ${what}. Latest: ${first}` : `The ${what} reported an error: ${first}`;
    this.bannerText.title = errors.map((p) => p.message).join('\n\n');
    this.fixButton.hidden = !this.onFix;
    this.banner.hidden = false;
    this.setStatus(`error: ${first}`, 'error');
  }

  private request(message: Record<string, unknown>, timeoutMs = 20_000): Promise<BridgeReply> {
    const frame = this.frame?.contentWindow;
    if (!frame) return Promise.resolve({ ok: false, page: '', error: 'the preview is not showing anything' });
    const id = this.nextRequest++;
    return new Promise((resolve) => {
      const finish = (result: BridgeReply) => {
        clearTimeout(timer);
        this.requests.delete(id);
        resolve(result);
        if (!this.requests.size && this.pendingPage) {
          const page = this.pendingPage;
          this.pendingPage = null;
          setTimeout(() => this.setPage(page));
        }
      };
      const timer = setTimeout(() => {
        finish({
          ok: false,
          page: '',
          error: this.bridgeReady
            ? `the preview did not answer within ${Math.round(timeoutMs / 1000)} s (the page may be busy)`
            : 'the preview did not answer: bot.computer\'s bridge did not start in the preview frame (reload bot.computer; if it persists, run npm run fetch:softn -- --ensure)',
        });
      }, timeoutMs);
      this.requests.set(id, finish);
      frame.postMessage({ __botComputer: true, id, ...message }, '*');
    });
  }

  /** A request to the frame, asked again once if the frame was re-rendered under it. */
  private async requestAgain(message: Record<string, unknown>, timeoutMs?: number): Promise<BridgeReply> {
    const first = await this.request(message, timeoutMs);
    if (first.error !== RELOADED) return first;
    await this.ready();
    return this.request(message, timeoutMs);
  }

  // --- what the agent asks -------------------------------------------------------------

  /** Render `target` unless it is already live and current. */
  private async ready(target?: PreviewTarget): Promise<PreviewResult> {
    if ((target && !sameTarget(target, this.target)) || this.stale || !this.frame || this.current.at === 0) return this.checkNow(target);
    return this.current;
  }

  /** Describe what the app or page shows, as text. */
  inspect(target?: PreviewTarget): Promise<PageReport> {
    return this.exclusive(() => this.inspectNow(target));
  }

  private async inspectNow(target?: PreviewTarget): Promise<PageReport> {
    const rendered = await this.ready(target);
    if (!this.frame) return { ok: false, page: '', error: rendered.errors.join('; ') || 'nothing is rendered', problems: [] };
    const since = Date.now();
    const result = await this.requestAgain({ type: 'bot:inspect' });
    return { ...result, problems: this.problems.filter((p) => p.at >= since) };
  }

  /** Click, fill, choose and press keys in the app or page, then describe it. */
  act(target: PreviewTarget | undefined, actions: PreviewAction[]): Promise<PageReport> {
    return this.exclusive(() => this.actNow(target, actions));
  }

  private async actNow(target: PreviewTarget | undefined, actions: PreviewAction[]): Promise<PageReport> {
    const rendered = await this.ready(target);
    if (!this.frame) return { ok: false, page: '', error: rendered.errors.join('; ') || 'nothing is rendered', problems: [] };
    const since = Date.now();
    const result = await this.request({ type: 'bot:act', actions }, 60_000);
    if (result.error === RELOADED) return { ok: false, page: '', error: 'the page was re-rendered while the actions ran (its files changed); run them again', problems: [] };
    // Errors a click causes can arrive a moment after it.
    await new Promise((r) => setTimeout(r, 200));
    return { ...result, problems: this.problems.filter((p) => p.at >= since) };
  }

  /** A PNG of what the app or page shows: the viewport, or the whole page. */
  screenshot(target: PreviewTarget | undefined, options: { fullPage?: boolean } = {}): Promise<Shot & { problems: Problem[] }> {
    return this.exclusive(async () => {
      const rendered = await this.ready(target);
      if (!this.frame) throw new Error(rendered.errors.join('; ') || 'nothing is rendered');
      const box = this.frame.getBoundingClientRect();
      if (box.width < 1 || box.height < 1) throw new Error('the preview is not on screen (on a narrow screen, open the Preview tab), so it cannot be drawn');
      // A size just chosen: the page lays itself out again first.
      const wait = this.resizedAt + 500 - Date.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      const since = Date.now();
      const result = await this.requestAgain({ type: 'bot:screenshot', fullPage: options.fullPage === true }, 60_000);
      if (!result.ok || !result.shot) throw new Error(result.error || 'the preview could not take a screenshot');
      return { ...result.shot, png: new Uint8Array(result.shot.png), problems: this.problems.filter((p) => p.at >= since) };
    });
  }

  private settle(result: Omit<PreviewResult, 'at'>): void {
    this.current = { ...result, at: this.current.at };
    for (const w of this.waiters.splice(0)) w(this.current);
  }

  /**
   * One check, inspect, act or screenshot at a time: agents working in
   * parallel would otherwise switch the preview to their own app under each other.
   */
  private lock: Promise<unknown> = Promise.resolve();
  private exclusive<T>(fn: () => Promise<T>): Promise<T> {
    this.busy++;
    const run = this.lock.then(fn, fn).finally(() => this.busy--);
    this.lock = run.catch(() => {});
    return run;
  }

  /** Render now and wait for the outcome (for the agent's softn_check and page_check). */
  check(target?: PreviewTarget): Promise<PreviewResult> {
    return this.exclusive(() => this.checkNow(target));
  }

  private async checkNow(target?: PreviewTarget): Promise<PreviewResult> {
    if (target && !sameTarget(target, this.target)) {
      this.target = target;
      this.refreshApps();
    }
    if (this.target?.kind !== 'page' && !(await softnRuntimeAvailable())) return { ok: false, errors: ['the SoftN preview runtime is not installed (npm run fetch:softn); only the file checks ran'], at: Date.now() };
    const done = new Promise<PreviewResult>((resolve) => this.waiters.push(resolve));
    this.render();
    return done;
  }

  // --- rendering --------------------------------------------------------------------------

  /** A new frame for this render, sized to the viewport. */
  private newFrame(title: string, src: string, sandbox: string): HTMLIFrameElement {
    const frame = h('iframe', { title }) as HTMLIFrameElement;
    // Opaque origin: the page cannot reach bot.computer's storage, files or keys.
    frame.setAttribute('sandbox', sandbox);
    // Lets the page embed the frame under its own COEP even where the host
    // cannot add headers to the frame's response.
    frame.setAttribute('credentialless', '');
    frame.setAttribute('allow', '');
    frame.src = src;
    return frame;
  }

  private place(frame: HTMLIFrameElement): void {
    clear(this.stage);
    this.stage.append(frame);
    this.layout();
  }

  async render(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const generation = ++this.generation;
    this.stale = false;
    // Requests to the frame being replaced: answered now, rather than never.
    for (const finish of [...this.requests.values()]) finish({ ok: false, page: '', error: RELOADED });
    this.bridgeReady = false;
    this.current = { ok: false, errors: [], at: Date.now() };
    this.problems = [];
    this.collecting = null;
    this.banner.hidden = true;
    this.frame = null;
    if (this.settleTimer) clearTimeout(this.settleTimer);
    const target = this.target;
    if (target?.kind === 'page') return this.renderPage(target.path, generation);
    return this.renderApp(target?.root ?? '', generation);
  }

  private async renderPage(path: string, generation: number): Promise<void> {
    if (!this.vfs.exists(`/${path}`)) {
      this.setStatus('');
      this.showMessage(`/${path} is not in the project any more.`);
      this.settle({ ok: false, errors: [`/${path} does not exist`] });
      return;
    }
    this.setStatus('rendering…');
    let bytes: ArrayBuffer;
    let glue: string;
    try {
      [bytes, glue] = await Promise.all([zippBytes(), zippGlue()]);
    } catch (error) {
      this.setStatus('failed', 'error');
      this.settle({ ok: false, errors: [(error as Error).message] });
      return;
    }
    if (generation !== this.generation) return;
    // Forms stay allowed so that a page's submit handlers run; the frame keeps them from going anywhere.
    const frame = this.newFrame('Web page preview', `${import.meta.env.BASE_URL}webpage/index.html?v=${generation}`, 'allow-scripts allow-forms');
    this.frame = frame;
    const errors: string[] = [];
    const warnings: string[] = [];
    this.collecting = {
      errors,
      warnings,
      // A page's errors are gathered until it has loaded, then reported together.
      settle: () => {},
    };
    const onMessage = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.data?.type !== 'webpage:ready') return;
      window.removeEventListener('message', onMessage);
      const channel = new MessageChannel();
      channel.port1.onmessage = (e: MessageEvent) => {
        if (generation !== this.generation) return;
        const data = e.data as { type?: string; ok?: boolean; errors?: string[]; warnings?: string[] };
        if (data?.type !== 'loaded') return;
        const all = [...new Set([...(data.errors ?? []), ...errors])];
        const warned = [...new Set([...(data.warnings ?? []), ...warnings])].filter((w) => !all.includes(w));
        this.collecting = null;
        if (all.length) this.setStatus(`error: ${all[all.length - 1]}`, 'error');
        else this.setStatus(`live · ${new Date().toLocaleTimeString()}`, 'ok');
        this.settle({ ok: all.length === 0, errors: all, warnings: warned });
      };
      frame.contentWindow!.postMessage(
        { type: 'webpage:init', files: pageFiles(this.vfs, path), page: path, glue, wasm: bytes.slice(0), facade: FACADE, storage: this.storage.get(path) ?? {} },
        '*',
        [channel.port2],
      );
    };
    window.addEventListener('message', onMessage);
    setTimeout(() => {
      if (generation !== this.generation || this.status.dataset.kind !== 'busy') return;
      window.removeEventListener('message', onMessage);
      this.setStatus('the page did not load', 'error');
      this.settle({ ok: false, errors: ['the page preview did not finish loading within 20 s'] });
    }, 20_000);
    this.place(frame);
  }

  private async renderApp(root: string, generation: number): Promise<void> {
    if (!isSoftnApp(this.vfs, root)) {
      this.setStatus('');
      this.showMessage(this.target ? `${appLabel(root)} is not a SoftN app any more.` : 'Nothing to show yet. A web page is any .html file of the project; a SoftN app is a folder whose manifest.json has a "main" .ui page (/softn new starts one).');
      this.settle({ ok: false, errors: [`${appLabel(root)} is not a SoftN app: manifest.json with a "main" .ui page is missing`] });
      return;
    }
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

    const frame = this.newFrame('SoftN app preview', `${import.meta.env.BASE_URL}softn/index.html?v=${generation}`, 'allow-scripts allow-forms allow-modals');
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
    this.place(frame);
  }
}
