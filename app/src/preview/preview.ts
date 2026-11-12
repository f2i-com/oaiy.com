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
 * - A 3D model (a .glb or .gltf file) is drawn by OAIY's own viewer
 *   (public/modelview/, three.js), which the person turns with the mouse and
 *   which screenshots it from four sides for the agent.
 *
 * OAIY's bridge (scripts/softn-bridge/bot-bridge.js, in both frames)
 * reports what goes wrong while the page runs, and describes, operates and
 * screenshots it for the agent. The frame can be given a viewport size
 * (phone, tablet, …) for responsive design; it is then scaled to fit the pane,
 * while the page inside lays out at the size chosen.
 *
 * A web page's or a model's frame takes one per load, so a change reloads it
 * (after edits settle) and the page's in-memory state starts over. A SoftN app's
 * frame that says it takes a newer version in place (liveFrame.ts) is sent the
 * files instead and keeps running; one that does not is opened afresh, as before.
 */
import { zippBase } from '../sandbox/runner';
import type { Vfs } from '../vfs/vfs';
import { clear, h } from '../ui/dom';
import { icon } from '../ui/icons';
import { onTheme, theme } from '../ui/theme';
import { appFiles, appLabel, findApps, isSoftnApp, logicSyntax } from '../softn/softn';
import { findPages, pageFiles, pageLabel } from './page';
import { findModels, modelFiles, modelLabel } from './model';
import { KEPT_STATUS, KEPT_TITLE, LiveFrame, engineMismatch, fitNote, frameSays, isLoadSyntaxError, plainProblem, updateSaid } from './liveFrame';
import FACADE from './zippDom.guest.js?raw';

export type PreviewTarget = { kind: 'app'; root: string } | { kind: 'page'; path: string } | { kind: 'model'; path: string };

export interface PreviewResult {
  ok: boolean;
  errors: string[];
  /** console.warn from the app while it loaded (the runtime warns about missing handlers); for a page, files it could not load. */
  warnings?: string[];
  /** When the render these errors belong to started. */
  at: number;
  /** The newest version would not start, and the preview still shows the last one that worked (what inspect, act and screenshot then see). */
  kept?: boolean;
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
  /**
   * Problems the app reported since it last loaded: those while this ran, and those before it (`since`), which a
   * timer that failed once and stopped leaves behind with the app still broken.
   */
  problems: Problem[];
  /** When this began: the problems before it were raised earlier, the app as it is still loaded. */
  since?: number;
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
  /** For a 3D model: what it is made of and how big it is, as text. */
  about?: string;
}

/** A 3D model's screenshot from one angle, in degrees: yaw 0 looks at its front (+Z), 90 at its right side (+X); pitch 90 looks down. */
export interface ModelView {
  yaw: number;
  pitch: number;
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

export type PreviewAction =
  | { click: string; nth?: number; times?: number }
  | { fill: string; value: string; nth?: number }
  | { select: string; value: string | string[]; nth?: number }
  | { key: string; times?: number; hold?: number }
  | { wait: number };

let engineBytes: Promise<ArrayBuffer> | null = null;
function zippBytes(): Promise<ArrayBuffer> {
  engineBytes ??= fetch(`${zippBase()}zipp_wasm_bg.wasm`).then((r) => {
    if (!r.ok) throw new Error(`the Zipp engine could not be loaded (${r.status})`);
    return r.arrayBuffer();
  });
  engineBytes.catch(() => (engineBytes = null));
  return engineBytes;
}

let softnEngine: Promise<ArrayBuffer> | null = null;
/**
 * The engine a SoftN app's frame is given: the Agent's own. SoftN's runtime takes the module from its parent and
 * binds it with the bindings of the ZIPP release it was built with, and another release's module does not fit them
 * (the frame fails with "Import … requires a callable"). The runtime's manifest names its engine by its digest, and
 * from SoftN v0.0.35 the runtime carries no copy of it: where that is not the Agent's engine, nothing here can run
 * the app, and the person is told which two releases to bring together (engineMismatch) rather than shown a frame
 * that fails.
 */
function softnZippBytes(): Promise<ArrayBuffer> {
  softnEngine ??= (async () => {
    const json = (url: string) =>
      fetch(url, { cache: 'no-store' })
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null);
    const [manifest, own] = await Promise.all([json(`${import.meta.env.BASE_URL}softn/runtime-manifest.json`), json(`${zippBase()}SOURCE.json`)]);
    const mismatch = engineMismatch(manifest, own);
    if (mismatch) throw new Error(mismatch);
    return zippBytes();
  })();
  softnEngine.catch(() => (softnEngine = null));
  return softnEngine;
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
/** How long a render has to report an error before it counts as clean, in a frame that does not say when the app is drawn. */
const SETTLE_MS = 2500;
/** After the frame says the app is drawn, or answers an update: errors the drawing raised come another way (the bridge), a moment behind. */
const DRAWN_GRACE_MS = 300;
/** How long a frame that says when the app is drawn has to say so (or to report an error). */
const OPEN_DEADLINE_MS = 20_000;
/** A request's answer when the frame was re-rendered under it. */
const RELOADED = 'the preview was re-rendered while this ran';

export const targetLabel = (target: PreviewTarget): string => (target.kind === 'app' ? appLabel(target.root) : target.kind === 'model' ? modelLabel(target.path) : pageLabel(target.path));
const sameTarget = (a: PreviewTarget | null, b: PreviewTarget | null): boolean =>
  !!a && !!b && a.kind === b.kind && (a.kind === 'app' ? a.root === (b as { root: string }).root : a.path === (b as { path: string }).path);
const targetValue = (t: PreviewTarget): string => (t.kind === 'app' ? `app:${t.root}` : `${t.kind}:${t.path}`);
/** Where an app folder's manifest is, in the project. */
const manifestPath = (root: string): string => `/${root ? `${root}/` : ''}manifest.json`;

type BridgeReply = Omit<PageReport, 'problems'> & { shot?: Omit<Shot, 'png'> & { png: ArrayBuffer } };

export class Preview {
  readonly element = h('section.preview');
  private readonly status = h('span.preview-status', '');
  private readonly picker = h('select.preview-app', { title: 'What to show: a SoftN app, a web page or a 3D model of the project' });
  private readonly sizePicker = h('select.preview-size', { title: 'The size of the screen the page is shown on' });
  private readonly sizeWidth = h('input.preview-size-input', { type: 'number', min: String(MIN_VIEWPORT), max: String(MAX_VIEWPORT), title: 'Width in CSS pixels' }) as HTMLInputElement;
  private readonly sizeHeight = h('input.preview-size-input', { type: 'number', min: String(MIN_VIEWPORT), max: String(MAX_VIEWPORT), title: 'Height in CSS pixels' }) as HTMLInputElement;
  private readonly sizeCustom = h('span.preview-size-custom', this.sizeWidth, '×', this.sizeHeight);
  /** Beside the size picker: how far a chosen size is shrunk; or, under "Fit pane", that the pane is phone-sized, with the Laptop size offered. */
  private readonly sizeNoteText = h('span.preview-size-note', '');
  private readonly sizeOffer = h('button.preview-size-offer', { type: 'button', title: `See the app at a laptop's size (${VIEWPORTS.laptop.width}×${VIEWPORTS.laptop.height}), scaled to fit the pane`, onclick: () => this.setViewport(VIEWPORTS.laptop) }, 'Laptop') as HTMLButtonElement;
  private readonly sizeNote = h('span.preview-size-said', this.sizeNoteText, this.sizeOffer);
  /** What is shown: an app folder ('' is the project root), a page or a 3D model. */
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
  /** The SoftN app's frame shown now, with its port: a newer version of the same app is sent to it where it takes one (liveFrame.ts). */
  private link: LiveFrame | null = null;
  /** The newest version would not start and the frame kept the last one that worked: the banner and the status say so. */
  private kept = false;
  /** The logic files the last syntax check found broken: the banner names the file. */
  private syntaxFiles: string[] = [];
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
  private readonly banner = h('div.preview-problem', { role: 'alert' });
  /** Up to two lines: what is wrong, the file first (the whole of it in the tooltip). */
  private readonly bannerText = h('span.preview-problem-text');
  private readonly fixButton = h('button.primary', { title: 'Send these errors to the agent and ask it to fix it' }, 'Fix with agent');
  /** Each page's localStorage, kept while OAIY is open (the frame starts afresh on every render). */
  private readonly storage = new Map<string, Record<string, string>>();
  /** Set by the page: ask the agent to fix what is shown. Returns false when the agent is busy. */
  onFix: ((target: PreviewTarget, problems: Problem[]) => boolean) | null = null;

  /** Shows the preview over the whole screen (or OAIY's whole window, where the screen cannot be had), and back. */
  private readonly fullLabel = h('span.label', 'Full screen');
  private readonly fullButton = h('button.preview-full', { title: 'Show the app or page over the whole screen (Esc to leave)', 'aria-label': 'Full screen', onclick: () => this.setFull(!this.full) }, icon('expand'), this.fullLabel);

  /** Whether the preview fills the screen ([`Preview::setFull`]). */
  get full(): boolean {
    return this.element.classList.contains('full');
  }

  /**
   * The preview over the whole screen, or back in its pane. The screen's own full-screen mode where the window may
   * take it; OAIY's whole window otherwise (the class alone does that). Esc leaves either: the screen's by itself,
   * the window's by the key here (or in the frame, which cannot be heard here: the button then).
   */
  setFull(on: boolean): void {
    if (on === this.full) return;
    this.element.classList.toggle('full', on);
    this.fullLabel.textContent = on ? 'Leave full screen' : 'Full screen';
    this.fullButton.replaceChildren(icon(on ? 'x' : 'expand'), this.fullLabel);
    this.fullButton.setAttribute('aria-label', this.fullLabel.textContent);
    this.fullButton.title = on ? 'Back to the pane (Esc)' : 'Show the app or page over the whole screen (Esc to leave)';
    if (on) {
      void this.element.requestFullscreen?.().catch(() => {
        /* the window stays as it is: the preview fills it */
      });
    } else if (document.fullscreenElement === this.element) {
      void document.exitFullscreen().catch(() => {});
    }
    this.layout();
  }

  constructor(private vfs: Vfs, private readonly projectId: () => string) {
    // (Reload starts the app or page over in a new frame, also where a change would have been taken in place)
    const reload = h('button.preview-reload', { title: 'Start it again from the beginning', 'aria-label': 'Reload', onclick: () => this.render(true) }, icon('refresh'), h('span.label', 'Reload'));
    // The app is told OAIY's own theme when it opens (not the system's, which can differ): a change of theme opens it again.
    onTheme(() => {
      if (this.target?.kind !== 'app' || !this.frame) return;
      this.link?.close();
      this.link = null;
      this.stale = true;
      if (this.visible) this.scheduleRender(50);
    });
    // (the screen's full-screen mode left by Esc or otherwise: back in the pane; Esc in OAIY's window leaves its own)
    document.addEventListener('fullscreenchange', () => {
      if (!document.fullscreenElement && this.full) this.setFull(false);
    });
    window.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.full && !document.fullscreenElement) this.setFull(false);
    });
    this.picker.addEventListener('change', () => {
      const [kind, ...rest] = this.picker.value.split(':');
      const value = rest.join(':');
      if (kind === 'page') this.setPage(value);
      else if (kind === 'model') this.setModel(value);
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
      h('span.preview-problem-icon', { 'aria-hidden': 'true' }, icon('alert')),
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
    // (the status after the size picker, then Reload and Full screen at the right; on a narrow pane the two are their icons)
    this.element.append(h('div.pane-title', h('span.preview-kicker', 'Preview'), this.picker, this.sizePicker, this.sizeCustom, this.sizeNote, this.status, reload, this.fullButton), this.banner, this.frameHost);
    this.picker.hidden = true;
    this.sizeCustom.hidden = true;
    this.sizeNote.hidden = true;
    this.showMessage('Open a web page (an .html file), a 3D model (a .glb file) or a SoftN app (/softn new) to see it here.');
  }

  /** The app folder shown ('' when it is the project root, or when a page is shown). */
  get app(): string {
    return this.target?.kind === 'app' ? this.target.root : '';
  }

  /** The apps, pages and 3D models of the project, for the picker; keeps the choice if it still exists. */
  refreshApps(): string[] {
    const apps = findApps(this.vfs);
    const pages = findPages(this.vfs);
    const models = findModels(this.vfs);
    const exists = (t: PreviewTarget | null) => !!t && (t.kind === 'app' ? apps.includes(t.root) : t.kind === 'model' ? models.includes(t.path) : pages.includes(t.path));
    // An app open in a frame that takes updates stays the target while its manifest is there but unreadable (half
    // written, say): the frame keeps the last working version and says what is wrong, rather than the preview
    // turning to some other page of the project.
    const held = this.target?.kind === 'app' && !!this.link?.takes(this.target.root) && this.vfs.exists(manifestPath(this.target.root));
    if (!exists(this.target) && !held)
      this.target = apps.length ? { kind: 'app', root: apps[0] } : pages.length ? { kind: 'page', path: pages[0] } : models.length ? { kind: 'model', path: models[0] } : this.target?.kind === 'app' ? this.target : null;
    clear(this.picker);
    const chosen = this.target ? targetValue(this.target) : '';
    const option = (t: PreviewTarget, text: string) => h('option', { value: targetValue(t), selected: targetValue(t) === chosen }, text);
    if (apps.length) this.picker.append(h('optgroup', { label: 'SoftN apps' }, ...apps.map((root) => option({ kind: 'app', root }, root ? `${root}/` : '/ (project root)'))));
    if (pages.length) this.picker.append(h('optgroup', { label: 'Web pages' }, ...pages.map((path) => option({ kind: 'page', path }, `/${path}`))));
    if (models.length) this.picker.append(h('optgroup', { label: '3D models' }, ...models.map((path) => option({ kind: 'model', path }, `/${path}`))));
    this.picker.hidden = apps.length + pages.length + models.length < 2;
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

  /** Show the 3D model at `path`. */
  setModel(path: string): void {
    this.show({ kind: 'model', path });
  }

  show(target: PreviewTarget): void {
    this.target = target;
    this.refreshApps();
    this.stale = true;
    if (this.visible) this.render();
  }

  setVfs(vfs: Vfs): void {
    this.vfs = vfs;
    // Another project: its app is another app (its own data), whatever its folder is called, so never an update of the one open.
    this.link?.close();
    this.link = null;
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
    if (path === null || /(^|\/)manifest\.json$/i.test(path) || /\.(html?|glb|gltf)$/i.test(path)) this.refreshApps();
    const target = this.target;
    if (path !== null && target) {
      const key = path.replace(/^\/+/, '');
      if (target.kind === 'app') {
        if (target.root && !key.startsWith(`${target.root}/`)) return;
        if (!/(^|\/)(manifest\.json|permission\.json)$|\.(ui|html?|logic|js|py|xdb|css)$|(^|\/)assets\//i.test(key)) return;
      } else if (target.kind === 'model') {
        // A .glb is all there is to it; a .gltf also reads its folder.
        const folder = target.path.includes('/') ? `${target.path.slice(0, target.path.lastIndexOf('/'))}/` : '';
        if (key !== target.path && !(/\.gltf$/i.test(target.path) && (!folder || key.startsWith(folder)))) return;
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
      // A SoftN app in a pane this narrow is in its phone layout (a Sidebar is a Menu button, a wide Table is cards):
      // said here, with the Laptop size a press away, so nobody takes the phone layout for the app's only one.
      const note = this.target?.kind === 'app' && frame ? fitNote(this.frameHost.getBoundingClientRect().width) : '';
      this.sizeNoteText.textContent = note;
      this.sizeNoteText.title = note ? 'The preview pane is as narrow as a phone, so the app shows its phone layout: a side menu becomes a Menu button, and a wide table becomes cards. Choose Laptop to see the wide layout.' : '';
      this.sizeOffer.hidden = !note;
      this.sizeNote.hidden = !note;
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
    const shrunk = this.scale < 0.995 ? `${Math.round(this.scale * 100)}%` : '';
    this.sizeNoteText.textContent = shrunk;
    this.sizeNoteText.title = shrunk ? `Shown at ${shrunk} of its size, to fit the pane` : '';
    this.sizeOffer.hidden = true;
    this.sizeNote.hidden = !shrunk;
  }

  // --- what the frame says ----------------------------------------------------------

  private showMessage(text: string): void {
    clear(this.stage);
    this.stage.append(h('div.empty', text));
  }

  private setStatus(text: string, kind: 'ok' | 'busy' | 'error' = 'busy', title = ''): void {
    this.status.textContent = text;
    this.status.dataset.kind = kind;
    this.status.title = title;
  }

  /**
   * An error as the status says it: the word alone, with what went wrong in its tooltip. The message itself is in
   * the banner (an app's own errors) or, where nothing could be drawn, in the pane.
   */
  private failStatus(message: string): void {
    this.setStatus('error', 'error', message);
  }

  private fromBridge(event: MessageEvent): void {
    const data = event.data as { __botComputer?: boolean; type?: string; level?: string; message?: string; at?: number; id?: number; result?: BridgeReply; path?: unknown; store?: unknown; kind?: string };
    if (!this.frame || event.source !== this.frame.contentWindow || data?.__botComputer !== true) return;
    if (data.type === 'bot:bridge') {
      this.bridgeReady = true;
    } else if (data.type === 'bot:problem' && typeof data.message === 'string') {
      const problem: Problem = { level: data.level === 'warning' ? 'warning' : 'error', message: data.message.slice(0, 2000), at: Date.now() };
      // The same failure the frame said on its port a moment ago (fromApp): one problem, in the console's fuller words.
      const same = problem.level === 'error' ? this.problems.find((p) => this.fromPort.has(p) && problem.message.includes(p.message)) : undefined;
      if (same) {
        this.fromPort.delete(same);
        same.message = problem.message;
        this.showProblems();
        return;
      }
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

  /** Problems the frame said on its port (an error while the app runs), which its console may say too. */
  private readonly fromPort = new WeakSet<Problem>();

  /**
   * Something in the running app failed, as its frame says on the port (`error` with `phase: 'runtime'`): the app is
   * still drawn, so this is a problem of a running app, in the banner with "Fix with agent", like the ones the
   * bridge hears from the console.
   */
  private fromApp(reason: string): void {
    if (!this.problems.some((p) => p.level === 'error' && p.message.includes(reason))) {
      const problem: Problem = { level: 'error', message: reason.slice(0, 2000), at: Date.now() };
      this.fromPort.add(problem);
      this.problems.push(problem);
      if (this.problems.length > 100) this.problems.shift();
      if (this.collecting) {
        this.collecting.errors.push(problem.message);
        this.collecting.settle();
      }
    }
    this.showProblems();
  }

  private showProblems(): void {
    const errors = this.problems.filter((p) => p.level === 'error');
    if (!errors.length) {
      this.banner.hidden = true;
      return;
    }
    // The banner carries the message, on up to two lines: the file and the part a person can act on first. (It was
    // one line that ended before that part, behind "[SoftN] Error loading script: SyntaxError:"; the status beside
    // the size picker repeated it and was cut shorter still: now it says "error" and no more.)
    // (while the last working version is kept, why the newest one does not start leads, whatever the running one says since)
    const latest = ((this.kept && errors.find((p) => this.keptProblems.has(p))) || errors[errors.length - 1]).message;
    const file = this.syntaxFiles.length === 1 && (this.kept || isLoadSyntaxError(latest)) ? this.syntaxFiles[0] : undefined;
    const said = plainProblem(latest, file);
    const more = errors.length > 1 && !this.kept ? ` (the last of ${errors.length} errors)` : '';
    this.bannerText.textContent = `${this.kept ? 'Kept the last working version. ' : ''}${said}${more}`;
    this.bannerText.title = errors.map((p) => p.message).join('\n\n');
    this.fixButton.hidden = !this.onFix;
    this.banner.hidden = false;
    this.setStatus(this.kept ? KEPT_STATUS : 'error', 'error', this.kept ? KEPT_TITLE : errors.length > 1 ? `${errors.length} errors: the banner below says the latest` : 'The banner below says what is wrong');
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
            : 'the preview did not answer: OAIY\'s bridge did not start in the preview frame (reload the page; if it persists, run npm run fetch:softn -- --ensure)',
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
    return { ...result, problems: [...this.problems], since };
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
    return { ...result, problems: [...this.problems], since };
  }

  /** A PNG of what the app or page shows: the viewport, or the whole page; a 3D model from four sides, or from `view`. */
  screenshot(target: PreviewTarget | undefined, options: { fullPage?: boolean; view?: ModelView } = {}): Promise<Shot & { problems: Problem[]; since: number }> {
    return this.exclusive(async () => {
      const rendered = await this.ready(target);
      if (!this.frame) throw new Error(rendered.errors.join('; ') || 'nothing is rendered');
      // A model that could not be read has nothing to draw: why is the answer.
      if (this.target?.kind === 'model' && !rendered.ok) throw new Error(rendered.errors.join('; ') || 'the model could not be shown');
      const box = this.frame.getBoundingClientRect();
      if (box.width < 1 || box.height < 1) throw new Error('the preview is not on screen (on a narrow screen, open the Preview tab), so it cannot be drawn');
      // A size just chosen: the page lays itself out again first.
      const wait = this.resizedAt + 500 - Date.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      const since = Date.now();
      const result = await this.requestAgain({ type: 'bot:screenshot', fullPage: options.fullPage === true, view: options.view }, 60_000);
      if (!result.ok || !result.shot) throw new Error(result.error || 'the preview could not take a screenshot');
      return { ...result.shot, png: new Uint8Array(result.shot.png), problems: [...this.problems], since };
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
    if (this.target?.kind === 'app' && !(await softnRuntimeAvailable())) return { ok: false, errors: ['the SoftN preview runtime is not installed (npm run fetch:softn); only the file checks ran'], at: Date.now() };
    const done = new Promise<PreviewResult>((resolve) => this.waiters.push(resolve));
    this.render();
    return done;
  }

  // --- rendering --------------------------------------------------------------------------

  /** A new frame for this render, sized to the viewport. */
  private newFrame(title: string, src: string, sandbox: string): HTMLIFrameElement {
    const frame = h('iframe', { title }) as HTMLIFrameElement;
    // Opaque origin: the page cannot reach OAIY's storage, files or keys.
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

  /**
   * Show the target as its files are now. A SoftN app whose frame takes a newer version in place is sent the files,
   * and goes on running. Everything else gets a new frame: a page, a model, another app, a frame that never said it
   * takes updates, and `fresh` (Reload, and a change of theme, which an app is told only as it opens).
   */
  async render(fresh = false): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const generation = ++this.generation;
    this.stale = false;
    if (this.settleTimer) clearTimeout(this.settleTimer);
    const target = this.target;
    const link = this.link;
    // (a manifest that is there and cannot be read, half-written say, goes to the frame too: it answers `kept`, and
    // the last working app stays on screen with the problem said, where a new frame would show only "not an app".
    // An app whose manifest is gone is not an app any more, and is said so as before.)
    if (!fresh && link && this.frame && target?.kind === 'app' && link.takes(target.root) && (isSoftnApp(this.vfs, target.root) || this.vfs.exists(manifestPath(target.root)))) return this.updateApp(target.root, generation, link);
    this.leaveFrame();
    if (target?.kind === 'page') return this.renderPage(target.path, generation);
    if (target?.kind === 'model') return this.renderModel(target.path, generation);
    return this.renderApp(target?.root ?? '', generation);
  }

  /** The frame shown is being replaced: what waited on it is answered, and what it reported is put away with it. */
  private leaveFrame(): void {
    // Requests to the frame being replaced: answered now, rather than never.
    for (const finish of [...this.requests.values()]) finish({ ok: false, page: '', error: RELOADED });
    this.link?.close();
    this.link = null;
    this.kept = false;
    this.syntaxFiles = [];
    this.bridgeReady = false;
    this.current = { ok: false, errors: [], at: Date.now() };
    this.problems = [];
    this.collecting = null;
    this.banner.hidden = true;
    this.frame = null;
  }

  /**
   * Errors found here rather than reported from inside the frame (a logic file that does not compile, a page's file
   * that could not be loaded): in the banner with the others, so they are read whole and "Fix with agent" sends them.
   */
  private report(messages: string[]): void {
    const line = (text: string) => text.split('\n')[0];
    for (const message of messages) {
      // (not a second time what the frame's bridge said already, worded a little differently)
      const said = this.problems.some((p) => p.level === 'error' && (line(p.message).includes(line(message)) || line(message).includes(line(p.message))));
      if (!said) this.problems.push({ level: 'error', message: message.slice(0, 2000), at: Date.now() });
    }
    this.showProblems();
  }

  /** The app's files as the frame takes them: its text files by path, and the rest (pictures, sounds, fonts) as base64. */
  private appSource(root: string): { client: Record<string, string>; assets: Record<string, string> } {
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
    return { client, assets };
  }

  /** The problems that are a version the frame would not start (not something the running app raised). */
  private readonly keptProblems = new WeakSet<Problem>();

  /**
   * A newer version of the app the frame has open, sent to it (the frame said `update: 1` when it opened). The answer
   * says what became of it (liveFrame.ts): drawn again where it stood, started again, or kept as it was because the
   * newest version does not start, in which case the last working app stays on screen and the banner says what is
   * wrong. `reopen`, or no answer within the deadline, opens a new frame exactly as a frame without updates gets.
   */
  private async updateApp(root: string, generation: number, link: LiveFrame): Promise<void> {
    this.current = { ok: false, errors: [], at: Date.now() };
    const errors: string[] = [];
    const warnings: string[] = [];
    // What the app reports from here on is the new version's; what it reported before is kept only where the
    // version that reported it is still the one running (see below).
    const earlier = this.problems.filter((p) => !this.keptProblems.has(p));
    this.problems = [];
    this.syntaxFiles = [];
    this.collecting = {
      errors,
      warnings,
      settle: () => {
        if (generation !== this.generation) return;
        this.failStatus(errors[errors.length - 1]);
        this.settle({ ok: false, errors: [...errors], warnings: [...warnings] });
      },
    };
    this.setStatus('updating…');
    const syntax = logicSyntax(this.vfs, root).catch(() => []);
    const { client, assets } = this.appSource(root);
    const answer = await link.update(client, assets);
    if (generation !== this.generation) return;
    if (!answer || answer.outcome === 'reopen') {
      this.leaveFrame();
      return this.renderApp(root, generation);
    }
    const said = updateSaid(answer, new Date().toLocaleTimeString());
    if (answer.outcome === 'kept') {
      // The version on screen is the one from before, still running: what it reported still holds, and the newest
      // version's problem leads.
      const problem: Problem = { level: 'error', message: answer.problem ?? 'The newest version of the app would not start.', at: Date.now() };
      this.keptProblems.add(problem);
      this.kept = true;
      this.problems = [...earlier, ...this.problems, problem];
      errors.push(problem.message);
      this.showProblems();
      this.settle({ ok: false, errors: [...errors], warnings: [...warnings], kept: true });
      // (the file the banner leads with, once the logic has been compiled here)
      void syntax.then((found) => {
        if (generation !== this.generation) return;
        this.syntaxFiles = found.map((f) => f.file);
        if (!this.banner.hidden) this.showProblems();
      });
      return;
    }
    this.kept = false;
    // Nothing changed: the app is as it was, with whatever it had reported.
    if (answer.outcome === 'unchanged') this.problems = [...earlier, ...this.problems];
    // Errors the new version raised as it was drawn come from the bridge, a moment behind the answer.
    const found = (await Promise.all([syntax, new Promise((r) => setTimeout(r, DRAWN_GRACE_MS))]))[0];
    if (generation !== this.generation) return;
    this.syntaxFiles = found.map((f) => f.file);
    if (errors.length) {
      if (!this.banner.hidden) this.showProblems();
      return;
    }
    const broken = found.map((f) => `${f.file}: ${f.message}`);
    const standing = this.problems.filter((p) => p.level === 'error').map((p) => p.message);
    if (broken.length || standing.length) {
      errors.push(...broken, ...standing);
      this.report(broken);
      this.settle({ ok: false, errors: [...errors], warnings: [...warnings] });
      return;
    }
    this.banner.hidden = true;
    this.setStatus(said.status, said.kind, said.title);
    this.settle({ ok: true, errors: [], warnings: [...warnings] });
  }

  private renderModel(path: string, generation: number): void {
    if (!this.vfs.exists(`/${path}`)) {
      this.setStatus('');
      this.showMessage(`/${path} is not in the project any more.`);
      this.settle({ ok: false, errors: [`/${path} does not exist`] });
      return;
    }
    this.setStatus('loading…');
    // No forms, no pop-ups: the viewer only draws the model it is sent.
    const frame = this.newFrame('3D model preview', `${import.meta.env.BASE_URL}modelview/index.html?v=${generation}`, 'allow-scripts');
    this.frame = frame;
    const errors: string[] = [];
    this.collecting = { errors, warnings: [], settle: () => {} };
    const onMessage = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.data?.type !== 'modelview:ready') return;
      window.removeEventListener('message', onMessage);
      const channel = new MessageChannel();
      channel.port1.onmessage = (e: MessageEvent) => {
        if (generation !== this.generation) return;
        const data = e.data as { type?: string; ok?: boolean; errors?: string[] };
        if (data?.type !== 'loaded') return;
        const all = [...new Set([...(data.errors ?? []), ...errors])];
        this.collecting = null;
        if (all.length) this.report(all);
        else this.setStatus(`shown · ${new Date().toLocaleTimeString()}`, 'ok');
        this.settle({ ok: all.length === 0, errors: all });
      };
      const files = modelFiles(this.vfs, path);
      // Copies, so the project's own bytes stay put when they are handed over.
      const sent: Record<string, Uint8Array> = {};
      for (const [name, data] of Object.entries(files)) sent[name] = data.slice();
      frame.contentWindow!.postMessage({ type: 'modelview:init', path, files: sent }, '*', [channel.port2, ...Object.values(sent).map((d) => d.buffer as ArrayBuffer)]);
    };
    window.addEventListener('message', onMessage);
    setTimeout(() => {
      // (taken off whatever came of it: a render that replaced this frame before it was ready left it listening)
      window.removeEventListener('message', onMessage);
      if (generation !== this.generation || this.status.dataset.kind !== 'busy') return;
      this.setStatus('the model did not load', 'error');
      this.settle({ ok: false, errors: ['the 3D model preview did not finish loading within 30 s'] });
    }, 30_000);
    this.place(frame);
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
        if (all.length) this.report(all);
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
      window.removeEventListener('message', onMessage);
      if (generation !== this.generation || this.status.dataset.kind !== 'busy') return;
      this.setStatus('the page did not load', 'error');
      this.settle({ ok: false, errors: ['the page preview did not finish loading within 20 s'] });
    }, 20_000);
    this.place(frame);
  }

  private async renderApp(root: string, generation: number): Promise<void> {
    if (!isSoftnApp(this.vfs, root)) {
      this.setStatus('');
      this.showMessage(this.target ? `${appLabel(root)} is not a SoftN app any more.` : 'Nothing to show yet. A web page is any .html file of the project; a SoftN app is a folder whose manifest.json names its main page, such as index.html, as "main" (/softn new starts one).');
      this.settle({ ok: false, errors: [`${appLabel(root)} is not a SoftN app: a manifest.json whose "main" names its main page (such as index.html) is missing`] });
      return;
    }
    if (!(await softnRuntimeAvailable())) {
      this.setStatus('');
      this.showMessage('The SoftN preview runtime is not installed. Run `npm run fetch:softn` and reload the page. You can still build, check and export .softn apps.');
      this.settle({ ok: false, errors: ['the SoftN preview runtime is not installed'] });
      return;
    }
    this.setStatus('rendering…');
    // The runtime shows a logic syntax error inside the frame without
    // reporting it, so the logic is also compiled here, on Zipp.
    const syntax = logicSyntax(this.vfs, root).catch(() => []);
    const { client, assets } = this.appSource(root);
    let bytes: ArrayBuffer;
    try {
      bytes = await softnZippBytes();
    } catch (error) {
      // Nothing can be drawn: why, and what to do about it, is said where the app would be.
      const message = (error as Error).message;
      this.failStatus(message);
      this.showMessage(message);
      this.settle({ ok: false, errors: [message] });
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
        this.failStatus(errors[errors.length - 1]);
        this.settle({ ok: false, errors: [...errors], warnings: [...warnings] });
      },
    };
    // (the file the banner leads with, when the logic does not compile)
    void syntax.then((found) => {
      if (generation !== this.generation) return;
      this.syntaxFiles = found.map((f) => f.file);
      if (!this.banner.hidden) this.showProblems();
    });
    const onMessage = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.data?.type !== 'formlogic:ready') return;
      window.removeEventListener('message', onMessage);
      const channel = new MessageChannel();
      // What this frame does past drawing the app once: saying when it is drawn, taking a newer version in place.
      // (with the media `init` carries: an update leaves its media out while it is the same)
      const link = new LiveFrame(root, channel.port1, frameSays(event.data), assets);
      this.link = link;
      // The app's first page is drawn: live, unless the logic does not compile or the drawing raised an error.
      const opened = async () => {
        const broken = (await syntax).map((f) => `${f.file}: ${f.message}`);
        if (generation !== this.generation || errors.length) return;
        if (broken.length) {
          errors.push(...broken);
          this.report(broken);
          this.settle({ ok: false, errors: [...errors], warnings: [...warnings] });
          return;
        }
        this.setStatus(`live · ${new Date().toLocaleTimeString()}`, 'ok');
        this.settle({ ok: true, errors: [], warnings: [...warnings] });
      };
      channel.port1.onmessage = (e: MessageEvent) => {
        // (the port outlives the render that opened it where the frame takes updates: its messages are this frame's
        // for as long as it is the one shown)
        if (this.link !== link) return;
        const data = e.data as { type?: string; reason?: string; phase?: string; id?: unknown };
        if (link.take(data)) return;
        if (data?.type === 'shown') {
          if (link.drawn) return;
          link.drawn = true;
          if (this.settleTimer) clearTimeout(this.settleTimer);
          this.settleTimer = setTimeout(() => void opened(), DRAWN_GRACE_MS);
        } else if (data?.type === 'error' && data.phase === 'runtime') {
          // The app is drawn and something in it failed: a problem of a running app (the banner, "Fix with agent"),
          // not an app that could not be opened.
          link.drawn = true;
          this.fromApp(data.reason || 'something in the app failed');
        } else if (data?.type === 'error') {
          // It did not open: nothing of it is shown, and there is nothing in this frame to update.
          link.failed = true;
          const reason = data.reason || 'the app failed to load';
          if (generation !== this.generation) return;
          errors.push(reason);
          this.report([reason]);
          this.settle({ ok: false, errors: [...errors], warnings: [...warnings] });
        } else if (data?.type === 'call') {
          channel.port1.postMessage({ id: data.id, result: { error: 'This preview has no backend: server/ actions run only when the app is deployed.' } });
        }
      };
      frame.contentWindow!.postMessage(
        // dark: OAIY's own theme (the dashboard's choice in its window), which the system's setting need not be.
        { type: 'formlogic:init', client, assets, appId: `bot.computer-${this.projectId()}-${root || 'root'}`, dark: theme() === 'dark', engine: 'zipp-web-python', zippWasm: bytes.slice(0) },
        '*',
        [channel.port2],
      );
      // A frame that says when the app is drawn is believed, and given a while to say it; one that does not is
      // counted as drawn once it has had time to report an error.
      this.settleTimer = link.says.shown
        ? setTimeout(() => {
            if (generation !== this.generation || link.drawn || errors.length) return;
            link.failed = true;
            const message = `the app did not finish opening within ${OPEN_DEADLINE_MS / 1000} s`;
            errors.push(message);
            this.failStatus(message);
            this.settle({ ok: false, errors: [...errors], warnings: [...warnings] });
          }, OPEN_DEADLINE_MS)
        : setTimeout(() => {
            link.drawn = true;
            void opened();
          }, SETTLE_MS);
    };
    window.addEventListener('message', onMessage);
    setTimeout(() => {
      window.removeEventListener('message', onMessage);
      if (generation !== this.generation || this.status.dataset.kind !== 'busy') return;
      this.setStatus('the preview runtime did not start', 'error');
      this.settle({ ok: false, errors: ['the SoftN preview runtime did not start within 20 s'] });
    }, 20_000);
    this.place(frame);
  }
}
