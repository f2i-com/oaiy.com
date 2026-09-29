import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TriangleAlert } from 'lucide-react';
import {
  API_BASE,
  bridge,
  companion,
  engines,
  plugins,
  voices,
  type CompanionOfferRequest,
  type ModulesSnapshot,
  type PluginRecord,
} from './api';
import { useToast } from './Toasts';
import { moduleOn, useModules } from './useModules';

/**
 * Host for a plugin-contributed screen.
 *
 * A plugin ships its screen as a BODY FRAGMENT plus css/js files (declared in
 * `manifest.ui.screens[].files`); the host composes them into one document and
 * runs it in a sandboxed, opaque-origin iframe under a strict CSP. The plugin's
 * scripts therefore have no network of their own — everything they need reaches
 * the host through `window.PluginHost`, a postMessage RPC bridge this component
 * injects before any plugin script executes.
 *
 * That indirection is the point: the screen is third-party code, so it gets a
 * named capability surface (invoke a connector command, subscribe to declared
 * events, read a snapshot) instead of a network and an origin.
 */

/** The dashboard pages a plugin screen may open with `PluginHost.navigate`. */
export const PLUGIN_NAV_TARGETS = ['agent', 'calendar', 'engines', 'services', 'overview', 'plugins', 'providers'] as const;
export type PluginNavTarget = (typeof PLUGIN_NAV_TARGETS)[number];

export function isPluginNavTarget(value: unknown): value is PluginNavTarget {
  return typeof value === 'string' && (PLUGIN_NAV_TARGETS as readonly string[]).includes(value);
}

/** The pages that are there only while their module is (a plugin provides it). */
const TARGET_MODULE: Partial<Record<PluginNavTarget, string>> = { calendar: 'calendar' };

/** May a plugin screen open `value` now: one of the pages, and not one whose module is off (or not yet known)? */
export function pluginNavAllowed(value: unknown, modules: ModulesSnapshot | null): value is PluginNavTarget {
  if (!isPluginNavTarget(value)) return false;
  const module = TARGET_MODULE[value];
  return !module || moduleOn(modules, module) === true;
}

/** What OAIY answers calls with, for a plugin screen to show: the voice
 *  chosen for calls and the engines' language model. Names and states only;
 *  a part that cannot be read comes back null rather than failing the rest. */
export interface PluginOaiyStatus {
  voice: { chosen: string | null } | null;
  llm: { running: boolean; state: string | null; resident: string | null } | null;
}

export async function pluginOaiyStatus(): Promise<PluginOaiyStatus> {
  const [voice, engine] = await Promise.allSettled([voices.list(), engines.status()]);
  return {
    voice: voice.status === 'fulfilled' ? { chosen: voice.value.chosen ?? null } : null,
    llm:
      engine.status === 'fulfilled'
        ? {
            running: !!engine.value.running,
            state: engine.value.llm?.state ?? null,
            resident: engine.value.llm?.resident ?? null,
          }
        : null,
  };
}

/** What a setup wizard's `screen` step hears from the screen (`PluginHost.setup`). */
export interface SetupScreenCalls {
  /** `fraction` 0..1 or null, and a short text, shown under the step. */
  progress: (fraction: number | null, text: string) => void;
  /** A hint: check the step's `done` now (a step with no check is recorded done). */
  done: (detail?: string) => void | Promise<void>;
  /** A message to show on the step. */
  fail: (message: string) => void;
  /** The plugin's own wizard is finished. */
  finish: () => void | Promise<void>;
}

/** A plugin screen shown as a step of its setup wizard. */
export interface SetupScreen {
  /** The step's id (the frame is a fresh document per step). */
  step: string;
  /** The view the screen shows (the step's `view`). */
  view: string;
  calls: SetupScreenCalls;
}

interface Props {
  pluginId: string;
  /** The `ui.nav[].id` that was clicked. */
  navId?: string;
  /** Or the `ui.screens[].id` to show (a setup step names its screen directly). */
  screenId?: string;
  /** Opens one of the dashboard's own pages for the screen's `navigate`. */
  onNavigate?: (view: PluginNavTarget) => void;
  /** Setup mode: the screen is a step of the plugin's setup wizard. */
  setup?: SetupScreen;
}

interface UiNav {
  id?: string;
  label?: string;
  screen?: string;
}
interface UiScreen {
  id?: string;
  title?: string;
  entry?: string;
  files?: string[];
}

/** Resolve provider routes at the host, which knows its configured API port.
 * Plugin screens cannot infer it from their opaque iframe origin. */
export function pluginAiSources(sources: unknown[], apiBase = API_BASE): unknown[] {
  return sources.map((source) => {
    if (!source || typeof source !== 'object' || Array.isArray(source)) return source;
    const record = source as Record<string, unknown>;
    if (record.kind !== 'provider') return source;
    const providerId = typeof record.providerId === 'string' ? record.providerId
      : typeof record.id === 'string' && record.id.startsWith('provider:') ? record.id.slice(9) : '';
    if (!providerId) return source;
    return {
      ...record,
      gatewayUrl: `${apiBase.replace(/\/+$/, '')}/api/ai/providers/${encodeURIComponent(providerId)}`,
    };
  });
}

/**
 * The dashboard's own fonts, as `@font-face` rules with data: URLs.
 *
 * A plugin screen's CSP allows fonts only from data:, so without this every
 * screen fell back to the system font and read as a different app. The files
 * are the ones the dashboard self-hosts; they are read once per session. Any
 * failure gives '' and the screen keeps its fallback fonts.
 */
let hostFontCss: Promise<string> | null = null;
export function pluginFontCss(): Promise<string> {
  if (!hostFontCss) {
    const face = async (family: string, file: string, weight: string) => {
      const resp = await fetch(`/fonts/${file}`);
      if (!resp.ok) throw new Error(`${file} → HTTP ${resp.status}`);
      const bytes = new Uint8Array(await resp.arrayBuffer());
      let binary = '';
      for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      return `@font-face{font-family:'${family}';src:url(data:font/woff2;base64,${btoa(binary)}) format('woff2');font-weight:${weight};font-display:swap}`;
    };
    hostFontCss = Promise.all([
      face('Public Sans', 'public-sans.woff2', '300 700'),
      face('JetBrains Mono', 'jetbrains-mono.woff2', '400 700'),
    ]).then(
      (faces) => faces.join('\n'),
      () => {
        hostFontCss = null;
        return '';
      },
    );
  }
  return hostFontCss;
}

/** The page's gutters as a padding (top, sides, bottom): a plugin screen runs
 *  edge to edge and draws them inside its own document, so its cards line up
 *  with the page's header as a built-in page's do. */
export function pageGutter(): string {
  const style = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return `${read('--page-pad-t', '24px')} ${read('--page-pad-x', '30px')} ${read('--page-pad-b', '40px')}`;
}

/** What every screen starts from, ahead of the plugin's own css (which may
 *  override any of it): the page's gutters, and scrollbars like the host's. */
export const SCREEN_BASE_CSS =
  'body { box-sizing: border-box; min-height: 100vh; padding: var(--host-page-pad, 24px 30px 40px); }' +
  ' * { scrollbar-width: thin; scrollbar-color: var(--strong-border, #9f998f) transparent; }';

/** The bootstrap injected ahead of the plugin's own scripts. Plain ES5-ish so it
 *  runs before any transform, and self-contained: the iframe has no imports. */
export const HOST_BOOTSTRAP = `
(function () {
  var seq = 0;
  var pending = {};
  var subs = [];
  // Setup mode: the host stamped the step's context into the document before
  // this ran (a wizard step). Absent on an ordinary screen.
  var SETUP = window.__oaiySetup && typeof window.__oaiySetup === 'object' ? window.__oaiySetup : null;
  function call(method, args) {
    return new Promise(function (resolve, reject) {
      var id = 'r' + ++seq;
      pending[id] = { resolve: resolve, reject: reject };
      pending[id].timer = setTimeout(function () {
        delete pending[id];
        reject(new Error('The desktop did not respond. The outcome may be unknown; check its status before trying the action again.'));
      }, 20000);
      parent.postMessage({ __pluginHost: 1, id: id, method: method, args: args || [] }, '*');
    });
  }
  // The plugin document is opaque-origin, so the host cannot reach in and
  // restyle it: the theme has to arrive as a message and be applied from in
  // here. Both conventions are written — 'fl-dark' is what the shipped plugin
  // stylesheets key their dark tokens off, and data-theme mirrors the host's
  // own attribute so a newer plugin can use either without a host change.
  function applyTheme(mode) {
    var el = document.documentElement;
    if (!el) return;
    if (mode === 'dark') el.classList.add('fl-dark');
    else el.classList.remove('fl-dark');
    el.setAttribute('data-theme', mode === 'dark' ? 'dark' : 'light');
  }
  window.addEventListener('message', function (e) {
    if (e.source !== parent) return;
    var m = e.data;
    if (!m || !m.__pluginHost) return;
    if (m.theme) { applyTheme(m.theme); return; }
    // The page's gutters: the screen runs edge to edge and draws them itself.
    // Not in a wizard step, whose pane draws its own padding (the gutter stays 0).
    if (m.gutter) { if (!SETUP) document.documentElement.style.setProperty('--host-page-pad', m.gutter); return; }
    if (m.event) { subs.forEach(function (s) { try { s(m.event); } catch (_) {} }); return; }
    var p = pending[m.id];
    if (!p) return;
    delete pending[m.id];
    clearTimeout(p.timer);
    if (m.ok) p.resolve(m.data); else p.reject(new Error(m.error || 'host call failed'));
  });
  window.PluginHost = {
    command: function (name, payload) { return call('command', [name, payload]); },
    toast: function (kind, message) { return call('toast', [kind, message]); },
    snapshot: function () { return call('snapshot', []); },
    aiSources: function () { return call('aiSources', []); },
    restartPlugin: function () { return call('restartPlugin', []); },
    // Open one of the dashboard's own pages ('agent', 'calendar', 'engines',
    // ...). The frame cannot navigate itself out of the sandbox, so the host
    // does it, and only to its fixed list.
    navigate: function (view) { return call('navigate', [view]); },
    // Read-only: the voice chosen for calls and the engines' language model.
    oaiyStatus: function () { return call('oaiyStatus', []); },
    events: {
      // Resolves to a HANDLE (with .unsubscribe()), not the function itself —
      // callers do subscribe(...).then(h => handle = h).
      subscribe: function (names, cb) {
        subs.push(cb);
        return call('subscribe', [names || []]).then(function () {
          return {
            unsubscribe: function () {
              subs = subs.filter(function (s) { return s !== cb; });
            }
          };
        }, function (error) {
          subs = subs.filter(function (s) { return s !== cb; });
          throw error;
        });
      }
    },
    // Consent is a plain connector surface on the plugin (consent.get/set/revoke
    // are declared commands), so it routes through the same gateway as any other
    // command — the plugin owns the grant, the host just carries it.
    consent: {
      get: function () { return call('command', ['consent.get']); },
      issue: function (grant) { return call('command', ['consent.set', grant]); },
      revoke: function () { return call('command', ['consent.revoke']); }
    },
    // Companion device trust. Served by the host, not by the plugin.
    //
    // The plugin does hold the endpoint SIGNING key — it hosts the WebRTC peer,
    // so it has to sign as this desktop. What it must not hold is the power to
    // change who is trusted: approving and revoking a phone stay here, on the
    // side of the boundary the plugin cannot reach.
    //
    // All SEVEN methods are defined, including the three the screen only calls
    // from a confirm step. A missing one throws synchronously inside a handler
    // that has already set busy = true and cannot clear it, which leaves every
    // button on the tab disabled until reload.
    companionPairing: {
      status: function () { return call('companion.status', []); },
      createOffer: function (body) { return call('companion.createOffer', [body]); },
      receiveResponse: function (response) { return call('companion.receiveResponse', [response]); },
      approve: function (id) { return call('companion.approve', [id]); },
      deny: function (id) { return call('companion.deny', [id]); },
      revoke: function (thumbprint) { return call('companion.revoke', [thumbprint]); },
      rotateDesktopKey: function () { return call('companion.rotate', []); }
    }
  };
  // The setup wizard's side of a step, defined only in setup mode, so a
  // screen can tell a wizard step from its normal page by its presence.
  // Every call returns a Promise.
  if (SETUP) {
    var context = { mode: 'setup', step: String(SETUP.step || ''), view: String(SETUP.view || '') };
    window.PluginHost.setup = {
      context: function () { return Promise.resolve({ mode: context.mode, step: context.step, view: context.view }); },
      progress: function (fraction, text) {
        var f = fraction == null || isNaN(Number(fraction)) ? null : Math.max(0, Math.min(1, Number(fraction)));
        return call('setup.progress', [f, text == null ? '' : String(text)]);
      },
      done: function (detail) { return call('setup.done', detail == null ? [] : [String(detail)]); },
      fail: function (message) { return call('setup.fail', [message == null ? '' : String(message)]); },
      finish: function () { return call('setup.finish', []); }
    };
  }
})();
`;

/** Escape text for an HTML attribute value. */
function attr(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** The pieces of a plugin screen's document. */
export interface ScreenParts {
  /** The screen's body fragment. */
  body: string;
  /** Its css, then the host's fonts. */
  css: string;
  fonts: string;
  /** Its scripts, one `<script>` each, in manifest order. */
  scripts: string[];
  dark: boolean;
  /** The page's gutters (ignored in setup mode, where the gutter is 0). */
  gutter: string;
  /** Setup mode: the wizard step's context, stamped before the bootstrap runs. */
  setup?: { step: string; view: string };
}

/**
 * A plugin screen's whole document, for the iframe's srcdoc (an opaque origin).
 *
 * In setup mode it is stamped for the step before first paint: the view on
 * `<html data-oaiy-setup>` (so the plugin's CSS can hide its own chrome), the
 * context as `window.__oaiySetup` ahead of the bootstrap (which then defines
 * `PluginHost.setup`), and a gutter of 0 (the wizard pane draws its own).
 */
export function assembleScreenDocument(parts: ScreenParts): string {
  const theme = parts.dark ? ' class="fl-dark" data-theme="dark"' : ' data-theme="light"';
  const setupAttr = parts.setup ? ` data-oaiy-setup="${attr(parts.setup.view)}"` : '';
  const pad = parts.setup ? '0' : parts.gutter;
  // JSON in a script: `<` escaped so a view can never close the tag.
  const stamp = parts.setup
    ? `<script>window.__oaiySetup=${JSON.stringify({ mode: 'setup', step: parts.setup.step, view: parts.setup.view }).replace(/</g, '\\u003c')};</script>`
    : '';
  const scripts = parts.scripts.map((s) => `<script>${s.replace(/<\/script/gi, '<\\/script')}</script>`).join('');
  return (
    `<!doctype html><html${theme}${setupAttr} style="--host-page-pad: ${attr(pad)}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">` +
    // No external anything: the plugin ships inline SVG and its own CSS.
    `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'">` +
    `<style>${SCREEN_BASE_CSS}\n${parts.fonts}\n${parts.css}</style>${stamp}</head><body>${parts.body}` +
    `<script>${HOST_BOOTSTRAP}</script>${scripts}</body></html>`
  );
}

export default function PluginScreenPage(props: Props) {
  // Changing screens discards the old document, pending RPCs and event cursor.
  // A setup step is a fresh document per step.
  const key = props.setup ? `${props.pluginId}:setup:${props.setup.step}` : `${props.pluginId}:${props.navId ?? ''}:${props.screenId ?? ''}`;
  return <PluginScreenContent key={key} {...props} />;
}

function PluginScreenContent({ pluginId, navId, screenId, onNavigate, setup }: Props) {
  const toast = useToast();
  // Held in a ref so a new callback each render does not re-register the
  // message pump (handleCall depends on nothing that changes per render).
  const navigateRef = useRef(onNavigate);
  navigateRef.current = onNavigate;
  // The step's context is fixed for this document (it is keyed by step); its
  // callbacks may change each render, so they are read through a ref.
  const setupRef = useRef(setup);
  setupRef.current = setup;
  const setupContext = useMemo(() => (setup ? { step: setup.step, view: setup.view } : undefined), [setup?.step, setup?.view]);
  // Which pages may be opened depends on the modules on now (the calendar only while there is one).
  const modules = useModules();
  const modulesRef = useRef(modules);
  modulesRef.current = modules;
  const [record, setRecord] = useState<PluginRecord | null | undefined>(undefined);
  // The manifest selects and assembles the iframe once. Runtime status changes
  // separately so a health refresh cannot remount a call console or transcript.
  const [runtimeStatus, setRuntimeStatus] = useState<Pick<PluginRecord, 'state' | 'reason'> | null>();
  const snapshotGeneration = useRef(0);
  const snapshotSequence = useRef(0);
  const [doc, setDoc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const frameRef = useRef<HTMLIFrameElement>(null);
  /** Event names the screen subscribed to, and the poll cursor. */
  const subscribed = useRef<string[]>([]);
  const cursor = useRef<number>(0);

  useEffect(() => {
    let cancelled = false;
    const generation = ++snapshotGeneration.current;
    setRecord(undefined);
    setRuntimeStatus(undefined);
    setDoc(null);
    setError(null);
    subscribed.current = [];
    cursor.current = 0;
    (async () => {
      try {
        const snap = await plugins.list();
        if (!cancelled) {
          const initial = snap.plugins.find((p) => p.id === pluginId) ?? null;
          setRecord(initial);
          setRuntimeStatus(initial ? { state: initial.state, reason: initial.reason } : null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
      if (snapshotGeneration.current === generation) snapshotGeneration.current++;
    };
  }, [pluginId, attempt]);

  const screen = useMemo(() => {
    const ui = (record?.manifest as unknown as { ui?: { nav?: UiNav[]; screens?: UiScreen[] } } | undefined)?.ui;
    if (!ui) return null;
    const wanted = screenId ?? (ui.nav ?? []).find((n) => n.id === navId)?.screen;
    return (ui.screens ?? []).find((s) => s.id === wanted) ?? null;
  }, [record, navId, screenId]);

  // Compose the document: fragment + inlined css + the bootstrap + the plugin's
  // scripts in manifest order. Assembled here (not server-side) so the iframe can
  // use srcdoc, which is what gives it an opaque origin.
  useEffect(() => {
    if (!screen?.id || !screen.entry) return;
    let cancelled = false;
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 15000);
    (async () => {
      const base = `${API_BASE}/api/plugins/${encodeURIComponent(pluginId)}/ui/${encodeURIComponent(screen.id!)}`;
      const fetchText = async (rel: string) => {
        const resp = await fetch(`${base}/${rel.split('/').map(encodeURIComponent).join('/')}`, {
          cache: 'no-store',
          signal: controller.signal,
        });
        if (!resp.ok) throw new Error(`${rel} → HTTP ${resp.status}`);
        return resp.text();
      };
      try {
        const files = screen.files ?? [screen.entry!];
        const body = await fetchText(screen.entry!);
        const css = (
          await Promise.all(files.filter((f) => f.endsWith('.css')).map(fetchText))
        ).join('\n');
        // Manifest order matters: app.js defines the registry the tab modules
        // register into, and it is listed first.
        //
        // One <script> PER FILE, not one concatenated blob: concatenating puts
        // every module in a single script, so a throw anywhere aborts all the
        // modules after it — which silently cost us every tab that registers
        // after the first failure. Separate tags give identical global semantics
        // and identical ordering, but isolate a bad module to itself.
        const jsFiles = files.filter((f) => f.endsWith('.js'));
        const [sources, fonts] = await Promise.all([Promise.all(jsFiles.map(fetchText)), pluginFontCss()]);
        if (cancelled) return;
        // Stamped at assembly rather than messaged in after load, so the screen
        // never paints light-then-flips. Read from the live attribute instead of
        // a prop on purpose: this must NOT be a dependency of this effect, or a
        // theme flip would re-assemble srcdoc and remount the plugin — losing a
        // live call console mid-call to change a colour.
        const dark = document.documentElement.getAttribute('data-theme') !== 'light';
        setDoc(assembleScreenDocument({ body, css, fonts, scripts: sources, dark, gutter: pageGutter(), setup: setupContext }));
        setError(null);
      } catch (e) {
        if (!cancelled) setError(controller.signal.aborted ? 'Loading took too long. Check the desktop connection and try again.' : e instanceof Error ? e.message : String(e));
      } finally {
        window.clearTimeout(timer);
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [screen, pluginId, setupContext]);

  /** Service one RPC from the screen. */
  const handleCall = useCallback(
    async (method: string, args: unknown[]): Promise<unknown> => {
      switch (method) {
        case 'command': {
          const [name, payload] = args as [string, unknown];
          // Always send an idempotency key: the gateway REQUIRES one for the
          // plugin's journalled (physically side-effecting) commands, and a
          // fresh key per user action is the correct semantics for the rest.
          const key = `ui-${name}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
          const res = await bridge.connectorRequest(pluginId, name, payload, key);
          if (res.ok === false) throw new Error('The desktop could not complete this plugin request.');
          // Plugins answer with the SDK envelope { ok, data }; hand the screen
          // the inner data, which is what its call sites expect.
          const r = res.result as { ok?: boolean; data?: unknown } | undefined;
          if (r?.ok === false) throw new Error(String((r as { error?: unknown }).error ?? 'The plugin could not complete this action.'));
          return r && typeof r === 'object' && 'data' in r ? r.data : r;
        }
        case 'toast': {
          const [kind, message] = args as [string, string];
          toast.push({
            kind: kind === 'error' ? 'error' : kind === 'success' ? 'success' : 'info',
            title: String(message ?? ''),
          });
          return true;
        }
        case 'snapshot': {
          const generation = snapshotGeneration.current;
          const sequence = ++snapshotSequence.current;
          const snap = await plugins.list();
          const current = snap.plugins.find((p) => p.id === pluginId) ?? null;
          if (generation === snapshotGeneration.current && sequence === snapshotSequence.current) {
            setRuntimeStatus((previous) => {
              if (!current) return null;
              return previous?.state === current.state && previous?.reason === current.reason
                ? previous : { state: current.state, reason: current.reason };
            });
          }
          return current;
        }
        case 'aiSources':
          return pluginAiSources((await bridge.aiSources()).sources);
        case 'restartPlugin':
          await plugins.stop(pluginId).catch(() => {});
          await plugins.start(pluginId);
          return true;
        case 'navigate': {
          const [target] = args as [unknown];
          const go = navigateRef.current;
          if (!go || !pluginNavAllowed(target, modulesRef.current)) throw new Error('That page cannot be opened from a plugin screen.');
          go(target);
          return true;
        }
        case 'oaiyStatus':
          return await pluginOaiyStatus();
        // Companion trust. The screen never names a plugin: it gets its OWN id
        // from the host, so a screen cannot administer another plugin's phones
        // by asking nicely.
        case 'companion.status':
          return await companion.status(pluginId);
        case 'companion.createOffer':
          return await companion.createOffer(pluginId, (args[0] ?? {}) as CompanionOfferRequest);
        case 'companion.receiveResponse':
          return await companion.receiveResponse(pluginId, args[0]);
        case 'companion.approve':
          return await companion.approve(pluginId, String(args[0] ?? ''));
        case 'companion.deny':
          // The DELETE/deny routes answer 204, so there is no body to hand
          // back; the screen only awaits completion and then refreshes.
          await companion.deny(pluginId, String(args[0] ?? ''));
          return true;
        case 'companion.revoke':
          await companion.revoke(pluginId, String(args[0] ?? ''));
          return true;
        case 'companion.rotate':
          return await companion.rotate(pluginId);
        case 'subscribe': {
          const [names] = args as [string[]];
          subscribed.current = Array.isArray(names) ? names : [];
          // Start from the tail so a screen doesn't replay history on open.
          cursor.current = (await bridge.events(0, 1)).next;
          return true;
        }
        // The setup wizard's calls: only a wizard step answers them.
        case 'setup.progress':
        case 'setup.done':
        case 'setup.fail':
        case 'setup.finish': {
          const calls = setupRef.current?.calls;
          if (!calls) throw new Error('This screen is not a setup step.');
          if (method === 'setup.progress') {
            const [fraction, text] = args as [unknown, unknown];
            const f = typeof fraction === 'number' && Number.isFinite(fraction) ? Math.max(0, Math.min(1, fraction)) : null;
            calls.progress(f, String(text ?? '').slice(0, 300));
          } else if (method === 'setup.done') {
            await calls.done(args[0] == null ? undefined : String(args[0]).slice(0, 300));
          } else if (method === 'setup.fail') {
            calls.fail(String(args[0] ?? '').slice(0, 500) || 'This step could not continue.');
          } else {
            await calls.finish();
          }
          return true;
        }
        default:
          throw new Error(`unsupported host call "${method}"`);
      }
    },
    [pluginId, toast],
  );

  // Follow the host's theme for as long as this screen is mounted.
  //
  // Watching the attribute rather than taking a prop keeps this independent of
  // who flips the theme (the header toggle today, anything else later) and, more
  // importantly, keeps the theme out of the srcdoc effect's dependencies — that
  // effect remounts the plugin, and a colour change must never do that.
  useEffect(() => {
    const send = () => {
      const mode = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
      frameRef.current?.contentWindow?.postMessage({ __pluginHost: 1, theme: mode }, '*');
    };
    const observer = new MutationObserver(send);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    // Also once now: the iframe may have finished loading after it was stamped,
    // and a re-assembled document starts from the host's current theme anyway.
    send();
    return () => observer.disconnect();
  }, [doc]);

  // And the page's gutters, which change with the window's width. Not for a
  // wizard step: its pane draws its own padding, and the screen's stays 0.
  useEffect(() => {
    if (setupContext) return;
    const send = () => frameRef.current?.contentWindow?.postMessage({ __pluginHost: 1, gutter: pageGutter() }, '*');
    window.addEventListener('resize', send);
    send();
    return () => window.removeEventListener('resize', send);
  }, [doc, setupContext]);

  // RPC pump: only messages from OUR iframe are serviced.
  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      const m = e.data as { __pluginHost?: number; id?: string; method?: string; args?: unknown[] };
      if (!m || !m.__pluginHost || !m.id || !m.method) return;
      const frame = frameRef.current;
      if (!frame || e.source !== frame.contentWindow) return;
      handleCall(m.method, m.args ?? [])
        .then((data) => frame.contentWindow?.postMessage({ __pluginHost: 1, id: m.id, ok: true, data }, '*'))
        .catch((err) =>
          frame.contentWindow?.postMessage(
            { __pluginHost: 1, id: m.id, ok: false, error: err instanceof Error ? err.message : String(err) },
            '*',
          ),
        );
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [handleCall]);

  // Forward declared plugin events to the screen.
  useEffect(() => {
    let busy = false;
    let cancelled = false;
    const id = window.setInterval(async () => {
      if (busy || document.hidden || subscribed.current.length === 0) return;
      busy = true;
      try {
        const res = await bridge.events(cursor.current, 100);
        if (cancelled) return;
        cursor.current = res.next;
        for (const e of res.events) {
          const env = e.envelope as { name?: string };
          if (env?.name && subscribed.current.includes(env.name)) {
            frameRef.current?.contentWindow?.postMessage({ __pluginHost: 1, event: env }, '*');
          }
        }
      } catch {
        /* transient — the next tick retries */
      } finally {
        busy = false;
      }
    }, 2000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [attempt]);

  if (error) {
    return (
      <div className="panel">
        <div className="banner banner-err" role="alert">
          <span>Couldn't load the plugin screen: {error}</span>
          <button className="btn" onClick={() => setAttempt((value) => value + 1)}>Retry loading</button>
        </div>
      </div>
    );
  }
  if (record === undefined) {
    return (
      <div className="panel">
        <div className="empty-state">Loading…</div>
      </div>
    );
  }
  if (record === null) {
    return (
      <div className="panel">
        <div className="empty-state">
          <TriangleAlert size={22} style={{ opacity: 0.5 }} />
          <p>That plugin is no longer installed.</p>
        </div>
      </div>
    );
  }
  if (!screen) {
    return (
      <div className="panel">
        <div className="empty-state">
          <TriangleAlert size={22} style={{ opacity: 0.5 }} />
          <p>This plugin doesn't ship that screen.</p>
          <p style={{ fontSize: 13, opacity: 0.7 }}>
            Its manifest declares no <code>ui.screens</code> entry for “{screenId ?? navId}”.
          </p>
        </div>
      </div>
    );
  }

  const currentStatus = runtimeStatus === undefined ? record : runtimeStatus;
  return (
    <div className={setup ? 'panel plugin-setup-frame' : 'panel'}>
      {currentStatus === null ? (
        <div className="banner banner-err" role="alert">This plugin is no longer installed. Its open screen has been kept so you can review the current information.</div>
      ) : setup ? // A wizard step keeps going while the plugin restarts (a driver
        // install restarts it): the wizard says how the plugin stands, and
        // the frame is never reloaded for it.
        null : currentStatus.state !== 'running' && (
        <div className="banner banner-err" role="alert">
          <span>
            <TriangleAlert size={13} /> “{record.manifest?.name ?? record.id}” is {currentStatus.state}
            {currentStatus.reason ? ` — ${currentStatus.reason}` : ''}
            {/* An unhealthy plugin is RUNNING and still answers commands (health
                is a coarse signal), so telling the user to start it would be
                wrong — and often the screen itself is where the fix lives. */}
            {currentStatus.state === 'unhealthy'
              ? '. The screen still works — you may be able to resolve this here.'
              : '. Start it from Plugins before using this screen.'}
          </span>
        </div>
      )}
      {doc === null ? (
        <div className="empty-state" role="status">Loading screen…</div>
      ) : (
        <iframe
          key={attempt}
          ref={frameRef}
          className="plugin-screen"
          title={screen.title ?? screenId ?? navId}
          srcDoc={doc}
          /* allow-scripts WITHOUT allow-same-origin: the screen runs at an opaque
             origin, so it has no access to the host's storage or the local API
             except through the PluginHost bridge above. */
          sandbox="allow-scripts allow-forms allow-modals"
        />
      )}
    </div>
  );
}
