import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react';
import {
  Bot,
  CalendarDays,
  Check,
  Copy,
  Cpu,
  ExternalLink,
  HardDrive,
  History,
  LayoutDashboard,
  LockKeyhole,
  Mail,
  MessageSquare,
  Moon,
  Package,
  Phone,
  Plug,
  Puzzle,
  Server,
  Settings2,
  ShieldCheck,
  Sparkles,
  Sun,
  TriangleAlert,
  Workflow,
  type LucideIcon,
} from 'lucide-react';
import { API_BASE, openExternal, phone as phoneApi, plugins as pluginsApi } from './api';
import { moduleOn, useModules } from './useModules';
import { applyTheme, initialTheme, THEME_LABEL, type ThemeMode } from './theme';
import ServicesPanel from './ServicesPanel';
import ModelsPanel from './ModelsPanel';
import PythonPanel from './PythonPanel';
import PluginsPanel from './PluginsPanel';
import AiProvidersPanel from './AiProvidersPanel';
import OverviewPanel from './OverviewPanel';
import RunsPanel from './RunsPanel';
import CalendarPanel from './CalendarPanel';
import PluginScreenPage from './PluginScreenPage';
import ConnectionsPanel from './ConnectionsPanel';
import PairingPrompt from './PairingPrompt';
import SettingsPanel from './SettingsPanel';
import EmbeddedPage from './EmbeddedPage';

/**
 * OAIY Desktop: a sidebar of a few sections, and a workspace of topbar / page /
 * endpoint dock. A section with more than one page shows them as tabs under the
 * topbar; every page keeps its own view id, so anything that opens a page by id
 * (the Overview, a plugin screen's `navigate`, the dock) still lands on it.
 *
 * The HTTP API is bound to 127.0.0.1:17972; the health poll below drives the
 * dock. If the Rust side isn't up, everything degrades to "unreachable"
 * without crashing.
 */

interface HealthResponse {
  status: string;
  /** Stable machine identity (`oaiy-desktop`). */
  product?: string;
  /** Bridge Protocol the sidecar speaks, e.g. `oaiy-bridge/1`. */
  protocol?: string;
  version: string;
}

type BuiltinView =
  | 'agent'
  | 'flows'
  | 'calendar'
  | 'engines'
  | 'overview'
  | 'services'
  | 'plugins'
  | 'runs'
  | 'models'
  | 'python'
  | 'providers'
  | 'connections'
  | 'settings';

/** A plugin-contributed screen, addressed as `plugin:<pluginId>:<navId>`. */
type View = BuiltinView | `plugin:${string}:${string}`;

/** The pages shown in webviews of their own, laid over the page by the desktop. */
const EMBEDDED = new Set<View>(['agent', 'flows', 'engines']);

/** One nav entry a plugin contributes (`manifest.ui.nav[]`). */
interface PluginNavEntry {
  pluginId: string;
  pluginName: string;
  navId: string;
  label: string;
  icon?: string;
  badge?: string;
}

type Group = 'Home' | 'Work' | 'Setup';

interface Section {
  id: string;
  label: string;
  icon: LucideIcon;
  group: Group;
  /** Its pages, the first the one it opens on. More than one shows as tabs. */
  tabs: BuiltinView[];
}

/** The sidebar: home, what you work in, and what this machine is set up with. Settings sits at the bottom. */
const SECTIONS: Section[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, group: 'Home', tabs: ['overview'] },
  { id: 'agent', label: 'Agent', icon: Bot, group: 'Work', tabs: ['agent'] },
  { id: 'flows', label: 'Flows', icon: Workflow, group: 'Work', tabs: ['flows', 'runs'] },
  { id: 'calendar', label: 'Calendar', icon: CalendarDays, group: 'Work', tabs: ['calendar'] },
  { id: 'engines', label: 'Engines', icon: Cpu, group: 'Setup', tabs: ['engines', 'models'] },
  { id: 'services', label: 'Services', icon: Server, group: 'Setup', tabs: ['services', 'python'] },
  { id: 'connections', label: 'Connections', icon: Plug, group: 'Setup', tabs: ['connections', 'providers', 'plugins'] },
];
const SETTINGS: Section = { id: 'settings', label: 'Settings', icon: Settings2, group: 'Setup', tabs: ['settings'] };

/** Each page: its tab's name and icon, and the line under the topbar's title. */
const PAGE: Record<BuiltinView, { tab: string; icon: LucideIcon; copy: string }> = {
  overview: {
    tab: 'Overview',
    icon: LayoutDashboard,
    copy: 'Today on this machine: the phone, the model, your appointments, and anything that needs you.',
  },
  agent: {
    tab: 'Agent',
    icon: Bot,
    copy: 'Your projects, and the agent that works in them and answers your texts and calls.',
  },
  flows: { tab: 'Editor', icon: Workflow, copy: 'Build flows, run them, and give the agent new tools.' },
  runs: { tab: 'Run history', icon: History, copy: 'Every flow this machine has run, and why any of them failed.' },
  calendar: {
    tab: 'Calendar',
    icon: CalendarDays,
    copy: 'Appointments, the requests your calls and texts bring in, and the hours the phone offers.',
  },
  engines: {
    tab: 'Engines',
    icon: Cpu,
    copy: 'The language, picture, video, speech, music and 3D models this machine runs, and their endpoints.',
  },
  models: { tab: 'Model files', icon: Package, copy: 'Download weights from Hugging Face and manage what is on disk.' },
  services: { tab: 'Services', icon: Server, copy: 'Start, stop and install the local AI services this machine offers OAIY.' },
  python: { tab: 'Python', icon: HardDrive, copy: 'A bundled Python and reusable venvs, so no system Python is needed.' },
  connections: { tab: 'Connections', icon: ShieldCheck, copy: 'Apps and accounts allowed to run flows and call plugins on this machine.' },
  providers: { tab: 'AI providers', icon: Sparkles, copy: 'Cloud or local AI providers your flows can call. Keys stay on this device.' },
  plugins: { tab: 'Plugins', icon: Puzzle, copy: 'Supervised extensions that add connectors and events to your flows.' },
  settings: { tab: 'Settings', icon: Settings2, copy: 'Where OAIY keeps its data and models, and your Hugging Face token.' },
};

/** The icons a plugin's nav entry may name (`manifest.ui.nav[].icon`); any other gets the plugin piece. */
const PLUGIN_ICONS: Record<string, LucideIcon> = {
  phone: Phone,
  calendar: CalendarDays,
  chat: MessageSquare,
  message: MessageSquare,
  mail: Mail,
  bot: Bot,
};

/** Plugin screens already opened here: their "New" badge is no longer news. */
const SEEN_KEY = 'oaiy.pluginNavSeen';
function readSeen(): Set<string> {
  try {
    const raw = window.localStorage.getItem(SEEN_KEY);
    const list: unknown = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(list) ? list.filter((v): v is string => typeof v === 'string') : []);
  } catch {
    return new Set();
  }
}
function writeSeen(seen: Set<string>) {
  try {
    window.localStorage.setItem(SEEN_KEY, JSON.stringify([...seen]));
  } catch {
    /* storage can be unavailable; the badge just shows again next time */
  }
}

function sectionOf(view: View): Section | null {
  if (view === 'settings') return SETTINGS;
  return SECTIONS.find((s) => (s.tabs as View[]).includes(view)) ?? null;
}

/** A section's pages as tabs: the dashboard's segmented control, keyboard-driven with the arrow keys. */
function SectionTabs({ section, view, onSelect }: { section: Section; view: View; onSelect: (v: BuiltinView) => void }) {
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = section.tabs.indexOf(view as BuiltinView);
    const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!step || i < 0) return;
    e.preventDefault();
    const next = section.tabs[(i + step + section.tabs.length) % section.tabs.length];
    onSelect(next);
    requestAnimationFrame(() => document.getElementById(`tab-${next}`)?.focus());
  };
  return (
    <div className="section-tabs">
      <div className="seg-tabs" role="tablist" aria-label={`${section.label} pages`} onKeyDown={onKey}>
        {section.tabs.map((t) => {
          const Icon = PAGE[t].icon;
          const on = t === view;
          return (
            <button
              type="button"
              role="tab"
              id={`tab-${t}`}
              key={t}
              aria-selected={on}
              tabIndex={on ? 0 : -1}
              className={on ? 'active' : undefined}
              onClick={() => onSelect(t)}
            >
              <Icon size={14} />
              <span>{PAGE[t].tab}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [view, setView] = useState<View>('overview');
  /** Nav entries contributed by installed plugins (`manifest.ui.nav[]`). */
  const [pluginNav, setPluginNav] = useState<PluginNavEntry[]>([]);
  /** The plugins have been listed once (until then an empty nav says nothing). */
  const [pluginNavKnown, setPluginNavKnown] = useState(false);
  /** The phone and the calendar are there while a plugin provides them (Aokie, the AI Receptionist). `null` until known. */
  const modules = useModules();
  const phoneOn = moduleOn(modules, 'phone');
  const calendarOn = moduleOn(modules, 'calendar');
  /** Views already opened this session — they skip the entrance animation. */
  const visited = useRef<Set<string>>(new Set()).current;
  /** The tab each section was last on, so the sidebar brings you back to it. */
  const lastTab = useRef<Record<string, BuiltinView>>({}).current;
  const [seen, setSeen] = useState<Set<string>>(readSeen);
  const [theme, setThemeState] = useState<ThemeMode>(initialTheme);
  const [copied, setCopied] = useState(false);

  const setTheme = useCallback((next: ThemeMode) => {
    setThemeState(next);
    applyTheme(next);
  }, []);

  const copyEndpoint = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(API_BASE);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard can be unavailable in a webview without a focused document.
    }
  }, []);

  useEffect(() => {
    let id: number | undefined;
    const tick = async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/health`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const body: HealthResponse = await resp.json();
        // A 200 from this port is NOT proof it's OUR server. Sibling apps built
        // from the same template expose an identically-shaped /api/health, so if
        // one of them owns the port first our bind fails and every *other* call
        // 401s against a stranger — while a version-only check reads as healthy.
        // Verify the identity, and name the squatter so the cause is obvious.
        if (body?.product !== 'oaiy-desktop') {
          throw new Error(
            `port ${new URL(API_BASE).port} is owned by "${body?.product ?? 'an unknown app'}", not OAIY`,
          );
        }
        setHealth(body);
        setHealthError(null);
      } catch (e) {
        setHealthError(e instanceof Error ? e.message : String(e));
        setHealth(null);
      }
    };
    // Tray-resident: run the interval only while the window is visible, so a
    // hidden window costs zero timer wake-ups (not just early-returning ticks).
    const start = () => {
      if (id !== undefined) return;
      tick();
      id = window.setInterval(tick, 3000);
    };
    const stop = () => {
      if (id !== undefined) {
        window.clearInterval(id);
        id = undefined;
      }
    };
    const onVis = () => (document.hidden ? stop() : start());
    if (!document.hidden) start();
    document.addEventListener('visibilitychange', onVis);
    // Second, reliable start trigger: the Rust tray-reveal calls set_focus() right
    // after show(), and some webview backends don't emit visibilitychange on a
    // programmatic show — without this the badge can stick on "checking…" on the
    // launch-to-tray-then-Open path. start() is idempotent, so this can't double-run.
    window.addEventListener('focus', start);
    return () => {
      stop();
      document.removeEventListener('visibilitychange', onVis);
      window.removeEventListener('focus', start);
    };
  }, []);

  // A call going on now: the Agent's nav entry says so (the Agent page shows the call itself).
  // Asked only while there is a phone.
  const [onCall, setOnCall] = useState(false);
  useEffect(() => {
    if (phoneOn !== true) {
      setOnCall(false);
      return;
    }
    let cancelled = false;
    const look = async () => {
      try {
        const calls = await phoneApi.calls();
        if (!cancelled) setOnCall(calls.length > 0);
      } catch {
        if (!cancelled) setOnCall(false);
      }
    };
    look();
    const id = window.setInterval(look, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [phoneOn]);

  // Plugins can contribute sidebar entries that open a screen they ship. Polled
  // (not one-shot) so installing or removing a plugin updates the nav without a
  // restart — the same cadence the Plugins panel uses.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const snap = await pluginsApi.list();
        if (cancelled) return;
        const entries: PluginNavEntry[] = [];
        // A plugin turned off contributes nothing (its screens come back with it).
        for (const p of snap.plugins.filter((p) => !p.userDisabled)) {
          const ui = (p.manifest as unknown as { ui?: { nav?: Array<Record<string, string>> } } | undefined)?.ui;
          for (const n of ui?.nav ?? []) {
            if (n.id && n.label) {
              entries.push({
                pluginId: p.id,
                pluginName: p.manifest?.name ?? p.id,
                navId: n.id,
                label: n.label,
                icon: n.icon,
                badge: n.badge,
              });
            }
          }
        }
        setPluginNav(entries);
        setPluginNavKnown(true);
      } catch {
        /* the Plugins panel surfaces the error; the nav just stays as it was */
      }
    };
    load();
    const id = window.setInterval(load, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  // The calendar is there while a plugin provides it: on it when it goes, go to Overview.
  useEffect(() => {
    if (view === 'calendar' && calendarOn === false) setView('overview');
  }, [view, calendarOn]);

  // A plugin screen the user is on can disappear (plugin removed/disabled) —
  // fall back to Overview rather than rendering a dead page.
  useEffect(() => {
    if (!view.startsWith('plugin:')) return;
    const [, pluginId, navId] = view.split(':');
    if (pluginNavKnown && !pluginNav.some((n) => n.pluginId === pluginId && n.navId === navId)) {
      setView('overview');
    }
  }, [view, pluginNav, pluginNavKnown]);

  // Mark AFTER render: the first open of a view still animates, a return does not.
  // A plugin screen once opened loses its "New" badge.
  useEffect(() => {
    visited.add(view);
    const section = sectionOf(view);
    if (section && section.tabs.length > 1) lastTab[section.id] = view as BuiltinView;
    if (view.startsWith('plugin:')) {
      const key = view.slice('plugin:'.length);
      setSeen((prev) => {
        if (prev.has(key)) return prev;
        const next = new Set(prev).add(key);
        writeSeen(next);
        return next;
      });
    }
  }, [view, visited, lastTab]);

  const openSection = (s: Section) => setView(lastTab[s.id] ?? s.tabs[0]);

  // One derived state drives the dock (and the topbar's warning when the API is down).
  const link: 'up' | 'down' | 'pending' = health ? 'up' : healthError ? 'down' : 'pending';
  const section = sectionOf(view);
  // A plugin screen has no PAGE entry — its heading comes from the nav entry the
  // plugin contributed.
  const pluginView = view.startsWith('plugin:') ? view.split(':') : null;
  const activePluginNav = pluginView
    ? pluginNav.find((n) => n.pluginId === pluginView[1] && n.navId === pluginView[2])
    : undefined;
  const header = pluginView
    ? {
        kicker: 'Work',
        title: activePluginNav?.label ?? 'Plugin screen',
        copy: `From the ${activePluginNav?.pluginName ?? pluginView[1]} plugin.`,
      }
    : {
        kicker: section?.group ?? 'Home',
        title: section?.label ?? PAGE[view as BuiltinView].tab,
        copy: PAGE[view as BuiltinView].copy,
      };
  const tabbed = section && section.tabs.length > 1 ? section : null;
  const embedded = EMBEDDED.has(view);
  const visibleSections = SECTIONS.filter((s) => s.id !== 'calendar' || calendarOn === true);

  const navButton = (s: Section) => {
    const Icon = s.icon;
    const active = section?.id === s.id;
    const calling = s.id === 'agent' && onCall;
    return (
      <button
        type="button"
        key={s.id}
        className={`${active ? 'active' : ''}${calling ? ' on-call' : ''}`}
        aria-current={active ? 'page' : undefined}
        /* Below 1240px the label <span> is display:none and the icon is
           aria-hidden, which would leave the button with no accessible name at
           all — so name it explicitly. The name stays the section's even on a
           call (scripts find the Agent by it); the call is said in its description. */
        aria-label={s.label}
        aria-describedby={calling ? 'nav-on-call' : undefined}
        title={calling ? 'Agent: on a call now. Open it to see the call.' : s.label}
        onClick={() => openSection(s)}
      >
        <Icon size={18} />
        <span>{s.label}</span>
        {calling && (
          <em className="nav-badge on-call" id="nav-on-call">
            On call
          </em>
        )}
      </button>
    );
  };

  return (
    <div className="app-shell">
      <a className="skip" href="#main">
        Skip to dashboard
      </a>

      <aside className="sidebar">
        <div className="brand">
          <strong>OAIY</strong>
          <span>Orchestrate AI Yourself</span>
        </div>

        <nav aria-label="Primary">
          {visibleSections.map((s, i) => {
            const prev = visibleSections[i - 1];
            const heading = s.group !== 'Home' && prev?.group !== s.group && (
              <span key={`group-${s.group}`} className="nav-group" aria-hidden>
                {s.group}
              </span>
            );
            // Screens installed plugins contribute go at the end of Work, after
            // the built-ins, so a plugin extends the app without displacing them.
            const plugins =
              s.group === 'Setup' && prev?.group === 'Work'
                ? pluginNav.map((n) => {
                    const value: View = `plugin:${n.pluginId}:${n.navId}`;
                    const Icon = (n.icon && PLUGIN_ICONS[n.icon]) || Puzzle;
                    const isNew = !!n.badge && !seen.has(`${n.pluginId}:${n.navId}`);
                    return (
                      <button
                        type="button"
                        key={value}
                        className={view === value ? 'active' : ''}
                        aria-current={view === value ? 'page' : undefined}
                        aria-label={n.label}
                        title={`${n.label}, from the ${n.pluginName} plugin`}
                        onClick={() => setView(value)}
                      >
                        <Icon size={18} />
                        <span>{n.label}</span>
                        {isNew && <em className="nav-badge">{n.badge}</em>}
                      </button>
                    );
                  })
                : null;
            return [plugins, heading, navButton(s)];
          })}
        </nav>

        <div className="sidebar-fill" />

        <button
          className={view === 'settings' ? 'settings active' : 'settings'}
          type="button"
          aria-current={view === 'settings' ? 'page' : undefined}
          aria-label="Settings"
          title="Settings"
          onClick={() => setView('settings')}
        >
          <Settings2 size={18} />
          <span>Settings</span>
        </button>
      </aside>

      <main id="main" className="workspace" tabIndex={-1}>
        <header className={tabbed ? 'topbar has-tabs' : 'topbar'}>
          <div className="top-title">
            <span className="top-kicker">{header.kicker}</span>
            <h1>{header.title}</h1>
            <p title={header.copy}>{header.copy}</p>
          </div>

          <div className="top-actions">
            {link === 'down' && (
              <button
                type="button"
                className="top-alert"
                title={healthError ?? undefined}
                onClick={() => setView('services')}
              >
                <TriangleAlert size={13} /> API unreachable
              </button>
            )}
            {/* One button showing the theme you would switch TO, matching the
                web app. The label is an action rather than a state, so
                aria-pressed is gone: this does something, it does not report
                whether it is on. */}
            <button
              type="button"
              className="icon-button"
              aria-label={theme === 'dark' ? `Switch to the ${THEME_LABEL.light} light theme` : `Switch to the ${THEME_LABEL.dark} dark theme`}
              title={theme === 'dark' ? `Switch to ${THEME_LABEL.light}` : `Switch to ${THEME_LABEL.dark}`}
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            >
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button
              className="button secondary top-site"
              type="button"
              title="oaiy.com, in your browser"
              onClick={() => openExternal('https://oaiy.com')}
            >
              <span>oaiy.com</span> <ExternalLink size={13} />
            </button>
          </div>
        </header>

        <section className={`view view-${view}`}>
          {tabbed && <SectionTabs section={tabbed} view={view} onSelect={setView} />}
          {/* A page in a webview of its own: the desktop lays it over this box. */}
          {embedded && <EmbeddedPage page={view as 'agent' | 'flows' | 'engines'} />}
          {/* Keyed so React remounts the scroller on a view change — otherwise
              the next view opens at the previous one's scroll offset. `revisit`
              suppresses the entrance animation the second time you open a panel,
              so navigation feels instant instead of replaying a staggered reveal
              on every switch. `page-fill`: a plugin screen fills the page down
              to the dock and scrolls on its own. */}
          <div
            hidden={embedded}
            className={`content-page${visited.has(view) ? ' revisit' : ''}${pluginView ? ' page-fill' : ''}`}
            key={view}
          >
            <PairingPrompt />
            {view === 'overview' && (
              <OverviewPanel
                onNavigate={(v) => setView(v)}
                onOpenPluginScreen={(pluginId, navId) => setView(`plugin:${pluginId}:${navId}`)}
              />
            )}
            {pluginView && (
              <PluginScreenPage pluginId={pluginView[1]} navId={pluginView[2]} onNavigate={(v) => setView(v)} />
            )}
            {view === 'services' && <ServicesPanel />}
            {view === 'plugins' && <PluginsPanel />}
            {view === 'runs' && <RunsPanel />}
            {view === 'calendar' && <CalendarPanel />}
            {view === 'models' && <ModelsPanel />}
            {view === 'providers' && <AiProvidersPanel />}
            {view === 'connections' && <ConnectionsPanel />}
            {view === 'python' && <PythonPanel />}
            {view === 'settings' && <SettingsPanel />}
          </div>
        </section>

        <footer className={link === 'down' ? 'endpoint-dock offline' : link === 'pending' ? 'endpoint-dock pending' : 'endpoint-dock'}>
          <button
            type="button"
            className="dock-status"
            title={healthError ?? 'The services this machine runs'}
            onClick={() => setView('services')}
          >
            <i />
            <span>
              <strong>
                {link === 'up' ? 'Local API running' : link === 'down' ? 'Local API down' : 'Local API…'}
              </strong>
              <small>
                {link === 'up' ? `v${health?.version} · ` : ''}localhost only
              </small>
            </span>
          </button>
          <div className="dock-endpoint">
            <code>{API_BASE}</code>
            <button
              type="button"
              className="icon-button"
              /* The confirmation is otherwise only an icon swap — name the button
                 for its current state so a screen reader hears the copy landed. */
              aria-label={copied ? 'Local API URL copied' : 'Copy the local API URL'}
              title={copied ? 'Copied' : 'Copy endpoint'}
              onClick={() => void copyEndpoint()}
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </div>
          <em className="dock-trust">
            <LockKeyhole size={12} /> Keys stay on this device
          </em>
        </footer>
      </main>
    </div>
  );
}
