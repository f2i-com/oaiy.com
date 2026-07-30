import { useCallback, useEffect, useState } from 'react';
import {
  Check,
  ChevronRight,
  Copy,
  ExternalLink,
  HardDrive,
  LoaderCircle,
  LockKeyhole,
  Moon,
  Package,
  Plug,
  Server,
  Settings2,
  Sparkles,
  Sun,
  TriangleAlert,
  Workflow,
  type LucideIcon,
} from 'lucide-react';
import { API_BASE, openExternal } from './api';
import { applyTheme, initialTheme, THEME_LABEL, type ThemeMode } from './theme';
import ServicesPanel from './ServicesPanel';
import ModelsPanel from './ModelsPanel';
import PythonPanel from './PythonPanel';
import PluginsPanel from './PluginsPanel';
import AiProvidersPanel from './AiProvidersPanel';
import PairingPrompt from './PairingPrompt';
import SettingsPanel from './SettingsPanel';

/**
 * OAIY Desktop — the OAIY design-system shell (222px sidebar + a workspace of
 * topbar / content page / endpoint dock), carrying the same four views it always
 * had: Services · Models · Python · Settings.
 *
 * The HTTP API is bound to 127.0.0.1:17972; the health poll below drives three
 * places at once — the sidebar's engine card, the topbar readout, and the
 * endpoint dock. If the Rust side isn't up, everything degrades to
 * "unreachable" without crashing.
 */

interface HealthResponse {
  status: string;
  /** Stable machine identity (`oaiy-desktop`). */
  product?: string;
  /** Bridge Protocol the sidecar speaks, e.g. `oaiy-bridge/1`. */
  protocol?: string;
  version: string;
}

type View = 'services' | 'plugins' | 'models' | 'python' | 'providers' | 'settings';

const WEB_APP_URL = 'https://oaiy.com/app.html';

const NAV: { value: View; label: string; icon: LucideIcon }[] = [
  { value: 'services', label: 'Services', icon: Server },
  { value: 'plugins', label: 'Plugins', icon: Plug },
  { value: 'models', label: 'Models', icon: Package },
  { value: 'providers', label: 'Providers', icon: Sparkles },
  { value: 'python', label: 'Python', icon: HardDrive },
];

/**
 * `crumb` is the short topbar name; `title` is the descriptive page heading.
 * Keeping them different avoids saying the same words twice on one screen.
 */
const PAGE: Record<View, { crumb: string; kicker: string; title: string; copy: string }> = {
  services: {
    crumb: 'Services',
    kicker: 'Connections',
    title: 'Local AI services',
    copy: 'Start, stop and install the engines this machine exposes to OAIY.',
  },
  plugins: {
    crumb: 'Plugins',
    kicker: 'Extensions',
    title: 'Plugins',
    copy: 'Supervised extensions that add connectors and events to your flows.',
  },
  models: {
    crumb: 'Models',
    kicker: 'Storage',
    title: 'Model library',
    copy: 'Download weights from Hugging Face and manage what is on disk.',
  },
  providers: {
    crumb: 'Providers',
    kicker: 'AI gateway',
    title: 'AI providers',
    copy: 'Cloud or local AI providers your flows can call — keys stay on this device.',
  },
  python: {
    crumb: 'Python',
    kicker: 'Runtime',
    title: 'Portable Python',
    copy: 'A bundled interpreter and reusable venvs — no system Python needed.',
  },
  settings: {
    crumb: 'Settings',
    kicker: 'Local workspace',
    title: 'Settings',
    copy: 'Your machine, your models, your orchestration.',
  },
};

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [view, setView] = useState<View>('services');
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

  // One derived state drives the engine card, the topbar readout and the dock.
  const link: 'up' | 'down' | 'pending' = health ? 'up' : healthError ? 'down' : 'pending';
  const engineClass =
    link === 'up' ? 'engine-card' : `engine-card ${link === 'down' ? 'offline' : 'pending'}`;
  const page = PAGE[view];

  return (
    <div className="app-shell">
      <a className="skip" href="#main">
        Skip to dashboard
      </a>

      <aside className="sidebar">
        <div className="brand">
          <strong>OAIY</strong>
          <span>Orchestrate AI Yourself</span>
          <small>Connect. Draw. Expose.</small>
        </div>

        <button
          className="open-editor"
          type="button"
          onClick={() => openExternal(WEB_APP_URL)}
          aria-label="Open the OAIY flow editor"
          title={WEB_APP_URL}
        >
          <Workflow size={17} />
          <span>Flow editor</span>
          <ExternalLink size={13} />
        </button>

        <nav aria-label="Primary">
          {NAV.map((item) => {
            const Icon = item.icon;
            return (
              <button
                type="button"
                key={item.value}
                className={view === item.value ? 'active' : ''}
                aria-current={view === item.value ? 'page' : undefined}
                /* Below 1240px the label <span> is display:none and the icon is
                   aria-hidden, which would leave the button with no accessible
                   name at all — so name it explicitly. */
                aria-label={item.label}
                title={item.label}
                onClick={() => setView(item.value)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-fill" />

        <div className={engineClass} title={healthError ?? undefined}>
          <Server size={17} />
          <span>
            <strong>Local engine</strong>
            <small>
              {link === 'up'
                ? `v${health?.version} · this device`
                : link === 'down'
                  ? 'Unreachable'
                  : 'Checking…'}
            </small>
          </span>
          <i />
        </div>

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

        <div className="trust">
          <LockKeyhole size={13} />
          <span>Keys stay on this device</span>
        </div>
      </aside>

      <main id="main" className="workspace" tabIndex={-1}>
        <header className="topbar">
          <div className="top-title">
            <span>
              OAIY <ChevronRight size={12} /> Desktop
            </span>
            <div>
              <h1>{page.crumb}</h1>
              <em>Local</em>
              <small
                className={
                  link === 'down' ? 'is-error' : link === 'pending' ? 'is-pending' : undefined
                }
                role="status"
                title={healthError ?? undefined}
              >
                {link === 'up' ? (
                  <>
                    <Check size={13} />
                    API v{health?.version}
                  </>
                ) : link === 'down' ? (
                  <>
                    <TriangleAlert size={13} />
                    API unreachable
                  </>
                ) : (
                  <>
                    <LoaderCircle size={13} />
                    Checking…
                  </>
                )}
              </small>
            </div>
          </div>

          <div className="top-actions">
            <div className="theme-toggle">
              <button
                type="button"
                className={`icon-button ${theme === 'light' ? 'active' : ''}`}
                aria-label="Paper Circuit (light) theme"
                title={THEME_LABEL.light}
                aria-pressed={theme === 'light'}
                onClick={() => setTheme('light')}
              >
                <Sun size={16} />
              </button>
              <button
                type="button"
                className={`icon-button ${theme === 'dark' ? 'active' : ''}`}
                aria-label="Prism Lab (dark) theme"
                title={THEME_LABEL.dark}
                aria-pressed={theme === 'dark'}
                onClick={() => setTheme('dark')}
              >
                <Moon size={16} />
              </button>
            </div>
            <button
              className="button secondary"
              type="button"
              onClick={() => openExternal('https://oaiy.com')}
            >
              oaiy.com <ExternalLink size={13} />
            </button>
          </div>
        </header>

        <section className={`view view-${view}`}>
          {/* keyed so React remounts the scroller on a view change — otherwise
              the next view opens at the previous one's scroll offset, i.e.
              mid-content. */}
          <div className="content-page" key={view}>
            <PairingPrompt />
            <div className="page-intro">
              <div>
                <span className="kicker">{page.kicker}</span>
                <h2>{page.title}</h2>
                <p>{page.copy}</p>
              </div>
            </div>
            {view === 'services' && <ServicesPanel />}
            {view === 'plugins' && <PluginsPanel />}
            {view === 'models' && <ModelsPanel />}
            {view === 'providers' && <AiProvidersPanel />}
            {view === 'python' && <PythonPanel />}
            {view === 'settings' && <SettingsPanel />}
          </div>
        </section>

        <footer className={link === 'down' ? 'endpoint-dock offline' : 'endpoint-dock'}>
          <button type="button" onClick={() => setView('services')}>
            <i />
            <span>
              <strong>
                {link === 'up'
                  ? 'Local API running'
                  : link === 'down'
                    ? 'Local API down'
                    : 'Local API…'}
              </strong>
              <small>localhost only</small>
            </span>
          </button>
          <div>
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
          <em className={link === 'up' ? 'ok' : link === 'down' ? 'bad' : ''}>
            {link === 'up' ? <Check size={12} /> : <LoaderCircle size={12} />}
            {link === 'up' ? `v${health?.version}` : link === 'down' ? 'offline' : 'checking'}
          </em>
          <em>
            <LockKeyhole size={12} /> Local only
          </em>
          <a
            href={WEB_APP_URL}
            onClick={(e) => {
              e.preventDefault();
              openExternal(WEB_APP_URL);
            }}
          >
            Open editor <ChevronRight size={13} />
          </a>
        </footer>
      </main>
    </div>
  );
}
