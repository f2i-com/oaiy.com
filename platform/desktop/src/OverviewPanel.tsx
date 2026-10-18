import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ChevronRight,
  CircleDot,
  History,
  ListChecks,
  Network,
  Package,
  Plug,
  RotateCcw,
  Server,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  X,
} from 'lucide-react';
import { peek, put } from './useCached';
import SetupGuidePanel from './SetupGuidePanel';
import UpdateBanner from './UpdateBanner';
import { BackupLine } from './BackupPanel';
import TodayPanel from './TodayPanel';
import { dismissGuide, reopenGuide, shouldAutoOpen } from './setupGuide';
import type { StepTarget } from './setupGuide';
import {
  aiProviders,
  codex,
  bridge,
  network,
  nodeRuntime,
  type NetworkStatus,
  type RuntimeStatus,
  pairing,
  plugins,
  services,
  type AiProviderPublic,
  type OverviewContribution,
  type PairedApp,
  type PluginRecord,
  type ServiceSnapshot,
} from './api';
import { bindText, pollsOf, usePluginPolls, type BindContext } from './bind';
import { PLUGIN_ICONS } from './sections';
import { useModules } from './useModules';
import { pluginSetupStatus, waitingForSetup } from './setupFlow';
import { useLiveSetup } from './useLiveSetup';
import { useSetupState } from './useSetupState';

/**
 * Overview — the control centre. One screen that answers "is this machine ready
 * to run my flows?" without clicking through five panels: what's running, what
 * needs attention, and which apps are connected.
 *
 * Plugins can contribute their own cards here (`ui.overview`, served in
 * `/api/modules` contributions: a `hero`, a `status` row or a `tile`), so a
 * plugin like the Aokie phone bridge can surface its own status and a deep link
 * into its screen instead of being invisible until you open Plugins. Their
 * texts are looked up by bind.ts: `$health.*` from the plugin's last health
 * report, `$poll.<card>.*` from its status-card polls.
 */

const POLL_MS = 4000;

/** How many failed runs the person has seen (opened Run history from here, or dismissed the warning). */
const FAILED_SEEN = 'oaiy.overview.failedSeen';

function readFailedSeen(): number {
  try {
    const n = Number(localStorage.getItem(FAILED_SEEN) ?? 0);
    return Number.isFinite(n) && n > 0 ? n : 0;
  } catch {
    return 0;
  }
}

export type OverviewNav =
  | 'agent'
  | 'calendar'
  | 'engines'
  | 'services'
  | 'plugins'
  | 'runs'
  | 'models'
  | 'providers'
  | 'python'
  | 'connections'
  | 'network'
  | 'settings';

interface Props {
  /** Jump to one of the built-in panels. */
  onNavigate: (view: OverviewNav) => void;
  /** Open a plugin-contributed screen (pluginId, navId). */
  onOpenPluginScreen: (pluginId: string, navId: string) => void;
}

/** A card's icon, by the name the plugin gave. */
const cardIcon = (name?: string) => (name && Object.prototype.hasOwnProperty.call(PLUGIN_ICONS, name) && PLUGIN_ICONS[name]) || Sparkles;

/** What a plugin's bindings are looked up in. Its health only while it runs: a stopped plugin's last report is old news. */
function bindContext(plugin: PluginRecord | undefined, answers: Record<string, unknown>, pluginId: string): BindContext {
  const live = plugin?.state === 'running' || plugin?.state === 'unhealthy';
  return { health: live ? plugin?.lastHealth : undefined, polls: pollsOf(answers, pluginId) };
}

/** A plugin's health status word as the person reads it, with its pill; null for a word not known here. */
function healthPill(status: string | null): { label: string; badge: string } | null {
  switch ((status ?? '').trim().toLowerCase()) {
    case 'ok':
    case 'healthy':
      return { label: 'Healthy', badge: 'badge-ok' };
    case 'degraded':
    case 'warn':
    case 'warning':
      return { label: 'Needs attention', badge: 'badge-neutral' };
    case 'unhealthy':
    case 'error':
    case 'down':
      return { label: 'Unhealthy', badge: 'badge-err' };
    default:
      return null;
  }
}

export default function OverviewPanel({ onNavigate, onOpenPluginScreen }: Props) {
  // Seeded from the last known values so returning to Overview paints straight
  // away instead of showing four em-dashes while the round trips land.
  const [svc, setSvc] = useState<ServiceSnapshot[] | null>(() => peek('services') ?? null);
  const [plug, setPlug] = useState<PluginRecord[] | null>(() => peek('plugins') ?? null);
  const [provs, setProvs] = useState<AiProviderPublic[] | null>(() => peek('providers') ?? null);
  const [apps, setApps] = useState<PairedApp[] | null>(() => peek('pairedApps') ?? null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(() => peek('runtimeStatus') ?? null);
  const [net, setNet] = useState<NetworkStatus | null>(() => peek('networkStatus') ?? null);
  // Shown until dismissed; the rows tick themselves, so it stays honest.
  const [guideOpen, setGuideOpen] = useState<boolean>(() => shouldAutoOpen());
  const [codexConnected, setCodexConnected] = useState(false);
  const [unavailable, setUnavailable] = useState<string[]>([]);
  const [failedSeen, setFailedSeen] = useState<number>(readFailedSeen);
  const [nodeBusy, setNodeBusy] = useState(false);
  const [nodeError, setNodeError] = useState<string | null>(null);
  const refreshing = useRef(false);

  const refresh = useCallback(async () => {
    if (refreshing.current) return;
    refreshing.current = true;
    // Each read is independent: one failing surface must not blank the others.
    const [s, p, a, c, r, cx, n] = await Promise.allSettled([
      services.list(),
      plugins.list(),
      aiProviders.list(),
      pairing.paired(),
      bridge.status(),
      codex.status(),
      // (OAIY Network's status: an older desktop, or a test's, may have none)
      (async () => network.status())(),
    ]);
    if (n.status === 'fulfilled') { setNet(n.value); put('networkStatus', n.value); }
    refreshing.current = false;
    setCodexConnected(cx.status === 'fulfilled' && cx.value.connected);
    const labels = ['services', 'plugins', 'AI providers', 'connections', 'runtime'];
    setUnavailable([s, p, a, c, r].flatMap((result, index) => result.status === 'rejected' ? [labels[index]] : []));
    if (s.status === 'fulfilled') {
      setSvc(s.value.services);
      put('services', s.value.services);
      // The WHOLE snapshot too, under the key ServicesPanel seeds from. Writing
      // only the array meant Overview's 4s poll never populated what Services
      // reads, so the first visit to Services still showed "Loading services…"
      // — the exact flash that seeding was meant to remove, surviving on the
      // one path everybody takes (the app opens on Overview).
      put('servicesSnapshot', s.value);
    }
    if (p.status === 'fulfilled') { setPlug(p.value.plugins); put('plugins', p.value.plugins); }
    if (a.status === 'fulfilled') { setProvs(a.value.providers); put('providers', a.value.providers); }
    if (c.status === 'fulfilled') { setApps(c.value.paired); put('pairedApps', c.value.paired); }
    if (r.status === 'fulfilled') { setRuntime(r.value); put('runtimeStatus', r.value); }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(() => { if (!document.hidden) void refresh(); }, POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const runningSvc = (svc ?? []).filter((s) => s.status === 'running').length;
  const runningPlug = (plug ?? []).filter((p) => p.state === 'running').length;
  // A plugin unhealthy only because its setup has not recorded what it needs is waiting for that setup (the guide
  // above offers it), not a fault to raise an alarm about.
  const setupState = useSetupState();
  const live = useLiveSetup(plug ?? null, setupState);
  const crashedPlug = (plug ?? []).filter(
    (p) => p.state === 'crashed' || (p.state === 'unhealthy' && !waitingForSetup(p, pluginSetupStatus(p, setupState, live))),
  );
  const readyProv = (provs ?? []).filter((p) => p.enabled && (p.hasKey || p.allowLocal)).length + (codexConnected ? 1 : 0);
  // The plugins' own cards, as the desktop serves them (none from a plugin turned off, or for a module that is off).
  const modules = useModules();
  const cards: OverviewContribution[] = modules?.contributions?.overview ?? [];
  const answers = usePluginPolls(modules?.contributions?.polls);
  const pluginById = new Map((plug ?? []).map((p) => [p.id, p]));
  const ctx = (card: OverviewContribution) => bindContext(pluginById.get(card.pluginId), answers, card.pluginId);
  /** Open a plugin page (`plugin:<pluginId>:<navId>`). */
  const openView = (view: string) => {
    const [kind, pluginId, navId] = view.split(':');
    if (kind === 'plugin' && pluginId && navId) onOpenPluginScreen(pluginId, navId);
  };
  const heroes = cards.filter((c) => c.kind === 'hero');
  const statuses = cards.filter((c) => c.kind === 'status');
  const tiles = cards.filter((c) => c.kind === 'tile');

  // The one-click Node fix — offered wherever the runtime is reported broken.
  const installNode =
    runtime?.nodeRuntime && !runtime.nodeRuntime.available ? (
      <span className="overview-fix">
        <button
          className="btn-tiny"
          disabled={nodeBusy || runtime.nodeRuntime.installing}
          onClick={() => {
            setNodeBusy(true);
            setNodeError(null);
            void nodeRuntime
              .install()
              .then(refresh)
              .catch((e: unknown) => setNodeError(`Node was not installed: ${e instanceof Error ? e.message.replace(/^\d{3}: /, '') : String(e)}`))
              .finally(() => setNodeBusy(false));
          }}
        >
          {nodeBusy || runtime.nodeRuntime.installing
            ? 'Installing Node…'
            : `Install Node ${runtime.nodeRuntime.installsVersion}`}
        </button>
        {nodeError && <small className="card-warn" role="alert">{nodeError}</small>}
      </span>
    ) : null;
  // The runtime's own step: Node installed in one click where that is what is missing, else read again (what is wrong
  // with it is said in the step; no other page fixes it).
  const runtimeFix = installNode ?? (
    <button className="btn-tiny" onClick={() => void refresh()}>
      <RotateCcw size={12} /> Check again
    </button>
  );

  // Failed runs since the person last looked (a run that failed last week is not news every time Overview opens).
  const failed = runtime?.runs.failed ?? 0;
  // (fewer than seen: the history was cleared since, and counts from nothing)
  const seen = failedSeen > failed ? 0 : failedSeen;
  const newFailed = failed - seen;
  const markFailedSeen = () => {
    try {
      localStorage.setItem(FAILED_SEEN, String(failed));
    } catch {
      /* storage can be unavailable: the warning comes back next time */
    }
    setFailedSeen(failed);
  };
  const plugName = (p: PluginRecord) => p.manifest?.name ?? p.id;

  return (
    <div className="panel">
      {unavailable.length > 0 && (
        <div className="banner banner-err banner-dismissable" role="status">
          <span>Couldn't refresh {unavailable.join(', ')}. Showing last known values where available.</span>
          <button className="btn" onClick={() => void refresh()}>Refresh status</button>
        </div>
      )}
      <UpdateBanner onOpenSettings={() => onNavigate('settings')} />
      <TodayPanel onNavigate={onNavigate} />
      {guideOpen && (
        <SetupGuidePanel
          codexConnected={codexConnected}
          runtime={runtime}
          providers={provs}
          services={svc}
          plugins={plug}
          connected={apps}
          live={live}
          actions={{ runtime: runtimeFix }}
          onNavigate={(t: StepTarget) => onNavigate(t === 'overview' ? 'runs' : t)}
          onDismiss={() => {
            dismissGuide();
            setGuideOpen(false);
          }}
        />
      )}

      {/* Plugin-contributed hero cards. A plugin declares these so it can own a
          spot on the control centre rather than hiding under Plugins. */}
      {heroes.map((card) => {
        const plugin = pluginById.get(card.pluginId);
        const running = plugin?.state === 'running';
        const Icon = cardIcon(card.icon);
        // The declared bindings (`$health.status`, `$poll.<card>.<path>`) are looked
        // up by bind.ts: the plugin's last health report while it runs, and its
        // status-card polls. What cannot be looked up falls back to what the host
        // itself knows, never to the raw "$health.status" text.
        const headline = bindText(card.bind.headline, ctx(card));
        const body = bindText(card.bind.body, ctx(card));
        // A headline that is the plugin's health status reads as one pill
        // ("Healthy"), in place of "running" beside a bare "ok".
        const health = running && card.bind.headline === '$health.status' ? healthPill(headline) : null;
        return (
          <div
            key={`${card.pluginId}-${card.id}`}
            className={`service-card overview-hero service-card-${running ? 'running' : 'stopped'}`}
          >
            <span className="overview-hero-icon" aria-hidden>
              <Icon size={16} />
            </span>
            <span className="overview-hero-text">
              <span className="overview-hero-title">
                <strong>{card.title}</strong>
                {plugin &&
                  (health ? (
                    <span className={`badge ${health.badge}`}>{health.label}</span>
                  ) : (
                    <span className={running ? 'badge badge-ok' : 'badge badge-neutral'}>{plugin.state}</span>
                  ))}
                {headline && !health && <span className="overview-hero-headline">{headline}</span>}
              </span>
              <small>
                {body ??
                  (running || !plugin
                    ? `From the ${card.pluginName} plugin.`
                    : plugin.reason ?? 'Start it from Connections, under Plugins.')}
              </small>
            </span>
            {card.cta && (
              <button className="btn btn-secondary" onClick={() => openView(card.cta!.view)}>
                {card.cta.label} <ChevronRight size={13} />
              </button>
            )}
          </div>
        );
      })}

      {/* Status rows: one line each, the value and what it means. */}
      {statuses.length > 0 && (
        <div className="overview-status-list">
          {statuses.map((card) => {
            const Icon = cardIcon(card.icon);
            const value = bindText(card.bind.value ?? card.bind.headline, ctx(card));
            const detail = bindText(card.bind.detail ?? card.bind.body, ctx(card));
            const target = card.view ?? card.cta?.view;
            return (
              <div key={`${card.pluginId}-${card.id}`} className="overview-status">
                <Icon size={14} aria-hidden />
                <strong>{card.title}</strong>
                <span className="overview-status-value">{value ?? '—'}</span>
                {detail && <small>{detail}</small>}
                {target && (
                  <button className="btn-tiny" onClick={() => openView(target)}>
                    {card.cta?.label ?? 'Open'} <ChevronRight size={12} />
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Tiles: a number or a word, and what it counts. */}
      {tiles.length > 0 && (
        <section className="service-section">
          <div className="section-title-row">
            <h3 className="section-title">From your plugins</h3>
          </div>
          <div className="overview-grid">
            {tiles.map((card) => {
              const Icon = cardIcon(card.icon);
              const value = bindText(card.bind.value ?? card.bind.headline, ctx(card));
              const inside = (
                <>
                  <Icon size={16} aria-hidden />
                  <strong>{value ?? '—'}</strong>
                  <small>{card.title}</small>
                </>
              );
              const key = `${card.pluginId}-${card.id}`;
              const label = `${card.title}: ${value ?? 'not known'}`;
              return card.view ? (
                <button key={key} className="overview-tile" aria-label={label} title={`From the ${card.pluginName} plugin`} onClick={() => openView(card.view!)}>
                  {inside}
                </button>
              ) : (
                <div key={key} role="group" className="overview-tile overview-tile-static" aria-label={label} title={`From the ${card.pluginName} plugin`}>
                  {inside}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Not inside "Next steps": that section yields to the setup guide, and a
          run that has ALREADY failed is not a setup step to work through. On a
          fresh install the guide is open — precisely when a failure is most
          likely and least excusable to hide. */}
      {newFailed > 0 && (
        <div className="banner banner-err banner-dismissable" role="alert">
          <span>
            <TriangleAlert size={13} />{' '}
            {seen > 0
              ? `${newFailed} more run${newFailed === 1 ? '' : 's'} failed since you last looked.`
              : `${failed} run${failed === 1 ? ' has' : 's have'} failed on this machine.`}
          </span>
          <span className="banner-actions">
            <button
              className="btn-tiny"
              onClick={() => {
                markFailedSeen();
                onNavigate('runs');
              }}
            >
              <History size={13} /> See why
            </button>
            <button className="banner-dismiss" aria-label="Dismiss: these failures are seen" title="Dismiss" onClick={markFailedSeen}>
              <X size={13} />
            </button>
          </span>
        </div>
      )}

      {crashedPlug.length > 0 && (
        <div className="banner banner-err banner-dismissable" role="alert">
          <span>
            <TriangleAlert size={13} />{' '}
            {crashedPlug.length === 1
              ? `${plugName(crashedPlug[0])} has ${crashedPlug[0].state === 'crashed' ? 'stopped' : 'a problem'}${crashedPlug[0].reason ? `: ${crashedPlug[0].reason}` : '.'}`
              : `${crashedPlug.length} plugins need attention: ${crashedPlug.map(plugName).join(', ')}.`}
          </span>
          <button className="btn-tiny" onClick={() => onNavigate('plugins')}>
            <Plug size={13} /> Open Plugins
          </button>
        </div>
      )}

      <section className="service-section">
        <div className="section-title-row">
          <h3 className="section-title">This machine</h3>
        </div>
        <div className="overview-grid">
          <button className="overview-tile" onClick={() => onNavigate('services')}>
            <Server size={16} aria-hidden />
            <strong>{svc === null ? '—' : `${runningSvc}/${svc.length}`}</strong>
            <small>Services running</small>
          </button>
          <button className="overview-tile" onClick={() => onNavigate('plugins')}>
            <Plug size={16} aria-hidden />
            <strong>{plug === null ? '—' : `${runningPlug}/${plug.length}`}</strong>
            <small>Plugins running</small>
          </button>
          <button className="overview-tile" onClick={() => onNavigate('providers')}>
            <Sparkles size={16} aria-hidden />
            <strong>{provs === null ? '—' : `${readyProv}/${provs.length + (codexConnected ? 1 : 0)}`}</strong>
            <small>AI providers ready</small>
          </button>
          <button className="overview-tile" onClick={() => onNavigate('connections')}>
            <ShieldCheck size={16} aria-hidden />
            <strong>{apps === null ? '—' : apps.length}</strong>
            <small>Connected apps</small>
          </button>
          {/* OAIY Network, once it is switched on: the workers answering (a primary), or whom this one lends to */}
          {net?.enabled && (
            <button className="overview-tile" onClick={() => onNavigate('network')}>
              <Network size={16} aria-hidden />
              <strong>{net.lends && net.waiting > 0 ? `${net.waiting} asking` : net.uses ? `${net.online ?? 0} online` : (net.usedBy ?? net.paired)}</strong>
              <small>
                {net.lends && net.waiting > 0
                  ? 'To pair with this computer'
                  : net.uses
                    ? `Of ${net.using ?? net.paired} paired computer${(net.using ?? net.paired) === 1 ? '' : 's'}`
                    : (net.usedBy ?? net.paired) === 1
                      ? 'Computer using this one'
                      : 'Computers using this one'}
              </small>
            </button>
          )}
        </div>
        <BackupLine onOpenSettings={() => onNavigate('settings')} />
      </section>

      {/* Anything actionable, rather than making the user hunt for it. While the
          setup guide is open it covers this same ground, so only one shows. */}
      {!guideOpen && (svc !== null || plug !== null || provs !== null) && (
        <section className="service-section">
          <div className="section-title-row">
            <h3 className="section-title">Next steps</h3>
          </div>
          {runtime && !runtime.ready && (
            <div className="datadir-note">
              <span>
                <TriangleAlert size={13} /> Flows cannot run on this machine —{' '}
                {runtime.flowRuntime.detail ?? 'the OAIY runtime is unavailable.'}
              </span>
              {/* When the only thing missing is Node, fixing it is one click —
                  don't make the user go and install a runtime by hand. */}
              {installNode}
            </div>
          )}
          {provs !== null && provs.length === 0 && !codexConnected && (
            <div className="datadir-note">
              <span>No AI provider configured — flows have no chat model to call.</span>
              <button className="btn-tiny" onClick={() => onNavigate('providers')}>
                <Sparkles size={13} /> Add a provider
              </button>
            </div>
          )}
          {plug !== null && plug.length === 0 && (
            <div className="datadir-note">
              <span>No plugins installed — connectors and device events come from plugins.</span>
              <button className="btn-tiny" onClick={() => onNavigate('plugins')}>
                <Plug size={13} /> Open Plugins
              </button>
            </div>
          )}
          {plug !== null && plug.length > 0 && runningPlug === 0 && (
            <div className="datadir-note">
              <span>No plugin is running — their connectors are unavailable to flows.</span>
              <button className="btn-tiny" onClick={() => onNavigate('plugins')}>
                <Plug size={13} /> Start one
              </button>
            </div>
          )}
          {svc !== null && svc.length > 0 && runningSvc === 0 && (
            <div className="datadir-note">
              <span>No local service is running — local models are offline.</span>
              <button className="btn-tiny" onClick={() => onNavigate('services')}>
                <Server size={13} /> Open Services
              </button>
            </div>
          )}
          {provs !== null &&
            provs.length > 0 &&
            readyProv === 0 && (
              <div className="datadir-note">
                <span>Every AI provider is disabled or missing a key.</span>
                <button className="btn-tiny" onClick={() => onNavigate('providers')}>
                  <Sparkles size={13} /> Fix providers
                </button>
              </div>
            )}
          {/* The all-clear, so the screen is never ambiguous about being fine. */}
          {runtime?.ready &&
            provs !== null &&
            plug !== null &&
            svc !== null &&
            readyProv > 0 &&
            crashedPlug.length === 0 &&
            (plug.length === 0 || runningPlug > 0) && (
              <div className="datadir-note">
                <span>
                  <CircleDot size={13} /> Ready — {readyProv} provider{readyProv === 1 ? '' : 's'} and{' '}
                  {runningPlug} plugin{runningPlug === 1 ? '' : 's'} available to your flows.
                </span>
              </div>
            )}
        </section>
      )}

      {!guideOpen && (
        <div className="setup-reopen">
          <button
            className="btn-tiny"
            onClick={() => {
              reopenGuide();
              setGuideOpen(true);
            }}
          >
            <ListChecks size={13} /> Show the setup guide
          </button>
        </div>
      )}

      {/* Only while nothing at all is installed: a service counts too (its tile above already shows it). */}
      {plug !== null && plug.length === 0 && svc !== null && svc.length === 0 && cards.length === 0 && (
        <div className="empty-state">
          <Package size={22} style={{ opacity: 0.5 }} />
          <p>Nothing installed yet.</p>
          <p style={{ fontSize: 13, opacity: 0.7 }}>
            Install a plugin, such as the phone, or a local service, and what it brings shows up here.
          </p>
        </div>
      )}
    </div>
  );
}
