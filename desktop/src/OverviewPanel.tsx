import { useCallback, useEffect, useState } from 'react';
import {
  ChevronRight,
  CircleDot,
  ListChecks,
  Package,
  Plug,
  Server,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
} from 'lucide-react';
import { peek, put } from './useCached';
import SetupGuidePanel from './SetupGuidePanel';
import { dismissGuide, reopenGuide, shouldAutoOpen } from './setupGuide';
import type { StepTarget } from './setupGuide';
import {
  aiProviders,
  bridge,
  nodeRuntime,
  type RuntimeStatus,
  pairing,
  plugins,
  services,
  type AiProviderPublic,
  type PairedApp,
  type PluginRecord,
  type ServiceSnapshot,
} from './api';

/**
 * Overview — the control centre. One screen that answers "is this machine ready
 * to run my flows?" without clicking through five panels: what's running, what
 * needs attention, and which apps are connected.
 *
 * Plugins can contribute their own hero card here (`manifest.ui.overview`), so a
 * plugin like the Aokie phone bridge can surface its own status and a deep link
 * into its screen instead of being invisible until you open Plugins.
 */

const POLL_MS = 4000;

export type OverviewNav =
  | 'services'
  | 'plugins'
  | 'models'
  | 'providers'
  | 'python'
  | 'connections'
  | 'settings';

interface Props {
  /** Jump to one of the built-in panels. */
  onNavigate: (view: OverviewNav) => void;
  /** Open a plugin-contributed screen (pluginId, navId). */
  onOpenPluginScreen: (pluginId: string, navId: string) => void;
}

interface OverviewCard {
  id?: string;
  kind?: string;
  title?: string;
  icon?: string;
  bind?: {
    headline?: string;
    body?: string;
    cta?: { label?: string; nav?: string };
  };
}

/** The `ui.overview` cards a plugin contributes, paired with the plugin itself. */
function overviewCards(list: PluginRecord[]): Array<{ plugin: PluginRecord; card: OverviewCard }> {
  const out: Array<{ plugin: PluginRecord; card: OverviewCard }> = [];
  for (const p of list) {
    const ui = (p.manifest as unknown as { ui?: { overview?: OverviewCard[] } } | undefined)?.ui;
    for (const card of ui?.overview ?? []) out.push({ plugin: p, card });
  }
  return out;
}

/** A plugin's `ui.nav` id for a contributed screen, if it declares one. */
function navIdFor(p: PluginRecord, wanted?: string): string | null {
  const ui = (p.manifest as unknown as { ui?: { nav?: Array<{ id?: string }> } } | undefined)?.ui;
  const nav = ui?.nav ?? [];
  if (wanted && nav.some((n) => n.id === wanted)) return wanted;
  return nav[0]?.id ?? null;
}

export default function OverviewPanel({ onNavigate, onOpenPluginScreen }: Props) {
  // Seeded from the last known values so returning to Overview paints straight
  // away instead of showing four em-dashes while the round trips land.
  const [svc, setSvc] = useState<ServiceSnapshot[] | null>(() => peek('services') ?? null);
  const [plug, setPlug] = useState<PluginRecord[] | null>(() => peek('plugins') ?? null);
  const [provs, setProvs] = useState<AiProviderPublic[] | null>(() => peek('providers') ?? null);
  const [apps, setApps] = useState<PairedApp[] | null>(() => peek('pairedApps') ?? null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(() => peek('runtimeStatus') ?? null);
  // Shown until dismissed; the rows tick themselves, so it stays honest.
  const [guideOpen, setGuideOpen] = useState<boolean>(() => shouldAutoOpen());

  const refresh = useCallback(async () => {
    // Each read is independent: one failing surface must not blank the others.
    const [s, p, a, c, r] = await Promise.allSettled([
      services.list(),
      plugins.list(),
      aiProviders.list(),
      pairing.paired(),
      bridge.status(),
    ]);
    if (s.status === 'fulfilled') { setSvc(s.value.services); put('services', s.value.services); }
    if (p.status === 'fulfilled') { setPlug(p.value.plugins); put('plugins', p.value.plugins); }
    if (a.status === 'fulfilled') { setProvs(a.value.providers); put('providers', a.value.providers); }
    if (c.status === 'fulfilled') { setApps(c.value.paired); put('pairedApps', c.value.paired); }
    if (r.status === 'fulfilled') { setRuntime(r.value); put('runtimeStatus', r.value); }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const runningSvc = (svc ?? []).filter((s) => s.status === 'running').length;
  const runningPlug = (plug ?? []).filter((p) => p.state === 'running').length;
  const crashedPlug = (plug ?? []).filter((p) => p.state === 'crashed' || p.state === 'unhealthy');
  const readyProv = (provs ?? []).filter((p) => p.enabled && (p.hasKey || p.allowLocal)).length;
  const cards = overviewCards(plug ?? []);

  // The one-click Node fix — offered wherever the runtime is reported broken.
  const installNode =
    runtime?.nodeRuntime && !runtime.nodeRuntime.available ? (
      <button
        className="btn-tiny"
        disabled={runtime.nodeRuntime.installing}
        onClick={() => void nodeRuntime.install().then(refresh).catch(() => {})}
      >
        {runtime.nodeRuntime.installing
          ? 'Installing Node…'
          : `Install Node ${runtime.nodeRuntime.installsVersion}`}
      </button>
    ) : null;

  return (
    <div className="panel">
      {guideOpen && (
        <SetupGuidePanel
          runtime={runtime}
          providers={provs}
          services={svc}
          plugins={plug}
          connected={apps}
          actions={installNode ? { runtime: installNode } : undefined}
          onNavigate={(t: StepTarget) => onNavigate(t === 'overview' ? 'services' : t)}
          onDismiss={() => {
            dismissGuide();
            setGuideOpen(false);
          }}
        />
      )}

      {/* Plugin-contributed hero cards. A plugin declares these so it can own a
          spot on the control centre rather than hiding under Plugins. */}
      {cards.map(({ plugin, card }) => {
        const running = plugin.state === 'running';
        const navId = navIdFor(plugin, card.bind?.cta?.nav);
        return (
          <div
            key={`${plugin.id}-${card.id ?? 'card'}`}
            className={`service-card service-card-${running ? 'running' : 'stopped'}`}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <Sparkles size={15} aria-hidden />
              <strong style={{ fontSize: 15 }}>{card.title ?? plugin.manifest?.name ?? plugin.id}</strong>
              <span className={running ? 'badge badge-ok' : 'badge badge-neutral'}>{plugin.state}</span>
            </div>
            {/* The declared bindings reference the plugin's own health feed; until
                that is wired, report the state the host actually knows rather than
                echoing an unresolved "$health.status" placeholder. */}
            {!running && plugin.reason && (
              <p style={{ fontSize: 12.5, opacity: 0.75, margin: '6px 0 0' }}>{plugin.reason}</p>
            )}
            {navId && (
              <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                <button className="btn btn-secondary" onClick={() => onOpenPluginScreen(plugin.id, navId)}>
                  {card.bind?.cta?.label ?? `Open ${card.title ?? plugin.id}`} <ChevronRight size={13} />
                </button>
              </div>
            )}
          </div>
        );
      })}

      {crashedPlug.length > 0 && (
        <div className="banner banner-err" role="alert">
          <span>
            <TriangleAlert size={13} />{' '}
            {crashedPlug.length === 1
              ? `Plugin "${crashedPlug[0].id}" is ${crashedPlug[0].state}.`
              : `${crashedPlug.length} plugins need attention.`}{' '}
            {crashedPlug[0].reason ?? ''}
          </span>
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
            <strong>{provs === null ? '—' : `${readyProv}/${provs.length}`}</strong>
            <small>AI providers ready</small>
          </button>
          <button className="overview-tile" onClick={() => onNavigate('connections')}>
            <ShieldCheck size={16} aria-hidden />
            <strong>{apps === null ? '—' : apps.length}</strong>
            <small>Connected apps</small>
          </button>
        </div>
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
          {provs !== null && provs.length === 0 && (
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

      {apps !== null && apps.length > 0 && (
        <section className="service-section">
          <div className="section-title-row">
            <h3 className="section-title">Connected apps</h3>
          </div>
          <ul className="pairing-list">
            {apps.map((a) => (
              <li key={a.id}>
                <span>
                  {a.label ?? a.product} <span style={{ opacity: 0.55 }}>({a.product})</span>
                </span>
                <span className="badge badge-ok">paired</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {plug !== null && plug.length === 0 && cards.length === 0 && (
        <div className="empty-state">
          <Package size={22} style={{ opacity: 0.5 }} />
          <p>Nothing installed yet.</p>
          <p style={{ fontSize: 13, opacity: 0.7 }}>
            Install a local service or a plugin and it shows up here.
          </p>
        </div>
      )}
    </div>
  );
}
