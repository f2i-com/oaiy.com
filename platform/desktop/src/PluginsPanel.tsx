import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CircleDot,
  Loader2,
  Play,
  Square,
  ScrollText,
  Plug,
  TriangleAlert,
  FolderOpen,
  Trash2,
  Boxes,
  ListChecks,
} from 'lucide-react';
import {
  appConfig,
  isTauri,
  plugins,
  openInExplorer,
  type PluginRecord,
  type PluginState,
  type PluginsSnapshot,
  serviceDefinitions,
  type ServiceDefinition,
} from './api';
import { refetchModules } from './useModules';
import { pluginSetupStatus, readSetup, type PluginSetupStatus } from './setupFlow';
import { useLiveSetup } from './useLiveSetup';
import { openSetup, refetchSetup, useSetupState } from './useSetupState';
import { peek, put } from './useCached';
import { useToast } from './Toasts';
import LogsViewer from './LogsViewer';

/**
 * Plugins panel — install, start/stop, and monitor Bridge-Protocol plugins.
 *
 * This is the GUI over the plugin host: `GET /api/plugins` and the
 * start/stop/enabled/logs routes. It exists because the backend can supervise
 * plugins but a user could not SEE them — a plugin dropped into the plugins
 * folder was invisible until something touched the API. The panel polls every
 * 2s so a crash → restart, or a health transition, shows without a manual
 * refresh, mirroring the Services panel.
 */

const POLL_MS = 2000;

/** Maps a plugin state to the same status vocabulary the CSS already styles. */
function badgeClass(state: PluginState): string {
  switch (state) {
    case 'running':
      return 'badge badge-ok';
    case 'starting':
      return 'badge badge-pending';
    case 'unhealthy':
    case 'crashed':
      return 'badge badge-err';
    default:
      return 'badge badge-neutral';
  }
}

/** A plugin the host cannot run — no manifest — never becomes startable. */
function isLoadable(p: PluginRecord): boolean {
  return p.manifest !== undefined;
}

export default function PluginsPanel() {
  const toast = useToast();
  // Last known snapshot, so a revisit renders the plugin list immediately.
  const [snapshot, setSnapshot] = useState<PluginsSnapshot | null>(() => peek('pluginsSnapshot') ?? null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [logsFor, setLogsFor] = useState<string | null>(null);
  /** Path to a plugin folder or .tar.gz to install. */
  const [installSource, setInstallSource] = useState('');
  const [installing, setInstalling] = useState(false);
  /** Action surfaces the installed plugins contribute. */
  const [definitions, setDefinitions] = useState<ServiceDefinition[]>(() => peek('definitions') ?? []);
  // Track state per plugin so we toast on a transition (crash, came up),
  // not on every poll — same pattern the Services panel uses.
  const seen = useRef<Map<string, PluginState>>(new Map());
  const firstPoll = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const snap = await plugins.list();
      setError(null);
      for (const p of snap.plugins) {
        const prev = seen.current.get(p.id);
        if (prev !== undefined && prev !== p.state && !firstPoll.current) {
          if (p.state === 'crashed') {
            toast.push({
              kind: 'error',
              title: `Plugin "${p.id}" crashed`,
              body: p.reason ?? 'The process exited unexpectedly.',
            });
          } else if (p.state === 'running' && prev !== 'running') {
            toast.push({ kind: 'success', title: `Plugin "${p.id}" is running` });
          } else if (p.state === 'unhealthy') {
            toast.push({
              kind: 'error',
              title: `Plugin "${p.id}" is unhealthy`,
              body: p.reason ?? 'Health probes are failing.',
            });
          }
        }
        seen.current.set(p.id, p.state);
      }
      firstPoll.current = false;
      setSnapshot(snap);
      put('pluginsSnapshot', snap);
      put('plugins', snap.plugins);
      try {
        const defs = (await serviceDefinitions.list()).definitions;
        setDefinitions(defs);
        put('definitions', defs);
      } catch {
        /* a definition listing failure must not blank the plugin list */
      }
    } catch (e) {
      // A poll failure is usually a momentary blip; keep the last snapshot and
      // show the error rather than blanking the list.
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [toast]);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const runAction = useCallback(
    async (id: string, fn: () => Promise<unknown>) => {
      setPending((s) => new Set(s).add(id));
      try {
        await fn();
        // Turning a plugin on or off, or removing it, can bring or take the phone and the calendar.
        void refetchModules();
        await refresh();
      } catch (e) {
        toast.push({
          kind: 'error',
          title: `Action failed for "${id}"`,
          body: e instanceof Error ? e.message : String(e),
        });
      } finally {
        setPending((s) => {
          const next = new Set(s);
          next.delete(id);
          return next;
        });
      }
    },
    [refresh, toast],
  );

  const list = useMemo(() => snapshot?.plugins ?? [], [snapshot]);
  const setupState = useSetupState();
  // A plugin set up by hand (or before the wizard) counts as set up once its steps' checks all pass.
  const live = useLiveSetup(snapshot?.plugins ?? null, setupState);

  const installPlugin = useCallback(async () => {
    const source = installSource.trim();
    if (!source) return;
    setInstalling(true);
    try {
      const out = await plugins.install(source);
      // A new plugin with a setup wizard: its wizard opens now. An update whose
      // setup went up gets a nudge on its card instead (it was set up before).
      const opensSetup = !!out.setup && !out.replaced;
      toast.push({
        kind: 'success',
        title: out.replaced
          ? `Updated ${out.name} to v${out.version}`
          : `Installed ${out.name} v${out.version}`,
        body: opensSetup ? 'Its setup opens now.' : out.setup ? 'If its setup has something new, its card says so.' : 'Click Start to run it.',
      });
      setInstallSource('');
      void refetchModules();
      await Promise.all([refresh(), refetchSetup()]);
      if (opensSetup) openSetup({ plugin: out.id });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setInstalling(false);
    }
  }, [installSource, refresh, toast]);

  const uninstallPlugin = useCallback(
    async (p: PluginRecord) => {
      const name = p.manifest?.name ?? p.id;
      if (!confirm(`Remove the “${name}” plugin? Its files are deleted from this machine.`)) return;
      await runAction(p.id, async () => {
        await plugins.uninstall(p.id);
        toast.push({ kind: 'success', title: `Removed ${name}` });
      });
    },
    [runAction, toast],
  );

  return (
    <div className="panel">
      {error && (
        <div className="banner banner-err banner-dismissable">
          <span>Couldn't reach the plugin host: {error}</span>
          <button className="banner-dismiss" onClick={() => setError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      {snapshot && (
        <div className="datadir-note datadir-row">
          <span>
            Plugins live in <code>{snapshot.root}</code>.
          </span>
          <button
            className="btn-tiny"
            onClick={() => openInExplorer(snapshot.root).catch(() => {})}
          >
            <FolderOpen size={13} /> Open folder
          </button>
        </div>
      )}

      {/* Install from a path on this machine. Until this existed the only way to
          add a plugin was to copy a folder in by hand — and every connector OAIY
          offers a flow comes from a plugin. */}
      <section className="service-section">
        <div className="section-title-row">
          <h3 className="section-title">Install a plugin</h3>
        </div>
        <form
          className="dl-form"
          onSubmit={(e) => {
            e.preventDefault();
            void installPlugin();
          }}
        >
          <label className="form-row">
            <span>Plugin folder, .zip or .tar.gz on this machine</span>
            <span className="setup-folder-row">
              <input
                type="text"
                placeholder="C:\path\to\my-plugin"
                value={installSource}
                onChange={(e) => setInstallSource(e.target.value)}
              />
              {isTauri() && (
                <button
                  type="button"
                  className="btn"
                  onClick={() => void appConfig.pickFolder().then((p) => p && setInstallSource(p)).catch(() => {})}
                >
                  <FolderOpen size={14} /> Browse
                </button>
              )}
            </span>
          </label>
          <p className="form-hint">
            Installing a plugin installs native code this app will run. Only install plugins you
            trust. Re-installing over an existing plugin updates it in place.
          </p>
          <div className="form-actions">
            <button
              className="btn btn-primary"
              type="submit"
              disabled={installing || !installSource.trim()}
            >
              {installing ? <Loader2 size={14} className="spin" /> : <Plug size={14} />} Install
            </button>
          </div>
        </form>
      </section>

      {!snapshot ? (
        <div className="empty-state">Loading plugins…</div>
      ) : list.length === 0 ? (
        <div className="empty-state">
          <Plug size={22} style={{ opacity: 0.5 }} />
          <p>No plugins installed.</p>
          <p style={{ fontSize: 13, opacity: 0.7 }}>
            A plugin is a folder with a <code>manifest.json</code> and an executable, placed in the
            plugins directory above. It runs supervised and speaks the Bridge Protocol, contributing
            connectors and events to your flows.
          </p>
        </div>
      ) : (
        <section className="service-section">
          {list.map((p) => (
            <PluginCard
              key={p.id}
              plugin={p}
              setupTitle={readSetup(p)?.title ?? null}
              setupStatus={p.userDisabled ? 'none' : pluginSetupStatus(p, setupState, live)}
              onSetup={() => openSetup({ plugin: p.id })}
              pending={pending.has(p.id)}
              onStart={() => runAction(p.id, () => plugins.start(p.id))}
              onStop={() => runAction(p.id, () => plugins.stop(p.id))}
              onToggleEnabled={() =>
                runAction(p.id, () => plugins.setEnabled(p.id, p.userDisabled))
              }
              onViewLogs={() => setLogsFor(p.id)}
              onUninstall={() => void uninstallPlugin(p)}
            />
          ))}
        </section>
      )}

      {/* Services a plugin contributes: named, schema'd actions a flow can call.
          Installed and removed with the plugin, so they belong here rather than
          alongside the managed local services. */}
      {definitions.length > 0 && (
        <section className="service-section">
          <div className="section-title-row">
            <h3 className="section-title">Services from plugins</h3>
          </div>
          {definitions.map((d) => (
            <div key={d.id} className="service-card service-card-running">
              <div className="card-head">
                <Boxes size={14} aria-hidden />
                <strong className="card-title">{d.name}</strong>
                <span className="badge badge-neutral">{d.category ?? 'service'}</span>
                <span className="card-note">
                  from <code>{d.pluginId}</code>
                </span>
              </div>
              {d.description && (
                <p className="card-desc">{d.description}</p>
              )}
              <div className="card-meta">
                {d.actions.length} action{d.actions.length === 1 ? '' : 's'}:{' '}
                {d.actions.map((a) => a.id).join(', ')}
              </div>
            </div>
          ))}
        </section>
      )}

      {logsFor && (
        <LogsViewer
          title={`${logsFor} — logs`}
          onClose={() => setLogsFor(null)}
          load={async () => (await plugins.logs(logsFor, 300)).lines}
        />
      )}
    </div>
  );
}

interface CardProps {
  plugin: PluginRecord;
  /** Its setup wizard's title, when it declares one. */
  setupTitle: string | null;
  /**
   * `needs-setup`: its setup was never finished and its checks do not all
   * pass, or its setup version went up (a nudge, not a forced wizard);
   * `set-up`: finished here, or everything its checks look at is true now.
   */
  setupStatus: PluginSetupStatus;
  onSetup: () => void;
  pending: boolean;
  onStart: () => void;
  onStop: () => void;
  onToggleEnabled: () => void;
  onViewLogs: () => void;
  onUninstall: () => void;
}

function PluginCard({
  plugin: p,
  setupTitle,
  setupStatus,
  onSetup,
  pending,
  onStart,
  onStop,
  onToggleEnabled,
  onViewLogs,
  onUninstall,
}: CardProps) {
  const loadable = isLoadable(p);
  const needsSetup = setupStatus === 'needs-setup';
  const running = p.state === 'running' || p.state === 'unhealthy' || p.state === 'starting';
  const connectorCount =
    p.manifest?.connectors?.reduce((n, c) => n + c.commands.length, 0) ?? 0;

  return (
    <div className={`service-card service-card-${cardStatus(p.state)}`}>
      <div className="card-head">
        <CircleDot size={14} aria-hidden />
        <strong className="card-title">{p.manifest?.name ?? p.id}</strong>
        <span className={badgeClass(p.state)}>{p.state}</span>
        {p.manifest?.version && (
          <span className="card-note">v{p.manifest.version}</span>
        )}
        {p.userDisabled && <span className="badge badge-neutral">disabled by you</span>}
        {setupTitle && setupStatus === 'set-up' && (
          <span className="badge badge-ok" title={`${setupTitle}: done`}>
            set up
          </span>
        )}
      </div>

      {p.manifest?.description && (
        <p className="card-desc">{p.manifest.description}</p>
      )}

      {/* Dropped in by hand, or updated with more to set up: a nudge, never a forced wizard. */}
      {needsSetup && setupTitle && (
        <div className="setup-nudge">
          <ListChecks size={14} aria-hidden />
          <span>Finish setting up {p.manifest?.name ?? p.id}.</span>
          <button className="btn btn-primary" onClick={onSetup} title={setupTitle}>
            Set up…
          </button>
        </div>
      )}

      {/* Every non-running state carries a reason the host wrote — surface it,
          because "it won't start" with no cause is the least useful state. */}
      {p.state !== 'running' && p.reason && (
        <p className="card-reason">{p.reason}</p>
      )}

      {loadable && (
        <div className="card-meta">
          {connectorCount > 0 && `${connectorCount} command${connectorCount === 1 ? '' : 's'}`}
          {connectorCount > 0 && (p.manifest?.events?.length ?? 0) > 0 && ' · '}
          {(p.manifest?.events?.length ?? 0) > 0 &&
            `${p.manifest!.events!.length} event${p.manifest!.events!.length === 1 ? '' : 's'}`}
        </div>
      )}

      {/* Legacy + unknown capabilities are surfaced, not hidden: a plugin using a
          pre-OAIY name, or asking for something OAIY has no equivalent for, is a
          real fact an operator wants before a mystery denial. */}
      {p.legacyCapabilities && p.legacyCapabilities.length > 0 && (
        <p className="card-meta">
          Uses legacy capability names: {p.legacyCapabilities.map(([o]) => o).join(', ')}
        </p>
      )}
      {p.unknownCapabilities && p.unknownCapabilities.length > 0 && (
        <p className="card-meta card-warn">
          <TriangleAlert size={12} /> Asks for {p.unknownCapabilities.join(', ')}, which this OAIY
          build does not provide.
        </p>
      )}

      <div className="card-actions">
        {pending ? (
          <button className="btn btn-secondary" disabled>
            <Loader2 size={14} className="spin" /> Working…
          </button>
        ) : running ? (
          <button className="btn btn-secondary" onClick={onStop}>
            <Square size={14} /> Stop
          </button>
        ) : (
          <button
            className="btn btn-primary"
            onClick={onStart}
            disabled={!loadable || p.userDisabled}
            title={
              !loadable
                ? 'This plugin cannot start — its manifest is invalid.'
                : p.userDisabled
                  ? 'Turned off. Enable it first.'
                  : undefined
            }
          >
            <Play size={14} /> Start
          </button>
        )}

        {setupTitle && !needsSetup && (
          <button className="btn btn-ghost" onClick={onSetup} title={setupTitle}>
            <ListChecks size={14} /> {setupStatus === 'set-up' ? 'Its setup' : 'Set up…'}
          </button>
        )}

        <button className="btn btn-ghost" onClick={onViewLogs}>
          <ScrollText size={14} /> Logs
        </button>

        {loadable && (
          <button className="btn btn-ghost" onClick={onToggleEnabled}>
            {p.userDisabled ? 'Enable' : 'Disable'}
          </button>
        )}

        <button
          className="btn btn-ghost btn-danger"
          onClick={onUninstall}
          disabled={pending}
          aria-label={`Remove the ${p.manifest?.name ?? p.id} plugin`}
        >
          <Trash2 size={14} /> Remove
        </button>
      </div>
    </div>
  );
}

/** Map a plugin state onto a `service-card-<x>` class the CSS already colours. */
function cardStatus(state: PluginState): string {
  switch (state) {
    case 'running':
      return 'running';
    case 'starting':
      return 'starting';
    case 'crashed':
    case 'unhealthy':
      return 'errored';
    default:
      return 'stopped';
  }
}
