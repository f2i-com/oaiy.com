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
} from 'lucide-react';
import {
  plugins,
  openInExplorer,
  type PluginRecord,
  type PluginState,
  type PluginsSnapshot,
  serviceDefinitions,
  type ServiceDefinition,
} from './api';
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
  const [snapshot, setSnapshot] = useState<PluginsSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [logsFor, setLogsFor] = useState<string | null>(null);
  /** Path to a plugin folder or .tar.gz to install. */
  const [installSource, setInstallSource] = useState('');
  const [installing, setInstalling] = useState(false);
  /** Action surfaces the installed plugins contribute. */
  const [definitions, setDefinitions] = useState<ServiceDefinition[]>([]);
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
      try {
        setDefinitions((await serviceDefinitions.list()).definitions);
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

  const installPlugin = useCallback(async () => {
    const source = installSource.trim();
    if (!source) return;
    setInstalling(true);
    try {
      const out = await plugins.install(source);
      toast.push({
        kind: 'success',
        title: out.replaced
          ? `Updated ${out.name} to v${out.version}`
          : `Installed ${out.name} v${out.version}`,
        body: 'Click Start to run it.',
      });
      setInstallSource('');
      await refresh();
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
        <div className="datadir-note">
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
            <span>Plugin folder or .tar.gz on this machine</span>
            <input
              type="text"
              placeholder="C:\path\to\my-plugin"
              value={installSource}
              onChange={(e) => setInstallSource(e.target.value)}
            />
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
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <Boxes size={14} aria-hidden />
                <strong style={{ fontSize: 15 }}>{d.name}</strong>
                <span className="badge badge-neutral">{d.category ?? 'service'}</span>
                <span style={{ fontSize: 12, opacity: 0.6 }}>
                  from <code>{d.pluginId}</code>
                </span>
              </div>
              {d.description && (
                <p style={{ fontSize: 13, opacity: 0.8, margin: '6px 0 0' }}>{d.description}</p>
              )}
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
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
  pending: boolean;
  onStart: () => void;
  onStop: () => void;
  onToggleEnabled: () => void;
  onViewLogs: () => void;
  onUninstall: () => void;
}

function PluginCard({
  plugin: p,
  pending,
  onStart,
  onStop,
  onToggleEnabled,
  onViewLogs,
  onUninstall,
}: CardProps) {
  const loadable = isLoadable(p);
  const running = p.state === 'running' || p.state === 'unhealthy' || p.state === 'starting';
  const connectorCount =
    p.manifest?.connectors?.reduce((n, c) => n + c.commands.length, 0) ?? 0;

  return (
    <div className={`service-card service-card-${cardStatus(p.state)}`}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <CircleDot size={14} aria-hidden />
        <strong style={{ fontSize: 15 }}>{p.manifest?.name ?? p.id}</strong>
        <span className={badgeClass(p.state)}>{p.state}</span>
        {p.manifest?.version && (
          <span style={{ fontSize: 12, opacity: 0.6 }}>v{p.manifest.version}</span>
        )}
        {p.userDisabled && <span className="badge badge-neutral">disabled by you</span>}
      </div>

      {p.manifest?.description && (
        <p style={{ fontSize: 13, opacity: 0.8, margin: '6px 0 0' }}>{p.manifest.description}</p>
      )}

      {/* Every non-running state carries a reason the host wrote — surface it,
          because "it won't start" with no cause is the least useful state. */}
      {p.state !== 'running' && p.reason && (
        <p style={{ fontSize: 12.5, opacity: 0.75, margin: '6px 0 0' }}>{p.reason}</p>
      )}

      {loadable && (
        <div style={{ fontSize: 12, opacity: 0.6, marginTop: 6 }}>
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
        <p style={{ fontSize: 12, opacity: 0.7, margin: '4px 0 0' }}>
          Uses legacy capability names: {p.legacyCapabilities.map(([o]) => o).join(', ')}
        </p>
      )}
      {p.unknownCapabilities && p.unknownCapabilities.length > 0 && (
        <p style={{ fontSize: 12, color: 'var(--warn, #b7791f)', margin: '4px 0 0', display: 'flex', gap: 6, alignItems: 'center' }}>
          <TriangleAlert size={12} /> Asks for {p.unknownCapabilities.join(', ')}, which this OAIY
          build does not provide.
        </p>
      )}

      <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
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
