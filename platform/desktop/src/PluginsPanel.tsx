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
  ShieldCheck,
} from 'lucide-react';
import {
  appConfig,
  isTauri,
  plugins,
  openInExplorer,
  setup as setupApi,
  type CatalogPlugin,
  type PluginRecord,
  type PluginState,
  type PluginsSnapshot,
  serviceDefinitions,
  type ServiceDefinition,
} from './api';
import { refetchModules } from './useModules';
import { pluginSetupStatus, readSetup, waitingForSetup, type PluginSetupStatus } from './setupFlow';
import { useLiveSetup } from './useLiveSetup';
import { openSetup, refetchSetup, useSetupState } from './useSetupState';
import { peek, put } from './useCached';
import { useToast } from './Toasts';
import { ask } from './ask';
import { useVisiblePoll } from './useVisiblePoll';
import LogsViewer from './LogsViewer';
import PackageTrustBadge from './PackageTrustBadge';
import LoadError from './LoadError';
import { trustQuestion } from './PluginRunNote';
import { catalogNote } from './setupFlow';

/** An example of the path the install box takes, as this system writes one (it said C:\path\to\my-plugin on Linux too). */
const PLUGIN_PATH_EXAMPLE = /Windows/.test(navigator.userAgent) ? 'C:\\path\\to\\my-plugin' : '/path/to/my-plugin';

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
  /** Why the last install failed (its own: the list's poll clears the host's error, not this). */
  const [installError, setInstallError] = useState<string | null>(null);
  /** Action surfaces the installed plugins contribute. */
  const [definitions, setDefinitions] = useState<ServiceDefinition[]>(() => peek('definitions') ?? []);
  // Track state per plugin so we toast on a transition (crash, came up),
  // not on every poll — same pattern the Services panel uses.
  const seen = useRef<Map<string, PluginState>>(new Map());
  const firstPoll = useRef(true);
  // The setup state as last rendered, for the poll: a plugin waiting for its setup is not toasted as unhealthy.
  const setupInfo = useRef<{ state: ReturnType<typeof useSetupState>; live: ReturnType<typeof useLiveSetup> }>({ state: null, live: {} });

  // (the latest reading wins: an older one landing after a newer one would toast a crash, then a recovery, that
  // did not happen)
  const asked = useRef(0);
  const refresh = useCallback(async () => {
    const mine = ++asked.current;
    try {
      const snap = await plugins.list();
      if (mine !== asked.current) return;
      setError(null);
      for (const p of snap.plugins) {
        const shown = p.manifest?.name ?? p.declared?.name ?? p.id;
        const prev = seen.current.get(p.id);
        if (prev !== undefined && prev !== p.state && !firstPoll.current) {
          if (p.state === 'crashed') {
            toast.push({
              kind: 'error',
              title: `${shown} crashed`,
              body: p.reason ?? 'The process exited unexpectedly.',
            });
          } else if (p.state === 'running' && prev !== 'running') {
            toast.push({ kind: 'success', title: `${shown} is running` });
          } else if (p.state === 'unhealthy' && !waitingForSetup(p, pluginSetupStatus(p, setupInfo.current.state, setupInfo.current.live))) {
            toast.push({
              kind: 'error',
              title: `${shown} is unhealthy`,
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
      if (mine === asked.current) setError(e instanceof Error ? e.message : String(e));
    }
  }, [toast]);

  // (not while the window is hidden in the tray)
  useVisiblePoll(() => void refresh(), POLL_MS);

  /** Do `fn` for plugin `id`; `failed` is what a toast says when it does not work ("eGPU did not start"). */
  const runAction = useCallback(
    async (id: string, failed: string, fn: () => Promise<unknown>) => {
      setPending((s) => new Set(s).add(id));
      try {
        await fn();
        // Turning a plugin on or off, or removing it, can bring or take the phone and the calendar.
        void refetchModules();
        await refresh();
      } catch (e) {
        toast.push({
          kind: 'error',
          title: failed,
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
  setupInfo.current = { state: setupState, live };

  const installPlugin = useCallback(async () => {
    const source = installSource.trim();
    if (!source) return;
    setInstalling(true);
    setInstallError(null);
    try {
      const out = await plugins.install(source);
      // A new plugin with a setup wizard: its wizard opens now. An update whose
      // setup went up gets a nudge on its card instead (it was set up before).
      const opensSetup = !!out.setup && !out.replaced;
      // A release build does not start a package nobody signed until the person says so:
      // not a failure, but "Click Start" would be the wrong thing to tell them.
      const needsTrust = out.trust?.state === 'unsigned';
      // (a signature that does not check out: it never starts, so nothing here says to start it)
      const quarantined = out.trust?.state === 'quarantined';
      // An update of a plugin that was running: started again with the new package, or not, and why.
      const notRestarted = out.replaced && out.notRestarted ? out.notRestarted : null;
      const restarted = out.replaced && out.restarted === true;
      toast.push({
        kind: quarantined ? 'error' : notRestarted ? 'info' : 'success',
        title: out.replaced
          ? notRestarted
            ? `Updated ${out.name} to v${out.version}. It was not started again`
            : restarted
              ? `Updated ${out.name} to v${out.version} and started it again`
              : `Updated ${out.name} to v${out.version}`
          : `Installed ${out.name} v${out.version}`,
        body: notRestarted
          ? `${notRestarted.charAt(0).toUpperCase()}${notRestarted.slice(1).replace(/\.$/, '')}.`
          : quarantined
            ? 'Its signature does not check out, so it will not start. Install a copy from its publisher.'
            : needsTrust
              ? 'It is not signed, so it will not start until you trust it.'
              : opensSetup
                ? 'Its setup opens now.'
                : out.setup
                  ? 'If its setup has something new, its card says so.'
                  : restarted
                    ? 'It runs the new version now.'
                    : 'Click Start to run it.',
      });
      setInstallSource('');
      void refetchModules();
      await Promise.all([refresh(), refetchSetup()]);
      if (opensSetup) openSetup({ plugin: out.id });
    } catch (e) {
      setInstallError(e instanceof Error ? e.message : String(e));
    } finally {
      setInstalling(false);
    }
  }, [installSource, refresh, toast]);

  const uninstallPlugin = useCallback(
    async (p: PluginRecord) => {
      const name = p.manifest?.name ?? declaredOf(p)?.name ?? p.id;
      if (!(await ask(`Remove the “${name}” plugin? Its files are deleted from this computer.`))) return;
      await runAction(p.id, `${name} was not removed`, async () => {
        await plugins.uninstall(p.id);
        toast.push({ kind: 'success', title: `Removed ${name}` });
      });
    },
    [runAction, toast],
  );

  const trustPlugin = useCallback(
    async (p: PluginRecord) => {
      // (a package not trusted yet has no manifest here: what it says of itself is shown instead, never acted on)
      const declared = declaredOf(p);
      const name = p.manifest?.name ?? declared?.name ?? p.id;
      const declaresAi = (p.manifest?.capabilities ?? declared?.capabilities ?? []).includes('oaiy.ai.complete');
      // (the same question a plugin's setup asks: PluginRunNote)
      if (!(await ask(trustQuestion(p)))) return;
      await runAction(p.id, `${name} was not trusted`, async () => {
        await plugins.trust(p.id);
        const running = p.state === 'running' || p.state === 'unhealthy' || p.state === 'starting';
        toast.push({ kind: 'success', title: `Trusted ${name}`, body: running ? (declaresAi ? 'Return to the plugin and read its host models again.' : 'The plugin is already running.') : 'Click Start to run it.' });
      });
    },
    [runAction, toast],
  );

  return (
    <div className="panel">
      {error && <LoadError what="The plugins" error={error} onRetry={refresh} />}

      {snapshot && (
        <div className="datadir-note datadir-row">
          <span>
            Plugins live in <code className="path-code">{snapshot.root}</code>
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
            <span>Plugin folder, .zip or .tar.gz on this computer</span>
            <span className="setup-folder-row">
              <input className="is-mono"
                type="text"
                placeholder={PLUGIN_PATH_EXAMPLE}
                value={installSource}
                onChange={(e) => {
                  setInstallSource(e.target.value);
                  setInstallError(null);
                }}
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
          {installError && (
            <div className="card-warn" role="alert">
              <TriangleAlert size={13} aria-hidden /> It was not installed: {installError}
            </div>
          )}
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
        // (not read, and the banner above says why: no "Loading" under it that never ends)
        !error && <div className="empty-state is-loading">Loading plugins…</div>
      ) : list.length === 0 ? (
        <PluginCatalog onInstall={(path) => setInstallSource(path)} />
      ) : (
        <section className="service-section">
          {list.map((p) => {
            const name = p.manifest?.name ?? declaredOf(p)?.name ?? p.id;
            return (
              <PluginCard
                key={p.id}
                plugin={p}
                setupTitle={readSetup(p)?.title ?? null}
                setupStatus={p.userDisabled ? 'none' : pluginSetupStatus(p, setupState, live)}
                onSetup={() => openSetup({ plugin: p.id })}
                pending={pending.has(p.id)}
                onStart={() => runAction(p.id, `${name} did not start`, () => plugins.start(p.id))}
                onStop={() => runAction(p.id, `${name} did not stop`, () => plugins.stop(p.id))}
                onToggleEnabled={() =>
                  runAction(p.id, `${name} was not ${p.userDisabled ? 'enabled' : 'disabled'}`, () => plugins.setEnabled(p.id, p.userDisabled))
                }
                onViewLogs={() => setLogsFor(p.id)}
                onTrust={() => void trustPlugin(p)}
                onUninstall={() => void uninstallPlugin(p)}
              />
            );
          })}
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
            // (coloured as the plugin that offers it stands: running, stopped, crashed)
            <div key={d.id} className={`service-card service-card-${cardStatus(list.find((p) => p.id === d.pluginId)?.state ?? 'stopped')}`}>
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
          title={`${nameOf(list.find((p) => p.id === logsFor)) ?? logsFor} — logs`}
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
  /** Trust this exact, unsigned package, including one running in a developer build. */
  onTrust: () => void;
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
  onTrust,
  onUninstall,
}: CardProps) {
  const loadable = isLoadable(p);
  const needsSetup = setupStatus === 'needs-setup';
  // Unhealthy only because its setup has not recorded what it needs yet: the next step, not a fault.
  const waiting = waitingForSetup(p, setupStatus);
  const running = p.state === 'running' || p.state === 'unhealthy' || p.state === 'starting';
  // The host will not start this package: it failed its signature, or it has none and
  // this is a release build. A quarantined package cannot be trusted by hand; an
  // unsigned one can, and only this build's policy holds it back.
  const quarantined = p.trust?.state === 'quarantined';
  const unsigned = p.trust?.state === 'unsigned';
  const unsignedDev = p.trust?.state === 'unsigned-dev';
  const canTrust = unsigned || unsignedDev;
  const heldBack = quarantined || unsigned;
  const connectorCount =
    p.manifest?.connectors?.reduce((n, c) => n + c.commands.length, 0) ?? 0;

  return (
    <div className={`service-card service-card-${waiting ? 'starting' : cardStatus(p.state)}`}>
      <div className="card-head">
        <CircleDot size={14} aria-hidden />
        <strong className="card-title">{p.manifest?.name ?? declaredOf(p)?.name ?? p.id}</strong>
        <span className={waiting ? 'badge badge-pending' : badgeClass(p.state)}>{waiting ? 'needs setup' : p.state}</span>
        {p.manifest?.version && (
          <span className="card-note">v{p.manifest.version}</span>
        )}
        {p.trust && <PackageTrustBadge trust={p.trust} />}
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
      {/* A plugin that is up keeps running when its folder stops verifying, and has no
          reason of its own to show: say what changed, or it looks fine. */}
      {p.state === 'running' && heldBack && p.trust?.reason && (
        <p className="card-reason">{p.trust.reason}</p>
      )}

      {loadable && (
        <div className="card-meta">
          {connectorCount > 0 && `${connectorCount} command${connectorCount === 1 ? '' : 's'}`}
          {connectorCount > 0 && (p.manifest?.events?.length ?? 0) > 0 && ' · '}
          {(p.manifest?.events?.length ?? 0) > 0 &&
            `${p.manifest!.events!.length} event${p.manifest!.events!.length === 1 ? '' : 's'}`}
        </div>
      )}

      {/* Something it asks for that OAIY has no equivalent for is said, before a mystery denial. (The names it used
          before OAIY's own, which OAIY maps, are not: nothing for the person to do about them.) */}
      {p.unknownCapabilities && p.unknownCapabilities.length > 0 && (
        <p className="card-meta card-warn">
          <TriangleAlert size={12} /> Asks for {p.unknownCapabilities.join(', ')}, which this OAIY
          build does not provide.
        </p>
      )}

      {unsignedDev && p.manifest?.capabilities?.includes('oaiy.ai.complete') && (
        <p className="card-meta">This developer build lets the unsigned plugin run. Its AI access requires you to trust this exact package.</p>
      )}
      {unsignedDev && p.manifest?.capabilities?.includes('oaiy.voice.session') && <p className="card-meta">Its local voice access requires you to trust this exact package. Microphone capture needs explicit consent in each session.</p>}

      <div className="card-actions">
        {/* A package the host holds back until it is trusted: trusting it is the next step, so it comes first. */}
        {unsigned && !pending && !running && (
          <button className="btn btn-primary" onClick={onTrust} title="Let this exact package run. A change to any of its files asks again.">
            <ShieldCheck size={14} /> Trust this plugin
          </button>
        )}
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
            className={heldBack ? 'btn btn-secondary' : 'btn btn-primary'}
            onClick={onStart}
            disabled={!loadable || p.userDisabled || heldBack}
            title={
              quarantined
                ? 'This package failed its signature check, so it will not start.'
                : unsigned
                  ? 'This package is not signed. Trust it first to be able to start it.'
                  : !loadable
                    ? 'This plugin cannot start — its manifest is invalid.'
                    : p.userDisabled
                      ? 'Turned off. Enable it first.'
                      : undefined
            }
          >
            <Play size={14} /> Start
          </button>
        )}

        {/* Only for a package nobody signed, including one allowed to run in development. A package
            that carries a signature is verified or quarantined by that signature alone. */}
        {canTrust && !pending && !(unsigned && !running) && (
          <button className="btn btn-secondary" onClick={onTrust} title="Let this exact package run. A change to any of its files asks again.">
            <ShieldCheck size={14} /> Trust this plugin
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
          <button className="btn btn-ghost" onClick={onToggleEnabled} disabled={pending}>
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

/**
 * No plugins yet: the ones OAIY knows how to install (the bundled catalogue), what each brings and needs, and where
 * to get it, as said for this computer. One found on this computer is put in the install box above with a press.
 */
function PluginCatalog({ onInstall }: { onInstall: (path: string) => void }) {
  const [catalog, setCatalog] = useState<CatalogPlugin[] | null>(null);
  useEffect(() => {
    setupApi
      .catalog()
      .then((c) => setCatalog(c.plugins))
      .catch(() => setCatalog([]));
  }, []);
  return (
    <div className="empty-state plugin-catalog">
      <Plug size={22} style={{ opacity: 0.5 }} />
      <p>No plugins installed yet.</p>
      <p className="form-hint">Plugins bring devices and services to OAIY, such as your business phone and its calendar.{catalog && catalog.length > 0 ? ' These are the ones OAIY knows:' : ''}</p>
      {catalog && catalog.length > 0 && (
        <ul className="plugin-catalog-list">
          {catalog.map((c) => (
            <li key={c.id}>
              <strong>{c.name}</strong>
              {c.description && <span>{c.description}</span>}
              {c.needs && <small>Needs: {c.needs}</small>}
              {c.source.path ? (
                <span className="plugin-catalog-found">
                  <small>Found on this computer.</small>
                  <button type="button" className="btn-tiny btn-primary" onClick={() => onInstall(c.source.path!)}>
                    Use this folder
                  </button>
                </span>
              ) : (
                catalogNote(c.source) && <small>{catalogNote(c.source)}</small>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** A plugin's name as a person knows it: its manifest's, else what it says of itself, else its id. */
function nameOf(p: PluginRecord | undefined): string | undefined {
  return p ? (p.manifest?.name ?? declaredOf(p)?.name ?? p.id) : undefined;
}

/** What a package that may not run says of itself (its name, version and capabilities): shown, never acted on. */
function declaredOf(p: PluginRecord): { name: string; version: string; capabilities: string[] } | undefined {
  return p.declared;
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
