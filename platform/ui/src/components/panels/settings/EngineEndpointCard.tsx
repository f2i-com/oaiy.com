/**
 * Where the OAIY engine lives.
 *
 * Loopback is right for the common case, and was compiled in as a constant —
 * which made one genuinely useful arrangement impossible: run the engine on a
 * desktop and drive the editor from a phone on the same network. There was
 * nowhere to put the desktop's address.
 *
 * The field takes what someone reads off a screen (`192.168.1.50:17972`) and
 * stores a clean origin. It also states plainly that the desktop has to be
 * allowing network access, because a correct address against a loopback-only
 * server fails in a way that looks like a typo.
 */
import { useEffect, useState } from 'react';
import {
  DEFAULT_ENGINE_BASE,
  getEngineBase,
  isRemoteEngine,
  setEngineBase,
} from '../../../lib/engineEndpoint';
import { subscribeDesktopStatus, type DesktopInfo } from '../../../lib/desktopDetection';

const INPUT =
  'w-full px-2 py-1 text-sm border border-slate-300 dark:border-slate-600 rounded ' +
  'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 font-mono ' +
  'focus:outline-none focus:ring-2 focus:ring-blue-500/50';

export default function EngineEndpointCard() {
  const [value, setValue] = useState(getEngineBase());
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [status, setStatus] = useState<DesktopInfo | null>(null);

  useEffect(() => subscribeDesktopStatus(setStatus), []);

  const apply = (raw: string | null) => {
    setError(null);
    try {
      const next = setEngineBase(raw);
      setValue(next);
      setSaved(true);
      window.setTimeout(() => setSaved(false), 1800);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const live = status?.baseUrl ?? getEngineBase();
  const remote = isRemoteEngine(live);

  return (
    <div className="rounded-lg border border-slate-300 dark:border-slate-700 p-4 space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">OAIY Desktop engine</h3>
        <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
          Where this editor looks for the engine that runs flows, hosts local models and drives
          plugins. Leave it as the default when the editor and the engine are on the same machine.
        </p>
      </div>

      <div className="flex items-center gap-2">
        <span
          className={`inline-block h-2 w-2 rounded-full ${status?.available ? 'bg-emerald-500' : 'bg-slate-400'}`}
          aria-hidden="true"
        />
        <span className="text-xs text-slate-600 dark:text-slate-400">
          {status?.available
            ? `Connected${status.version ? ` · v${status.version}` : ''} at ${live}`
            : `Not reachable at ${live}`}
        </span>
      </div>

      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          apply(value);
        }}
      >
        <input
          className={INPUT}
          value={value}
          spellCheck={false}
          autoCapitalize="none"
          autoCorrect="off"
          inputMode="url"
          aria-label="Engine address"
          placeholder="192.168.1.50:17972"
          onChange={(e) => {
            setValue(e.target.value);
            setError(null);
          }}
        />
        <button type="submit" className="btn btn-primary whitespace-nowrap">Save</button>
        <button
          type="button"
          className="btn whitespace-nowrap"
          onClick={() => apply(null)}
          disabled={value === DEFAULT_ENGINE_BASE}
          title={`Back to ${DEFAULT_ENGINE_BASE}`}
        >
          Reset
        </button>
      </form>

      {error && <p className="text-xs text-red-600 dark:text-red-400">{error}</p>}
      {saved && !error && <p className="text-xs text-emerald-600 dark:text-emerald-400">Saved — reconnecting…</p>}

      {remote && (
        /* A correct address against a loopback-only server fails exactly like a
           typo, so say which one to check. */
        <p className="text-xs text-amber-700 dark:text-amber-400">
          This is another machine. OAIY Desktop only answers the network when you turn that on in
          its own settings, and it will ask you to approve this browser the first time it connects.
        </p>
      )}
    </div>
  );
}
