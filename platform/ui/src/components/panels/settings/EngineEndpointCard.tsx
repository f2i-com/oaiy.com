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
import { mayLookOnLoad, startDesktopDetection, subscribeDesktopStatus, type DesktopInfo } from '../../../lib/desktopDetection';
import { startDesktopServiceSync } from '../../../lib/desktopServices';
import { Card } from '../../chrome/SectionPage';
import ConnectDesktop from '../../ConnectDesktop';
import DownloadDesktop from '../../DownloadDesktop';

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
      // An address of its own is a link (lib/desktopLink.ts): a tab that had none starts keeping up with the desktop there. The
      // address change has already asked once; nothing here asks again.
      startDesktopDetection({ probeNow: false });
      startDesktopServiceSync({ probeNow: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const live = status?.baseUrl ?? getEngineBase();
  const remote = isRemoteEngine(live);
  // A tab in a browser has not looked for the desktop until its person presses Connect (or has a link): neither there nor not there.
  const notAsked = !status?.checked && !mayLookOnLoad();

  return (
    <Card
      title="OAIY's engine"
      actions={
        <span className={status?.available ? 'oaiy-pill dot ok' : 'oaiy-pill dot'}>
          {status?.available ? `connected${status.version ? ` · v${status.version}` : ''}` : notAsked ? 'not connected' : 'not reachable'}
        </span>
      }
    >
      <p className="oaiy-card-text">
        Where this editor looks for the engine that runs flows, hosts local models and drives
        plugins. Leave it as the default when the editor and the engine are on the same machine.
      </p>

      <form
        className="flex flex-wrap items-end gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          apply(value);
        }}
      >
        <label className="oaiy-field min-w-0 flex-1" style={{ flexBasis: 240 }}>
          <span>Address</span>
          <input
            className="oaiy-input mono"
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
        </label>
        <button type="submit" className="btn btn-primary">Save</button>
        <button
          type="button"
          className="btn"
          onClick={() => apply(null)}
          disabled={value === DEFAULT_ENGINE_BASE}
          title={`Back to ${DEFAULT_ENGINE_BASE}`}
        >
          Reset
        </button>
      </form>

      <p className="oaiy-help faint">
        {status?.available ? `Answering at ${live}.` : notAsked ? `Not looked for yet at ${live}.` : `Nothing answers at ${live}.`}
      </p>
      {error && <p className="oaiy-error-text">{error}</p>}
      {saved && !error && <p className="oaiy-ok-text">Saved. Reconnecting…</p>}

      {/* A tab in a browser looks for the desktop when its person presses this, and says why. Nothing in OAIY's own window. */}
      <ConnectDesktop variant="card" />

      {!status?.available && (
        /* No desktop answers: where to get one. Nothing in OAIY's own window, which is one. */
        <DownloadDesktop variant="card" />
      )}

      {remote && (
        /* A correct address against a loopback-only server fails exactly like a
           typo, so say which one to check. */
        <p className="oaiy-note warn m-0">
          This is another machine. OAIY Desktop only answers the network when you turn that on in
          its own settings, and it will ask you to approve this browser the first time it connects.
        </p>
      )}
    </Card>
  );
}
