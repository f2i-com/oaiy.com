import { useEffect, useReducer, useState } from 'react';
import { looksOnLoad, readHost } from '@oaiy/shared/capabilities/host';
import { connectDesktop, disconnectDesktop } from '../lib/desktopConnect';
import { subscribeDesktopStatus, type DesktopInfo } from '../lib/desktopDetection';
import { linkedByConnect, subscribeDesktopLink } from '../lib/desktopLink';
import { getEngineBase } from '../lib/engineEndpoint';

/**
 * Connect: how a tab in a browser is linked to OAIY Desktop (lib/desktopConnect.ts).
 *
 * A tab does not look for the desktop when it opens: on a public address that would be a permission prompt for the visitor and
 * would show the site what is on their network. Pressing Connect is the person asking, so this says what it asks and why the
 * browser may put a question of its own. Once the desktop has answered the link is kept and the editor looks for it as it opens,
 * until Disconnect. Not shown in OAIY's own window, which is given the desktop.
 *
 * Variants:
 *   card     a paragraph, the button, and what came of it (the engine's card in Settings)
 *   compact  a line for the editor's sidebar, under the engine card
 */
export const CONNECT_WHY =
  'Connect asks OAIY Desktop, on this computer or at the address above, whether it is running: one request. Until you press it this page sends nothing to your computer or your network. Your browser may ask whether this site may connect to devices on your network: that question is this request, and Connect needs the answer to be yes.';

export default function ConnectDesktop({ variant }: { variant: 'card' | 'compact' }) {
  const [status, setStatus] = useState<DesktopInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [said, setSaid] = useState<string | null>(null);
  const [, refresh] = useReducer((n: number) => n + 1, 0);
  useEffect(() => subscribeDesktopStatus(setStatus), []);
  useEffect(() => subscribeDesktopLink(refresh), []);
  if (looksOnLoad(readHost())) return null;

  const base = status?.baseUrl ?? getEngineBase();
  const connected = status?.available === true;
  const press = async () => {
    setBusy(true);
    setSaid(null);
    try {
      const info = await connectDesktop();
      setSaid(info.available ? null : `Nothing answered at ${base}. Start OAIY Desktop, then press Connect again.`);
    } finally {
      setBusy(false);
    }
  };
  const button = connected ? (
    linkedByConnect() ? (
      <button type="button" className={variant === 'card' ? 'btn' : undefined} onClick={() => { setSaid(null); disconnectDesktop(); }} title="Forget the link: this page asks nothing of your computer when it opens until you press Connect again">
        Disconnect
      </button>
    ) : null
  ) : (
    <button type="button" className={variant === 'card' ? 'btn btn-primary' : undefined} onClick={() => void press()} disabled={busy} title={CONNECT_WHY} data-connect-desktop>
      {busy ? 'Connecting…' : 'Connect'}
    </button>
  );

  if (variant === 'compact') {
    if (!button) return null;
    return (
      <p className="oaiy-engine-get">
        {connected ? 'Linked to OAIY Desktop. ' : 'Already running OAIY Desktop? '}
        {button}
        {said && <span role="status"> {said}</span>}
      </p>
    );
  }
  return (
    <div className="oaiy-connect">
      {button && <div className="flex flex-wrap items-center gap-2">{button}</div>}
      <p className="oaiy-help faint" style={{ marginTop: 6 }}>
        {connected
          ? linkedByConnect()
            ? 'Connected. This browser remembers the link and looks for OAIY Desktop when the editor opens; Disconnect forgets it.'
            : 'Connected, at the address above (saved in Settings).'
          : CONNECT_WHY}
      </p>
      {said && (
        <p className="oaiy-error-text" role="status">
          {said}
        </p>
      )}
    </div>
  );
}
