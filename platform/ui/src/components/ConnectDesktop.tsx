import { useEffect, useReducer, useState } from 'react';
import { DEFAULT_ENGINE_BASE, getEngineBase } from '../lib/engineEndpoint';
import { looksOnLoad, pageHost } from '@oaiy/shared/capabilities/host';
import { connectDesktop, disconnectDesktop } from '../lib/desktopConnect';
import { subscribeDesktopStatus, type DesktopInfo } from '../lib/desktopDetection';
import { linkedByConnect, subscribeDesktopLink } from '../lib/desktopLink';
import { connectLead, connectState, connectWords } from '../lib/connectWords';

/**
 * Connect: how a tab in a browser is linked to OAIY Desktop (lib/desktopConnect.ts).
 *
 * A tab does not look for the desktop when it opens: on a public address that would be a permission prompt for the visitor and
 * would show the site what is on their network. Pressing Connect is the person asking. What the button and its card say depends on
 * the state (lib/connectWords.ts): never connected (nothing is sent until it is pressed), connected (the editor asks the desktop each
 * time it opens and every ten seconds while it is open), an engine address of its own (the same, at that address), or disconnected
 * (never again). Disconnect is there whenever Connect made the link, answering or not. Not shown in OAIY's own window, which is
 * given the desktop.
 *
 * Variants:
 *   card     a paragraph, the buttons, and what came of it (the engine's card in Settings)
 *   compact  a line for the editor's sidebar, under the engine card
 */
export default function ConnectDesktop({ variant }: { variant: 'card' | 'compact' }) {
  const [status, setStatus] = useState<DesktopInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [said, setSaid] = useState<string | null>(null);
  const [, refresh] = useReducer((n: number) => n + 1, 0);
  useEffect(() => subscribeDesktopStatus(setStatus), []);
  useEffect(() => subscribeDesktopLink(refresh), []);
  if (looksOnLoad(pageHost())) return null;

  const base = getEngineBase();
  const available = status?.available === true;
  const state = connectState({ linkedByConnect: linkedByConnect(), addressGiven: base !== DEFAULT_ENGINE_BASE, base, available });
  const words = connectWords(state);
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
  // Connect asks (again) where the desktop does not answer; Disconnect forgets the link Connect made, whether it answers or not.
  const connect = available ? null : (
    <button type="button" className={variant === 'card' ? 'btn btn-primary' : undefined} onClick={() => void press()} disabled={busy} title={words} data-connect-desktop>
      {busy ? 'Connecting…' : 'Connect'}
    </button>
  );
  const disconnect = state.kind === 'connected' ? (
    <button type="button" className={variant === 'card' ? 'btn' : undefined} onClick={() => { setSaid(null); disconnectDesktop(); }} title="Forget the link: the editor stops asking OAIY Desktop, and asks nothing of your computer when it opens until you press Connect again." data-disconnect-desktop>
      Disconnect
    </button>
  ) : null;

  if (variant === 'compact') {
    if (!connect && !disconnect) return null;
    return (
      <p className="oaiy-engine-get">
        {connectLead(state)}
        {connect}
        {connect && disconnect ? ' · ' : null}
        {disconnect}
        {said && <span role="status"> {said}</span>}
      </p>
    );
  }
  return (
    <div className="oaiy-connect">
      {(connect || disconnect) && <div className="flex flex-wrap items-center gap-2">{connect}{disconnect}</div>}
      <p className="oaiy-help faint" style={{ marginTop: 6 }} data-connect-words={state.kind}>
        {words}
      </p>
      {said && (
        <p className="oaiy-error-text" role="status">
          {said}
        </p>
      )}
    </div>
  );
}
