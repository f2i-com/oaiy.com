import { useState } from 'react';
import { Loader2, Play, ShieldCheck, TriangleAlert } from 'lucide-react';
import { plugins as pluginsApi, type PluginRecord } from './api';
import { ask } from './ask';

/**
 * A plugin that is not running, said once where something needs it (a setup
 * step that talks to it, one of its screens), with the one thing that gets it
 * running: Start it; Trust this plugin first for a package nobody signed (the
 * host does not start one until the person trusts it, so Start alone was a dead
 * end); Turn it on for one the person turned off. A package whose signature
 * does not check out never starts, and is said so, with nothing to press.
 */

/** What the person is asked before trusting an unsigned package: the same on the Plugins page and here. */
export function trustQuestion(p: Pick<PluginRecord, 'id' | 'manifest' | 'declared'>): string {
  // (a package not trusted yet may have no manifest here: what it says of itself is shown instead, never acted on)
  const name = p.manifest?.name ?? p.declared?.name ?? p.id;
  const capabilities = p.manifest?.capabilities ?? p.declared?.capabilities ?? [];
  const declaresAi = capabilities.includes('oaiy.ai.complete');
  const declaresVoice = capabilities.includes('oaiy.voice.session');
  return (
    `Trust the "${name}" plugin?\n\nIt is not signed, so OAIY cannot tell who made it. Trusting it lets it run as native code with your permissions. ` +
    'The trust is for this exact package only: if any of its files change, you are asked again.\n\n' +
    (declaresAi ? 'This package declares AI completion access to your configured providers. Trusting it enables that access.\n\n' : '') +
    (declaresVoice ? 'This package declares local voice session access. Trusting it permits ready local speech; microphone capture still requires explicit consent in each session and native microphone permission.\n\n' : '') +
    'Only trust a plugin you built yourself or got from someone you trust.'
  );
}

type RunRecord = Pick<PluginRecord, 'id' | 'state' | 'reason' | 'trust' | 'userDisabled' | 'manifest' | 'declared'>;

/** Why it is not running, in words, where the host's reason says more than the state ("Not started yet" says nothing). */
function reasonWorthSaying(record: RunRecord): string | null {
  const reason = (record.reason ?? '').trim().replace(/\.+$/, '');
  if (!reason || /^not started( yet)?$/i.test(reason)) return null;
  // (the trust is said in its own words, and its button is there)
  if (record.trust?.state === 'unsigned' || record.trust?.state === 'quarantined') return null;
  if (record.userDisabled) return null;
  return reason;
}

export default function PluginRunNote({
  record,
  place,
  onChanged,
}: {
  record: RunRecord;
  /** `step`: a setup step that talks to it; `screen`: one of its own screens. */
  place: 'step' | 'screen';
  /** It was started, trusted or turned on: read it again. */
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (record.state === 'running' || record.state === 'unhealthy') return null;
  const name = record.manifest?.name ?? record.declared?.name ?? record.id;
  const starting = record.state === 'starting';
  const unsigned = record.trust?.state === 'unsigned';
  const quarantined = record.trust?.state === 'quarantined';
  const off = record.userDisabled;
  const after = place === 'step' ? 'This step talks to it, and carries on once it runs.' : 'This screen works once it runs.';
  const reason = reasonWorthSaying(record);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message.replace(/^\d{3}: /, '') : String(e));
    } finally {
      setBusy(false);
    }
  };
  const start = () => act(() => pluginsApi.start(record.id));
  // Trusted, it is started at once: what the person pressed this for.
  const trust = () =>
    void (async () => {
      if (!(await ask(trustQuestion(record)))) return;
      await act(async () => {
        await pluginsApi.trust(record.id);
        await pluginsApi.start(record.id);
      });
    })();
  // (turning it on does not start it: both, as one press)
  const turnOn = () =>
    act(async () => {
      await pluginsApi.setEnabled(record.id, true);
      await pluginsApi.start(record.id);
    });

  let said: string;
  if (starting) said = `${name} is starting…`;
  else if (quarantined) said = `${name} will not start: its signature does not check out. Install a copy from its publisher.`;
  else if (unsigned) said = `${name} is not signed, and you have not trusted it yet, so it does not run.`;
  else if (off) said = `${name} is turned off.`;
  else if (record.state === 'crashed') said = `${name} stopped${reason ? `: ${reason}` : ' unexpectedly'}.`;
  else said = `${name} is not running${reason ? `: ${reason}` : ''}.`;

  return (
    <div className={place === 'screen' ? 'setup-run-note is-page' : 'setup-run-note'} role="status">
      <TriangleAlert size={13} aria-hidden />
      <span>
        {said} {!quarantined && after}
        {error && <small>{error}</small>}
      </span>
      {!starting && !quarantined && (
        <button
          type="button"
          className={unsigned ? 'btn-tiny btn-primary' : 'btn-tiny'}
          disabled={busy}
          title={unsigned ? 'Let this exact package run. A change to any of its files asks again.' : undefined}
          onClick={() => (unsigned ? trust() : off ? void turnOn() : void start())}
        >
          {busy ? <Loader2 size={12} className="spin" /> : unsigned ? <ShieldCheck size={12} /> : <Play size={12} />}{' '}
          {unsigned ? 'Trust this plugin' : off ? 'Turn it on' : 'Start it'}
        </button>
      )}
    </div>
  );
}
