import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, Loader2, TriangleAlert } from 'lucide-react';
import { companion, isNotFound, ring, type CompanionApproved, type RingFeatures, type RingSettings } from './api';
import { nothingWouldRing } from './transfersModel';
import { openSetup } from './useSetupState';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * Transfers: whether the receptionist may try to reach you for a caller who asks
 * for a person, and how. Both switches are off until you turn them on, so the
 * phone answers exactly as before. With "Transfer calls to me" on, the receptionist
 * tells the caller it will try, this computer rings (a notification and a dialog),
 * and the Companion on this computer or on a second phone can take the call; if
 * nobody does, or you decline, or it is quiet hours, it takes a message instead.
 * The Companion needs your consent for it (the Phone page); without it the
 * receptionist takes a message.
 */

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

const lines = (list: string[]) => list.join('\n');
const fromLines = (text: string) => text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);

export default function TransfersPanel() {
  const [saved, setSaved] = useState<RingSettings | null>(null);
  const [draft, setDraft] = useState<RingSettings | null>(null);
  const [features, setFeatures] = useState<RingFeatures | null>(null);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [devices, setDevices] = useState<CompanionApproved[] | null>(null);
  /** The urgent phrases and VIP numbers are typed as lines: kept as typed until saved. */
  const [urgentText, setUrgentText] = useState('');
  const [vipText, setVipText] = useState('');

  const adopt = useCallback((s: RingSettings, f: RingFeatures) => {
    setSaved(s);
    setDraft(s);
    setFeatures(f);
    setUrgentText(lines(s.urgentPhrases));
    setVipText(lines(s.vipNumbers));
  }, []);

  useEffect(() => {
    let live = true;
    ring.settings().then(
      (r) => live && adopt(r.settings, r.features),
      (e) => {
        if (!live) return;
        if (isNotFound(e)) setMissing(true);
        else setError(errText(e));
      },
    );
    return () => {
      live = false;
    };
  }, [adopt]);

  // The devices that may take a call: the Companions approved for the phone.
  useVisiblePoll(() => {
    companion.status('aokie').then(
      (s) => setDevices(s.approvedMobiles ?? []),
      () => setDevices([]),
    );
  }, 20_000);

  const current = useMemo(() => (draft ? { ...draft, urgentPhrases: fromLines(urgentText), vipNumbers: fromLines(vipText) } : null), [draft, urgentText, vipText]);
  const dirty = !!saved && !!current && JSON.stringify(current) !== JSON.stringify(saved);

  const patch = (change: Partial<RingSettings>) => {
    setJustSaved(false);
    setDraft((d) => (d ? { ...d, ...change } : d));
  };
  const patchQuiet = (change: Partial<RingSettings['quietHours']>) => patch({ quietHours: { ...draft!.quietHours, ...change } });
  const patchLimits = (change: Partial<RingSettings['limits']>) => patch({ limits: { ...draft!.limits, ...change } });
  const toggleIn = (key: 'windowsCompanions' | 'excludedDevices', id: string, on: boolean) => {
    const list = draft![key].filter((x) => x !== id);
    patch({ [key]: on ? [...list, id] : list } as Partial<RingSettings>);
  };

  const save = async () => {
    if (!current) return;
    setSaving(true);
    setError(null);
    try {
      const r = await ring.save({
        enabled: current.enabled,
        takeMessages: current.enabled ? true : current.takeMessages,
        initiative: current.initiative,
        urgentPhrases: current.urgentPhrases,
        ringSeconds: Number(current.ringSeconds),
        phoneRing: current.phoneRing,
        desktopRing: current.desktopRing,
        away: current.away,
        quietHours: current.quietHours,
        vipNumbers: current.vipNumbers,
        limits: {
          perCall: Number(current.limits.perCall),
          gapSeconds: Number(current.limits.gapSeconds),
          perCallerHour: Number(current.limits.perCallerHour),
          globalHour: Number(current.limits.globalHour),
        },
        windowsCompanions: current.windowsCompanions,
        excludedDevices: current.excludedDevices,
      });
      adopt(r.settings, r.features);
      setJustSaved(true);
    } catch (e) {
      setError(`Not saved: ${errText(e)}`);
    } finally {
      setSaving(false);
    }
  };

  if (missing) {
    return (
      <div className="panel transfers-page">
        <p className="form-hint">This OAIY cannot put callers through to you yet: update it to have the receptionist try to reach you.</p>
      </div>
    );
  }
  if (!current) {
    return (
      <div className="panel transfers-page">
        {error ? <p className="form-error" role="alert">Could not read the settings: {error}</p> : <p className="form-hint">Loading…</p>}
      </div>
    );
  }

  const on = current.enabled;
  // What is saved is what rings: a warning about a setting not yet saved would say something the desktop is not doing.
  const nothingRings = saved ? nothingWouldRing(saved, devices) : null;
  return (
    <div className="panel transfers-page">
      <section className="model-section">
        <h3 className="section-title">Calls to you</h3>
        <p className="form-hint">
          When a caller asks for you, the receptionist says it will try to reach you (never that they are put through until you have taken the call), rings your
          devices, and, if you do not answer, offers to take a message. Nothing changes until you turn this on.
        </p>
        <label className="switch-row">
          <input type="checkbox" checked={on} onChange={(e) => patch({ enabled: e.target.checked, takeMessages: e.target.checked ? true : current.takeMessages })} />
          <span>
            <strong>Transfer calls to me</strong>
            <small>
              Needs the Companion on this computer or on a second phone (not the phone that carries the calls), and the Companion’s consent for taking calls on the Phone page.
              Without either, the receptionist takes a message.
            </small>
          </span>
        </label>
        <label className="switch-row">
          <input type="checkbox" checked={on || current.takeMessages} disabled={on} onChange={(e) => patch({ takeMessages: e.target.checked })} />
          <span>
            <strong>Take messages</strong>
            <small>
              {on
                ? 'On while transfers are on: a message is what the receptionist offers when nobody answers.'
                : 'The receptionist asks the caller what they want you to know and keeps it under Messages.'}
            </small>
          </span>
        </label>
        {nothingRings && (
          <div className="transfers-warning" data-testid="nothing-would-ring">
            <TriangleAlert size={14} aria-hidden />
            <span>{nothingRings}</span>
            <button type="button" className="btn-tiny" onClick={() => openSetup({ plugin: 'aokie', step: 'pair' })}>
              Set up a Companion
            </button>
          </div>
        )}
        {features && (
          <p className="form-hint" data-testid="what-it-may-do">
            Now: {features.transfer ? 'the receptionist may try to reach you' : 'it does not try to reach you'}; {features.messages ? 'it takes messages' : 'it takes no messages'}.
          </p>
        )}
      </section>

      <section className="model-section transfers-help" aria-label="How a call is put through">
        <details>
          <summary>How a call is put through to you</summary>
          <ol>
            <li>A caller asks for you, or for a person. The receptionist says it will try to reach you: never that they are put through.</li>
            <li>
              This computer tells you (a notification, and a box on this window with who is calling and what they said), and your Companion rings: the one on this computer while
              you are at it, and one on a second phone when you are away. You answer on the Companion, and whoever answers first takes the call. The box on this window can only
              decline and have the receptionist take a message, or be put away with Not now.
            </li>
            <li>
              Nothing rings unless a Companion is set up to take the call: tick the one that runs on this computer (below), or approve one on a second phone. With none, callers are
              offered a message and you are told someone asked for you.
            </li>
            <li>The receptionist stops speaking, tells the caller it is connecting them, and you talk to the caller. The receptionist does not come back unless you hand the call back.</li>
            <li>
              If nobody answers in time, you decline, it is quiet hours, you are away with no device to ring, or a limit is reached, the receptionist offers to take a message. It
              always ends with the caller being spoken to.
            </li>
            <li>It only tries when the caller’s own words asked for a person, and no more than the limits below allow, so a caller cannot make it ring over and over.</li>
            <li>What callers leave is kept under Messages, on this computer.</li>
          </ol>
        </details>
      </section>

      <section className="model-section">
        <h3 className="section-title">How it rings</h3>
        <div className="transfers-grid">
          <label>
            Ring for (seconds)
            <input type="number" min={20} max={90} value={current.ringSeconds} onChange={(e) => patch({ ringSeconds: Number(e.target.value) })} />
          </label>
          <label>
            This computer
            <select value={current.desktopRing} onChange={(e) => patch({ desktopRing: e.target.value as RingSettings['desktopRing'] })}>
              <option value="auto">Rings while I am at it</option>
              <option value="always">Always rings</option>
              <option value="never">Never rings</option>
            </select>
          </label>
          <label>
            Phones
            <select value={current.phoneRing} onChange={(e) => patch({ phoneRing: e.target.value as RingSettings['phoneRing'] })}>
              <option value="when_away">Ring when I am away</option>
              <option value="always">Always ring</option>
              <option value="never">Never ring</option>
            </select>
          </label>
          <label>
            I am
            <select value={current.away} onChange={(e) => patch({ away: e.target.value as RingSettings['away'] })}>
              <option value="auto">Away when I am not at this computer</option>
              <option value="on">Away</option>
              <option value="off">Here</option>
            </select>
          </label>
          <label>
            The receptionist may ask on its own
            <select value={current.initiative} onChange={(e) => patch({ initiative: e.target.value as RingSettings['initiative'] })}>
              <option value="on_request">Only when the caller asks for a person</option>
              <option value="on_request_or_urgent">Also when they say one of my urgent phrases</option>
            </select>
          </label>
        </div>
        <label className="transfers-grid">
          <span className="form-hint">Urgent phrases, one a line (used only when the setting above allows it)</span>
          <textarea rows={3} value={urgentText} onChange={(e) => { setJustSaved(false); setUrgentText(e.target.value); }} placeholder={'gas leak\nburst pipe'} aria-label="Urgent phrases" />
        </label>
      </section>

      <section className="model-section">
        <h3 className="section-title">Quiet hours</h3>
        <label className="switch-row">
          <input type="checkbox" checked={current.quietHours.enabled} onChange={(e) => patchQuiet({ enabled: e.target.checked })} />
          <span>
            <strong>Do not ring me during quiet hours</strong>
            <small>The receptionist takes a message instead. By this computer’s clock.</small>
          </span>
        </label>
        {current.quietHours.enabled && (
          <>
            <div className="transfers-grid">
              <label>
                From
                <input type="time" value={current.quietHours.start} onChange={(e) => patchQuiet({ start: e.target.value })} />
              </label>
              <label>
                To
                <input type="time" value={current.quietHours.end} onChange={(e) => patchQuiet({ end: e.target.value })} />
              </label>
            </div>
            <div className="transfers-days" role="group" aria-label="The days quiet hours begin on">
              {DAYS.map((d, i) => (
                <label key={d}>
                  <input type="checkbox" checked={(current.quietHours.days & (1 << i)) !== 0} onChange={(e) => patchQuiet({ days: e.target.checked ? current.quietHours.days | (1 << i) : current.quietHours.days & ~(1 << i) })} />
                  {d}
                </label>
              ))}
            </div>
            <label className="switch-row">
              <input type="checkbox" checked={current.quietHours.allowVip} onChange={(e) => patchQuiet({ allowVip: e.target.checked })} />
              <span><strong>Still ring me for my VIPs</strong></span>
            </label>
            <label className="switch-row">
              <input type="checkbox" checked={current.quietHours.allowUrgent} onChange={(e) => patchQuiet({ allowUrgent: e.target.checked })} />
              <span><strong>Still ring me when it is urgent</strong><small>Only for the urgent phrases above.</small></span>
            </label>
          </>
        )}
        <label className="transfers-grid">
          <span className="form-hint">VIP numbers, one a line: they ring through quiet hours. A caller ID can be faked, so the limits above still apply to them.</span>
          <textarea rows={3} value={vipText} onChange={(e) => { setJustSaved(false); setVipText(e.target.value); }} placeholder="0491 570 006" aria-label="VIP numbers" />
        </label>
      </section>

      <section className="model-section">
        <h3 className="section-title">Limits</h3>
        <p className="form-hint">So one caller cannot make your devices ring over and over.</p>
        <div className="transfers-grid">
          <label>
            Tries a call
            <input type="number" min={1} max={5} value={current.limits.perCall} onChange={(e) => patchLimits({ perCall: Number(e.target.value) })} />
          </label>
          <label>
            Seconds between tries
            <input type="number" min={10} max={600} value={current.limits.gapSeconds} onChange={(e) => patchLimits({ gapSeconds: Number(e.target.value) })} />
          </label>
          <label>
            Tries a caller in an hour
            <input type="number" min={1} max={10} value={current.limits.perCallerHour} onChange={(e) => patchLimits({ perCallerHour: Number(e.target.value) })} />
          </label>
          <label>
            Tries in an hour, everyone
            <input type="number" min={1} max={100} value={current.limits.globalHour} onChange={(e) => patchLimits({ globalHour: Number(e.target.value) })} />
          </label>
        </div>
      </section>

      <section className="model-section">
        <h3 className="section-title">Devices that may take a call</h3>
        <p className="form-hint">
          The Companions you approved on the Phone page. The phone that carries your calls is never one of them. Say which one is the Companion on this computer, and which
          never to ring.
        </p>
        {devices === null ? (
          <p className="form-hint">Looking…</p>
        ) : devices.length === 0 ? (
          <p className="form-hint">No Companion is approved yet: pair one on the Phone page. Until then the receptionist takes a message.</p>
        ) : (
          <ul className="transfers-devices">
            {devices.map((d) => {
              const id = d.endpointKey.thumbprint;
              return (
                <li key={id}>
                  <strong>{d.displayName || 'A device'}</strong>
                  <label>
                    <input type="checkbox" checked={current.windowsCompanions.includes(id)} onChange={(e) => toggleIn('windowsCompanions', id, e.target.checked)} /> This is the Companion on this computer
                  </label>
                  <label>
                    <input type="checkbox" checked={current.excludedDevices.includes(id)} onChange={(e) => toggleIn('excludedDevices', id, e.target.checked)} /> Never ring it
                  </label>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {error && (
        <p className="form-error" role="alert">
          <TriangleAlert size={13} /> {error}
        </p>
      )}
      <div className="section-title-row">
        <button type="button" className="button" disabled={!dirty || saving} onClick={() => void save()}>
          {saving ? <Loader2 size={14} className="spin" aria-label="Saving" /> : null} Save changes
        </button>
        {justSaved && !dirty && (
          <span className="transfers-saved">
            <Check size={13} /> Saved
          </span>
        )}
      </div>
    </div>
  );
}
