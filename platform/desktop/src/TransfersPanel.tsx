import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, Loader2, TriangleAlert } from 'lucide-react';
import { companion, isNotFound, ring, type CompanionApproved, type RingFeatures, type RingPreview, type RingSettings } from './api';
import { useUnsaved } from './unsavedChanges';
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

/**
 * Passing a call to you needs the phone's consent for taking calls, and on this computer that consent is not signed: a plugin that
 * can reach it could change what it allows. So the switch below is best kept off unless every installed plugin is trusted.
 */
export const CONSENT_NOT_SIGNED = 'The permission you give for taking calls is not protected on this computer, so another plugin you installed could change it. Keep this off unless you trust every plugin you have installed.';

/** These settings are reached over OAIY's local address, which a program that OAIY has paired (or that has an administrator's token) can use as well as this window. */
export const PAIRED_APP_CAN_CHANGE = 'A program on this computer that OAIY has paired can also change these settings, and turn transfers on. Pair only programs you trust.';

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** What an approved Companion is, for when it rings: a phone or tablet, the app on this computer, or one never rung. */
export type DeviceKind = 'phone' | 'companion' | 'never';

export const DEVICE_KINDS: Array<{ kind: DeviceKind; label: string; hint: string }> = [
  { kind: 'phone', label: 'A phone or tablet', hint: 'Rings by the Phones setting above.' },
  { kind: 'companion', label: 'The Companion app on this computer', hint: 'Rings only while you are at this computer.' },
  { kind: 'never', label: 'Never ring it', hint: 'For the phone that carries your calls: it cannot take its own.' },
];

/** The kind a device is saved as: never rung wins (it is rung by neither rule), then this computer's Companion, else a phone. */
export function deviceKind(settings: Pick<RingSettings, 'windowsCompanions' | 'excludedDevices'>, id: string): DeviceKind {
  if (settings.excludedDevices.includes(id)) return 'never';
  return settings.windowsCompanions.includes(id) ? 'companion' : 'phone';
}

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

const lines = (list: string[]) => list.join('\n');
const fromLines = (text: string) => text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);

/**
 * What `next` changes of `saved`: the settings that differ, `quietHours` and `limits` member by member (the desktop
 * keeps the others as they are), so a setting a paired program changed meanwhile is not put back.
 */
export function changedFrom<T extends object>(saved: object, next: T): Partial<T> {
  const out: Record<string, unknown> = {};
  const was = saved as Record<string, unknown>;
  for (const [key, value] of Object.entries(next)) {
    const before = was[key];
    if (value && typeof value === 'object' && !Array.isArray(value) && before && typeof before === 'object' && !Array.isArray(before)) {
      const inner = Object.fromEntries(Object.entries(value).filter(([k, v]) => JSON.stringify(v) !== JSON.stringify((before as Record<string, unknown>)[k])));
      if (Object.keys(inner).length) out[key] = inner;
    } else if (JSON.stringify(value) !== JSON.stringify(before)) {
      out[key] = value;
    }
  }
  return out as Partial<T>;
}

export default function TransfersPanel() {
  const [saved, setSaved] = useState<RingSettings | null>(null);
  const [draft, setDraft] = useState<RingSettings | null>(null);
  const [features, setFeatures] = useState<RingFeatures | null>(null);
  const [missing, setMissing] = useState(false);
  /** Why the desktop could not use the file the settings are kept in (it says where the file is kept), when it could not. */
  const [loadProblem, setLoadProblem] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [devices, setDevices] = useState<CompanionApproved[] | null>(null);
  /** Why the approved Companions could not be read, when they could not (not the same as none approved). */
  const [devicesError, setDevicesError] = useState<string | null>(null);
  /** What a caller who asks for you would get now, from the desktop's own policy. */
  const [preview, setPreview] = useState<RingPreview | null>(null);
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

  // Whether the form differs from what is saved, as of now: a reading that lands while the owner edits leaves the form alone.
  const editing = useRef(false);
  // (each save counted: a reading asked before a save answers what was there before it, and is not taken)
  const writes = useRef(0);
  const load = useCallback(
    (live: () => boolean = () => true) => {
      const asked = writes.current;
      return ring.settings().then(
        (r) => {
          if (!live() || editing.current || asked !== writes.current) return;
          adopt(r.settings, r.features);
          setLoadProblem(r.loadProblem ?? null);
          setError(null);
        },
        (e) => {
          if (!live()) return;
          if (isNotFound(e)) setMissing(true);
          else setError(errText(e));
        },
      );
    },
    [adopt],
  );
  useEffect(() => {
    let live = true;
    void load(() => live);
    return () => {
      live = false;
    };
  }, [load]);

  const look = useCallback(() => {
    ring.preview().then(setPreview, () => setPreview(null));
  }, []);

  // The devices that may take a call: the Companions approved for the phone. And what a caller would get now, which changes as the owner
  // comes and goes and as the phone plugin calls.
  // And, while nothing is being changed here, the settings as saved: a paired program may change them too.
  useVisiblePoll(() => {
    companion.status('aokie').then(
      (s) => {
        setDevices(s.approvedMobiles ?? []);
        setDevicesError(null);
      },
      (e) => setDevicesError(errText(e)),
    );
    look();
    if (saved) void load();
  }, 20_000);

  const current = useMemo(() => (draft ? { ...draft, urgentPhrases: fromLines(urgentText), vipNumbers: fromLines(vipText) } : null), [draft, urgentText, vipText]);
  const dirty = !!saved && !!current && JSON.stringify(current) !== JSON.stringify(saved);
  // (a text box with the caret in it counts as being edited too: a reading would put its text back under the caret,
  // a trailing empty line the person just typed gone)
  const typing = useRef(false);
  editing.current = dirty || typing.current;
  useUnsaved('transfers', dirty, 'the transfer settings');

  const patch = (change: Partial<RingSettings>) => {
    setJustSaved(false);
    setDraft((d) => (d ? { ...d, ...change } : d));
  };
  const patchQuiet = (change: Partial<RingSettings['quietHours']>) => patch({ quietHours: { ...draft!.quietHours, ...change } });
  const patchLimits = (change: Partial<RingSettings['limits']>) => patch({ limits: { ...draft!.limits, ...change } });
  /** What an approved device is, said as one choice: it sets the two lists the desktop keeps (this computer's Companions, and the devices never rung). */
  const setDeviceKind = (id: string, kind: DeviceKind) => {
    const without = (list: string[]) => list.filter((x) => x !== id);
    const windows = without(draft!.windowsCompanions);
    const excluded = without(draft!.excludedDevices);
    patch({
      windowsCompanions: kind === 'companion' ? [...windows, id] : kind === 'never' ? draft!.windowsCompanions : windows,
      excludedDevices: kind === 'never' ? [...excluded, id] : excluded,
    });
  };
  /** "Who else can change this", opened by the person (it opens by itself while transfers are being turned on). */
  const [whoOpen, setWhoOpen] = useState(false);

  const save = async () => {
    if (!current || !saved) return;
    writes.current += 1;
    setSaving(true);
    setError(null);
    try {
      const r = await ring.save(changedFrom(saved, {
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
      }));
      editing.current = false;
      adopt(r.settings, r.features);
      setJustSaved(true);
      look();
    } catch (e) {
      setError(`Not saved: ${errText(e)}`);
    } finally {
      setSaving(false);
    }
  };

  // Back to what is saved: the form, and the lines typed for the urgent phrases and VIPs.
  const discard = () => {
    if (!saved) return;
    editing.current = false;
    setDraft(saved);
    setUrgentText(lines(saved.urgentPhrases));
    setVipText(lines(saved.vipNumbers));
    setJustSaved(false);
    setError(null);
  };

  if (missing) {
    return (
      <div className="panel is-form transfers-page">
        <p className="form-hint">This OAIY cannot put callers through to you yet: update it to have the receptionist try to reach you.</p>
      </div>
    );
  }
  if (!current) {
    return (
      <div className="panel is-form transfers-page">
        {error ? (
          <>
            <p className="form-error" role="alert">Could not read the settings: {error}</p>
            <button type="button" className="button secondary" onClick={() => void load()}>
              Try again
            </button>
          </>
        ) : (
          <p className="form-hint">Loading…</p>
        )}
      </div>
    );
  }

  const on = current.enabled;
  // What is saved is what rings, and the desktop says what a caller would get: a warning about a setting not yet saved would say something it is not doing.
  const nothingRings = preview?.enabled && !preview.rings && preview.cause ? preview : null;
  // Being turned on, and not saved yet: the moment to read who else can change it, so that line opens by itself.
  const turningOn = on && !saved!.enabled;
  // The switch's one line: what a caller who asks for you would get. As saved, by the desktop's own words; a switch
  // moved and not saved yet says that saving is what makes it so.
  const outcome =
    on !== saved!.enabled
      ? on
        ? 'When you save, the receptionist will try to reach you for a caller who asks for you, and take a message if you do not answer.'
        : 'When you save, the receptionist will stop trying to reach you.'
      : !on
        ? 'Off: nothing rings. Turned on, the receptionist says it will try to reach you when a caller asks for you, rings your devices, and takes a message if you do not answer.'
        : preview?.enabled && !nothingRings && preview.text
          ? preview.text
          : 'On: when a caller asks for you, the receptionist tries to reach you, and takes a message if you do not answer.';
  return (
    <div className="panel is-form transfers-page">
      <section className="model-section">
        <h3 className="section-title">Calls to you</h3>
        {/* The one decision first: may the receptionist ring me. Its line under it says what a caller would get. */}
        <label className="switch-row">
          <input type="checkbox" className="switch" checked={on} onChange={(e) => patch({ enabled: e.target.checked, takeMessages: e.target.checked ? true : current.takeMessages })} />
          <span>
            <strong>Transfer calls to me</strong>
            <small data-testid={on === saved!.enabled && on && preview?.enabled && !nothingRings && preview.text ? 'what-would-ring' : undefined}>{outcome}</small>
          </span>
        </label>
        {/* Who else could turn this on: one closed line under the switch, open by itself while it is being turned on. */}
        <details className="transfers-who" open={turningOn || whoOpen} onToggle={(e) => setWhoOpen((e.currentTarget as HTMLDetailsElement).open)}>
          <summary>Who else can change this</summary>
          <div className="transfers-warning" data-testid="consent-not-signed">
            <TriangleAlert size={14} aria-hidden />
            <span>{CONSENT_NOT_SIGNED}</span>
          </div>
          <div className="transfers-warning" data-testid="paired-app-can-change">
            <TriangleAlert size={14} aria-hidden />
            <span>{PAIRED_APP_CAN_CHANGE}</span>
          </div>
        </details>
        <label className="switch-row">
          <input type="checkbox" className="switch" checked={on || current.takeMessages} disabled={on} onChange={(e) => patch({ takeMessages: e.target.checked })} />
          <span>
            <strong>Take messages</strong>
            <small>
              {on
                ? 'On while transfers are on: a message is what the receptionist offers when nobody answers.'
                : 'The receptionist asks the caller what they want you to know and keeps it under Messages.'}
            </small>
          </span>
        </label>
        {loadProblem && (
          <div className="transfers-warning" role="alert" data-testid="settings-problem">
            <TriangleAlert size={14} aria-hidden />
            <span>{loadProblem}</span>
          </div>
        )}
        {nothingRings && (
          <div className="transfers-warning" data-testid="nothing-would-ring">
            <TriangleAlert size={14} aria-hidden />
            <span>{nothingRings.text}</span>
            {nothingRings.cause === 'noCompanion' && (
              <button type="button" className="btn-tiny" onClick={() => openSetup({ plugin: 'aokie', step: 'pair' })}>
                Set up a Companion
              </button>
            )}
          </div>
        )}
        {preview?.enabled && preview.plugin && (
          <div className="transfers-warning" data-testid="plugin-state">
            <TriangleAlert size={14} aria-hidden />
            <span>{preview.plugin}</span>
          </div>
        )}
        {/* (what would ring now is the switch's own line, above) */}
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
              Your Companions ring: the one on this computer while you are at it, and one on a second phone when you are away. When this computer is one of the devices rung,
              it also tells you (a notification, and a box on this window with who is calling and what they said). It starts no Companion: one that is not running rings only if
              you open it while the caller is still being rung. You answer on the Companion, and whoever answers first takes the call. The box on this window can only decline
              and have the receptionist take a message, or be put away with Not now.
            </li>
            <li>
              Nothing rings unless a Companion is set up to take the call: approve one on the Phone page, on this computer or on a second phone (not the phone that carries the
              calls), and give the Companion’s consent for taking calls there. A phone rings when you are away, and also when no Companion on this computer can take the call, so
              one phone is enough; say “The Companion app on this computer” only of a Companion that is the Windows app here. With none, callers are offered a message and you are
              told someone asked for you.
            </li>
            <li>The receptionist stops speaking, tells the caller it is connecting them, and you talk to the caller. The receptionist does not come back unless you hand the call back.</li>
            <li>
              If nobody answers in time, you decline, it is quiet hours, you are away with no device to ring, or a limit is reached, the receptionist offers to take a message. It
              always ends with the caller being spoken to.
            </li>
            <li>
              It only tries when the caller’s own words asked for a person (or said one of your own urgent phrases, if you turned that on), as this computer heard them, and an ask
              counts for one try: what they said is used up once it rings. So a caller cannot make it ring over and over, and the limits below cap what is left.
            </li>
            <li>What callers leave is kept under Messages, on this computer.</li>
          </ol>
        </details>
      </section>

      <section className="model-section">
        <h3 className="section-title">How it rings</h3>
        {/* Each setting a sentence with its choice in it, as Hours & Services writes its booking rules. */}
        <div className="transfers-two">
        <ul className="rules transfers-rules">
          <li>
            <label htmlFor="ring-seconds">Ring for</label>
            <input id="ring-seconds" type="number" min={20} max={90} value={current.ringSeconds} onChange={(e) => patch({ ringSeconds: Number(e.target.value) })} />
            <span>seconds, then offer to take a message.</span>
          </li>
          <li>
            <label htmlFor="ring-desktop">This computer</label>
            <select id="ring-desktop" value={current.desktopRing} onChange={(e) => patch({ desktopRing: e.target.value as RingSettings['desktopRing'] })}>
              <option value="auto">Rings while I am at it</option>
              <option value="always">Always rings</option>
              <option value="never">Never rings</option>
            </select>
          </li>
          <li>
            <label htmlFor="ring-phones">Phones</label>
            <select id="ring-phones" value={current.phoneRing} onChange={(e) => patch({ phoneRing: e.target.value as RingSettings['phoneRing'] })}>
              <option value="when_away">Ring when I am away</option>
              <option value="always">Always ring</option>
              <option value="never">Never ring</option>
            </select>
            <small className="form-hint">A phone also rings when no Companion on this computer can take the call.</small>
          </li>
          <li>
            <label htmlFor="ring-away">Treat me as away</label>
            <select id="ring-away" value={current.away} onChange={(e) => patch({ away: e.target.value as RingSettings['away'] })}>
              <option value="auto">when I am not at this computer</option>
              <option value="on">all the time</option>
              <option value="off">never: I am here</option>
            </select>
          </li>
          <li>
            <label htmlFor="ring-when">Ring me</label>
            <select id="ring-when" value={current.initiative} onChange={(e) => patch({ initiative: e.target.value as RingSettings['initiative'] })}>
              <option value="on_request">only when the caller asks for a person</option>
              <option value="on_request_or_urgent">also when they say one of my urgent phrases</option>
            </select>
          </li>
        </ul>
        <label className="form-row transfers-vip">
          <span>Urgent phrases, one a line</span>
          <textarea rows={3} value={urgentText} onChange={(e) => { setJustSaved(false); setUrgentText(e.target.value); }} placeholder={'gas leak\nburst pipe'} aria-label="Urgent phrases" onFocus={() => (typing.current = true)} onBlur={() => (typing.current = false)} />
          <small className="form-hint">Used when you have the receptionist ring you for them, here or in quiet hours.</small>
        </label>
        </div>
      </section>

      <section className="model-section">
        <h3 className="section-title">Quiet hours</h3>
        <div className="transfers-two">
        <div className="transfers-quiet">
        <label className="switch-row">
          <input type="checkbox" className="switch" checked={current.quietHours.enabled} onChange={(e) => patchQuiet({ enabled: e.target.checked })} />
          <span>
            <strong>Do not ring me during quiet hours</strong>
            <small>The receptionist takes a message instead. By this computer’s clock.</small>
          </span>
        </label>
        {current.quietHours.enabled && (
          <>
            <div className="transfers-grid transfers-grid-fields">
              <label className="form-row">
                <span>From</span>
                <input type="time" value={current.quietHours.start} onChange={(e) => patchQuiet({ start: e.target.value })} />
              </label>
              <label className="form-row">
                <span>To</span>
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
              <input type="checkbox" className="switch" checked={current.quietHours.allowVip} onChange={(e) => patchQuiet({ allowVip: e.target.checked })} />
              <span><strong>Still ring me for my VIPs</strong></span>
            </label>
            <label className="switch-row">
              <input type="checkbox" className="switch" checked={current.quietHours.allowUrgent} onChange={(e) => patchQuiet({ allowUrgent: e.target.checked })} />
              <span><strong>Still ring me when it is urgent</strong><small>Only for the urgent phrases above.</small></span>
            </label>
          </>
        )}
        </div>
        <label className="form-row transfers-vip">
          <span>VIP numbers, one a line</span>
          <textarea rows={3} value={vipText} onChange={(e) => { setJustSaved(false); setVipText(e.target.value); }} placeholder="0491 570 006" aria-label="VIP numbers" onFocus={() => (typing.current = true)} onBlur={() => (typing.current = false)} />
          <small className="form-hint">They ring through quiet hours. A caller ID can be faked, so the limits below still apply to them.</small>
        </label>
        </div>
      </section>

      <section className="model-section">
        <h3 className="section-title">Limits</h3>
        <p className="form-hint">So one caller cannot make your devices ring over and over.</p>
        {/* The four limits as two sentences, each number in its place. */}
        <ul className="rules transfers-rules">
          <li>
            <span>Ring me up to</span>
            <input type="number" aria-label="Times to ring in one call" min={1} max={5} value={current.limits.perCall} onChange={(e) => patchLimits({ perCall: Number(e.target.value) })} />
            <span>times in one call,</span>
            <input type="number" aria-label="Seconds between rings" min={10} max={600} value={current.limits.gapSeconds} onChange={(e) => patchLimits({ gapSeconds: Number(e.target.value) })} />
            <span>seconds apart.</span>
          </li>
          <li>
            <span>No more than</span>
            <input type="number" aria-label="Times an hour for one caller" min={1} max={10} value={current.limits.perCallerHour} onChange={(e) => patchLimits({ perCallerHour: Number(e.target.value) })} />
            <span>times an hour for one caller, and</span>
            <input type="number" aria-label="Times an hour for everyone" min={1} max={100} value={current.limits.globalHour} onChange={(e) => patchLimits({ globalHour: Number(e.target.value) })} />
            <span>an hour in all.</span>
          </li>
        </ul>
      </section>

      <section className="model-section">
        <h3 className="section-title">Devices that may take a call</h3>
        <p className="form-hint">
          The Companions you approved on the Phone page. Say what each one is: this computer cannot tell which of them is on the phone that carries your calls, and that phone
          cannot take its own calls.
        </p>
        {devicesError && devices === null ? (
          <p className="form-error" role="alert">
            Could not read the approved Companions: {devicesError}
          </p>
        ) : devices === null ? (
          <p className="form-hint">Looking…</p>
        ) : devices.length === 0 ? (
          <p className="form-hint">No Companion is approved yet: pair one on the Phone page. Until then the receptionist takes a message.</p>
        ) : (
          <ul className="transfers-devices">
            {devices.map((d) => {
              const id = d.endpointKey.thumbprint;
              const name = d.displayName || 'A device';
              const is = deviceKind(current, id);
              return (
                <li key={id}>
                  <strong id={`device-${id}`}>{name}</strong>
                  {/* One choice of three, each with what it means: what the device is decides when it rings. */}
                  <div className="transfers-kinds" role="radiogroup" aria-labelledby={`device-${id}`}>
                    {DEVICE_KINDS.map((k) => (
                      <label key={k.kind} className={is === k.kind ? 'is-on' : undefined}>
                        <input type="radio" name={`device-kind-${id}`} checked={is === k.kind} onChange={() => setDeviceKind(id, k.kind)} />
                        <span>
                          <b>{k.label}</b>
                          <small>{k.hint}</small>
                        </span>
                      </label>
                    ))}
                  </div>
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
      {/* The Hours page's save bar: in view while something is not saved, a quiet line at the end once it is. */}
      <div className={`hours-savebar transfers-savebar${dirty ? ' is-dirty' : ''}`} role="region" aria-label="Save">
        <div className="hours-savebar-text" aria-live="polite">
          {dirty ? (
            <>
              <strong>
                <i aria-hidden /> Unsaved changes
              </strong>
              <small>Calls are put through as saved until you save.</small>
            </>
          ) : justSaved ? (
            <strong className="transfers-saved">
              <Check size={13} /> Saved
            </strong>
          ) : (
            <small>These are saved. Change anything above, then save it here.</small>
          )}
        </div>
        {dirty && (
          <button type="button" className="btn btn-ghost" disabled={saving} onClick={discard}>
            Discard
          </button>
        )}
        <button type="button" className="btn btn-primary" disabled={!dirty || saving} onClick={() => void save()}>
          {saving ? <Loader2 size={14} className="spin" aria-label="Saving" /> : null} Save changes
        </button>
      </div>
    </div>
  );
}
