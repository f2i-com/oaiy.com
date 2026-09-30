import { useCallback, useEffect, useRef, useState } from 'react';
import { Archive, FolderOpen } from 'lucide-react';
import {
  backup,
  formatBytes,
  formatTimestamp,
  openInExplorer,
  type BackupCreateResult,
  type BackupStatus,
  type RestoreClassId,
  type RestorePreview,
  type StagedRestore,
} from './api';

/**
 * Backup and restore: one encrypted file with the person's OAIY setup, made and read back here.
 *
 * The desktop opens the native save and open dialogs itself, so nothing here handles a path
 * except to show one. The passphrase lives only in this component's state, is handed to a command
 * as an argument and is cleared as soon as the command ends: it is never written to storage,
 * logged or put in a message.
 */

/** The fewest characters a backup passphrase may have. */
export const MIN_PASSPHRASE = 12;
/** How often the status is asked for while nothing is happening, and while a backup is being made. */
const IDLE_POLL_MS = 3000;
const RUNNING_POLL_MS = 700;
/** Days without a backup after which the Overview says so. */
export const NUDGE_AFTER_DAYS = 30;

const DAY_MS = 24 * 60 * 60 * 1000;

/** How long the panel waits for the check or the preparing of a restore before it stops waiting. */
export const RESTORE_TIMEOUT_MS = 20 * 60 * 1000;

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** Stops waiting for `pending` after `ms`, so a command that never answers cannot leave the panel stuck. */
function withTimeout<T>(pending: Promise<T>, ms: number, why: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(why)), ms);
    pending.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

/** How long ago something was, as a person says it. */
export function agoWords(iso: string, now: Date | number = new Date()): string {
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) return 'a while ago';
  const minutes = Math.max(0, Math.floor((new Date(now).getTime() - then) / 60_000));
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? '' : 's'} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
  return `${Math.floor(hours / 24)} days ago`;
}

/** The number of characters as a person counts them (an emoji is one, not two). */
const length = (text: string): number => Array.from(text).length;

// ---------------------------------------------------------------------------
// The Overview's "Last backup" line
// ---------------------------------------------------------------------------

export type BackupLineKind = 'recent' | 'stale' | 'never' | 'failed' | 'pending';

export interface BackupLineText {
  /** What the line says. */
  text: string;
  /** Whether it is a gentle nudge (a note with a button) rather than a plain status row. */
  nudge: boolean;
  kind: BackupLineKind;
}

/** "42 days" or "3 months": how long, as a person says it. */
function spanWords(days: number): string {
  if (days < 60) return `${days} day${days === 1 ? '' : 's'}`;
  return `${Math.floor(days / 30.4)} months`;
}

/**
 * What the Overview's line says about the last backup. Pure: `now` is passed in.
 * A restore that waits for a restart comes first, then a failed backup, then no backup at
 * all, then one that is more than 30 days old (a nudge), else how long ago it was.
 */
export function describeLastBackup(status: BackupStatus, now: Date | number = new Date()): BackupLineText {
  if (status.pendingRestore) {
    return { text: 'A restore is waiting for a restart', nudge: false, kind: 'pending' };
  }
  if (status.lastBackupOk === false) {
    return {
      text: 'The last backup did not finish. A backup takes a minute: try again from Settings.',
      nudge: true,
      kind: 'failed',
    };
  }
  const at = status.lastBackupAt ? Date.parse(status.lastBackupAt) : NaN;
  if (!Number.isFinite(at)) {
    return { text: 'You have not made a backup yet.', nudge: true, kind: 'never' };
  }
  const days = Math.max(0, Math.floor((new Date(now).getTime() - at) / DAY_MS));
  if (days > NUDGE_AFTER_DAYS) {
    return {
      text: `It has been ${spanWords(days)} since your last backup. A backup takes a minute.`,
      nudge: true,
      kind: 'stale',
    };
  }
  const ago = days === 0 ? 'today' : days === 1 ? 'yesterday' : `${spanWords(days)} ago`;
  return { text: `Last backup: ${ago}`, nudge: false, kind: 'recent' };
}

/**
 * The Overview's one line about backups. It asks for the status once, on its own, so the
 * Overview's poll is not touched; when the desktop cannot answer it shows nothing.
 */
export function BackupLine({ onOpenSettings }: { onOpenSettings: () => void }) {
  const [status, setStatus] = useState<BackupStatus | null>(null);
  useEffect(() => {
    let cancelled = false;
    backup
      .status()
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch(() => {
        /* an older desktop, or none to ask: no line */
      });
    return () => {
      cancelled = true;
    };
  }, []);
  if (!status) return null;
  const line = describeLastBackup(status);
  const label = line.kind === 'pending' ? 'Open Settings' : line.nudge ? 'Back up now' : 'Back up';
  if (line.nudge) {
    return (
      <div className="datadir-note datadir-row" role="status" data-kind={line.kind} style={{ marginTop: 8 }}>
        <span>{line.text}</span>
        <button className="btn-tiny" onClick={onOpenSettings}>
          <Archive size={13} /> {label}
        </button>
      </div>
    );
  }
  return (
    <div className="overview-status" role="status" data-kind={line.kind} style={{ marginTop: 8 }}>
      <Archive size={14} aria-hidden />
      <strong>{line.kind === 'pending' ? 'Backup' : line.text}</strong>
      {line.kind === 'pending' && <small>{line.text}</small>}
      <button className="btn-tiny" onClick={onOpenSettings}>
        {label}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The Settings section
// ---------------------------------------------------------------------------

/** Show a file in the file manager. */
function OpenButton({ path, onError }: { path: string; onError: (message: string) => void }) {
  return (
    <button
      className="btn-tiny"
      title="Show in file explorer"
      onClick={() => openInExplorer(path).catch((e) => onError(message(e)))}
    >
      <FolderOpen size={13} /> Show
    </button>
  );
}

/** A short list of plain sentences. */
function Sentences({ items }: { items: string[] }) {
  return (
    <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
      {items.map((t, i) => (
        <li key={i}>{t}</li>
      ))}
    </ul>
  );
}

/**
 * What an undo really does, said before it is done. The snapshot it uses is the last restore's (or,
 * for a redo, the last undo's): it puts files back AND takes away the ones that restore added, and
 * whatever the person changed in them since goes with them, though it is saved first.
 */
export function undoConfirmText(kind: 'restore' | 'undo' | null): string {
  if (kind === 'undo') {
    return (
      'Redo? This puts back what the last undo took away, and replaces the files the undo put back, ' +
      'including anything you changed in them since. What it replaces is saved first, so you can undo it again. ' +
      'It happens when you restart OAIY.'
    );
  }
  return (
    'Undo the last restore? This puts back the files the last restore replaced and REMOVES the files it added, ' +
    'and it puts the Agent’s settings back as they were, each one, empty ones included: the AI providers (any the restore added is taken away), its network gate, how it answers texts and calls, and the image, video and audio service. ' +
    'That includes anything you changed or added in them since. What it replaces or removes is saved first, so you can put it back with Redo. ' +
    'It happens when you restart OAIY.'
  );
}

const AGENT_STATE_WORDS: Record<string, string> = {
  applied: 'The Agent’s conversations and projects were restored.',
  pending: 'The Agent’s conversations and projects are restored when the Agent opens.',
  failed: 'The Agent’s conversations and projects could not be restored.',
};

type PendingRestore = NonNullable<BackupStatus['pendingRestore']>;

/**
 * A restore or undo the desktop has just made ready, as the pending banner reads it (until the
 * desktop's own status says it). A prepared restore is thrown away after a day.
 */
function stagedNow(staged: StagedRestore, classes: string[] = [], classLabels: string[] = classes): PendingRestore {
  const now = Date.now();
  return {
    id: staged.id,
    kind: staged.kind,
    stagedAt: new Date(now).toISOString(),
    expiresAt: new Date(now + DAY_MS).toISOString(),
    expired: false,
    files: staged.files,
    agentStorage: staged.agentStorage,
    classes,
    classLabels,
  };
}

const EMPTY_STATUS: BackupStatus = {
  lastBackupAt: null,
  lastBackupOk: null,
  lastBackupSize: null,
  pendingRestore: null,
  lastRestore: null,
  undoAvailable: false,
  undoKind: null,
  running: null,
};

export function BackupSection() {
  const [status, setStatus] = useState<BackupStatus | null>(null);

  // ----- create -----
  const [passphrase, setPassphrase] = useState('');
  const [again, setAgain] = useState('');
  const [includeKeys, setIncludeKeys] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [result, setResult] = useState<BackupCreateResult | null>(null);

  // ----- restore -----
  const [restorePass, setRestorePass] = useState('');
  const [checking, setChecking] = useState(false);
  const [preview, setPreview] = useState<RestorePreview | null>(null);
  const [staging, setStaging] = useState(false);
  const [restoreError, setRestoreError] = useState<string | null>(null);
  // What the person ticked to bring back besides the data. Nothing is ticked until they tick it.
  const [ticked, setTicked] = useState<RestoreClassId[]>([]);
  const [keysTicked, setKeysTicked] = useState(false);
  // What the desktop left out of the restore it has just prepared.
  const [skipped, setSkipped] = useState<string[]>([]);

  // ----- what waits for a restart -----
  const [restartError, setRestartError] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);
  const [pendingBusy, setPendingBusy] = useState(false);
  const [undoError, setUndoError] = useState<string | null>(null);

  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  /**
   * Ask for the status. `staged` is a restore the desktop made ready a moment ago: it stays on
   * show, even if this answer was read before the desktop wrote it down or could not be read at
   * all, until a later poll says otherwise.
   */
  const refresh = useCallback(async (staged?: PendingRestore) => {
    try {
      const s = await backup.status();
      if (mounted.current) setStatus(staged && !s.pendingRestore ? { ...s, pendingRestore: staged } : s);
    } catch {
      if (staged && mounted.current) setStatus((s) => ({ ...(s ?? EMPTY_STATUS), pendingRestore: staged }));
    }
  }, []);

  // The status: every few seconds, and quickly while a backup is being made.
  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), creating ? RUNNING_POLL_MS : IDLE_POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh, creating]);

  const tooShort = length(passphrase) > 0 && length(passphrase) < MIN_PASSPHRASE;
  const mismatch = length(again) > 0 && passphrase !== again;
  const canCreate = length(passphrase) >= MIN_PASSPHRASE && passphrase === again && !creating;

  const create = async () => {
    if (!canCreate) return;
    const pass = passphrase;
    const keys = includeKeys;
    setCreating(true);
    setCreateError(null);
    setResult(null);
    try {
      const made = await backup.create(pass, keys);
      if (mounted.current && made) setResult(made);
    } catch (e) {
      if (mounted.current) setCreateError(message(e));
    } finally {
      if (mounted.current) {
        // Whatever happened, the passphrase does not stay in the page.
        setPassphrase('');
        setAgain('');
        setIncludeKeys(false);
        setCreating(false);
      }
      void refresh();
    }
  };

  const check = async () => {
    if (!restorePass || checking || staging) return;
    setChecking(true);
    setRestoreError(null);
    setPreview(null);
    setTicked([]);
    setKeysTicked(false);
    setSkipped([]);
    try {
      const found = await withTimeout(backup.inspectRestore(restorePass), RESTORE_TIMEOUT_MS, 'That took too long: the check was stopped, try again.');
      if (mounted.current) {
        setPreview(found);
        // Nothing chosen: the passphrase is not kept for a file that was never opened.
        if (!found) setRestorePass('');
      }
    } catch (e) {
      if (mounted.current) {
        setRestoreError(message(e));
        setRestorePass('');
      }
    } finally {
      if (mounted.current) setChecking(false);
    }
  };

  const stage = async () => {
    if (!preview || staging) return;
    const pass = restorePass;
    const id = preview.inspectId;
    // Only what is ticked, and in the order the desktop listed the kinds.
    const classes = preview.classes.filter((c) => ticked.includes(c.id)).map((c) => c.id);
    const keys = keysTicked && preview.keys.inBackup;
    setStaging(true);
    setRestoreError(null);
    let staged: PendingRestore | undefined;
    try {
      const made = await withTimeout(
        backup.stageRestore(id, pass, { classes, keys }),
        RESTORE_TIMEOUT_MS,
        'That took too long: the restore was not prepared, try again.',
      );
      staged = stagedNow(made, classes, preview.classes.filter((c) => ticked.includes(c.id)).map((c) => c.label));
      if (mounted.current) {
        setPreview(null);
        setTicked([]);
        setKeysTicked(false);
        setSkipped(made.skipped ?? []);
      }
    } catch (e) {
      if (mounted.current) {
        // The passphrase is gone with the command, so the summary goes too: check the file again to retry.
        setRestoreError(message(e));
        setPreview(null);
        setTicked([]);
        setKeysTicked(false);
      }
    } finally {
      if (mounted.current) {
        setRestorePass('');
        setStaging(false);
      }
      void refresh(staged);
    }
  };

  const cancelPreview = () => {
    setPreview(null);
    setRestorePass('');
    setRestoreError(null);
    setTicked([]);
    setKeysTicked(false);
  };

  const toggleClass = (id: RestoreClassId) =>
    setTicked((now) => (now.includes(id) ? now.filter((x) => x !== id) : [...now, id]));

  /** An explicit click: tick every kind listed. It does not tick the keys. */
  const selectAll = () => setTicked((preview?.classes ?? []).filter((c) => c.count > 0).map((c) => c.id));

  const restart = async () => {
    setRestarting(true);
    setRestartError(null);
    try {
      await backup.restartToApply();
    } catch (e) {
      if (mounted.current) setRestartError(message(e));
    } finally {
      if (mounted.current) setRestarting(false);
    }
  };

  const discard = async () => {
    setPendingBusy(true);
    setRestartError(null);
    try {
      await backup.discardPending();
      if (mounted.current) {
        setStatus((s) => (s ? { ...s, pendingRestore: null } : s));
        setSkipped([]);
      }
    } catch (e) {
      if (mounted.current) setRestartError(message(e));
    } finally {
      if (mounted.current) setPendingBusy(false);
      void refresh();
    }
  };

  const undo = async () => {
    if (!confirm(undoConfirmText(status?.undoKind ?? null))) return;
    setPendingBusy(true);
    setUndoError(null);
    let staged: PendingRestore | undefined;
    try {
      const made = await backup.undo();
      staged = stagedNow(made);
      if (mounted.current) setSkipped(made.skipped ?? []);
    } catch (e) {
      if (mounted.current) setUndoError(message(e));
    } finally {
      if (mounted.current) setPendingBusy(false);
      void refresh(staged);
    }
  };

  const pending = status?.pendingRestore ?? null;
  const last = status?.lastRestore ?? null;
  const working = creating || checking || staging;

  return (
    <section className="model-section" aria-label="Backup and restore">
      <h3 className="section-title">Backup and restore</h3>
      <p className="form-hint">
        A backup is one encrypted file with your contacts, calendar, conversations, flows, triggers, settings and
        plugin data. Plugin data is backed up plugin by plugin, with PINs, keys and sealed values left out. Keep the
        file somewhere safe, and use it on this computer or a new one to get your data back.
      </p>
      <div className="banner banner-pending" role="note">
        <strong>If you lose the passphrase, the backup is lost.</strong> Nobody, including OAIY, can open it without
        it and nobody can reset it. The passphrase is not stored anywhere.
      </div>
      <p className="form-hint">
        The FormLogic link, the phone pairing, the ChatGPT sign-in and the Hugging Face token are never in a backup:
        you set them up again after a restore. Your API provider keys are added only if you tick the box below, and even then they come back only if you tick them again when you restore.
      </p>

      {/* ---------- a restore waits for a restart ---------- */}
      {pending && (
        <div className="banner banner-pending" role="status" aria-label="Restore ready">
          <p style={{ margin: 0 }}>
            {pending.kind === 'undo'
              ? `An undo is ready (${pending.files} file${pending.files === 1 ? '' : 's'}). Restart OAIY to finish undoing the last restore.`
              : `A restore is ready (${pending.files} file${pending.files === 1 ? '' : 's'}). Restart OAIY to finish restoring.`}
          </p>
          {pending.expired ? (
            <p role="alert" style={{ margin: '6px 0 0' }}>
              Prepared {agoWords(pending.stagedAt)}: that is more than a day, so it will be discarded, not applied, at the
              next start. Cancel it and prepare it again if you still want it.
            </p>
          ) : (
            <p style={{ margin: '6px 0 0' }}>
              Prepared {agoWords(pending.stagedAt)}; it is discarded, not applied, at the next start after{' '}
              {formatTimestamp(pending.expiresAt)}.
            </p>
          )}
          {pending.kind !== 'undo' && (
            <p style={{ margin: '6px 0 0' }}>
              {(pending.classes ?? []).length > 0
                ? `You ticked: ${(pending.classLabels ?? pending.classes ?? []).join(', ')}.`
                : 'Only your data is brought back: nothing that can run or change settings was ticked.'}
            </p>
          )}
          {skipped.length > 0 && (
            <div style={{ margin: '6px 0 0' }}>
              Left out of it:
              <Sentences items={skipped} />
            </div>
          )}
          {pending.agentStorage && (
            <p style={{ margin: '6px 0 0' }}>
              Your Agent’s conversations and projects are put back when the Agent opens after the restart.
            </p>
          )}
          {restartError && (
            <p role="alert" style={{ margin: '6px 0 0' }}>
              {restartError}
            </p>
          )}
          <div className="form-actions" style={{ marginTop: 8 }}>
            <button className="btn btn-primary" disabled={restarting || pendingBusy || pending.expired} onClick={() => void restart()}>
              {pending.kind === 'undo' ? 'Restart to finish the undo' : 'Restart to finish restoring'}
            </button>
            <button className="btn btn-ghost" disabled={restarting || pendingBusy} onClick={() => void discard()}>
              {pending.kind === 'undo' ? 'Cancel undo' : 'Cancel restore'}
            </button>
          </div>
        </div>
      )}

      {/* ---------- how the last restore went ---------- */}
      {/* The reason is the desktop's own words: it says whether everything was put back, or which files were not. */}
      {last && !last.ok && (
        <div className="banner banner-err" role="alert">
          {last.kind === 'undo'
            ? `The undo did not finish (${formatTimestamp(last.at)}). ${last.error ?? 'No reason was given.'}`
            : `The restore did not finish (${formatTimestamp(last.at)}). ${last.error ?? 'No reason was given.'}`}
          {(last.notes ?? []).length > 0 && <Sentences items={last.notes} />}
        </div>
      )}
      {last && last.ok && (
        <div className="datadir-note" role="status">
          {last.kind === 'undo'
            ? `The undo was applied on ${formatTimestamp(last.at)}.`
            : `Restored on ${formatTimestamp(last.at)}.`}
          {AGENT_STATE_WORDS[last.agentStorage] && <div>{AGENT_STATE_WORDS[last.agentStorage]}</div>}
          {(last.notes ?? []).length > 0 && (
            <div>
              Left out or changed:
              <Sentences items={last.notes} />
            </div>
          )}
          {last.redo.length > 0 && (
            <div>
              You still need to:
              <Sentences items={last.redo} />
            </div>
          )}
        </div>
      )}
      {undoError && (
        <div className="banner banner-err" role="alert">
          {undoError}
        </div>
      )}
      {status?.undoAvailable && !pending && (
        <div className="form-actions">
          <button className="btn btn-secondary" disabled={pendingBusy} onClick={() => void undo()}>
            {status.undoKind === 'undo' ? 'Redo: put back what the last undo took away' : 'Undo the last restore'}
          </button>
        </div>
      )}

      {/* ---------- make a backup ---------- */}
      <h4 className="settings-label" style={{ marginTop: 14 }}>
        Make a backup
      </h4>
      <form
        className="dl-form"
        onSubmit={(e) => {
          e.preventDefault();
          void create();
        }}
      >
        <label className="form-row">
          <span>Passphrase (at least {MIN_PASSPHRASE} characters)</span>
          <input
            type="password"
            autoComplete="new-password"
            aria-label="Backup passphrase"
            value={passphrase}
            onChange={(e) => setPassphrase(e.target.value)}
            disabled={creating}
          />
        </label>
        <label className="form-row">
          <span>Type it again</span>
          <input
            type="password"
            autoComplete="new-password"
            aria-label="Backup passphrase again"
            value={again}
            onChange={(e) => setAgain(e.target.value)}
            disabled={creating}
          />
        </label>
        {tooShort && <p className="form-hint">Use at least {MIN_PASSPHRASE} characters.</p>}
        {mismatch && <p className="form-hint">The two passphrases are not the same.</p>}
        <label className="form-row form-row-inline">
          <input
            type="checkbox"
            checked={includeKeys}
            onChange={(e) => setIncludeKeys(e.target.checked)}
            disabled={creating}
          />
          <span>Include my API provider keys</span>
        </label>
        <p className="form-hint">
          Off unless you tick it. It adds the keys OAIY’s own AI gateway holds and, from the Agent, the keys of its
          own providers.
        </p>
        {includeKeys && (
          <p className="form-hint">
            The backup is then as sensitive as the keys themselves: anyone who has the file and the passphrase can
            use them. When you restore, they come back only if you tick them again.
          </p>
        )}
        <div className="form-actions">
          <button type="submit" className="btn btn-primary" disabled={!canCreate}>
            Create backup
          </button>
        </div>
      </form>

      {creating && (
        <div role="status" aria-live="polite">
          <div className="progress-bar" role="progressbar" aria-label="Backup progress" style={{ margin: '8px 0' }}>
            <div className="progress-fill progress-indet" style={{ width: '40%' }} />
          </div>
          <span className="form-hint">{status?.running?.label ?? 'Starting…'}</span>
        </div>
      )}
      {createError && (
        <div className="banner banner-err" role="alert">
          {createError}
        </div>
      )}
      {result && (
        <div className="banner banner-pending" role="status" aria-label="Backup created">
          <strong>Backup created.</strong>
          <div className="settings-value" style={{ marginTop: 6 }}>
            <code className="path-code">{result.path}</code>
            <OpenButton path={result.path} onError={setCreateError} />
          </div>
          <p style={{ margin: '6px 0 0' }}>
            {formatBytes(result.size)} · {result.counts.files} file{result.counts.files === 1 ? '' : 's'}
            {result.counts.agentProjects + result.counts.agentConversations > 0 &&
              ` · the Agent: ${result.counts.agentProjects} project${result.counts.agentProjects === 1 ? '' : 's'}, ${result.counts.agentConversations} conversation${result.counts.agentConversations === 1 ? '' : 's'}`}
          </p>
          {result.verified && (
            <p style={{ margin: '6px 0 0' }}>Verified: the file was decrypted and every item checked after writing.</p>
          )}
          {result.includesKeys && (
            <p style={{ margin: '6px 0 0' }}>
              This backup includes your API provider keys: treat it as carefully as the keys.
            </p>
          )}
          {result.partial.length > 0 && (
            <div style={{ marginTop: 6 }}>
              <strong>Not everything was included:</strong>
              <Sentences items={result.partial} />
            </div>
          )}
        </div>
      )}

      {/* ---------- restore ---------- */}
      <h4 className="settings-label" style={{ marginTop: 14 }}>
        Restore from a backup
      </h4>
      <p className="form-hint">
        Only restore a backup you made yourself. First OAIY checks the file and shows what would change; nothing is
        changed until you restart.
      </p>
      <form
        className="dl-form"
        onSubmit={(e) => {
          e.preventDefault();
          void check();
        }}
      >
        <label className="form-row">
          <span>Passphrase of the backup</span>
          <input
            type="password"
            autoComplete="off"
            aria-label="Passphrase of the backup to restore"
            value={restorePass}
            onChange={(e) => setRestorePass(e.target.value)}
            disabled={working || !!preview}
          />
        </label>
        <div className="form-actions">
          <button type="submit" className="btn btn-secondary" disabled={!restorePass || working || !!preview}>
            Choose backup file and check it
          </button>
        </div>
      </form>
      {restoreError && (
        <div className="banner banner-err" role="alert">
          {restoreError}
        </div>
      )}
      {preview && (
        <div className="datadir-note" role="region" aria-label="What restoring would do">
          <strong>{preview.fileName}</strong>
          <div>
            Made {formatTimestamp(preview.createdAt)} · OAIY {preview.appVersion} · {preview.platform} ·{' '}
            {preview.totalFiles} file{preview.totalFiles === 1 ? '' : 's'}, {formatBytes(preview.totalBytes)}
            {preview.includesKeys && ' · includes API provider keys'}
          </div>
          <div className="settings-grid" style={{ marginTop: 8 }}>
            {preview.categories.map((c) => (
              <div className="settings-row" key={c.id}>
                <span className="settings-label">{c.label}</span>
                <span>
                  {c.added} added · {c.replaced} replaced · {c.unchanged} unchanged · {c.leftAlone} left alone
                </span>
              </div>
            ))}
          </div>
          {preview.lacks.length > 0 && (
            <div style={{ marginTop: 8 }}>
              This backup does not have: {preview.lacks.join(', ')}.
            </div>
          )}
          {preview.partial.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <strong>Warnings from when it was made:</strong>
              <Sentences items={preview.partial} />
            </div>
          )}
          {preview.excluded.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary>Left out on purpose ({preview.excluded.length})</summary>
              <Sentences items={preview.excluded.map((x) => `${x.pattern}: ${x.reason}`)} />
            </details>
          )}
          {preview.notRestored.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary>Not restored ({preview.notRestored.length})</summary>
              <ul style={{ margin: '4px 0 0', paddingLeft: 18, maxHeight: 220, overflowY: 'auto' }}>
                {preview.notRestored.map((n, i) => (
                  <li key={`${n.name}-${i}`}>
                    <code className="path-code">{n.name}</code>: {n.why}
                  </li>
                ))}
              </ul>
            </details>
          )}
          {preview.redo.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <strong>After restoring you will need to:</strong>
              <Sentences items={preview.redo} />
            </div>
          )}
          {preview.notes.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <strong>Notes:</strong>
              <Sentences items={preview.notes} />
            </div>
          )}

          {/* What can run or reconfigure things: every item by name, each kind brought back only if ticked. */}
          <div className="banner banner-pending" role="group" aria-label="What can run or change settings" style={{ marginTop: 12 }}>
            <strong>What can run or change settings</strong>
            <p style={{ margin: '6px 0 0' }}>
              These can run programs, send messages, point OAIY at other servers, or change what OAIY and the Agent are
              allowed to do. Nothing here is brought back unless you tick it, so tick only what you recognise as yours.
              A backup someone else made can hold things you do not want.
            </p>
            {preview.classes
              .filter((c) => c.count > 0)
              .map((c) => {
                const items = preview.items.filter((i) => i.class === c.id);
                return (
                  <div key={c.id} style={{ marginTop: 10 }}>
                    <label className="form-row form-row-inline">
                      <input
                        type="checkbox"
                        aria-label={`Bring back: ${c.label}`}
                        checked={ticked.includes(c.id)}
                        onChange={() => toggleClass(c.id)}
                        disabled={staging}
                      />
                      <span>
                        <strong>{c.label}</strong> ({c.count})
                      </span>
                    </label>
                    <p className="form-hint" style={{ margin: '2px 0 0 24px' }}>
                      {c.description}
                    </p>
                    {items.length > 0 && (
                      <details open style={{ margin: '4px 0 0 24px' }}>
                        <summary>
                          The {items.length} item{items.length === 1 ? '' : 's'}
                        </summary>
                        <ul style={{ margin: '4px 0 0', paddingLeft: 18, maxHeight: 220, overflowY: 'auto' }}>
                          {items.map((i, n) => (
                            <li key={`${i.name}-${n}`}>
                              <code className="path-code">{i.name}</code> — {i.title}: {i.what}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </div>
                );
              })}
            {preview.classes.every((c) => c.count === 0) && <p style={{ margin: '6px 0 0' }}>This backup holds nothing of that kind.</p>}
            <div className="form-actions" style={{ marginTop: 10 }}>
              <button className="btn btn-secondary" disabled={staging} onClick={selectAll}>
                Select all of my own backup
              </button>
            </div>
            {ticked.length === 0 && (
              <p style={{ margin: '6px 0 0' }}>
                Nothing is ticked, so only what carries no words and does nothing comes back (opening hours, the steps between the times offered, and numbers not to be contacted). Everything listed above stays behind until you tick it.
              </p>
            )}
            {preview.keys.inBackup && (
              <div style={{ marginTop: 10 }}>
                <label className="form-row form-row-inline">
                  <input
                    type="checkbox"
                    aria-label="Bring back the API keys that are in this backup"
                    checked={keysTicked}
                    onChange={(e) => setKeysTicked(e.target.checked)}
                    disabled={staging}
                  />
                  <span>Bring back the API keys that are in this backup</span>
                </label>
                <p className="form-hint" style={{ margin: '2px 0 0 24px' }}>
                  The keys come back only when this is ticked, and only for the provider lists you tick above (OAIY’s AI
                  providers and the Agent’s settings). Otherwise those come back without keys.
                </p>
              </div>
            )}
          </div>
          <div className="form-actions" style={{ marginTop: 10 }}>
            <button className="btn btn-primary" disabled={staging} onClick={() => void stage()}>
              Prepare restore
            </button>
            <button className="btn btn-ghost" disabled={staging} onClick={cancelPreview}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
