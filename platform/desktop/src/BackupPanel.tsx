import { useCallback, useEffect, useRef, useState } from 'react';
import { Archive, FolderOpen } from 'lucide-react';
import {
  backup,
  formatBytes,
  formatTimestamp,
  openInExplorer,
  type BackupCreateResult,
  type BackupStatus,
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

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

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

const AGENT_STATE_WORDS: Record<string, string> = {
  applied: 'The Agent’s conversations and projects were restored.',
  pending: 'The Agent’s conversations and projects are restored when the Agent opens.',
  failed: 'The Agent’s conversations and projects could not be restored.',
};

type PendingRestore = NonNullable<BackupStatus['pendingRestore']>;

/** A restore or undo the desktop has just made ready, as the pending banner reads it. */
function stagedNow(staged: StagedRestore): PendingRestore {
  return { id: staged.id, kind: staged.kind, stagedAt: new Date().toISOString(), files: staged.files, agentStorage: staged.agentStorage };
}

const EMPTY_STATUS: BackupStatus = {
  lastBackupAt: null,
  lastBackupOk: null,
  lastBackupSize: null,
  pendingRestore: null,
  lastRestore: null,
  undoAvailable: false,
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
    try {
      const found = await backup.inspectRestore(restorePass);
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
    setStaging(true);
    setRestoreError(null);
    let staged: PendingRestore | undefined;
    try {
      staged = stagedNow(await backup.stageRestore(id, pass));
      if (mounted.current) setPreview(null);
    } catch (e) {
      if (mounted.current) setRestoreError(message(e));
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
  };

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
      if (mounted.current) setStatus((s) => (s ? { ...s, pendingRestore: null } : s));
    } catch (e) {
      if (mounted.current) setRestartError(message(e));
    } finally {
      if (mounted.current) setPendingBusy(false);
      void refresh();
    }
  };

  const undo = async () => {
    if (!confirm('Undo the last restore? OAIY puts back the files it replaced, when you restart.')) return;
    setPendingBusy(true);
    setUndoError(null);
    let staged: PendingRestore | undefined;
    try {
      staged = stagedNow(await backup.undo());
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
        plugin data. Keep it somewhere safe, and use it on this computer or a new one to get everything back.
      </p>
      <div className="banner banner-pending" role="note">
        <strong>If you lose the passphrase, the backup is lost.</strong> Nobody, including OAIY, can open it without
        it and nobody can reset it. The passphrase is not stored anywhere.
      </div>
      <p className="form-hint">
        Sign-ins and keys are never in a backup unless you choose to add your API provider keys below: the FormLogic
        link, the phone pairing, the ChatGPT sign-in and the Hugging Face token are set up again after a restore.
      </p>

      {/* ---------- a restore waits for a restart ---------- */}
      {pending && (
        <div className="banner banner-pending" role="status" aria-label="Restore ready">
          <p style={{ margin: 0 }}>
            {pending.kind === 'undo'
              ? `An undo is ready (${pending.files} file${pending.files === 1 ? '' : 's'}). Restart OAIY to finish undoing the last restore.`
              : `A restore is ready (${pending.files} file${pending.files === 1 ? '' : 's'}). Restart OAIY to finish restoring.`}
          </p>
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
            <button className="btn btn-primary" disabled={restarting || pendingBusy} onClick={() => void restart()}>
              {pending.kind === 'undo' ? 'Restart to finish the undo' : 'Restart to finish restoring'}
            </button>
            <button className="btn btn-ghost" disabled={restarting || pendingBusy} onClick={() => void discard()}>
              {pending.kind === 'undo' ? 'Cancel undo' : 'Cancel restore'}
            </button>
          </div>
        </div>
      )}

      {/* ---------- how the last restore went ---------- */}
      {last && !last.ok && (
        <div className="banner banner-err" role="alert">
          {last.kind === 'undo'
            ? `The undo did not finish and your files were put back as they were: ${last.error ?? 'no reason was given'}`
            : `The restore did not finish and your files were put back: ${last.error ?? 'no reason was given'}`}
        </div>
      )}
      {last && last.ok && (
        <div className="datadir-note" role="status">
          {last.kind === 'undo'
            ? `The last restore was undone on ${formatTimestamp(last.at)}.`
            : `Restored on ${formatTimestamp(last.at)}.`}
          {AGENT_STATE_WORDS[last.agentStorage] && <div>{AGENT_STATE_WORDS[last.agentStorage]}</div>}
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
            Undo the last restore
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
        {includeKeys && (
          <p className="form-hint">
            The backup is then as sensitive as the keys themselves: anyone who has the file and the passphrase can
            use them.
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
          {preview.redo.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <strong>After restoring you will need to:</strong>
              <Sentences items={preview.redo} />
            </div>
          )}
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
