/**
 * useBackendIntegration — wire the oaiy-api dispatcher into the UI.
 *
 * Three jobs, all gated on the `isBackendEnabled()` toggle:
 *   1. **Share state** — remember the (hash_view, hash_edit, owner_token)
 *      we got from the last `POST /api/flows` so the Share button can
 *      show "already shared" vs "create share". Persists in
 *      localStorage under `oaiy.activeShare`.
 *   2. **Long-poll dispatcher** — when a flow is shared, run the
 *      `startBackendDispatcher` loop so external POSTs to `/runs`
 *      land in this browser.
 *   3. **Run handler** — wires the run to the caller's `executeRun`
 *      callback. OAIYApp passes an executeRun that resolves the bound
 *      flow, submits it to the live JobQueue runtime (submitJob), and
 *      reports the real outcome back to the backend (with a 10-min
 *      timeout). The end-to-end remote-run path is fully wired.
 *
 * The hook returns:
 *   - `share`        — null when not shared, else the hashes + URL
 *   - `createShare`  — async callback: takes the current flow snapshot
 *                      + optional password, returns the new share
 *   - `dispatchState` — 'off' | 'idle' | 'busy' | 'error' + last detail
 *   - `lastRun`      — the most recently dispatched run (for the toast)
 */

import { useEffect, useRef, useState } from 'react';
import {
  createFlow,
  startBackendDispatcher,
  isBackendEnabled,
  type CreatedFlow,
  type FlowSnapshot,
  type QueuedRun,
  type RunOutcome,
} from '../lib/backendDispatcher';
import {
  setFlowPassword,
  onSharingChanged,
} from '../lib/sharingPrefs';

const SHARE_KEY = 'oaiy.activeShare';

export interface ShareState {
  hash_view: string;
  hash_edit: string;
  owner_token: string;
  /** Composed "open this URL" for sharing. */
  viewUrl: string;
  editUrl: string;
  /** Whether this share is currently password-protected. */
  encrypted: boolean;
  /** The local flow id this share points at — the dispatcher needs it
   *  to know which flow to submit when a remote run arrives. Optional
   *  on the type for forward-compat with shares written before this
   *  field existed; new shares always populate it. */
  flowId?: string;
}

export type DispatchStatus = 'off' | 'idle' | 'busy' | 'error';

export interface DispatchedRun {
  id: number;
  reason: string;
  /** Local clock when we received the run from /runs/pending. */
  receivedAt: number;
}

export interface UseBackendIntegrationOptions {
  /**
   * Async callback to actually run a queued workflow with the supplied
   * inputs. The hook hands you `{id, inputs, reason}`; you typically:
   *   1. Apply the inputs to the active flow's input nodes.
   *   2. Trigger the existing runWorkflow.
   *   3. Read collected outputs.
   *   4. Return them as the RunOutcome.
   *
   * If you haven't wired the runtime yet, return e.g.
   * `{status: 'error', error: 'remote runs are not wired locally yet'}`
   * — the dispatcher will report the error back to the API caller
   * (ChatGPT/Claude/curl) and keep polling.
   */
  executeRun: (run: QueuedRun) => Promise<RunOutcome>;
}

function readShareFromStorage(): ShareState | null {
  try {
    const raw = localStorage.getItem(SHARE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (
      typeof parsed?.hash_view === 'string'
      && typeof parsed?.hash_edit === 'string'
      && typeof parsed?.owner_token === 'string'
    ) {
      return parsed as ShareState;
    }
  } catch { /* ignore corrupt JSON */ }
  return null;
}

function writeShareToStorage(share: ShareState | null): void {
  try {
    if (share === null) localStorage.removeItem(SHARE_KEY);
    else localStorage.setItem(SHARE_KEY, JSON.stringify(share));
  } catch { /* ignore quota errors */ }
}

function composeShareUrls(hashView: string, hashEdit: string): { viewUrl: string; editUrl: string } {
  // Use the current origin + a `?flow=<hash>` query so the loader on
  // boot can pick it up. If the user later deploys behind a different
  // origin, the recipient's browser hits this URL and the same `?flow`
  // logic applies regardless of which deployment served the share.
  const base = `${window.location.origin}${window.location.pathname}`;
  return {
    viewUrl: `${base}?flow=${encodeURIComponent(hashView)}`,
    editUrl: `${base}?flow=${encodeURIComponent(hashEdit)}`,
  };
}

export interface UseBackendIntegrationResult {
  /** Current share state. Null when the user hasn't shared this flow yet. */
  share: ShareState | null;
  /**
   * Create a new share on the backend. Pass a password (non-empty) to
   * encrypt the flow at rest. Throws if the backend is disabled.
   */
  createShare: (snapshot: FlowSnapshot, opts?: { title?: string; password?: string; flowId?: string }) => Promise<ShareState>;
  /** Forget the local copy of the share. Doesn't delete from the backend. */
  forgetShare: () => void;
  dispatchState: DispatchStatus;
  dispatchDetail: string | null;
  lastRun: DispatchedRun | null;
  /** True when backend sharing is currently enabled (build + user toggle). */
  enabled: boolean;
}

export function useBackendIntegration(
  opts: UseBackendIntegrationOptions,
): UseBackendIntegrationResult {
  // The toggle can flip mid-session via Settings — re-evaluate on the
  // 'oaiy:sharing-changed' event so we stop/start the dispatcher
  // without a full reload.
  const [enabled, setEnabled] = useState<boolean>(() => isBackendEnabled());
  useEffect(() => {
    return onSharingChanged(() => setEnabled(isBackendEnabled()));
  }, []);

  const [share, setShare] = useState<ShareState | null>(() => readShareFromStorage());
  const [dispatchState, setDispatchState] = useState<DispatchStatus>('off');
  const [dispatchDetail, setDispatchDetail] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<DispatchedRun | null>(null);

  // Stash the latest executeRun in a ref so the dispatcher's runHandler
  // always sees the freshest closure without restarting the loop.
  const executeRef = useRef(opts.executeRun);
  useEffect(() => { executeRef.current = opts.executeRun; }, [opts.executeRun]);

  const createShare = async (
    snapshot: FlowSnapshot,
    createOpts: { title?: string; password?: string; flowId?: string } = {},
  ): Promise<ShareState> => {
    if (!isBackendEnabled()) {
      throw new Error('backend sharing is disabled — turn it on in Settings → Defaults');
    }
    const created: CreatedFlow = await createFlow(snapshot, createOpts);
    const urls = composeShareUrls(created.hash_view, created.hash_edit);
    const next: ShareState = {
      hash_view:   created.hash_view,
      hash_edit:   created.hash_edit,
      owner_token: created.owner_token,
      viewUrl:     urls.viewUrl,
      editUrl:     urls.editUrl,
      encrypted:   !!createOpts.password,
      flowId:      createOpts.flowId,
    };
    // Remember the password against the edit hash so the dispatcher
    // can decrypt incoming runs without re-prompting.
    if (createOpts.password) {
      setFlowPassword(created.hash_edit, createOpts.password);
    }
    setShare(next);
    writeShareToStorage(next);
    return next;
  };

  const forgetShare = () => {
    setShare(null);
    writeShareToStorage(null);
  };

  // Lifecycle: start the dispatcher when we have a share AND backend
  // is enabled. Stop it on either condition flipping false. Pass a
  // wrapper runHandler that pulls from the executeRef.
  useEffect(() => {
    if (!enabled || share === null) {
      setDispatchState('off');
      setDispatchDetail(null);
      return;
    }
    setDispatchState('idle');
    setDispatchDetail(null);
    const stop = startBackendDispatcher({
      hashEdit: share.hash_edit,
      runHandler: async (run): Promise<RunOutcome> => {
        setLastRun({ id: run.id, reason: run.reason, receivedAt: Date.now() });
        return executeRef.current(run);
      },
      onPoll: (state, detail) => {
        setDispatchState(state);
        setDispatchDetail(detail ?? null);
      },
      onError: (err) => {
        // The dispatcher already calls onPoll('error', ...) — this is
        // just the fallback log. Don't double-set state.
        // eslint-disable-next-line no-console
        console.warn('[backendDispatcher]', err);
      },
    });
    return stop;
  }, [enabled, share]);

  return {
    share,
    createShare,
    forgetShare,
    dispatchState,
    dispatchDetail,
    lastRun,
    enabled,
  };
}
