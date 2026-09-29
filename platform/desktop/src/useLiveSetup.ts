import { useEffect, useRef, useState } from 'react';
import { setup as setupApi, type CheckOutcome, type PluginRecord, type SetupState } from './api';
import { pluginsToCheck, readSetup, setUpByChecks, type LiveChecks } from './setupFlow';

/**
 * Is a plugin set up by what is true now? For each plugin whose setup version
 * was never finished here but whose steps carry `done` checks (a plugin set up
 * by hand, or before the wizard existed), its checks are run on the desktop
 * now and every 15 seconds while the window is visible; when they all pass it
 * counts as set up, with no clicking through (`pluginSetupStatus`).
 *
 * By plugin id: `true` set up, `false` not, `null` not told yet.
 */

const POLL_MS = 15000;

async function outcome(pluginId: string, stepId: string, which: 'done' | 'when'): Promise<CheckOutcome> {
  try {
    return await setupApi.check(pluginId, stepId, which);
  } catch (e) {
    // A check that cannot run is "not yet", never an error to show.
    return { passed: false, detail: e instanceof Error ? e.message : String(e) };
  }
}

/** Run a plugin's `done` checks (and the `when` of each step that has one), and judge them. */
export async function liveSetUp(record: PluginRecord): Promise<boolean | null> {
  const declared = readSetup(record);
  if (!declared) return null;
  const checks: LiveChecks = { done: {}, when: {} };
  await Promise.all(
    declared.steps
      .filter((s) => s.done !== undefined)
      .flatMap((s) => [
        outcome(record.id, s.id, 'done').then((o) => void (checks.done[s.id] = o)),
        ...(s.when !== undefined ? [outcome(record.id, s.id, 'when').then((o) => void (checks.when[s.id] = o))] : []),
      ]),
  );
  return setUpByChecks(declared, checks);
}

export function useLiveSetup(records: PluginRecord[] | null, state: SetupState | null): Record<string, boolean | null> {
  const [live, setLive] = useState<Record<string, boolean | null>>({});
  const targets = pluginsToCheck(records, state);
  const targetsRef = useRef(targets);
  targetsRef.current = targets;
  // Asked again at once when a plugin starts or stops, or its setup changes.
  const key = targets.map((p) => `${p.id}:${readSetup(p)?.version}:${p.state}`).join(',');

  useEffect(() => {
    if (!key) return;
    let stopped = false;
    const run = async () => {
      const found = await Promise.all(targetsRef.current.map(async (p) => [p.id, await liveSetUp(p)] as const));
      if (!stopped) setLive((prev) => ({ ...prev, ...Object.fromEntries(found) }));
    };
    void run();
    const timer = window.setInterval(() => {
      if (!document.hidden) void run();
    }, POLL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [key]);

  return live;
}
