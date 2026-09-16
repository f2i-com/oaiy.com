/**
 * `oaiy capabilities --json` — what this build of the CLI is, and can run.
 *
 * Read by whoever is about to hand this process flows: the Desktop's probe
 * before it queues a run, a heartbeat that reports which engines a host has.
 * Everything in it is either the build's own record or measured now:
 *
 *   * `version`   — cli/package.json's, the one place it is written;
 *   * `protocols` — the `run` payload shape (`success`, `status`, `results`,
 *                   `output`, `error`, `errorCode`, `engine`), version 1;
 *   * `engine`    — the ZIPP release the build was made from (`__ZIPP_ENGINE__`,
 *                   from the installed release's SOURCE.json at build time) and
 *                   whether the staged artifact beside this bundle is that
 *                   release NOW: `status` is `ready` only after the staged
 *                   bytes were read, hashed against the build's digest and
 *                   compiled — the same check every run makes — otherwise
 *                   `unavailable` with the reason. Never taken from the define
 *                   alone: a bundle whose artifact was not staged with it says
 *                   so here rather than at the first run;
 *   * `run`       — what THIS host runs. `run.languages` is deliberately not
 *                   `engine.languages`: the bundle can execute JavaScript and
 *                   Python, this CLI runs JavaScript flows only, and a heartbeat
 *                   that read the engine's list would advertise Python before
 *                   any host could run it. The two step figures come from the
 *                   release's PROFILE.json at build time (`__ZIPP_RUN_LIMITS__`).
 *
 * Imports `./zipp/artifact` and nothing of the engine: the probe must stay
 * cheap, and must not construct a runtime to answer a question about one.
 */
import { version } from '../package.json';
import { loadZippArtifact, zippEngineIdentity, type ZippEngineIdentity } from './zipp/artifact';

export interface RunLimits {
  defaultInstructionSteps: number;
  maxInstructionSteps: number;
}

export interface Capabilities {
  version: string;
  protocols: { run: 1 };
  engine: ZippEngineIdentity & { status: 'ready' | 'unavailable'; reason?: string };
  run: { languages: ['javascript'] } & RunLimits;
}

/** The two instruction-step figures the build recorded from PROFILE.json. */
export function runLimits(): RunLimits {
  const limits = JSON.parse(__ZIPP_RUN_LIMITS__) as Partial<RunLimits>;
  const figure = (n: unknown, name: string): number => {
    if (!Number.isInteger(n) || (n as number) < 1) {
      throw new Error(`this build recorded no usable ${name} (${String(n)})`);
    }
    return n as number;
  };
  return {
    defaultInstructionSteps: figure(limits.defaultInstructionSteps, 'defaultInstructionSteps'),
    maxInstructionSteps: figure(limits.maxInstructionSteps, 'maxInstructionSteps'),
  };
}

/**
 * Assemble the report. Resolves whether or not the engine is available — the
 * unavailability is the report — and rejects only when the build itself
 * carries no engine identity, which no artifact could make good.
 */
export async function capabilities(): Promise<Capabilities> {
  const identity = zippEngineIdentity();
  let status: Capabilities['engine'];
  try {
    const artifact = await loadZippArtifact();
    status = { ...artifact.identity, status: 'ready' };
  } catch (e) {
    status = { ...identity, status: 'unavailable', reason: e instanceof Error ? e.message : String(e) };
  }
  return {
    version,
    protocols: { run: 1 },
    engine: status,
    run: { languages: ['javascript'], ...runLimits() },
  };
}
