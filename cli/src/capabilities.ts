/**
 * `oaiy capabilities --json` — what this build of the CLI is, and can run.
 *
 * Read by whoever is about to hand this process flows: the Desktop's probe
 * before it queues a run, a heartbeat that reports which engines a host has.
 * Everything in it is either the build's own record or measured now:
 *
 *   * `version`   — cli/package.json's, the one place it is written;
 *   * `protocols` — `run`: the `oaiy run` payload shape (`success`, `status`,
 *                   `results`, `output`, `error`, `errorCode`, `engine`);
 *                   `script`: the leaf-script envelope `oaiy script` speaks
 *                   (`protocol/v1/script-{request,result}.schema.json`, and
 *                   `--serve`'s NDJSON framing, `src/script.ts`); `profile`:
 *                   the script-profile document `run --profile` accepts
 *                   (`script-profile.schema.json`). Each version 1;
 *   * `engine`    — the ZIPP release the build was made from (`__ZIPP_ENGINE__`,
 *                   from the installed release's SOURCE.json at build time) and
 *                   whether the staged artifact beside this bundle is that
 *                   release NOW: `status` is `ready` only after the staged
 *                   bytes were read, hashed against the build's digest and
 *                   compiled — the same check every run makes — otherwise
 *                   `unavailable` with the reason. Never taken from the define
 *                   alone: a bundle whose artifact was not staged with it says
 *                   so here rather than at the first run;
 *   * `run`       — what THIS host runs as a WORKFLOW (`oaiy run`, `oaiy
 *                   worker`, the Desktop's flows). `run.languages` is
 *                   deliberately not `engine.languages`: the bundle can execute
 *                   JavaScript and Python, but a flow's logic blocks and
 *                   conditions compile to JavaScript only until the compilers
 *                   emit Python (O1), and a heartbeat that read the engine's
 *                   list would advertise Python flows before any host could run
 *                   them. The two step figures come from the release's
 *                   PROFILE.json at build time (`__ZIPP_RUN_LIMITS__`);
 *   * `script`    — what `oaiy script` runs as a LEAF (one job of the
 *                   envelope). `script.languages` is the engine's list: a
 *                   `python-project` job runs on this build's web-python engine
 *                   today, so saying `['javascript']` here would be the lie the
 *                   other way. The two figures are the `--serve` watchdog's
 *                   default per-job budget and the largest `budgetMs` a job may
 *                   ask for (the schema's ceiling).
 *
 * Imports `./zipp/artifact`, the script host's policy constants and nothing of
 * the engine: the probe must stay cheap, and must not construct a runtime to
 * answer a question about one.
 */
import { version } from '../package.json';
import { SCRIPT_MAX_BUDGET_MS } from 'oaiy-core/src/zipp-script';
import { loadZippArtifact, zippEngineIdentity, type ZippEngineIdentity } from './zipp/artifact';
import { SCRIPT_DEFAULT_BUDGET_MS } from './zipp/script-host';

export interface RunLimits {
  defaultInstructionSteps: number;
  maxInstructionSteps: number;
}

export interface Capabilities {
  version: string;
  protocols: { run: 1; script: 1; profile: 1 };
  engine: ZippEngineIdentity & { status: 'ready' | 'unavailable'; reason?: string };
  run: { languages: ['javascript'] } & RunLimits;
  script: { languages: string[]; defaultBudgetMs: number; maxBudgetMs: number };
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
    protocols: { run: 1, script: 1, profile: 1 },
    engine: status,
    run: { languages: ['javascript'], ...runLimits() },
    // The build's engine record (SOURCE.json via the define), the same list as
    // `engine.languages`: the envelope decides `unsupported` from exactly it.
    script: { languages: [...identity.languages], defaultBudgetMs: SCRIPT_DEFAULT_BUDGET_MS, maxBudgetMs: SCRIPT_MAX_BUDGET_MS },
  };
}
