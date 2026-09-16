/**
 * `oaiy` CLI entry. Subcommands:
 *   oaiy run <flow.json>   — execute a flow headlessly, print/save its outputs
 *   oaiy inputs <flow>     — list the inputs a flow expects
 *   oaiy validate <flow>   — parse + structural check, no execution
 *   oaiy capabilities      — what this build is and can run (--json)
 *   oaiy script            — leaf scripts on the ZIPP engine (--request <file> | --serve)
 */
import { Command, InvalidArgumentError } from 'commander';
import fs from 'node:fs';
import { version } from '../package.json';
import { loadFlowFile, discoverInputs, parseInputs, gatherConstants } from './flow-io';
import { registerManageCommands } from './manage';
// Node builtins only (see its imports), so a `--help` still costs nothing.
import { EngineUnavailableError } from './zipp/artifact';

// Make stdout/stderr writes blocking so an explicit process.exit() never
// truncates buffered output (on Windows, writes to a pipe are async). The
// node:sqlite experimental warning is silenced in bin/oaiy.mjs, which runs
// before this bundle (and its hoisted node:sqlite import) loads.
for (const stream of [process.stdout, process.stderr]) {
  const handle = (stream as unknown as { _handle?: { setBlocking?: (b: boolean) => void } })._handle;
  handle?.setBlocking?.(true);
}

// A CLI must keep stdout clean (the result is machine-readable). The shared
// engine emits diagnostic chatter via console.log/info/debug; route those to
// stderr so only our explicit result lands on stdout. console.warn/error keep
// their stderr destination. (The engine itself is lazy-imported in run/worker,
// so management commands stay lightweight.)
for (const m of ['log', 'info', 'debug'] as const) {
  console[m] = (...args: unknown[]) => console.error(...args);
}

/**
 * The engine's step ceiling and default, recorded from its PROFILE.json at
 * build time.
 *
 * `typeof` first, exactly as `capabilities.ts` and `artifact.ts` guard their
 * own defines. This read the define bare, so a bundle built outside
 * `cli/esbuild.mjs` threw a ReferenceError out of an argument parser: exit 1,
 * no result file, and the Desktop's bridge lane reporting a retryable
 * `Failed{}` with a stderr tail for a build that simply has no engine.
 * `capabilities.ts` OWNS this figure, but that module pulls the script host in
 * with it and this is a `--help`-weight argument parser, so the read stays
 * here and only the guard is shared.
 */
function runLimits(): { defaultInstructionSteps: number; maxInstructionSteps: number } {
  if (typeof __ZIPP_RUN_LIMITS__ !== 'string') {
    throw new EngineUnavailableError(
      'ZIPP engine unavailable: this build recorded no run limits (not built by cli/esbuild.mjs)',
    );
  }
  return JSON.parse(__ZIPP_RUN_LIMITS__);
}

const program = new Command();
program
  .name('oaiy')
  .description('Run oaiy flows headlessly, with no GUI.')
  .version(version);

program
  .command('run')
  .description('Execute a flow and print or save its outputs')
  .argument('<flow>', 'path to a .json flow')
  .option('--input <kv...>', 'flow input as KEY=value (repeatable)')
  .option('--inputs <file>', 'JSON file of inputs (merged under --input)')
  .option('--constant <kv...>', 'constant KEY=value (e.g. OPENAI_API_KEY=...) (repeatable)')
  .option('--constants <file>', 'JSON file of constants')
  .option(
    '--connector <file>',
    'JSON connector config from a linked provider; its node types are registered before the run',
  )
  .option('-o, --out <file>', 'write result JSON to a file (default: stdout)')
  .option('--timeout <seconds>', 'run timeout in seconds', (v) => {
    const n = parseInt(v, 10);
    if (!Number.isFinite(n) || n <= 0) {
      throw new InvalidArgumentError('--timeout must be a positive integer (seconds)');
    }
    return n;
  })
  .option(
    '--instruction-budget <steps>',
    "ZIPP instruction budget per script entry, in steps (default: the engine's own; see `capabilities --json`)",
    (v) => {
      // Refused here, before any flow is read or engine imported: a budget the
      // engine cannot take must not become a run on some other budget.
      const { maxInstructionSteps } = runLimits();
      const n = /^\d+$/.test(v) ? Number(v) : NaN;
      if (!Number.isSafeInteger(n) || n < 1 || n > maxInstructionSteps) {
        throw new InvalidArgumentError(`--instruction-budget must be an integer from 1 to ${maxInstructionSteps} (steps)`);
      }
      return n;
    },
  )
  .option(
    '--profile <file>',
    'a script profile (protocol/v1/script-profile.schema.json): its preamble is placed before every flow script and its instructionSteps is the budget (not with --instruction-budget)',
  )
  .option('--quiet', 'suppress progress logs on stderr', false)
  .action(async (flowPath: string, opts: Record<string, any>) => {
    // The profile is loaded, checked and reconciled with --instruction-budget
    // BEFORE the flow is read or the engine imported: a profile that does not
    // check, or two budgets, is a usage error and no result file is written.
    const { loadScriptProfile, resolveInstructionBudget } = await import('./zipp/profile');
    const profile = opts.profile ? loadScriptProfile(opts.profile) : undefined;
    resolveInstructionBudget(profile, opts.instructionBudget); // refuses the pair; `runFlow` resolves the figure
    const { runFlow } = await import('./engine');
    const { graph, name, flows } = loadFlowFile(flowPath);
    const inputs = parseInputs(opts.input ?? [], opts.inputs);
    const constants = gatherConstants(opts.constant ?? [], opts.constants);

    if (!opts.quiet) {
      process.stderr.write(`oaiy: running '${name}' (${graph.nodes.length} nodes)…\n`);
      if (opts.connector) {
        process.stderr.write(`oaiy: connector config ${opts.connector}\n`);
      }
    }

    const res = await runFlow(graph, {
      inputs,
      constants,
      // A linked provider's node types, built from its config at run time.
      connectorPath: opts.connector,
      // Without this, a flow containing a subflow or macro node can't run
      // headlessly — the compiler resolves those targets by id out of this list.
      availableFlows: flows,
      flowName: name,
      timeoutMs: opts.timeout ? opts.timeout * 1000 : undefined,
      instructionBudgetSteps: opts.instructionBudget,
      profile,
    });

    const payload: Record<string, unknown> = {
      success: res.success,
      status: res.status,
      jobId: res.jobId,
      results: res.results,
      // The flow's own answer, beside the per-node map. A host reporting a run
      // to whoever queued it wants THIS; `results` is this engine's working.
      output: res.output,
      error: res.error,
      // Why the host, rather than the flow, ended a run that did not complete
      // (`timeout`, `engine_unavailable`); absent for a flow's own failure.
      errorCode: res.errorCode,
      // The engine the flow ran on — the ZIPP VM in a worker thread, always.
      engine: res.engine,
    };
    if (process.env.OAIY_DEBUG) {
      payload.result = res.result;
      payload.logs = res.logs;
    }
    const text = JSON.stringify(payload, null, 2);
    if (opts.out) {
      fs.writeFileSync(opts.out, text);
      if (!opts.quiet) process.stderr.write(`oaiy: wrote ${opts.out}\n`);
    } else {
      process.stdout.write(text + '\n');
    }

    if (!opts.quiet && !res.success) {
      process.stderr.write(`oaiy: run ${res.status}${res.error ? ` — ${res.error}` : ''}\n`);
    }
    // Let pending socket close callbacks finish before Node tears down libuv.
    // Forced exit after an HTTP flow can crash on Windows despite a successful result.
    process.exitCode = res.success ? 0 : 1;
  });

program
  .command('inputs')
  .description('List the inputs a flow expects')
  .argument('<flow>', 'path to a .json flow')
  .action((flowPath: string) => {
    const { graph } = loadFlowFile(flowPath);
    process.stdout.write(JSON.stringify(discoverInputs(graph), null, 2) + '\n');
  });

program
  .command('validate')
  .description('Parse a flow and check its structure (no execution)')
  .argument('<flow>', 'path to a .json flow')
  .action((flowPath: string) => {
    const { graph, name } = loadFlowFile(flowPath);
    process.stdout.write(
      `ok: '${name}' parsed — ${graph.nodes.length} nodes, ${graph.edges.length} edges\n`,
    );
  });

program
  .command('capabilities')
  .description('Report what this build is and can run: version, run protocol, engine identity and health')
  .option('--json', 'machine-readable JSON on stdout (the only form today)', false)
  .action(async (opts: Record<string, any>) => {
    if (!opts.json) {
      throw new Error('capabilities: pass --json (the report has no other form yet)');
    }
    // src/capabilities imports the artifact check only, never the engine, so
    // a probe costs a hash and a compile of the staged .wasm — not a runtime.
    const { capabilities } = await import('./capabilities');
    const report = await capabilities();
    process.stdout.write(JSON.stringify(report, null, 2) + '\n');
    // The report is printed either way; an unavailable engine is a failed
    // probe, and the exit code says so to a caller that only checks that.
    process.exitCode = report.engine.status === 'ready' ? 0 : 1;
  });

program
  .command('script')
  .description(
    'Run leaf scripts (protocol/v1/script-request.schema.json) on the ZIPP engine: one request from a file, or serve requests over stdin/stdout as NDJSON',
  )
  .option('--request <file>', 'a script-request JSON document; the response goes to --out or stdout')
  .option('-o, --out <file>', 'write the response JSON to a file (default: stdout)')
  .option('--serve', 'keep one warm engine worker and speak NDJSON on stdin/stdout ({op:"batch"|"ping"|"shutdown"})', false)
  .action(async (opts: Record<string, any>) => {
    if (opts.serve && opts.request) throw new InvalidArgumentError('script: pass --request <file> or --serve, not both');
    if (!opts.serve && !opts.request) throw new InvalidArgumentError('script: pass --request <file> or --serve');
    if (opts.serve && opts.out) throw new InvalidArgumentError('script: --out has no meaning with --serve (responses go to stdout)');
    // src/script imports the artifact check and the script host only — never
    // engine.ts, so no __TAURI__ is installed and no module registry loads.
    const { runScriptFile, serveScript } = await import('./script');
    if (opts.serve) {
      const code = await serveScript(process.stdin, process.stdout);
      // Like `worker`: the streams are blocking, the worker has been
      // terminated, and a paused stdin pipe must not keep the process alive.
      process.exit(code);
    }
    process.exitCode = await runScriptFile(opts.request, opts.out);
  });

program
  .command('worker')
  .description('Drive an oaiy-api run queue from a server (long-poll + execute + report)')
  .requiredOption('--backend <url>', 'oaiy-api base URL, e.g. http://localhost:8080')
  .requiredOption('--flow <hashEdit>', 'the flow edit-hash to listen on')
  .option('--constant <kv...>', 'constant KEY=value (repeatable)')
  .option('--constants <file>', 'JSON file of constants')
  .option('--once', 'process at most one queued run, then exit', false)
  .action(async (opts: Record<string, any>) => {
    // Server mode executes untrusted queued inputs — confine absolute fs paths to the
    // sandbox roots + block remote ffmpeg protocols (the node-host reads OAIY_FS_CONFINE
    // at call time). Opt out with OAIY_FS_ALLOW_ABSOLUTE=1 for a trusted single-tenant box.
    if (process.env.OAIY_FS_ALLOW_ABSOLUTE !== '1') process.env.OAIY_FS_CONFINE = '1';
    const { runWorker } = await import('./worker');
    const constants = gatherConstants(opts.constant ?? [], opts.constants);
    const { fatal } = await runWorker({
      backend: opts.backend,
      hashEdit: opts.flow,
      constants,
      once: opts.once,
    });
    // Exit non-zero when the worker bailed on a permanent error so a supervisor
    // (systemd/k8s/pm2) restarts/alerts instead of seeing a false clean stop.
    process.exit(fatal ? 1 : 0);
  });

// Management commands: oaiy python / venv / service / model / server (drive oaiy-server).
registerManageCommands(program);

program.parseAsync(process.argv).catch((e) => {
  process.stderr.write(`oaiy: ${e instanceof Error ? e.message : String(e)}\n`);
  process.exit(1);
});
