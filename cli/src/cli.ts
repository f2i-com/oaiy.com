/**
 * `oaiy` CLI entry. Subcommands:
 *   oaiy run <flow.json>   — execute a flow headlessly, print/save its outputs
 *   oaiy inputs <flow>     — list the inputs a flow expects
 *   oaiy validate <flow>   — parse + structural check, no execution
 */
import { Command, InvalidArgumentError } from 'commander';
import fs from 'node:fs';
import { loadFlowFile, discoverInputs, parseInputs, gatherConstants } from './flow-io';
import { registerManageCommands } from './manage';

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

const program = new Command();
program
  .name('oaiy')
  .description('Run oaiy flows headlessly, with no GUI.')
  .version('0.1.0');

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
  .option('--quiet', 'suppress progress logs on stderr', false)
  .action(async (flowPath: string, opts: Record<string, any>) => {
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
    process.exit(res.success ? 0 : 1);
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
