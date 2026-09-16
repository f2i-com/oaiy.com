/**
 * `oaiy script` — leaf scripts on the ZIPP engine, without a workflow around
 * them: the `protocol/v1/script-*.schema.json` envelope, served by one warm
 * `worker_threads` Worker (`zipp/script-host.ts`).
 *
 *   oaiy script --request <file> [-o <file>]   one request, one response
 *   oaiy script --serve                        requests in, responses out, NDJSON
 *
 * Imports `zipp/artifact` and `zipp/script-host` and nothing of the workflow
 * engine: no `engine.ts`, no `node-host/core` (which installs `__TAURI__`),
 * no module registry. A leaf script has no nodes to call.
 *
 * # `--serve`, the wire protocol (one JSON object per line, both ways)
 *
 *   → { "op": "batch", "id": <string|number>, "request": <script-request> }
 *   ← { "op": "result", "id": <same>, "result": <script-result> }
 *         `result` is the response (`{v, engine, results}`) or the refusal
 *         (`{v, error: {code: "invalid_request", message}}`) — the envelope's
 *         own answer, verbatim. Batches are served one at a time in arrival
 *         order; a batch arriving while one runs waits, it is not refused.
 *   → { "op": "ping" }                     (an "id", if given, is echoed)
 *   ← { "op": "pong", "engine": <engine identity>, "instance": <n>, "jobs": <n> }
 *         `instance` counts engine workers spawned so far — it goes up by one
 *         at every recycle — and `jobs` is how many jobs the current one has
 *         served. Answered at once, even while a batch runs.
 *   → { "op": "shutdown" }
 *         Nothing further is read. Every batch already received is answered,
 *         the worker is terminated, and the process exits 0. No reply line:
 *         the exit is the acknowledgement. EOF on stdin does the same.
 *   ← { "op": "error", "id": <id|null>, "error": { "code": <code>, "message": <text> } }
 *         For a line that is not a batch, ping or shutdown: not JSON, not an
 *         object, an unknown `op`, a batch without an `id` or a `request`
 *         (`code: "invalid_line"`). The stream continues. Also the one line
 *         written before exit 1 when the engine cannot be loaded at all
 *         (`code: "engine_unavailable"`).
 *
 * Whitespace-only lines are ignored. stdout carries protocol lines only: the
 * worker's streams and every diagnostic go to stderr (`cli.ts` already routes
 * `console.log` there). Everything fails closed: a malformed line is one
 * error line, a malformed request is the refusal object, a worker that dies
 * mid-job answers that job `host` and is replaced, a job past its `budgetMs`
 * is `timeout` and its worker is replaced — and the next job runs.
 */
import fs from 'node:fs';
import readline from 'node:readline';

type Writable = { write(chunk: string): unknown };
type Readable = NodeJS.ReadableStream & { pause?: () => unknown };

interface ErrorLine {
  op: 'error';
  id: string | number | null;
  error: { code: 'invalid_line' | 'engine_unavailable'; message: string };
}

const errorLine = (id: string | number | null, code: ErrorLine['error']['code'], message: string): ErrorLine => ({
  op: 'error',
  id,
  error: { code, message },
});

const isPlainObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

/**
 * `--request <file>`: read, run, write. Exit code 0 when every job was
 * attempted (a job's own failure is a result, not a process failure), 1 for a
 * refused request, an unreadable or non-JSON file, or no engine — in which
 * last case nothing is written to `out` and the reason is on stderr.
 */
export async function runScriptFile(requestPath: string, out?: string): Promise<number> {
  let text: string;
  try {
    text = fs.readFileSync(requestPath, 'utf8');
  } catch (e) {
    process.stderr.write(`oaiy script: --request ${requestPath} cannot be read (${(e as NodeJS.ErrnoException).code ?? String(e)})\n`);
    return 1;
  }
  let request: unknown;
  try {
    request = JSON.parse(text);
  } catch (e) {
    process.stderr.write(`oaiy script: --request ${requestPath} is not JSON (${e instanceof Error ? e.message : String(e)})\n`);
    return 1;
  }

  const { loadZippArtifact, EngineUnavailableError } = await import('./zipp/artifact');
  const { ScriptHost } = await import('./zipp/script-host');
  let host: InstanceType<typeof ScriptHost>;
  try {
    host = new ScriptHost(await loadZippArtifact());
  } catch (e) {
    if (e instanceof EngineUnavailableError) {
      process.stderr.write(`oaiy script: ${e.message}\n`);
      return 1;
    }
    throw e;
  }
  try {
    const response = await host.run(request);
    const json = JSON.stringify(response, null, 2);
    if (out) {
      fs.writeFileSync(out, json);
    } else {
      process.stdout.write(json + '\n');
    }
    return 'error' in response ? 1 : 0;
  } finally {
    await host.shutdown();
  }
}

/**
 * `--serve`: the NDJSON loop over `input`/`output` (stdin/stdout). Resolves
 * with the exit code once the stream has ended (EOF or `shutdown`) and every
 * accepted batch has been answered: 0, or 1 when the engine could not load.
 */
export async function serveScript(input: Readable, output: Writable): Promise<number> {
  const write = (line: unknown) => {
    output.write(JSON.stringify(line) + '\n');
  };

  const { loadZippArtifact, EngineUnavailableError } = await import('./zipp/artifact');
  const { ScriptHost } = await import('./zipp/script-host');
  let host: InstanceType<typeof ScriptHost>;
  try {
    host = new ScriptHost(await loadZippArtifact(), { keepWarm: true });
    // The worker is up and has reported the engine before the first line is
    // read, so a first-line ping has an identity to answer with.
    await host.warm();
  } catch (e) {
    if (e instanceof EngineUnavailableError) {
      write(errorLine(null, 'engine_unavailable', e.message));
      return 1;
    }
    throw e;
  }

  const rl = readline.createInterface({ input, crlfDelay: Infinity });
  let inFlight = 0;
  let closing = false;
  let finish!: () => void;
  const done = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const maybeFinish = () => {
    if (closing && inFlight === 0) finish();
  };

  rl.on('line', (line: string) => {
    // `rl.close()` still delivers the lines already buffered in the same
    // chunk; after `shutdown` nothing further is read, as promised.
    if (closing) return;
    if (line.trim() === '') return;
    let message: unknown;
    try {
      message = JSON.parse(line);
    } catch (e) {
      write(errorLine(null, 'invalid_line', `not JSON: ${e instanceof Error ? e.message : String(e)}`));
      return;
    }
    if (!isPlainObject(message)) {
      write(errorLine(null, 'invalid_line', 'a line must be a JSON object with an "op"'));
      return;
    }
    const id = typeof message.id === 'string' || typeof message.id === 'number' ? message.id : null;
    switch (message.op) {
      case 'ping': {
        const pong: Record<string, unknown> = { op: 'pong', engine: host.engine, instance: host.instance, jobs: host.jobsOnInstance };
        if (id !== null) pong.id = id;
        write(pong);
        return;
      }
      case 'shutdown':
        // Stop reading; what was already received is still answered.
        closing = true;
        rl.close();
        maybeFinish();
        return;
      case 'batch': {
        if (id === null) {
          write(errorLine(null, 'invalid_line', 'a batch needs an "id" (string or number)'));
          return;
        }
        if (!('request' in message)) {
          write(errorLine(id, 'invalid_line', 'a batch needs a "request"'));
          return;
        }
        inFlight += 1;
        void host.run(message.request).then(
          (result) => {
            write({ op: 'result', id, result });
          },
          (e) => {
            // `run` does not reject for anything a request did; this is the host's own fault.
            write(errorLine(id, 'invalid_line', `the host failed to run the batch: ${e instanceof Error ? e.message : String(e)}`));
          },
        ).finally(() => {
          inFlight -= 1;
          maybeFinish();
        });
        return;
      }
      default:
        write(errorLine(id, 'invalid_line', `unknown op ${JSON.stringify(message.op)}`));
    }
  });
  rl.on('close', () => {
    closing = true;
    maybeFinish();
  });

  await done;
  input.pause?.();
  await host.shutdown();
  return 0;
}
