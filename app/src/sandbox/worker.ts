/// <reference lib="webworker" />
/**
 * A sandbox Worker: one guest program on a fresh Zipp engine, then done.
 *
 * The guest reaches the page through `__coderHostCall`, which rides Zipp's
 * synchronous `ls.getItem` bridge with a reserved key prefix; everything else
 * a guest asks `localStorage` for is inert. The bridge blocks this Worker on
 * shared memory until the page answers (channel.ts). The page ends the run by
 * terminating the Worker at its deadline, which is the only wall clock a
 * guest cannot outwait.
 */
import { blockingCall } from './channel';
import { buildPackage } from './pystd/package';
import type { ConsoleLine, FromWorker, GuestSources, RunRequest, RunResult, ToWorker } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

const MAGIC = '\u0000bot.computer:';

/** The operations a guest may ask the page for (answered by host.ts). */
const HOST_KINDS = ['fs.read', 'fs.readb64', 'fs.write', 'fs.append', 'fs.writeb64', 'fs.stat', 'fs.list', 'fs.walk', 'fs.grep', 'fs.mkdir', 'fs.remove', 'fs.rename', 'fs.copy', 'net.fetch', 'proc.run', 'sys.sleep', 'fs.archive'];

// Installed ahead of the guest: captures the host channel before any guest
// code can shadow it, and turns the page's {ok}|{err} reply into a return
// value or an exception. An engine with Zipp's app bridge answers through
// `host.callSync`; an older one through the `ls.getItem` tunnel.
function hostCallShim(appBridge: boolean): string {
  const send = appBridge
    ? 'var reply = channel.callSync.apply(channel, [String(kind)].concat(args));'
    : `var reply = JSON.parse(channel("ls.getItem", ${JSON.stringify(MAGIC)} + JSON.stringify({ k: String(kind), a: args })));`;
  return `var __coderHostCall = (function (channel) {
  return function (kind) {
    var args = [];
    for (var i = 1; i < arguments.length; i++) args.push(String(arguments[i]));
    ${send}
    if (reply === null || typeof reply !== "object") throw new Error("the sandbox host did not answer");
    if (typeof reply.err === "string") throw new Error(reply.err);
    return String(reply.ok === undefined ? "" : reply.ok);
  };
})(${appBridge ? 'host' : '__zippHostCall'});
`;
}
let HOSTCALL = hostCallShim(false);

interface ZippEngine {
  setLocalStorageBridge(bridge: unknown): void;
  setAppBridge?(bridge: unknown): void;
  setSyncHostCapabilities(ops: string[]): void;
  setInstructionBudget(steps: number): boolean;
  initScript(source: string): unknown;
  initPythonProject(files: unknown, entry: string, argv: unknown): unknown;
  pythonCall(name: string, args: unknown): unknown;
  callFunction(name: string, args: unknown): unknown;
  evalInContext(expr: string): unknown;
  pump(): void;
  takeConsole(): unknown;
  takeFailedConsole(): unknown;
  lastErrorKind(): string;
  dispose(): void;
  free(): void;
}

function post(message: FromWorker): void {
  self.postMessage(message);
}

function consoleLines(raw: unknown): ConsoleLine[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((entry: unknown) => {
    if (Array.isArray(entry)) return { err: /err/i.test(String(entry[0])), text: String(entry[1] ?? '') };
    if (entry && typeof entry === 'object') {
      const e = entry as Record<string, unknown>;
      return { err: /err|warn/i.test(String(e.stream ?? e.level ?? '')), text: String(e.text ?? e.line ?? e.message ?? '') };
    }
    return { err: false, text: String(entry) };
  });
}

function drain(engine: ZippEngine, failed = false): ConsoleLine[] {
  const out: ConsoleLine[] = [];
  for (let i = 0; i < 10_000; i++) {
    let page: ConsoleLine[];
    try {
      page = consoleLines(failed ? engine.takeFailedConsole() : engine.takeConsole());
    } catch {
      break;
    }
    if (!page.length) break;
    out.push(...page);
  }
  return out;
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function isLimit(engine: ZippEngine, message: string): string | undefined {
  let kind = '';
  try {
    kind = engine.lastErrorKind();
  } catch {
    /* disposed */
  }
  if (/budget|limit|exhaust|heap|output/i.test(kind) || /instruction budget|limit exceeded|out of memory|heap limit|output limit/i.test(message)) {
    return message || kind;
  }
  return undefined;
}

function jsProgram(request: RunRequest, guest: GuestSources): string {
  const json = (v: unknown) => JSON.stringify(v);
  return `${HOSTCALL}var __coder_input = ${json({ cwd: request.cwd ?? '/', stdin: request.stdin ?? '', env: request.env ?? {} })}, __coder_argv = ${json(request.argv ?? [])}, __coder_file = ${json(request.fileName ?? 'main.js')}, __coder_source = ${json(request.source)};\n${guest.prelude}\n__coder_main();\n`;
}

function shellProgram(request: RunRequest, guest: GuestSources): string {
  const input = { command: request.source, cwd: request.cwd ?? '/', env: request.env ?? {}, stdin: request.stdin ?? '' };
  return `${HOSTCALL}var __coder_input = ${JSON.stringify(input)};\nvar __shell_tools = ${guest.tools};\nvar __shell_result = ${guest.shell}\n`;
}

async function run(message: Extract<ToWorker, { type: 'run' }>): Promise<RunResult> {
  const glue = (await import(/* @vite-ignore */ message.glueUrl)) as {
    default: (opts: unknown) => Promise<unknown>;
    Engine: new () => ZippEngine;
    addPythonPackage?: (archive: Uint8Array, kernels: unknown) => string;
    pythonPackages?: () => string;
  };
  await glue.default({ module_or_path: message.module });
  const request = message.request;
  // bot.computer's Python additions (standard modules, stdin, the working folder), before any Python engine exists.
  let packageError: string | undefined;
  if (request.lang === 'python' && glue.addPythonPackage && glue.pythonPackages) {
    try {
      const info = JSON.parse(glue.pythonPackages()) as { engineAbi: string; installed: Array<{ name: string }> };
      if (!info.installed.some((p) => p.name === 'botcomputer-stdlib')) glue.addPythonPackage(await buildPackage(info.engineAbi), null);
    } catch (error) {
      packageError = errorText(error);
    }
  }
  const engine = new glue.Engine();
  const sab = message.sab;
  // Never throw from a bridge: Zipp would hand the guest an opaque failure.
  const answer = (kind: string, args: string[]) => {
    try {
      const reply = blockingCall(sab, (m) => self.postMessage(m), { type: 'call', kind, args });
      return JSON.parse(reply);
    } catch (error) {
      return { err: `Error: ${errorText(error)}` };
    }
  };
  const appBridge = typeof engine.setAppBridge === 'function';
  HOSTCALL = hostCallShim(appBridge);
  if (appBridge) {
    engine.setAppBridge!({
      call(kind: string, args: string[]) {
        return HOST_KINDS.includes(kind) ? answer(kind, Array.from(args, String)) : { err: `TypeError: \`${kind}\` is not available in the bot.computer sandbox` };
      },
    });
    engine.setSyncHostCapabilities(HOST_KINDS.map((k) => `app.${k}`));
  } else {
    engine.setLocalStorageBridge({
      getItem(key: string) {
        if (typeof key !== 'string' || !key.startsWith(MAGIC)) return null;
        let call: { k: string; a: string[] };
        try {
          call = JSON.parse(key.slice(MAGIC.length));
        } catch {
          return { err: 'TypeError: malformed host call' };
        }
        return answer(String(call.k), (call.a ?? []).map(String));
      },
      setItem() {},
      removeItem() {},
      clear() {},
    });
    engine.setSyncHostCapabilities(['ls.getItem']);
  }
  engine.setInstructionBudget(request.limits.maxSteps);

  const result: RunResult = { console: [] };
  if (packageError) result.console.push({ text: `[bot.computer's Python additions could not load: ${packageError}]`, err: true });
  try {
    if (request.lang === 'python') {
      const entry = request.fileName ?? 'main.py';
      const files = { ...(request.files ?? {}), [entry]: request.source };
      try {
        engine.initPythonProject(files, entry, request.argv ?? []);
      } catch (error) {
        result.error = errorText(error);
      }
      result.console.push(...drain(engine, result.error !== undefined));
      if (result.error === undefined || !/disposed/i.test(result.error)) {
        try {
          const changed = engine.pythonCall('__zipp_py_vfs_changed', []);
          const parsed = JSON.parse(String(changed));
          if (Array.isArray(parsed?.changes)) result.vfsChanges = parsed.changes;
        } catch {
          /* the program never started */
        }
        try {
          const status = engine.pythonCall('__zipp_py_exit_status', []);
          if (typeof status === 'number' || typeof status === 'bigint') result.exit = Number(status);
          else if (typeof status === 'string' && status) result.exit = status;
        } catch {
          /* no status */
        }
      }
    } else {
      const program = request.lang === 'shell' ? shellProgram(request, message.guest) : jsProgram(request, message.guest);
      let started = true;
      try {
        engine.initScript(program);
        for (let i = 0; i < 64; i++) engine.pump();
      } catch (error) {
        result.error = errorText(error);
        started = !/disposed|source/i.test(safeKind(engine));
      }
      result.console.push(...drain(engine, !started));
      if (started) {
        if (request.lang === 'shell') {
          try {
            const text = engine.evalInContext('__shell_result');
            result.shell = JSON.parse(String(text));
          } catch (error) {
            result.error ??= errorText(error);
          }
        } else {
          try {
            // The prelude defines its hooks on the global object, which
            // `callFunction` (program bindings only) does not search.
            const outcome = JSON.parse(String(engine.evalInContext('__coder_finish()')));
            if (typeof outcome.error === 'string') result.error ??= outcome.error;
            if (typeof outcome.value === 'string') result.value = outcome.value;
            if (outcome.exit !== null && outcome.exit !== undefined) result.exit = outcome.exit;
          } catch (error) {
            result.error ??= errorText(error);
          }
          result.console.push(...drain(engine));
        }
      }
    }
    if (result.error) result.limit = isLimit(engine, result.error);
  } finally {
    try {
      engine.dispose();
      engine.free();
    } catch {
      /* already gone */
    }
  }
  return result;
}

function safeKind(engine: ZippEngine): string {
  try {
    return engine.lastErrorKind();
  } catch {
    return 'disposed';
  }
}

self.onmessage = (event: MessageEvent<ToWorker>) => {
  if (event.data?.type !== 'run') return;
  run(event.data).then(
    (result) => post({ type: 'done', result }),
    (error) => post({ type: 'fatal', message: errorText(error) }),
  );
};
