/**
 * Executing untrusted workflow scripts on the Zipp JavaScript engine.
 *
 * `untrusted-executor.ts` moves untrusted package code into a Worker so a
 * recovered `globalThis` reaches only the worker's realm. That realm is still a
 * full browser realm, though, so the defence is *subtractive*: take something
 * that has every capability and `defineProperty` the dangerous names away, walk
 * the prototype chain, then scan the source for escape patterns. The comments
 * on both layers say what that costs — "BEST-EFFORT ONLY — NOT a security
 * boundary".
 *
 * Zipp inverts it. Its guest global is a positive allowlist that never held a
 * host object, so a *successful* realm recovery yields nothing:
 *
 *     Function("return this")()            -> object   (Zipp's own global)
 *     Function("return this")().fetch      -> undefined
 *     Function("return this")().Worker     -> undefined
 *     Function("return this")().process    -> undefined
 *     (function(){}).constructor chain     -> undefined
 *
 * The capability is not hidden, it is absent. That makes
 * `HARDENED_SHADOW_PREAMBLE` and `detectSandboxEscape` belt-and-braces rather
 * than the boundary itself. Zipp also enforces an instruction budget, so a
 * runaway `while (true)` stops on its own instead of pinning a core until the
 * user aborts.
 *
 * # Why this needs no change to the compiled script
 *
 * OAIY's emitted program is already a generator trampoline (see
 * `runtime.ts` — `rewriteAwaitToYield` turns every `await` into `yield`, and
 * `__step` drives it). It reaches the host through exactly one function:
 *
 *     host.call(kind: string, args: unknown[], callback: (res) => void): void
 *
 * Zipp's WASM preamble defines that same function with the same deferred
 * contract: the call is queued, the embedder drains it after the current VM
 * re-entry returns, and the callback fires later via `resolveHostCallback`.
 * So the script text does not change at all — what OAIY has to supply is the
 * pump loop that Zipp deliberately leaves to the embedder, which is
 * `ZippSession` below.
 *
 * # The wrapper, and why it is load-bearing
 *
 * One genuine mismatch remains. OAIY evaluates its script as a `new Function`
 * BODY; Zipp prepends its own preamble and compiles a single Program. Left
 * alone that difference is silently destructive:
 *
 *   * `'use strict'` is no longer in the directive prologue, so it degrades to
 *     a no-op string expression and the whole script runs sloppy;
 *   * `HARDENED_SHADOW_PREAMBLE`'s `var window = undefined` stops shadowing a
 *     binding and starts *overwriting Zipp's own* `window`/`navigator`/
 *     `localStorage` bridges at global scope;
 *   * a module named `host`, `db`, `window`, `navigator` or `localStorage`
 *     makes `getModulesShim`'s `let <Name> = {}` a hard SyntaxError against the
 *     preamble's `var` of the same name;
 *   * `console` is a parameter in the `new Function` path but an intrinsic
 *     under Zipp, and the intrinsic cannot be reassigned — output would silently
 *     divert into Zipp's internal buffer, losing both level and interleaving.
 *
 * Wrapping the script in an immediately-invoked function restores `new
 * Function` semantics exactly and fixes all four at once: the directive regains
 * prologue position, the shadow `var`s become function-scoped again, the module
 * `let`s land in a fresh scope, and `host`/`console` become parameters we
 * supply. `buildZippScript` is that wrapper.
 */

/** The subset of Zipp's `Engine` this module drives. */
export interface ZippEngine {
  initScript(source: string): unknown;
  drainPendingHostCalls(): unknown;
  resolveHostCallback(callId: number, result: unknown): void;
  renewInstructionBudget(): boolean;
  takeOutput(): unknown;
  dispose(): void;
}

/** One `host.call` the guest queued, as Zipp hands it back. */
export interface ZippHostCall {
  id: number;
  kind: string;
  args: string[];
}

export type ZippConsoleLevel = 'log' | 'warn' | 'error' | 'info' | 'debug';

export interface ZippSessionHooks {
  /**
   * Service one module call. Mirrors the `host.call` broker in `runtime.ts`:
   * resolve with the handler's return value, or with `{ __error__: string }`
   * to make the guest's `await` throw.
   */
  onHostCall(kind: string, args: string[]): Promise<unknown>;
  onConsole(level: ZippConsoleLevel, args: string[]): void;
  onFinish(value: unknown): void;
  onError(message: string): void;
}

/**
 * Names Zipp's preamble binds at global scope. `getModulesShim` emits
 * `let <ModuleName> = {}`, and `let` after the preamble's `var` of the same
 * name is a SyntaxError that would take out the whole script — so a collision
 * has to be caught before we hand anything to the engine, not after.
 *
 * The IIFE wrapper already moves the module `let`s into their own scope, which
 * makes this unreachable in the current wiring. It stays as a fail-closed
 * assertion: if the wrapper is ever changed or bypassed, the failure it
 * prevents is a SyntaxError with no obvious cause.
 */
export const ZIPP_PREAMBLE_GLOBALS: readonly string[] = [
  'window',
  'navigator',
  'localStorage',
  'db',
  'host',
  '__zEvents',
  '__zHostQueue',
  '__zHostCbs',
  '__zHostId',
  '__zippHostCall',
];

/** The console levels the wrapper bridges, matching `consoleProxy` in `runtime.ts`. */
const CONSOLE_LEVELS: readonly ZippConsoleLevel[] = ['log', 'warn', 'error', 'info', 'debug'];

/** Reserved `kind` values that never reach the module broker. */
const KIND_FINISH = '__system.finish';
const KIND_FINISH_ERROR = '__system.finish_error';
const KIND_CONSOLE = '__system.console';

/**
 * Wrap a compiled OAIY workflow so Zipp evaluates it with `new Function`
 * semantics. See the module comment for why each piece is here.
 *
 * `hardened` mirrors `runtime.ts`: the strict directive and the null-prototype
 * `this` are applied exactly when the in-thread path would have applied them,
 * so the two engines agree on behaviour rather than only on output.
 */
export function buildZippScript(fullScript: string, hardened: boolean): string {
  const strict = hardened ? `'use strict';\n` : '';
  const thisArg = hardened ? 'Object.create(null)' : 'undefined';

  return `var __oaiyConsole = {};
(function () {
  var levels = ${JSON.stringify(CONSOLE_LEVELS)};
  for (var i = 0; i < levels.length; i++) {
    (function (level) {
      __oaiyConsole[level] = function () {
        var parts = [];
        for (var a = 0; a < arguments.length; a++) parts.push(String(arguments[a]));
        // No callback: console is fire-and-forget, so a log line costs a queue
        // entry rather than a full re-entry round trip. Ordering still holds —
        // it is the same FIFO every other host call is drained from.
        host.call(${JSON.stringify(KIND_CONSOLE)}, [level, JSON.stringify(parts)], undefined);
      };
    })(levels[i]);
  }
})();
(function (host, console) {
${strict}// Zipp's synchronous host bridge. It is capability-denied by default (the
// embedder never calls setSyncHostCapabilities), so this only removes a
// misleading affordance — but an untrusted script has no business seeing it.
var __zippHostCall = undefined;
${ZIPP_GUEST_SHIMS}
${fullScript}
}).call(${thisArg}, host, __oaiyConsole);
`;
}

/**
 * Host-realm APIs a code node might reach for, made safe inside the engine.
 *
 * Declared with `var` inside the wrapper's function scope, so they shadow
 * Zipp's intrinsics for the flow's code and are themselves overridden by
 * `HARDENED_SHADOW_PREAMBLE`'s `var x = undefined` when a package flow follows
 * — hardened flows keep exactly the surface they had.
 *
 * Two groups:
 *
 *   * Pure computation the engine simply lacks — UTF-8 text coding, base64,
 *     a structural clone, a microtask hook, a monotonic clock. Provided.
 *   * Anything needing the event loop or the network. Replaced by a function
 *     that throws a message naming the alternative, and NOT left as Zipp's
 *     own. This is load-bearing for timers: Zipp's `setTimeout` with a
 *     non-zero delay tries to sleep a thread that WebAssembly does not have,
 *     panics with "can't sleep", and takes the whole instance down as an
 *     unrecoverable trap. A trusted flow gets no shadow preamble, so without
 *     this one stray `setTimeout(fn, 100)` in a code node would kill its run.
 *
 * `structuredClone` is a JSON round-trip: it handles the plain data flows
 * carry and documents what it drops. `crypto` is deliberately a throwing stub
 * rather than a `Math.random` imitation — code that asks for cryptographic
 * randomness must not silently receive something weaker.
 */
const ZIPP_GUEST_SHIMS = String.raw`// ---- host-realm APIs, made safe inside the engine (see ZIPP_GUEST_SHIMS) ----
var __oaiyUnavailable = function (name, hint) {
  return function () {
    throw new TypeError(name + " is not available inside the Zipp sandbox. " + hint);
  };
};
var __oaiyNoTimers = "Flow code resumes through node calls, not timers; put the work after an await on a node instead.";
var __oaiyNoNet = "Use an HTTP Request node so the host can apply its network policy.";
var setTimeout = __oaiyUnavailable("setTimeout", __oaiyNoTimers);
var setInterval = __oaiyUnavailable("setInterval", __oaiyNoTimers);
var setImmediate = __oaiyUnavailable("setImmediate", __oaiyNoTimers);
var requestAnimationFrame = __oaiyUnavailable("requestAnimationFrame", __oaiyNoTimers);
var requestIdleCallback = __oaiyUnavailable("requestIdleCallback", __oaiyNoTimers);
var clearTimeout = function () {};
var clearInterval = function () {};
var fetch = __oaiyUnavailable("fetch", __oaiyNoNet);
var XMLHttpRequest = __oaiyUnavailable("XMLHttpRequest", __oaiyNoNet);
var WebSocket = __oaiyUnavailable("WebSocket", __oaiyNoNet);
var EventSource = __oaiyUnavailable("EventSource", __oaiyNoNet);
var importScripts = __oaiyUnavailable("importScripts", "Flow code cannot load scripts.");
var Worker = __oaiyUnavailable("Worker", "Flow code cannot start workers.");
var crypto = {
  getRandomValues: __oaiyUnavailable("crypto.getRandomValues", "The sandbox has no entropy source; generate secrets in a node on the host."),
  randomUUID: __oaiyUnavailable("crypto.randomUUID", "The sandbox has no entropy source; generate ids in a node on the host."),
  subtle: undefined
};
var queueMicrotask = function (fn) { Promise.resolve().then(fn); };
var __oaiyT0 = Date.now();
var performance = { now: function () { return Date.now() - __oaiyT0; } };
var structuredClone = function (value) {
  // Plain data only: functions and undefined are dropped, Map/Set/Date become
  // {} / {} / an ISO string, exactly as JSON does. Flows carry JSON-shaped
  // values across nodes, so that is the shape this needs to clone.
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
};
var TextEncoder = function TextEncoder() { this.encoding = "utf-8"; };
TextEncoder.prototype.encode = function (input) {
  var s = String(input === undefined ? "" : input);
  var out = [];
  for (var i = 0; i < s.length; i++) {
    var c = s.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
      var d = s.charCodeAt(i + 1);
      if (d >= 0xdc00 && d <= 0xdfff) { c = 0x10000 + ((c - 0xd800) << 10) + (d - 0xdc00); i++; }
    }
    if (c >= 0xd800 && c <= 0xdfff) c = 0xfffd;
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
    else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  return new Uint8Array(out);
};
var TextDecoder = function TextDecoder(label) {
  var l = String(label || "utf-8").toLowerCase();
  if (l !== "utf-8" && l !== "utf8") throw new RangeError("TextDecoder: only utf-8 is available inside the Zipp sandbox");
  this.encoding = "utf-8";
};
TextDecoder.prototype.decode = function (input) {
  if (input === undefined) return "";
  var b = input instanceof Uint8Array ? input : new Uint8Array(input.buffer ? input.buffer : input);
  var s = "";
  for (var i = 0; i < b.length;) {
    var c = b[i++], cp;
    if (c < 0x80) cp = c;
    else if (c < 0xe0) cp = ((c & 31) << 6) | (b[i++] & 63);
    else if (c < 0xf0) cp = ((c & 15) << 12) | ((b[i++] & 63) << 6) | (b[i++] & 63);
    else cp = ((c & 7) << 18) | ((b[i++] & 63) << 12) | ((b[i++] & 63) << 6) | (b[i++] & 63);
    if (cp > 0xffff) { cp -= 0x10000; s += String.fromCharCode(0xd800 + (cp >> 10), 0xdc00 + (cp & 1023)); }
    else s += String.fromCharCode(cp);
  }
  return s;
};
var __oaiyB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
var btoa = function (input) {
  var s = String(input), out = "";
  for (var i = 0; i < s.length; i += 3) {
    var a = s.charCodeAt(i), b = s.charCodeAt(i + 1), c = s.charCodeAt(i + 2);
    if (a > 255 || (b === b && b > 255) || (c === c && c > 255)) throw new Error("btoa: string contains characters outside the Latin1 range");
    var n = (a << 16) | ((b || 0) << 8) | (c || 0);
    out += __oaiyB64[(n >> 18) & 63] + __oaiyB64[(n >> 12) & 63] +
      (i + 1 < s.length ? __oaiyB64[(n >> 6) & 63] : "=") +
      (i + 2 < s.length ? __oaiyB64[n & 63] : "=");
  }
  return out;
};
var atob = function (input) {
  var s = String(input).replace(/[\t\n\f\r ]/g, "");
  if (s.length % 4 === 1 || /[^A-Za-z0-9+\/=]/.test(s.replace(/=+$/, ""))) throw new Error("atob: the string is not correctly encoded");
  s = s.replace(/=+$/, "");
  var out = "", bits = 0, acc = 0;
  for (var i = 0; i < s.length; i++) {
    acc = (acc << 6) | __oaiyB64.indexOf(s[i]);
    bits += 6;
    if (bits >= 8) { bits -= 8; out += String.fromCharCode((acc >> bits) & 255); }
  }
  return out;
};
// ---- end host-realm shims ----`;

/**
 * Reject a script whose module names would collide with Zipp's preamble.
 * Returns the offending name, or `null` when the script is safe to run.
 */
export function detectZippPreambleCollision(moduleNames: readonly string[]): string | null {
  for (const name of moduleNames) {
    if (ZIPP_PREAMBLE_GLOBALS.includes(name)) return name;
  }
  return null;
}

function asHostCalls(raw: unknown): ZippHostCall[] {
  if (!Array.isArray(raw)) return [];
  const out: ZippHostCall[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') continue;
    const call = entry as { id?: unknown; kind?: unknown; args?: unknown };
    if (typeof call.id !== 'number' || typeof call.kind !== 'string') continue;
    out.push({
      id: call.id,
      kind: call.kind,
      args: Array.isArray(call.args) ? call.args.map(String) : [],
    });
  }
  return out;
}

/**
 * Drives one Zipp `Engine` through one workflow.
 *
 * The loop Zipp leaves to the embedder:
 *
 *   initScript(script)                 // runs the top level; __step() queues call #1
 *   loop {
 *     drainPendingHostCalls()          // [] once the guest is waiting on nothing
 *     for each call:
 *       __system.finish       -> done
 *       __system.finish_error -> failed
 *       __system.console      -> forward, no resolve (the guest never waits)
 *       otherwise             -> await the broker, then
 *                                renewInstructionBudget() + resolveHostCallback()
 *                                which re-enters the guest and may queue more
 *   }
 *
 * Two details that are easy to get wrong and are the difference between this
 * working and deadlocking:
 *
 *   * The completion signal travels on the same queue as ordinary calls, so a
 *     driver that stops draining once it has dispatched everything never learns
 *     the workflow finished. The loop only exits on a terminal kind or a
 *     genuinely empty drain.
 *   * `resolveHostCallback` re-enters the guest synchronously, so calls queued
 *     by that re-entry appear only in a *subsequent* drain. Resolving a batch
 *     without re-draining between entries silently drops work — hence the
 *     explicit FIFO rather than a `for` loop over one drain.
 *
 * The budget is renewed before each re-entry. Zipp's ceiling is a lifetime
 * total, which would put a fuse on the whole workflow; renewing per re-entry
 * converts it into a per-step bound, so no single step can run away while a
 * legitimately long workflow still completes. Renewal is host-only — a method
 * on the Engine binding, unreachable from guest code — so a guest cannot raise
 * its own ceiling.
 */
export class ZippSession {
  private readonly engine: ZippEngine;
  private readonly hooks: ZippSessionHooks;
  private readonly pending: ZippHostCall[] = [];
  private settled = false;
  private aborted = false;

  constructor(engine: ZippEngine, hooks: ZippSessionHooks) {
    this.engine = engine;
    this.hooks = hooks;
  }

  /**
   * Compile and run `script` to completion. Resolves when the workflow has
   * settled; the result itself is delivered through `onFinish` / `onError`,
   * matching the callback shape the Worker protocol already uses.
   */
  async run(script: string): Promise<void> {
    try {
      this.engine.initScript(script);
    } catch (e) {
      this.fail(`Zipp failed to compile the workflow: ${errText(e)}`);
      return;
    }

    try {
      await this.pump();
    } catch (e) {
      // Two things land here. The ordinary one is a resource ceiling —
      // `RangeError: script exceeded its memory budget` or its instruction
      // equivalent — which is sticky and terminal for the run but leaves the
      // Engine usable.
      //
      // The other is a WebAssembly trap (`RuntimeError: unreachable`). That is
      // terminal for the Engine too, and worse: `dispose()` then throws
      // "recursive use of an object detected", so the instance leaks. The
      // module itself survives, and the Worker is discarded after every run
      // regardless, so a trap costs one Worker that was going away anyway.
      //
      // Engine builds before zipp.org `833680d8` could be driven into that
      // trap deliberately, because the heap ceiling was re-checked on an
      // instruction stride and a single instruction can commit megabytes. That
      // is fixed upstream and the vendored artifact carries the fix, but the
      // handling stays: a trap is always possible in principle, and treating
      // one as anything other than terminal would be a way to keep using a
      // poisoned Engine.
      this.fail(errText(e));
    }
  }

  /** Stop servicing calls. The Worker terminates itself right after. */
  abort(): void {
    this.aborted = true;
  }

  private async pump(): Promise<void> {
    this.drainInto();

    while (!this.settled && !this.aborted) {
      const call = this.pending.shift();
      if (!call) {
        // Nothing queued and nothing settled: the guest is not waiting on us
        // and will never wake up on its own — Zipp has no event loop and no
        // firing timers. Treat it as a stalled workflow rather than spinning.
        this.fail(
          'Workflow stalled: the script is not waiting on a host call and never signalled completion',
        );
        return;
      }

      if (call.kind === KIND_FINISH) {
        this.finish(parseFinishValue(call.args[0]));
        return;
      }
      if (call.kind === KIND_FINISH_ERROR) {
        this.fail(String(call.args[0] ?? 'Workflow failed'));
        return;
      }
      if (call.kind === KIND_CONSOLE) {
        this.forwardConsole(call.args);
        continue;
      }

      let result: unknown;
      try {
        result = await this.hooks.onHostCall(call.kind, call.args);
      } catch (e) {
        // The broker itself failed. Hand it to the guest as a normal rejected
        // await so user code can catch it, exactly as the in-thread path does.
        result = { __error__: errText(e) };
      }
      if (this.settled || this.aborted) return;

      this.engine.renewInstructionBudget();
      this.engine.resolveHostCallback(call.id, result);
      this.drainInto();
    }
  }

  private drainInto(): void {
    const calls = asHostCalls(this.engine.drainPendingHostCalls());
    for (const call of calls) this.pending.push(call);
    this.drainEngineOutput();
  }

  /**
   * Anything the guest logged through Zipp's intrinsic `console` rather than
   * the injected one. The wrapper's parameter shadows the intrinsic for all
   * user code, so this is normally empty — it exists so output from an
   * unexpected path is surfaced instead of silently dropped. Level is not
   * recoverable here, so it is reported as `log`.
   */
  private drainEngineOutput(): void {
    let lines: unknown;
    try {
      lines = this.engine.takeOutput();
    } catch {
      return;
    }
    if (!Array.isArray(lines) || lines.length === 0) return;
    for (const line of lines) this.hooks.onConsole('log', [String(line)]);
  }

  private forwardConsole(args: string[]): void {
    const level = CONSOLE_LEVELS.includes(args[0] as ZippConsoleLevel)
      ? (args[0] as ZippConsoleLevel)
      : 'log';
    let parts: string[] = [];
    try {
      const parsed = JSON.parse(args[1] ?? '[]');
      if (Array.isArray(parsed)) parts = parsed.map(String);
    } catch {
      parts = [String(args[1] ?? '')];
    }
    this.hooks.onConsole(level, parts);
  }

  private finish(value: unknown): void {
    if (this.settled) return;
    this.settled = true;
    this.drainEngineOutput();
    this.hooks.onFinish(value);
  }

  private fail(message: string): void {
    if (this.settled) return;
    this.settled = true;
    this.drainEngineOutput();
    this.hooks.onError(message);
  }
}

/**
 * `__system.finish` carries `JSON.stringify(item.value)`. An undefined
 * workflow result stringifies to `undefined`, which is not JSON — fall back to
 * the raw value rather than failing a workflow that simply returned nothing.
 */
function parseFinishValue(raw: string | undefined): unknown {
  if (typeof raw !== 'string' || raw === '') return null;
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

function errText(e: unknown): string {
  if (e instanceof Error) return e.message || String(e);
  return String(e);
}
