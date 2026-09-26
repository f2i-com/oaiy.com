/**
 * What the page and a sandbox Worker say to each other.
 *
 * One Worker runs one guest program on a fresh Zipp engine. The guest's only
 * way out is `__coderHostCall(kind, ...args)`, tunnelled through Zipp's
 * synchronous `ls.getItem` bridge: the Worker posts the call to the page and
 * blocks on a SharedArrayBuffer until the page has answered (see channel.ts).
 * The page decides every call (see host.ts), exactly as coder-cli's parent
 * process answers its runner child.
 */

export type Lang = 'js' | 'python' | 'shell';

export interface RunLimits {
  /** Zipp instruction budget for the run. */
  maxSteps: number;
}

export interface RunRequest {
  lang: Lang;
  /** JS/Python source, or the shell command line for `shell`. */
  source: string;
  /** Program file name for messages (`main.py`, `script.js`). */
  fileName?: string;
  argv?: string[];
  stdin?: string;
  /** Guest working directory ("/" is the project root). */
  cwd?: string;
  /** The environment: the shell's exported variables (carried between calls), and a program's. */
  env?: Record<string, string>;
  /** Python only: the files `open()` sees, relative to `cwd`. */
  files?: Record<string, string | { base64: string }>;
  limits: RunLimits;
}

export interface ConsoleLine {
  err: boolean;
  text: string;
}

export interface RunResult {
  console: ConsoleLine[];
  /** JS: the program's completion value, as text. */
  value?: string;
  error?: string;
  /** A resource ceiling hit (instructions, heap, output). */
  limit?: string;
  /** Python `sys.exit` status or traceback text; JS `process.exit` code. */
  exit?: number | string | null;
  /** Python: files created, changed or deleted in its virtual filesystem. */
  vfsChanges?: Array<{ path: string; base64?: string; deleted?: boolean }>;
  /** Shell: its report. */
  shell?: { stdout: string; stderr: string; exit_code: number; cwd: string; env: Record<string, string> };
}

export type ToWorker =
  | { type: 'run'; module: WebAssembly.Module; glueUrl: string; sab: SharedArrayBuffer; request: RunRequest; guest: GuestSources };

export type FromWorker =
  | { type: 'call'; kind: string; args: string[] }
  | { type: 'more' }
  | { type: 'done'; result: RunResult }
  | { type: 'fatal'; message: string };

export interface GuestSources {
  prelude: string;
  shell: string;
}
