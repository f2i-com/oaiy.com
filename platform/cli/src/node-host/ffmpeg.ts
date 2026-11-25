/**
 * Native media ops for the Node host — the `run_command` / `get_video_info` /
 * `extract_video_frames(_batch)` commands the runtime's audio/video modules
 * call. The browser shim drives ffmpeg.wasm; here we shell out to a real
 * `ffmpeg`/`ffprobe` on PATH (override with OAIY_FFMPEG / OAIY_FFPROBE), which is
 * faster and — for ffprobe — actually accurate (the wasm shim fakes ffprobe).
 *
 * Contracts mirror ui/src/tauri-shim/ffmpeg.ts so the shared modules consume
 * the results unchanged.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { resolvePath, TMP_ROOT, ensureDir, confineAbsolute, fsConfinementOn } from './context';

const pexecFile = promisify(execFile);

const FFMPEG = process.env.OAIY_FFMPEG || 'ffmpeg';
const FFPROBE = process.env.OAIY_FFPROBE || 'ffprobe';
const MAX_BUF = 1024 * 1024 * 128; // ffmpeg can be chatty on stderr

export interface RunCommandResult {
  code: number;
  stdout: string;
  stderr: string;
}
export interface FrameInfo {
  timestamp: number;
  dataUrl: string;
  index: number;
  path: string;
}
export interface VideoInfoLite {
  duration: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  format: string;
}

/** Map virtual-root / absolute path args onto real files; leave flags alone. */
function resolveArg(a: string): string {
  const n = a.replace(/\\/g, '/');
  if (n.startsWith('/tmp') || n.startsWith('/oaiy') || n.startsWith('/downloads')) {
    return resolvePath(a);
  }
  if (fsConfinementOn()) {
    // Server mode (`oaiy worker`): args are attacker-controlled. ALLOWLIST, not a denylist —
    // reject ANY arg shaped like a URI scheme (`file:`, `http:`, `concat:`, `subfile:`,
    // `crypto:`, `cache:`, `gopher:`, `rtmpt:`, `unix:`, …) so ffmpeg can't read a remote/exotic
    // /local-file source. A denylist always loses to ffmpeg's protocol set + wrapper schemes
    // (`cache:http://…`). Requiring 2+ chars before `:` spares Windows drive letters (`C:`) and
    // value tokens that use `=` before any `:` (`scale=W:H`, `afade=t=out:st=…`).
    if (/^[a-z][a-z0-9+.-]+:/i.test(n)) {
      throw new Error(`ffmpeg arg uses a disallowed protocol/URI scheme in server mode: ${a}`);
    }
    // Drive-relative `C:foo` slips past path.isAbsolute — confine it like an absolute path.
    if (/^[A-Za-z]:(?![\\/])/.test(a) || path.isAbsolute(a)) {
      return confineAbsolute(a);
    }
    // Block filtergraph expressions and virtual inputs that read or write arbitrary files
    // on disk without going through `-i` (e.g., `movie=`, `movie@id=`, `amovie=`, `subtitles=`,
    // `ass=`, `drawtext=textfile=`, `drawtext=fontfile=`, `metadata=mode=print:file=`,
    // `sendcmd`, `asendcmd`, `lut3d`, `haldclut`, `lavfi`, `stats_file=`, `log_path=`,
    // `filename=`, `model_path=`, `sofa=`).
    if (
      /(^|[;,:=\[\]\s])(a?movie|subtitles|ass|a?sendcmd|lut3d|haldclut)(?:@[^=,:;\[\]\s]+)?\s*=/i.test(a) ||
      /(^|[;,:=\[\]\s])(textfile|fontfile|stats_file|log_path|file|filename|model_path|sofa)\s*=/i.test(a)
    ) {
      throw new Error(`ffmpeg filtergraph file-reading expression is not allowed in server mode: ${a}`);
    }
    // A bare relative path that traverses upward escapes the worker CWD/sandbox — reject it.
    // Value tokens (`copy`, `concat`, `0`, filtergraphs) contain no `..`, so they pass through.
    if (/(^|\/)\.\.(\/|$)/.test(n)) {
      throw new Error(`ffmpeg arg uses an upward (..) path in server mode: ${a}`);
    }
  }
  return a;
}

/**
 * Flags that let ffmpeg read filtergraphs/presets from disk, write arbitrary auxiliary files,
 * or disable concat path safety — all rejected under server-mode confinement.
 * Covers stream-specifier suffixes (`-filter_script:v`, `-fpre:a`, `-preset:v`, …) and
 * ffmpeg's `-/option` read-value-from-file syntax.
 */
const DISALLOWED_CONFINED_FLAG =
  /^-(?:\/|filter_script|filter_complex_script|(?:[fvas]pre|preset|sdp_file|vstats_file|passlogfile)(?::|$))/i;

/**
 * `ffprobe` flags that legitimately consume their immediately following argv token as a
 * non-input option value. Any other non-flag token passed to `ffprobe` under confinement
 * is treated as a positional input path and inspected via `inspectConfinedInput`.
 */
const FFPROBE_VALUE_FLAGS = new Set([
  '-v',
  '-loglevel',
  '-print_format',
  '-of',
  '-show_entries',
  '-select_streams',
  '-f',
  '-safe',
  '-probesize',
  '-analyzeduration',
  '-read_intervals',
  '-max_probe_packets',
  '-fpsprobesize',
  '-threads',
]);

/** Max leading bytes inspected in `inspectConfinedInput` for playlists / concat manifests (256 KiB). */
const INSPECT_PROBE_BYTES = 256 * 1024;

/** Whether the `-i` at `at` reads a concat list (`-f concat` among the options before it, after the last input). */
function concatInput(args: string[], at: number): boolean {
  for (let j = at - 1; j >= 0 && args[j] !== '-i'; j--) {
    if (args[j] === '-f' && String(args[j + 1] ?? '').toLowerCase() === 'concat') return true;
  }
  return false;
}

/**
 * Inspect an input file under confinement regardless of whether `-f concat` was specified:
 * auto-probed `ffconcat` manifests are checked through `checkConcatList`, and HLS/DASH playlist
 * headers (`#EXTM3U`, `#EXT-X-`, `<MPD`) are rejected so an attacker-supplied playlist cannot
 * smuggle arbitrary file/network references past CLI arg checks.
 */
function inspectConfinedInput(
  inputArg: string,
  explicitConcat: boolean,
  seen = new Set<string>(),
  requireExists = false,
): void {
  if (inputArg === '-' && !explicitConcat) return;
  const diskPath = path.isAbsolute(inputArg)
    ? confineAbsolute(inputArg)
    : confineAbsolute(path.resolve(TMP_ROOT, inputArg));
  if (seen.has(diskPath)) return;
  if (seen.size >= 16) {
    throw new Error('ffmpeg concat list nesting limit exceeded in server mode');
  }
  seen.add(diskPath);
  if (explicitConcat) {
    checkConcatList(diskPath, seen);
    return;
  }
  let fd: number;
  try {
    fd = fs.openSync(diskPath, 'r');
  } catch (e) {
    const code = (e as NodeJS.ErrnoException)?.code;
    if (!requireExists && (code === 'ENOENT' || code === 'EISDIR' || code === 'ENOTDIR')) return;
    throw new Error(`ffmpeg input could not be inspected: ${e instanceof Error ? e.message : String(e)}`);
  }
  let bytesRead = 0;
  const buf = Buffer.alloc(INSPECT_PROBE_BYTES);
  try {
    while (bytesRead < buf.length) {
      const n = fs.readSync(fd, buf, bytesRead, buf.length - bytesRead, bytesRead);
      if (n <= 0) break;
      bytesRead += n;
    }
  } finally {
    fs.closeSync(fd);
  }
  if (bytesRead === 0) return;
  const slice = buf.subarray(0, bytesRead);
  const nul = slice.indexOf(0);
  const head = (nul === -1 ? slice : slice.subarray(0, nul))
    .toString('utf-8')
    .replace(/^\uFEFF/, '');
  if (/^\s*ffconcat\b/i.test(head)) {
    checkConcatList(diskPath, seen);
    return;
  }
  if (/(?:^|[\r\n])\s*(?:#EXTM3U\b|#EXT-X-)/i.test(head)) {
    throw new Error(`ffmpeg playlist/HLS input is not allowed in server mode: ${inputArg}`);
  }
  if (/<MPD(?:\s|>|$)/i.test(head)) {
    throw new Error(`ffmpeg DASH/MPD input is not allowed in server mode: ${inputArg}`);
  }
}

/**
 * A concat list's entries, checked as ffmpeg args are under confinement: each `file '…'` is a local file in the
 * sandbox roots, with no protocol; any other directive than `file`, `duration`, `inpoint` and `outpoint` is
 * refused. Throws on the first that is not.
 */
function checkConcatList(listPath: string, seen = new Set<string>()): void {
  let text: string;
  try {
    text = fs.readFileSync(listPath, 'utf-8');
  } catch (e) {
    throw new Error(`ffmpeg concat list could not be read: ${e instanceof Error ? e.message : String(e)}`);
  }
  const baseDir = path.dirname(listPath);
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#') || line === 'ffconcat version 1.0') continue;
    const m = /^(\w+)\s+(.*)$/.exec(line);
    const directive = m?.[1] ?? line;
    if (directive !== 'file') {
      if (['duration', 'inpoint', 'outpoint'].includes(directive)) continue;
      throw new Error(`ffmpeg concat list directive '${directive}' is not allowed in server mode`);
    }
    // Only accept either a strict single-quoted literal (`'...'` with no quotes,
    // backslashes, or control chars inside) or a safe bare path (no quotes,
    // backslashes, whitespace, or control chars). This prevents parser differentials
    // against ffmpeg's `av_get_token` (e.g., `http\://...`, `'http'://...`, `'a'\''b'`).
    const quoted = (m?.[2] ?? '').trim();
    if (!quoted) {
      throw new Error('ffmpeg concat list file entry is empty in server mode');
    }
    let file: string;
    if (quoted.startsWith("'")) {
      if (!quoted.endsWith("'") || quoted.length < 2) {
        throw new Error(`ffmpeg concat list file entry has malformed quoting in server mode: ${quoted}`);
      }
      const inner = quoted.slice(1, -1);
      if (!inner || /['"\\\u0000-\u001f]/.test(inner)) {
        throw new Error(`ffmpeg concat list file entry contains disallowed quotes or backslashes in server mode: ${quoted}`);
      }
      file = inner;
    } else {
      if (/['"\\\s\u0000-\u001f]/.test(quoted)) {
        throw new Error(`ffmpeg concat list file entry contains disallowed quotes or backslashes in server mode: ${quoted}`);
      }
      file = quoted;
    }
    const entryResolved = resolveArg(file);
    const entryDiskPath = path.isAbsolute(entryResolved)
      ? confineAbsolute(entryResolved)
      : confineAbsolute(path.resolve(baseDir, entryResolved));
    // Fail closed on missing/unreadable concat entries so a parser/path mismatch cannot skip inspection.
    inspectConfinedInput(entryDiskPath, false, seen, true);
  }
}

const mimeFor = (fmt: string): string =>
  fmt === 'png' ? 'image/png' : fmt === 'jpg' || fmt === 'jpeg' ? 'image/jpeg' : `image/${fmt}`;

/** Resolve and validate `run_command` argv for `ffmpeg`/`ffprobe` (exported for unit tests). */
export function buildRunCommandArgs(command: 'ffmpeg' | 'ffprobe', rawArgs: unknown[]): string[] {
  let resolved = rawArgs.map((x) => resolveArg(String(x)));
  // Defense-in-depth under confinement: restrict ffmpeg/ffprobe to local file/pipe protocols,
  // so even a protocol that slips past resolveArg's scheme check (e.g. `subfile` with
  // comma-options, or a nested/wrapper protocol) can't reach a remote/exotic source.
  // For ffmpeg, `-protocol_whitelist` is a per-input option emitted before each `-i`.
  // For ffprobe, inputs may be passed positionally without `-i`, so `-protocol_whitelist file,pipe`
  // is prepended unconditionally before all args and every positional input is inspected.
  if (fsConfinementOn()) {
    const guarded: string[] = [];
    if (command === 'ffprobe') {
      guarded.push('-protocol_whitelist', 'file,pipe');
    }
    const ffprobeExplicitConcat =
      command === 'ffprobe' &&
      resolved.some((tok, idx) => /^-f(?::.*)?$/i.test(tok) && String(resolved[idx + 1] ?? '').toLowerCase() === 'concat');
    let afterDashDash = false;
    for (let i = 0; i < resolved.length; i++) {
      const a = resolved[i];
      if (!afterDashDash && /^-protocol_whitelist(?::.*)?$/i.test(a)) {
        i++;
        continue;
      }
      if (!afterDashDash && DISALLOWED_CONFINED_FLAG.test(a)) {
        throw new Error(`ffmpeg option ${a} is not allowed in server mode`);
      }
      if (!afterDashDash && /^-safe(?::.*)?$/i.test(a)) {
        const val = resolved[i + 1];
        if (val !== '1') {
          throw new Error(`ffmpeg ${a} ${val ?? ''} is not allowed in server mode`.trim());
        }
      }
      if (!afterDashDash && /^-f(?::.*)?$/i.test(a) && resolved[i + 1] && String(resolved[i + 1]).toLowerCase() === 'lavfi') {
        throw new Error('ffmpeg -f lavfi is not allowed in server mode');
      }
      if (!afterDashDash && a === '-i') {
        if (command === 'ffmpeg') {
          guarded.push('-protocol_whitelist', 'file,pipe');
        }
        guarded.push(a);
        const inputArg = resolved[i + 1];
        if (inputArg !== undefined) {
          inspectConfinedInput(inputArg, command === 'ffprobe' ? ffprobeExplicitConcat : concatInput(resolved, i));
          guarded.push(inputArg);
          i++;
        }
        continue;
      }
      if (command === 'ffprobe') {
        if (!afterDashDash && a === '--') {
          afterDashDash = true;
          guarded.push(a);
          continue;
        }
        if (!afterDashDash && FFPROBE_VALUE_FLAGS.has(a.toLowerCase())) {
          guarded.push(a);
          if (i + 1 < resolved.length) {
            guarded.push(resolved[++i]);
          }
          continue;
        }
        if (afterDashDash || !a.startsWith('-') || a === '-') {
          inspectConfinedInput(a, ffprobeExplicitConcat);
          guarded.push(a);
          continue;
        }
      }
      guarded.push(a);
    }
    resolved = guarded;
  }
  return resolved;
}

/** `run_command`: spawn ffmpeg/ffprobe (only those) with path-resolved args. */
export async function runCommand(args: Record<string, unknown>): Promise<RunCommandResult> {
  const command = String(args.command ?? 'ffmpeg');
  if (command !== 'ffmpeg' && command !== 'ffprobe') {
    return {
      code: -1,
      stdout: '',
      stderr: `run_command: only ffmpeg/ffprobe are permitted, got '${command}'`,
    };
  }
  const bin = command === 'ffprobe' ? FFPROBE : FFMPEG;
  const resolved = buildRunCommandArgs(command, (args.args as unknown[]) ?? []);
  // Pin child CWD to TMP_ROOT so bare relative paths can never read from the inherited worker CWD.
  ensureDir(TMP_ROOT);
  // ffmpeg won't mkdir an output's parent — make sure resolved dirs exist.
  for (const a of resolved) {
    if (a.startsWith(TMP_ROOT) || path.isAbsolute(a)) {
      try {
        ensureDir(path.dirname(a));
      } catch {
        /* best effort */
      }
    }
  }
  try {
    const { stdout, stderr } = await pexecFile(bin, resolved, { maxBuffer: MAX_BUF, cwd: TMP_ROOT });
    return { code: 0, stdout: stdout.toString(), stderr: stderr.toString() };
  } catch (e) {
    const err = e as { code?: number; stdout?: Buffer | string; stderr?: Buffer | string; message?: string };
    return {
      code: typeof err.code === 'number' ? err.code : 1,
      stdout: err.stdout?.toString() ?? '',
      stderr: err.stderr?.toString() ?? String(err.message ?? e),
    };
  }
}

/** `get_video_info`: native ffprobe JSON → the lite shape the runtime expects. */
export async function getVideoInfo(args: Record<string, unknown>): Promise<VideoInfoLite> {
  const p = resolvePath(String(args.path ?? ''));
  if (fsConfinementOn()) {
    inspectConfinedInput(p, false);
  }
  ensureDir(TMP_ROOT);
  const empty: VideoInfoLite = { duration: 0, width: 0, height: 0, fps: 0, codec: '', format: '' };
  const probeArgs = fsConfinementOn()
    ? ['-v', 'quiet', '-protocol_whitelist', 'file,pipe', '-print_format', 'json', '-show_format', '-show_streams', p]
    : ['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', p];
  try {
    const { stdout } = await pexecFile(
      FFPROBE,
      probeArgs,
      { maxBuffer: 1024 * 1024 * 16, cwd: TMP_ROOT },
    );
    const j = JSON.parse(stdout.toString()) as {
      streams?: Array<Record<string, unknown>>;
      format?: Record<string, unknown>;
    };
    const v = (j.streams ?? []).find((s) => s.codec_type === 'video') ?? {};
    const fmt = j.format ?? {};
    let fps = 0;
    if (typeof v.r_frame_rate === 'string') {
      const [n, d] = v.r_frame_rate.split('/').map(Number);
      if (d) fps = n / d;
    }
    return {
      duration: parseFloat(String(fmt.duration ?? v.duration ?? '0')) || 0,
      width: Number(v.width ?? 0),
      height: Number(v.height ?? 0),
      fps,
      codec: String(v.codec_name ?? ''),
      format: String(fmt.format_name ?? '').split(',')[0] ?? '',
    };
  } catch {
    return empty;
  }
}

interface ExtractOptions {
  intervalSeconds?: number;
  startTime?: number;
  endTime?: number;
  maxFrames?: number;
  outputFormat?: string;
  scaleWidth?: number;
  scaleHeight?: number;
}

/** Build the ffmpeg argv for `extractVideoFrames` (exported for unit tests). */
export function buildExtractFrameArgs(
  input: string,
  options: ExtractOptions,
  outDir: string,
  confined: boolean = fsConfinementOn(),
): { args: string[]; interval: number; fmt: string } {
  // Floor the interval so a tiny value can't request an enormous fps (CPU/disk DoS).
  const interval = Math.max(Number(options.intervalSeconds ?? 1) || 1, 0.1);
  // Sanitize the format to a bare alphanumeric extension. It is interpolated into the
  // output filename below, so `../../etc/...` would let ffmpeg write OUTSIDE TMP_ROOT
  // (arbitrary file write). Strip everything but [a-z0-9].
  const fmt = String(options.outputFormat ?? 'png').replace(/[^a-z0-9]/gi, '') || 'png';

  const args = ['-hide_banner', '-loglevel', 'error'];
  if (options.startTime != null) args.push('-ss', String(options.startTime));
  if (options.endTime != null && options.startTime != null) {
    args.push('-t', String(Number(options.endTime) - Number(options.startTime)));
  } else if (options.endTime != null) {
    args.push('-to', String(options.endTime));
  }
  const vf = [`fps=1/${interval}`];
  if (options.scaleWidth || options.scaleHeight) {
    // Coerce to integers — these are interpolated into the -vf filtergraph, so a string
    // like "1,movie=/etc/passwd[v]" would inject extra ffmpeg filters.
    const sw = Math.trunc(Number(options.scaleWidth ?? -1));
    const sh = Math.trunc(Number(options.scaleHeight ?? -1));
    vf.push(`scale=${Number.isFinite(sw) ? sw : -1}:${Number.isFinite(sh) ? sh : -1}`);
  }
  if (confined) {
    args.push('-protocol_whitelist', 'file,pipe');
  }
  args.push('-i', input, '-vf', vf.join(','));
  // Always bound frame extraction — an untrusted input could otherwise request millions
  // of frames and fill the disk. Configurable via OAIY_FFMPEG_MAX_FRAMES.
  const maxFramesCap = Math.max(1, Math.trunc(Number(process.env.OAIY_FFMPEG_MAX_FRAMES) || 10_000));
  const requested =
    options.maxFrames != null ? Math.trunc(Number(options.maxFrames) || 0) : maxFramesCap;
  args.push('-frames:v', String(Math.min(Math.max(1, requested), maxFramesCap)));
  args.push('-y', path.join(outDir, `frame-%05d.${fmt}`));
  return { args, interval, fmt };
}

/** `extract_video_frames`: sample frames at an interval into temp images. */
export async function extractVideoFrames(
  srcPath: string,
  options: ExtractOptions,
): Promise<FrameInfo[]> {
  const input = resolvePath(srcPath);
  if (fsConfinementOn()) {
    inspectConfinedInput(input, false);
  }
  const outDir = path.join(TMP_ROOT, `frames-${randomUUID()}`);
  ensureDir(outDir);

  const { args, interval, fmt } = buildExtractFrameArgs(input, options, outDir);

  // A failure is said, not answered as no frames: a missing ffmpeg, or a file it could not read, gave an empty list,
  // and the steps after it ran on nothing as if the video had no frames.
  let failed: string | null = null;
  await pexecFile(FFMPEG, args, { maxBuffer: MAX_BUF, cwd: TMP_ROOT }).catch((e: NodeJS.ErrnoException & { stderr?: Buffer | string }) => {
    failed = e.code === 'ENOENT'
      ? `ffmpeg is not installed here (looked for '${FFMPEG}'; set OAIY_FFMPEG to it)`
      : `ffmpeg could not read the video: ${(e.stderr?.toString() || e.message || String(e)).trim().slice(0, 500)}`;
  });

  const files = (await fsp.readdir(outDir).catch(() => []))
    .filter((f) => f.endsWith(`.${fmt}`))
    .sort();
  if (failed && files.length === 0) {
    await fsp.rm(outDir, { recursive: true, force: true }).catch(() => {});
    throw new Error(failed);
  }
  const mime = mimeFor(fmt);
  const startT = Number(options.startTime ?? 0);
  const frames: FrameInfo[] = [];
  for (let i = 0; i < files.length; i++) {
    const fp = path.join(outDir, files[i]);
    const buf = await fsp.readFile(fp);
    frames.push({
      timestamp: startT + i * interval,
      dataUrl: `data:${mime};base64,${buf.toString('base64')}`,
      index: i,
      path: fp,
    });
  }
  return frames;
}

/** `extract_video_frames_batch`: a windowed slice + the BatchResult wrapper. */
export async function extractVideoFramesBatch(args: Record<string, unknown>): Promise<unknown> {
  const srcPath = String(args.path ?? '');
  const intervalSeconds = Math.max(Number(args.intervalSeconds ?? 1) || 1, 0.1);
  // Clamp the batch size (it caps frames per call) so a huge value can't request a
  // runaway extraction window.
  const batchSize = Math.min(Math.max(1, Math.trunc(Number(args.batchSize ?? 32) || 32)), 1000);
  const batchIndex = Math.max(0, Math.trunc(Number(args.batchIndex ?? 0) || 0));
  const startTime = batchIndex * batchSize * intervalSeconds;
  const endTime = (batchIndex + 1) * batchSize * intervalSeconds;

  const frames = await extractVideoFrames(srcPath, {
    intervalSeconds,
    startTime,
    endTime,
    maxFrames: batchSize,
    outputFormat: String(args.outputFormat ?? 'png'),
    scaleWidth: args.scaleWidth as number | undefined,
    scaleHeight: args.scaleHeight as number | undefined,
  });
  for (let i = 0; i < frames.length; i++) {
    frames[i].index = batchIndex * batchSize + i;
    frames[i].timestamp = startTime + i * intervalSeconds;
  }
  return {
    frames,
    batchIndex,
    totalBatches: 0,
    totalFrames: frames.length,
    hasMore: frames.length === batchSize,
    nextStartTime: endTime,
  };
}
