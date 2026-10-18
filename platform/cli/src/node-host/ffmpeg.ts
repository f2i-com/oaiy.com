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
    // A bare relative path that traverses upward escapes the worker CWD/sandbox — reject it.
    // Value tokens (`copy`, `concat`, `0`, filtergraphs) contain no `..`, so they pass through.
    if (/(^|\/)\.\.(\/|$)/.test(n)) {
      throw new Error(`ffmpeg arg uses an upward (..) path in server mode: ${a}`);
    }
  }
  return a;
}

/** Whether the `-i` at `at` reads a concat list (`-f concat` among the options before it, after the last input). */
function concatInput(args: string[], at: number): boolean {
  for (let j = at - 1; j >= 0 && args[j] !== '-i'; j--) {
    if (args[j] === '-f' && args[j + 1] === 'concat') return true;
  }
  return false;
}

/**
 * A concat list's entries, checked as ffmpeg args are under confinement: each `file '…'` is a local file in the
 * sandbox roots, with no protocol; any other directive than `file`, `duration`, `inpoint` and `outpoint` is
 * refused. Throws on the first that is not.
 */
function checkConcatList(listPath: string): void {
  let text: string;
  try {
    text = fs.readFileSync(listPath, 'utf-8');
  } catch (e) {
    throw new Error(`ffmpeg concat list could not be read: ${e instanceof Error ? e.message : String(e)}`);
  }
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#') || line === 'ffconcat version 1.0') continue;
    const m = /^(\w+)\s+(.*)$/.exec(line);
    const directive = m?.[1] ?? line;
    if (directive !== 'file') {
      if (['duration', 'inpoint', 'outpoint'].includes(directive)) continue;
      throw new Error(`ffmpeg concat list directive '${directive}' is not allowed in server mode`);
    }
    // `file 'a'\''b'` is a'b: the list's own quoting, undone
    const quoted = (m?.[2] ?? '').trim();
    const file = quoted.startsWith("'") ? quoted.replace(/^'|'$/g, '').replace(/'\\''/g, "'") : quoted;
    resolveArg(file);
  }
}

const mimeFor = (fmt: string): string =>
  fmt === 'png' ? 'image/png' : fmt === 'jpg' || fmt === 'jpeg' ? 'image/jpeg' : `image/${fmt}`;

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
  let resolved = ((args.args as unknown[]) ?? []).map((x) => resolveArg(String(x)));
  // Defense-in-depth under confinement: restrict ffmpeg to local file/pipe protocols, so even a
  // protocol that slips past resolveArg's scheme check (e.g. `subfile` with comma-options, or a
  // nested/wrapper protocol) can't reach a remote/exotic source. `-protocol_whitelist` is an
  // input option — emit it before each `-i`. A whitelist the caller set is replaced, not kept: the
  // video append node sets `file,http,https,tcp,tls` for its concat list, and keeping it let a list
  // entry (a URL from the run's input) be fetched from anywhere. A concat list's own entries are
  // files ffmpeg opens without them being args, so each is checked as an arg is.
  if (fsConfinementOn()) {
    const guarded: string[] = [];
    for (let i = 0; i < resolved.length; i++) {
      const a = resolved[i];
      if (a === '-protocol_whitelist') {
        i++;
        continue;
      }
      if (a === '-i') {
        guarded.push('-protocol_whitelist', 'file,pipe');
        const list = resolved[i + 1];
        if (list !== undefined && concatInput(resolved, i)) checkConcatList(list);
      }
      guarded.push(a);
    }
    resolved = guarded;
  }
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
    const { stdout, stderr } = await pexecFile(bin, resolved, { maxBuffer: MAX_BUF });
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
  const empty: VideoInfoLite = { duration: 0, width: 0, height: 0, fps: 0, codec: '', format: '' };
  try {
    const { stdout } = await pexecFile(
      FFPROBE,
      ['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', p],
      { maxBuffer: 1024 * 1024 * 16 },
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

/** `extract_video_frames`: sample frames at an interval into temp images. */
export async function extractVideoFrames(
  srcPath: string,
  options: ExtractOptions,
): Promise<FrameInfo[]> {
  const input = resolvePath(srcPath);
  // Floor the interval so a tiny value can't request an enormous fps (CPU/disk DoS).
  const interval = Math.max(Number(options.intervalSeconds ?? 1) || 1, 0.1);
  // Sanitize the format to a bare alphanumeric extension. It is interpolated into the
  // output filename below, so `../../etc/...` would let ffmpeg write OUTSIDE TMP_ROOT
  // (arbitrary file write). Strip everything but [a-z0-9].
  const fmt = String(options.outputFormat ?? 'png').replace(/[^a-z0-9]/gi, '') || 'png';
  const outDir = path.join(TMP_ROOT, `frames-${randomUUID()}`);
  ensureDir(outDir);

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
  args.push('-i', input, '-vf', vf.join(','));
  // Always bound frame extraction — an untrusted input could otherwise request millions
  // of frames and fill the disk. Configurable via OAIY_FFMPEG_MAX_FRAMES.
  const maxFramesCap = Math.max(1, Math.trunc(Number(process.env.OAIY_FFMPEG_MAX_FRAMES) || 10_000));
  const requested =
    options.maxFrames != null ? Math.trunc(Number(options.maxFrames) || 0) : maxFramesCap;
  args.push('-frames:v', String(Math.min(Math.max(1, requested), maxFramesCap)));
  args.push('-y', path.join(outDir, `frame-%05d.${fmt}`));

  // A failure is said, not answered as no frames: a missing ffmpeg, or a file it could not read, gave an empty list,
  // and the steps after it ran on nothing as if the video had no frames.
  let failed: string | null = null;
  await pexecFile(FFMPEG, args, { maxBuffer: MAX_BUF }).catch((e: NodeJS.ErrnoException & { stderr?: Buffer | string }) => {
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
