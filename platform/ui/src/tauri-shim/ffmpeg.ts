/**
 * Browser ffmpeg.wasm shim.
 *
 * Replaces the desktop's `plugin:oaiy-filesystem|run_command` sidecar call
 * (which on desktop spawns the bundled `ffmpeg` / `ffprobe` binary) with
 * an in-browser execution via @ffmpeg/ffmpeg.
 *
 * Strategy
 * --------
 * Desktop ffmpeg ops pass real disk paths in their args list and expect
 * ffmpeg to read/write those files. The browser equivalent uses ffmpeg.wasm's
 * MEMFS — a virtual filesystem inside the WASM module — so we have to:
 *
 *   1. Scan the args for path-shaped strings.
 *   2. For each path that already exists in the renderer's webVfs
 *      (see tauri-shim/core.ts), copy its bytes into MEMFS at the same path.
 *   3. Run ffmpeg.
 *   4. For each path that was an OUTPUT (existed in args but not in webVfs
 *      before the run), read MEMFS and write the bytes back to webVfs so
 *      downstream nodes can consume the result via plugin:oaiy-filesystem|read_file.
 *   5. Clean up MEMFS so we don't leak across calls.
 *
 * Loading
 * -------
 * Lazy singleton. First call loads the ~25 MB WASM (and the smaller JS
 * glue) from unpkg.com via the @ffmpeg/util toBlobURL helper, which
 * sidesteps CORP for the WASM/JS fetches. Subsequent calls reuse the same
 * FFmpeg instance.
 *
 * Threading
 * ---------
 * Currently uses the SINGLE-threaded ffmpeg-core build for portability.
 * The multi-threaded build (@ffmpeg/core-mt) needs cross-origin-isolation
 * (COOP/COEP headers) and SharedArrayBuffer; we set those headers in
 * vite.config.ts so we can switch later, but the default avoids a
 * mysterious "Failed to load core" error if a deployer forgets the
 * headers in production.
 */

import { FFmpeg, FFFSType } from '@ffmpeg/ffmpeg';
import { toBlobURL } from '@ffmpeg/util';

// Pin a specific @ffmpeg/core version to match the @ffmpeg/ffmpeg API.
// Bumped 2026-05-28 to match @ffmpeg/ffmpeg ^0.12.15.
//
// SELF-HOSTED by default. This used to fetch the core from
// `https://unpkg.com/@ffmpeg/core@<v>/dist/umd`, which meant every video and
// audio node needed the public internet and trusted a third-party CDN at
// runtime — in an app whose whole claim is that media processing happens on your
// machine. The `?url` imports below make the bundler emit both files as hashed
// assets instead, so they are versioned with the app and work offline.
//
// The cost is honest and large: ffmpeg-core.wasm is ~31 MB. It is lazily
// fetched, only when a flow first touches a media node, so it never affects page
// load. If you would rather serve it from a CDN or your own mirror, set
// VITE_FFMPEG_CORE_BASE at build time to a base URL holding ffmpeg-core.js and
// ffmpeg-core.wasm.
// Relative, not '@ffmpeg/core/...': the package's exports map only exposes "."
// and "./wasm" and both point at the ESM build, while ffmpeg.wasm's coreURL
// needs the UMD one. A relative path resolves the real file directly.
import ffmpegCoreJsUrl from '../../node_modules/@ffmpeg/core/dist/umd/ffmpeg-core.js?url';
import ffmpegCoreWasmUrl from '../../node_modules/@ffmpeg/core/dist/umd/ffmpeg-core.wasm?url';

const FFMPEG_CORE_OVERRIDE = (
  (import.meta.env?.VITE_FFMPEG_CORE_BASE as string | undefined) ?? ''
).replace(/\/$/, '');

let ffmpegInstance: FFmpeg | null = null;
let loadPromise: Promise<FFmpeg> | null = null;

/**
 * Lazy-init the FFmpeg singleton. Safe to call concurrently — multiple
 * callers share the same loadPromise.
 */
async function getFFmpeg(): Promise<FFmpeg> {
  if (ffmpegInstance) return ffmpegInstance;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    const ff = new FFmpeg();

    // Wire log + progress to the console so the user sees what's going on
    // during the (initially long) load + first transcode. Each node can
    // listen for these too via the ffmpeg events API if it wants UI
    // progress; for now the console is plenty.
    ff.on('log', ({ type, message }) => {
      // ffmpeg-wasm uses 'fferr' / 'ffout' / 'info' — surface the actual
      // ffmpeg stderr/stdout as `[ffmpeg]` so it's grep-able.
      if (type === 'fferr' || type === 'stderr') {
        // ffmpeg writes ALL diagnostics to stderr including progress, so
        // route at info level not error.
        // eslint-disable-next-line no-console
        console.info(`[ffmpeg] ${message}`);
      } else {
        // eslint-disable-next-line no-console
        console.debug(`[ffmpeg:${type}] ${message}`);
      }
    });

    // eslint-disable-next-line no-console
    console.info('[ffmpeg-shim] loading ffmpeg-core (~25 MB, first time only)…');
    const t0 = performance.now();
    await ff.load({
      coreURL: await toBlobURL(
        FFMPEG_CORE_OVERRIDE ? `${FFMPEG_CORE_OVERRIDE}/ffmpeg-core.js` : ffmpegCoreJsUrl,
        'text/javascript',
      ),
      wasmURL: await toBlobURL(
        FFMPEG_CORE_OVERRIDE ? `${FFMPEG_CORE_OVERRIDE}/ffmpeg-core.wasm` : ffmpegCoreWasmUrl,
        'application/wasm',
      ),
    });
    // eslint-disable-next-line no-console
    console.info(`[ffmpeg-shim] loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

    ffmpegInstance = ff;
    return ff;
  })();

  // If the load fails (unpkg down, network blip, hostile firewall),
  // clear loadPromise so the NEXT call gets a fresh attempt instead of
  // being permanently stuck on the same rejected promise. Use a tail
  // catch — leave the rejection visible to the awaiter that triggered
  // the load so they still see the error.
  loadPromise.catch(() => {
    loadPromise = null;
  });

  return loadPromise;
}

/**
 * Best-effort: split the args list into (inputs, outputs, all-paths).
 *
 * Heuristic — ffmpeg conventions:
 *   - `-i <path>` flags an input. Take the next arg literally.
 *   - The output path is the FINAL non-flag arg.
 *   - Everything else might also be a path (filter sidecars, drawtext
 *     textfile=, concat list files). We additionally scan for any string
 *     that looks like an absolute path AND exists in the provided VFS map,
 *     copying those into MEMFS too — covers the cases where the path is
 *     baked inside a filter expression.
 */
interface ClassifiedArgs {
  /** Paths that should be present in MEMFS BEFORE ffmpeg runs. */
  inputs: Set<string>;
  /** Paths ffmpeg is expected to write — read back after. */
  outputs: Set<string>;
}

function classifyArgs(
  args: string[],
  vfsHas: (p: string) => boolean,
  vfsKeys: () => string[],
): ClassifiedArgs {
  const inputs = new Set<string>();
  const outputs = new Set<string>();

  // Track which arg INDICES are the value of an `-i` flag — those are
  // input filenames and must not be misclassified as the output by the
  // "last positional wins" heuristic below. (Without this, an args list
  // like `['-i', 'FILE']` ends up with FILE in both inputs and outputs,
  // the subtraction step removes it from inputs, and ffmpeg runs
  // against a path that was never staged into MEMFS.)
  const inputArgIndices = new Set<number>();
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '-i' && i + 1 < args.length) {
      inputs.add(args[i + 1]);
      inputArgIndices.add(i + 1);
      i++;
      continue;
    }
  }

  // Output = last non-flag arg that is NOT an input value, IF it looks
  // path-shaped. For ffprobe-style `ffmpeg -i FILE` (no real output)
  // the only positional candidate IS an input — leave outputs empty.
  for (let i = args.length - 1; i >= 0; i--) {
    const a = args[i];
    if (typeof a !== 'string') continue;
    if (a.startsWith('-')) break;
    if (inputArgIndices.has(i)) continue;
    // Skip values that are obviously not paths (codec ids, integers, etc.)
    if (!a.includes('/') && !a.includes('.')) break;
    outputs.add(a);
    break;
  }

  // Anything else in the args that exists as an EXACT VFS key — picks up
  // concat list paths passed without -i, subtitle sidecars passed bare,
  // and the textfile= arg of a drawtext when the runtime quotes the path
  // standalone.
  for (const a of args) {
    if (typeof a !== 'string') continue;
    if (vfsHas(a)) inputs.add(a);
  }

  // Some args embed a VFS path INSIDE a larger expression — e.g. ffmpeg
  // filter graphs like `drawtext=textfile='/tmp/captions.txt':font='…'`
  // or `subtitles='/tmp/subs.srt':force_style='…'`. For each existing
  // VFS key, see if any arg contains it as a substring; if yes, stage
  // that key too. Skips very short keys (<3 chars) to avoid spurious
  // matches on numeric strings.
  const allKeys = vfsKeys();
  // Chars that may legitimately precede an embedded VFS path in a filter
  // expression (delimiter, quote, assignment, list-sep, or a leading dir
  // separator), and chars that may legitimately follow it (quote, delimiter,
  // list-sep, or whitespace). end-of-string is accepted on both sides so the
  // bare `subtitles=/path` (path at the very end) case still matches. Using
  // boundaries instead of a raw substring test avoids prefix false-positives
  // (e.g. key `/tmp/a.txt` matching arg `/tmp/a.txt.bak`).
  const beforeOk = new Set(["'", '"', '=', ':', ',', '/']);
  const afterOk = new Set(["'", '"', ':', ',', ' ']);
  for (const key of allKeys) {
    if (key.length < 3) continue;
    for (const a of args) {
      if (typeof a !== 'string') continue;
      if (a === key) continue;
      let found = false;
      let from = 0;
      for (;;) {
        const idx = a.indexOf(key, from);
        if (idx < 0) break;
        const before = idx === 0 ? null : a[idx - 1];
        const afterPos = idx + key.length;
        const after = afterPos >= a.length ? null : a[afterPos];
        const beforeAccepted = before === null || beforeOk.has(before);
        const afterAccepted = after === null || afterOk.has(after);
        if (beforeAccepted && afterAccepted) {
          found = true;
          break;
        }
        from = idx + 1;
      }
      if (found) {
        inputs.add(key);
        break;
      }
    }
  }

  // If a path ended up in both buckets (legitimate edge case: in-place
  // transform writing to its own input), trust the output classification
  // and don't stage from VFS. The runtimes don't currently do in-place
  // updates, but the rule is safer here than upstream.
  for (const o of outputs) inputs.delete(o);

  return { inputs, outputs };
}

/**
 * If any of the staged input files looks like an ffmpeg concat list
 * (lines of `file '/path/to/something'`), parse out those referenced
 * paths and return them so the caller can stage them into MEMFS too.
 * Other-than-concat-list text inputs return [].
 */
function extractConcatListRefs(textContent: string): string[] {
  const refs: string[] = [];
  // ffmpeg concat-demuxer format: `file 'path'` or `file "path"` or `file path`.
  // We tolerate single-quotes (most common), double-quotes, and bareword.
  const re = /^\s*file\s+(?:'([^']+)'|"([^"]+)"|(\S+))\s*$/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(textContent)) !== null) {
    const p = m[1] ?? m[2] ?? m[3];
    if (p) refs.push(p);
  }
  return refs;
}

/**
 * Strip any leading slashes and convert path separators so the same
 * "path" works as a MEMFS filename. MEMFS keys are flat strings; if we
 * keep the original `/foo/bar/baz.mp4` ffmpeg treats it as an absolute
 * mount — works on the linux-emulated wasm FS but matches up cleanly
 * with the input arg, so we keep paths verbatim.
 *
 * Currently a no-op (kept as a hook in case future ffmpeg-core versions
 * become picky about leading slashes — both 0.12 builds I've tested
 * accept them).
 */
function memfsName(p: string): string {
  return p;
}

/**
 * MEMFS doesn't auto-create parent directories. For a path like
 * `/oaiy/output/foo.mp4`, writeFile or ffmpeg's own output write fails
 * with "No such file or directory" unless `/oaiy` and `/oaiy/output`
 * already exist. Walk the parents from root and createDir each one;
 * the calls are idempotent (existing dirs throw, we swallow).
 *
 * Skips paths inside a WORKERFS mount because that filesystem is
 * read-only and creating dirs there errors. Mount paths are identified
 * by the `/wfs_` prefix we use when generating mount points.
 */
async function ensureParentDirs(ff: FFmpeg, filePath: string): Promise<void> {
  if (filePath.startsWith('/wfs_')) return;
  const segments = filePath.split('/').filter((s) => s.length > 0);
  // The last segment is the filename — slice off; everything else
  // is the directory chain from root.
  let accum = '';
  for (let i = 0; i < segments.length - 1; i++) {
    accum += '/' + segments[i];
    try { await ff.createDir(accum); } catch { /* already exists or root */ }
  }
}

/**
 * Pick a filename for a WORKERFS mount that preserves the original
 * extension (ffmpeg uses it for codec probing).
 *
 * The picker shim embeds the original filename in the blob: URL as a
 * URL fragment — e.g. `blob:http://origin/uuid#photo.png`. Prefer that
 * when present (it's the actual filename the user chose); fall back to
 * heuristics on the path tail for non-picker paths.
 */
function basenameForMount(originalPath: string): string {
  // Pickker-embedded filename in the URL fragment wins.
  const hashIdx = originalPath.lastIndexOf('#');
  if (hashIdx >= 0 && hashIdx < originalPath.length - 1) {
    const frag = originalPath.slice(hashIdx + 1);
    try {
      const decoded = decodeURIComponent(frag);
      if (decoded && /\.[A-Za-z0-9]{1,6}$/.test(decoded)) return decoded;
    } catch {
      // malformed encoding — fall through to tail heuristics
    }
  }
  const tail = originalPath.split(/[/\\]/).pop() || '';
  // Strip any blob:/data: scheme noise, query strings, AND fragments.
  const clean = tail
    .replace(/^blob:.*\//, '')
    .replace(/\?.*$/, '')
    .replace(/#.*$/, '');
  if (clean && /\.[A-Za-z0-9]{1,6}$/.test(clean)) return clean;
  // Try to recover an extension from the rest of the path.
  const extMatch = originalPath.match(/\.([A-Za-z0-9]{1,6})(?:[?#]|$)/);
  const ext = extMatch ? extMatch[1].toLowerCase() : 'bin';
  return `mounted.${ext}`;
}

export interface RunCommandResult {
  code: number;
  stdout: string;
  stderr: string;
}

/**
 * Optional helpers passed by the caller (core.ts). We can't import core.ts
 * back here without creating a cycle, so we take the VFS bridge as
 * parameters.
 */
export interface VfsBridge {
  /** Return raw bytes for a VFS path, or null if absent. Async to allow
   *  blob:/data: URL fetches + IndexedDB recovery for paths uploaded via
   *  the file picker. */
  readBytes: (path: string) => Promise<Uint8Array | null>;
  /** Persist bytes for a VFS path so other shim handlers (read_file etc.) can serve them. */
  writeBytes: (path: string, bytes: Uint8Array, contentType?: string) => void;
  /** Whether the path currently exists in VFS. Sync — best-effort hint
   *  for the classifier; readBytes is the authoritative path. */
  has: (path: string) => boolean;
  /** Enumerate every key currently in VFS. Used to detect paths embedded
   *  inside filter expressions (drawtext=textfile='...', subtitles='...'). */
  keys: () => string[];
  /**
   * Return a Blob handle for the path WITHOUT materialising its bytes
   * into a Uint8Array. The shim uses this to mount the file via WORKERFS
   * (zero-copy disk-backed mount) instead of copying bytes through the
   * WASM heap. Multi-GB video inputs become feasible because Blobs are
   * disk-backed by the browser; ffmpeg streams chunks as it reads.
   *
   * Return null for paths that don't have an underlying Blob (typed
   * webVfs entries that already live in JS memory — using a Blob there
   * wouldn't save anything, and the writeFile fallback handles them).
   */
  getBlob?: (path: string) => Promise<Blob | null>;
}

/**
 * Run an ffmpeg command using ffmpeg.wasm.
 *
 * Mirrors the desktop run_command result shape:
 *   { code: number; stdout: string; stderr: string }
 *
 * On any thrown error we still return a result object with code !== 0
 * so the calling runtime sees the same failure shape as the desktop
 * sidecar (which would set code = exit code on failure).
 */
export async function runFFmpegCommand(
  command: 'ffmpeg' | 'ffprobe',
  args: string[],
  vfs: VfsBridge,
): Promise<RunCommandResult> {
  // ffprobe support — ffmpeg.wasm doesn't ship ffprobe as a separate
  // binary. For the small number of probe queries the runtimes do (mainly
  // "get duration"), we re-implement using `ffmpeg -i <file>` which writes
  // the same metadata to stderr, then parse out the requested field.
  //
  // The audio_fade node is the only current ffprobe caller and it asks
  // for `format=duration`. Implementing the general -show_entries query
  // language inside the shim is overkill; we recognise the duration query
  // specifically and synthesise stdout. Extend this switch as new probe
  // patterns appear.
  if (command === 'ffprobe') {
    return runFfprobeShim(args, vfs);
  }

  let ff: FFmpeg;
  try {
    ff = await getFFmpeg();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { code: -1, stdout: '', stderr: `ffmpeg-shim: failed to load ffmpeg-core: ${msg}` };
  }

  const { inputs, outputs } = classifyArgs(args, vfs.has, vfs.keys);

  // Stage inputs into MEMFS. Three passes:
  //   1. If the VFS bridge can give us a Blob handle (file picker
  //      uploads have one), mount it via WORKERFS — zero-copy, the
  //      browser streams chunks from disk. Critical for multi-GB videos.
  //   2. Otherwise (typed webVfs entries, intermediates from chained
  //      nodes), fall back to writeFile + bytes copy.
  //   3. For text inputs (concat lists, subtitle sidecars), parse out
  //      nested `file '...'` references and stage those too — required
  //      by audio_append / video_append.
  //
  // Track both copied paths (deleteFile to clean up) and mount points
  // (unmount + remove the synthetic dir). Also build a path-rewrite map
  // so we can swap the original arg string for the in-MEMFS mount path
  // before passing to ff.exec — ffmpeg sees the mount path as a real
  // file.
  const stagedPaths: string[] = [];
  const mountedDirs: string[] = [];
  const pathRewrites = new Map<string, string>();

  const mountInputAsBlob = async (
    path: string,
    blob: Blob,
  ): Promise<RunCommandResult | null> => {
    // Pick a stable name inside a per-input mount dir. Reuse the
    // basename of the original path when possible so ffmpeg's filename
    // probing (extension-based codec detection, etc.) still works.
    const base = basenameForMount(path);
    const dir = `/wfs_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const mountedPath = `${dir}/${base}`;
    try {
      await ff.createDir(dir);
    } catch {
      // ignored — createDir is idempotent in practice
    }
    try {
      await ff.mount(
        FFFSType.WORKERFS,
        { blobs: [{ name: base, data: blob }] },
        dir,
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return { code: -1, stdout: '', stderr: `ffmpeg-shim: WORKERFS mount failed for ${path}: ${msg}` };
    }
    mountedDirs.push(dir);
    pathRewrites.set(path, mountedPath);
    return null;
  };

  const stageFile = async (path: string): Promise<RunCommandResult | null> => {
    if (stagedPaths.includes(path) || pathRewrites.has(path)) return null;

    // Prefer mount when available. Skip mounting for tiny inputs (<1 MB)
    // because the per-mount overhead (createDir + mount + unmount +
    // deleteDir) costs more than the byte copy at that size, and concat
    // lists / subtitle sidecars are always tiny utf-8.
    if (vfs.getBlob) {
      try {
        const blob = await vfs.getBlob(path);
        if (blob && blob.size >= 1024 * 1024) {
          const err = await mountInputAsBlob(path, blob);
          if (err) return err;
          // For text-likeness check we still need bytes — but only
          // tiny inputs are text and they bypass the mount path, so
          // we don't need to re-read here. Return early.
          return null;
        }
      } catch {
        // Fall through to byte-copy path on any blob fetch failure.
      }
    }

    const bytes = await vfs.readBytes(path);
    if (!bytes) {
      return {
        code: -1,
        stdout: '',
        stderr: `ffmpeg-shim: input file not found in browser VFS: ${path}`,
      };
    }
    try {
      await ensureParentDirs(ff, memfsName(path));
      await ff.writeFile(memfsName(path), bytes);
      stagedPaths.push(path);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return { code: -1, stdout: '', stderr: `ffmpeg-shim: writeFile failed for ${path}: ${msg}` };
    }
    // Pass 3: concat-list expansion for small text inputs.
    if (bytes.length < 64 * 1024) {
      let asText = '';
      try {
        asText = new TextDecoder('utf-8', { fatal: false }).decode(bytes);
      } catch {
        return null;
      }
      const refs = extractConcatListRefs(asText);
      for (const ref of refs) {
        if (vfs.has(ref)) {
          const sub = await stageFile(ref);
          if (sub) return sub;
        } else {
          return {
            code: -1,
            stdout: '',
            stderr: `ffmpeg-shim: concat list ${path} references missing file: ${ref}`,
          };
        }
      }
    }
    return null;
  };

  for (const path of inputs) {
    const err = await stageFile(path);
    if (err) return err;
  }

  // Rewrite args so any mounted-input path is swapped for the mount
  // location. Done as a substring replace because the path can appear
  // as a bare arg (`-i path`) OR embedded in a filter (`subtitles=path:…`).
  const execArgs = args.map((a) => {
    if (typeof a !== 'string') return a;
    let out = a;
    for (const [orig, mounted] of pathRewrites) {
      if (out === orig) {
        out = mounted;
      } else if (out.includes(orig)) {
        out = out.split(orig).join(mounted);
      }
    }
    return out;
  });

  // Pre-create the parent directory tree for every output path.
  // ffmpeg's output write errors with "No such file or directory" if
  // the parent dir isn't present in MEMFS.
  for (const out of outputs) {
    await ensureParentDirs(ff, out);
  }

  // Capture ffmpeg's stderr (where it logs progress + diagnostics) so we
  // can return it to the caller for nodes that grep it (e.g. duration
  // parsing). The log listener already prints to console; here we just
  // tee into a buffer for this run.
  const stderrChunks: string[] = [];
  const teeListener = ({ type, message }: { type: string; message: string }) => {
    if (type === 'fferr' || type === 'stderr') stderrChunks.push(message);
  };
  ff.on('log', teeListener);

  let code = 0;
  let runError: unknown;
  try {
    // exec returns 0 on success, non-zero on ffmpeg failure.
    code = await ff.exec(execArgs);
  } catch (e) {
    runError = e;
    code = -1;
  } finally {
    ff.off('log', teeListener);
  }

  if (runError) {
    const msg = runError instanceof Error ? runError.message : String(runError);
    stderrChunks.push(`ffmpeg-shim: exec threw: ${msg}`);
  }

  // Read outputs back even on non-zero exit — some ffmpeg failures still
  // produce a partial output we want to surface (or at least not lose
  // silently).
  if (code === 0) {
    for (const path of outputs) {
      try {
        const data = await ff.readFile(memfsName(path));
        if (data instanceof Uint8Array) {
          vfs.writeBytes(path, data, guessContentType(path));
        } else if (typeof data === 'string') {
          // readFile with default 'binary' returns Uint8Array; in case
          // someone switches it to 'utf8' in the future, handle that too.
          const enc = new TextEncoder().encode(data);
          vfs.writeBytes(path, enc, guessContentType(path));
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        stderrChunks.push(`ffmpeg-shim: output not produced (${path}): ${msg}`);
        code = code || -1;
      }
    }
  }

  // Best-effort cleanup so we don't accumulate MEMFS state across calls.
  // Failures are non-fatal — files/dirs may already be gone (e.g. ffmpeg
  // crashed mid-write) and surfacing those errors would mask the real
  // exec failure above.
  for (const path of [...stagedPaths, ...outputs]) {
    try {
      await ff.deleteFile(memfsName(path));
    } catch {
      // intentionally ignored
    }
  }
  // Unmount + remove the synthetic mount dirs from the WORKERFS inputs.
  // Must unmount BEFORE deleteDir or Emscripten errors with EBUSY.
  for (const dir of mountedDirs) {
    try { await ff.unmount(dir); } catch { /* ignored */ }
    try { await ff.deleteDir(dir); } catch { /* ignored */ }
  }

  return { code, stdout: '', stderr: stderrChunks.join('') };
}

/**
 * Tiny shim for the `ffprobe -show_entries format=duration` query, which
 * is the only probe currently invoked from the audio_fade runtime. Runs
 * `ffmpeg -i <file>` and parses the `Duration: HH:MM:SS.ms` line out of
 * stderr.
 */
async function runFfprobeShim(args: string[], vfs: VfsBridge): Promise<RunCommandResult> {
  // The duration-query call passes the file as the LAST arg.
  const file = args[args.length - 1];
  if (!file) {
    return { code: -1, stdout: '', stderr: 'ffprobe-shim: no file path in args' };
  }

  const wantsDuration =
    args.includes('format=duration') ||
    args.some((a) => typeof a === 'string' && a.includes('duration'));

  // Run `ffmpeg -i <file>` and discard stdout — duration is in stderr.
  const probe = await runFFmpegCommand('ffmpeg', ['-i', file], vfs);
  if (probe.code !== 0 && !probe.stderr.includes('Duration:')) {
    // ffmpeg returns non-zero when there's no output spec, but it still
    // prints metadata first. Only fail if we got nothing useful.
    return probe;
  }

  if (!wantsDuration) {
    // For unknown probe queries, return ffmpeg's stderr as stdout so the
    // caller at least gets the raw metadata to parse.
    return { code: 0, stdout: probe.stderr, stderr: '' };
  }

  // Parse "Duration: HH:MM:SS.ms"
  const m = probe.stderr.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/);
  if (!m) {
    return { code: -1, stdout: '', stderr: 'ffprobe-shim: could not parse Duration from ffmpeg output' };
  }
  const seconds =
    parseInt(m[1], 10) * 3600 + parseInt(m[2], 10) * 60 + parseFloat(m[3]);
  return { code: 0, stdout: `${seconds}\n`, stderr: '' };
}

// ---------------------------------------------------------------------------
// Higher-level helpers used by node runtimes that historically called
// dedicated Tauri commands (extract_video_frames, get_video_info) instead
// of run_command. We re-implement them on top of the ffmpeg.wasm core so
// the calling runtimes (core-video's frame extractor + last-frame helpers)
// don't have to detect browser vs desktop.
// ---------------------------------------------------------------------------

export interface FrameInfo {
  timestamp: number;
  dataUrl: string;
  index: number;
  path: string;
}

export interface ExtractFramesOptions {
  intervalSeconds: number;
  startTime?: number;
  endTime?: number;
  maxFrames?: number;
  outputFormat?: string;
  scaleWidth?: number;
  scaleHeight?: number;
}

/**
 * Extract frames from a video at the requested cadence. Returns one
 * FrameInfo per emitted frame; each frame is also written to the VFS so
 * downstream nodes can read it via plugin:oaiy-filesystem|read_file.
 */
export async function extractVideoFrames(
  sourcePath: string,
  options: ExtractFramesOptions,
  vfs: VfsBridge,
): Promise<FrameInfo[]> {
  let ff: FFmpeg;
  try {
    ff = await getFFmpeg();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`extractVideoFrames: failed to load ffmpeg-core: ${msg}`);
  }

  // Stage the input — prefer WORKERFS mount for blob-backed sources
  // (file picker uploads) so multi-GB videos don't try to live in the
  // WASM heap. Falls back to copy for typed VFS entries.
  let inputName: string;
  let mountedDir: string | null = null;
  let writeFileName: string | null = null;

  let blob: Blob | null = null;
  if (vfs.getBlob) {
    try {
      blob = await vfs.getBlob(sourcePath);
    } catch {
      blob = null;
    }
  }

  if (blob && blob.size >= 1024 * 1024) {
    const base = basenameForMount(sourcePath);
    const dir = `/wfs_fx_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    try { await ff.createDir(dir); } catch { /* idempotent */ }
    await ff.mount(FFFSType.WORKERFS, { blobs: [{ name: base, data: blob }] }, dir);
    mountedDir = dir;
    inputName = `${dir}/${base}`;
  } else {
    const bytes = await vfs.readBytes(sourcePath);
    if (!bytes) {
      throw new Error(`extractVideoFrames: source not in VFS: ${sourcePath}`);
    }
    inputName = memfsName(sourcePath);
    writeFileName = inputName;
    await ensureParentDirs(ff, inputName);
    await ff.writeFile(inputName, bytes);
  }

  // Pick an output dir + filename pattern. Use a per-call subdir so
  // listDir afterwards only sees frames from this run. Random suffix
  // because two extractions kicked off in the same millisecond would
  // collide on Date.now() alone (the calling runtime sometimes does
  // exactly this for timestamp-based last-frame probes).
  const outDir = `frames_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
  try {
    await ff.createDir(outDir);
  } catch {
    // already exists — fine
  }
  const ext = (options.outputFormat || 'png').toLowerCase().replace(/^\./, '');
  const pattern = `${outDir}/frame_%05d.${ext}`;

  // Build the ffmpeg args.
  const fps = options.intervalSeconds > 0
    ? `1/${options.intervalSeconds}`
    : '30';
  const vf: string[] = [`fps=${fps}`];
  if (options.scaleWidth || options.scaleHeight) {
    const w = options.scaleWidth ?? -2;
    const h = options.scaleHeight ?? -2;
    vf.push(`scale=${w}:${h}`);
  }

  const args: string[] = [];
  if (options.startTime !== undefined && options.startTime > 0) {
    args.push('-ss', String(options.startTime));
  }
  if (options.endTime !== undefined && options.endTime > 0) {
    args.push('-to', String(options.endTime));
  }
  args.push('-i', inputName);
  args.push('-vf', vf.join(','));
  if (options.maxFrames && options.maxFrames > 0) {
    args.push('-frames:v', String(options.maxFrames));
  }
  args.push('-y', pattern);

  // Tee ffmpeg stderr while we run, mostly so failures land in console.
  const stderrChunks: string[] = [];
  const listener = ({ type, message }: { type: string; message: string }) => {
    if (type === 'fferr' || type === 'stderr') stderrChunks.push(message);
  };
  ff.on('log', listener);

  let code = 0;
  try {
    code = await ff.exec(args);
  } finally {
    ff.off('log', listener);
  }

  if (code !== 0) {
    // Clean up before throwing.
    if (writeFileName) { try { await ff.deleteFile(writeFileName); } catch { /* noop */ } }
    if (mountedDir) {
      try { await ff.unmount(mountedDir); } catch { /* noop */ }
      try { await ff.deleteDir(mountedDir); } catch { /* noop */ }
    }
    throw new Error(
      `extractVideoFrames: ffmpeg exec returned ${code}. stderr:\n${stderrChunks.join('').slice(-2000)}`,
    );
  }

  // Discover the frames ffmpeg emitted.
  let entries: { name: string; isDir?: boolean }[] = [];
  try {
    entries = (await ff.listDir(outDir)) as { name: string; isDir?: boolean }[];
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (writeFileName) { try { await ff.deleteFile(writeFileName); } catch { /* noop */ } }
    if (mountedDir) {
      try { await ff.unmount(mountedDir); } catch { /* noop */ }
      try { await ff.deleteDir(mountedDir); } catch { /* noop */ }
    }
    throw new Error(`extractVideoFrames: listDir failed: ${msg}`);
  }

  // listDir returns "." and ".." too; filter them and our prefix.
  const frameFiles = entries
    .filter((e) => !e.isDir && e.name.startsWith('frame_'))
    .map((e) => e.name)
    .sort();

  const frames: FrameInfo[] = [];
  const baseTime = (options.startTime ?? 0);
  const mimeType = `image/${ext === 'jpg' ? 'jpeg' : ext}`;

  for (let i = 0; i < frameFiles.length; i++) {
    const memfsPath = `${outDir}/${frameFiles[i]}`;
    const data = await ff.readFile(memfsPath);
    const frameBytes = data instanceof Uint8Array ? data : new TextEncoder().encode(String(data));

    // Persist in VFS at a stable path so downstream nodes can read it.
    const vfsPath = `/tmp/${outDir}/${frameFiles[i]}`;
    vfs.writeBytes(vfsPath, frameBytes, mimeType);

    // Also build a data: URL so callers can use the frame inline without
    // a second read_file round-trip. Chunked btoa for multi-MB images.
    let bin = '';
    const CHUNK = 8192;
    for (let j = 0; j < frameBytes.length; j += CHUNK) {
      const slice = frameBytes.subarray(j, j + CHUNK);
      bin += String.fromCharCode.apply(null, Array.from(slice));
    }
    const dataUrl = `data:${mimeType};base64,${btoa(bin)}`;

    // Approximate timestamp: ffmpeg writes frames at the requested fps,
    // so frame_i lands at start + i*interval.
    const timestamp = baseTime + i * (options.intervalSeconds || 0);

    frames.push({ timestamp, dataUrl, index: i, path: vfsPath });

    // Clean MEMFS — we already have what we need in VFS / dataUrl.
    try { await ff.deleteFile(memfsPath); } catch { /* noop */ }
  }

  try { await ff.deleteDir(outDir); } catch { /* noop */ }
  // Mirror the same cleanup fork the failure branches use: WORKERFS
  // mounts must be unmounted (deleteFile on a path inside a WORKERFS
  // mount is a no-op since the FS is read-only — the previous code
  // leaked one mount + one synthetic dir per call into MEMFS).
  if (writeFileName) {
    try { await ff.deleteFile(writeFileName); } catch { /* noop */ }
  }
  if (mountedDir) {
    try { await ff.unmount(mountedDir); } catch { /* noop */ }
    try { await ff.deleteDir(mountedDir); } catch { /* noop */ }
  }

  return frames;
}

export interface VideoInfoLite {
  duration: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  format: string;
}

/**
 * Probe a video for the metadata the runtime cares about. Parses
 * `ffmpeg -i <file>` stderr — same path as runFfprobeShim but extracts
 * more fields.
 */
export async function getVideoInfoFromVfs(
  sourcePath: string,
  vfs: VfsBridge,
): Promise<VideoInfoLite> {
  // Probe via `ffmpeg -i` and let the standard ffmpeg path handle staging.
  const probe = await runFFmpegCommand('ffmpeg', ['-i', sourcePath], vfs);
  const text = probe.stderr || '';

  // Duration: HH:MM:SS.ms, ...
  const dm = text.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/);
  const duration = dm
    ? parseInt(dm[1], 10) * 3600 + parseInt(dm[2], 10) * 60 + parseFloat(dm[3])
    : 0;

  // Stream line: "Stream #0:0[0x1](und): Video: h264 (...), yuv420p(tv...),
  // 1920x1080 [SAR 1:1 DAR 16:9], 30 fps, 30 tbr, 90k tbn"
  const sm = text.match(
    /Video:\s*([^\s,]+)[^,]*,\s*(?:[^,]+,\s*)?(\d+)x(\d+)[^,]*,(?:[^,]+,\s*)*\s*([\d.]+)\s*fps/,
  );
  const codec = sm?.[1] ?? '';
  const width = sm ? parseInt(sm[2], 10) : 0;
  const height = sm ? parseInt(sm[3], 10) : 0;
  const fps = sm ? parseFloat(sm[4]) : 0;

  // Format/container: "Input #0, mov,mp4,m4a,3gp,3g2,mj2"
  const fm = text.match(/Input #0,\s*([^,\n]+)/);
  const format = fm?.[1].trim() ?? '';

  return { duration, width, height, fps, codec, format };
}

function guessContentType(path: string): string {
  const lower = path.toLowerCase();
  if (lower.endsWith('.mp4') || lower.endsWith('.m4v')) return 'video/mp4';
  if (lower.endsWith('.webm')) return 'video/webm';
  if (lower.endsWith('.mov')) return 'video/quicktime';
  if (lower.endsWith('.mkv')) return 'video/x-matroska';
  if (lower.endsWith('.mp3')) return 'audio/mpeg';
  if (lower.endsWith('.wav')) return 'audio/wav';
  if (lower.endsWith('.ogg')) return 'audio/ogg';
  if (lower.endsWith('.flac')) return 'audio/flac';
  if (lower.endsWith('.aac') || lower.endsWith('.m4a')) return 'audio/aac';
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
  if (lower.endsWith('.webp')) return 'image/webp';
  return 'application/octet-stream';
}
