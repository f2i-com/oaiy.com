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

/**
 * Match any URI scheme or comma-option protocol wrapper (`file:`, `http:`, `http,a=b://…`,
 * `tcp,timeout=5://…`, `subfile,,start,0,,:`, `concat:`, `crypto:`, `cache:`, `gopher:`, …).
 * Single-letter Windows drive prefixes (`C:\…`, `C:/…`, `C:foo`) are excluded via
 * `WINDOWS_DRIVE_RE` and immediately routed through `confineAbsolute`.
 */
const DISALLOWED_PROTOCOL_RE = /^[a-z][a-z0-9+.-]*(?:,[^:]*)*:/i;
const WINDOWS_DRIVE_RE = /^[A-Za-z]:/;

/** Strip any FFmpeg `:stream_spec` suffix and lowercase the base flag name (e.g. `-bitexact:v` -> `-bitexact`). */
function stripStreamSpec(flag: string): string {
  const colon = flag.indexOf(':');
  return (colon === -1 ? flag : flag.slice(0, colon)).toLowerCase();
}

/** Map virtual-root / absolute path args onto real files; leave flags alone. */
function resolveArg(a: string): string {
  const n = a.replace(/\\/g, '/');
  if (
    n === '/tmp' ||
    n.startsWith('/tmp/') ||
    n === '/oaiy' ||
    n.startsWith('/oaiy/') ||
    n === '/downloads' ||
    n.startsWith('/downloads/')
  ) {
    return resolvePath(a);
  }
  if (fsConfinementOn()) {
    // Server mode (`oaiy worker`): args are attacker-controlled. ALLOWLIST, not a denylist —
    // reject ANY arg shaped like a URI scheme or comma-option protocol wrapper (including
    // comma-options containing `=` like `http,a=b://…` or `tcp,timeout=5://…`) so ffmpeg
    // can't read or write a remote/exotic/local-file source. Single-letter Windows drive
    // letters (`C:\…`, `C:/…`, `C:foo`) are excluded here and confined immediately below.
    if (!WINDOWS_DRIVE_RE.test(n) && DISALLOWED_PROTOCOL_RE.test(n)) {
      throw new Error(`ffmpeg arg uses a disallowed protocol/URI scheme in server mode: ${a}`);
    }
    // Windows drive paths (including drive-relative `C:foo`) and POSIX absolute paths must be confined.
    if (WINDOWS_DRIVE_RE.test(a) || path.isAbsolute(a)) {
      return confineAbsolute(a);
    }
    // A bare relative path that traverses upward escapes the worker CWD/sandbox — reject it.
    // Value tokens (`copy`, `concat`, `0`, simple scalars) contain no `..`, so they pass through.
    if (/(^|\/)\.\.(\/|$)/.test(n)) {
      throw new Error(`ffmpeg arg uses an upward (..) path in server mode: ${a}`);
    }
  }
  return a;
}

/**
 * Flags that let ffmpeg/ffprobe read filtergraphs/presets/options from disk, widen protocols,
 * enable external media references, pass arbitrary encoder file params, or write auxiliary files —
 * all rejected under server-mode confinement (after stripping any `:stream_spec` suffix).
 */
const DISALLOWED_CONFINED_FLAGS = new Set([
  '-protocol_whitelist',
  '-filter_script',
  '-filter_complex_script',
  '-pre',
  '-fpre',
  '-vpre',
  '-apre',
  '-spre',
  '-sdp_file',
  '-vstats_file',
  '-vstats',
  '-novstats',
  '-passlogfile',
  '-enable_drefs',
  '-use_absolute_path',
  '-x264-params',
  '-x264opts',
  '-x265-params',
  '-x265opts',
  '-svtav1-params',
  '-rav1e-params',
  '-aom-params',
  '-vpx-params',
  '-hls_key_info_file',
  '-hls_enc_key_url',
  '-hls_segment_filename',
  '-hls_base_url',
  '-hls_fmp4_init_filename',
  '-var_stream_map',
  '-master_pl_name',
  '-segment_list',
  '-init_seg_name',
  '-media_seg_name',
  '-maximum_manifest_size',
  '-allowed_extensions',
  '-allowed_segment_extensions',
  '-progress',
  '-report',
  '-noreport',
  '-dump_attachment',
  '-attach',
  '-stats_enc_pre',
  '-stats_enc_post',
  '-stats_mux_pre',
  '-stats_mux_post',
  '-lavfi',
  '-print_graphs_file',
  '-o',
  '-output_file',
]);

/** True if `flag` (including `-/option` read-from-file or `:stream_spec` forms) is forbidden under confinement. */
function isDisallowedConfinedFlag(flag: string): boolean {
  if (flag.startsWith('-/')) return true;
  return DISALLOWED_CONFINED_FLAGS.has(stripStreamSpec(flag));
}

/** Flags that accept a filtergraph expression (validated via `validateConfinedFiltergraph`). */
const FILTERGRAPH_FLAG = /^-(?:[va]f|filter|filter_complex)(?::.*)?$/i;

/** Built-in x264/x265 speed presets permitted for `-preset` under server confinement. */
const SAFE_CODEC_PRESET =
  /^(?:ultrafast|superfast|veryfast|faster|fast|medium|slow|slower|veryslow|placebo)$/;

/**
 * Explicit allowlist of safe non-playlist, non-device `-f` container/demuxer/muxer formats
 * permitted under server-mode confinement. Excludes `lavfi`, `tee`, `dash`, `hls`, `sdp`,
 * `fifo`, `segment`, and all capture devices (`x11grab`, `gdigrab`, `dshow`, `avfoundation`, …).
 */
const ALLOWED_CONFINED_FORMATS = new Set([
  'mp4',
  'mov',
  'mkv',
  'matroska',
  'webm',
  'ogg',
  'opus',
  'mp3',
  'wav',
  'flac',
  'aac',
  'adts',
  'm4a',
  'aiff',
  'alac',
  'concat',
  'image2',
  'image2pipe',
  'rawvideo',
  'mpegts',
  'avi',
  'gif',
  'apng',
  'mjpeg',
  's16le',
  'f32le',
  'u8',
  'null',
]);

/**
 * Explicit allowlist of pure audio/video filters permitted in confined filtergraphs.
 * Filters that read or write external files (`movie`, `amovie`, `subtitles`, `ass`, `sendcmd`,
 * `asendcmd`, `lut3d`, `haldclut`, `ssim`, `psnr`, `libvmaf`, `metadata`, `ametadata`, …) are
 * excluded by default.
 */
const ALLOWED_CONFINED_FILTERS = new Set([
  'scale',
  'fps',
  'crop',
  'pad',
  'trim',
  'atrim',
  'setpts',
  'asetpts',
  'concat',
  'overlay',
  'format',
  'aformat',
  'volume',
  'fade',
  'afade',
  'aresample',
  'amix',
  'amerge',
  'loudnorm',
  'transpose',
  'hflip',
  'vflip',
  'rotate',
  'setsar',
  'setdar',
  'null',
  'anull',
  'copy',
  'acopy',
  'split',
  'asplit',
  'adelay',
  'atempo',
  'aloop',
  'loop',
  'tpad',
  'apad',
  'pan',
  'channelmap',
  'channelsplit',
  'join',
  'dynaudnorm',
  'compand',
  'highpass',
  'lowpass',
  'equalizer',
  'bass',
  'treble',
  'silenceremove',
  'silencedetect',
  'volumedetect',
  'ebur128',
  'drawtext',
  'drawbox',
  'eq',
  'hue',
  'boxblur',
  'gblur',
  'unsharp',
  'colorkey',
  'chromakey',
  'xfade',
  'acrossfade',
  'hstack',
  'vstack',
  'xstack',
  'tile',
  'thumbnail',
  'select',
  'aselect',
  'showinfo',
  'ashowinfo',
  'reverse',
  'areverse',
  'color',
  'anullsrc',
  'geq',
]);

/** Filter option keys that accept filesystem paths — always rejected even on allowlisted filters. */
const DISALLOWED_FILTER_OPTION_KEYS = new Set([
  'textfile',
  'fontfile',
  'stats_file',
  'log_path',
  'file',
  'filename',
  'model_path',
  'sofa',
  'psfile',
  'result',
  'input',
  'f',
]);

/**
 * FFmpeg boolean (`OPT_BOOL` / info) flags that take no argument token.
 * Matched via `isFfmpegBoolFlag`, which strips `:stream_spec` suffixes and automatically
 * recognizes `-no<bool>` negations (e.g. `-nohide_banner`, `-noy`, `-non`, `-nore`,
 * `-nobenchmark`, `-noauto_conversion_filters`, `-noprint_graphs`, `-bitexact:v`).
 */
const FFMPEG_BOOL_FLAGS = new Set([
  '--',
  '-y',
  '-n',
  '-vn',
  '-an',
  '-sn',
  '-dn',
  '-hide_banner',
  '-stdin',
  '-stats',
  '-shortest',
  '-re',
  '-benchmark',
  '-benchmark_all',
  '-auto_conversion_filters',
  '-print_graphs',
  '-ignore_unknown',
  '-copy_unknown',
  '-accurate_seek',
  '-seek_timestamp',
  '-copyts',
  '-start_at_zero',
  '-fix_sub_duration',
  '-fix_sub_duration_heartbeat',
  '-autorotate',
  '-autoscale',
  '-bitexact',
  '-dump',
  '-hex',
  '-xerror',
  '-psnr',
  '-qphist',
  '-find_stream_info',
  '-recast_media',
  '-copyinkf',
  '-copypriorss',
  '-drop_changed',
  '-debug_ts',
  '-deinterlace',
  '-force_fps',
  '-display_hflip',
  '-display_vflip',
  '-apad',
  '-intra',
  '-sameq',
  '-same_quant',
  '-override_ffserver',
  '-l',
  '-h',
  '-?',
  '-help',
  '--help',
  '-version',
  '-buildconf',
  '-formats',
  '-muxers',
  '-demuxers',
  '-devices',
  '-codecs',
  '-decoders',
  '-encoders',
  '-bsfs',
  '-protocols',
  '-filters',
  '-pix_fmts',
  '-layouts',
  '-sample_fmts',
  '-dispositions',
  '-colors',
  '-hwaccels',
]);

/** True when `flag` is a 0-arg FFmpeg boolean/info flag, including `:stream_spec` and `-no<bool>` forms. */
function isFfmpegBoolFlag(flag: string): boolean {
  const base = stripStreamSpec(flag);
  if (FFMPEG_BOOL_FLAGS.has(base)) return true;
  if (base.startsWith('-no') && base.length > 3 && FFMPEG_BOOL_FLAGS.has(`-${base.slice(3)}`)) {
    return true;
  }
  return false;
}

/**
 * Known 1-arg FFmpeg value flags (matched after stripping `:stream_spec`).
 * Flags outside both `FFMPEG_BOOL_FLAGS` and `FFMPEG_VALUE_FLAGS` are treated conservatively:
 * they cannot swallow a positional output, protocol URL, or path/symlink token.
 */
const FFMPEG_VALUE_FLAGS = new Set([
  '-c',
  '-codec',
  '-vcodec',
  '-acodec',
  '-scodec',
  '-dcodec',
  '-map',
  '-map_metadata',
  '-map_chapters',
  '-map_channel',
  '-streamid',
  '-program',
  '-stream_group',
  '-discard',
  '-disposition',
  '-bsf',
  '-tag',
  '-vtag',
  '-atag',
  '-stag',
  '-b',
  '-vb',
  '-ab',
  '-crf',
  '-qp',
  '-q',
  '-qscale',
  '-aq',
  '-profile',
  '-level',
  '-tune',
  '-g',
  '-keyint_min',
  '-bf',
  '-refs',
  '-sc_threshold',
  '-maxrate',
  '-minrate',
  '-bufsize',
  '-pass',
  '-rc_override',
  '-intra_matrix',
  '-inter_matrix',
  '-chroma_intra_matrix',
  '-bits_per_raw_sample',
  '-strict',
  '-compression_level',
  '-qmin',
  '-qmax',
  '-qcomp',
  '-qdiff',
  '-mbd',
  '-trellis',
  '-me_method',
  '-subq',
  '-coder',
  '-flags',
  '-fflags',
  '-movflags',
  '-avioflags',
  '-fdebug',
  '-r',
  '-s',
  '-aspect',
  '-pix_fmt',
  '-fps_mode',
  '-vsync',
  '-fpsmax',
  '-frames',
  '-vframes',
  '-aframes',
  '-dframes',
  '-display_rotation',
  '-timecode',
  '-force_key_frames',
  '-time_base',
  '-enc_time_base',
  '-apply_cropping',
  '-canvas_size',
  '-color_primaries',
  '-color_trc',
  '-colorspace',
  '-color_range',
  '-field_order',
  '-start_number',
  '-top',
  '-reinit_filter',
  '-ar',
  '-ac',
  '-sample_fmt',
  '-channel_layout',
  '-ch_layout',
  '-async',
  '-guess_layout_max',
  '-t',
  '-to',
  '-ss',
  '-sseof',
  '-itsoffset',
  '-itsscale',
  '-isync',
  '-stream_loop',
  '-copytb',
  '-muxdelay',
  '-muxpreload',
  '-max_muxing_queue_size',
  '-muxing_queue_data_threshold',
  '-thread_queue_size',
  '-shortest_buf_duration',
  '-readrate',
  '-readrate_initial_burst',
  '-readrate_catchup',
  '-dts_delta_threshold',
  '-dts_error_threshold',
  '-frame_drop_threshold',
  '-timestamp',
  '-fs',
  '-target',
  '-metadata',
  '-v',
  '-loglevel',
  '-threads',
  '-filter_threads',
  '-filter_complex_threads',
  '-filter_buffered_frames',
  '-stats_period',
  '-max_error_rate',
  '-abort_on',
  '-max_alloc',
  '-cpuflags',
  '-cpucount',
  '-timelimit',
  '-fpsprobesize',
  '-max_probe_packets',
  '-max_delay',
  '-indexmem',
  '-rtbufsize',
  '-seek2any',
  '-hwaccel',
  '-hwaccel_output_format',
  '-e',
]);

/**
 * Scan one token using FFmpeg's `av_get_token` semantics up to any unquoted/unescaped terminator.
 * Inside `'...'`, characters (including backslashes) are taken verbatim until the closing `'`.
 * Outside `'...'`, `\` escapes the immediately following character.
 */
function scanFfmpegToken(
  input: string,
  start: number,
  terminators: string,
): { unquoted: string; end: number } {
  let i = start;
  while (i < input.length && (input[i] === ' ' || input[i] === '\t')) i++;
  let unquoted = '';
  let lastNonSpace = 0;
  while (i < input.length && !terminators.includes(input[i])) {
    const ch = input[i];
    if (ch === '\\') {
      if (i + 1 >= input.length) {
        throw new Error('ffmpeg filtergraph has a trailing backslash escape and is not allowed in server mode');
      }
      unquoted += input[i + 1];
      lastNonSpace = unquoted.length;
      i += 2;
    } else if (ch === "'") {
      i++;
      while (i < input.length && input[i] !== "'") {
        unquoted += input[i];
        i++;
      }
      if (i >= input.length) {
        throw new Error('ffmpeg filtergraph has an unclosed single quote and is not allowed in server mode');
      }
      i++;
      lastNonSpace = unquoted.length;
    } else {
      unquoted += ch;
      if (ch !== ' ' && ch !== '\t') {
        lastNonSpace = unquoted.length;
      }
      i++;
    }
  }
  return { unquoted: unquoted.slice(0, lastNonSpace), end: i };
}

/** Find the first unquoted, unescaped `=` inside a filter option segment. */
function findUnquotedEquals(seg: string): number {
  let inQuote = false;
  for (let i = 0; i < seg.length; i++) {
    const ch = seg[i];
    if (inQuote) {
      if (ch === "'") inQuote = false;
      continue;
    }
    if (ch === '\\') {
      i++;
      continue;
    }
    if (ch === "'") {
      inQuote = true;
      continue;
    }
    if (ch === '=') return i;
  }
  return -1;
}

/** Validate a filter option segment's key (rejecting quotes, backslashes, and file-path keys). */
function validateFilterOptionSegment(rawSeg: string, filterBase: string, requireNamedForDrawtext: boolean): void {
  const eqIdx = findUnquotedEquals(rawSeg);
  if (eqIdx === -1) {
    if (requireNamedForDrawtext && filterBase === 'drawtext') {
      throw new Error(
        `ffmpeg filtergraph drawtext positional argument is not allowed in server mode: ${rawSeg}`,
      );
    }
    return;
  }
  const rawKey = rawSeg.slice(0, eqIdx).trim();
  if (!rawKey || rawKey.includes("'") || rawKey.includes('\\')) {
    throw new Error(
      `ffmpeg filtergraph option key '${rawKey}' is not allowed in server mode`,
    );
  }
  if (DISALLOWED_FILTER_OPTION_KEYS.has(rawKey.toLowerCase())) {
    throw new Error(
      `ffmpeg filtergraph option '${rawKey}' is not allowed in server mode`,
    );
  }
  if (filterBase !== 'pan' && !/^[a-z][a-z0-9_-]*$/i.test(rawKey)) {
    throw new Error(
      `ffmpeg filtergraph option key '${rawKey}' is not allowed in server mode`,
    );
  }
}

/**
 * Strictly parse and allowlist a confined FFmpeg filtergraph (`-vf`, `-af`, `-filter*`,
 * `-filter_complex*`). Mirrors FFmpeg's two-level `av_get_token` parsing so quoted or
 * backslash-escaped filter names and option keys (`'movie'=…`, `m\ovie=…`, `'textfile'=…`,
 * `drawtext='textfile=/etc/passwd'`) cannot bypass validation.
 */
function validateConfinedFiltergraph(spec: string): void {
  if (!spec || !spec.trim()) {
    throw new Error('ffmpeg filtergraph is empty and is not allowed in server mode');
  }
  if (/[\u0000-\u001f\u007f]/.test(spec)) {
    throw new Error('ffmpeg filtergraph contains control characters and is not allowed in server mode');
  }

  let pos = 0;
  let filterCount = 0;
  const skipWs = () => {
    while (pos < spec.length && (spec[pos] === ' ' || spec[pos] === '\t')) pos++;
  };
  const consumeLinkLabels = () => {
    while (pos < spec.length && spec[pos] === '[') {
      const close = spec.indexOf(']', pos + 1);
      if (close === -1) {
        throw new Error('ffmpeg filtergraph has an unclosed link label and is not allowed in server mode');
      }
      const label = spec.slice(pos + 1, close);
      if (!/^[A-Za-z0-9_.:-]+$/.test(label)) {
        throw new Error(`ffmpeg filtergraph link label '[${label}]' is not allowed in server mode`);
      }
      pos = close + 1;
      skipWs();
    }
  };

  while (pos < spec.length) {
    skipWs();
    if (pos >= spec.length) break;

    consumeLinkLabels();

    // Parse raw filter name up to `=,;[` without unescaping so quotes/backslashes are caught directly.
    const nameStart = pos;
    while (pos < spec.length && !'=;,['.includes(spec[pos])) {
      pos++;
    }
    const rawName = spec.slice(nameStart, pos).trim();
    if (!rawName) {
      throw new Error(`ffmpeg filtergraph is missing a filter name and is not allowed in server mode: ${spec}`);
    }
    if (rawName.includes("'") || rawName.includes('\\')) {
      throw new Error(`ffmpeg filtergraph filter '${rawName}' is not allowed in server mode`);
    }
    const nameMatch = /^([a-z][a-z0-9_]*)(?:@([a-z0-9_-]+))?$/i.exec(rawName);
    if (!nameMatch) {
      throw new Error(`ffmpeg filtergraph filter '${rawName}' is not allowed in server mode`);
    }
    const filterBase = nameMatch[1].toLowerCase();
    if (!ALLOWED_CONFINED_FILTERS.has(filterBase)) {
      throw new Error(`ffmpeg filtergraph filter '${filterBase}' is not allowed in server mode`);
    }
    filterCount++;

    if (pos < spec.length && spec[pos] === '=') {
      pos++;
      const rawArgsStart = pos;
      const level1 = scanFfmpegToken(spec, pos, ',;[');
      const rawArgs = spec.slice(rawArgsStart, level1.end);
      pos = level1.end;

      // Pass 1 (pre-Level-1): reject single quotes/backslashes in raw option keys (e.g. `scale='w'=640`).
      let rawOptPos = 0;
      while (rawOptPos < rawArgs.length) {
        const seg = scanFfmpegToken(rawArgs, rawOptPos, ':');
        const rawSeg = rawArgs.slice(rawOptPos, seg.end).trim();
        validateFilterOptionSegment(rawSeg, filterBase, false);
        if (seg.end < rawArgs.length && rawArgs[seg.end] === ':') {
          rawOptPos = seg.end + 1;
        } else {
          break;
        }
      }

      // Pass 2 (post-Level-1, matching FFmpeg's `av_opt_set_from_string`): validate unquoted options
      // so level-1 quoted keys like `drawtext='textfile=/etc/passwd'` or `drawtext=/etc/passwd` are caught.
      const unquotedArgs = level1.unquoted;
      let optPos = 0;
      if (unquotedArgs.length === 0 && filterBase === 'drawtext') {
        throw new Error('ffmpeg filtergraph drawtext requires named options in server mode');
      }
      while (optPos < unquotedArgs.length) {
        const seg = scanFfmpegToken(unquotedArgs, optPos, ':');
        const rawSeg = unquotedArgs.slice(optPos, seg.end).trim();
        validateFilterOptionSegment(rawSeg, filterBase, true);
        if (seg.end < unquotedArgs.length && unquotedArgs[seg.end] === ':') {
          optPos = seg.end + 1;
          if (optPos === unquotedArgs.length && filterBase === 'drawtext') {
            throw new Error('ffmpeg filtergraph drawtext trailing separator is not allowed in server mode');
          }
        } else {
          break;
        }
      }
    }

    skipWs();
    consumeLinkLabels();
    skipWs();

    if (pos < spec.length) {
      if (spec[pos] !== ',' && spec[pos] !== ';') {
        throw new Error(`ffmpeg filtergraph has unexpected syntax near '${spec.slice(pos, pos + 16)}' in server mode`);
      }
      pos++;
      skipWs();
      if (pos >= spec.length) {
        throw new Error('ffmpeg filtergraph has a trailing separator and is not allowed in server mode');
      }
    }
  }

  if (filterCount === 0) {
    throw new Error('ffmpeg filtergraph contains no filters and is not allowed in server mode');
  }
}

/**
 * `ffprobe` flags that legitimately consume their immediately following argv token as a
 * non-input option value. Any other non-flag token passed to `ffprobe` under confinement
 * is treated as a positional input path and inspected via `inspectConfinedInput`.
 */
const FFPROBE_VALUE_FLAGS = new Set([
  '-v',
  '-loglevel',
  '-print_format',
  '-output_format',
  '-of',
  '-o',
  '-output_file',
  '-show_entries',
  '-select_streams',
  '-f',
  '-safe',
  '-probesize',
  '-formatprobesize',
  '-analyzeduration',
  '-read_intervals',
  '-max_probe_packets',
  '-fpsprobesize',
  '-threads',
]);

/** Max leading bytes inspected in `inspectConfinedInput` for playlists / concat manifests (1 MiB, matching FFmpeg's default format probe window). */
const INSPECT_PROBE_BYTES = 1024 * 1024;
/** Max `-analyzeduration` permitted under confinement (60 seconds in microseconds). */
const MAX_ANALYZEDURATION_US = 60_000_000;

/**
 * Parse a confined FFmpeg byte-size value (`-probesize`, `-formatprobesize`).
 * Only accepts non-negative decimal integers with optional K/M/G / KiB/MiB/GiB suffixes so
 * FFmpeg expressions or hex literals (`0x...`, `2*1024*1024`) cannot bypass the cap.
 */
function parseConfinedByteSize(raw: string): number | null {
  const m = /^(\d+)(?:([kKmMgG])(i?[bB])?)?$/.exec(raw.trim());
  if (!m) return null;
  const n = Number(m[1]);
  if (!Number.isSafeInteger(n)) return null;
  const unit = m[2]?.toLowerCase();
  const mult = unit === 'k' ? 1024 : unit === 'm' ? 1024 * 1024 : unit === 'g' ? 1024 * 1024 * 1024 : 1;
  const bytes = n * mult;
  return Number.isFinite(bytes) ? bytes : null;
}

/**
 * Parse a confined FFmpeg microsecond duration (`-analyzeduration`).
 * Accepts non-negative decimal integers with optional K/M decimal suffixes.
 */
function parseConfinedDurationUs(raw: string): number | null {
  const m = /^(\d+)([kKmM])?$/.exec(raw.trim());
  if (!m) return null;
  const n = Number(m[1]);
  if (!Number.isSafeInteger(n)) return null;
  const unit = m[2]?.toLowerCase();
  const mult = unit === 'k' ? 1_000 : unit === 'm' ? 1_000_000 : 1;
  const us = n * mult;
  return Number.isFinite(us) ? us : null;
}

/**
 * Resolve and confine a positional `ffmpeg` output path against `TMP_ROOT` via `confineAbsolute`
 * (which also enforces `refuseSecretStore` and live/dangling `assertNoSymlinkEscape`).
 */
function confinePositionalOutput(outArg: string): string {
  const candidate = path.isAbsolute(outArg) ? outArg : path.resolve(TMP_ROOT, outArg);
  return confineAbsolute(candidate);
}

/**
 * Decode candidate header text slices from an input file buffer: UTF-8 plus UTF-16LE/UTF-16BE
 * (both BOM-prefixed and BOM-less ASCII XML/text) so UTF-16 DASH/HLS/concat manifests cannot
 * evade prefix checks via NUL bytes.
 */
function decodeHeaderCandidates(slice: Buffer): string[] {
  const candidates: string[] = [];
  const nul = slice.indexOf(0);
  const utf8Head = (nul === -1 ? slice : slice.subarray(0, nul))
    .toString('utf-8')
    .replace(/^\uFEFF/, '');
  if (utf8Head.length > 0) candidates.push(utf8Head);

  const trimUtf16 = (s: string): string => {
    const z = s.indexOf('\u0000');
    return (z === -1 ? s : s.slice(0, z)).replace(/^\uFEFF/, '');
  };

  if (slice.length >= 4 && slice[0] === 0xff && slice[1] === 0xfe) {
    const body = slice.subarray(2, 2 + ((slice.length - 2) & ~1));
    candidates.push(trimUtf16(body.toString('utf16le')));
  } else if (slice.length >= 4 && slice[0] === 0xfe && slice[1] === 0xff) {
    const body = Buffer.from(slice.subarray(2, 2 + ((slice.length - 2) & ~1)));
    body.swap16();
    candidates.push(trimUtf16(body.toString('utf16le')));
  } else if (slice.length >= 4) {
    if (slice[1] === 0 && slice[3] === 0 && slice[0] !== 0) {
      const body = slice.subarray(0, slice.length & ~1);
      candidates.push(trimUtf16(body.toString('utf16le')));
    } else if (slice[0] === 0 && slice[2] === 0 && slice[1] !== 0) {
      const body = Buffer.from(slice.subarray(0, slice.length & ~1));
      body.swap16();
      candidates.push(trimUtf16(body.toString('utf16le')));
    }
  }
  return candidates;
}

/**
 * Inspect an input file under confinement regardless of whether `-f concat` was specified:
 * fails closed when the file does not exist (except `-` for stdin), checks auto-probed `ffconcat`
 * manifests through `checkConcatList`, and rejects UTF-8/UTF-16 HLS/DASH playlist headers
 * (`#EXTM3U`, `#EXT-X-`, `<MPD`, `<ns:MPD`) so an attacker-supplied playlist cannot smuggle
 * arbitrary file/network references past CLI arg checks.
 */
function inspectConfinedInput(
  inputArg: string,
  explicitConcat: boolean,
  seen = new Set<string>(),
): void {
  if (inputArg === '-') {
    if (explicitConcat) {
      throw new Error('ffmpeg -f concat cannot read from stdin in server mode');
    }
    return;
  }
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
  for (const head of decodeHeaderCandidates(slice)) {
    if (/^\s*ffconcat\b/i.test(head)) {
      checkConcatList(diskPath, seen);
      return;
    }
    if (/(?:^|[\r\n])[ \t]*(?:#EXTM3U\b|#EXT-X-)/i.test(head)) {
      throw new Error(`ffmpeg playlist/HLS input is not allowed in server mode: ${inputArg}`);
    }
    if (/<(?:[a-z_][a-z0-9._-]*:)?MPD(?:\s|\/|>|$)/i.test(head)) {
      throw new Error(`ffmpeg DASH/MPD input is not allowed in server mode: ${inputArg}`);
    }
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
    text = fs.readFileSync(listPath, 'utf-8').replace(/^\uFEFF/, '');
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
    inspectConfinedInput(entryDiskPath, false, seen);
  }
}

const mimeFor = (fmt: string): string =>
  fmt === 'png' ? 'image/png' : fmt === 'jpg' || fmt === 'jpeg' ? 'image/jpeg' : `image/${fmt}`;

/** Resolve and validate `run_command` argv for `ffmpeg`/`ffprobe` (exported for unit tests). */
export function buildRunCommandArgs(command: 'ffmpeg' | 'ffprobe', rawArgs: unknown[]): string[] {
  const raw = rawArgs.map((x) => String(x));
  if (!fsConfinementOn()) {
    return raw.map((x) => resolveArg(x));
  }
  // Defense-in-depth under confinement: restrict ffmpeg/ffprobe to local file/pipe protocols
  // on BOTH inputs and outputs, reject caller `-protocol_whitelist` and file-writing flags
  // (preventing token-adjacency desync), allowlist `-f` formats and filtergraph filters, cap
  // `-probesize`/`-formatprobesize` to `INSPECT_PROBE_BYTES`, confine positional outputs, and
  // inspect every input file (failing closed when missing).
  const guarded: string[] = [];
  if (command === 'ffprobe') {
    guarded.push('-protocol_whitelist', 'file,pipe');
  }
  let ffprobeExplicitConcat = false;
  if (command === 'ffprobe') {
    for (let j = 0; j < raw.length; j++) {
      if (raw[j] === '--') break;
      if (stripStreamSpec(raw[j]) === '-f' && String(raw[j + 1] ?? '').toLowerCase() === 'concat') {
        ffprobeExplicitConcat = true;
        break;
      }
    }
  }
  let afterDashDash = false;
  let inputIsConcat = false;
  let seenInput = false;
  let outputWhitelistPending = false;
  for (let i = 0; i < raw.length; i++) {
    const rawTok = raw[i];
    const baseFlag = !afterDashDash && rawTok.startsWith('-') ? stripStreamSpec(rawTok) : '';
    if (!afterDashDash && rawTok.startsWith('-')) {
      if (isDisallowedConfinedFlag(rawTok)) {
        throw new Error(`ffmpeg option ${rawTok} is not allowed in server mode`);
      }
      if (baseFlag === '-safe') {
        const val = raw[i + 1];
        if (val !== '1') {
          throw new Error(`ffmpeg ${rawTok} ${val ?? ''} is not allowed in server mode`.trim());
        }
        guarded.push(rawTok, val);
        i++;
        continue;
      }
      if (baseFlag === '-preset') {
        const val = raw[i + 1];
        if (val === undefined || !SAFE_CODEC_PRESET.test(val)) {
          throw new Error(`ffmpeg option ${rawTok} ${val ?? ''} is not allowed in server mode`.trim());
        }
        guarded.push(rawTok, val);
        i++;
        continue;
      }
      if (baseFlag === '-f') {
        const val = raw[i + 1];
        if (val === undefined || !ALLOWED_CONFINED_FORMATS.has(val.toLowerCase())) {
          throw new Error(`ffmpeg format -f ${val ?? '(missing)'} is not allowed in server mode`);
        }
        if (command === 'ffmpeg' && val.toLowerCase() === 'concat') {
          inputIsConcat = true;
        }
        guarded.push(rawTok, val);
        i++;
        continue;
      }
      if (baseFlag === '-probesize' || baseFlag === '-formatprobesize') {
        const val = raw[i + 1];
        if (val === undefined) {
          throw new Error(`ffmpeg option ${rawTok} requires a value in server mode`);
        }
        const bytes = parseConfinedByteSize(val);
        if (bytes === null || bytes > INSPECT_PROBE_BYTES) {
          throw new Error(
            `ffmpeg option ${rawTok} ${val} exceeds maximum confined probe size (${INSPECT_PROBE_BYTES}) and is not allowed in server mode`,
          );
        }
        guarded.push(rawTok, val);
        i++;
        continue;
      }
      if (baseFlag === '-analyzeduration') {
        const val = raw[i + 1];
        if (val === undefined) {
          throw new Error(`ffmpeg option ${rawTok} requires a value in server mode`);
        }
        const us = parseConfinedDurationUs(val);
        if (us === null || us > MAX_ANALYZEDURATION_US) {
          throw new Error(`ffmpeg option ${rawTok} ${val} is not allowed in server mode`);
        }
        guarded.push(rawTok, val);
        i++;
        continue;
      }
      if (FILTERGRAPH_FLAG.test(rawTok)) {
        const val = raw[i + 1];
        if (val === undefined) {
          throw new Error(`ffmpeg filtergraph option ${rawTok} requires a value in server mode`);
        }
        validateConfinedFiltergraph(val);
        guarded.push(rawTok, val);
        i++;
        continue;
      }
      if (rawTok === '-i') {
        const rawIn = raw[i + 1];
        if (rawIn === undefined) {
          throw new Error('ffmpeg -i requires an input path in server mode');
        }
        const inputArg = resolveArg(rawIn);
        if (command === 'ffmpeg') {
          guarded.push('-protocol_whitelist', 'file,pipe');
        }
        inspectConfinedInput(inputArg, command === 'ffprobe' ? ffprobeExplicitConcat : inputIsConcat);
        inputIsConcat = false;
        seenInput = true;
        outputWhitelistPending = false;
        guarded.push(rawTok, inputArg);
        i++;
        continue;
      }
    }
    if (command === 'ffprobe') {
      if (!afterDashDash && rawTok === '--') {
        afterDashDash = true;
        guarded.push(rawTok);
        continue;
      }
      if (!afterDashDash && FFPROBE_VALUE_FLAGS.has(baseFlag)) {
        guarded.push(rawTok);
        if (i + 1 < raw.length) {
          guarded.push(resolveArg(raw[++i]));
        }
        continue;
      }
      if (afterDashDash || !rawTok.startsWith('-') || rawTok === '-') {
        const inputArg = resolveArg(rawTok);
        inspectConfinedInput(inputArg, ffprobeExplicitConcat);
        guarded.push(inputArg);
        continue;
      }
      guarded.push(rawTok);
      continue;
    }
    // command === 'ffmpeg'
    if (rawTok === '-' || !rawTok.startsWith('-')) {
      // Positional output path (including stdout `-` and platform null sink):
      // enforce output protocol whitelist and confine relative/symlinked outputs.
      const isNullSink =
        rawTok === '-' ||
        (process.platform !== 'win32' && rawTok === '/dev/null') ||
        (process.platform === 'win32' && /^nul$/i.test(rawTok));
      const outArg = isNullSink ? rawTok : confinePositionalOutput(resolveArg(rawTok));
      guarded.push('-protocol_whitelist', 'file,pipe', outArg);
      outputWhitelistPending = false;
      continue;
    }
    if (isFfmpegBoolFlag(rawTok)) {
      guarded.push(rawTok);
      continue;
    }
    if (FFMPEG_VALUE_FLAGS.has(baseFlag)) {
      guarded.push(rawTok);
      // Consume one value token for known value flags, unless the next token looks like another flag
      // (sparing negative numeric/stream values like `-5` or `-0:a` and flag bitmasks like `-movflags -faststart`).
      const nextRaw = raw[i + 1];
      if (
        nextRaw !== undefined &&
        (!nextRaw.startsWith('-') || /^-\d/.test(nextRaw) || /^-(?:movflags|flags|fflags)$/i.test(baseFlag))
      ) {
        guarded.push(resolveArg(nextRaw));
        i++;
      }
      continue;
    }
    // Defense-in-depth for unknown/unclassified flags:
    // Emit `-protocol_whitelist file,pipe` before an unknown flag after an input so even an
    // unknown 0-arg flag cannot leave a subsequent output unguarded, and never let an unknown
    // flag swallow a positional output, protocol URL, path separator, or symlink/secret target.
    if (seenInput && !outputWhitelistPending) {
      guarded.push('-protocol_whitelist', 'file,pipe');
      outputWhitelistPending = true;
    }
    guarded.push(rawTok);
    const nextRaw = raw[i + 1];
    const nextIsLastAfterInput = seenInput && i + 1 === raw.length - 1;
    if (
      nextRaw !== undefined &&
      !nextIsLastAfterInput &&
      nextRaw !== '-' &&
      (!nextRaw.startsWith('-') || /^-\d/.test(nextRaw)) &&
      !/[\\/:\u0000]/.test(nextRaw)
    ) {
      const nextVal = resolveArg(nextRaw);
      confinePositionalOutput(nextVal);
      guarded.push(nextVal);
      i++;
    }
  }
  return guarded;
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
  // Ensure TMP_ROOT exists before confinement checks and child spawn (CWD is pinned to TMP_ROOT).
  ensureDir(TMP_ROOT);
  const resolved = buildRunCommandArgs(command, (args.args as unknown[]) ?? []);
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
  if (confined) {
    args.push('-protocol_whitelist', 'file,pipe');
  }
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
