/**
 * Video and audio editing in the page: what a file holds, frames out of a
 * video, cutting it up, and composing clips and sounds on a timeline into a
 * new video (or an audio file).
 *
 * Decoding and encoding are the browser's (WebCodecs: hardware H.264 and AAC
 * where it has them, Opus where AAC is missing); Mediabunny reads and writes
 * the MP4, WebM and WAV containers. The mix is an OfflineAudioContext, so
 * placement, trims, loops, volume, fades and ducking are Web Audio nodes.
 * Everything is re-encoded: a cut is exact to the frame, at the cost of one
 * encode.
 */
import {
  ALL_FORMATS,
  AudioBufferSink,
  AudioBufferSource,
  BlobSource,
  BufferTarget,
  CanvasSink,
  CanvasSource,
  Input,
  Mp4OutputFormat,
  Output,
  QUALITY_HIGH,
  WebMOutputFormat,
  getFirstEncodableAudioCodec,
  getFirstEncodableVideoCodec,
  type AudioCodec,
  type InputAudioTrack,
  type InputVideoTrack,
  type VideoCodec,
} from 'mediabunny';

export class EditError extends Error {}

const IMAGE_MIME: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif', bmp: 'image/bmp', avif: 'image/avif' };
const AUDIO_ONLY = /\.(mp3|wav|ogg|oga|opus|flac|m4a|aac|weba)$/i;
/** The longest timeline composed in one go. */
const MAX_SECONDS = 30 * 60;
/** Frames written out of a video in one call. */
const MAX_FRAMES_OUT = 300;
const SAMPLE_RATE = 48_000;

export const extOf = (name: string) => name.slice(name.lastIndexOf('.') + 1).toLowerCase();
const isImage = (name: string) => extOf(name) in IMAGE_MIME;
const round = (n: number, places = 3) => Math.round(n * 10 ** places) / 10 ** places;

function openInput(bytes: Uint8Array): Input {
  return new Input({ formats: ALL_FORMATS, source: new BlobSource(new Blob([bytes as BlobPart])) });
}

async function readable(input: Input, name: string): Promise<void> {
  if (!(await input.canRead().catch(() => false))) throw new EditError(`${name} is not a video or audio file this browser can read`);
}

// --- What a file holds -----------------------------------------------------

export interface MediaInfo {
  kind: 'video' | 'audio' | 'image';
  container?: string;
  duration?: number;
  width?: number;
  height?: number;
  video?: { codec: string | null; width: number; height: number; fps: number; frames: number };
  audio?: { codec: string | null; sampleRate: number; channels: number };
}

async function videoFacts(track: InputVideoTrack): Promise<{ width: number; height: number; fps: number; frames: number; codec: string | null }> {
  const stats = await track.computePacketStats();
  return {
    codec: await track.getCodec(),
    width: await track.getDisplayWidth(),
    height: await track.getDisplayHeight(),
    fps: round(stats.averagePacketRate, 3),
    frames: stats.packetCount,
  };
}

export async function mediaInfo(bytes: Uint8Array, name: string): Promise<MediaInfo> {
  if (isImage(name)) {
    const bitmap = await createImageBitmap(new Blob([bytes as BlobPart], { type: IMAGE_MIME[extOf(name)] }));
    const info: MediaInfo = { kind: 'image', width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return info;
  }
  const input = openInput(bytes);
  try {
    await readable(input, name);
    const format = await input.getFormat();
    const video = await input.getPrimaryVideoTrack();
    const audio = await input.getPrimaryAudioTrack();
    const info: MediaInfo = { kind: video ? 'video' : 'audio', container: format.name, duration: round(await input.computeDuration()) };
    if (video) {
      info.video = await videoFacts(video);
      info.width = info.video.width;
      info.height = info.video.height;
    }
    if (audio) info.audio = { codec: await audio.getCodec(), sampleRate: await audio.getSampleRate(), channels: await audio.getNumberOfChannels() };
    return info;
  } finally {
    input.dispose();
  }
}

// --- Frames out of a video -------------------------------------------------

export interface FrameRequest {
  /** Times in seconds. */
  times?: number[];
  /** Frame numbers, from 0. */
  frames?: number[];
  /** One frame every this many seconds. */
  every?: number;
  first?: boolean;
  last?: boolean;
  /** The longest side of the images (default: the video's size). */
  maxSize?: number;
}

export interface Frame {
  frame: number;
  time: number;
  png: Uint8Array;
  width: number;
  height: number;
}

async function canvasPng(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<Uint8Array> {
  const blob = 'convertToBlob' in canvas
    ? await canvas.convertToBlob({ type: 'image/png' })
    : await new Promise<Blob>((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new EditError('the frame could not be encoded'))), 'image/png'));
  return new Uint8Array(await blob.arrayBuffer());
}

export async function videoFrames(bytes: Uint8Array, name: string, req: FrameRequest): Promise<{ frames: Frame[]; fps: number; duration: number; count: number }> {
  const input = openInput(bytes);
  try {
    await readable(input, name);
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new EditError(`${name} has no video`);
    const facts = await videoFacts(track);
    const fps = facts.fps || 30;
    const start = await track.getFirstTimestamp();
    const duration = await track.computeDuration();
    const count = facts.frames;
    // Frame n is shown from start + n / fps: look a little into it, so rounding never lands on the one before.
    const wanted = new Map<number, number>();
    const addFrame = (n: number) => {
      const f = Math.max(0, Math.min(count - 1, Math.round(n)));
      wanted.set(f, start + (f + 0.25) / fps);
    };
    for (const f of req.frames ?? []) addFrame(f);
    for (const t of req.times ?? []) addFrame(Math.floor((Math.max(0, t) - 0) * fps + 1e-6));
    if (req.every && req.every > 0) for (let t = 0; t < duration && wanted.size < MAX_FRAMES_OUT + 1; t += req.every) addFrame(Math.floor(t * fps + 1e-6));
    if (req.first) addFrame(0);
    if (req.last) addFrame(count - 1);
    if (!wanted.size) throw new EditError('say which frames: times, frames, every, first or last');
    if (wanted.size > MAX_FRAMES_OUT) throw new EditError(`${wanted.size} frames asked for; at most ${MAX_FRAMES_OUT} at a time`);
    const scale = req.maxSize && Math.max(facts.width, facts.height) > req.maxSize ? req.maxSize / Math.max(facts.width, facts.height) : 1;
    const width = Math.max(2, Math.round(facts.width * scale));
    const height = Math.max(2, Math.round(facts.height * scale));
    const sink = new CanvasSink(track, { width, height, fit: 'fill' });
    const order = [...wanted.entries()].sort((a, b) => a[1] - b[1]);
    const frames: Frame[] = [];
    let i = 0;
    for await (const wrapped of sink.canvasesAtTimestamps(order.map(([, t]) => t))) {
      const [frame] = order[i++];
      if (!wrapped) continue;
      frames.push({ frame, time: round(frame / fps), png: await canvasPng(wrapped.canvas), width, height });
    }
    return { frames, fps, duration: round(duration), count };
  } finally {
    input.dispose();
  }
}

// --- The timeline ----------------------------------------------------------

export interface ClipSpec {
  bytes: Uint8Array;
  name: string;
  /** In the source, seconds (a video): where the clip starts and ends. */
  start?: number;
  end?: number;
  /** How long a still image shows (seconds, default 3). */
  duration?: number;
  /** Fades from and to black, seconds. */
  fadeIn?: number;
  fadeOut?: number;
  /** The clip's own sound: 1 as it is, 0 silent. */
  volume?: number;
}

export interface AudioSpec {
  bytes: Uint8Array;
  name: string;
  /** Where it starts on the timeline, seconds (default 0). */
  at?: number;
  /** The part of the file to use, seconds. */
  start?: number;
  end?: number;
  /** 1 as it is, 0.5 half, 2 double. */
  volume?: number;
  fadeIn?: number;
  fadeOut?: number;
  /** Repeat it to the end of the timeline (or to `until`). */
  loop?: boolean;
  until?: number;
  /** While this plays, every other sound drops to this level (0-1): a voice over music. */
  duck?: number;
}

export interface ComposeSpec {
  clips: ClipSpec[];
  audio: AudioSpec[];
  /** mp4 or webm for a video; wav, m4a or webm/ogg (Opus) for sound only. */
  output: 'mp4' | 'webm' | 'wav' | 'm4a' | 'ogg';
  width?: number;
  height?: number;
  fps?: number;
  /** contain (letterbox, default) or cover (crop) for clips of another shape. */
  fit?: 'contain' | 'cover';
  /** The clips' own sound (default true). */
  clipAudio?: boolean;
  /** The whole mix. */
  volume?: number;
  fadeIn?: number;
  fadeOut?: number;
  /** The length of a sound-only timeline (default: to the end of its last sound). */
  duration?: number;
}

export interface ComposeResult {
  bytes: Uint8Array;
  duration: number;
  width?: number;
  height?: number;
  fps?: number;
  videoCodec?: string;
  audioCodec?: string;
  /** The loudest sample before any limiting, 1 is full scale. */
  peak: number;
  /** The mix was turned down this much to keep it from clipping. */
  limitedBy?: number;
}

interface PlacedClip {
  spec: ClipSpec;
  offset: number;
  length: number;
  sourceStart: number;
  input?: Input;
  video?: InputVideoTrack;
  audio?: InputAudioTrack | null;
  image?: ImageBitmap;
  width: number;
  height: number;
  fps?: number;
}

async function loadClips(specs: ClipSpec[]): Promise<PlacedClip[]> {
  const clips: PlacedClip[] = [];
  let offset = 0;
  for (const spec of specs) {
    if (isImage(spec.name)) {
      const image = await createImageBitmap(new Blob([spec.bytes as BlobPart], { type: IMAGE_MIME[extOf(spec.name)] }));
      const length = spec.duration && spec.duration > 0 ? spec.duration : 3;
      clips.push({ spec, offset, length, sourceStart: 0, image, width: image.width, height: image.height });
      offset += length;
      continue;
    }
    const input = openInput(spec.bytes);
    await readable(input, spec.name);
    const video = await input.getPrimaryVideoTrack();
    if (!video) {
      input.dispose();
      throw new EditError(`${spec.name} has no video: put sound in audio, not clips`);
    }
    const first = await video.getFirstTimestamp();
    const full = (await video.computeDuration()) - first;
    const sourceStart = Math.max(0, Math.min(spec.start ?? 0, full));
    const sourceEnd = Math.max(sourceStart, Math.min(spec.end ?? full, full));
    const length = sourceEnd - sourceStart;
    if (length <= 0) throw new EditError(`${spec.name}: the part from ${sourceStart} s to ${sourceEnd} s is empty`);
    const facts = await videoFacts(video);
    clips.push({ spec, offset, length, sourceStart: first + sourceStart, input, video, audio: await input.getPrimaryAudioTrack(), width: facts.width, height: facts.height, fps: facts.fps });
    offset += length;
  }
  return clips;
}

/** Decode a whole sound file (or a video's sound) to an AudioBuffer. */
async function decodeSound(ctx: BaseAudioContext, spec: { bytes: Uint8Array; name: string }): Promise<AudioBuffer> {
  if (AUDIO_ONLY.test(spec.name)) {
    try {
      return await ctx.decodeAudioData(spec.bytes.slice().buffer);
    } catch {
      /* try the container reader */
    }
  }
  const input = openInput(spec.bytes);
  try {
    await readable(input, spec.name);
    const track = await input.getPrimaryAudioTrack();
    if (!track) throw new EditError(`${spec.name} has no sound`);
    return (await joinBuffers(ctx, track, 0, Infinity)).buffer;
  } finally {
    input.dispose();
  }
}

/**
 * A track's sound from `start` to `end` (source seconds) as one AudioBuffer,
 * and how far into it `start` falls (the first decoded block can begin a
 * little before it).
 */
async function joinBuffers(ctx: BaseAudioContext, track: InputAudioTrack, start: number, end: number): Promise<{ buffer: AudioBuffer; lead: number }> {
  const parts: AudioBuffer[] = [];
  let firstAt: number | null = null;
  for await (const wrapped of new AudioBufferSink(track).buffers(start, Number.isFinite(end) ? end : undefined)) {
    firstAt ??= wrapped.timestamp;
    parts.push(wrapped.buffer);
  }
  if (!parts.length) return { buffer: ctx.createBuffer(2, 1, ctx.sampleRate), lead: 0 };
  const rate = parts[0].sampleRate;
  const channels = Math.max(...parts.map((p) => p.numberOfChannels));
  const length = parts.reduce((n, p) => n + p.length, 0);
  const joined = new AudioBuffer({ length, sampleRate: rate, numberOfChannels: channels });
  let at = 0;
  for (const p of parts) {
    for (let c = 0; c < channels; c++) joined.copyToChannel(p.getChannelData(Math.min(c, p.numberOfChannels - 1)), c, at);
    at += p.length;
  }
  return { buffer: joined, lead: Math.max(0, start - (firstAt ?? start)) };
}

/** Merge overlapping [start, end) intervals. */
function merged(intervals: Array<[number, number]>): Array<[number, number]> {
  const sorted = intervals.slice().sort((a, b) => a[0] - b[0]);
  const out: Array<[number, number]> = [];
  for (const [s, e] of sorted) {
    const last = out[out.length - 1];
    if (last && s <= last[1]) last[1] = Math.max(last[1], e);
    else out.push([s, e]);
  }
  return out;
}

/** Fade a gain in at `from` and out before `to`. */
function fades(gain: AudioParam, level: number, from: number, to: number, fadeIn = 0, fadeOut = 0): void {
  if (fadeIn > 0) {
    gain.setValueAtTime(0, from);
    gain.linearRampToValueAtTime(level, from + Math.min(fadeIn, (to - from) / 2));
  } else gain.setValueAtTime(level, from);
  if (fadeOut > 0) {
    const s = Math.max(from, to - fadeOut);
    gain.setValueAtTime(level, s);
    gain.linearRampToValueAtTime(0, to);
  }
}

/** The whole mix: clips' sound in place, the audio tracks placed, trimmed, looped, faded, ducked. */
async function mixSound(clips: PlacedClip[], specs: AudioSpec[], total: number, spec: ComposeSpec): Promise<{ buffer: AudioBuffer | null; peak: number; limitedBy?: number }> {
  const withClipSound = spec.clipAudio !== false ? clips.filter((c) => c.audio && (c.spec.volume ?? 1) > 0) : [];
  if (!withClipSound.length && !specs.length) return { buffer: null, peak: 0 };
  const length = Math.max(1, Math.ceil(total * SAMPLE_RATE));
  const ctx = new OfflineAudioContext({ numberOfChannels: 2, length, sampleRate: SAMPLE_RATE });
  const master = ctx.createGain();
  fades(master.gain, spec.volume ?? 1, 0, total, spec.fadeIn, spec.fadeOut);
  master.connect(ctx.destination);
  // Every sound goes through its own level and a ducking stage.
  const voices: Array<{ duck: GainNode; from: number; to: number; ducks?: number }> = [];
  for (const clip of withClipSound) {
    const sound = await joinBuffers(ctx, clip.audio!, clip.sourceStart, clip.sourceStart + clip.length);
    const source = ctx.createBufferSource();
    source.buffer = sound.buffer;
    const level = ctx.createGain();
    const duck = ctx.createGain();
    fades(level.gain, clip.spec.volume ?? 1, clip.offset, clip.offset + clip.length, clip.spec.fadeIn, clip.spec.fadeOut);
    source.connect(level).connect(duck).connect(master);
    source.start(clip.offset, sound.lead, clip.length);
    voices.push({ duck, from: clip.offset, to: clip.offset + clip.length });
  }
  for (const track of specs) {
    const sound = await decodeSound(ctx, track);
    const start = Math.max(0, Math.min(track.start ?? 0, sound.duration));
    const end = Math.max(start, Math.min(track.end ?? sound.duration, sound.duration));
    if (end - start <= 0) throw new EditError(`${track.name}: the part from ${start} s to ${end} s is empty`);
    const at = Math.max(0, track.at ?? 0);
    const until = track.loop ? Math.min(total, track.until ?? total) : Math.min(total, at + (end - start));
    if (until <= at) continue;
    const source = ctx.createBufferSource();
    source.buffer = sound;
    if (track.loop) {
      source.loop = true;
      source.loopStart = start;
      source.loopEnd = end;
    }
    const level = ctx.createGain();
    const duck = ctx.createGain();
    fades(level.gain, track.volume ?? 1, at, until, track.fadeIn, track.fadeOut);
    source.connect(level).connect(duck).connect(master);
    source.start(at, start, track.loop ? undefined : end - start);
    if (track.loop) source.stop(until);
    voices.push({ duck, from: at, to: until, ducks: track.duck });
  }
  // Ducking: while a voice that ducks plays, every other sound drops to its level (ramps of a fifth of a second).
  const RAMP = 0.2;
  for (const voice of voices) {
    const others = voices.filter((o) => o !== voice && o.ducks !== undefined && o.ducks < 1 && o.to > voice.from && o.from < voice.to);
    const intervals = merged(others.map((o) => [o.from, o.to] as [number, number]));
    const level = Math.min(...others.map((o) => o.ducks!), 1);
    voice.duck.gain.setValueAtTime(1, 0);
    for (const [s, e] of intervals) {
      voice.duck.gain.setValueAtTime(1, Math.max(0, s - RAMP));
      voice.duck.gain.linearRampToValueAtTime(level, s);
      voice.duck.gain.setValueAtTime(level, e);
      voice.duck.gain.linearRampToValueAtTime(1, e + RAMP);
    }
  }
  const buffer = await ctx.startRendering();
  let peak = 0;
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < data.length; i++) {
      const v = Math.abs(data[i]);
      if (v > peak) peak = v;
    }
  }
  // Louder than full scale would clip: turn the whole mix down just enough.
  let limitedBy: number | undefined;
  if (peak > 0.99) {
    const k = 0.99 / peak;
    limitedBy = round(k, 3);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const data = buffer.getChannelData(c);
      for (let i = 0; i < data.length; i++) data[i] *= k;
    }
  }
  return { buffer, peak: round(peak, 3), limitedBy };
}

/** A slice of an AudioBuffer, for feeding the encoder a second at a time. */
function sliceBuffer(buffer: AudioBuffer, from: number, to: number): AudioBuffer {
  const length = Math.max(1, to - from);
  const out = new AudioBuffer({ length, sampleRate: buffer.sampleRate, numberOfChannels: buffer.numberOfChannels });
  for (let c = 0; c < buffer.numberOfChannels; c++) out.copyToChannel(buffer.getChannelData(c).subarray(from, to), c);
  return out;
}

/** 16-bit PCM WAV. */
function wavBytes(buffer: AudioBuffer): Uint8Array {
  const channels = buffer.numberOfChannels;
  const frames = buffer.length;
  const data = new DataView(new ArrayBuffer(44 + frames * channels * 2));
  const text = (at: number, s: string) => [...s].forEach((ch, i) => data.setUint8(at + i, ch.charCodeAt(0)));
  text(0, 'RIFF');
  data.setUint32(4, 36 + frames * channels * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  data.setUint32(16, 16, true);
  data.setUint16(20, 1, true);
  data.setUint16(22, channels, true);
  data.setUint32(24, buffer.sampleRate, true);
  data.setUint32(28, buffer.sampleRate * channels * 2, true);
  data.setUint16(32, channels * 2, true);
  data.setUint16(34, 16, true);
  text(36, 'data');
  data.setUint32(40, frames * channels * 2, true);
  const chans = Array.from({ length: channels }, (_, c) => buffer.getChannelData(c));
  let at = 44;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < channels; c++) {
      const v = Math.max(-1, Math.min(1, chans[c][i]));
      data.setInt16(at, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      at += 2;
    }
  }
  return new Uint8Array(data.buffer);
}

function even(n: number): number {
  return Math.max(2, Math.round(n / 2) * 2);
}

/** Draw a picture into the frame: letterboxed (contain) or cropped (cover), faded towards black. */
function drawFitted(g: OffscreenCanvasRenderingContext2D, source: CanvasImageSource, sw: number, sh: number, w: number, h: number, fit: 'contain' | 'cover', alpha: number): void {
  g.globalAlpha = 1;
  g.fillStyle = '#000';
  g.fillRect(0, 0, w, h);
  const scale = fit === 'cover' ? Math.max(w / sw, h / sh) : Math.min(w / sw, h / sh);
  const dw = sw * scale;
  const dh = sh * scale;
  g.globalAlpha = Math.max(0, Math.min(1, alpha));
  g.drawImage(source, (w - dw) / 2, (h - dh) / 2, dw, dh);
  g.globalAlpha = 1;
}

function fadeAlpha(t: number, length: number, fadeIn = 0, fadeOut = 0): number {
  let a = 1;
  if (fadeIn > 0 && t < fadeIn) a = Math.min(a, t / fadeIn);
  if (fadeOut > 0 && t > length - fadeOut) a = Math.min(a, (length - t) / fadeOut);
  return a;
}

/** Clips one after another, with the sound mixed over them, into a new file. */
export async function compose(spec: ComposeSpec, onProgress: (message: string) => void = () => {}): Promise<ComposeResult> {
  const soundOnly = spec.output === 'wav' || spec.output === 'm4a' || spec.output === 'ogg';
  if (soundOnly && spec.clips.length) throw new EditError(`a .${spec.output} output is sound only: give the videos as audio, or write .mp4`);
  if (!soundOnly && !spec.clips.length) throw new EditError('a video needs clips (videos or pictures); for sound only, write .wav or .m4a');
  const clips = await loadClips(spec.clips);
  try {
    const videoLength = clips.reduce((n, c) => n + c.length, 0);
    let total = videoLength;
    if (soundOnly) {
      if (spec.duration && spec.duration > 0) total = spec.duration;
      else {
        // To the end of the last sound that does not loop.
        const ctx = new OfflineAudioContext({ numberOfChannels: 1, length: 1, sampleRate: SAMPLE_RATE });
        for (const a of spec.audio) {
          if (a.loop && !a.until) continue;
          const sound = await decodeSound(ctx, a);
          const len = Math.min(a.end ?? sound.duration, sound.duration) - Math.max(0, a.start ?? 0);
          total = Math.max(total, a.loop ? a.until! : (a.at ?? 0) + len);
        }
      }
      if (!(total > 0)) throw new EditError('nothing to hear: give audio, and a duration for sounds that loop');
    }
    if (total > MAX_SECONDS) throw new EditError(`the timeline is ${Math.round(total)} s; at most ${MAX_SECONDS / 60} minutes at a time`);
    onProgress('mixing the sound…');
    const mix = await mixSound(clips, spec.audio, total, spec);

    if (soundOnly) {
      if (!mix.buffer) throw new EditError('nothing to hear');
      if (spec.output === 'wav') return { bytes: wavBytes(mix.buffer), duration: round(total), peak: mix.peak, limitedBy: mix.limitedBy, audioCodec: 'pcm-s16' };
      const codec = spec.output === 'm4a' ? await getFirstEncodableAudioCodec(['aac', 'opus'], { numberOfChannels: 2, sampleRate: SAMPLE_RATE }) : await getFirstEncodableAudioCodec(['opus'], { numberOfChannels: 2, sampleRate: SAMPLE_RATE });
      if (!codec) throw new EditError(`this browser cannot encode ${spec.output} audio; write .wav`);
      const target = new BufferTarget();
      const output = new Output({ format: spec.output === 'm4a' ? new Mp4OutputFormat({ fastStart: 'in-memory' }) : new WebMOutputFormat(), target });
      const source = new AudioBufferSource({ codec, bitrate: QUALITY_HIGH });
      output.addAudioTrack(source);
      await output.start();
      for (let at = 0; at < mix.buffer.length; at += SAMPLE_RATE * 5) await source.add(sliceBuffer(mix.buffer, at, Math.min(mix.buffer.length, at + SAMPLE_RATE * 5)));
      await output.finalize();
      return { bytes: new Uint8Array(target.buffer!), duration: round(total), peak: mix.peak, limitedBy: mix.limitedBy, audioCodec: codec };
    }

    // The frame: the given size, else the first clip's; even sides, as H.264 wants.
    const width = even(spec.width ?? clips[0].width);
    const height = even(spec.height ?? clips[0].height);
    const fps = Math.max(1, Math.min(60, spec.fps ?? Math.round(clips.find((c) => c.fps)?.fps ?? 30)));
    const webm = spec.output === 'webm';
    const videoCodec: VideoCodec | null = await getFirstEncodableVideoCodec(webm ? ['vp9', 'vp8', 'av1'] : ['avc', 'vp9', 'av1'], { width, height });
    if (!videoCodec) throw new EditError(`this browser cannot encode ${width}×${height} video`);
    const audioCodec: AudioCodec | null = mix.buffer ? await getFirstEncodableAudioCodec(webm ? ['opus', 'vorbis'] : ['aac', 'opus'], { numberOfChannels: 2, sampleRate: SAMPLE_RATE }) : null;
    if (mix.buffer && !audioCodec) throw new EditError('this browser cannot encode the sound');
    const canvas = new OffscreenCanvas(width, height);
    const g = canvas.getContext('2d')!;
    const target = new BufferTarget();
    const output = new Output({ format: webm ? new WebMOutputFormat() : new Mp4OutputFormat({ fastStart: 'in-memory' }), target });
    const videoSource = new CanvasSource(canvas, { codec: videoCodec, bitrate: QUALITY_HIGH, keyFrameInterval: 2 });
    output.addVideoTrack(videoSource, { frameRate: fps });
    const audioSource = mix.buffer && audioCodec ? new AudioBufferSource({ codec: audioCodec, bitrate: QUALITY_HIGH }) : null;
    if (audioSource) output.addAudioTrack(audioSource);
    await output.start();
    const fit = spec.fit ?? 'contain';
    const frameCount = Math.max(1, Math.round(total * fps));
    let audioAt = 0;
    let frame = 0;
    let lastReport = 0;
    for (const clip of clips) {
      const first = Math.round(clip.offset * fps);
      const last = Math.min(frameCount, Math.round((clip.offset + clip.length) * fps));
      const times = Array.from({ length: Math.max(0, last - first) }, (_, k) => (first + k) / fps);
      const draw = async (t: number, picture: CanvasImageSource | null, sw: number, sh: number) => {
        if (picture) drawFitted(g, picture, sw, sh, width, height, fit, fadeAlpha(t - clip.offset, clip.length, clip.spec.fadeIn, clip.spec.fadeOut));
        await videoSource.add(t, 1 / fps);
        frame++;
        // The sound goes in alongside, a second at a time, so the file interleaves.
        if (audioSource && mix.buffer) {
          const due = Math.min(mix.buffer.length, Math.ceil((t + 1) * SAMPLE_RATE));
          if (due - audioAt >= SAMPLE_RATE) {
            await audioSource.add(sliceBuffer(mix.buffer, audioAt, due));
            audioAt = due;
          }
        }
        if (frame - lastReport >= Math.max(fps, frameCount / 20)) {
          lastReport = frame;
          onProgress(`making the video: ${Math.floor((frame / frameCount) * 100)}%`);
        }
      };
      if (clip.image) {
        for (const t of times) await draw(t, clip.image, clip.width, clip.height);
        continue;
      }
      const sink = new CanvasSink(clip.video!, { poolSize: 2 });
      let previous: CanvasImageSource | null = null;
      let i = 0;
      // Source time for each output frame; a little into the frame, so rounding never picks the one before.
      for await (const wrapped of sink.canvasesAtTimestamps(times.map((t) => clip.sourceStart + (t - clip.offset) + 0.25 / (clip.fps || fps)))) {
        const t = times[i++];
        if (wrapped) previous = wrapped.canvas;
        await draw(t, previous, clip.width, clip.height);
      }
    }
    if (audioSource && mix.buffer && audioAt < mix.buffer.length) await audioSource.add(sliceBuffer(mix.buffer, audioAt, mix.buffer.length));
    onProgress('finishing the file…');
    await output.finalize();
    return { bytes: new Uint8Array(target.buffer!), duration: round(frameCount / fps), width, height, fps, videoCodec, audioCodec: audioCodec ?? undefined, peak: mix.peak, limitedBy: mix.limitedBy };
  } finally {
    for (const c of clips) {
      c.input?.dispose();
      c.image?.close();
    }
  }
}

/** Cut a video at points (seconds): each part re-encoded, exact to the frame. */
export async function splitVideo(bytes: Uint8Array, name: string, points: number[], onProgress: (message: string) => void = () => {}): Promise<Array<{ bytes: Uint8Array; start: number; end: number; frames: number }>> {
  const info = await mediaInfo(bytes, name);
  if (info.kind !== 'video' || !info.duration) throw new EditError(`${name} is not a video`);
  const cuts = [...new Set(points.filter((p) => p > 0 && p < info.duration!).map((p) => round(p, 4)))].sort((a, b) => a - b);
  if (!cuts.length) throw new EditError(`no cut points inside the video (it is ${info.duration} s long)`);
  const bounds = [0, ...cuts, info.duration];
  const parts: Array<{ bytes: Uint8Array; start: number; end: number; frames: number }> = [];
  for (let i = 0; i < bounds.length - 1; i++) {
    onProgress(`cutting part ${i + 1} of ${bounds.length - 1}…`);
    const result = await compose({ clips: [{ bytes, name, start: bounds[i], end: bounds[i + 1] }], audio: [], output: extOf(name) === 'webm' ? 'webm' : 'mp4', fps: info.video?.fps ? Math.round(info.video.fps) : undefined });
    parts.push({ bytes: result.bytes, start: bounds[i], end: bounds[i + 1], frames: Math.round((bounds[i + 1] - bounds[i]) * (result.fps ?? 30)) });
  }
  return parts;
}
