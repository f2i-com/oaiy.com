/** Trusted parent media only. The opaque plugin frame receives transcripts,
 * never a microphone stream, audio bytes, object URL or AudioContext. */
export interface VoiceRecording { stop(): Promise<Uint8Array>; cancel(): void }
export interface VoiceMedia {
  enable(): Promise<void>;
  record(): Promise<VoiceRecording>;
  play(pcm: Uint8Array, signal: AbortSignal): Promise<void>;
  dispose(): void;
}
const RATE = 16_000;
const SECONDS = 15;

/** Canonical mono PCM16 WAV, shared wire format with voice/audio.rs. */
export function voiceWav(samples: Float32Array, sampleRate: number): Uint8Array {
  if (!Number.isFinite(sampleRate) || sampleRate < RATE || sampleRate > 192_000 || samples.length > sampleRate * SECONDS) {
    throw new Error('Recording exceeds its bounded sample limit.');
  }
  const count = Math.floor(samples.length * RATE / sampleRate);
  if (!count || count > RATE * SECONDS) throw new Error('The recording contains no bounded audio.');
  const bytes = new Uint8Array(44 + count * 2);
  const view = new DataView(bytes.buffer);
  const tag = (offset: number, text: string) => [...text].forEach((c, i) => { bytes[offset + i] = c.charCodeAt(0); });
  tag(0, 'RIFF'); view.setUint32(4, bytes.length - 8, true); tag(8, 'WAVE'); tag(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, RATE, true); view.setUint32(28, RATE * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); tag(36, 'data'); view.setUint32(40, count * 2, true);
  // Average the input interval before downsampling, rather than dropping
  // two of every three 48 kHz samples. No assumed device sample rate.
  for (let i = 0; i < count; i++) {
    const start = Math.floor(i * sampleRate / RATE);
    const end = Math.max(start + 1, Math.min(samples.length, Math.floor((i + 1) * sampleRate / RATE)));
    let sum = 0;
    for (let n = start; n < end; n++) sum += Number.isFinite(samples[n]) ? samples[n] : 0;
    const value = Math.max(-1, Math.min(1, sum / (end - start)));
    view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return bytes;
}
export function voiceBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}
export class BrowserVoiceMedia implements VoiceMedia {
  private context: AudioContext | undefined;
  private capture: VoiceRecording | undefined;
  private playback: AudioBufferSourceNode | undefined;
  private closed = false;
  async enable(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) throw new Error('Microphone capture is unavailable in this desktop.');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 }, video: false });
    stream.getTracks().forEach((track) => track.stop());
    if (this.closed) throw new Error('Voice session closed.');
    this.context ??= new AudioContext();
    await this.context.resume();
    if (this.closed) { void this.context.close(); throw new Error('Voice session closed.'); }
  }
  async record(): Promise<VoiceRecording> {
    if (this.closed || !this.context) throw new Error('Enable this microphone session first.');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 }, video: false });
    if (this.closed) { stream.getTracks().forEach((track) => track.stop()); throw new Error('Voice session closed.'); }
    const context = this.context;
    let source: MediaStreamAudioSourceNode | undefined;
    let processor: ScriptProcessorNode | undefined;
    let silent: GainNode | undefined;
    let released = false;
    const release = () => {
      if (released) return; released = true;
      if (processor) processor.onaudioprocess = null;
      try {source?.disconnect();} catch {} try {processor?.disconnect();} catch {} try {silent?.disconnect();} catch {}
      stream.getTracks().forEach((track) => track.stop());
    };
    try {
      if (context.sampleRate < RATE || context.sampleRate > 192_000) throw new Error('Unsupported microphone sample rate.');
      await context.resume();
      if (this.closed) throw new Error('Voice session closed.');
      source = context.createMediaStreamSource(stream);
      // The desktop WebView supports ScriptProcessor. A silent sink drives
      // capture without feedback. Resampling is incremental: even a192kHz
      // device retains at most960kB of16kHz float samples, not11.5MiB raw.
      processor = context.createScriptProcessor(4096, 1, 1);
      silent = context.createGain(); silent.gain.value = 0;
      const chunks: Float32Array[] = []; let count = 0; let consumed = false;
      let inputCount = 0; let sum = 0; let interval = 0; let nextBoundary = Math.floor(context.sampleRate / RATE);
      processor.onaudioprocess = (event) => {
        if (released || consumed) return;
        const input = event.inputBuffer.getChannelData(0);
        const output: number[] = [];
        for (let i = 0; i < input.length && inputCount < context.sampleRate * SECONDS; i++) {
          sum += Number.isFinite(input[i]) ? input[i] : 0; interval++; inputCount++;
          if (inputCount >= nextBoundary && count < RATE * SECONDS) {
            output.push(sum / interval); count++; sum = 0; interval = 0;
            nextBoundary = Math.floor((count + 1) * context.sampleRate / RATE);
          }
        }
        if (output.length) chunks.push(Float32Array.from(output));
        // The sample cap independently stops native tracks if UI timers are
        // throttled. Keep only the bounded samples for the owner's Stop.
        if (inputCount >= context.sampleRate * SECONDS || count >= RATE * SECONDS) release();
      };
      source.connect(processor); processor.connect(silent); silent.connect(context.destination);
      const recording: VoiceRecording = {
        stop: async () => {
          if (consumed) throw new Error('The recording has stopped.');
          consumed = true; release(); if (this.capture === recording) this.capture = undefined;
          const samples = new Float32Array(count); let offset = 0;
          for (const chunk of chunks) {samples.set(chunk, offset); offset += chunk.length;}
          chunks.length = 0; return voiceWav(samples, RATE);
        },
        cancel: () => {consumed = true; release(); chunks.length = 0; if (this.capture === recording) this.capture = undefined;},
      };
      this.capture = recording; return recording;
    } catch (error) {release(); throw error;}
  }
  async play(pcm: Uint8Array, signal: AbortSignal): Promise<void> {
    if (this.closed || !this.context || !pcm.length || pcm.length % 2 || pcm.length > 960_000) throw new Error('Invalid bounded speech audio.');
    const context = this.context; await context.resume();
    if (this.closed || signal.aborted) throw new Error('Speech playback cancelled.');
    const buffer = context.createBuffer(1, pcm.length / 2, 24_000);
    const output = buffer.getChannelData(0); const input = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    for (let i = 0; i < output.length; i++) output[i] = input.getInt16(i * 2, true) / 32768;
    const source = context.createBufferSource(); source.buffer = buffer; source.connect(context.destination); this.playback = source;
    await new Promise<void>((resolve, reject) => {
      const finish = () => { signal.removeEventListener('abort', abort); source.disconnect(); if (this.playback === source) this.playback = undefined; };
      const abort = () => {source.onended = null; try {source.stop();} catch {} finish(); reject(new Error('Speech playback cancelled.'));};
      source.onended = () => {finish(); resolve();}; signal.addEventListener('abort', abort, {once:true});
      if (signal.aborted) abort(); else {try {source.start();} catch (error) {source.onended=null;finish();reject(error);}}
    });
  }
  dispose(): void {
    this.closed = true; this.capture?.cancel();
    try {this.playback?.stop();} catch {}
    if (this.context) void this.context.close().catch(() => {});
  }
}
