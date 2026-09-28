/**
 * Speech to text for the agent: the words spoken in an audio or video file of
 * the project. The page decodes the file (any format the browser plays) to
 * 16 kHz mono and cuts it where it is quiet into pieces of at most half a
 * minute; OAIY Desktop's speech-to-text writes each piece out.
 */
import type { SessionTool } from '../agent/agent';
import type { Vfs } from '../vfs/vfs';
import type { Desktop } from './bridge';

export const RATE = 16_000;
/** The longest piece sent at once. */
const PIECE_SECONDS = 30;
/** How far back from a piece's end to look for a quiet place to cut. */
const CUT_WINDOW_SECONDS = 5;
/** A transcript longer than this is only saved, not handed back whole. */
const MAX_REPLY = 20_000;

/** 16-bit mono WAV of `samples` (-1..1) at `rate`. */
export function wav(samples: Float32Array, rate = RATE): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(44 + samples.length * 2);
  const view = new DataView(out.buffer);
  const text = (at: number, s: string) => [...s].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
  text(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), true);
  return out;
}

/**
 * Where to cut `samples` into pieces of at most `pieceSeconds`: each cut in the
 * quietest 20 ms of the piece's last `windowSeconds`, so words stay whole.
 * Returns [start, end) sample ranges covering it all.
 */
export function pieces(samples: Float32Array, rate = RATE, pieceSeconds = PIECE_SECONDS, windowSeconds = CUT_WINDOW_SECONDS): Array<[number, number]> {
  const most = Math.round(pieceSeconds * rate);
  const frame = Math.round(rate / 50);
  const out: Array<[number, number]> = [];
  let start = 0;
  while (samples.length - start > most) {
    let cut = start + most;
    let quietest = Infinity;
    for (let at = start + most - frame; at >= Math.max(start + frame, start + most - Math.round(windowSeconds * rate)); at -= frame) {
      let energy = 0;
      for (let i = at; i < at + frame; i++) energy += samples[i] * samples[i];
      if (energy < quietest) {
        quietest = energy;
        cut = at + (frame >> 1);
      }
    }
    out.push([start, cut]);
    start = cut;
  }
  if (samples.length > start) out.push([start, samples.length]);
  return out;
}

/** The file's sound as 16 kHz mono samples (the browser decodes it). */
export async function decode16k(bytes: Uint8Array): Promise<Float32Array> {
  const context = new OfflineAudioContext(1, 1, RATE);
  const buffer = await context.decodeAudioData(bytes.slice().buffer);
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0);
  const mono = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const channel = buffer.getChannelData(c);
    for (let i = 0; i < mono.length; i++) mono[i] += channel[i] / buffer.numberOfChannels;
  }
  return mono;
}

export const TRANSCRIBE_TOOL = 'transcribe_audio';

/** The agent's speech-to-text tool: `vfs` is the project's files; `decode` is the browser's decoder (tests give their own). */
export function transcribeTool(vfs: () => Vfs, desktop: () => Desktop | null, decode: (bytes: Uint8Array) => Promise<Float32Array> = decode16k): SessionTool {
  return {
    spec: {
      name: TRANSCRIBE_TOOL,
      description:
        'Writes out the words spoken in an audio or video file of the project (speech to text, run on this computer by OAIY Desktop, in English). ' +
        'Give `save_to` to write the transcript into a file as well; a transcript over 20,000 characters is only saved there. A minute of speech takes a few seconds.',
      parameters: {
        type: 'object',
        required: ['path'],
        properties: {
          path: { type: 'string', description: 'The audio or video file (wav, mp3, m4a, ogg, webm, mp4, …)' },
          save_to: { type: 'string', description: 'A text file to write the transcript into (optional)' },
        },
      },
    },
    run: async (input, signal) => {
      const d = desktop();
      if (!d) throw new Error('OAIY Desktop is not connected, so speech cannot be written out');
      const path = String(input.path ?? '');
      const files = vfs();
      let samples: Float32Array;
      try {
        samples = await decode(files.readBytes(path));
      } catch (e) {
        throw new Error(`${path} could not be played as sound: ${e instanceof Error ? e.message : String(e)}`);
      }
      const words: string[] = [];
      for (const [start, end] of pieces(samples)) {
        signal?.throwIfAborted();
        const text = (await d.transcribe(wav(samples.subarray(start, end)), signal)).trim();
        if (text) words.push(text);
      }
      const transcript = words.join(' ');
      const seconds = Math.round(samples.length / RATE);
      const saveTo = typeof input.save_to === 'string' && input.save_to.trim() ? input.save_to.trim() : '';
      if (saveTo) files.writeFile(saveTo, transcript, { parents: true });
      const heading = `${path}: ${seconds} s of sound${saveTo ? `, the transcript saved to ${saveTo}` : ''}.`;
      if (!transcript) return `${heading} No speech was heard.`;
      if (transcript.length > MAX_REPLY) {
        if (saveTo) return `${heading} It is ${transcript.length.toLocaleString('en')} characters long; it begins:\n${transcript.slice(0, 2000)}…`;
        return `${heading}\n${transcript.slice(0, MAX_REPLY)}\n[cut at 20,000 characters: give save_to to keep it all]`;
      }
      return `${heading}\n${transcript}`;
    },
  };
}
