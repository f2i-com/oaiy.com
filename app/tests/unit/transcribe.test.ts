import { describe, expect, it } from 'vitest';
import { Vfs } from '../../src/vfs/vfs';
import { RATE, pieces, transcribeTool, wav } from '../../src/desktop/transcribe';
import type { Desktop } from '../../src/desktop/bridge';

/** `seconds` of a 200 Hz tone, with silence from `quietAt` for `quietFor` seconds. */
function speech(seconds: number, quietAt = -1, quietFor = 0): Float32Array {
  const out = new Float32Array(Math.round(seconds * RATE));
  for (let i = 0; i < out.length; i++) {
    const t = i / RATE;
    out[i] = t >= quietAt && t < quietAt + quietFor ? 0 : 0.5 * Math.sin(2 * Math.PI * 200 * t);
  }
  return out;
}

describe('speech to text for the agent', () => {
  it('writes 16-bit mono WAV', () => {
    const file = wav(new Float32Array([0, 1, -1, 2]), 16_000);
    const view = new DataView(file.buffer);
    expect(String.fromCharCode(...file.slice(0, 4))).toBe('RIFF');
    expect(String.fromCharCode(...file.slice(8, 12))).toBe('WAVE');
    expect([view.getUint16(22, true), view.getUint32(24, true), view.getUint16(34, true)]).toEqual([1, 16_000, 16]);
    expect(view.getUint32(40, true)).toBe(8);
    // Clipped at full scale.
    expect([view.getInt16(44, true), view.getInt16(46, true), view.getInt16(48, true), view.getInt16(50, true)]).toEqual([0, 32767, -32767, 32767]);
  });

  it('cuts long speech into pieces of at most 30 s, where it is quiet', () => {
    // 70 s with a pause at 27–27.5 s: the first cut falls in it, not at 30 s mid-word.
    const samples = speech(70, 27, 0.5);
    const cuts = pieces(samples);
    expect(cuts[0][0]).toBe(0);
    expect(cuts[0][1] / RATE).toBeGreaterThan(27);
    expect(cuts[0][1] / RATE).toBeLessThan(27.5);
    for (const [start, end] of cuts) expect(end - start).toBeLessThanOrEqual(30 * RATE);
    // They cover it all, in order.
    for (let i = 1; i < cuts.length; i++) expect(cuts[i][0]).toBe(cuts[i - 1][1]);
    expect(cuts.at(-1)![1]).toBe(samples.length);
    expect(pieces(speech(5))).toEqual([[0, 5 * RATE]]);
  });

  it('the tool: the file decoded, each piece written out by the desktop, the transcript saved when asked', async () => {
    const vfs = new Vfs();
    vfs.writeFile('/talk.mp3', new Uint8Array([1, 2, 3]));
    const sent: number[] = [];
    const desktop = {
      transcribe: async (file: Uint8Array) => {
        sent.push(file.length);
        return sent.length === 1 ? 'Hello there.' : 'General Kenobi.';
      },
    } as unknown as Desktop;
    const tool = transcribeTool(() => vfs, () => desktop, async () => speech(40, 25, 0.5));
    expect(tool.spec.name).toBe('transcribe_audio');
    const out = await tool.run({ path: '/talk.mp3', save_to: '/notes/talk.txt' }, new AbortController().signal);
    expect(sent).toHaveLength(2);
    expect(out).toContain('/talk.mp3: 40 s of sound, the transcript saved to /notes/talk.txt.');
    expect(out).toContain('Hello there. General Kenobi.');
    // Its folder is made for it.
    expect(vfs.readText('/notes/talk.txt')).toBe('Hello there. General Kenobi.');
    await expect(transcribeTool(() => vfs, () => null).run({ path: '/talk.mp3' }, new AbortController().signal)).rejects.toThrow('not connected');
    await expect(transcribeTool(() => vfs, () => desktop, async () => { throw new Error('bad data'); }).run({ path: '/talk.mp3' }, new AbortController().signal)).rejects.toThrow('could not be played as sound: bad data');
  });
});
