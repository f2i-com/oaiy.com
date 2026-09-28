/**
 * A synchronous call from a Worker to the page over a SharedArrayBuffer.
 *
 * The Worker posts the request as an ordinary message, then blocks in
 * `Atomics.wait` until the page writes the reply into shared memory and
 * notifies it. Replies larger than the buffer arrive in chunks: the page sets
 * `more`, the Worker asks for the next one. The page never blocks.
 *
 *   ctrl[0] state: IDLE, WAITING (Worker blocked), READY (a chunk is there)
 *   ctrl[1] chunk length in bytes
 *   ctrl[2] 1 when more chunks follow
 */

export const IDLE = 0;
export const WAITING = 1;
export const READY = 2;

const CTRL_BYTES = 16;
export const CHUNK_BYTES = 1 << 20;

export function createChannelBuffer(): SharedArrayBuffer {
  return new SharedArrayBuffer(CTRL_BYTES + CHUNK_BYTES);
}

function views(sab: SharedArrayBuffer): { ctrl: Int32Array; data: Uint8Array } {
  return { ctrl: new Int32Array(sab, 0, 4), data: new Uint8Array(sab, CTRL_BYTES, CHUNK_BYTES) };
}

/** Page side: feeds one reply to the blocked Worker, chunk by chunk. */
export class ReplyWriter {
  private readonly ctrl: Int32Array;
  private readonly data: Uint8Array;
  private pending: Uint8Array | null = null;
  private offset = 0;

  constructor(sab: SharedArrayBuffer) {
    ({ ctrl: this.ctrl, data: this.data } = views(sab));
  }

  /** Start a reply; writes its first chunk. */
  begin(reply: string): void {
    this.pending = new TextEncoder().encode(reply);
    this.offset = 0;
    this.next();
  }

  /** The Worker asked for the next chunk. */
  next(): void {
    const bytes = this.pending ?? new Uint8Array();
    const end = Math.min(bytes.byteLength, this.offset + CHUNK_BYTES);
    this.data.set(bytes.subarray(this.offset, end), 0);
    Atomics.store(this.ctrl, 1, end - this.offset);
    Atomics.store(this.ctrl, 2, end < bytes.byteLength ? 1 : 0);
    this.offset = end;
    if (end >= bytes.byteLength) this.pending = null;
    Atomics.store(this.ctrl, 0, READY);
    Atomics.notify(this.ctrl, 0);
  }
}

/** Worker side: send a request and block until the whole reply is in. */
export function blockingCall(sab: SharedArrayBuffer, send: (message: unknown) => void, request: unknown): string {
  const { ctrl, data } = views(sab);
  const decoder = new TextDecoder();
  Atomics.store(ctrl, 0, WAITING);
  send(request);
  let out = '';
  for (;;) {
    Atomics.wait(ctrl, 0, WAITING);
    const length = Atomics.load(ctrl, 1);
    const more = Atomics.load(ctrl, 2) === 1;
    // Copy out of shared memory: TextDecoder refuses shared views.
    out += decoder.decode(data.slice(0, length), { stream: more });
    if (!more) break;
    Atomics.store(ctrl, 0, WAITING);
    send({ type: 'more' });
  }
  Atomics.store(ctrl, 0, IDLE);
  return out;
}
