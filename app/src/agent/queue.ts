/**
 * The queue sub-agents wait in: tasks start in order, at most `limit` at a
 * time. A local model server usually answers one request at a time, so it
 * gets one agent at a time; an API can take a few. A stopped run takes its
 * waiting tasks out of the queue, and its running ones see the abort.
 */
export class TaskQueue {
  private running = 0;
  private waiting: Array<{ start: () => void; signal?: AbortSignal; cancel: () => void }> = [];
  private limitSource: number | (() => number);

  constructor(limit: number | (() => number), private readonly onIdle?: () => void) {
    this.limitSource = limit;
  }

  /** Update the concurrency limit (e.g. without retaining a stale closure from an earlier caller). */
  setLimit(limit: number): void {
    this.limitSource = limit;
    this.next();
  }

  private getLimit(): number {
    const raw = typeof this.limitSource === 'function' ? this.limitSource() : this.limitSource;
    return Math.max(1, Number.isFinite(raw) ? Math.floor(raw) : 1);
  }

  /** How many are running and how many wait. */
  get size(): { running: number; waiting: number } {
    return { running: this.running, waiting: this.waiting.length };
  }

  /** Run `job` when there is room; `onStart` tells the caller it left the queue. */
  run<T>(job: () => Promise<T>, signal?: AbortSignal, onStart?: () => void, limit?: number): Promise<T> {
    if (limit !== undefined) this.limitSource = limit;
    return new Promise<T>((resolve, reject) => {
      if (signal?.aborted) {
        reject(new DOMException('Stopped.', 'AbortError'));
        return;
      }
      const onAbort = () => {
        const i = this.waiting.indexOf(entry);
        if (i >= 0) {
          this.waiting.splice(i, 1);
          entry.cancel();
          this.checkIdle();
        }
      };
      const start = () => {
        signal?.removeEventListener('abort', onAbort);
        this.running++;
        let result: Promise<T>;
        try {
          onStart?.();
          result = Promise.resolve(job());
        } catch (error) {
          result = Promise.reject(error);
        }
        const settle = (done: () => void) => {
          this.running--;
          this.checkIdle();
          done();
          if (this.waiting.length) queueMicrotask(() => this.next());
        };
        result.then(
          (val) => settle(() => resolve(val)),
          (err) => settle(() => reject(err)),
        );
      };
      const entry = {
        start,
        signal,
        cancel: () => reject(new DOMException('Stopped.', 'AbortError')),
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      this.waiting.push(entry);
      this.next();
    });
  }

  private next(): void {
    while (this.running < this.getLimit() && this.waiting.length) {
      this.waiting.shift()!.start();
    }
  }

  private checkIdle(): void {
    if (this.running === 0 && this.waiting.length === 0) {
      this.onIdle?.();
    }
  }
}

const queues = new Map<string, TaskQueue>();

/** The queue for a provider: every agent using it shares one. Empty queues are evicted on idle. */
export function queueFor(key: string, limit: number | (() => number)): TaskQueue {
  const n = typeof limit === 'function' ? limit() : limit;
  let queue = queues.get(key);
  if (!queue) {
    queue = new TaskQueue(n, () => {
      if (queues.get(key) === queue) queues.delete(key);
    });
    queues.set(key, queue);
  } else {
    queue.setLimit(n);
  }
  return queue;
}

/** Number of currently active provider queues (for diagnostics/tests). */
export function activeQueueCount(): number {
  return queues.size;
}
