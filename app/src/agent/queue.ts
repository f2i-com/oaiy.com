/**
 * The queue sub-agents wait in: tasks start in order, at most `limit` at a
 * time. A local model server usually answers one request at a time, so it
 * gets one agent at a time; an API can take a few. A stopped run takes its
 * waiting tasks out of the queue, and its running ones see the abort.
 */
export class TaskQueue {
  private running = 0;
  private waiting: Array<{ start: () => void; signal?: AbortSignal; cancel: () => void }> = [];

  constructor(private readonly limit: () => number) {}

  /** How many are running and how many wait. */
  get size(): { running: number; waiting: number } {
    return { running: this.running, waiting: this.waiting.length };
  }

  /** Run `job` when there is room; `onStart` tells the caller it left the queue. */
  run<T>(job: () => Promise<T>, signal?: AbortSignal, onStart?: () => void): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const start = () => {
        this.running++;
        onStart?.();
        job()
          .then(resolve, reject)
          .finally(() => {
            this.running--;
            this.next();
          });
      };
      if (signal?.aborted) {
        reject(new DOMException('Stopped.', 'AbortError'));
        return;
      }
      const entry = {
        start,
        signal,
        cancel: () => reject(new DOMException('Stopped.', 'AbortError')),
      };
      signal?.addEventListener('abort', () => {
        const i = this.waiting.indexOf(entry);
        if (i >= 0) {
          this.waiting.splice(i, 1);
          entry.cancel();
        }
      }, { once: true });
      this.waiting.push(entry);
      this.next();
    });
  }

  private next(): void {
    while (this.running < Math.max(1, this.limit()) && this.waiting.length) {
      this.waiting.shift()!.start();
    }
  }
}

const queues = new Map<string, TaskQueue>();

/** The queue for a provider: every agent using it shares one. */
export function queueFor(key: string, limit: () => number): TaskQueue {
  let queue = queues.get(key);
  if (!queue) {
    queue = new TaskQueue(limit);
    queues.set(key, queue);
  }
  return queue;
}
