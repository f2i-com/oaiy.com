/**
 * FIFO queue for local-network permission prompts.
 *
 * Concurrent jobs (or parallel nodes in one job) can request local-network
 * permission at the same time. Holding a single resolver ref silently overwrote
 * earlier resolvers, hanging those jobs forever. This queue serializes prompts
 * in FIFO order and guarantees every enqueued promise eventually settles.
 */
import type {
  LocalNetworkPermissionRequest,
  LocalNetworkPermissionResponse,
} from 'oaiy-core';

interface QueueEntry {
  request: LocalNetworkPermissionRequest;
  resolve: (response: LocalNetworkPermissionResponse) => void;
}

export class PermissionRequestQueue {
  private queue: QueueEntry[] = [];

  /** Number of pending permission requests (including the active one). */
  get length(): number {
    return this.queue.length;
  }

  /** Currently active request (head of queue), or null when idle. */
  get current(): LocalNetworkPermissionRequest | null {
    return this.queue[0]?.request ?? null;
  }

  /**
   * Enqueue a permission request. Returns the promise for the caller and
   * whether this entry became the active head (meaning the dialog should show it).
   */
  enqueue(request: LocalNetworkPermissionRequest): {
    promise: Promise<LocalNetworkPermissionResponse>;
    becameActive: boolean;
  } {
    let resolveFn!: (response: LocalNetworkPermissionResponse) => void;
    const promise = new Promise<LocalNetworkPermissionResponse>((resolve) => {
      resolveFn = resolve;
    });
    this.queue.push({ request, resolve: resolveFn });
    return { promise, becameActive: this.queue.length === 1 };
  }

  /**
   * Resolve the active head request and advance to the next queued request.
   * When the user allows AND chooses "remember" for a hostPort, any queued
   * duplicates for the exact same hostPort are also resolved immediately so the
   * user isn't prompted multiple times for the same host.
   */
  respond(allowed: boolean, remember: boolean): LocalNetworkPermissionRequest | null {
    const head = this.queue.shift();
    if (head) {
      head.resolve({ allowed, remember });
      if (allowed && remember && head.request.hostPort) {
        const remaining: QueueEntry[] = [];
        for (const entry of this.queue) {
          if (entry.request.hostPort === head.request.hostPort) {
            entry.resolve({ allowed: true, remember: true });
          } else {
            remaining.push(entry);
          }
        }
        this.queue = remaining;
      }
    }
    return this.current;
  }

  /** Deny and clear all pending requests (e.g., on unmount). */
  drainDeny(): void {
    const pending = this.queue;
    this.queue = [];
    for (const entry of pending) {
      entry.resolve({ allowed: false, remember: false });
    }
  }
}
