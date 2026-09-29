/**
 * Updating the installed app.
 *
 * A new build brings a new service worker. It installs beside the running one
 * and waits (public/sw.js does not take over by itself), so a tab that is open
 * keeps running the version it started with. The page says a new version is
 * ready; when the person chooses Reload, the page first asks whether it may be
 * left (see pwa/leaveGuard.ts), then the waiting worker is told to take over
 * and the page reloads once when it has. Nothing reloads the page by itself,
 * and nothing is sent to the worker until the person has agreed to leave.
 *
 * The page also asks the browser whether there is a new worker when it is shown
 * again after more than an hour (the browser checks on every navigation, but an
 * installed app can stay open for days).
 *
 * The controller takes its surroundings as arguments so that it can be tried
 * with stubs; `watchUpdates` gives it the real ones.
 */

/** The message a waiting worker takes to activate itself (public/sw.js). */
export const SKIP_WAITING = { type: 'skip-waiting' } as const;

export interface WorkerLike {
  state?: string;
  postMessage(message: unknown): void;
  addEventListener(type: 'statechange', listener: () => void): void;
}

export interface RegistrationLike {
  waiting: WorkerLike | null;
  installing: WorkerLike | null;
  update(): Promise<unknown>;
  addEventListener(type: 'updatefound', listener: () => void): void;
}

export interface ContainerLike {
  /** The worker that controls the page, or null before one does. */
  controller: unknown;
  addEventListener(type: 'controllerchange', listener: () => void): void;
}

export interface UpdateEnvironment {
  registration: RegistrationLike;
  container: ContainerLike;
  reload(): void;
  now(): number;
  /**
   * Asked before the worker is told to take over: may the page be left now? False
   * when the person chose to stay (the page's leave guard would have stopped them,
   * and they said no), in which case nothing is sent and nothing changes.
   */
  mayLeave(): boolean | Promise<boolean>;
}

/** `ready`: a new version is waiting to be used (or another tab already switched to it). */
export type UpdateState = 'none' | 'ready';

/** An hour, in milliseconds. */
export const CHECK_AFTER_MS = 60 * 60 * 1000;

export class UpdateController {
  private ready = false;
  /** A click told the waiting worker to take over, and the page reloads when it has. */
  private asked = false;
  /** The reload for that click was made: a second `controllerchange` for the same click does not make another. */
  private reloadedForChange = false;
  private controlled: boolean;
  private lastCheck: number;
  private readonly listeners = new Set<(state: UpdateState) => void>();

  constructor(
    private readonly env: UpdateEnvironment,
    private readonly checkAfterMs = CHECK_AFTER_MS,
  ) {
    this.controlled = !!env.container.controller;
    this.lastCheck = env.now();
    if (env.registration.waiting) this.ready = true;
    env.registration.addEventListener('updatefound', () => this.watch(env.registration.installing));
    env.container.addEventListener('controllerchange', () => this.controllerChanged());
    // A worker may already be installing when the page starts to look.
    this.watch(env.registration.installing);
  }

  get state(): UpdateState {
    return this.ready ? 'ready' : 'none';
  }

  /** Called with the new state whenever it changes; returns the function that stops it. */
  onChange(listener: (state: UpdateState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * The Reload button. The page is asked whether it may be left first; only when it
   * may, the waiting worker is told to take over, and the page reloads once it has.
   * Returns false when there is nothing to update to, or the person stayed.
   *
   * A click always does something: if the worker has already taken over (an earlier
   * click whose reload was declined by the browser's own leave prompt, or another
   * tab), the page is only reloaded.
   */
  async apply(): Promise<boolean> {
    if (!this.ready) return false;
    if (!(await this.env.mayLeave())) return false;
    const waiting = this.env.registration.waiting;
    if (waiting) {
      this.asked = true;
      this.reloadedForChange = false;
      waiting.postMessage(SKIP_WAITING);
      return true;
    }
    this.env.reload();
    return true;
  }

  /** The page is visible again: ask the browser for a new worker when it has been more than an hour. */
  async checkIfDue(): Promise<boolean> {
    const now = this.env.now();
    if (now - this.lastCheck <= this.checkAfterMs) return false;
    this.lastCheck = now;
    try {
      await this.env.registration.update();
    } catch {
      // offline, or the host is down: the next visible page tries again after an hour
    }
    return true;
  }

  private watch(worker: WorkerLike | null): void {
    if (!worker) return;
    worker.addEventListener('statechange', () => {
      // A first install has no worker in charge yet and simply starts; only a page that is
      // in one's charge has an older version running that this one would replace.
      if (worker.state === 'installed' && this.env.container.controller) this.setReady();
    });
  }

  private controllerChanged(): void {
    const wasControlled = this.controlled;
    this.controlled = true;
    if (this.asked) {
      if (!this.reloadedForChange) {
        this.reloadedForChange = true;
        this.env.reload();
      }
      return;
    }
    // A tab of its own took the new version: this page still runs the old one, and says so.
    if (wasControlled) this.setReady();
  }

  private setReady(): void {
    if (this.ready) return;
    this.ready = true;
    for (const listener of [...this.listeners]) listener('ready');
  }
}

/**
 * The controller for this page's registration: reloads with `location.reload()`,
 * asks `mayLeave` before it does anything, and looks for a new version whenever
 * the page is shown after more than an hour.
 */
export function watchUpdates(registration: ServiceWorkerRegistration, mayLeave: () => boolean | Promise<boolean>): UpdateController {
  const controller = new UpdateController({
    registration,
    container: navigator.serviceWorker,
    reload: () => location.reload(),
    now: () => Date.now(),
    mayLeave,
  });
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') void controller.checkIfDue();
  });
  return controller;
}
