/**
 * A new build of the editor, and how the person hears about it.
 *
 * A new worker installs beside the one in charge and waits. Nothing reloads the page under
 * someone who is editing a flow: the editor says a new version is ready and the person
 * chooses when (`apply`). Then the waiting worker is told to take over, and the page reloads
 * once the browser says it has (`controllerchange`), so the page and its files are the same build.
 *
 * Two things are told apart from that:
 *   - The first worker to install takes over the open page too, and that is not an update:
 *     nothing is said and nothing reloads.
 *   - A worker that took over because the person applied the update in another tab of the
 *     editor: the page here is still the old build, its lazy files are gone from the caches, and
 *     it says so ('updated-elsewhere') rather than break later.
 */

/** The parts of a ServiceWorker the controller uses. */
export interface WorkerLike {
  state: string;
  postMessage(message: unknown): void;
  addEventListener(type: 'statechange', listener: () => void): void;
}

export interface RegistrationLike {
  installing: WorkerLike | null;
  waiting: WorkerLike | null;
  addEventListener(type: 'updatefound', listener: () => void): void;
}

export interface ContainerLike {
  controller: unknown;
  addEventListener(type: 'controllerchange', listener: () => void): void;
}

/** 'none': nothing to say; 'ready': a new version is waiting; 'updated-elsewhere': another tab applied one. */
export type UpdateState = 'none' | 'ready' | 'updated-elsewhere';

export interface UpdateController {
  getState(): UpdateState;
  subscribe(listener: () => void): () => void;
  /** Follow this registration. */
  attach(container: ContainerLike, registration: RegistrationLike): void;
  /** The person chose to reload: the waiting worker takes over and the page reloads. */
  apply(): void;
  /** The person chose to carry on for now: the message goes until the next update. */
  dismiss(): void;
}

export function createUpdateController(deps: { reload: () => void }): UpdateController {
  let state: UpdateState = 'none';
  let waiting: WorkerLike | null = null;
  let applying = false;
  let reloaded = false;
  const listeners = new Set<() => void>();

  const set = (next: UpdateState) => {
    if (state === next) return;
    state = next;
    for (const listener of [...listeners]) listener();
  };
  const reloadOnce = () => {
    if (reloaded) return;
    reloaded = true;
    deps.reload();
  };

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    attach(container, registration) {
      // Was this page under a worker's control when it started? A page that was not is being taken over for the first time.
      let controlled = Boolean(container.controller);
      const consider = (worker: WorkerLike | null) => {
        if (worker && controlled) {
          waiting = worker;
          set('ready');
        }
      };
      registration.addEventListener('updatefound', () => {
        const installing = registration.installing;
        if (!installing) return;
        installing.addEventListener('statechange', () => {
          if (installing.state === 'installed') consider(installing);
        });
      });
      consider(registration.waiting);
      container.addEventListener('controllerchange', () => {
        const was = controlled;
        controlled = Boolean(container.controller);
        if (applying) reloadOnce();
        else if (was) set('updated-elsewhere');
      });
    },
    apply() {
      if (state === 'none') return;
      applying = true;
      if (waiting && state === 'ready') waiting.postMessage({ type: 'skip-waiting' });
      else reloadOnce(); // another tab already updated: nothing to wait for
    },
    dismiss() {
      set('none');
    },
  };
}

/** The words the editor shows for each state. */
export function updateMessage(state: UpdateState): string | null {
  if (state === 'ready') return 'A new version is ready.';
  if (state === 'updated-elsewhere') return 'OAIY was updated in another tab.';
  return null;
}
