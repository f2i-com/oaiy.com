import { describe, expect, it, vi } from 'vitest';
import { CHECK_AFTER_MS, SKIP_WAITING, UpdateController, type ContainerLike, type RegistrationLike, type UpdateState, type WorkerLike } from '../../src/pwa/update';

class FakeWorker implements WorkerLike {
  state = 'installing';
  postMessage = vi.fn();
  private listeners: Array<() => void> = [];
  addEventListener(_type: 'statechange', listener: () => void): void {
    this.listeners.push(listener);
  }
  moveTo(state: string): void {
    this.state = state;
    for (const l of this.listeners) l();
  }
}

class FakeRegistration implements RegistrationLike {
  waiting: FakeWorker | null = null;
  installing: FakeWorker | null = null;
  update = vi.fn(async () => {});
  private found: Array<() => void> = [];
  addEventListener(_type: 'updatefound', listener: () => void): void {
    this.found.push(listener);
  }
  /** The browser found a new worker file and starts installing it. */
  found_(worker = new FakeWorker()): FakeWorker {
    this.installing = worker;
    for (const l of this.found) l();
    return worker;
  }
  /** It installed and waits behind the running one. */
  installed(worker: FakeWorker): void {
    this.installing = null;
    this.waiting = worker;
    worker.moveTo('installed');
  }
}

class FakeContainer implements ContainerLike {
  controller: unknown;
  private listeners: Array<() => void> = [];
  constructor(controlled: boolean) {
    this.controller = controlled ? {} : null;
  }
  addEventListener(_type: 'controllerchange', listener: () => void): void {
    this.listeners.push(listener);
  }
  /** A worker took over the page. */
  takeOver(): void {
    this.controller = {};
    for (const l of this.listeners) l();
  }
}

function setup({ controlled = true, waiting = false, mayLeave = async () => true }: { controlled?: boolean; waiting?: boolean; mayLeave?: () => boolean | Promise<boolean> } = {}) {
  const registration = new FakeRegistration();
  if (waiting) registration.waiting = new FakeWorker();
  const container = new FakeContainer(controlled);
  const reload = vi.fn();
  const asked = vi.fn(mayLeave);
  let now = 1_800_000_000_000;
  const controller = new UpdateController({ registration, container, reload, now: () => now, mayLeave: asked });
  const states: UpdateState[] = [];
  controller.onChange((s) => states.push(s));
  return {
    registration,
    container,
    reload,
    asked,
    controller,
    states,
    advance: (ms: number) => {
      now += ms;
    },
  };
}

describe('a new version is ready', () => {
  it('says nothing while there is no new version, and has nothing to apply', async () => {
    const { controller, registration, reload, asked } = setup();
    expect(controller.state).toBe('none');
    expect(await controller.apply()).toBe(false);
    expect(asked).not.toHaveBeenCalled();
    expect(registration.update).not.toHaveBeenCalled();
    expect(reload).not.toHaveBeenCalled();
  });

  it('is ready when a new worker was already waiting as the page started', () => {
    expect(setup({ waiting: true }).controller.state).toBe('ready');
  });

  it('becomes ready when a new worker has installed and waits behind the one in charge', () => {
    const { registration, controller, states, reload } = setup();
    const worker = registration.found_();
    expect(controller.state).toBe('none');
    registration.installed(worker);
    expect(controller.state).toBe('ready');
    expect(states).toEqual(['ready']);
    expect(reload).not.toHaveBeenCalled();
  });

  it('is not ready for a worker that is still installing, or that gave up', () => {
    const { registration, controller } = setup();
    const worker = registration.found_();
    worker.moveTo('installing');
    worker.moveTo('redundant');
    expect(controller.state).toBe('none');
  });

  it('is not ready for the very first worker (nothing was running before it)', () => {
    const { registration, container, controller, reload } = setup({ controlled: false });
    const worker = registration.found_();
    registration.installed(worker);
    expect(controller.state).toBe('none');
    // It then takes over the page as it starts: that is no update, and no reload of the page from here.
    container.takeOver();
    expect(controller.state).toBe('none');
    expect(reload).not.toHaveBeenCalled();
  });

  it('watches a worker that was already installing when the controller started', () => {
    const registration = new FakeRegistration();
    const worker = new FakeWorker();
    registration.installing = worker;
    const container = new FakeContainer(true);
    const controller = new UpdateController({ registration, container, reload: vi.fn(), now: () => 0, mayLeave: () => true });
    registration.installed(worker);
    expect(controller.state).toBe('ready');
  });
});

describe('the Reload button', () => {
  it('tells the waiting worker to take over, and reloads only when it has, once', async () => {
    const { registration, container, controller, reload } = setup({ waiting: true });
    const waiting = registration.waiting!;
    expect(await controller.apply()).toBe(true);
    expect(waiting.postMessage).toHaveBeenCalledWith(SKIP_WAITING);
    expect(reload).not.toHaveBeenCalled();
    container.takeOver();
    expect(reload).toHaveBeenCalledTimes(1);
    // A second controllerchange for the same click makes no second reload.
    container.takeOver();
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('does not reload by itself: the worker taking over without a click only says so', () => {
    const { registration, container, controller, reload } = setup({ waiting: true });
    container.takeOver();
    expect(reload).not.toHaveBeenCalled();
    expect(controller.state).toBe('ready');
    expect(registration.waiting!.postMessage).not.toHaveBeenCalled();
  });

  it('says a new version is ready when another tab switched to it, and then only loads it', async () => {
    const { container, registration, controller, states, reload } = setup();
    container.takeOver();
    expect(controller.state).toBe('ready');
    expect(states).toEqual(['ready']);
    expect(reload).not.toHaveBeenCalled();
    // Nothing waits any more: Reload just loads the page again.
    registration.waiting = null;
    expect(await controller.apply()).toBe(true);
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('tells the worker again when pressed again before it took over, and still reloads once', async () => {
    const { registration, container, controller, reload } = setup({ waiting: true });
    await controller.apply();
    await controller.apply();
    expect(registration.waiting!.postMessage).toHaveBeenCalledTimes(2);
    container.takeOver();
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('tells each listener once when it becomes ready', () => {
    const { registration, states } = setup();
    registration.installed(registration.found_());
    registration.installed(registration.found_());
    expect(states).toEqual(['ready']);
  });
});

describe('leaving the page for the update', () => {
  it('asks whether the page may be left before it tells the worker anything', async () => {
    let answer: (yes: boolean) => void = () => {};
    const question = new Promise<boolean>((resolve) => (answer = resolve));
    const { registration, controller, asked, reload } = setup({ waiting: true, mayLeave: () => question });
    const waiting = registration.waiting!;
    const pending = controller.apply();
    // The person has not answered: nothing is sent, nothing reloads.
    await Promise.resolve();
    expect(asked).toHaveBeenCalledTimes(1);
    expect(waiting.postMessage).not.toHaveBeenCalled();
    expect(reload).not.toHaveBeenCalled();
    answer(true);
    expect(await pending).toBe(true);
    expect(waiting.postMessage).toHaveBeenCalledWith(SKIP_WAITING);
  });

  it('sends nothing, changes nothing and reloads nothing when the person stays', async () => {
    const { registration, container, controller, states, reload } = setup({ waiting: true, mayLeave: () => false });
    const waiting = registration.waiting!;
    expect(await controller.apply()).toBe(false);
    expect(waiting.postMessage).not.toHaveBeenCalled();
    expect(reload).not.toHaveBeenCalled();
    expect(controller.state).toBe('ready');
    // The worker does not take over on its own either, so the page does not reload when some other tab makes it.
    container.takeOver();
    expect(reload).not.toHaveBeenCalled();
    expect(states).toEqual([]);
  });

  it('does not ask when there is nothing to update to', async () => {
    const { controller, asked } = setup();
    await controller.apply();
    expect(asked).not.toHaveBeenCalled();
  });

  it('works on a second click after the person stayed', async () => {
    let yes = false;
    const { registration, container, controller, reload } = setup({ waiting: true, mayLeave: () => yes });
    const waiting = registration.waiting!;
    expect(await controller.apply()).toBe(false);
    yes = true;
    expect(await controller.apply()).toBe(true);
    expect(waiting.postMessage).toHaveBeenCalledTimes(1);
    container.takeOver();
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('works on a second click after the browser prompt stopped the reload (the worker has taken over, the page stayed)', async () => {
    const { registration, container, controller, reload } = setup({ waiting: true });
    const waiting = registration.waiting!;
    await controller.apply();
    container.takeOver();
    // The reload was made and the browser's own leave prompt answered "Stay": the page runs on, the worker is in charge.
    registration.waiting = null;
    expect(reload).toHaveBeenCalledTimes(1);
    // The button works again: this click reloads again, and sends nothing (nothing waits).
    expect(await controller.apply()).toBe(true);
    expect(reload).toHaveBeenCalledTimes(2);
    expect(waiting.postMessage).toHaveBeenCalledTimes(1);
    // ...and it goes on working.
    expect(await controller.apply()).toBe(true);
    expect(reload).toHaveBeenCalledTimes(3);
  });

  it('a click after a Stay tells a newer waiting worker to take over, and reloads for that once', async () => {
    const { registration, container, controller, reload } = setup({ waiting: true });
    await controller.apply();
    container.takeOver();
    registration.waiting = null;
    expect(reload).toHaveBeenCalledTimes(1);
    // A newer version arrives and waits; the person presses Reload.
    const newer = registration.found_();
    registration.installed(newer);
    expect(await controller.apply()).toBe(true);
    expect(newer.postMessage).toHaveBeenCalledWith(SKIP_WAITING);
    container.takeOver();
    container.takeOver();
    expect(reload).toHaveBeenCalledTimes(2);
  });
});

describe('looking for a new version when the page is shown again', () => {
  it('does not ask within the hour', async () => {
    const { controller, registration, advance } = setup();
    advance(CHECK_AFTER_MS);
    await expect(controller.checkIfDue()).resolves.toBe(false);
    expect(registration.update).not.toHaveBeenCalled();
  });

  it('asks the browser after more than an hour, and then waits another hour', async () => {
    const { controller, registration, advance } = setup();
    advance(CHECK_AFTER_MS + 1);
    await expect(controller.checkIfDue()).resolves.toBe(true);
    expect(registration.update).toHaveBeenCalledTimes(1);
    advance(60_000);
    await expect(controller.checkIfDue()).resolves.toBe(false);
    advance(CHECK_AFTER_MS);
    await expect(controller.checkIfDue()).resolves.toBe(true);
    expect(registration.update).toHaveBeenCalledTimes(2);
  });

  it('carries on when the browser cannot reach the host', async () => {
    const { controller, registration, advance } = setup();
    registration.update.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    advance(CHECK_AFTER_MS + 1);
    await expect(controller.checkIfDue()).resolves.toBe(true);
  });
});
