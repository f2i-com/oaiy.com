// `oaiy://navigate` (CONTROL_API.md §1, "Showing a page"): the desktop asks
// the dashboard for a setup step, a plugin's page, or one of its own views.
import { afterEach, describe, expect, it, vi } from 'vitest';

const listeners = vi.hoisted(() => ({ handler: null as null | ((e: { payload: unknown }) => void), unlisten: vi.fn(), listen: vi.fn() }));
vi.mock('@tauri-apps/api/event', () => ({
  listen: (event: string, handler: (e: { payload: unknown }) => void) => {
    listeners.listen(event);
    listeners.handler = handler;
    return Promise.resolve(listeners.unlisten);
  },
}));

import { NAVIGATE_EVENT, onNavigate, parseNavigate } from './navigate';

afterEach(() => {
  delete (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  listeners.handler = null;
  listeners.listen.mockReset();
  listeners.unlisten.mockReset();
});

describe('a navigation from the desktop', () => {
  it('opens a plugin’s setup at a step, or the first-run wizard', () => {
    expect(parseNavigate({ view: 'setup', pluginId: 'aokie', stepId: 'pair' })).toEqual({ kind: 'setup', pluginId: 'aokie', stepId: 'pair' });
    expect(parseNavigate({ view: 'setup', pluginId: 'aokie' })).toEqual({ kind: 'setup', pluginId: 'aokie' });
    expect(parseNavigate({ view: 'setup' })).toEqual({ kind: 'setup' });
    expect(parseNavigate({ view: 'setup', stepId: 'plugin:aokie' })).toEqual({ kind: 'setup', stepId: 'plugin:aokie' });
  });

  it('opens a plugin’s page, or one of the dashboard’s own views', () => {
    expect(parseNavigate({ view: 'plugin:aokie:receptionist' })).toEqual({ kind: 'view', view: 'plugin:aokie:receptionist' });
    expect(parseNavigate({ view: 'engines' })).toEqual({ kind: 'view', view: 'engines' });
    expect(parseNavigate({ view: 'agent-settings' })).toEqual({ kind: 'view', view: 'agent-settings' });
  });

  it('opens the Calendar by its old id, and Hours & Services by its own', () => {
    expect(parseNavigate({ view: 'calendar' })).toEqual({ kind: 'view', view: 'calendar' });
    expect(parseNavigate({ view: 'hours' })).toEqual({ kind: 'view', view: 'hours' });
  });

  it('ignores anything else, and names it cannot be', () => {
    expect(parseNavigate(null)).toBeNull();
    expect(parseNavigate('setup')).toBeNull();
    expect(parseNavigate({})).toBeNull();
    expect(parseNavigate({ view: 'somewhere' })).toBeNull();
    expect(parseNavigate({ view: 'plugin:Aokie!:x' })).toBeNull();
    expect(parseNavigate({ view: 'plugin:aokie' })).toBeNull();
    // A bad plugin or step id is dropped; the setup page still opens.
    expect(parseNavigate({ view: 'setup', pluginId: '../x', stepId: 'Pair Now' })).toEqual({ kind: 'setup' });
  });

  it('is a no-op in a plain browser', () => {
    const cb = vi.fn();
    const stop = onNavigate(cb);
    expect(listeners.listen).not.toHaveBeenCalled();
    stop();
  });

  it('listens for oaiy://navigate in OAIY’s window, and stops', async () => {
    (window as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = { invoke: vi.fn() };
    const cb = vi.fn();
    const stop = onNavigate(cb);
    expect(listeners.listen).toHaveBeenCalledWith(NAVIGATE_EVENT);
    listeners.handler!({ payload: { view: 'setup', pluginId: 'aokie', stepId: 'pair' } });
    listeners.handler!({ payload: { view: 'nowhere' } });
    expect(cb).toHaveBeenCalledTimes(1);
    expect(cb).toHaveBeenCalledWith({ kind: 'setup', pluginId: 'aokie', stepId: 'pair' });
    await Promise.resolve();
    stop();
    expect(listeners.unlisten).toHaveBeenCalled();
    listeners.handler!({ payload: { view: 'engines' } });
    expect(cb).toHaveBeenCalledTimes(1);
  });
});
