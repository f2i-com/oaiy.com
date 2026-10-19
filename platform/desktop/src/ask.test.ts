// Asking before something is done: a real answer, inside OAIY's window and outside it. Inside, the dialog plugin's
// `message` command with OK and Cancel; never the `confirm` the plugin puts in the window's place, whose Promise was
// always taken for a yes.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ask } from './ask';

type Internals = { invoke: ReturnType<typeof vi.fn> };
const w = window as unknown as { __TAURI_INTERNALS__?: Internals };

beforeEach(() => {
  delete w.__TAURI_INTERNALS__;
});
afterEach(() => {
  delete w.__TAURI_INTERNALS__;
  vi.restoreAllMocks();
  vi.stubGlobal('confirm', () => true);
});

describe('ask', () => {
  it('outside the window, is the browser’s own confirm, as the person answers', async () => {
    const confirm = vi.fn().mockReturnValueOnce(true).mockReturnValueOnce(false);
    vi.stubGlobal('confirm', confirm);
    expect(await ask('Delete it?')).toBe(true);
    expect(await ask('Delete it?')).toBe(false);
    expect(confirm).toHaveBeenCalledWith('Delete it?');
  });

  it('inside the window, shows the plugin’s message dialog with OK and Cancel, and says yes only for OK', async () => {
    const invoke = vi.fn().mockResolvedValueOnce('Ok').mockResolvedValueOnce('Cancel');
    w.__TAURI_INTERNALS__ = { invoke };
    const confirm = vi.fn(() => true);
    vi.stubGlobal('confirm', confirm);
    expect(await ask('Restart now?')).toBe(true);
    expect(await ask('Restart now?')).toBe(false);
    expect(invoke).toHaveBeenCalledWith('plugin:dialog|message', { message: 'Restart now?', title: 'OAIY', kind: 'warning', buttons: 'OkCancel' }, undefined);
    // (the window's confirm is the plugin's broken one there: never asked)
    expect(confirm).not.toHaveBeenCalled();
  });

  it('takes a title and a kind', async () => {
    const invoke = vi.fn().mockResolvedValue('Ok');
    w.__TAURI_INTERNALS__ = { invoke };
    await ask('Remove it?', { title: 'Plugins', kind: 'info' });
    expect(invoke.mock.calls[0][1]).toMatchObject({ title: 'Plugins', kind: 'info' });
  });

  it('where the dialog cannot be shown, asks with confirm, and takes only a real true from it', async () => {
    w.__TAURI_INTERNALS__ = { invoke: vi.fn().mockRejectedValue(new Error('Command plugin:dialog|message not found')) };
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('confirm', () => true);
    expect(await ask('Unpair it?')).toBe(true);
    // The plugin's replacement: an async confirm calling a command 2.7 no longer has. A Promise is no yes.
    vi.stubGlobal('confirm', () => Promise.reject(new Error('Command plugin:dialog|confirm not found')));
    expect(await ask('Unpair it?')).toBe(false);
    vi.stubGlobal('confirm', () => Promise.resolve({}));
    expect(await ask('Unpair it?')).toBe(false);
    vi.stubGlobal('confirm', () => undefined);
    expect(await ask('Unpair it?')).toBe(false);
  });
});
