// The wizard's offer of the NVIDIA engine (SetupParts.tsx): shown in the desktop's window on a computer with an NVIDIA
// card that has no NVIDIA engine yet, it fetches it (the desktop checks its signature) and says when the engines run
// it; on a computer without one, or once it is there, it shows nothing.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const m = vi.hoisted(() => ({ nvidia: vi.fn(), fetchNvidia: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, isTauri: () => true, engines: { ...real.engines, nvidia: m.nvidia, fetchNvidia: m.fetchNvidia } };
});

import { NvidiaEngineOffer } from './SetupParts';
import type { NvidiaEngine } from './api';

const status = (over: Partial<NvidiaEngine>): NvidiaEngine => ({
  offered: true,
  nvidia: true,
  installed: false,
  version: '0.2.0',
  fetch: { state: 'idle', got: 0, total: null, error: null },
  ...over,
});

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  m.nvidia.mockReset();
  m.fetchNvidia.mockReset();
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const render = async (onInstalled = vi.fn()) => {
  await act(async () => {
    root.render(<NvidiaEngineOffer onInstalled={onInstalled} />);
  });
  return onInstalled;
};

const button = () => [...host.querySelectorAll('button')].find((b) => b.textContent?.includes('NVIDIA engine') || b.textContent?.includes('Try again'));

describe('the NVIDIA engine offer', () => {
  it('offers the engine on a computer with an NVIDIA card, and fetches it', async () => {
    m.nvidia.mockResolvedValue(status({}));
    m.fetchNvidia.mockImplementation(async () => {
      m.nvidia.mockResolvedValue(status({ installed: true, fetch: { state: 'done', got: 9, total: 9, error: null } }));
      return status({ installed: true });
    });
    const onInstalled = await render();
    expect(host.textContent).toContain('This computer has an NVIDIA card');
    await act(async () => {
      button()!.click();
    });
    expect(m.fetchNvidia).toHaveBeenCalledTimes(1);
    expect(onInstalled).toHaveBeenCalledTimes(1);
    expect(host.textContent).toContain('Installed');
    expect(button()).toBeUndefined();
  });

  it('shows nothing without an NVIDIA card, where no build is made, or once the engine is there', async () => {
    for (const s of [status({ nvidia: false }), status({ offered: false }), status({ installed: true })]) {
      // A new card each time: one asks for the status when it is shown.
      act(() => root.unmount());
      root = createRoot(host);
      m.nvidia.mockResolvedValue(s);
      await render();
      expect(host.textContent).toBe('');
    }
  });

  it('says why a fetch failed and offers it again', async () => {
    m.nvidia.mockResolvedValue(status({}));
    m.fetchNvidia.mockRejectedValue(new Error('The downloaded NVIDIA engine does not match its signature, so it was thrown away.'));
    await render();
    await act(async () => {
      button()!.click();
    });
    expect(host.textContent).toContain('does not match its signature');
    expect(button()?.textContent).toContain('Try again');
  });
});
