// The setup guide's rendering contract. setupGuide.test.ts already pins WHICH
// rows are done; this pins what the panel actually puts on screen — that a
// blocker replaces the generic detail, that a caller-supplied control wins over
// the navigation link, and that a done row offers neither.
//
// Same convention as AiProvidersPanel.test.tsx: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import SetupGuidePanel from './SetupGuidePanel';
import type { SetupInput } from './setupGuide';
import type { RuntimeStatus } from './api';

const READY = { ready: true, flowRuntime: { cliResolved: true, cliKind: 'node' } } as RuntimeStatus;
const BROKEN = {
  ready: false,
  flowRuntime: { cliResolved: false, cliKind: 'missing', detail: 'Node is not installed.' },
} as RuntimeStatus;

const state: SetupInput = {
  runtime: BROKEN,
  providers: [],
  services: [],
  plugins: [],
  connected: [],
};

let host: HTMLDivElement;
let root: Root;

function render(props: Partial<React.ComponentProps<typeof SetupGuidePanel>> = {}) {
  act(() => {
    root.render(
      <SetupGuidePanel
        {...state}
        onNavigate={vi.fn()}
        onDismiss={vi.fn()}
        {...props}
      />,
    );
  });
}

const rows = () => Array.from(host.querySelectorAll('.setup-step'));
const rowFor = (title: string) =>
  rows().find((li) => li.querySelector('strong')?.textContent?.includes(title))!;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('SetupGuidePanel', () => {
  it('renders every step, with progress counted over required ones only', () => {
    render();
    expect(rows()).toHaveLength(4);
    // Runtime broken, no AI: 0 of the 2 required steps.
    expect(host.querySelector('.section-title')?.textContent).toContain('0 of 2');
  });

  it('shows the backend’s blocker in place of the generic detail', () => {
    render();
    expect(rowFor('flow runtime').textContent).toContain('Node is not installed.');
    expect(rowFor('flow runtime').textContent).not.toContain('bundled engine');
  });

  it('marks a satisfied step done and drops its call to action', () => {
    render({ runtime: READY });
    const row = rowFor('flow runtime');
    expect(row.className).toContain('is-done');
    expect(row.querySelector('button')).toBeNull();
    expect(row.querySelector('.setup-done-label')).not.toBeNull();
  });

  it('prefers a caller-supplied action over the navigation link', () => {
    const onNavigate = vi.fn();
    render({
      onNavigate,
      actions: { runtime: <button className="install-node">Install Node</button> },
    });
    const row = rowFor('flow runtime');
    expect(row.querySelector('.install-node')).not.toBeNull();
    // The link it replaced must be gone, not merely hidden beside it.
    expect(row.textContent).not.toContain('Fix the runtime');

    // Other rows keep their links, and those still navigate.
    act(() => {
      rowFor('a model').querySelector('button')!.dispatchEvent(
        new MouseEvent('click', { bubbles: true }),
      );
    });
    expect(onNavigate).toHaveBeenCalledWith('providers');
  });

  it('congratulates only when the REQUIRED steps pass, optional ones outstanding', () => {
    render({
      runtime: READY,
      providers: [
        { id: 'p', enabled: true, hasKey: true, allowLocal: false } as never,
      ],
    });
    expect(host.textContent).toContain('You’re set up');
    // …and still lists the optional work rather than hiding it.
    expect(rowFor('plugin').className).not.toContain('is-done');
  });

  it('reports dismissal to its caller', () => {
    const onDismiss = vi.fn();
    render({ onDismiss });
    act(() => {
      host
        .querySelector('[aria-label="Dismiss the setup guide"]')!
        .dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(onDismiss).toHaveBeenCalled();
  });
});
