// The badge says what the host found out about a plugin's package, and where it has a
// reason, puts it in the tooltip rather than on the card.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import PackageTrustBadge, { trustLabel } from './PackageTrustBadge';
import type { PackageTrust } from './api';

let host: HTMLDivElement;
let root: Root;

async function badge(trust: PackageTrust): Promise<HTMLElement> {
  await act(async () => {
    root.render(<PackageTrustBadge trust={trust} />);
  });
  return host.querySelector('.badge') as HTMLElement;
}

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('the package trust badge', () => {
  it('a verified package names its publisher, in the green badge, and the tooltip says which key and release', async () => {
    const el = await badge({ state: 'verified', publisher: 'Aokie', keyId: 'fl-aokie-2026a', version: '0.1.0' });
    expect(el.textContent).toBe('verified · Aokie');
    expect(el.className).toContain('badge-ok');
    expect(el.getAttribute('title')).toBe('Signed by Aokie (key fl-aokie-2026a), release 0.1.0. Every file is as signed.');
  });

  it('a package the person trusted is neutral, and says so', async () => {
    const el = await badge({ state: 'trusted-local', reason: 'You trusted this exact package on 2026-09-29.', trustedAt: '2026-09-29T01:02:03Z' });
    expect(el.textContent).toBe('trusted by you');
    expect(el.className).toContain('badge-neutral');
    expect(el.getAttribute('title')).toContain('You trusted this exact package');
  });

  it('an unsigned package in a developer build is neutral, not an alarm', async () => {
    const el = await badge({ state: 'unsigned-dev', reason: 'Not signed. It runs because this is a developer build.' });
    expect(el.textContent).toBe('unsigned (dev)');
    expect(el.className).toContain('badge-neutral');
  });

  it('an unsigned package a release build holds back, and a quarantined one, are the red badge with the host’s reason', async () => {
    const unsigned = await badge({ state: 'unsigned', reason: 'Not signed by a publisher this OAIY trusts.' });
    expect(unsigned.textContent).toBe('unsigned');
    expect(unsigned.className).toContain('badge-err');
    expect(unsigned.getAttribute('title')).toBe('Not signed by a publisher this OAIY trusts.');

    const quarantined = await badge({ state: 'quarantined', reason: 'Quarantined: digest mismatch: aokie-plugin.exe.' });
    expect(quarantined.textContent).toBe('quarantined');
    expect(quarantined.className).toContain('badge-err');
    expect(quarantined.getAttribute('title')).toBe('Quarantined: digest mismatch: aokie-plugin.exe.');
  });

  it('a verified package with no publisher name is still called verified', () => {
    expect(trustLabel({ state: 'verified' })).toBe('verified');
  });

  it('marks the state on the element, for anything that wants to find it', async () => {
    const el = await badge({ state: 'quarantined', reason: 'x' });
    expect(el.getAttribute('data-trust')).toBe('quarantined');
  });
});
