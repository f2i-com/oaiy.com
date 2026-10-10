// The setup wizard's download line. The engines report a download's speed in
// bytes a second; the line once printed that number with "MB/s" after it
// ("24117248.3 MB/s"), seen on a Mac's first model download.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { DownloadProgress } from './SetupParts';
import type { EngineDownload } from './api';

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const line = (dl: EngineDownload): string => {
  act(() => root.render(<DownloadProgress name="Catalog LLM" dl={dl} />));
  return host.querySelector('small')?.textContent ?? '';
};

const GIB = 1024 * 1024 * 1024;
const MIB = 1024 * 1024;

describe('the setup wizard’s download line', () => {
  it('says the speed in a unit, from the bytes a second the engines give', () => {
    const text = line({ id: 'catalog-llm', status: 'downloading', done: 1.5 * GIB, total: 6 * GIB, speed: 24 * MIB + 0.3 });
    expect(text).toBe('1.50 GiB of 6.00 GiB · 24.0 MiB/s');
    expect(text).not.toMatch(/\d{5,}/);
  });

  it('keeps to a unit for a slow download and for a fast one', () => {
    expect(line({ id: 'm', status: 'downloading', done: 0, total: GIB, speed: 300 * 1024 })).toContain(' · 300.0 KiB/s');
    expect(line({ id: 'm', status: 'downloading', done: 0, total: GIB, speed: 1.2 * GIB })).toContain(' · 1.20 GiB/s');
  });

  it('says no speed while nothing is moving, and which file of several it is on', () => {
    expect(line({ id: 'm', status: 'downloading', done: 0, total: GIB, speed: 0 })).toBe('0 B of 1.00 GiB');
    expect(line({ id: 'm', status: 'queued', done: 0, total: 0 })).toBe('Waiting to start…');
    expect(line({ id: 'm', status: 'downloading', done: MIB, total: GIB, speed: MIB, filesDone: 1, filesTotal: 3 })).toBe('1.0 MiB of 1.00 GiB · 1.0 MiB/s · file 2 of 3');
  });
});
