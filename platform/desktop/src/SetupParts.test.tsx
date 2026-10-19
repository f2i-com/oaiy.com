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

const GB = 1e9;
const MB = 1e6;

describe('the setup wizard’s download line', () => {
  it('says the speed in a unit, from the bytes a second the engines give, in decimal units as the catalogue does', () => {
    const text = line({ id: 'catalog-llm', status: 'downloading', done: 1.5 * GB, total: 6.1 * GB, speed: 24 * MB + 0.3 });
    expect(text).toBe('1.5 GB of 6.1 GB · 24.0 MB/s');
    expect(text).not.toMatch(/\d{5,}/);
  });

  it('keeps to a unit for a slow download and for a fast one', () => {
    expect(line({ id: 'm', status: 'downloading', done: 0, total: GB, speed: 300 * 1000 })).toContain(' · 300.0 kB/s');
    expect(line({ id: 'm', status: 'downloading', done: 0, total: GB, speed: 1.2 * GB })).toContain(' · 1.2 GB/s');
  });

  it('says no speed while nothing is moving, and which file of several it is on', () => {
    expect(line({ id: 'm', status: 'downloading', done: 0, total: GB, speed: 0 })).toBe('0 B of 1.0 GB');
    expect(line({ id: 'm', status: 'queued', done: 0, total: 0 })).toBe('Waiting to start…');
    expect(line({ id: 'm', status: 'downloading', done: MB, total: GB, speed: MB, filesDone: 1, filesTotal: 3 })).toBe('1.0 MB of 1.0 GB · 1.0 MB/s · file 2 of 3');
  });
});
