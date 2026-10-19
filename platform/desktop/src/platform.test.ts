// The kind of computer the window runs on, from what its webview says, and the path examples written its way.
import { describe, expect, it } from 'vitest';
import { hostOS, pathExamples } from './platform';

describe('hostOS', () => {
  it('reads each system’s webview', () => {
    expect(hostOS({ platform: 'MacIntel', userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15' })).toBe('macos');
    expect(hostOS({ platform: 'Win32', userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edg/141.0' })).toBe('windows');
    expect(hostOS({ platform: 'Linux x86_64', userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15' })).toBe('linux');
    expect(hostOS({})).toBe('other');
  });
});

describe('pathExamples', () => {
  it('writes a drive letter on Windows, a volume on a Mac, and a mount elsewhere', () => {
    expect(pathExamples('windows')).toEqual({ data: 'D:\\OAIY', models: 'E:\\models', extra: 'E:\\ckpts' });
    expect(pathExamples('macos').models).toBe('/Volumes/Data/models');
    expect(pathExamples('linux').extra).toBe('/mnt/data/ckpts');
    expect(Object.values(pathExamples('macos')).some((p) => p.includes(':\\'))).toBe(false);
  });
});
