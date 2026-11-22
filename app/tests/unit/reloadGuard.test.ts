import { describe, expect, it } from 'vitest';
import { RELOAD_KEY, mayReloadForIsolation } from '../../src/pwa/reloadGuard';

/** A Storage stand-in that can also refuse. */
function store(initial: Record<string, string> = {}, fail = false) {
  const data = new Map(Object.entries(initial));
  return {
    data,
    getItem: (k: string) => {
      if (fail) throw new Error('blocked');
      return data.get(k) ?? null;
    },
    setItem: (k: string, v: string) => {
      if (fail) throw new Error('blocked');
      data.set(k, v);
    },
    removeItem: (k: string) => {
      if (fail) throw new Error('blocked');
      data.delete(k);
    },
  };
}

const NOW = 1_800_000_000_000;

describe('the isolation reload guard', () => {
  it('lets the first visit reload once and records it under the OAIY key', () => {
    const s = store();
    expect(mayReloadForIsolation(s, NOW)).toBe(true);
    expect(s.data.get(RELOAD_KEY)).toBe(String(NOW));
  });

  it('refuses a second reload within a minute and allows one after it', () => {
    const s = store();
    expect(mayReloadForIsolation(s, NOW)).toBe(true);
    expect(mayReloadForIsolation(s, NOW + 59_999)).toBe(false);
    expect(mayReloadForIsolation(s, NOW + 60_000)).toBe(true);
  });

  it('treats a value that is not a number as never having reloaded', () => {
    const s = store({ [RELOAD_KEY]: 'nonsense' });
    expect(mayReloadForIsolation(s, NOW)).toBe(true);
  });

  it('does not reload when storage cannot say (no loop through a blocked store)', () => {
    expect(mayReloadForIsolation(store({}, true), NOW)).toBe(false);
  });
});
