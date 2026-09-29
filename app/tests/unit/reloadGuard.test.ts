import { describe, expect, it } from 'vitest';
import { OLD_RELOAD_KEY, RELOAD_KEY, mayReloadForIsolation } from '../../src/pwa/reloadGuard';

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
    expect(s.data.has(OLD_RELOAD_KEY)).toBe(false);
  });

  it('refuses a second reload within a minute and allows one after it', () => {
    const s = store();
    expect(mayReloadForIsolation(s, NOW)).toBe(true);
    expect(mayReloadForIsolation(s, NOW + 59_999)).toBe(false);
    expect(mayReloadForIsolation(s, NOW + 60_000)).toBe(true);
  });

  it('honours a reload the old key recorded a moment ago, once, and moves it to the new key', () => {
    const s = store({ [OLD_RELOAD_KEY]: String(NOW - 5_000) });
    expect(mayReloadForIsolation(s, NOW)).toBe(false);
    expect(s.data.has(OLD_RELOAD_KEY)).toBe(false);
    expect(s.data.get(RELOAD_KEY)).toBe(String(NOW - 5_000));
    // The record survived the move: still no reload a little later, then one after the minute.
    expect(mayReloadForIsolation(s, NOW + 10_000)).toBe(false);
    expect(mayReloadForIsolation(s, NOW + 60_000)).toBe(true);
  });

  it('ignores an old key that is a minute or more old, and still drops it', () => {
    const s = store({ [OLD_RELOAD_KEY]: String(NOW - 120_000) });
    expect(mayReloadForIsolation(s, NOW)).toBe(true);
    expect(s.data.has(OLD_RELOAD_KEY)).toBe(false);
    expect(s.data.get(RELOAD_KEY)).toBe(String(NOW));
  });

  it('takes the later of the two keys', () => {
    const s = store({ [OLD_RELOAD_KEY]: String(NOW - 30_000), [RELOAD_KEY]: String(NOW - 90_000) });
    expect(mayReloadForIsolation(s, NOW)).toBe(false);
    expect(s.data.get(RELOAD_KEY)).toBe(String(NOW - 30_000));
  });

  it('treats a value that is not a number as never having reloaded', () => {
    const s = store({ [RELOAD_KEY]: 'nonsense', [OLD_RELOAD_KEY]: '' });
    expect(mayReloadForIsolation(s, NOW)).toBe(true);
  });

  it('does not reload when storage cannot say (no loop through a blocked store)', () => {
    expect(mayReloadForIsolation(store({}, true), NOW)).toBe(false);
  });
});
