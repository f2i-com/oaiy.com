import { describe, expect, it } from 'vitest';
import { AGREEMENT_MS, agreeToLeave, agreedToLeave, leaveGuardBlocks } from '../../src/pwa/leaveGuard';

describe('asking the page leave guard', () => {
  it('finds no objection when no handler objects', () => {
    const page = new EventTarget();
    expect(leaveGuardBlocks(page)).toBe(false);
    page.addEventListener('beforeunload', () => {});
    expect(leaveGuardBlocks(page)).toBe(false);
  });

  it('finds the objection of a handler that prevents the default, as the page own guard does', () => {
    const page = new EventTarget();
    page.addEventListener('beforeunload', (event) => event.preventDefault());
    expect(leaveGuardBlocks(page)).toBe(true);
  });

  it('runs every handler, so the page can save before anything else happens', () => {
    const page = new EventTarget();
    const seen: string[] = [];
    page.addEventListener('beforeunload', () => seen.push('save'));
    page.addEventListener('beforeunload', (event) => {
      seen.push('guard');
      event.preventDefault();
    });
    expect(leaveGuardBlocks(page)).toBe(true);
    expect(seen).toEqual(['save', 'guard']);
  });

  it('follows the handler as it is now (an objection that has passed is no objection)', () => {
    const page = new EventTarget();
    let running = true;
    page.addEventListener('beforeunload', (event) => {
      if (running) event.preventDefault();
    });
    expect(leaveGuardBlocks(page)).toBe(true);
    running = false;
    expect(leaveGuardBlocks(page)).toBe(false);
  });
});

describe('the person agreed to leave', () => {
  it('lasts for the reload that follows, and no longer', () => {
    const t = 1_000_000;
    expect(agreedToLeave(t)).toBe(false);
    agreeToLeave(t);
    expect(agreedToLeave(t + 1)).toBe(true);
    expect(agreedToLeave(t + AGREEMENT_MS - 1)).toBe(true);
    expect(agreedToLeave(t + AGREEMENT_MS)).toBe(false);
  });
});
