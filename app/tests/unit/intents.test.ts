// What OAIY Desktop may ask of the Agent's page by name (its setup wizard's
// "Answer calls and texts with OAIY"), evaluated into the page by embed.rs.
import { describe, expect, it, vi } from 'vitest';
import { answeringOn, installIntents, type IntentWindow } from '../../src/desktop/intents';
import { DEFAULT_MESSAGE_SETTINGS } from '../../src/settings';

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('intents from OAIY Desktop', () => {
  it('does a named intent, and nothing for a name it has no handler for', async () => {
    const w: IntentWindow = {};
    const answerWithOaiy = vi.fn();
    installIntents({ answerWithOaiy }, w);
    expect(w.__oaiyIntent!('answerWithOaiy')).toBe(true);
    expect(w.__oaiyIntent!('toString')).toBe(false);
    expect(w.__oaiyIntent!('alert(1)')).toBe(false);
    expect(w.__oaiyIntent!({ name: 'answerWithOaiy' })).toBe(false);
    await settle();
    expect(answerWithOaiy).toHaveBeenCalledTimes(1);
  });

  it('does the intents that came while the page was starting, once each', async () => {
    const w: IntentWindow = { __OAIY_INTENTS__: ['answerWithOaiy', 'answerWithOaiy', 'other'] };
    const answerWithOaiy = vi.fn();
    installIntents({ answerWithOaiy }, w);
    await settle();
    expect(answerWithOaiy).toHaveBeenCalledTimes(1);
    expect(w.__OAIY_INTENTS__).toEqual([]);
  });

  it('a handler that fails does not throw into the desktop, and stopping removes it', async () => {
    const w: IntentWindow = {};
    const stop = installIntents({ answerWithOaiy: () => Promise.reject(new Error('storage full')) }, w);
    expect(() => w.__oaiyIntent!('answerWithOaiy')).not.toThrow();
    await settle();
    stop();
    expect(w.__oaiyIntent).toBeUndefined();
  });

  it('turns on answering texts and calls, keeping the instructions', () => {
    const before = { ...DEFAULT_MESSAGE_SETTINGS, instructions: 'Be brief.', callBack: true };
    expect(answeringOn(before)).toEqual({ ...before, answer: true, calls: true });
    expect(DEFAULT_MESSAGE_SETTINGS.answer).toBe(false);
  });
});
