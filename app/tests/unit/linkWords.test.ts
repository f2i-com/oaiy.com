import { describe, expect, it } from 'vitest';
import { desktopLookingWords, findOaiyTitle, oaiyFoundWords, type DesktopState, type OaiyState } from '../../src/ui/linkWords';

const FOUND = 'http://192.168.1.20:8080';

describe('what Settings says about how OAIY is found, in each state the page can be in', () => {
  const own: OaiyState = { kind: 'own' };
  const never: OaiyState = { kind: 'never' };
  const saved: OaiyState = { kind: 'saved', origin: FOUND, withKey: false };
  const savedWithKey: OaiyState = { kind: 'saved', origin: FOUND, withKey: true };

  it("OAIY's own window: it is found on its own, and nothing is said about a button", () => {
    expect(oaiyFoundWords(own)).toBe('OAIY is found on its own');
    expect(findOaiyTitle(own)).not.toMatch(/reaches out|also asks/);
  });

  it('a tab that has found none, or has forgotten it: nothing is sent until Find OAIY is pressed, and that is all it says', () => {
    const words = oaiyFoundWords(never);
    expect(words).toMatch(/only when you press Find OAIY/);
    expect(words).toMatch(/sends nothing to your computer or your network/);
    expect(words).toMatch(/may ask you to allow it/);
    expect(words).not.toContain(FOUND);
    expect(findOaiyTitle(never)).toMatch(/reaches out to your computer only when you press this/);
  });

  it('a tab that has found an OAIY: it asks it at that address each time it opens (and says with the key, or without), and how to stop that', () => {
    for (const state of [saved, savedWithKey]) {
      const words = oaiyFoundWords(state);
      expect(words).toContain(FOUND);
      expect(words).toMatch(/each time it opens/);
      expect(words).toMatch(/Forget OAIY, or clearing the address, ends that/);
      // What is true of a tab that has found none is not true of this one, and must not be said.
      expect(words).not.toMatch(/sends nothing|only when you press|never/);
      expect(findOaiyTitle(state)).toContain(FOUND);
      expect(findOaiyTitle(state)).toMatch(/also asks OAIY at .* each time it opens/);
      expect(findOaiyTitle(state)).not.toMatch(/only when you press this/);
    }
    expect(oaiyFoundWords(savedWithKey)).toMatch(/sending the key below/);
    expect(oaiyFoundWords(saved)).not.toMatch(/key/);
  });

  it('each state says something the others do not', () => {
    const all = [own, never, saved].map(oaiyFoundWords);
    expect(new Set(all).size).toBe(3);
  });
});

describe('what the pairing dialog says as it opens, in each state', () => {
  const origin = 'http://127.0.0.1:17972';
  const given: DesktopState = { kind: 'given' };
  const unpaired: DesktopState = { kind: 'unpaired', origin };
  const paired: DesktopState = { kind: 'paired', origin };

  it("OAIY's own window keeps its words", () => {
    expect(desktopLookingWords(given, true)).toBe("This is OAIY's own window: texts and calls to the phone come here.");
    expect(desktopLookingWords(given, false)).toBe("This is OAIY's own window: it is OAIY Desktop's.");
  });

  it('a tab that is not paired asks only because the dialog is open, and says so, with the address and the browser prompt', () => {
    const words = desktopLookingWords(unpaired, true);
    expect(words).toContain(origin);
    expect(words).toMatch(/because you opened this dialog/);
    expect(words).toMatch(/asks nothing of your computer for OAIY Desktop until then/);
    expect(words).toMatch(/may ask whether it may connect to your network/);
    expect(words).not.toMatch(/paired with|keeps in touch/);
  });

  it('a tab that is paired keeps in touch with the desktop while it is open: it does not say it reaches out only from here', () => {
    const words = desktopLookingWords(paired, true);
    expect(words).toContain(origin);
    expect(words).toMatch(/is paired with OAIY Desktop/);
    expect(words).toMatch(/keeps in touch with it while it is open/);
    expect(words).not.toMatch(/only from here|asks nothing|because you opened this dialog/);
  });
});
