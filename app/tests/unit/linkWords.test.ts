import { describe, expect, it } from 'vitest';
import { desktopLookingWords, findOaiyTitle, oaiyFoundWords, oaiyStateOf, pairedTabWords, type DesktopState, type OaiyLink, type OaiyState } from '../../src/ui/linkWords';

const FOUND = 'http://192.168.1.20:8080';
const TYPED = 'http://192.168.1.30:8080/v1';
const DESK = 'http://127.0.0.1:17972';

const NONE: OaiyLink = { kind: 'none' };
const FOUND_LINK: OaiyLink = { kind: 'found', origin: FOUND, withKey: false };
const FOUND_KEY: OaiyLink = { kind: 'found', origin: FOUND, withKey: true };
const TYPED_LINK: OaiyLink = { kind: 'typed', address: TYPED };

/** Every state a page can be in: the window, and a tab that is paired or not, holding nothing of OAIY, a found OAIY (with a key or without) or a typed address. */
const TABS: Array<[string, OaiyState]> = [];
for (const [desktopName, desktop] of [['not paired', null], ['paired', DESK]] as const) {
  for (const [linkName, oaiy] of [['holding nothing', NONE], ['with an OAIY found', FOUND_LINK], ['with an OAIY found, and a key', FOUND_KEY], ['with a media address typed', TYPED_LINK]] as const) {
    TABS.push([`a tab, ${desktopName}, ${linkName}`, { kind: 'tab', desktop, oaiy }]);
  }
}
const STATES: Array<[string, OaiyState]> = [["OAIY's own window", { kind: 'own' }], ...TABS];

/** Everything the settings say of a state: the clause, the sentence about the desktop, and the tooltip of Find OAIY. */
const said = (state: OaiyState) => [oaiyFoundWords(state), pairedTabWords(state) ?? '', findOaiyTitle(state)].join('\n');

describe('what Settings says about how OAIY is found, in every state the page can be in', () => {
  it("OAIY's own window: it is found on its own, and nothing is said about a button or a pairing", () => {
    const own: OaiyState = { kind: 'own' };
    expect(oaiyFoundWords(own)).toBe('OAIY is found on its own');
    expect(pairedTabWords(own)).toBeNull();
    expect(findOaiyTitle(own)).toBe('Ask OAIY for its details (/v1/discovery) and fill everything in');
  });

  it('a tab that is not paired and holds nothing of OAIY: nothing is sent to look for it until Find OAIY is pressed, and that is all it says', () => {
    const state: OaiyState = { kind: 'tab', desktop: null, oaiy: NONE };
    const words = oaiyFoundWords(state);
    expect(words).toMatch(/only when you press Find OAIY/);
    expect(words).toMatch(/sends nothing to your computer or your network to look for it/);
    expect(words).toMatch(/may ask you to allow it/);
    expect(words).not.toContain(FOUND);
    expect(pairedTabWords(state)).toBeNull();
    expect(findOaiyTitle(state)).toMatch(/looks for OAIY only when you press this/);
  });

  it('a tab that has found an OAIY: it asks it at that address each time it opens (and says with the key, or without), and how to stop that', () => {
    for (const oaiy of [FOUND_LINK, FOUND_KEY]) {
      for (const desktop of [null, DESK]) {
        const state: OaiyState = { kind: 'tab', desktop, oaiy };
        const words = oaiyFoundWords(state);
        expect(words).toContain(FOUND);
        expect(words).toMatch(/each time it opens/);
        expect(words).toMatch(/Forget OAIY, or clearing the address, ends that/);
        // What is true of a tab that holds nothing is not true of this one, and must not be said.
        expect(words).not.toMatch(/sends nothing|only when you press|never/);
        expect(findOaiyTitle(state)).toMatch(/also asks OAIY at .* each time it opens/);
        expect(findOaiyTitle(state)).not.toMatch(/only when you press this/);
      }
    }
    expect(oaiyFoundWords({ kind: 'tab', desktop: null, oaiy: FOUND_KEY })).toMatch(/sending the key below/);
    expect(oaiyFoundWords({ kind: 'tab', desktop: null, oaiy: FOUND_LINK })).not.toMatch(/key/);
  });

  it('a tab with a media address typed and no OAIY found: the agent uses the address when it makes media, nothing goes to it before, and Find OAIY asks there', () => {
    for (const desktop of [null, DESK]) {
      const state: OaiyState = { kind: 'tab', desktop, oaiy: TYPED_LINK };
      const words = oaiyFoundWords(state);
      expect(words).toContain(TYPED);
      expect(words).toMatch(/uses it when it makes media/);
      expect(words).toMatch(/sends nothing to it before then/);
      expect(words).toMatch(/only when you press Find OAIY/);
      expect(words).not.toMatch(/each time it opens/);
      expect(findOaiyTitle(state)).toContain(`asks the address you typed (${TYPED})`);
    }
  });

  it('a tab that is paired keeps in touch with its desktop whatever it holds of OAIY, and says so where it says how OAIY is found and on the button', () => {
    for (const [name, state] of TABS.filter(([, s]) => s.kind === 'tab' && s.desktop)) {
      const words = pairedTabWords(state);
      expect(words, name).toContain(DESK);
      expect(words, name).toMatch(/is paired with OAIY Desktop/);
      expect(words, name).toMatch(/keeps in touch with it while it is open/);
      expect(words, name).toMatch(/whether or not OAIY is found/);
      expect(findOaiyTitle(state), name).toContain(`paired desktop at ${DESK}`);
      // The claim of a tab that is not paired is not made: a paired tab asks its desktop in the background, and says nothing goes out only when a button is pressed.
      expect(oaiyFoundWords(state), name).not.toMatch(/sends nothing to your computer/);
      expect(findOaiyTitle(state), name).not.toMatch(/reaches out to your computer only/);
    }
  });

  it('the claim that nothing goes out is made in one state only: a tab that is not paired and holds nothing of OAIY', () => {
    for (const [name, state] of STATES) {
      const claims = /sends nothing to your computer or your network/.test(said(state));
      expect(claims, name).toBe(state.kind === 'tab' && !state.desktop && state.oaiy.kind === 'none');
    }
  });

  it('what a state names is what it holds: the desktop only when paired, the address only when found or typed, and never the other state\'s', () => {
    for (const [name, state] of STATES) {
      const text = said(state);
      const desktop = state.kind === 'tab' ? state.desktop : null;
      const oaiy = state.kind === 'tab' ? state.oaiy : NONE;
      expect(text.includes(DESK), name).toBe(!!desktop);
      expect(text.includes(FOUND), name).toBe(oaiy.kind === 'found');
      expect(text.includes(TYPED), name).toBe(oaiy.kind === 'typed');
    }
  });

  it('each state says something the others do not', () => {
    const all = STATES.map(([, state]) => `${oaiyFoundWords(state)}\n${pairedTabWords(state) ?? ''}`);
    expect(new Set(all).size).toBe(STATES.length);
    const titles = STATES.map(([, state]) => findOaiyTitle(state));
    // The button does not mention the key, so found with one and without say the same there: the window, and three holdings of OAIY paired or not.
    expect(new Set(titles).size).toBe(1 + 3 * 2);
  });
});

describe('the state from what the page holds (Settings asks this)', () => {
  const base = { own: false, desktop: null, withKey: false, typed: '' };

  it("OAIY's own window is its own state whatever else it holds", () => {
    expect(oaiyStateOf({ ...base, own: true, desktop: DESK, discovered: FOUND, typed: TYPED })).toEqual({ kind: 'own' });
  });

  it('a tab holds nothing, a found OAIY, or a typed address; and is paired or not', () => {
    expect(oaiyStateOf(base)).toEqual({ kind: 'tab', desktop: null, oaiy: NONE });
    expect(oaiyStateOf({ ...base, desktop: DESK })).toEqual({ kind: 'tab', desktop: DESK, oaiy: NONE });
    expect(oaiyStateOf({ ...base, discovered: FOUND })).toEqual({ kind: 'tab', desktop: null, oaiy: FOUND_LINK });
    expect(oaiyStateOf({ ...base, discovered: FOUND, withKey: true })).toEqual({ kind: 'tab', desktop: null, oaiy: FOUND_KEY });
    expect(oaiyStateOf({ ...base, typed: TYPED })).toEqual({ kind: 'tab', desktop: null, oaiy: TYPED_LINK });
    expect(oaiyStateOf({ ...base, desktop: DESK, typed: TYPED })).toEqual({ kind: 'tab', desktop: DESK, oaiy: TYPED_LINK });
    expect(oaiyStateOf({ ...base, desktop: DESK, discovered: FOUND })).toEqual({ kind: 'tab', desktop: DESK, oaiy: FOUND_LINK });
  });

  it('an OAIY that was found outranks the address (it is what the address says), and blank space is no address', () => {
    expect(oaiyStateOf({ ...base, discovered: FOUND, typed: `${FOUND}/v1` })).toEqual({ kind: 'tab', desktop: null, oaiy: FOUND_LINK });
    expect(oaiyStateOf({ ...base, typed: '   ' })).toEqual({ kind: 'tab', desktop: null, oaiy: NONE });
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
