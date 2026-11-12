/**
 * What the Connect button and its card say, for each state a tab can be in with OAIY Desktop. Each is true of that state and no other:
 * a tab that has never connected sends nothing; one that has, asks the desktop each time the editor opens and every ten seconds while
 * it is open, and reads its services when it answers; one whose engine has an address of its own does the same at that address; and
 * after Disconnect it is the first again.
 *
 * Pure (no globals, no React), so a test holds each state's words and a look at the page shows the same words.
 */

/** How often the editor asks the desktop while it is open and linked (lib/desktopDetection.ts). */
export const POLL_SECONDS = 10;

export type ConnectState =
  /** No link: never connected, disconnected, or the engine's address is the default. */
  | { kind: 'never'; base: string }
  /** A link kept by Connect (the desktop answered it once). */
  | { kind: 'connected'; base: string; available: boolean }
  /** No Connect link, but the engine has an address of its own in Settings, which is a link too. */
  | { kind: 'address'; base: string; available: boolean };

export function connectState(input: { linkedByConnect: boolean; addressGiven: boolean; base: string; available: boolean }): ConnectState {
  if (input.linkedByConnect) return { kind: 'connected', base: input.base, available: input.available };
  if (input.addressGiven) return { kind: 'address', base: input.base, available: input.available };
  return { kind: 'never', base: input.base };
}

const EACH = `each time it opens and every ${POLL_SECONDS} seconds while it is open`;

/**
 * What stands under the button in Settings: for a tab that has never connected, the two sentences a person needs
 * before pressing it (what it asks, and that nothing is asked until then); for the others, all there is to say.
 */
export function connectSummary(state: ConnectState): string {
  if (state.kind !== 'never') return connectWords(state);
  return `Connect asks ${state.base} whether OAIY Desktop is running. Until you press it, this page asks nothing of your computer or your network.`;
}

/**
 * The rest, for a tab that has never connected, under "What Connect does": the one request it is, the browser's own
 * question, what the editor refuses to load until then, and what one answer leads to. '' for the other states.
 */
export function connectDetails(state: ConnectState): string {
  if (state.kind !== 'never') return '';
  return `It is one request (GET /api/health). Your browser may ask whether this site may connect to devices on your network: that question is this request. Until you connect, the editor also loads no picture, video or sound from an address on your computer or your network, judging by the address as written: a link that redirects there, or a name that points there, is left to your browser's own question. If the desktop answers, this browser keeps the link: from then on the editor asks it again ${EACH}, and reads its services. Disconnect ends that.`;
}

/** Everything said of a state, in one piece: the button's tooltip in the sidebar (the card shows it in the two parts above). */
export function connectWords(state: ConnectState): string {
  switch (state.kind) {
    case 'never':
      return `${connectSummary(state)} ${connectDetails(state)}`;
    case 'connected':
      return state.available
        ? `Connected to ${state.base}. This browser keeps the link: the editor asks OAIY Desktop whether it is running, and reads its services, ${EACH}. Disconnect ends that.`
        : `This browser has a link to OAIY Desktop at ${state.base}, so the editor asks there ${EACH}; nothing answers now. Connect asks again now; Disconnect forgets the link and ends that.`;
    case 'address':
      return `The engine's address in Settings is ${state.base}, so the editor asks there ${EACH}${state.available ? '' : '; nothing answers now'}. Reset above gives the default address back and ends that (unless you have also pressed Connect).`;
  }
}

/** The sidebar's line before the button. */
export function connectLead(state: ConnectState): string {
  if (state.kind === 'never') return 'Already running OAIY Desktop? ';
  if (state.kind === 'connected' && state.available) return 'Linked to OAIY Desktop. ';
  return 'Linked to OAIY Desktop, which does not answer now. ';
}
