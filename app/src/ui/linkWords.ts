/**
 * What the settings and the pairing dialog say about when this page reaches out to the person's computer, for each state it can be in.
 * Each sentence is true of that state and no other, and says what stops it: a page that says it "never" reaches out and does, or
 * that says it reaches out "only from here" while it has been asking in the background since it opened, tells the person less than
 * they need.
 *
 * What a tab does is decided by two things the person did, and they are independent: whether it is PAIRED with OAIY Desktop (it then
 * keeps in touch with the desktop while it is open, whether or not OAIY is found), and what it holds of OAIY itself (the engine's
 * gateway and media service): nothing, an OAIY it FOUND (asked at each opening), or a media address the person TYPED (used when the
 * agent makes media, asked only by Find OAIY). OAIY's own window is a fourth state: it looks on its own.
 *
 * Pure, so a test holds each state's words, and one look at the page shows the same words.
 */

/** What this page holds of OAIY (the engine's gateway and media service). */
export type OaiyLink =
  /** Nothing: OAIY was never found here, or was forgotten, and no media address was typed. */
  | { kind: 'none' }
  /** An OAIY that was found: the page asks it at that address each time it opens. */
  | { kind: 'found'; origin: string; withKey: boolean }
  /** A media address typed by hand, with no OAIY found: used when the agent makes media, and asked only by Find OAIY. */
  | { kind: 'typed'; address: string };

/** Where this page stands with OAIY and with OAIY Desktop. */
export type OaiyState =
  /** OAIY's own window or the desktop shell: it looks for OAIY as it opens, and the desktop is given. */
  | { kind: 'own' }
  /** A tab in a browser: paired with the desktop at `desktop` (null: not paired), and holding `oaiy`. */
  | { kind: 'tab'; desktop: string | null; oaiy: OaiyLink };

/** The state from what the page holds. `discovered` is the origin of the OAIY that was found; `typed` the media address as typed. */
export function oaiyStateOf(input: { own: boolean; desktop: string | null; discovered?: string; withKey: boolean; typed: string }): OaiyState {
  if (input.own) return { kind: 'own' };
  const typed = input.typed.trim();
  const oaiy: OaiyLink = input.discovered ? { kind: 'found', origin: input.discovered, withKey: input.withKey } : typed ? { kind: 'typed', address: typed } : { kind: 'none' };
  return { kind: 'tab', desktop: input.desktop, oaiy };
}

/** The clause in Settings → Images, video and audio about how OAIY is found. */
export function oaiyFoundWords(state: OaiyState): string {
  if (state.kind === 'own') return 'OAIY is found on its own';
  const { oaiy, desktop } = state;
  switch (oaiy.kind) {
    case 'none':
      // Only an unpaired tab has nothing else it sends: a paired one says what it keeps in touch with (pairedTabWords), and this says only what is true of OAIY.
      return desktop
        ? 'OAIY is looked for only when you press Find OAIY, and your browser may ask you to allow it when you do'
        : 'OAIY is looked for only when you press Find OAIY: until you do, this page sends nothing to your computer or your network to look for it, and your browser may ask you to allow it when you do';
    case 'found':
      return `OAIY was found at ${oaiy.origin}: this page asks it there each time it opens${oaiy.withKey ? ' (sending the key below)' : ''}, and now and then to follow the model chosen in OAIY's Engines. Forget OAIY, or clearing the address, ends that`;
    case 'typed':
      return `You typed the address of a media service (${oaiy.address}): the agent uses it when it makes media, and this page sends nothing to it before then; OAIY is looked for only when you press Find OAIY, which asks that address, and your browser may ask you to allow it`;
  }
}

/** What a paired tab keeps doing whatever it holds of OAIY, said next to the words above; null for a tab that is not paired and for the window. */
export function pairedTabWords(state: OaiyState): string | null {
  if (state.kind !== 'tab' || !state.desktop) return null;
  return `This page is paired with OAIY Desktop at ${state.desktop} and keeps in touch with it while it is open (its calls, texts, settings and flows), whether or not OAIY is found. Forget the pairing, in the phone dialog, ends that.`;
}

/** The tooltip of the Find OAIY button. */
export function findOaiyTitle(state: OaiyState): string {
  const base = 'Ask OAIY for its details (/v1/discovery) and fill everything in';
  if (state.kind === 'own') return base;
  const { oaiy, desktop } = state;
  const about =
    oaiy.kind === 'found'
      ? `${base}. This page also asks OAIY at ${oaiy.origin} each time it opens.`
      : oaiy.kind === 'typed'
        ? `${base}. This asks the address you typed (${oaiy.address}), and your browser may ask you to allow it.`
        : `${base}. This page looks for OAIY only when you press this, and your browser may ask you to allow it.`;
  return desktop ? `${about} It also keeps in touch with its paired desktop at ${desktop}, whether or not you press this.` : about;
}

/** Where this page stands with OAIY Desktop, for the pairing dialog. */
export type DesktopState =
  /** OAIY's own window: the desktop is given. */
  | { kind: 'given' }
  /** A tab that is not paired: it asks about the desktop only while this dialog is open. */
  | { kind: 'unpaired'; origin: string }
  /** A tab that is paired: it keeps in touch with the desktop while it is open. */
  | { kind: 'paired'; origin: string };

/** What the pairing dialog says as it opens, before the desktop has answered. */
export function desktopLookingWords(state: DesktopState, phone: boolean): string {
  switch (state.kind) {
    case 'given':
      return phone ? "This is OAIY's own window: texts and calls to the phone come here." : "This is OAIY's own window: it is OAIY Desktop's.";
    case 'unpaired':
      return `Looking for OAIY Desktop at ${state.origin} now, because you opened this dialog: this page asks nothing of your computer for OAIY Desktop until then. Your browser may ask whether it may connect to your network: that is this request.`;
    case 'paired':
      return `This page is paired with OAIY Desktop at ${state.origin} and keeps in touch with it while it is open (its calls, texts, settings and flows). Checking it now.`;
  }
}
