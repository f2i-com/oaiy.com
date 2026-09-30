/**
 * What the settings and the pairing dialog say about when this page reaches out to the person's computer, for each state it can be in.
 * Each sentence is true of that state and no other, and says what stops it: a page that says it "never" reaches out and does, or
 * that says it reaches out "only from here" while it has been asking in the background since it opened, tells the person less than
 * they need. The states are OAIY's own window (looks as it opens), a tab that has no link, and a tab that has one.
 *
 * Pure, so a test holds each state's words, and one look at the page shows the same words.
 */

/** Where this page stands with OAIY (the engine's gateway and media service). */
export type OaiyState =
  /** OAIY's own window or the desktop shell: it looks for OAIY as it opens. */
  | { kind: 'own' }
  /** A tab that has found no OAIY, or has forgotten the one it found: it looks only when Find OAIY is pressed. */
  | { kind: 'never' }
  /** A tab that found an OAIY: it asks it at that address each time it opens. */
  | { kind: 'saved'; origin: string; withKey: boolean };

/** The sentence in Settings → Images, video and audio about how OAIY is found. */
export function oaiyFoundWords(state: OaiyState): string {
  switch (state.kind) {
    case 'own':
      return 'OAIY is found on its own';
    case 'never':
      return 'OAIY is looked for only when you press Find OAIY: until you do, this page sends nothing to your computer or your network, and your browser may ask you to allow it when you do';
    case 'saved':
      return `OAIY was found at ${state.origin}: this page asks it there each time it opens${state.withKey ? ' (sending the key below)' : ''}, and now and then to follow the model chosen in OAIY's Engines. Forget OAIY, or clearing the address, ends that`;
  }
}

/** The tooltip of the Find OAIY button. */
export function findOaiyTitle(state: OaiyState): string {
  const base = 'Ask OAIY for its details (/v1/discovery) and fill everything in';
  switch (state.kind) {
    case 'own':
      return base;
    case 'never':
      return `${base}. This page reaches out to your computer only when you press this, and your browser may ask you to allow it.`;
    case 'saved':
      return `${base}. This page also asks OAIY at ${state.origin} each time it opens.`;
  }
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
