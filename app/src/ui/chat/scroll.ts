/**
 * How the chat's log follows what is new. It reads oldest first, the newest
 * just above the message box, and it keeps to the bottom while the reader is
 * there (or near it): new messages, a reply as it streams, a row opened. A
 * reader who has scrolled up is left where they are, and told how much is new
 * below. Older turns drawn in above what they read leave it where it was.
 */

/** Where a scroller is: how far down, how tall its content, how tall its window. */
export interface Metrics {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
}

/** How near the bottom still counts as at it (px). */
export const STICK_SLACK = 80;
/** How far up the reader must be before "Latest" is offered with nothing new below (px). */
export const JUMP_OFFER = 240;

/** How far the bottom is below what shows (px). */
export function fromBottom(m: Metrics): number {
  return Math.max(0, m.scrollHeight - m.scrollTop - m.clientHeight);
}

/** Whether the reader is at the bottom, or within `slack` of it. */
export function nearBottom(m: Metrics, slack = STICK_SLACK): boolean {
  return fromBottom(m) <= slack;
}

/**
 * Whether the log follows new content, and how many new messages came while it
 * did not. The reader leaves the bottom only by scrolling up: content growing
 * under the log (a reply streaming, faster than the log keeps up with it within
 * a frame) moves the bottom away without anyone scrolling, and must not stop it
 * following. They come back by reaching the bottom again, or with `jump()`.
 */
export class Follow {
  /** Keep to the bottom as content comes. */
  stick = true;
  /** Messages that came while the reader was away from the bottom. */
  unseen = 0;
  /** Where the log was at the last scroll (-Infinity: nothing to go up from, as after a jump). */
  private lastTop = Number.NEGATIVE_INFINITY;

  constructor(private readonly slack = STICK_SLACK) {}

  /** The log scrolled (the reader, or the log itself). */
  scrolled(m: Metrics): void {
    if (nearBottom(m, this.slack)) this.stick = true;
    else if (m.scrollTop < this.lastTop - 1) this.stick = false;
    this.lastTop = m.scrollTop;
    if (this.stick) this.unseen = 0;
  }

  /** Something came: a message counts when the reader is away. */
  arrived(message: boolean): void {
    if (!this.stick && message) this.unseen++;
  }

  /**
   * Back to the latest, following from there. `landed`: where an instant jump
   * put the log (the reader's next move up from there leaves it). A smooth one
   * gives none: the positions it passes on its way down are not going up.
   */
  jump(landed = Number.NEGATIVE_INFINITY): void {
    this.stick = true;
    this.unseen = 0;
    this.lastTop = landed;
  }

  /** Whether "Latest" is offered: away from the bottom, with something new below or a way to go. */
  offerJump(m: Metrics): boolean {
    return !this.stick && (this.unseen > 0 || fromBottom(m) > JUMP_OFFER);
  }
}

/** An element's top and bottom, from the top of what shows (px). */
export interface Span {
  top: number;
  bottom: number;
}

/** The element to hold in place while more goes in above it: the first one that shows (at least in part). -1 for none. */
export function anchorIndex(spans: Span[], viewTop = 0): number {
  return spans.findIndex((s) => s.bottom > viewTop);
}

/** Where to scroll so the anchor stays where it was: its move is added back. */
export function anchoredScrollTop(scrollTop: number, anchorTopBefore: number, anchorTopAfter: number): number {
  return Math.max(0, scrollTop + (anchorTopAfter - anchorTopBefore));
}
