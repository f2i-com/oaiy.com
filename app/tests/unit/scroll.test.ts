import { describe, expect, it } from 'vitest';
import { Follow, JUMP_OFFER, STICK_SLACK, anchorIndex, anchoredScrollTop, fromBottom, nearBottom } from '../../src/ui/chat/scroll';

/** A log 600px tall holding `height` of content, scrolled `top` down. */
const at = (top: number, height = 3000) => ({ scrollTop: top, scrollHeight: height, clientHeight: 600 });
const bottom = (height = 3000) => at(height - 600, height);

describe('the chat keeps to the bottom while the reader is there', () => {
  it('counts the bottom, and within 80px of it, as at the bottom', () => {
    expect(fromBottom(bottom())).toBe(0);
    expect(nearBottom(bottom())).toBe(true);
    expect(nearBottom(at(3000 - 600 - STICK_SLACK))).toBe(true);
    expect(nearBottom(at(3000 - 600 - STICK_SLACK - 1))).toBe(false);
    // A log shorter than its window is at its bottom.
    expect(nearBottom({ scrollTop: 0, scrollHeight: 300, clientHeight: 600 })).toBe(true);
  });

  it('follows at first; stops when the reader scrolls up; starts again when they come back down to the bottom', () => {
    const follow = new Follow();
    expect(follow.stick).toBe(true);
    follow.scrolled(bottom());
    follow.scrolled(at(2000));
    expect(follow.stick).toBe(false);
    // Down a little, still well above the bottom: still away.
    follow.scrolled(at(2100));
    expect(follow.stick).toBe(false);
    follow.scrolled(at(2350));
    expect(follow.stick).toBe(true);
  });

  it('keeps following when content grows faster than the log keeps up (the bottom moves away, but nobody scrolled up)', () => {
    const follow = new Follow();
    follow.scrolled(bottom(3000));
    // A reply streamed 500px in the frame before the log was pinned again: the scroll event sees the bottom far below.
    follow.scrolled(at(2400, 3500));
    expect(follow.stick).toBe(true);
    follow.scrolled(bottom(3500));
    expect(follow.stick).toBe(true);
  });

  it('counts the messages that came while the reader was up, and offers "Latest"', () => {
    const follow = new Follow();
    follow.scrolled(bottom());
    follow.scrolled(at(1000));
    follow.arrived(true);
    follow.arrived(false);
    follow.arrived(true);
    expect(follow.unseen).toBe(2);
    expect(follow.offerJump(at(1000))).toBe(true);
    // Back at the bottom: nothing is unseen, and there is nothing to offer.
    follow.scrolled(bottom());
    expect(follow.unseen).toBe(0);
    expect(follow.offerJump(bottom())).toBe(false);
  });

  it('counts nothing while following', () => {
    const follow = new Follow();
    follow.arrived(true);
    expect(follow.unseen).toBe(0);
  });

  it('offers "Latest" a little way up only when something new came, and further up always', () => {
    const follow = new Follow();
    follow.scrolled(bottom());
    const near = at(3000 - 600 - STICK_SLACK - 20);
    follow.scrolled(near);
    expect(follow.stick).toBe(false);
    expect(follow.offerJump(near)).toBe(false);
    follow.arrived(true);
    expect(follow.offerJump(near)).toBe(true);
    const far = at(3000 - 600 - JUMP_OFFER - 1);
    follow.scrolled(far);
    expect(follow.offerJump(far)).toBe(true);
  });

  it('a jump to the latest follows at once, through a smooth scroll on its way down; scrolling up stops it again', () => {
    const follow = new Follow();
    follow.scrolled(bottom());
    follow.scrolled(at(500));
    follow.arrived(true);
    follow.jump();
    expect(follow.stick && follow.unseen === 0).toBe(true);
    // On its way down.
    follow.scrolled(at(900));
    follow.scrolled(at(1600));
    expect(follow.stick).toBe(true);
    // The reader scrolls up during it: they are left there.
    follow.scrolled(at(1200));
    expect(follow.stick).toBe(false);
  });

  it('an instant jump goes from where it landed, not from where the log was before (another conversation, say)', () => {
    const follow = new Follow();
    follow.scrolled(at(5000, 9000));
    // A shorter conversation, drawn and put at its end: 1800 down.
    follow.jump(1800);
    // More came before the scroll was read: the bottom moved away, but nobody went up.
    follow.scrolled(at(1800, 2800));
    expect(follow.stick).toBe(true);
    // The reader's first move up leaves the bottom, with no scroll read in between.
    const again = new Follow();
    again.jump(1800);
    again.scrolled(at(1200, 2400));
    expect(again.stick).toBe(false);
  });
});

describe('older turns drawn in above what the reader sees', () => {
  it('holds the first entry that shows, even in part', () => {
    const spans = [
      { top: -900, bottom: -300 },
      { top: -300, bottom: -10 },
      { top: -10, bottom: 200 },
      { top: 200, bottom: 700 },
    ];
    expect(anchorIndex(spans)).toBe(2);
    expect(anchorIndex([{ top: 0, bottom: 50 }])).toBe(0);
    expect(anchorIndex([{ top: -50, bottom: 0 }])).toBe(-1);
    expect(anchorIndex([])).toBe(-1);
  });

  it('scrolls down by as much as went in above, so the anchor stays where it was', () => {
    // The anchor was 40px down the window; 1200px went in above it, moving it to 1240px.
    expect(anchoredScrollTop(20, 40, 1240)).toBe(1220);
    // Nothing moved.
    expect(anchoredScrollTop(300, 12, 12)).toBe(300);
    // Never above the top.
    expect(anchoredScrollTop(0, 100, 20)).toBe(0);
  });
});
