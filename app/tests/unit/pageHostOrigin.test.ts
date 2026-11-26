import { describe, expect, it } from 'vitest';
import HOST_SOURCE from '../../public/webpage/page-host.js?raw';
import PREVIEW_SOURCE from '../../src/preview/preview.ts?raw';

/**
 * The web page preview's frame (public/webpage/page-host.js) run against a window of the given origin that a parent
 * frames: whether it went on to start, and what it told that parent. A window that is not a browser's stops it a few
 * lines further on, which is past what is asked here.
 */
function startIn(origin: string, framed = true) {
  const posted: unknown[] = [];
  const listened: string[] = [];
  let wentOn = false;
  const win: Record<string, unknown> = {
    origin,
    addEventListener: (type: string) => listened.push(type),
    // The first thing the frame takes from its window once it has decided to start.
    get setTimeout() {
      wentOn = true;
      return setTimeout;
    },
  };
  win.parent = framed ? { postMessage: (message: unknown) => posted.push(message) } : win;
  try {
    new Function('window', HOST_SOURCE)(win);
  } catch {
    // (no DOMParser, no document: this is not a browser)
  }
  return { wentOn, posted, listened };
}

describe("the web page preview's frame starts only in its sandbox", () => {
  it('starts with the opaque origin OAIY frames it for', () => {
    expect(startIn('null').wentOn).toBe(true);
  });

  it('does not start in the origin of whoever serves it: it would import what its parent posts into that origin', () => {
    for (const origin of ['http://127.0.0.1:17974', 'http://localhost:5173', 'https://oaiy.com', 'oaiy://localhost', '']) {
      const { wentOn, posted, listened } = startIn(origin);
      expect(wentOn, origin).toBe(false);
      // It neither listens for `webpage:init` nor tells the page that framed it that it is ready for one.
      expect(listened, origin).toEqual([]);
      expect(posted, origin).toEqual([]);
    }
  });

  it('does not start as a page of its own, as before', () => {
    expect(startIn('null', false).wentOn).toBe(false);
  });

  it('is framed by the Agent with no same-origin, so its origin there is opaque', () => {
    const framings = PREVIEW_SOURCE.split('\n').filter((line) => line.includes('this.newFrame('));
    expect(framings.length).toBe(3);
    for (const line of framings) expect(line).not.toContain('allow-same-origin');
    expect(PREVIEW_SOURCE).toContain("frame.setAttribute('sandbox', sandbox);");
  });
});
