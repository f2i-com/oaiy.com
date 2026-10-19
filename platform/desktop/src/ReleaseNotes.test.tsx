// A release's notes, read as the few Markdown forms GitHub's release notes use, and shown as text: nothing in them is
// ever HTML, and only an https address is a link (opened in the browser, never in OAIY's window).
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const h = vi.hoisted(() => ({ openExternal: vi.fn() }));
vi.mock('./api', () => ({ openExternal: (...a: unknown[]) => h.openExternal(...a) }));

import ReleaseNotes, { parseBlocks, safeHref } from './ReleaseNotes';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  h.openExternal.mockReset();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});
const render = (text: string) => act(() => root.render(<ReleaseNotes text={text} />));

const NOTES = [
  '## OAIY 0.1.9',
  '',
  '### Calls',
  '- The greeting plays sooner ([#412](https://github.com/f2i-com/oaiy.com/pull/412)).',
  '- **Front desk**: redials fixed,',
  '  and call waiting too.',
  '',
  'Plain words, with `code`.',
  '',
  '**Full Changelog**: https://github.com/f2i-com/oaiy.com/compare/v0.1.8...v0.1.9',
].join('\n');

describe('ReleaseNotes', () => {
  it('reads headings, lists (an item’s next line its own) and paragraphs', () => {
    expect(parseBlocks(NOTES)).toEqual([
      { kind: 'heading', level: 2, text: 'OAIY 0.1.9' },
      { kind: 'heading', level: 3, text: 'Calls' },
      { kind: 'list', ordered: false, items: ['The greeting plays sooner ([#412](https://github.com/f2i-com/oaiy.com/pull/412)).', '**Front desk**: redials fixed, and call waiting too.'] },
      { kind: 'para', text: 'Plain words, with `code`.' },
      { kind: 'para', text: '**Full Changelog**: https://github.com/f2i-com/oaiy.com/compare/v0.1.8...v0.1.9' },
    ]);
  });

  it('shows them as text, with bold, code and links that open in the browser', () => {
    render(NOTES);
    expect(host.querySelectorAll('.update-notes-heading')).toHaveLength(2);
    expect(host.querySelectorAll('li')).toHaveLength(2);
    expect(host.querySelector('strong')?.textContent).toBe('Front desk');
    expect(host.querySelector('code')?.textContent).toBe('code');
    expect(host.textContent).not.toContain('##');
    expect(host.textContent).not.toContain('**');
    const links = Array.from(host.querySelectorAll('a'));
    expect(links.map((a) => [a.textContent, a.getAttribute('href')])).toEqual([
      ['#412', 'https://github.com/f2i-com/oaiy.com/pull/412'],
      ['https://github.com/f2i-com/oaiy.com/compare/v0.1.8...v0.1.9', 'https://github.com/f2i-com/oaiy.com/compare/v0.1.8...v0.1.9'],
    ]);
    // (the sentence's own bracket and full stop after the first link stay text)
    expect(links[0].parentElement?.textContent).toContain('(#412).');
    act(() => links[0].click());
    expect(h.openExternal).toHaveBeenCalledWith('https://github.com/f2i-com/oaiy.com/pull/412');
  });

  it('never reads HTML, and makes no link of an address that is not https', () => {
    render('<img src=x onerror="alert(1)"> <script>alert(2)</script>\n\n[click](javascript:alert(3)) [plain](http://example.test/) [file](file:///etc/passwd)');
    expect(host.querySelector('img, script')).toBeNull();
    expect(host.querySelectorAll('a')).toHaveLength(0);
    expect(host.textContent).toContain('<img src=x onerror="alert(1)">');
    expect(host.textContent).toContain('click plain file');
    expect(safeHref('https://example.test/a')).toBe('https://example.test/a');
    expect(safeHref('javascript:alert(1)')).toBeNull();
    expect(safeHref('not a url')).toBeNull();
  });
});
