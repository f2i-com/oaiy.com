// Layout rules that a page's test cannot see (jsdom lays nothing out), held by reading the stylesheets:
//
// - Services at a phone's width (the dashboard in a phone's browser, through oaiy-server): a service's buttons go
//   under its name and state, from the left. Side by side they took the row and lay over the words, which were left
//   50px, one word a line.
// - There is no bar at the foot of the window, and the pages that measure against the header and the foot (the
//   Calendar's and Contacts' side panels) take both heights from the workspace, so they cannot fall out of step.
//   (The pages in webviews, the Agent, Flows and Engines, need neither: the desktop lays each over the box the
//   dashboard measures, src/EmbeddedPage.tsx.)
// - A section with a page in a webview has a one-line header.
//
// What these look like is in the pictures a browser takes of the page; this holds the rules they rest on. Here,
// beside the other plain-JS checks, because it reads files (the TypeScript build has no declarations for Node's).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const src = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'src');
const css = (name) => fs.readFileSync(path.join(src, name), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');

/** Where the block that opens at `open` (a `{`) ends. */
function closing(sheet, open) {
  let depth = 0;
  for (let i = open; i < sheet.length; i++) {
    if (sheet[i] === '{') depth++;
    else if (sheet[i] === '}' && --depth === 0) return i;
  }
  return sheet.length;
}

/** The bodies of every `@media <query> { … }` block in `sheet` whose query is exactly `query`. */
function media(sheet, query) {
  const out = [];
  const opener = `@media ${query}`;
  for (let at = sheet.indexOf(opener); at >= 0; ) {
    const open = sheet.indexOf('{', at);
    const end = closing(sheet, open);
    if (sheet.slice(at + opener.length, open).trim() === '') out.push(sheet.slice(open + 1, end));
    at = sheet.indexOf(opener, end);
  }
  return out;
}

/** `sheet` without its `@media` blocks: the rules that hold at every width. */
function everyWidth(sheet) {
  let out = '';
  let at = 0;
  for (;;) {
    const next = sheet.indexOf('@media', at);
    if (next < 0) return out + sheet.slice(at);
    out += sheet.slice(at, next);
    at = closing(sheet, sheet.indexOf('{', next)) + 1;
  }
}

/** What `selector { … }` declares in `block` (every rule of that exact selector, joined). */
function rule(block, selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return [...block.matchAll(new RegExp(`(?:^|[{},])\\s*${escaped}\\s*\\{([^}]*)\\}`, 'g'))].map((m) => m[1]).join(';');
}

describe('layout rules the pages rest on', () => {
  const styles = css('styles.css');

  it('stacks a service’s buttons under its words at a phone’s width', () => {
    const phone = media(styles, '(max-width: 560px)').join('\n');
    expect(rule(phone, '.service-row')).toMatch(/flex-direction:\s*column/);
    expect(rule(phone, '.service-actions')).toMatch(/justify-content:\s*flex-start/);
    // (side by side everywhere wider, as before)
    expect(rule(everyWidth(styles), '.service-row')).toMatch(/display:\s*flex/);
    expect(rule(everyWidth(styles), '.service-row')).not.toMatch(/flex-direction/);
  });

  it('has a header and a page, no bar at the foot, and one place that says how tall each is', () => {
    const everywhere = everyWidth(styles);
    // The header's two heights, on the root (the notices' place reads the page's too): its first row, and the row
    // with the line on what the page is for under it.
    expect(everywhere).toMatch(/:root\s*\{\s*--topbar-row:\s*48px;\s*--topbar-h:\s*78px;\s*\}/);
    expect(rule(everywhere, ':root:has(.workspace.is-compact)')).toMatch(/--topbar-h:\s*var\(--topbar-row\)/);
    const workspace = rule(everywhere, '.workspace');
    expect(workspace).toMatch(/--dock-h:\s*0px/);
    expect(workspace).toMatch(/grid-template-rows:\s*var\(--topbar-h\)\s+minmax\(0,\s*1fr\)\s*(;|$)/);
    expect(styles).not.toMatch(/endpoint-dock|\.dock-/);
    // The pages that measure against them read the same two names.
    expect(css('calendar.css')).toMatch(/max-height:\s*calc\(100dvh - var\(--topbar-h, 78px\) - var\(--dock-h, 0px\) - 24px\)/);
    expect(css('contacts.css')).toMatch(/calc\(100dvh - var\(--topbar-h, 78px\) - var\(--dock-h, 0px\) - var\(--page-pad-t\) - 66px\)/);
    // (no stylesheet keeps a header's or a foot's height of its own)
    for (const sheet of fs.readdirSync(src).filter((f) => f.endsWith('.css'))) expect(css(sheet), sheet).not.toMatch(/100d?vh - \d+px/);
  });

  it('has one header: the same first row on every page, and the line on what the page is for as a row of its own', () => {
    const everywhere = everyWidth(styles);
    const px = (decl, name) => Number(new RegExp(`(?:^|;|\\s)${name}:\\s*([\\d.]+)px`).exec(decl)?.[1]);
    // The first row: one height, and the name at one size, whatever the page (no rule gives a page another).
    expect(rule(everywhere, '.top-row')).toMatch(/height:\s*calc\(var\(--topbar-row\) - 1px\)/);
    const name = rule(everywhere, '.top-heading h1');
    expect([px(name, 'font-size'), px(name, 'line-height')]).toEqual([16, 22]);
    expect(styles).not.toMatch(/\.is-compact[^{]*h1|\.top-kicker|\.top-title/);
    // The second row: a line height in pixels, room over it (what is left of the first row under the name) and
    // under it, and the two rows are the page's height.
    const copy = rule(everywhere, '.top-copy');
    expect([px(copy, 'font-size'), px(copy, 'line-height'), px(copy, 'height')]).toEqual([13, 18, 30]);
    expect(copy).toMatch(/padding:\s*0 12px 12px var\(--page-pad-x\)/);
    expect(48 + px(copy, 'height')).toBe(78);
    expect((48 - px(name, 'line-height')) / 2).toBeGreaterThanOrEqual(12);
    // A short window and a phone have the first row alone, as a section with a page in a webview has.
    const small = media(styles, '(max-width: 700px), (max-height: 640px)').join('\n');
    expect(rule(small, ':root')).toMatch(/--topbar-h:\s*var\(--topbar-row\)/);
    expect(rule(small, '.top-copy')).toMatch(/display:\s*none/);
    expect(rule(everywhere, '.topbar .seg-tabs')).toMatch(/flex-wrap:\s*nowrap/);
    // The page draws no group word over the name, and the one row holds the name, the tabs and the buttons.
    const app = fs.readFileSync(path.join(src, 'App.tsx'), 'utf8');
    expect(app).not.toMatch(/className="top-kicker"/);
    expect(app).toMatch(/<div className="top-row">\s*<div className="top-heading">/);
    // The same gap wherever a small label, a heading and a line stack: a setup step, an empty state, the sidebar's
    // name, who is calling.
    expect(px(rule(everywhere, ':root'), '--stack-gap')).toBeGreaterThanOrEqual(6);
    expect(rule(css('setup.css'), '.setup-step-head h2')).toMatch(/margin:\s*var\(--stack-gap\) 0 0/);
    expect(rule(css('setup.css'), '.setup-step-head p')).toMatch(/margin:\s*var\(--stack-gap\) 0 0/);
    expect(rule(everywhere, '.empty-state')).toMatch(/gap:\s*var\(--stack-gap\)/);
    expect(rule(css('sidebar.css'), '.brand > span')).toMatch(/margin-top:\s*var\(--stack-gap\)/);
    expect(rule(css('messages.css'), '.ring-dialog .ring-who')).toMatch(/gap:\s*var\(--stack-gap\)/);
  });

  it('keeps a drop-down’s one arrow: no rule sets the whole `background` of a select after its arrow is placed', () => {
    // A select's arrow is a background image with its own place, size and no repeat. The `background` shorthand
    // resets all three, and a theme's rule that then puts back only the image tiles it across the field (the light
    // theme's drop-downs were covered by a zigzag of arrows). A colour is set with `background-color`.
    const rules = [...styles.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ selector: m[1].trim(), body: m[2] }));
    const onSelect = (selector) => selector.split(',').some((s) => /(^|[\s>+~])select(?![\w-])/.test(s.trim()));
    const arrow = rules.findIndex((r) => onSelect(r.selector) && /background-repeat:\s*no-repeat/.test(r.body));
    expect(arrow).toBeGreaterThan(0);
    const shorthand = /(^|;)\s*background\s*:/;
    const after = rules.slice(arrow).filter((r) => onSelect(r.selector) && shorthand.test(r.body)).map((r) => r.selector);
    expect(after).toEqual([]);
    // And in no theme's rule, wherever it is written (a theme's selector outweighs the arrow's own).
    const themed = rules.filter((r) => /\[data-theme/.test(r.selector) && onSelect(r.selector) && shorthand.test(r.body)).map((r) => r.selector);
    expect(themed).toEqual([]);
    // (the check would catch the rule that did it)
    expect(shorthand.test(' background: var(--surface-raised); ')).toBe(true);
    expect(shorthand.test(' background-color: var(--surface-raised); ')).toBe(false);
    // The other stylesheets set no background on a bare select either.
    for (const sheet of fs.readdirSync(src).filter((f) => f.endsWith('.css') && f !== 'styles.css')) {
      const bad = [...css(sheet).matchAll(/([^{}]+)\{([^{}]*)\}/g)].filter((m) => onSelect(m[1]) && shorthand.test(m[2])).map((m) => m[1].trim());
      expect(bad, sheet).toEqual([]);
    }
  });

  it('moves nothing for a notice: a stack under the header, or one line in the header’s row over a page in a webview', () => {
    const everywhere = everyWidth(styles);
    const stack = rule(everywhere, '.toast-stack');
    expect(stack).toMatch(/position:\s*fixed/);
    expect(stack).toMatch(/top:\s*calc\(var\(--topbar-h\) \+ 10px\)/);
    const inHeader = rule(everywhere, '.toast-stack.is-in-header');
    expect(inHeader).toMatch(/position:\s*static/);
    expect(rule(everywhere, '.toast-stack.is-in-header > .toast')).toMatch(/height:\s*32px/);
    expect(rule(everywhere, '.top-notices')).toMatch(/flex:\s*1 1 0/);
    // No room is made for a notice any more: the box a webview follows does not change when one shows.
    const app = fs.readFileSync(path.join(src, 'App.tsx'), 'utf8');
    expect(app).not.toMatch(/toast-room|roomForToasts/);
    expect(styles).not.toMatch(/toast-room/);
    expect(app).toMatch(/\{embedded && <div className="top-notices" ref=\{toastSlot\} \/>\}/);
  });

  it('keeps the window’s smallest size a desktop’s: no phone rule starts at its width, and a finger’s sizes are for a finger', () => {
    // The window cannot be narrower than 760px (tauri.conf.json). A rule for "760px and under" is a phone's rule
    // that the smallest window would get too: its buttons grew to 44px and its rows broke in two.
    const conf = JSON.parse(fs.readFileSync(path.join(src, '..', 'src-tauri', 'tauri.conf.json'), 'utf8'));
    const smallest = conf.app.windows[0].minWidth;
    expect(smallest).toBe(760);
    for (const sheet of fs.readdirSync(src).filter((f) => f.endsWith('.css'))) {
      for (const [, w] of css(sheet).matchAll(/@media[^{]*\(max-width:\s*(\d+)px\)/g)) {
        expect(Number(w) < 720 || Number(w) >= 900, `${sheet}: a rule for ${w}px and under`).toBe(true);
      }
    }
    const touch = media(css('workspace.css'), '(max-width: 700px), (pointer: coarse)').join('\n');
    expect(rule(touch, '.btn, .btn-tiny, .icon-button')).toMatch(/min-height:\s*44px/);
    expect(rule(everyWidth(css('workspace.css')), '.btn, .icon-button')).toMatch(/min-height:\s*34px/);
  });

  it('has one rule for how wide a page runs: a form stops at a form’s width, a list uses the window', () => {
    const root = rule(everyWidth(styles), ':root');
    expect(root).toMatch(/--w-form:\s*1000px/);
    expect(root).toMatch(/--w-wide:\s*1400px/);
    // (at the window's default size the two are the same width: a form's 1000px is more than the 988px the page
    // has at 1280 wide, so Settings' cards and Services' line up there)
    const conf = JSON.parse(fs.readFileSync(path.join(src, '..', 'src-tauri', 'tauri.conf.json'), 'utf8'));
    expect(conf.app.windows[0].width - 222 - 2 * 30 - 10).toBeLessThanOrEqual(1000);
    const workspace = everyWidth(css('workspace.css'));
    expect(rule(workspace, '.panel')).toMatch(/max-width:\s*var\(--w-wide\)/);
    expect(rule(workspace, '.panel.is-form')).toMatch(/max-width:\s*var\(--w-form\)/);
    expect(rule(workspace, '.form-narrow')).toMatch(/max-width:\s*var\(--w-form\)/);
    // No page keeps a width of its own (the Calendar's week takes all the window gives: `none`).
    for (const sheet of ['calendar.css', 'contacts.css', 'messages.css', 'setup.css', 'sidebar.css']) {
      expect(css(sheet), sheet).not.toMatch(/\.panel\.[a-z-]+\s*\{[^}]*max-width:\s*\d+px/);
    }
    // The pages that are forms say so.
    for (const page of ['SettingsPanel.tsx', 'AgentSettingsPanel.tsx', 'TransfersPanel.tsx', 'NetworkPanel.tsx', 'ConnectionsPanel.tsx', 'AiProvidersPanel.tsx', 'PythonPanel.tsx']) {
      expect(fs.readFileSync(path.join(src, page), 'utf8'), page).toMatch(/className="panel is-form/);
    }
  });

  it('folds a sub-menu away in the narrow sidebar, so the sections under it are never pushed out of a short window', () => {
    const narrow = media(css('sidebar.css'), '(max-width: 1240px)').join('\n');
    expect(narrow).toMatch(/\.sidebar nav \.nav-sub\.has-active \.nav-children,\s*\.sidebar nav \.nav-children\s*\{\s*display:\s*none;\s*\}/);
    expect(narrow).not.toMatch(/\.nav-children\s*\{[^}]*display:\s*flex/);
    // (its pages are the header's tabs there: App.tsx asks the same width)
    expect(fs.readFileSync(path.join(src, 'App.tsx'), 'utf8')).toMatch(/NARROW_SIDEBAR = '\(max-width: 1240px\)'/);
  });

  it('keeps a banner’s lone close mark beside its words, not on a line of its own', () => {
    expect(rule(styles, '.banner-dismissable:has(> .banner-dismiss)')).toMatch(/flex-wrap:\s*nowrap/);
  });
});
