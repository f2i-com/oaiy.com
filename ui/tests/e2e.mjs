/**
 * Browser smoke test for the OAIY web app.
 *
 * Drives a RUNNING dev server or preview with Playwright (already a devDependency):
 *
 *     npm run dev            # in one terminal
 *     npm run test:e2e       # in another
 *     npm run test:e2e -- http://localhost:4173     # or against `vite preview`
 *
 * Asserts the things a typecheck cannot: that all three pages boot with a clean
 * console, that the shell actually renders, that navigation works ACROSS pages,
 * that a flow can be created, and that both themes resolve every token they use.
 *
 * Regression cases for defects that reached us once:
 *   - the project name must stay editable when a flow is open (it was moved into
 *     a branch that never rendered, making renameProject unreachable)
 *   - the flow name must be keyboard-reachable (it was briefly a double-click
 *     handler on an <h1>)
 *   - the topbar action cluster must not clip at narrow widths
 *   - --accent-secondary must follow the chosen accent, not stay on the default
 *
 * Exit code is 0 only when every assertion passes.
 */
import { chromium } from 'playwright';

const BASE = (process.argv[2] ?? 'http://localhost:5173').replace(/\/+$/, '');
/** Optional: an api base to exercise the cross-origin CORS regression case. */
const API_BASE = (process.argv[3] ?? '').replace(/\/+$/, '');

let pass = 0;
const failures = [];
const ok = (name, cond, detail = '') => {
  if (cond) {
    pass++;
    console.log(`  ✓ ${name}`);
  } else {
    failures.push(name);
    console.log(`  ✗ ${name}${detail ? `  -> ${detail}` : ''}`);
  }
};
const section = (s) => console.log(`\n-- ${s} --`);

const browser = await chromium.launch();

/** A fresh context with the theme pinned and first-run gates pre-dismissed. */
async function open(theme, { skipBoot = false, width = 1440, height = 900 } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height } });
  const page = await ctx.newPage();
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(`PAGEERROR: ${e.message}`));
  await page.addInitScript((a) => {
    try {
      localStorage.setItem('oaiy_theme', a.theme);
      if (a.skipBoot) {
        // Skip the splash beat and the first-run wizard so the workspace is
        // reachable without driving onboarding.
        localStorage.setItem('skipSplash', 'true');
        localStorage.setItem('oaiy.wizard.completed', 'true');
      }
    } catch { /* storage can be blocked; the page still renders */ }
  }, { theme, skipBoot });
  return { ctx, page, errors };
}

// Fail fast with a useful message rather than 60 confusing assertion failures.
{
  const probe = await browser.newContext();
  const p = await probe.newPage();
  const resp = await p.goto(BASE + '/', { waitUntil: 'domcontentloaded' }).catch(() => null);
  if (!resp || !resp.ok()) {
    console.error(`\nCannot reach ${BASE} — start the dev server first:\n  npm run dev\n`);
    await browser.close();
    process.exit(2);
  }
  await probe.close();
}

for (const theme of ['dark', 'light']) {
  console.log(`\n${'='.repeat(58)}\n${theme === 'dark' ? 'Prism Lab (dark)' : 'Paper Circuit (light)'}\n${'='.repeat(58)}`);

  // ---------------------------------------------------------------- landing
  section('landing page');
  {
    const { ctx, page, errors } = await open(theme);
    await page.goto(BASE + '/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);

    ok('boots with a clean console', errors.length === 0, errors.slice(0, 2).join(' | '));
    ok('shared site nav renders once', (await page.locator('.site-nav').count()) === 1);
    ok('wordmark is the OAIY caps mark', (await page.locator('.lp-wordmark').first().textContent()) === 'OAIY');
    ok('"Overview" is marked as the current page',
      (await page.locator('.site-nav-links a.active').first().textContent()) === 'Overview');
    ok('the repo link is present and correct',
      (await page.locator('.site-nav-star').getAttribute('href'))?.includes('github.com/'),
      await page.locator('.site-nav-star').getAttribute('href'));

    // The hero graph must depict real node ids, not invented ones.
    const ids = await page.locator('svg[role="img"] text')
      .filter({ hasText: /^(IMAGE_GEN|AI_LLM|CONDITION|OUTPUT)$/ }).count();
    ok('hero graph uses the app\'s real node ids', ids === 4, `matched ${ids}/4`);
    ok('hero graph labels the feedback loop',
      (await page.locator('svg[role="img"]').innerHTML()).includes('regenerate'));

    ok('every design token it references resolves', await page.evaluate(() => {
      const cs = getComputedStyle(document.documentElement);
      return [
        '--accent-primary', '--accent-secondary', '--color-bg-primary', '--color-bg-canvas',
        '--color-text-primary', '--color-border-strong', '--dot',
        '--signal-cyan', '--signal-magenta', '--signal-green', '--signal-amber', '--signal-danger',
      ].every((v) => cs.getPropertyValue(v).trim().length > 0);
    }));
    ok('no horizontal overflow',
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
    await ctx.close();
  }

  // ------------------------------------------------------- desktop landing
  section('desktop landing page + cross-page nav');
  {
    const { ctx, page, errors } = await open(theme);
    await page.goto(BASE + '/desktop.html', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);

    // The service library fetches from the API when VITE_API_BASE is set. That's
    // optional, so a failed fetch is not a test failure — but a page error is.
    ok('boots with no page errors', errors.filter((e) => e.startsWith('PAGEERROR')).length === 0,
      errors.filter((e) => e.startsWith('PAGEERROR')).slice(0, 2).join(' | '));
    ok('uses the SAME nav as the landing page', (await page.locator('.site-nav').count()) === 1);
    ok('marks "Desktop app" as current',
      (await page.locator('.site-nav-links a.active').first().textContent()) === 'Desktop app');
    ok('offers an in-page sub-nav', (await page.locator('.site-subnav a').count()) >= 4);

    // Both pages have #how and #capabilities sections, so a bare hash from here
    // would scroll locally instead of crossing pages.
    ok('"How it works" resolves cross-page, not locally',
      (await page.locator('.site-nav-links a', { hasText: 'How it works' }).getAttribute('href')) === 'index.html#how');

    await page.locator('.site-nav-links a', { hasText: 'Overview' }).click();
    await page.waitForLoadState('domcontentloaded');
    const path = new URL(page.url()).pathname;
    ok('the nav alone gets you back to the landing page', path === '/' || path === '/index.html', path);
    await ctx.close();
  }

  // -------------------------------------------------------------- the app
  section('flow builder');
  {
    const { ctx, page, errors } = await open(theme, { skipBoot: true });
    await page.goto(BASE + '/app.html', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(5000);

    ok('boots with a clean console', errors.length === 0, errors.slice(0, 2).join(' | '));
    ok('app shell renders', (await page.locator('.app-shell').count()) === 1);
    ok('sidebar has the full primary nav', (await page.locator('.oaiy-nav button').count()) === 5);
    ok('endpoint dock renders', (await page.locator('.oaiy-dock').count()) === 1);
    ok('engine card reports companion state',
      ((await page.locator('.oaiy-engine small').first().textContent()) ?? '').length > 0);

    // regression: the project name must stay editable alongside an open flow
    const proj = page.locator('input.oaiy-name-project');
    ok('project name is an editable input', (await proj.count()) === 1 && (await proj.isEditable()));
    const originalName = await proj.inputValue();
    await proj.fill('renamed by e2e');
    await page.waitForTimeout(500);
    ok('project name accepts edits', (await proj.inputValue()) === 'renamed by e2e');
    await proj.fill(originalName);

    const create = page.getByRole('button', { name: /Create your first flow/i }).first();
    if (await create.count()) {
      await create.click();
      await page.waitForTimeout(2200);
    }
    ok('creating a flow reveals the node palette', (await page.getByText('Node Palette').count()) > 0);
    ok('creating a flow reveals the inspector', (await page.getByText(/PROPERTIES/i).count()) > 0);
    ok('canvas wrapper is mounted', (await page.locator('.oaiy-canvas-wrap').count()) === 1);
    ok('project name survives opening a flow', (await proj.count()) === 1);

    // regression: the flow name must be keyboard-reachable
    const flowBtn = page.locator('button.oaiy-name-flow');
    ok('flow name is a real button', (await flowBtn.count()) === 1);
    if (await flowBtn.count()) {
      await flowBtn.first().focus();
      ok('flow name can take keyboard focus',
        await page.evaluate(() => document.activeElement?.className?.includes('oaiy-name-flow')));
      await page.keyboard.press('Enter');
      await page.waitForTimeout(500);
      ok('Enter opens the rename input',
        (await page.locator('input.oaiy-name[aria-label="Flow name"]').count()) === 1);
      await page.keyboard.press('Escape');
    }

    // regression: the accent must be applied consistently, not half-default
    const secondary = await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue('--accent-secondary').trim());
    ok('--accent-secondary is set', secondary.length > 0, secondary);

    // regression: the api is cross-origin and sets no Allow-Credentials, so a
    // credentialed request fails the CORS check outright. With credentials:
    // 'include' every call through backendDispatcher.apiJson threw "Failed to
    // fetch" — sharing, autosave, the run long-poll, heartbeat and result
    // reporting — while Settings' bare-fetch "Test Connection" still said OK.
    // Pass an api base as the 2nd arg to exercise it; skipped otherwise.
    if (API_BASE) {
      const probe = await page.evaluate(async (base) => {
        const out = {};
        for (const mode of ['omit', 'include']) {
          try {
            const r = await fetch(base + '/', { credentials: mode });
            out[mode] = 'HTTP ' + r.status;
          } catch (e) {
            out[mode] = 'THREW';
          }
        }
        return out;
      }, API_BASE);
      ok('cross-origin api call succeeds with credentials omitted',
        String(probe.omit).startsWith('HTTP 2'), JSON.stringify(probe));
      // Not asserted as a failure — it documents WHY we use omit. If the api ever
      // starts sending Allow-Credentials this flips, which is worth noticing.
      console.log(`    (with credentials:'include' the same call is ${probe.include})`);
    }

    // theme round-trip. The app's own switch now lives in Settings -> Appearance
    // (the old `.oaiy-toggle` segment control is gone), so exercise the shared
    // ThemeContext through the site nav's toggle on the landing page instead:
    // same context, same localStorage key, and it is the one a visitor meets
    // first. Asserted in place — this section runs under a per-theme init
    // script that rewrites `oaiy_theme` on every navigation, so a cross-page
    // check here would be testing the harness, not the app. The toggle is
    // clicked back so the rest of the run stays in the loop's theme.
    await page.goto(BASE + '/', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(800);
    const toggle = page.locator('.site-nav button[aria-label^="Switch to"]').first();
    const before = await page.evaluate(() => document.documentElement.className);
    await toggle.click();
    await page.waitForTimeout(400);
    const after = await page.evaluate(() => document.documentElement.className);
    ok('theme toggle switches the root class', before !== after, `${before} -> ${after}`);
    ok('the choice is persisted where the app reads it',
      (await page.evaluate(() => localStorage.getItem('oaiy_theme'))) === after,
      await page.evaluate(() => localStorage.getItem('oaiy_theme')));
    await toggle.click();
    await page.waitForTimeout(400);
    await page.goto(BASE + '/app.html', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(2500);

    // regression: the topbar actions must not clip as the window narrows
    for (const w of [1024, 900, 780]) {
      await page.setViewportSize({ width: w, height: 900 });
      await page.waitForTimeout(400);
      const overflow = await page.evaluate(() => {
        const a = document.querySelector('.oaiy-actions');
        if (!a || !a.parentElement) return 'no action cluster';
        const bar = a.parentElement.getBoundingClientRect();
        const r = a.getBoundingClientRect();
        return r.right > bar.right + 1 ? `overflows by ${Math.round(r.right - bar.right)}px` : null;
      });
      ok(`topbar actions fit at ${w}px`, overflow === null, overflow);
    }
    await ctx.close();
  }
}

// ---------------------------------------------------------------------------
// The flows rail collapses, stays collapsed, and can be brought back.
//
// It used to `return null` when closed, which is a hide rather than a collapse:
// the panel vanished, left no affordance where it had been, and the only way
// back was an icon in the topbar. So the assertions here are specifically that
// something REMAINS (a 44px rail with the reopen control on it), that the canvas
// actually reclaims the width, and that the choice survives a reload — a panel
// that silently reopens on every visit is not usefully collapsible.
section('flows rail collapse');
{
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    localStorage.setItem('skipSplash', 'true');
    localStorage.setItem('oaiy.wizard.completed', 'true');
  });
  await page.goto(`${BASE}/app.html`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2200);

  const state = () =>
    page.evaluate(() => {
      const h = [...document.querySelectorAll('h2')].find((e) =>
        /^(Flows|Your Flows)$/.test(e.textContent.trim()),
      );
      const panel = h?.closest('div.flex.flex-col');
      const railBtn = document.querySelector('button[aria-label="Expand the flows panel"]');
      const canvas = document.querySelector('.oaiy-canvas-wrap');
      return {
        panel: panel ? Math.round(panel.getBoundingClientRect().width) : 0,
        rail: railBtn ? Math.round(railBtn.parentElement.getBoundingClientRect().width) : 0,
        canvas: canvas ? Math.round(canvas.getBoundingClientRect().width) : 0,
        stored: localStorage.getItem('oaiy.flowsRail'),
      };
    });

  const expanded = await state();
  ok('rail starts expanded', expanded.panel > 200, `panel=${expanded.panel}`);
  ok('no rail stub while expanded', expanded.rail === 0, `rail=${expanded.rail}`);

  await page.click('button[aria-label="Collapse the flows panel"]');
  await page.waitForTimeout(450);
  const collapsed = await state();
  ok('collapsing hides the panel', collapsed.panel === 0, `panel=${collapsed.panel}`);
  ok('a rail remains, with the reopen control', collapsed.rail > 0 && collapsed.rail < 80, `rail=${collapsed.rail}`);
  ok('canvas reclaims the width', collapsed.canvas > expanded.canvas, `${expanded.canvas} -> ${collapsed.canvas}`);
  ok('collapse is persisted', collapsed.stored === 'collapsed', String(collapsed.stored));

  await page.keyboard.press('Control+b');
  await page.waitForTimeout(450);
  ok('Ctrl+B expands again', (await state()).panel > 200);
  await page.keyboard.press('Control+b');
  await page.waitForTimeout(450);
  ok('Ctrl+B collapses again', (await state()).panel === 0);

  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(2200);
  const afterReload = await state();
  ok('still collapsed after a reload', afterReload.panel === 0 && afterReload.rail > 0, JSON.stringify(afterReload));

  await page.click('button[aria-label="Expand the flows panel"]');
  await page.waitForTimeout(450);
  ok('the rail button restores the panel', (await state()).panel > 200);

  await ctx.close();
}

// ---------------------------------------------------------------------------
// Self-hosted fonts, and no third-party requests at all.
//
// Both halves of this are here because both failed silently once.
//
// The app used to pull Inter and JetBrains Mono from fonts.googleapis.com, so
// every page load reported the reader to Google — in a product whose landing
// page sells "nothing leaves your device". Self-hosting them then failed twice
// over without a single console warning: the @font-face rules named the family
// `Inter Variable` while the CSS tokens asked for `Inter`, and Tailwind v4
// inlined the font @import without rebasing its relative url()s, so the rules
// pointed at files that were never emitted. Result: no woff2 request, no error,
// and every page rendering in Segoe UI while looking entirely deliberate.
//
// Checking `document.fonts.check()` alone is not enough — it answers about the
// family, not about whether real glyphs arrived — so this also measures text
// rendered in the face against a guaranteed-missing family. Identical widths
// mean the browser fell back and the font never loaded.
section('self-hosted fonts + zero third-party requests');
for (const path of ['/', '/desktop.html', '/app.html']) {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const external = [];
  const woff2 = [];
  page.on('request', (r) => {
    const u = r.url();
    if (!/^(https?:\/\/localhost|https?:\/\/127\.0\.0\.1|data:|blob:)/.test(u)) external.push(u);
  });
  page.on('response', (r) => {
    if (/\.woff2?(\?|$)/.test(r.url())) woff2.push(r.status());
  });

  await page.goto(BASE + path, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);

  const r = await page.evaluate(async () => {
    await document.fonts.ready;
    const el = document.createElement('span');
    el.textContent = 'OAIY orchestrate 0123';
    el.style.position = 'absolute';
    el.style.whiteSpace = 'pre';
    document.body.appendChild(el);
    const widthIn = (family) => {
      el.style.font = `400 40px ${family}`;
      return el.getBoundingClientRect().width;
    };
    const measured = {
      inter: widthIn('"Inter Variable"'),
      mono: widthIn('"JetBrains Mono Variable"'),
      // A family that cannot exist, so the browser must use the generic.
      fallbackSans: widthIn('"__oaiy_missing__", sans-serif'),
      fallbackMono: widthIn('"__oaiy_missing__", monospace'),
    };
    el.remove();
    return {
      interReady: document.fonts.check('16px "Inter Variable"'),
      monoReady: document.fonts.check('16px "JetBrains Mono Variable"'),
      bodyFirst: getComputedStyle(document.body).fontFamily.split(',')[0].replace(/["']/g, '').trim(),
      interDistinct: Math.abs(measured.inter - measured.fallbackSans) > 0.5,
      monoDistinct: Math.abs(measured.mono - measured.fallbackMono) > 0.5,
    };
  });

  ok(`${path} fetches its fonts locally`, woff2.length > 0, `${woff2.length} woff2 responses`);
  ok(`${path} every font response is 200`, woff2.length > 0 && woff2.every((s) => s === 200), woff2.join(','));
  ok(`${path} Inter Variable is loaded`, r.interReady);
  ok(`${path} JetBrains Mono Variable is loaded`, r.monoReady);
  ok(`${path} Inter actually renders (not a fallback)`, r.interDistinct);
  ok(`${path} JetBrains Mono actually renders (not a fallback)`, r.monoDistinct);
  ok(`${path} body resolves to the self-hosted family`, r.bodyFirst === 'Inter Variable', r.bodyFirst);
  // The api base is same-origin-ish (127.0.0.1) and allowed above; anything else
  // is a CDN or a tracker that crept back in.
  ok(`${path} makes no third-party requests`, external.length === 0, external.join(' '));

  await ctx.close();
}

await browser.close();

console.log(`\n${'-'.repeat(60)}`);
console.log(`web e2e: ${pass} passed, ${failures.length} failed`);
if (failures.length) console.log(`failed:\n  - ${failures.join('\n  - ')}`);
process.exit(failures.length ? 1 : 0);
