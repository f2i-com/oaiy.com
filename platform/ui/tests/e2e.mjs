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
 * that a flow can be created, that both themes resolve every token they use,
 * that the desktop page's links and library states are right, and that a flow's
 * API keys are sealed in IndexedDB rather than left in plaintext localStorage.
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

/**
 * What a marketing page may ask for beyond its own site: nothing, except the desktop page's one read of the
 * service library (`/api/service-library`), which goes to the build's API base (another origin when the site is
 * built with VITE_API_BASE). Anything else (GitHub, a probe of a desktop on a product port, a CDN) is stray.
 */
function strayRequests(urls, pagePath, site) {
  let libraryReads = 0;
  return urls.filter((u) => {
    if (/^(data|blob):/.test(u)) return false;
    const url = new URL(u);
    if (url.origin === site) return false;
    if (pagePath === '/desktop.html' && url.pathname === '/api/service-library' && ++libraryReads === 1) return false;
    return true;
  });
}
const section = (s) => console.log(`\n-- ${s} --`);

const browser = await chromium.launch();

// The editor looks for OAIY Desktop on this machine (127.0.0.1:17972) from the moment it opens, and the
// browser it runs in is a browser on a machine that may have one running. A test must neither talk to it
// nor depend on it, so every request to the product's own ports is refused before it leaves, in every
// context this file opens. What answers instead is something that is not OAIY Desktop (it says another
// product), so the editor sees what it sees on a machine with no desktop, and a refused connection
// does not put an error in the console that the "clean console" checks would count.
const OWN_PORTS = new Set(['17972', '17872', '17973', '8080', '7860', '8783', '9333']);
const newContext = browser.newContext.bind(browser);
browser.newContext = async (...args) => {
  const context = await newContext(...args);
  await context.route(
    (url) => ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) && OWN_PORTS.has(url.port),
    (route) => {
      const headers = { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'access-control-allow-private-network': 'true' };
      if (route.request().method() === 'OPTIONS') return route.fulfill({ status: 204, headers });
      return route.fulfill({ status: 200, headers, contentType: 'application/json', body: JSON.stringify({ status: 'ok', product: 'e2e-stand-in' }) });
    },
  );
  return context;
};

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

    // The example flow (the Flows section; the hero is the Agent) must depict real node ids, not invented ones.
    const ids = await page.locator('svg[role="img"] text')
      .filter({ hasText: /^(IMAGE_GEN|AI_LLM|CONDITION|OUTPUT)$/ }).count();
    ok('the example flow uses the app\'s real node ids', ids === 4, `matched ${ids}/4`);
    ok('the example flow labels the feedback loop',
      (await page.locator('svg[role="img"]').innerHTML()).includes('regenerate'));

    // The screenshots: four, each loaded at its own size and described. (They load as they are scrolled to.)
    const shots = page.locator('img.oaiy-shot');
    const count = await shots.count();
    for (let i = 0; i < count; i++) await shots.nth(i).scrollIntoViewIfNeeded();
    await page.waitForTimeout(800);
    const loaded = await shots.evaluateAll((els) => els.map((e) => ({
      ok: e.complete && e.naturalWidth === Number(e.getAttribute('width')) && e.naturalHeight === Number(e.getAttribute('height')),
      alt: e.alt.length > 40,
    })));
    ok('the four screenshots load at the size the page gives them, and are described', count === 4 && loaded.every((s) => s.ok && s.alt), JSON.stringify(loaded));
    ok('the page compares the browser and the desktop, in a table a screen reader can read',
      (await page.locator('table.lp-compare th[scope="col"]').count()) === 3 && (await page.locator('table.lp-compare tbody th[scope="row"]').count()) >= 6);
    ok('it says what needs the desktop, plainly',
      /NVIDIA GPU/.test((await page.locator('#why').textContent()) ?? '') && /not published for download yet/.test((await page.locator('#why').textContent()) ?? ''));

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

  // ------------------------------------ desktop page: repository links, library
  section('desktop page: links into the repository + the service library');
  // The standalone release build serves no /api: a static host answers a 404, or its front page
  // for every path. Either way the library says so and offers no retry.
  for (const [how, reply] of [
    ['a 404', { status: 404, contentType: 'text/plain', body: 'Not Found' }],
    ['the front page', { status: 200, contentType: 'text/html; charset=utf-8', body: '<!doctype html><title>OAIY</title>' }],
  ]) {
    const { ctx, page } = await open(theme);
    await page.route('**/api/service-library', (route) => route.fulfill(reply));
    await page.goto(BASE + '/desktop.html', { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);
    const library = page.locator('#library');
    ok(`with no /api (${how}) the library says this copy of the site does not serve it`,
      (await library.locator('.lp-note', { hasText: 'does not serve the service library' }).count()) === 1);
    ok(`and (${how}) offers no retry, which could not help`, (await library.getByRole('button', { name: 'Try again' }).count()) === 0);
    const browse = await library.getByRole('link', { name: 'Browse templates' }).getAttribute('href');
    ok('"Browse templates" goes to the folder where it is now', !!browse?.endsWith('/tree/main/platform/api/service-library'), browse ?? '');
    const docs = await page.getByRole('link', { name: 'Installation documentation' }).getAttribute('href');
    ok('"Installation documentation" goes to the desktop folder where it is now', !!docs?.endsWith('/tree/main/platform/desktop'), docs ?? '');
    await ctx.close();
  }
  {
    // A server that should have a library and fails is worth another try.
    const { ctx, page } = await open(theme);
    let asked = 0;
    await page.route('**/api/service-library', (route) => { asked++; return route.fulfill({ status: 503, contentType: 'text/plain', body: 'down' }); });
    await page.goto(BASE + '/desktop.html', { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);
    const library = page.locator('#library');
    ok('a server error says the library could not be loaded and offers a retry',
      (await library.getByRole('button', { name: 'Try again' }).count()) === 1
        && (await library.locator('.lp-note', { hasText: 'couldn’t load the service library' }).count()) === 1);
    const before = asked;
    await library.getByRole('button', { name: 'Try again' }).click();
    await page.waitForTimeout(500);
    ok('and the retry asks again', asked === before + 1, `asked ${asked}, was ${before}`);
    await ctx.close();
  }
  {
    // A library the API serves is listed, and each template downloads from the API.
    const { ctx, page } = await open(theme);
    await page.route('**/api/service-library', (route) => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ services: [{ file: 'ollama.json', name: 'Ollama', description: 'Local language models.', category: 'LLM', downloadUrl: '/api/service-library/ollama.json' }] }),
    }));
    await page.goto(BASE + '/desktop.html', { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);
    const items = page.locator('#library .lp-library-item');
    ok('a served library is listed', (await items.count()) === 1 && (await items.first().locator('h3').textContent()) === 'Ollama');
    ok('and each template downloads from the API', ((await items.first().locator('a[download]').getAttribute('href')) ?? '').endsWith('/api/service-library/ollama.json'));
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
    // Workflows, Data, Queue, Packages; Settings sits at the foot of the rail.
    ok('sidebar has the full primary nav', (await page.locator('.oaiy-nav button').count()) === 4);
    ok('Settings is in the rail too', (await page.locator('.oaiy-settings-btn').count()) === 1);
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

    // New flow asks for a name in the one dialog, then opens the canvas.
    const create = page.locator('.oaiy-new').first();
    if (await create.count()) {
      await create.click();
      await page.waitForTimeout(400);
      ok('New flow opens the dialog', (await page.locator('[data-testid="new-flow-dialog"]').count()) === 1);
      await page.getByRole('button', { name: /Create flow/i }).click();
      await page.waitForTimeout(2200);
    }
    // An empty flow opens with the palette; the inspector is a toolbar button
    // away (docked from the start only where the canvas has room for both).
    ok('creating a flow reveals the node palette', (await page.locator('[data-testid="node-palette"]').count()) > 0);
    const propsButton = page.locator('.oaiy-toolbar button[aria-label$="the properties"]');
    ok('the canvas toolbar offers the properties', (await propsButton.count()) === 1);
    if ((await page.locator('aside[aria-label="Inspector"]').count()) === 0) await propsButton.click();
    await page.waitForTimeout(300);
    ok('the properties open beside the canvas', (await page.locator('aside[aria-label="Inspector"]').count()) === 1);
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
// The marketing pages on a phone: nothing scrolls sideways, and the comparison reads as cards, not a wide table.
section('marketing pages at phone width');
for (const theme of ['dark', 'light']) {
  const { ctx, page } = await open(theme, { width: 390, height: 800 });
  for (const path of ['/', '/desktop.html']) {
    await page.goto(BASE + path, { waitUntil: 'networkidle' });
    await page.waitForTimeout(800);
    ok(`${theme}: ${path} does not scroll sideways at 390px`, await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
      String(await page.evaluate(() => document.documentElement.scrollWidth)));
  }
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  const wrap = await page.evaluate(() => {
    const el = document.querySelector('.lp-compare-wrap');
    const table = document.querySelector('table.lp-compare');
    return { fits: el.scrollWidth <= el.clientWidth + 1, stacked: getComputedStyle(table.querySelector('tbody tr')).display === 'block' };
  });
  ok(`${theme}: the comparison fits a phone, each row a card`, wrap.fits && wrap.stacked, JSON.stringify(wrap));
  await ctx.close();
}

// ---------------------------------------------------------------------------
// "Download OAIY Desktop": the device it is for, the link, and what it never does.
//
// The button is worked out in the page from what the browser says of itself (no request) and links to
// the release its build was made for: a versioned build names the file, a local build goes to the
// latest release. Both are accepted here (this file runs against the dev server too); tests/downloads.mjs
// has the exact names. A device OAIY Desktop is not built for is told so and is not given a button.
section('download OAIY Desktop');
{
  const ua = {
    windows: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
    linux: 'Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0',
    mac: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15',
    iphone: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1',
    android: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36',
  };
  const desktopSite = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36';
  const linuxArmFirefox = 'Mozilla/5.0 (X11; Linux aarch64; rv:143.0) Gecko/20100101 Firefox/143.0';
  // navigator.userAgentData (Chrome, Edge) or none (Firefox, Safari); maxTouchPoints 5 is what an iPad reports. `high` is what
  // getHighEntropyValues answers: the user agent string is frozen (ARM Linux says x86_64, Windows on ARM says x64).
  const x64 = { architecture: 'x86', bitness: '64' };
  const devices = [
    { name: 'Windows in Chrome', userAgent: ua.windows, uaData: 'Windows', high: x64, touch: 0, os: 'windows' },
    { name: 'Linux in Firefox', userAgent: ua.linux, uaData: null, touch: 0, os: 'linux' },
    { name: 'Linux in Chrome', userAgent: desktopSite, uaData: 'Linux', high: x64, touch: 0, os: 'linux' },
    { name: 'a Mac in Safari', userAgent: ua.mac, uaData: null, touch: 0, os: 'mac' },
    { name: 'an iPhone', userAgent: ua.iphone, uaData: null, touch: 5, os: 'iphone' },
    { name: 'an iPad (a Mac that has a touch screen)', userAgent: ua.mac, uaData: null, touch: 5, os: 'ipad' },
    { name: 'Android in Chrome', userAgent: ua.android, uaData: 'Android', high: x64, touch: 5, os: 'android' },
    { name: 'Android asked for the desktop site (says Linux x86_64, has a touch screen)', userAgent: desktopSite, uaData: 'Linux', high: x64, touch: 5, mobile: true, os: 'android' },
    { name: 'Linux on ARM64 in Chrome (its user agent says x86_64)', userAgent: desktopSite, uaData: 'Linux', high: { architecture: 'arm', bitness: '64' }, touch: 0, os: 'wrong-processor' },
    { name: 'Windows on ARM in Edge (its user agent says x64)', userAgent: ua.windows, uaData: 'Windows', high: { architecture: 'arm', bitness: '64' }, touch: 0, os: 'wrong-processor' },
    { name: '32-bit Windows in Chrome', userAgent: ua.windows, uaData: 'Windows', high: { architecture: 'x86', bitness: '32' }, touch: 0, os: 'wrong-processor' },
    { name: 'Linux on ARM in Firefox (no userAgentData: its user agent says aarch64)', userAgent: linuxArmFirefox, uaData: null, touch: 0, os: 'wrong-processor' },
  ];
  let versioned = null; // whether the build names files: read from the Windows button
  const site = new URL(BASE).origin;
  for (const d of devices) {
    const ctx = await browser.newContext({ userAgent: d.userAgent, viewport: { width: 1280, height: 900 } });
    await ctx.addInitScript(({ uaData, high, touch, mobile }) => {
      Object.defineProperty(navigator, 'userAgentData', {
        configurable: true,
        value: uaData ? { platform: uaData, mobile: !!mobile, brands: [], getHighEntropyValues: async () => high ?? {} } : undefined,
      });
      Object.defineProperty(navigator, 'maxTouchPoints', { configurable: true, value: touch });
      try { localStorage.setItem('oaiy_theme', 'dark'); } catch { /* blocked */ }
    }, { uaData: d.uaData, high: d.high, touch: d.touch, mobile: d.mobile });
    const page = await ctx.newPage();
    const requests = [];
    page.on('request', (r) => requests.push(r.url()));
    for (const path of ['/', '/desktop.html']) {
      requests.length = 0;
      await page.goto(BASE + path, { waitUntil: 'networkidle' });
      await page.waitForTimeout(600);
      const box = page.locator('.oaiy-download').first();
      const button = box.locator('a.btn').first();
      const href = (await button.getAttribute('href')) ?? '';
      const text = ((await box.textContent()) ?? '').replace(/\s+/g, ' ');
      const where = `${d.name} on ${path}`;
      if (d.os === 'windows') {
        ok(`${where}: the button is the Windows installer (or the latest release)`, /releases\/(latest|download\/v?\d+\.\d+\.\d+\/oaiy-desktop-\d+\.\d+\.\d+-windows-x64-setup\.exe)$/.test(href) && /Download OAIY Desktop/.test(await button.textContent()), href);
        ok(`${where}: it says the installer is not code-signed yet`, /Not code-signed yet/.test(text), text);
        versioned = /releases\/download\//.test(href);
      } else if (d.os === 'wrong-processor') {
        // After the first draw the page asks the browser what processor it has: no button that would install what cannot run.
        const others = await box.locator('.oaiy-download-more a').count();
        ok(`${where}: no installer button, the sentence about 64-bit Intel or AMD, and a way to the web app or the desktop page`,
          /OAIY Desktop needs a 64-bit Intel or AMD computer, and this looks like (an ARM|a 32-bit) one\. The web app works in your browser\./.test(text) && !/releases\/(latest|download)/.test(href) && href === (path === '/' ? 'desktop.html' : 'app.html'), `${href} | ${text}`);
        ok(`${where}: the files are still listed under other downloads (${versioned ? 'a versioned build' : 'an unversioned build lists only the releases page'})`,
          versioned === false ? others === 0 : others >= 4, `${others} links`);
      } else if (d.os === 'linux') {
        ok(`${where}: the button is the AppImage (or the latest release)`, /releases\/(latest|download\/v?\d+\.\d+\.\d+\/oaiy-desktop-\d+\.\d+\.\d+-linux-x86_64\.AppImage)$/.test(href) && /Download OAIY Desktop/.test(await button.textContent()), href);
      } else {
        // The button goes to the desktop page, or on the desktop page itself (where that would go nowhere) to the web app.
        ok(`${where}: no download, the honest sentence, and a way to the web app or the desktop page`, /OAIY Desktop is for Windows and Linux\. The web app works in your browser\./.test(text) && !/releases\/(latest|download)/.test(href) && href === (path === '/' ? 'desktop.html' : 'app.html'), `${href} | ${text}`);
      }
      ok(`${where}: "All downloads" is the releases page`, (await box.locator('a', { hasText: 'All downloads on GitHub' }).count()) >= 1
        && /github\.com\/f2i-com\/oaiy\.com\/releases$/.test((await box.locator('a', { hasText: 'All downloads on GitHub' }).first().getAttribute('href')) ?? ''));
      const stray = strayRequests(requests, path, site);
      ok(`${where}: nothing asked of anyone but its own site (no GitHub, no probe of a desktop; the desktop page's one read of the service library aside)`, stray.length === 0, stray.join(' '));
    }
    await ctx.close();
  }

  // In the editor: under the engine card in the sidebar, and in Settings, while no desktop answers.
  const ctx = await browser.newContext({ userAgent: ua.windows, viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(() => {
    try {
      localStorage.setItem('skipSplash', 'true');
      localStorage.setItem('oaiy.wizard.completed', 'true');
      localStorage.setItem('oaiy_theme', 'dark');
    } catch { /* blocked */ }
  });
  const page = await ctx.newPage();
  await page.goto(`${BASE}/app.html`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);
  const get = page.locator('.oaiy-engine-get a');
  ok('the sidebar offers OAIY Desktop under the engine card while none answers', (await get.count()) === 1 && /releases\/(latest|download\/v?\d+\.\d+\.\d+\/oaiy-desktop-\d+\.\d+\.\d+-windows-x64-setup\.exe)$/.test((await get.getAttribute('href')) ?? ''));
  await page.locator('.oaiy-settings-btn').first().click();
  await page.waitForTimeout(600);
  const card = page.locator('.oaiy-download-card');
  ok('Settings offers it in the engine card', (await card.count()) === 1 && (await card.locator('a.btn').count()) === 1 && /Download OAIY Desktop for Windows|Download OAIY Desktop/.test(await card.locator('a.btn').textContent()));
  await ctx.close();

  // OAIY's own window is a desktop already: no offer.
  const inOaiy = await browser.newContext({ userAgent: ua.windows, viewport: { width: 1440, height: 900 } });
  await inOaiy.addInitScript(() => {
    window.__OAIY_DESKTOP__ = { origin: 'http://127.0.0.1:1', token: 'e2e', theme: 'dark' };
    try { localStorage.setItem('skipSplash', 'true'); localStorage.setItem('oaiy.wizard.completed', 'true'); } catch { /* blocked */ }
  });
  const inPage = await inOaiy.newPage();
  await inPage.goto(`${BASE}/app.html`, { waitUntil: 'networkidle' });
  await inPage.waitForTimeout(2000);
  ok('in OAIY\'s own window there is no offer to download it', (await inPage.locator('.oaiy-engine-get, .oaiy-download').count()) === 0);
  await inOaiy.close();
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
      const panel = document.querySelector('[data-testid="flows-rail"]');
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
// API keys are sealed in IndexedDB, and the browser is asked to keep the editor's storage.
//
// A flow's API keys were plain text in localStorage (`oaiy_web_secrets`). They are
// sealed in IndexedDB now (AES-GCM under a key that cannot be exported), and a
// plaintext map left from before is moved in when the editor first loads. That
// keeps them out of localStorage; it is not protection against a copy of the
// profile (see tauri-shim/secretVault.ts). The vault's logic is
// tests/secret-vault.mjs against a stand-in database; this is the real IndexedDB
// and WebCrypto. (A page that is not a secure context has no WebCrypto, keeps
// the keys as it always did, and is skipped here.)
section('API keys sealed in IndexedDB + persistent storage');
{
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.addInitScript(() => {
    window.__persistCalls = 0;
    try {
      const proto = Object.getPrototypeOf(navigator.storage);
      const persist = proto.persist;
      proto.persist = function () { window.__persistCalls++; return persist.call(this); };
    } catch { /* a browser without it: the editor has to cope */ }
    try {
      if (!localStorage.getItem('__e2e_seeded')) {
        localStorage.setItem('__e2e_seeded', '1');
        localStorage.setItem('oaiy_web_secrets', JSON.stringify({ E2E_OPENAI_KEY: 'sk-e2e-plain-1', E2E_OTHER_KEY: 'sk-e2e-plain-2' }));
        localStorage.setItem('skipSplash', 'true');
        localStorage.setItem('oaiy.wizard.completed', 'true');
      }
    } catch { /* blocked storage */ }
  });
  await page.goto(`${BASE}/app.html`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2200);
  const invoke = (cmd, args) => page.evaluate(([c, a]) => window.__TAURI__.core.invoke(c, a), [cmd, args]);
  const plain = () => page.evaluate(() => localStorage.getItem('oaiy_web_secrets'));

  ok('the editor asked the browser to keep its storage, once', (await page.evaluate(() => window.__persistCalls)) === 1);
  if (!(await page.evaluate(() => window.isSecureContext && !!crypto.subtle))) {
    console.log('  (not a secure context: keys stay in plaintext here, so the sealing checks are skipped)');
  } else {
    await page.waitForFunction(() => localStorage.getItem('oaiy_web_secrets') === null, null, { timeout: 10000 }).catch(() => {});
    ok('the plaintext key map was moved out of localStorage', (await plain()) === null, String(await plain()));
    const sealed = await page.evaluate(async () => {
      const db = await new Promise((resolve, reject) => { const r = indexedDB.open('oaiy-web-secrets'); r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); });
      const all = (store) => new Promise((resolve) => { const q = db.transaction(store).objectStore(store).getAll(); q.onsuccess = () => resolve(q.result); });
      const names = await new Promise((resolve) => { const q = db.transaction('secrets').objectStore('secrets').getAllKeys(); q.onsuccess = () => resolve(q.result); });
      const [key] = await all('keys');
      const records = await all('secrets');
      let exportable = true;
      try { await crypto.subtle.exportKey('raw', key); } catch { exportable = false; }
      const latin1 = new TextDecoder('latin1');
      const leaks = records.some((r) => `${latin1.decode(r.data)}${latin1.decode(r.iv)}`.includes('sk-e2e'));
      db.close();
      return { names: [...names].sort(), alg: key?.algorithm?.name, extractable: key?.extractable, exportable, leaks, ivBytes: records.map((r) => r.iv.byteLength) };
    });
    ok('each key is its own sealed record', JSON.stringify(sealed.names) === JSON.stringify(['E2E_OPENAI_KEY', 'E2E_OTHER_KEY']), JSON.stringify(sealed.names));
    ok('sealed with an AES-GCM key that cannot be exported', sealed.alg === 'AES-GCM' && sealed.extractable === false && sealed.exportable === false, JSON.stringify(sealed));
    ok('and the records are ciphertext', !sealed.leaks && sealed.ivBytes.every((n) => n === 12), JSON.stringify(sealed.ivBytes));
    const got = await invoke('get_secrets', { keys: ['E2E_OPENAI_KEY', 'E2E_OTHER_KEY', 'E2E_NOT_SET'] });
    ok('the editor still gets the keys it had', got.E2E_OPENAI_KEY === 'sk-e2e-plain-1' && got.E2E_OTHER_KEY === 'sk-e2e-plain-2' && !('E2E_NOT_SET' in got), JSON.stringify(got));

    await invoke('store_secret', { key: 'E2E_NEW_KEY', value: 'sk-e2e-new' });
    ok('a key saved now is sealed, never plaintext', (await plain()) === null);
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(2200);
    const after = await invoke('get_secrets', { keys: ['E2E_OPENAI_KEY', 'E2E_NEW_KEY'] });
    ok('after a reload the keys are still there', after.E2E_OPENAI_KEY === 'sk-e2e-plain-1' && after.E2E_NEW_KEY === 'sk-e2e-new', JSON.stringify(after));
    await invoke('store_secret', { key: 'E2E_NEW_KEY', value: '' });
    ok('an empty value deletes a key', !('E2E_NEW_KEY' in (await invoke('get_secrets', { keys: ['E2E_NEW_KEY'] }))));
  }
  ok('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));

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
  const asked = [];
  const woff2 = [];
  page.on('request', (r) => {
    const u = r.url();
    if (!/^(https?:\/\/localhost|https?:\/\/127\.0\.0\.1|data:|blob:)/.test(u)) external.push(u);
    asked.push(u);
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
  // The check above lets loopback through, which is where OAIY Desktop is looked for: the marketing pages
  // never look (only the editor does), so they may ask for nothing but their own site (and the desktop page's
  // one read of the service library, which goes to the API base of a build that has one).
  if (path !== '/app.html') {
    const stray = strayRequests(asked, path, new URL(BASE).origin);
    ok(`${path} makes no request beyond its own site (no probe of a desktop)`, stray.length === 0, stray.join(' '));
  }

  await ctx.close();
}

await browser.close();

console.log(`\n${'-'.repeat(60)}`);
console.log(`web e2e: ${pass} passed, ${failures.length} failed`);
if (failures.length) console.log(`failed:\n  - ${failures.join('\n  - ')}`);
process.exit(failures.length ? 1 : 0);
