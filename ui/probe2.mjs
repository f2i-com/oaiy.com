import { chromium } from 'playwright';

const styleOf = (page, sel) => page.evaluate((s) => {
  const el = document.querySelector(s);
  if (!el) return null;
  const cs = getComputedStyle(el);
  const r = el.getBoundingClientRect();
  const cx = Math.round(r.x + r.width / 2), cy = Math.round(r.y + r.height / 2);
  const top = document.elementFromPoint(cx, cy);
  return {
    background: cs.backgroundColor, color: cs.color, cursor: cs.cursor,
    hitTestSelf: !!top && (top === el || el.contains(top)),
    topEl: top ? (top.tagName + '.' + String(top.className).slice(0, 40)) : null,
  };
}, sel);

const probe = async (page, label, tag) => {
  const sel = `.oaiy-icon-btn[aria-label="${label}"]`;
  const el = page.locator(sel).first();
  if (!(await el.count())) return console.log(`  !! not found: ${label}`);
  await page.mouse.move(3, 3); await page.waitForTimeout(350);
  const before = await styleOf(page, sel);
  const cls = (await el.getAttribute('class')).trim();
  // hover by raw mouse move to the centre -- no actionability wait, so occlusion doesn't throw
  const box = await el.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForTimeout(450);
  const after = await styleOf(page, sel);
  const changed = before.background !== after.background || before.color !== after.color;
  console.log(`  [${tag}] "${label}" class="${cls}"`);
  console.log(`      hittable=${before.hitTestSelf} topElementAtCentre=${before.topEl}`);
  console.log(`      bg  ${before.background}  ->  ${after.background}`);
  console.log(`      fg  ${before.color}  ->  ${after.color}`);
  console.log(`      HOVER FEEDBACK: ${changed ? 'yes' : 'NONE'}`);
  await page.mouse.move(3, 3); await page.waitForTimeout(250);
  return { changed, hittable: before.hitTestSelf };
};

const run = async () => {
  const browser = await chromium.launch();
  for (const vp of [{ width: 1440, height: 900 }, { width: 1920, height: 1080 }]) {
    const ctx = await browser.newContext({ viewport: vp });
    const page = await ctx.newPage();
    await page.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);
    console.log(`\n\n=========== VIEWPORT ${vp.width}x${vp.height} ===========`);

    for (const theme of ['dark', 'light']) {
      const tlabel = theme === 'dark' ? 'Prism Lab (dark) theme' : 'Paper Circuit (light) theme';
      await page.locator(`.oaiy-icon-btn[aria-label="${tlabel}"]`).first().click();
      await page.waitForTimeout(600);
      console.log(`\n### theme=${theme}`);

      console.log('\n  -- flows rail: DEFAULT state (page load default) --');
      await probe(page, 'Toggle the flows rail', theme);

      console.log('\n  -- flows rail: after one click (flipped) --');
      await page.locator('.oaiy-icon-btn[aria-label="Toggle the flows rail"]').first().click();
      await page.waitForTimeout(600);
      await probe(page, 'Toggle the flows rail', theme);
      // restore
      await page.locator('.oaiy-icon-btn[aria-label="Toggle the flows rail"]').first().click();
      await page.waitForTimeout(600);

      console.log('\n  -- job queue: OFF then ON --');
      await probe(page, 'Toggle the job queue', theme);
      await page.locator('.oaiy-icon-btn[aria-label="Toggle the job queue"]').first().click();
      await page.waitForTimeout(800);
      const r = await probe(page, 'Toggle the job queue', theme);
      // geometry of the queue panel vs topbar
      const geo = await page.evaluate(() => {
        const d = document.querySelector('[role="dialog"][aria-modal="true"]');
        const b = document.querySelector('.oaiy-icon-btn[aria-label="Toggle the job queue"]');
        const f = document.querySelector('.oaiy-icon-btn[aria-label="Toggle the flows rail"]');
        const rr = (e) => e ? { x: Math.round(e.getBoundingClientRect().x), y: Math.round(e.getBoundingClientRect().y), w: Math.round(e.getBoundingClientRect().width), h: Math.round(e.getBoundingClientRect().height), z: getComputedStyle(e).zIndex } : null;
        return { dialog: rr(d), queueBtn: rr(b), flowsBtn: rr(f) };
      });
      console.log('      geometry:', JSON.stringify(geo));
      // close it (Escape or its own close button)
      await page.keyboard.press('Escape'); await page.waitForTimeout(600);
      const stillOpen = await page.locator('[role="dialog"][aria-modal="true"]').count();
      if (stillOpen) {
        const c = page.locator('[role="dialog"][aria-modal="true"] [aria-label*="Close" i], [role="dialog"][aria-modal="true"] button').first();
        try { await c.click({ timeout: 1500 }); } catch {}
        await page.waitForTimeout(600);
      }
      console.log(`      dialog still open after Escape: ${stillOpen ? 'YES' : 'no'}`);
    }
    await ctx.close();
  }
  await browser.close();
};
run().catch((e) => { console.error(e); process.exit(1); });
