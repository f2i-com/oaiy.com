import { chromium } from 'playwright';

const URL = 'http://localhost:5173/';
const WIDTHS = [360, 390, 420, 480, 560, 600, 640, 700, 719, 721, 740, 759, 761, 768, 800, 830, 833, 860, 900, 959, 961, 1000, 1024, 1100, 1240, 1440, 1600];

const run = async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const rows = [];
  for (const w of WIDTHS) {
    await page.setViewportSize({ width: w, height: 900 });
    await page.waitForTimeout(220);
    const d = await page.evaluate(() => {
      const R = (n) => Math.round(n * 10) / 10;
      const vw = document.documentElement.clientWidth;
      const bar = document.querySelector('.site-nav-bar');
      const brand = bar.children[0];
      const links = document.querySelector('.site-nav-links');
      const navBtn = links.querySelector('a.btn');
      const heroGrid = document.querySelector('#top div.grid');
      const h1 = document.querySelector('#top h1');
      const svg = document.querySelector('#top svg[role="img"]');
      const remoteGrid = [...document.querySelectorAll('main .grid')].find(g => g.className.includes('lg:grid-cols-[1fr_1fr]'));
      const remoteCard = remoteGrid ? remoteGrid.children[1].querySelector('.overflow-hidden.rounded-xl') : null;
      const capGrid = document.querySelector('#capabilities .grid');
      const howGrid = document.querySelector('#how .grid');
      const vpGrid = [...document.querySelectorAll('main .grid')].find(g => g.className.includes('gap-4 sm:grid-cols-2') && !g.className.includes('lg:grid-cols-4'));
      const footerInner = document.querySelector('footer').firstElementChild;
      const footCopy = footerInner.lastElementChild;
      const footNavA = footerInner.querySelector('nav a');
      const brandTag = brand.querySelector('.lp-tagline');
      // h1 line count
      const rng = document.createRange(); rng.selectNodeContents(h1);
      const tops = new Set([...rng.getClientRects()].filter(r => r.width > 0.5).map(r => Math.round(r.top)));
      // widest overflow in main
      let worst = 0, worstEl = '';
      document.querySelectorAll('main *, footer *, header *').forEach(el => {
        const b = el.getBoundingClientRect();
        const c = getComputedStyle(el);
        if (c.position === 'absolute' && el.getAttribute('aria-hidden') === 'true') return;
        if (el.closest('.lp-marquee-mask')) return;
        if (b.width > 0 && b.right - vw > worst) { worst = R(b.right - vw); worstEl = el.tagName + '.' + (el.className || '').toString().slice(0, 34); }
      });
      return {
        vw,
        navBarW: R(bar.getBoundingClientRect().width),
        navBarScrollW: bar.scrollWidth,
        brandW: R(brand.getBoundingClientRect().width),
        tagShown: brandTag ? getComputedStyle(brandTag).display : 'none',
        linksW: R(links.getBoundingClientRect().width),
        linksRight: R(links.getBoundingClientRect().right),
        navBtnRight: R(navBtn.getBoundingClientRect().right),
        navOverflow: R(navBtn.getBoundingClientRect().right - vw),
        heroCols: getComputedStyle(heroGrid).gridTemplateColumns,
        heroColGap: getComputedStyle(heroGrid).columnGap,
        h1fs: getComputedStyle(h1).fontSize,
        h1w: R(h1.getBoundingClientRect().width),
        h1lines: tops.size,
        svgW: svg ? R(svg.getBoundingClientRect().width) : 0,
        svgScale: svg ? R((svg.getBoundingClientRect().width / 740) * 1000) / 1000 : 0,
        capCols: getComputedStyle(capGrid).gridTemplateColumns.split(' ').length,
        howCols: getComputedStyle(howGrid).gridTemplateColumns.split(' ').length,
        vpCols: vpGrid ? getComputedStyle(vpGrid).gridTemplateColumns.split(' ').length : 0,
        remoteCols: remoteGrid ? getComputedStyle(remoteGrid).gridTemplateColumns : '',
        remoteCardW: remoteCard ? R(remoteCard.getBoundingClientRect().width) : 0,
        remoteCardRight: remoteCard ? R(remoteCard.getBoundingClientRect().right) : 0,
        remoteOverflow: remoteCard ? R(remoteCard.getBoundingClientRect().right - vw) : 0,
        footDir: getComputedStyle(footerInner).flexDirection,
        footH: R(footerInner.getBoundingClientRect().height),
        footCopyH: R(footCopy.getBoundingClientRect().height),
        footNavAH: R(footNavA.getBoundingClientRect().height),
        worstOverflow: worst, worstEl,
        docScrollW: document.documentElement.scrollWidth,
      };
    });
    rows.push(d);
  }
  await browser.close();
  console.log(JSON.stringify(rows, null, 1));
};
run().catch(e => { console.error(e); process.exit(1); });
