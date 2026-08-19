import { chromium } from 'playwright';

const URL = 'http://localhost:5173/';
const VIEWPORTS = [
  { name: '390x844', width: 390, height: 844 },
  { name: '768x1024', width: 768, height: 1024 },
  { name: '1440x900', width: 1440, height: 900 },
];

const run = async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const vp of VIEWPORTS) {
    const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1800);

    const data = await page.evaluate(() => {
      const R = (n) => Math.round(n * 10) / 10;
      const box = (el) => {
        const b = el.getBoundingClientRect();
        return { x: R(b.x), y: R(b.y + window.scrollY), w: R(b.width), h: R(b.height), top: R(b.top + window.scrollY), bottom: R(b.bottom + window.scrollY) };
      };

      const result = {};
      result.docScrollWidth = document.documentElement.scrollWidth;
      result.docClientWidth = document.documentElement.clientWidth;
      result.bodyScrollWidth = document.body.scrollWidth;
      result.innerWidth = window.innerWidth;

      const main = document.querySelector('main');
      const sections = [...main.children];
      result.sections = sections.map((s, i) => {
        const b = box(s);
        const c = getComputedStyle(s);
        const h = s.querySelector('h1,h2');
        return {
          i,
          id: s.id || null,
          cls: s.className,
          top: b.top, bottom: b.bottom, h: b.h, w: b.w, x: b.x,
          padT: c.paddingTop, padB: c.paddingBottom,
          marT: c.marginTop, marB: c.marginBottom,
          heading: h ? h.textContent.trim().slice(0, 50) : null,
        };
      });

      result.sectionGaps = [];
      for (let i = 0; i < sections.length - 1; i++) {
        const a = box(sections[i]), b2 = box(sections[i + 1]);
        result.sectionGaps.push({ from: i, to: i + 1, gap: R(b2.top - a.bottom) });
      }

      result.contentGaps = [];
      for (let i = 0; i < sections.length - 1; i++) {
        const a = sections[i], b2 = sections[i + 1];
        const ael = [...a.querySelectorAll('*')].filter(e => e.getBoundingClientRect().height > 0 && getComputedStyle(e).position !== 'absolute');
        const bel = [...b2.querySelectorAll('*')].filter(e => e.getBoundingClientRect().height > 0 && getComputedStyle(e).position !== 'absolute');
        const aBottom = Math.max(...ael.map(e => box(e).bottom));
        const bTop = Math.min(...bel.map(e => box(e).top));
        result.contentGaps.push({ from: i, to: i + 1, inkGap: R(bTop - aBottom) });
      }

      const hero = document.querySelector('#top');
      const heroGrid = hero.querySelector('div.grid');
      const heroCS = getComputedStyle(heroGrid);
      const h1 = hero.querySelector('h1');
      const heroSvg = hero.querySelector('svg[role="img"]');
      const heroGraphWrap = heroSvg ? heroSvg.closest('div.rounded-2xl') : null;
      result.hero = {
        section: box(hero),
        cols: heroCS.gridTemplateColumns,
        colGap: heroCS.columnGap,
        rowGap: heroCS.rowGap,
        padT: heroCS.paddingTop,
        padB: heroCS.paddingBottom,
        padL: heroCS.paddingLeft,
        maxW: heroCS.maxWidth,
        h1: Object.assign(box(h1), { fontSize: getComputedStyle(h1).fontSize, lineHeight: getComputedStyle(h1).lineHeight, letterSpacing: getComputedStyle(h1).letterSpacing }),
        copyCol: box(heroGrid.children[0]),
        graphCol: box(heroGrid.children[1]),
        graphCard: heroGraphWrap ? box(heroGraphWrap) : null,
        svg: heroSvg ? Object.assign(box(heroSvg), { viewBox: heroSvg.getAttribute('viewBox') }) : null,
      };
      const heroP = hero.querySelector('p.lp-reveal');
      if (heroP) result.hero.para = Object.assign(box(heroP), { fontSize: getComputedStyle(heroP).fontSize, maxWidth: getComputedStyle(heroP).maxWidth });
      const trust = [...hero.querySelectorAll('p')].find(p => p.textContent.includes('Runs in your browser'));
      if (trust) {
        result.hero.trustRow = Object.assign(box(trust), { fontSize: getComputedStyle(trust).fontSize });
        result.hero.trustSpans = [...trust.querySelectorAll('span.inline-flex')].map(s => Object.assign(box(s), { t: s.textContent.trim().slice(0, 24) }));
      }
      const pill = hero.querySelector('.lp-pill');
      if (pill) result.hero.pill = box(pill);
      const heroBtnRow = hero.querySelector('div.lp-reveal.mt-8');
      if (heroBtnRow) result.hero.btnRow = Object.assign(box(heroBtnRow), { flexWrap: getComputedStyle(heroBtnRow).flexWrap, gap: getComputedStyle(heroBtnRow).gap });

      result.buttons = [...document.querySelectorAll('a.btn, button.btn')].map(b => {
        const c = getComputedStyle(b);
        return Object.assign({ text: b.textContent.trim().slice(0, 40), cls: b.className }, box(b), {
          fontSize: c.fontSize, padL: c.paddingLeft, padR: c.paddingRight,
          padT: c.paddingTop, minHeight: c.minHeight, radius: c.borderRadius, gap: c.gap,
        });
      });

      const allGrids = [...document.querySelectorAll('main .grid')];
      result.grids = allGrids.map((g, gi) => {
        const c = getComputedStyle(g);
        const kids = [...g.children].filter(k => k.getBoundingClientRect().height > 0);
        const kboxes = kids.map(k => {
          const innerEl = k.querySelector('article, .card') || k.firstElementChild || k;
          return { outer: box(k), inner: box(innerEl), text: k.textContent.trim().slice(0, 26) };
        });
        const rows = {};
        kboxes.forEach(kb => {
          const key = Math.round(kb.outer.top / 4) * 4;
          (rows[key] = rows[key] || []).push(kb);
        });
        const rowInfo = Object.entries(rows).map(([top, items]) => ({
          top: Number(top),
          n: items.length,
          heights: items.map(i => i.outer.h),
          innerHeights: items.map(i => i.inner.h),
          spread: R(Math.max(...items.map(i => i.outer.h)) - Math.min(...items.map(i => i.outer.h))),
          innerSpread: R(Math.max(...items.map(i => i.inner.h)) - Math.min(...items.map(i => i.inner.h))),
          labels: items.map(i => i.text),
        }));
        return {
          gi,
          sectionId: (g.closest('section') && (g.closest('section').id || g.closest('section').className.slice(0, 24))) || '',
          cls: g.className.slice(0, 90),
          cols: c.gridTemplateColumns,
          gap: c.gap, rowGap: c.rowGap, colGap: c.columnGap,
          alignItems: c.alignItems,
          nKids: kids.length,
          box: box(g),
          rows: rowInfo,
        };
      });

      const lineInfo = (el) => {
        const range = document.createRange();
        range.selectNodeContents(el);
        const rects = [...range.getClientRects()].filter(rr => rr.width > 0.5 && rr.height > 0.5);
        const lineTops = {};
        rects.forEach(rr => { const k = Math.round(rr.top); (lineTops[k] = lineTops[k] || []).push(rr); });
        const sortedTops = Object.keys(lineTops).map(Number).sort((a, b) => a - b);
        const lastRects = sortedTops.length ? lineTops[sortedTops[sortedTops.length - 1]] : [];
        const lastLineWidth = lastRects.length ? R(Math.max(...lastRects.map(rr => rr.right)) - Math.min(...lastRects.map(rr => rr.left))) : 0;
        const w = R(el.getBoundingClientRect().width);
        return { lines: sortedTops.length, w, lastLineWidth, lastRatio: w ? R(lastLineWidth / w) : 0 };
      };

      result.headings = [...document.querySelectorAll('main h1, main h2, main h3')].map(h => {
        const c = getComputedStyle(h);
        return Object.assign({
          tag: h.tagName, text: h.textContent.trim().slice(0, 60),
          fontSize: c.fontSize, lineHeight: c.lineHeight, align: c.textAlign, wrapMode: c.textWrap || c.textWrapStyle || 'n/a',
        }, lineInfo(h), { top: box(h).top });
      });

      result.paragraphs = [...document.querySelectorAll('main p')].map(p => Object.assign({
        text: p.textContent.trim().slice(0, 42), fontSize: getComputedStyle(p).fontSize,
      }, lineInfo(p))).filter(p => p.lines >= 2);

      const marqueeMask = document.querySelector('.lp-marquee-mask');
      const marqueeTrack = document.querySelector('.lp-marquee-track');
      if (marqueeMask && marqueeTrack) {
        const mc = getComputedStyle(marqueeMask), tc = getComputedStyle(marqueeTrack);
        const chips = [...marqueeTrack.children].map(ch => Object.assign({ text: ch.textContent.trim() }, box(ch)));
        result.marquee = {
          mask: box(marqueeMask),
          maskOverflow: mc.overflow,
          maskImage: (mc.maskImage || mc.webkitMaskImage || '').slice(0, 120),
          track: box(marqueeTrack),
          trackScrollWidth: marqueeTrack.scrollWidth,
          trackGap: tc.gap,
          anim: tc.animationName + ' ' + tc.animationDuration,
          transform: tc.transform,
          nChips: marqueeTrack.children.length,
          chipHeights: [...new Set(chips.map(c => c.h))],
          chipW: chips.slice(0, 10).map(c => ({ t: c.text, w: c.w, x: c.x })),
          halfWidth: R(marqueeTrack.scrollWidth / 2),
          speedPxPerSec: R((marqueeTrack.scrollWidth / 2) / parseFloat(tc.animationDuration)),
        };
        const strip = marqueeMask.closest('section');
        result.marquee.section = box(strip);
        result.marquee.sectionPad = getComputedStyle(strip).paddingTop + ' / ' + getComputedStyle(strip).paddingBottom;
        const label = strip.querySelector('p');
        if (label) result.marquee.label = Object.assign(box(label), { fs: getComputedStyle(label).fontSize, ls: getComputedStyle(label).letterSpacing, lines: lineInfo(label).lines });
      }

      const footer = document.querySelector('footer');
      const innerF = footer.firstElementChild;
      const ic = getComputedStyle(innerF);
      result.footer = {
        box: box(footer),
        inner: box(innerF),
        flexDirection: ic.flexDirection,
        gap: ic.gap,
        justify: ic.justifyContent,
        align: ic.alignItems,
        pad: ic.paddingTop + ' ' + ic.paddingRight + ' ' + ic.paddingBottom + ' ' + ic.paddingLeft,
        children: [...innerF.children].map(ch => Object.assign({ tag: ch.tagName, text: ch.textContent.trim().slice(0, 40) }, box(ch))),
        navLinks: [...footer.querySelectorAll('nav a')].map(a => Object.assign({ text: a.textContent.trim(), fs: getComputedStyle(a).fontSize }, box(a))),
      };
      const lastSection = sections[sections.length - 1];
      result.footer.gapFromLastSection = R(box(footer).top - box(lastSection).bottom);

      const nav = document.querySelector('.site-nav');
      if (nav) result.nav = Object.assign(box(nav), { position: getComputedStyle(nav).position });

      const vw = document.documentElement.clientWidth;
      result.overflowing = [];
      document.querySelectorAll('main *, footer *, header *, nav *').forEach(el => {
        const b = el.getBoundingClientRect();
        const c = getComputedStyle(el);
        if (b.width > 0 && (b.right > vw + 1 || b.left < -1)) {
          result.overflowing.push({ tag: el.tagName, cls: (el.className || '').toString().slice(0, 55), left: R(b.left), right: R(b.right), w: R(b.width), pos: c.position, ariaHidden: el.getAttribute('aria-hidden'), text: el.textContent.trim().slice(0, 26) });
        }
      });
      result.overflowing = result.overflowing.slice(0, 30);

      const named = {};
      const find = (frag) => [...document.querySelectorAll('main .grid')].find(g => g.className.includes(frag));
      const pg = find('0.9fr_1.1fr');
      if (pg) named.privacy = { cols: getComputedStyle(pg).gridTemplateColumns, gap: getComputedStyle(pg).gap, box: box(pg), kids: [...pg.children].map(k => box(k)) };
      const dg = find('1fr_0.85fr');
      if (dg) named.desktop = { cols: getComputedStyle(dg).gridTemplateColumns, gap: getComputedStyle(dg).gap, box: box(dg), kids: [...dg.children].map(k => box(k)) };
      const rg = find('lg:grid-cols-[1fr_1fr]');
      if (rg) named.remote = { cols: getComputedStyle(rg).gridTemplateColumns, gap: getComputedStyle(rg).gap, box: box(rg), kids: [...rg.children].map(k => box(k)) };
      result.namedGrids = named;

      const pre = document.querySelector('main pre');
      if (pre) result.pre = Object.assign(box(pre), { scrollWidth: pre.scrollWidth, clientWidth: pre.clientWidth, overflowX: getComputedStyle(pre).overflowX, fontSize: getComputedStyle(pre).fontSize });

      result.docHeight = document.documentElement.scrollHeight;
      return result;
    });

    out[vp.name] = data;
    await page.screenshot({ path: 'probe-shot-' + vp.width + '-top.png' });
    await ctx.close();
  }
  await browser.close();
  console.log(JSON.stringify(out, null, 1));
};
run().catch(e => { console.error(e); process.exit(1); });
