/**
 * Logo candidate detector — runs in the page context via Browser.evaluateJson.
 *
 * Walks a priority list of selectors that real sites use for logos,
 * skipping tiny icons. SVG logos are reported with a sentinel src so
 * callers can decide whether to inline them rather than download.
 *
 * Returns the best candidate plus the full ranked list.
 */

export interface LogoCandidate {
  src: string;
  isSvg: boolean;
  width: number;
  height: number;
  selector: string;
  alt: string;
}

export interface LogoDetectionResult {
  best?: LogoCandidate;
  candidates: LogoCandidate[];
}

export const extractLogoScript = /* js */ `
(() => {
  const selectors = [
    'header img[alt*="logo" i]',
    'header img[src*="logo"]',
    'header img[class*="logo" i]',
    'header img[id*="logo" i]',
    'nav img[alt*="logo" i]',
    'nav img[src*="logo"]',
    '.site-logo img',
    '.logo img',
    '#logo img',
    'a[class*="logo" i] img',
    'a[href="/"] img',
    'header svg[class*="logo" i]',
    '.logo svg',
    'header img',
    'nav img',
    'link[rel*="icon"]',
  ];

  const out = [];
  const seen = new Set();
  for (const sel of selectors) {
    document.querySelectorAll(sel).forEach((el) => {
      let src = "";
      let isSvg = false;
      const tag = el.tagName.toLowerCase();
      if (tag === "img") {
        const raw = el.getAttribute("src") || "";
        try { src = new URL(raw, location.href).href; } catch (_) { src = raw; }
        const w = el.naturalWidth || el.width || 0;
        const h = el.naturalHeight || el.height || 0;
        if (w && w < 32 && h && h < 32) return;
        if (seen.has(src)) return;
        seen.add(src);
        out.push({ src, isSvg, width: w, height: h, selector: sel, alt: el.getAttribute("alt") || "" });
      } else if (tag === "svg") {
        isSvg = true;
        src = "__svg__";
        const key = sel + "::svg::" + (out.length);
        if (seen.has(key)) return;
        seen.add(key);
        const r = el.getBoundingClientRect();
        out.push({ src, isSvg, width: r.width, height: r.height, selector: sel, alt: "" });
      } else if (tag === "link") {
        const raw = el.getAttribute("href") || "";
        try { src = new URL(raw, location.href).href; } catch (_) { src = raw; }
        if (seen.has(src)) return;
        seen.add(src);
        out.push({ src, isSvg: false, width: 0, height: 0, selector: sel, alt: "" });
      }
    });
  }

  return { best: out[0], candidates: out };
})()
`;

export type LogoExtractionResult = LogoDetectionResult;
