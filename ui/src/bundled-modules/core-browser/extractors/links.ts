/**
 * Link extractor — runs in the page context via Browser.evaluateJson.
 *
 * Returns every `<a href>` resolved to an absolute URL, with its visible
 * text. Optional sameOriginOnly filters to the current origin.
 */

export interface LinkExtractionOptions {
  selector?: string;
  sameOriginOnly?: boolean;
}

export interface BrowserLink {
  text: string;
  href: string;
}

export function extractLinksScript(opts: LinkExtractionOptions = {}): string {
  const selector = opts.selector ? JSON.stringify(opts.selector) : "null";
  const sameOriginOnly = opts.sameOriginOnly === true;

  return /* js */ `
(() => {
  const sel = ${selector};
  const root = sel ? document.querySelector(sel) : document;
  if (!root) return [];
  const sameOrigin = ${sameOriginOnly};
  const origin = location.origin;
  const out = [];
  const seen = new Set();
  root.querySelectorAll("a[href]").forEach((a) => {
    let href;
    try {
      href = new URL(a.getAttribute("href"), location.href).href;
    } catch (_) {
      return;
    }
    if (sameOrigin) {
      try {
        if (new URL(href).origin !== origin) return;
      } catch (_) { return; }
    }
    if (seen.has(href)) return;
    seen.add(href);
    out.push({
      text: (a.innerText || a.textContent || "").trim(),
      href,
    });
  });
  return out;
})()
`;
}

export type LinkExtractionResult = BrowserLink[];
