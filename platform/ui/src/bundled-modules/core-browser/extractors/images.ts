/**
 * Image extractor — runs in the page context via Browser.evaluateJson.
 *
 * Returns every `<img src>` resolved to an absolute URL with alt and
 * intrinsic dimensions. Optional minSize skips icons.
 */

export interface ImageExtractionOptions {
  selector?: string;
  minSize?: number;
}

export interface BrowserImage {
  src: string;
  alt: string;
  width: number;
  height: number;
}

export function extractImagesScript(opts: ImageExtractionOptions = {}): string {
  const selector = opts.selector ? JSON.stringify(opts.selector) : "null";
  const minSize = Number.isFinite(opts.minSize) ? Math.max(0, opts.minSize as number) : 0;

  return /* js */ `
(() => {
  const sel = ${selector};
  const root = sel ? document.querySelector(sel) : document;
  if (!root) return [];
  const min = ${minSize};
  const out = [];
  const seen = new Set();
  root.querySelectorAll("img[src]").forEach((img) => {
    let src;
    try {
      src = new URL(img.getAttribute("src"), location.href).href;
    } catch (_) { return; }
    const w = img.naturalWidth || img.width || 0;
    const h = img.naturalHeight || img.height || 0;
    if (min > 0 && w < min && h < min) return;
    if (seen.has(src)) return;
    seen.add(src);
    out.push({
      src,
      alt: img.getAttribute("alt") || "",
      width: w,
      height: h,
    });
  });
  return out;
})()
`;
}

export type ImageExtractionResult = BrowserImage[];
