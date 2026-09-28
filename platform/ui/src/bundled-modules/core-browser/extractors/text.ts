/**
 * Visible text extractor — runs in the page context via Browser.evaluateJson.
 *
 * Strips scripts, styles, nav, footer, header, aside elements before
 * reading innerText, then collapses whitespace. Optional scope selector
 * narrows the source. Optional maxLength truncates without breaking JSON.
 */

export interface TextExtractionOptions {
  selector?: string;
  maxLength?: number;
}

export function extractTextScript(opts: TextExtractionOptions = {}): string {
  const selector = opts.selector ? JSON.stringify(opts.selector) : "null";
  const maxLength = Number.isFinite(opts.maxLength) ? Math.max(0, opts.maxLength as number) : 0;

  return /* js */ `
(() => {
  const sel = ${selector};
  const root = sel ? document.querySelector(sel) : document.body;
  if (!root) return "";
  const clone = root.cloneNode(true);
  clone.querySelectorAll("script, style, noscript, nav, footer, header, aside, .nav, .footer, .header").forEach((n) => n.remove());
  let text = (clone.innerText || clone.textContent || "").replace(/\\s+/g, " ").trim();
  const max = ${maxLength};
  if (max > 0 && text.length > max) text = text.slice(0, max);
  return text;
})()
`;
}

export type TextExtractionResult = string;
