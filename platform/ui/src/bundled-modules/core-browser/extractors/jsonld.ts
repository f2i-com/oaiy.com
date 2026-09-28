/**
 * JSON-LD extractor — runs in the page context via Browser.evaluateJson.
 *
 * Returns the parsed contents of every `<script type="application/ld+json">`
 * block. Invalid JSON is skipped silently.
 */

export const extractJsonLdScript = /* js */ `
(() => {
  const out = [];
  document.querySelectorAll('script[type="application/ld+json"]').forEach((s) => {
    const text = (s.textContent || "").trim();
    if (!text) return;
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        for (const item of parsed) out.push(item);
      } else {
        out.push(parsed);
      }
    } catch (_) { /* skip invalid */ }
  });
  return out;
})()
`;

export type JsonLdExtractionResult = unknown[];
