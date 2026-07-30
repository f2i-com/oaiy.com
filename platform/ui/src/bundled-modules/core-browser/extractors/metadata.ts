/**
 * Meta-tag extractor — runs in the page context via Browser.evaluateJson.
 *
 * Returns title, description, canonical URL, and the union of the
 * common meta-name and Open Graph / Twitter property tags as a flat map.
 */

export interface PageMetadata {
  title: string;
  description?: string;
  canonical?: string;
  meta: Record<string, string>;
}

export const extractMetadataScript = /* js */ `
(() => {
  const meta = {};
  document.querySelectorAll("meta").forEach((m) => {
    const name = m.getAttribute("name") || m.getAttribute("property");
    const content = m.getAttribute("content");
    if (name && content) meta[name.toLowerCase()] = content;
  });
  const canonicalEl = document.querySelector('link[rel="canonical"]');
  return {
    title: document.title || "",
    description: meta["description"] || meta["og:description"] || meta["twitter:description"],
    canonical: canonicalEl ? canonicalEl.getAttribute("href") || undefined : undefined,
    meta,
  };
})()
`;

export type MetadataExtractionResult = PageMetadata;
