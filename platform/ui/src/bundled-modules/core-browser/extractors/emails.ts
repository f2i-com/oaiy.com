/**
 * Email extractor — runs in the page context via Browser.evaluateJson.
 *
 * Looks at mailto: links and the visible body text. Filters obvious
 * placeholders (example.com, noreply@, sentry.io, wixpress.com, etc).
 */

export const extractEmailsScript = /* js */ `
(() => {
  const out = new Set();
  const skip = (e) =>
    e.includes("example.com") ||
    e.includes("email.com") ||
    e.includes("domain.com") ||
    e.includes("sentry.io") ||
    e.includes("wixpress.com") ||
    e.startsWith("noreply@") ||
    e.startsWith("no-reply@");

  document.querySelectorAll('a[href^="mailto:"]').forEach((a) => {
    const href = a.getAttribute("href") || "";
    const e = href.replace(/^mailto:/i, "").split("?")[0].trim().toLowerCase();
    if (e && e.includes("@") && !skip(e)) out.add(e);
  });

  const text = (document.body && document.body.innerText) || "";
  const re = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const e = m[0].toLowerCase();
    if (!skip(e)) out.add(e);
    if (out.size > 200) break;
  }

  return Array.from(out);
})()
`;

export type EmailExtractionResult = string[];
