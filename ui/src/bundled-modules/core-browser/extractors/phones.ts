/**
 * Phone extractor — runs in the page context via Browser.evaluateJson.
 *
 * Combines tel: hrefs and visible text matches. Region selects the
 * regex flavour; unknown / empty region falls through to GENERIC.
 *
 * Free-form input is accepted (the JSON node passes whatever the user
 * typed) but only the listed regions have a tuned regex. The fallback
 * still finds most phone-shaped strings — it just won't filter out
 * false positives quite as well.
 */

export type PhoneRegion = string; // freeform; unknown values use GENERIC

export function extractPhonesScript(region: PhoneRegion = "AU"): string {
  const re = phonesRegexForRegion(region);

  return /* js */ `
(() => {
  const out = new Set();
  document.querySelectorAll('a[href^="tel:"]').forEach((a) => {
    const href = a.getAttribute("href") || "";
    const raw = href.replace(/^tel:/i, "").trim();
    const norm = raw.replace(/[^\\d+]/g, "");
    if (norm.length >= 8) out.add(raw);
  });

  const text = (document.body && document.body.innerText) || "";
  const re = new RegExp(${JSON.stringify(re)}, "g");
  let m;
  while ((m = re.exec(text)) !== null) {
    out.add(m[0].replace(/[\\s.-]+/g, " ").trim());
    if (out.size > 100) break;
  }

  return Array.from(out);
})()
`;
}

export type PhoneExtractionResult = string[];

/**
 * Tuned regexes for common locales. Unknown region → GENERIC fallback,
 * which captures most phone-shaped strings (international prefix + 7-12
 * digits in 2-3 chunks). The country list is small on purpose; tuning a
 * regex per locale is a long tail and the GENERIC pattern is good
 * enough for arbitrary regions.
 */
function phonesRegexForRegion(region: string): string {
  const r = region.trim().toUpperCase();
  switch (r) {
    case "AU":
      return String.raw`(?:\+?61|0)[\s.-]?(?:[2-9])[\s.-]?\d{4}[\s.-]?\d{4}|(?:1300|1800)[\s.-]?\d{3}[\s.-]?\d{3}`;
    case "US":
    case "CA":
      // NANP: optional +1, then 3-3-4 with various separators
      return String.raw`(?:\+?1[\s.-]?)?\(?[2-9]\d{2}\)?[\s.-]?[2-9]\d{2}[\s.-]?\d{4}`;
    case "UK":
    case "GB":
      // +44 / 0 prefix, area code 1-5 digits, total 10-11 digits
      return String.raw`(?:\+?44[\s.-]?|0)\d{2,5}[\s.-]?\d{3,4}[\s.-]?\d{3,4}`;
    case "NZ":
      return String.raw`(?:\+?64|0)[\s.-]?\d[\s.-]?\d{3}[\s.-]?\d{4}`;
    case "IN":
      return String.raw`(?:\+?91[\s.-]?|0)[6-9]\d{4}[\s.-]?\d{5}`;
    case "DE":
      return String.raw`(?:\+?49[\s.-]?|0)\d{2,5}[\s.-]?\d{3,8}`;
    case "FR":
      return String.raw`(?:\+?33[\s.-]?|0)[1-9](?:[\s.-]?\d{2}){4}`;
    case "":
    case "GENERIC":
    default:
      return String.raw`(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}`;
  }
}
