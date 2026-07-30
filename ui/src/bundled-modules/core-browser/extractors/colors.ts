/**
 * Color extractor — runs in the page context via Browser.evaluateJson.
 *
 * Samples computed styles from header/nav, buttons, headings, and the
 * first batch of links, then ranks colors by weighted frequency.
 * Near-white and near-black are filtered so the result is brand-ish.
 *
 * Returns a primary/secondary/accent shorthand plus the full ranked
 * palette so callers can reach further if they need more.
 */

export interface ColorExtractionOptions {
  paletteSize?: number;
}

export interface BrowserColorResult {
  primary?: string;
  secondary?: string;
  accent?: string;
  palette: string[];
}

export function extractColorsScript(opts: ColorExtractionOptions = {}): string {
  const paletteSize = Math.min(16, Math.max(1, opts.paletteSize ?? 5));

  return /* js */ `
(() => {
  const counts = {};
  const rgbToHex = (r, g, b) =>
    "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("");
  const parseColor = (color) => {
    if (!color || color === "transparent" || color === "rgba(0, 0, 0, 0)") return null;
    if (color.startsWith("#")) {
      return color.length === 4
        ? "#" + color[1] + color[1] + color[2] + color[2] + color[3] + color[3]
        : color;
    }
    const m = color.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
    if (!m) return null;
    const r = +m[1], g = +m[2], b = +m[3];
    if (r > 240 && g > 240 && b > 240) return null;
    if (r < 15 && g < 15 && b < 15) return null;
    return rgbToHex(r, g, b);
  };

  const elements = [
    ...Array.from(document.querySelectorAll("header, nav, .header, .nav, .navbar")),
    ...Array.from(document.querySelectorAll("button, .btn, .button, a.cta")),
    ...Array.from(document.querySelectorAll("h1, h2, h3")),
    ...Array.from(document.querySelectorAll("a")).slice(0, 30),
  ];

  for (const el of elements) {
    let style;
    try { style = window.getComputedStyle(el); } catch (_) { continue; }
    const bg = parseColor(style.backgroundColor);
    if (bg) counts[bg] = (counts[bg] || 0) + 2;
    const text = parseColor(style.color);
    if (text) counts[text] = (counts[text] || 0) + 1;
    const border = parseColor(style.borderColor);
    if (border) counts[border] = (counts[border] || 0) + 1;
  }

  const palette = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([color]) => color)
    .slice(0, ${paletteSize});

  return {
    primary: palette[0],
    secondary: palette[1],
    accent: palette[2],
    palette,
  };
})()
`;
}

export type ColorExtractionResult = BrowserColorResult;
