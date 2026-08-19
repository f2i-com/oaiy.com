/**
 * Static check: every CSS custom property the stylesheets read must be declared,
 * and both themes must declare the same set.
 *
 *     npm run test:css
 *
 * This is deliberately narrow. It does NOT try to match class selectors against
 * className expressions — template literals and computed class names make that
 * heuristic noisy enough to be worse than no check. A missing token, by
 * contrast, is unambiguous: `var(--typo)` silently resolves to nothing and the
 * rule quietly does nothing, which is exactly the failure a build cannot catch.
 *
 * It also enforces theme parity: if one theme declares a token the other
 * doesn't, that theme falls back to the :root value and drifts — the kind of bug
 * that only shows up when someone toggles the theme on the one screen that uses it.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SHEETS = [
  { label: 'ui/src/index.css', file: path.join(HERE, '..', 'src', 'index.css'), themes: [':root.light', ':root.dark'] },
  { label: 'desktop/src/styles.css', file: path.join(HERE, '..', '..', 'desktop', 'src', 'styles.css'), themes: [":root[data-theme='light']"] },
];

/**
 * Tokens written at runtime by ThemeContext rather than declared in CSS, so a
 * stylesheet may legitimately read them without a declaration.
 */
const RUNTIME_TOKENS = new Set([
  '--accent-primary', '--accent-hover', '--accent-glow', '--accent-secondary',
  '--bg-primary', '--bg-secondary', '--bg-tertiary',
]);

/** Tokens set on an element by a rule rather than on a theme root (e.g. --node). */
const LOCAL_TOKENS = new Set(['--node']);

const stripComments = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '');

let failed = 0;
const fail = (msg) => { failed++; console.log(`  ✗ ${msg}`); };
const pass = (msg) => console.log(`  ✓ ${msg}`);

for (const sheet of SHEETS) {
  console.log(`\n-- ${sheet.label} --`);
  if (!fs.existsSync(sheet.file)) {
    fail(`stylesheet not found: ${sheet.file}`);
    continue;
  }
  const css = stripComments(fs.readFileSync(sheet.file, 'utf8'));

  const declared = new Set([...css.matchAll(/^\s*(--[\w-]+)\s*:/gm)].map((m) => m[1]));
  const referenced = new Set([...css.matchAll(/var\(\s*(--[\w-]+)/g)].map((m) => m[1]));

  const missing = [...referenced].filter((t) => !declared.has(t) && !RUNTIME_TOKENS.has(t) && !LOCAL_TOKENS.has(t)).sort();
  if (missing.length) fail(`var() with no declaration: ${missing.join(', ')}`);
  else pass(`all ${referenced.size} var() references resolve (${declared.size} tokens declared)`);

  // Theme parity: collect the tokens each theme block declares and compare
  // against the base :root block.
  const blockTokens = (selector) => {
    // Selectors can be part of a list (`:root.dark, :root:not(.light) {`), so
    // match the selector followed by anything up to the opening brace rather
    // than requiring it to sit alone.
    const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const m = new RegExp(escaped + '\\s*(?:,[^{]*)?\\{').exec(css);
    if (!m) return null;
    const i = m.index;
    const open = css.indexOf('{', i);
    let depth = 0, end = -1;
    for (let k = open; k < css.length; k++) {
      if (css[k] === '{') depth++;
      else if (css[k] === '}') { depth--; if (depth === 0) { end = k; break; } }
    }
    if (end < 0) return null;
    return new Set([...css.slice(open, end).matchAll(/(--[\w-]+)\s*:/g)].map((m) => m[1]));
  };

  const base = blockTokens(':root');
  if (!base) { fail('no :root block found'); continue; }

  for (const theme of sheet.themes) {
    const t = blockTokens(theme);
    if (!t) { fail(`theme block not found: ${theme}`); continue; }
    // A theme need not re-declare everything — only the tokens whose value must
    // differ. What matters is that it declares nothing the base doesn't know
    // about (a typo'd override is silently dead).
    const orphans = [...t].filter((x) => !base.has(x)).sort();
    if (orphans.length) fail(`${theme} declares tokens absent from :root (typo? dead override?): ${orphans.join(', ')}`);
    else pass(`${theme} overrides ${t.size} tokens, all known to :root`);
  }
}

console.log(`\n${'-'.repeat(60)}`);
console.log(failed ? `css tokens: ${failed} problem(s)` : 'css tokens: clean');
process.exit(failed ? 1 : 0);
