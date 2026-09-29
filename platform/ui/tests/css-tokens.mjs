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
  { label: 'app/src/styles.css (the agent)', file: path.join(HERE, '..', '..', '..', 'app', 'src', 'styles.css'), themes: [":root[data-theme='light']"] },
];

/**
 * Tokens written at runtime by ThemeContext rather than declared in CSS, so a
 * stylesheet may legitimately read them without a declaration.
 */
const RUNTIME_TOKENS = new Set([
  '--accent-primary', '--accent-hover', '--accent-glow', '--accent-secondary',
  '--bg-primary', '--bg-secondary', '--bg-tertiary',
  // Set alongside --accent-secondary in the same effect (ThemeContext.tsx),
  // and read only by the filled-control gradient, which already spells its own
  // fallback: var(--accent-fill, var(--accent-primary)).
  '--accent-fill', '--accent-fill-2',
]);

/**
 * Tokens set on an element by a rule or an inline style rather than on a theme
 * root: --node on canvas nodes, --tone on landing-page sections (each section
 * sets the signal colour its rail, list markers and chips read).
 */
const LOCAL_TOKENS = new Set(['--node', '--tone']);

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

// ---------------------------------------------------------------------------
// The editor's own surfaces draw their chrome from the theme tokens.
//
// The shell, the section pages (Data, Queue, Packages, Settings), the side
// panels, the canvas toolbar, the toasts and every dialog are drawn from the
// tokens, so they follow both themes and the accent. A hard-coded Tailwind
// neutral or blue (`text-slate-500`, `bg-blue-600`, `dark:bg-slate-800`…) is
// the cool slate/blue look the editor had before, and it cannot follow the
// warm Paper Circuit canvas or a changed accent. The list is explicit rather
// than a glob: node kinds keep their own colours (nodeSwatches.ts,
// useModuleNodes' getNodeColorClasses), and so do the node UIs.
//
// And a modal is only ever the one Dialog: none of these files draws its own
// dimmed overlay (`fixed inset-0 … bg-black`).
// ---------------------------------------------------------------------------
const SRC = path.join(HERE, '..', 'src');
const SURFACES = [
  'components/OAIYApp.tsx',
  'components/OAIYBuilder.tsx',
  'components/ImportMenu.tsx',
  'components/Toast.tsx',
  'components/chrome/ShellChrome.tsx',
  'components/chrome/SectionPage.tsx',
  'components/panels/FlowsSidebar.tsx',
  'components/panels/NodePalette.tsx',
  'components/panels/PropertiesPanel.tsx',
  'components/panels/LogConsole.tsx',
  'components/panels/QueuePanel.tsx',
  'components/panels/DataViewer.tsx',
  'components/panels/SettingsPanel.tsx',
  'components/panels/PackageServicesPanel.tsx',
  'components/panels/settings/ServicesTab.tsx',
  'components/panels/settings/EngineEndpointCard.tsx',
  'components/panels/settings/SecurityTab.tsx',
  'components/panels/settings/AppearanceTab.tsx',
  'components/PackageManager/PackageBrowser.tsx',
  'components/PackageManager/TrustDialog.tsx',
  'components/PackageManager/DependencyDialog.tsx',
  'components/PackageManager/ServiceStartupDialog.tsx',
  'components/dialogs/AgentToolDialog.tsx',
  'components/dialogs/ShareFlowDialog.tsx',
  'components/dialogs/PasswordPromptModal.tsx',
  'components/dialogs/LocalNetworkPermissionDialog.tsx',
  'components/dialogs/ConvertToMacroDialog.tsx',
  'components/dialogs/NewFlowDialog.tsx',
  'components/ui/Dialog.tsx',
  'components/ui/Menu.tsx',
  'components/ui/ConfirmDialog.tsx',
  'components/ui/RunWorkflowModal.tsx',
  'components/ui/MacroRunnerModal.tsx',
  'components/ui/CopyButton.tsx',
  'components/OAIYBuilder/CanvasContextMenu.tsx',
  'components/OAIYBuilder/QuickConnectPopup.tsx',
  'components/wizards/WelcomeWizard.tsx',
  'lib/addServiceDialog.tsx',
  'bundled-modules/core-image/ui/ComfyUIWorkflowDialog.tsx',
];
const HARD_CODED = /\b(?:[a-z-]+:)*(?:bg|text|border|ring|ring-offset|from|to|via|divide|placeholder|fill|stroke|outline|shadow|decoration|accent|caret)-(?:slate|gray|zinc|neutral|stone|blue|sky|indigo)-\d{2,3}(?:\/\d+)?\b/g;
const OWN_OVERLAY = /fixed inset-0[^"'`]*bg-black|bg-black[^"'`]*fixed inset-0/;

console.log('\n-- the editor\'s surfaces, on tokens --');
let surfaceProblems = 0;
for (const rel of SURFACES) {
  const file = path.join(SRC, rel);
  if (!fs.existsSync(file)) { fail(`${rel}: not found (update the list)`); surfaceProblems++; continue; }
  const text = stripComments(fs.readFileSync(file, 'utf8')).replace(/^\s*\/\/.*$/gm, '');
  const hits = [...new Set(text.match(HARD_CODED) ?? [])];
  if (hits.length) {
    fail(`${rel}: hard-coded slate/blue classes: ${hits.slice(0, 8).join(' ')}${hits.length > 8 ? ` (+${hits.length - 8})` : ''}`);
    surfaceProblems++;
  }
  if (OWN_OVERLAY.test(text)) {
    fail(`${rel}: draws its own modal overlay; use components/ui/Dialog`);
    surfaceProblems++;
  }
}
if (!surfaceProblems) pass(`${SURFACES.length} surfaces: no hard-coded slate/blue classes, and no modal of their own`);

// The check itself: a known offender is caught, and a token class is not.
const probe = 'className="text-slate-500 dark:bg-slate-800 hover:bg-blue-600/20 text-content-secondary bg-surface-tertiary"';
const caught = probe.match(HARD_CODED) ?? [];
if (caught.length === 3) pass('the check catches slate/blue classes (with variants and alpha) and passes token classes');
else fail(`the check itself: expected 3 hits in the probe, got ${caught.length} (${caught.join(' ')})`);

console.log(`\n${'-'.repeat(60)}`);
console.log(failed ? `css tokens: ${failed} problem(s)` : 'css tokens: clean');
process.exit(failed ? 1 : 0);
