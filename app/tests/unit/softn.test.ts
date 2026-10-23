import { describe, expect, it } from 'vitest';
import { checkEvents } from '../../src/softn/knowledge';
import {
  SOFTN_BLANK,
  SOFTN_STARTER,
  checkLogicSyntax,
  checkProject,
  findApps,
  isSoftnApp,
  moduleSyntaxAsScript,
  normalizedManifest,
  pageLogicSources,
  resolveApp,
  softnAppOf,
} from '../../src/softn/softn';
import { Vfs } from '../../src/vfs/vfs';

const write = (files: Record<string, string>) => {
  const vfs = new Vfs();
  for (const [path, text] of Object.entries(files)) vfs.writeFile(`/${path}`, text, { parents: true });
  return vfs;
};

const INDEX = [
  '<import About from="./pages/about.html" />',
  '<script src="main.js"></script>',
  '<!-- <script src="old.js"></script> -->',
  '<script type="application/json" src="data.js"></script>',
  '<script src="https://cdn.example.com/x.js"></script>',
  '<script src="softn/softn.js"></script>',
  '<link rel="stylesheet" href="styles.css">',
  '<App>',
  '  <Text>{total()}</Text>',
  '</App>',
  '',
].join('\n');

/** An app written as HTML, in the folder shop/, with what an app may carry beside its code. */
const shop = (extra: Record<string, string> = {}) =>
  write({
    'shop/manifest.json': JSON.stringify({ name: 'Shop', version: '1.0.0', main: 'index.html', files: { ui: ['index.html'], logic: ['main.js'] } }),
    'shop/index.html': INDEX,
    'shop/pages/about.html': '<Text>About</Text>\n',
    'shop/main.js': 'import { sum } from "./lib/sum.js"\nexport function total() { return sum([1, 2]) }\n',
    'shop/lib/sum.js': 'export const sum = (xs) => xs.reduce((a, b) => a + b, 0)\n',
    'shop/old.js': 'a library nothing links (\n',
    'shop/data.js': '{ "not": "code" (\n',
    'shop/vendor/chart.js': 'window.Chart = (\n',
    'shop/styles.css': 'p { color: red }\n',
    'shop/assets/help.html': '<h1>Help</h1>\n',
    ...extra,
  });

/** The engine's part, played by this process: the program's completion value. */
const run = async (source: string) => String((0, eval)(source));

describe('a SoftN app written as HTML', () => {
  it('is an app: its main page is index.html, and its pages are the app\'s, shown as the app', () => {
    const vfs = shop();
    expect(findApps(vfs)).toEqual(['shop']);
    expect(resolveApp(vfs, 'shop')).toEqual({ ok: true, root: 'shop' });
    expect(softnAppOf(vfs, 'shop/index.html')).toBe('shop');
    expect(softnAppOf(vfs, '/shop/pages/about.html')).toBe('shop');
    // what the app shows or serves, and its code, are no pages
    expect(softnAppOf(vfs, 'shop/assets/help.html')).toBeNull();
    expect(softnAppOf(vfs, 'shop/main.js')).toBeNull();
    expect(softnAppOf(vfs, 'site/index.html')).toBeNull();
  });

  it('names its code with <script src>, as the composer reads it', () => {
    // a commented-out script, one that is data, one from another site and the runtime's own loader are not the app's code
    expect(pageLogicSources(INDEX)).toEqual(['main.js']);
    expect(pageLogicSources('<logic src="../logic/main.logic" />\n<App />')).toEqual(['../logic/main.logic']);
    expect(pageLogicSources('<script title="a>b" src=\'main.py\' type="text/python"></script>')).toEqual(['main.py']);
  });

  it('exports with its pages under ui and the .js its pages link (and what that imports) under logic', () => {
    const manifest = normalizedManifest(shop(), 'shop');
    expect(manifest.main).toBe('index.html');
    const files = manifest.files as Record<string, string[]>;
    expect([...files.ui].sort()).toEqual(['index.html', 'pages/about.html']);
    // the entry the manifest lists first; a library, data and a script nothing links are not the app's code
    expect(files.logic).toEqual(['main.js', 'lib/sum.js']);
    expect(files.assets).toEqual(['assets/help.html']);
  });

  it('keeps a file the manifest lists under assets there, a page or a script', () => {
    const vfs = shop({
      'shop/manifest.json': JSON.stringify({ name: 'Shop', version: '1.0.0', main: 'index.html', files: { ui: ['index.html'], logic: ['main.js'], assets: ['help/page.html', 'vendor/chart.js'] } }),
      'shop/help/page.html': '<h1>Help</h1>\n',
    });
    const files = normalizedManifest(vfs, 'shop').files as Record<string, string[]>;
    expect(files.ui).not.toContain('help/page.html');
    expect([...files.assets].sort()).toEqual(['assets/help.html', 'help/page.html', 'vendor/chart.js']);
  });

  it('checks clean, saying which pages and logic the manifest does not list', () => {
    const findings = checkProject(shop(), 'shop');
    expect(findings.filter((f) => f.level === 'error')).toEqual([]);
    expect(findings).toEqual(
      expect.arrayContaining([
        { level: 'warning', file: 'pages/about.html', message: 'not listed in manifest.json `files.ui` (export adds it)' },
        { level: 'warning', file: 'lib/sum.js', message: 'not listed in manifest.json `files.logic` (export adds it)' },
      ]),
    );
    expect(findings.some((f) => /old\.js|data\.js|chart\.js|styles\.css/.test(f.file))).toBe(false);
  });

  it('refuses a main that is no page, and logic that is both JavaScript and Python', () => {
    const notPage = shop({ 'shop/manifest.json': JSON.stringify({ name: 'Shop', version: '1.0.0', main: 'main.js', files: {} }) });
    expect(isSoftnApp(notPage, 'shop')).toBe(false);
    expect(checkProject(notPage, 'shop')).toContainEqual({ level: 'error', file: 'manifest.json', message: '`main` names main.js, which is not a page: it names the app\'s main page, e.g. "index.html" (or a .ui page)' });
    const mixed = checkProject(shop({ 'shop/extra.py': 'x = 1\n' }), 'shop');
    expect(mixed).toContainEqual({ level: 'error', file: 'extra.py', message: 'an app\'s logic is JavaScript (.js or .logic) or Python (.py), never both' });
  });

  it('compiles the .js its pages link as the runtime reads it, module syntax and all, and names the file that does not compile', async () => {
    expect(await checkLogicSyntax(shop(), 'shop', run)).toEqual([]);
    const broken = await checkLogicSyntax(shop({ 'shop/lib/sum.js': 'export function sum(xs) {\n  return xs.reduce((a, b) => a + b, 0\n}\n' }), 'shop', run);
    expect(broken).toEqual([{ level: 'error', file: 'lib/sum.js', message: expect.stringMatching(/^syntax error: /) }]);
  });

  it('reads logic written as a module as the script it is, line for line', () => {
    const module = [
      'export function a() {}',
      'export default class B {}',
      'export { a, B }',
      'import { x } from "./x.js"',
      'import { y as z } from "./y.js"',
      '  export const c = 1',
      'let exported = "export function"',
    ].join('\n');
    expect(moduleSyntaxAsScript(module).split('\n')).toEqual([
      'function a() {}',
      'class B {}',
      '',
      'import "./x.js"',
      'import { y as z } from "./y.js"',
      '  const c = 1',
      'let exported = "export function"',
    ]);
  });

  it('says so of a handler on a component that takes none, in an .html page too', async () => {
    const vfs = shop({ 'shop/index.html': '<script src="main.js"></script>\n<App>\n  <Text @click={total}>Total</Text>\n</App>\n' });
    expect(await checkEvents(vfs, 'shop')).toEqual([
      { level: 'error', file: 'index.html', message: 'line 3: <Text> takes no events, so its @click never runs. Use <Button>, or put it inside a <Box @click={…}>.' },
    ]);
  });
});

describe('a new app', () => {
  it.each([
    ['the task-list starter', SOFTN_STARTER],
    ['the blank app', SOFTN_BLANK],
  ])('%s is written as HTML and checks clean', async (_, files) => {
    const vfs = new Vfs();
    for (const [path, text] of files) vfs.writeFile(`/tasks/${path}`, text, { parents: true });
    expect(findApps(vfs)).toEqual(['tasks']);
    expect(files.map(([path]) => path).sort()).toEqual(['index.html', 'main.js', 'manifest.json']);
    expect(checkProject(vfs, 'tasks')).toEqual([]);
    expect(await checkLogicSyntax(vfs, 'tasks', run)).toEqual([]);
    expect((normalizedManifest(vfs, 'tasks').files as Record<string, string[]>).logic).toEqual(['main.js']);
  });
});
