import { describe, expect, it } from 'vitest';
import { findPages, pageFiles, resolvePage } from '../../src/preview/page';
import { Vfs } from '../../src/vfs/vfs';

describe('web pages for the preview', () => {
  const project = () => {
    const vfs = new Vfs();
    vfs.writeFile('/about.html', '<h1>About</h1>', { parents: true });
    vfs.writeFile('/index.html', '<h1>Home</h1>', { parents: true });
    vfs.writeFile('/site/index.html', '<h1>Site</h1>', { parents: true });
    vfs.writeFile('/site/css/style.css', 'h1 { color: red }', { parents: true });
    vfs.writeFile('/node_modules/pkg/index.html', 'x', { parents: true });
    // A SoftN app's files are not pages.
    vfs.writeFile('/app/manifest.json', JSON.stringify({ name: 'A', main: 'ui/main.ui' }), { parents: true });
    vfs.writeFile('/app/ui/main.ui', '<App />', { parents: true });
    vfs.writeFile('/app/help.html', 'x', { parents: true });
    return vfs;
  };

  it('finds the pages, index pages first, leaving out packages and SoftN apps', () => {
    expect(findPages(project())).toEqual(['index.html', 'about.html', 'site/index.html']);
  });

  it('resolves the page a tool names: a file, a folder with an index, or the only index', () => {
    const vfs = project();
    expect(resolvePage(vfs, 'site')).toEqual({ ok: true, path: 'site/index.html' });
    expect(resolvePage(vfs, '/about.html')).toEqual({ ok: true, path: 'about.html' });
    expect(resolvePage(vfs, undefined)).toEqual({ ok: true, path: 'index.html' });
    const missing = resolvePage(vfs, 'nope.html');
    expect(missing.ok).toBe(false);
    expect(!missing.ok && missing.reason).toContain('index.html');
    expect(resolvePage(new Vfs(), undefined)).toEqual({ ok: false, reason: 'the project has no .html page yet' });
  });

  it('sends a page its own folder, and only small files from elsewhere', () => {
    const vfs = project();
    vfs.writeFile('/big.bin', new Uint8Array(3 * 1024 * 1024));
    vfs.writeFile('/logo.png', new Uint8Array(100));
    const files = pageFiles(vfs, 'site/index.html');
    expect(Object.keys(files)).toEqual(expect.arrayContaining(['site/index.html', 'site/css/style.css', 'logo.png']));
    expect(files['big.bin']).toBeUndefined();
    expect(Object.keys(files).some((p) => p.startsWith('node_modules/'))).toBe(false);
    // A page at the root gets the project, but not its large media.
    vfs.writeFile('/video.mp4', new Uint8Array(40 * 1024 * 1024));
    const root = pageFiles(vfs, 'index.html');
    expect(root['video.mp4']).toBeUndefined();
    expect(root['big.bin']).toBeDefined();
  });

  it('leaves out the pages of a SoftN app written as HTML at the project root, which are the app\'s', () => {
    const vfs = new Vfs();
    vfs.writeFile('/manifest.json', JSON.stringify({ name: 'A', version: '1.0.0', main: 'index.html' }));
    vfs.writeFile('/index.html', '<script src="main.js"></script>\n<App />');
    vfs.writeFile('/pages/about.html', '<Text>About</Text>', { parents: true });
    vfs.writeFile('/assets/help.html', '<h1>Help</h1>', { parents: true });
    vfs.writeFile('/uploads/page.html', '<h1>Upload</h1>', { parents: true });
    vfs.writeFile('/main.js', 'let a = 1\n');
    // assets/ holds what the app shows, and uploads/ the user's files: those are still pages of the project
    expect(findPages(vfs)).toEqual(['assets/help.html', 'uploads/page.html']);
    const app = resolvePage(vfs, 'index.html');
    expect(app.ok).toBe(false);
    expect(!app.ok && app.reason).toContain('/index.html is a page of the SoftN app in the project root, not a web page');
    expect(resolvePage(vfs, 'pages/about.html').ok).toBe(false);
    expect(resolvePage(vfs, 'assets/help.html')).toEqual({ ok: true, path: 'assets/help.html' });
  });
});
