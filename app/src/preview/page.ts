/**
 * Web pages in a project: which .html files the preview can show, and the
 * files one of them gets. The page itself is built in the preview frame
 * (public/webpage/page-host.js), with its JavaScript on the Zipp VM.
 */
import { findApps } from '../softn/softn';
import type { Vfs } from '../vfs/vfs';

/** Folders whose files are nobody's page. */
const IGNORED = new Set(['node_modules', '.git', 'dist', 'build', '.cache']);
/** A page's own folder is sent (large media only up to a point); files elsewhere only while they are small. */
const OWN_FILE_MAX = 32 * 1024 * 1024;
const OWN_TOTAL_MAX = 160 * 1024 * 1024;
const OTHER_FILE_MAX = 2 * 1024 * 1024;
const OTHER_TOTAL_MAX = 48 * 1024 * 1024;
/** What a page is made of is always sent, whatever its size. */
const PAGE_PART = /\.(html?|css|m?js|json|svg|txt|md|csv|xml|woff2?|ttf|otf)$/i;

export const isPagePath = (path: string): boolean => /\.html?$/i.test(path);

/** The .html pages of the project (outside SoftN apps), index pages first. */
export function findPages(vfs: Vfs): string[] {
  const apps = findApps(vfs).filter(Boolean);
  const pages: string[] = [];
  for (const entry of vfs.walk('/', { limit: 20_000 }).entries) {
    if (entry.type !== 'file' || !isPagePath(entry.path)) continue;
    const path = entry.path.replace(/^\/+/, '');
    const segs = path.split('/');
    if (segs.slice(0, -1).some((s) => IGNORED.has(s) || s.startsWith('.'))) continue;
    if (apps.some((root) => path.startsWith(`${root}/`))) continue;
    pages.push(path);
  }
  const rank = (p: string) => (/(^|\/)index\.html?$/i.test(p) ? 0 : 1);
  return pages.sort((a, b) => a.split('/').length - b.split('/').length || rank(a) - rank(b) || a.localeCompare(b)).slice(0, 200);
}

/** A page path as the agent or the person wrote it: "site/index.html", "/site/", or a folder with an index.html. */
export function resolvePage(vfs: Vfs, asked: string | undefined, pages = findPages(vfs)): { ok: true; path: string } | { ok: false; reason: string } {
  const wanted = (asked ?? '').trim().replace(/^\/+/, '').replace(/\/+$/, '');
  if (!wanted) {
    if (pages.length === 1) return { ok: true, path: pages[0] };
    if (!pages.length) return { ok: false, reason: 'the project has no .html page yet' };
    const index = pages.filter((p) => /^index\.html?$/i.test(p));
    if (index.length === 1) return { ok: true, path: index[0] };
    return { ok: false, reason: `the project has ${pages.length} pages; name one with \`path\`: ${pages.slice(0, 12).join(', ')}` };
  }
  if (isPagePath(wanted) && vfs.exists(`/${wanted}`)) return { ok: true, path: wanted };
  for (const name of ['index.html', 'index.htm']) {
    const inside = wanted ? `${wanted}/${name}` : name;
    if (vfs.exists(`/${inside}`)) return { ok: true, path: inside };
  }
  return { ok: false, reason: `/${wanted} is not an .html page of the project${pages.length ? ` (pages: ${pages.slice(0, 12).join(', ')})` : ''}` };
}

/**
 * The project's files for a page, keyed by project path: everything in the
 * page's folder, and what is small elsewhere (a page may reach up with ../).
 */
export function pageFiles(vfs: Vfs, page: string): Record<string, Uint8Array> {
  const folder = page.includes('/') ? `${page.slice(0, page.lastIndexOf('/'))}/` : '';
  const out: Record<string, Uint8Array> = {};
  let others = 0;
  let own = 0;
  for (const [raw, data] of vfs.files()) {
    const path = raw.replace(/^\/+/, '');
    const segs = path.split('/');
    if (segs.slice(0, -1).some((s) => IGNORED.has(s))) continue;
    if (!folder || path.startsWith(folder)) {
      if (!PAGE_PART.test(path) && (data.byteLength > OWN_FILE_MAX || own + data.byteLength > OWN_TOTAL_MAX)) continue;
      if (!PAGE_PART.test(path)) own += data.byteLength;
      out[path] = data;
      continue;
    }
    if (data.byteLength > OTHER_FILE_MAX || others + data.byteLength > OTHER_TOTAL_MAX) continue;
    others += data.byteLength;
    out[path] = data;
  }
  return out;
}

/** A page for people: its project path. */
export const pageLabel = (path: string): string => `/${path}`;
