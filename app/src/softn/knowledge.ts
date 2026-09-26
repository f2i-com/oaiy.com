/**
 * The SoftN reference the agent works from, and a harness around it: a map of
 * what there is, reading by topic, the exact reference for components, a
 * search across all of it, and complete example apps to read or copy.
 *
 * Sources (generated, Apache-2.0, see scripts/generate-softn-knowledge.mjs):
 * SoftN Studio's writing guide and agent knowledge, softn.com's published
 * guides, and apps from softn.com's catalogue. The large parts load on first
 * use; main.ts warms them after start so they are cached for offline use.
 */
import { unzipSync } from 'fflate';
import type { Vfs } from '../vfs/vfs';
import { COMPONENT_INDEX, DOC_INDEX } from './knowledge/index.generated';
import { importSoftn } from './softn';

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');

const components = () => import('./knowledge/components.generated').then((m) => m.COMPONENT_REFERENCE);
const docs = () => import('./knowledge/docs.generated').then((m) => m.DOC_PAGES);
const examples = () => import('./knowledge/examples.generated').then((m) => m.SOFTN_EXAMPLES);

/** Load everything once, so the service worker caches it for offline use. */
export function warmKnowledge(): void {
  void Promise.all([components(), docs(), examples()]).catch(() => {});
}

/** The writing guide's sections: "## Heading" blocks. */
function guideSections(guide: string): Array<{ title: string; text: string }> {
  return guide.split(/\n(?=## )/).map((block) => ({ title: block.split('\n')[0].replace(/^#+\s*/, ''), text: block }));
}

/** What the reference holds and how to reach it: softn_docs with no arguments. */
export async function docsMap(guide: string): Promise<string> {
  const groups = new Map<string, string[]>();
  for (const c of COMPONENT_INDEX) {
    if (!c.registered) continue;
    groups.set(c.category, [...(groups.get(c.category) ?? []), c.name]);
  }
  const list = await examples();
  return [
    'SoftN reference. Start with the writing guide; look up components before using props you are unsure of; read an example app that does something similar; search when you do not know where something is.',
    '',
    '## The writing guide (softn_docs topic "guide", or "guide#<words from a heading>")',
    ...guideSections(guide).filter((s) => s.title && !s.title.startsWith('#')).map((s) => `- ${s.title}`),
    '',
    '## Published guides (softn_docs topic "<slug>" or "<slug>#<section>")',
    ...DOC_INDEX.map((d) => `- ${d.slug}: ${d.title} (sections: ${d.sections.join(', ')})`),
    '',
    '## Components (softn_components names [...] for props, events and an example)',
    ...[...groups].map(([group, names]) => `- ${group}: ${names.join(', ')}`),
    '',
    '## Example apps (softn_examples name "<slug>" to see its files; file to read one; install_to to copy it into the project)',
    ...list.map((e) => `- ${e.slug}: ${e.name}: ${e.description}`),
    '',
    'softn_docs search "<words>" finds a term across the guide, the guides, the components and the example apps.',
  ].join('\n');
}

/** A guide section or a published guide by slug, optionally one section of it. */
export async function readTopic(topic: string, guide: string): Promise<{ ok: boolean; text: string }> {
  const [pagePart, sectionPart = ''] = topic.trim().replace(/^\/+/, '').split('#');
  if (norm(pagePart) === 'guide' || norm(pagePart) === 'writingguide') {
    if (!sectionPart) return { ok: true, text: guide };
    const want = sectionPart.toLowerCase();
    const hits = guideSections(guide).filter((s) => s.title.toLowerCase().includes(want));
    if (hits.length) return { ok: true, text: hits.map((s) => s.text).join('\n\n') };
    return { ok: false, text: `No heading in the writing guide contains "${sectionPart}". Its sections: ${guideSections(guide).map((s) => s.title).join('; ')}` };
  }
  const pages = await docs();
  const page = pages.find((p) => p.slug === pagePart) ?? pages.find((p) => norm(p.slug) === norm(pagePart) || norm(p.title) === norm(pagePart));
  if (!page) return { ok: false, text: `There is no guide "${pagePart}". Topics: "guide" (the writing guide) or one of: ${DOC_INDEX.map((d) => d.slug).join(', ')}. Or search.` };
  const sections = sectionPart ? page.sections.filter((s) => s.id === sectionPart || norm(s.title) === norm(sectionPart) || s.title.toLowerCase().includes(sectionPart.toLowerCase())) : page.sections;
  if (!sections.length) return { ok: false, text: `${page.slug} has no section "${sectionPart}". Its sections: ${page.sections.map((s) => s.id).join(', ')}.` };
  const text = `# ${page.title}\n${page.summary}\n\n${sections.map((s) => `## ${s.title} (#${s.id})\n${s.text}`).join('\n\n')}`;
  return { ok: true, text: text.length > 24_000 ? `${text.slice(0, 24_000)}\n… (cut; read one section with ${page.slug}#<section>)` : text };
}

export async function lookupComponents(names: string[]): Promise<{ ok: boolean; text: string }> {
  const reference = await components();
  const byNorm = new Map(Object.keys(reference).map((n) => [norm(n), n]));
  const found: string[] = [];
  const unknown: string[] = [];
  for (const raw of names.slice(0, 12)) {
    const name = byNorm.get(norm(raw.replace(/[<>/]/g, '')));
    if (name) found.push(reference[name]);
    else unknown.push(raw);
  }
  const parts = [...found];
  if (unknown.length) parts.push(`Not components: ${unknown.join(', ')}. The registered components are: ${COMPONENT_INDEX.filter((c) => c.registered).map((c) => c.name).join(', ')}.`);
  if (names.length > 12) parts.push('Only the first 12 names were looked up; ask again for the rest.');
  return { ok: found.length > 0, text: parts.join('\n\n') };
}

// --- search ------------------------------------------------------------------

interface Doc {
  /** How to open it. */
  ref: string;
  title: string;
  text: string;
}

const textDecoder = new TextDecoder();
let exampleFilesCache: Map<string, Record<string, Uint8Array>> | null = null;
async function exampleFiles(slug: string): Promise<Record<string, Uint8Array> | null> {
  exampleFilesCache ??= new Map();
  const cached = exampleFilesCache.get(slug);
  if (cached) return cached;
  const example = (await examples()).find((e) => e.slug === slug);
  if (!example) return null;
  const files = unzipSync(Uint8Array.from(atob(example.softn), (c) => c.charCodeAt(0)));
  exampleFilesCache.set(slug, files);
  return files;
}

const TEXT_FILE = /\.(ui|logic|py|json|md|txt|xdb|wgsl|css|svg|js)$/i;

async function corpus(guide: string): Promise<Doc[]> {
  const out: Doc[] = [];
  for (const s of guideSections(guide)) out.push({ ref: `softn_docs topic "guide#${s.title}"`, title: `Writing guide: ${s.title}`, text: s.text });
  for (const p of await docs()) for (const s of p.sections) out.push({ ref: `softn_docs topic "${p.slug}#${s.id}"`, title: `${p.title}: ${s.title}`, text: s.text });
  for (const [name, text] of Object.entries(await components())) out.push({ ref: `softn_components names ["${name}"]`, title: `Component ${name}`, text });
  for (const e of await examples()) {
    const files = await exampleFiles(e.slug);
    for (const [path, data] of Object.entries(files ?? {})) {
      if (!TEXT_FILE.test(path)) continue;
      out.push({ ref: `softn_examples name "${e.slug}" file "${path}"`, title: `Example ${e.slug}: ${path}`, text: textDecoder.decode(data) });
    }
  }
  return out;
}

/** A term matches at the start of a word ("play" in "play()", not in "display"). */
function termPattern(term: string): RegExp {
  return new RegExp(`(?:^|[^a-z0-9_])${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'g');
}

function count(haystack: string, pattern: RegExp): number {
  pattern.lastIndex = 0;
  let n = 0;
  while (n < 50 && pattern.exec(haystack)) n++;
  return n;
}

/** Where the words are, best first, each with a snippet and how to open it. */
export async function searchKnowledge(query: string, guide: string, limit = 10): Promise<string> {
  const phrase = query.trim().toLowerCase();
  const terms = [...new Set(phrase.split(/[^\w@:#.<>-]+/).map((t) => t.replace(/^[<]+|[>]+$/g, '')).filter((t) => t.length > 1))];
  if (!terms.length) return 'Give some words to search for.';
  const patterns = terms.map(termPattern);
  const scored: Array<{ doc: Doc; score: number; at: number }> = [];
  for (const doc of await corpus(guide)) {
    const text = doc.text.toLowerCase();
    const title = doc.title.toLowerCase();
    let score = 0;
    let matched = 0;
    for (const pattern of patterns) {
      const inText = count(text, pattern);
      const inTitle = count(title, pattern);
      if (inText || inTitle) matched++;
      score += Math.min(inText, 10) + inTitle * 6;
    }
    if (!matched) continue;
    // Every word present counts for more than one word many times.
    score *= matched / terms.length;
    if (terms.length > 1 && text.includes(phrase)) score += 8;
    const first = patterns.map((p) => text.search(new RegExp(p.source))).filter((i) => i >= 0);
    const at = first.length ? Math.min(...first) : 0;
    scored.push({ doc, score, at });
  }
  scored.sort((a, b) => b.score - a.score);
  const hits = scored.slice(0, limit);
  if (!hits.length) return `Nothing matches "${query}". Try other words, or softn_docs with no arguments for the map.`;
  return [
    `${scored.length} places mention ${terms.map((t) => `"${t}"`).join(', ')}; the best ${hits.length}:`,
    ...hits.map(({ doc, at }, i) => {
      const start = Math.max(0, at - 100);
      const snippet = doc.text.slice(start, start + 260).replace(/\s+/g, ' ').trim();
      return `${i + 1}. ${doc.title}\n   ${start > 0 ? '…' : ''}${snippet}…\n   open: ${doc.ref}`;
    }),
  ].join('\n');
}

// --- examples ----------------------------------------------------------------

export async function listExamples(): Promise<string> {
  const list = await examples();
  return [
    'Complete SoftN apps from softn.com\'s catalogue. name "<slug>" lists an app\'s files with its manifest; add file "<path>" to read one; install_to "<folder>" copies the app into the project (to run it in the preview, or start from it).',
    ...list.map((e) => `- ${e.slug}: ${e.name}: ${e.description} [${e.tags.join(', ')}] (${e.files.length} files)`),
  ].join('\n');
}

export async function describeExample(slug: string, file?: string): Promise<{ ok: boolean; text: string }> {
  const list = await examples();
  const example = list.find((e) => e.slug === slug) ?? list.find((e) => norm(e.slug) === norm(slug) || norm(e.name) === norm(slug));
  if (!example) return { ok: false, text: `There is no example "${slug}". The examples: ${list.map((e) => e.slug).join(', ')}.` };
  const files = (await exampleFiles(example.slug))!;
  if (file) {
    const key = file.replace(/^\/+/, '');
    const data = files[key];
    if (!data) return { ok: false, text: `${example.slug} has no file "${file}". Its files: ${Object.keys(files).filter((n) => !n.endsWith('/')).join(', ')}.` };
    if (!TEXT_FILE.test(key)) return { ok: true, text: `${example.slug}/${key} is a binary file (${data.byteLength.toLocaleString()} bytes); install the example to use it.` };
    const text = textDecoder.decode(data);
    return { ok: true, text: `${example.slug}/${key} (${text.split('\n').length} lines):\n${text.length > 40_000 ? `${text.slice(0, 40_000)}\n… (cut at 40,000 characters; install the example and read the file in parts)` : text}` };
  }
  const manifest = files['manifest.json'] ? textDecoder.decode(files['manifest.json']) : '(none)';
  return {
    ok: true,
    text: [
      `${example.name} (${example.slug}): ${example.description}`,
      'Files:',
      ...Object.entries(files).filter(([n]) => !n.endsWith('/')).map(([n, d]) => `  ${n} (${d.byteLength.toLocaleString()} bytes)`),
      '',
      `manifest.json:\n${manifest}`,
      '',
      `Read a file with softn_examples name "${example.slug}" file "<path>" (start with the manifest's main page and its logic).`,
    ].join('\n'),
  };
}

/** Copy an example into the project; returns the folder. */
export async function installExample(vfs: Vfs, slug: string, parent: string): Promise<{ root: string; files: number; name: string }> {
  const list = await examples();
  const example = list.find((e) => e.slug === slug) ?? list.find((e) => norm(e.slug) === norm(slug) || norm(e.name) === norm(slug));
  if (!example) throw new Error(`There is no example "${slug}". The examples: ${list.map((e) => e.slug).join(', ')}.`);
  return importSoftn(vfs, Uint8Array.from(atob(example.softn), (c) => c.charCodeAt(0)), `${example.slug}.softn`, parent);
}

/** The example apps, for choosing one to start from. */
export async function exampleCatalogue(): Promise<Array<{ slug: string; name: string; description: string; tags: string[] }>> {
  return (await examples()).map(({ slug, name, description, tags }) => ({ slug, name, description, tags }));
}

/** An example app's files, by path inside the app. */
export async function exampleBundle(slug: string): Promise<Array<[string, Uint8Array]>> {
  const files = await exampleFiles(slug);
  if (!files) throw new Error(`There is no example "${slug}".`);
  return Object.entries(files).filter(([path]) => !path.endsWith('/') && !path.split('/').includes('..'));
}
