/**
 * No AI model's name in what the flow editor ships as words.
 *
 *     npm run test:model-names          (LIST=1 prints every string looked at that matched, allowed or not)
 *
 * The rule: a model or a model family is never written into the editor: not as a label, a placeholder, an example,
 * a default or an option. Names change faster than the editor ships, and a name written here is a model somebody's
 * server does not have (the welcome wizard's starter flow asked Ollama for one nobody had chosen, and its first run
 * failed). A model's name comes from the provider's own list, and where there is no list the field is empty and
 * says what to type.
 *
 * What is looked at (the words a person can meet, and the values the editor fills in for them):
 *   - src/**: every string literal, template string and piece of JSX text in a .ts/.tsx file (read with the
 *     TypeScript parser, so comments are not looked at: they ship to nobody), and every string VALUE in a .json
 *     file (the bundled modules' node and module definitions: labels, descriptions, placeholders, defaults, options);
 *   - the pages: index.html, app.html, desktop.html (their text and attributes, tags and scripts taken out);
 *   - ../examples/flows/*.flow.json: a flow's `name` and `description`, which the Flows rail shows, and every node's
 *     `label`, which its card shows. What a node needs in order to run stays inside it and is not looked at: the
 *     service it calls (`engine:image:<the model's id>`), and the prompts and code written for that model. The
 *     flow's `notes` (what it was made with, step by step) are its author's own words, as a saved flow's are.
 * What is not: vendor/ (the engine's own source, vendored), tests/, generated files and node_modules.
 *
 * Providers and servers are not models: Ollama, LM Studio, llama.cpp, llama-server, vLLM, ComfyUI, whisper.cpp,
 * OpenAI, Anthropic, Google, OpenRouter, Groq. The patterns below are built so those do not match.
 *
 * ALLOWED lists the few places where code must tell one provider's models apart to talk to them at all (a request
 * shape that depends on the model's name). Each says why, so it can be checked again rather than trusted.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const rel = (file) => path.relative(UI, file).replaceAll('\\', '/');

/** Model families and versioned model names. Case does not matter. */
export const MODEL_NAMES = [
  ['GPT', /\bgpt[- ]?(?:\d|image|4o)|\bgpt\b(?=[- ]?image)|\bGPT Image\b/i],
  ['DALL-E', /\bdall[- ]?e\b/i],
  ['Claude', /\bclaude\b|\b(?:sonnet|opus|haiku)[- ]?\d|\bclaude-(?:sonnet|opus|haiku)/i],
  ['Gemini (a named model)', /\bgemini[- ]?\d|\bgemini[- ](?:pro|flash|ultra|nano)\b/i],
  ['Gemma', /\bgemma\b/i],
  // "Ollama", "llama.cpp" and "llama-server" are servers; "llama3", "Llama 3.2" and "Llama" alone are models.
  ['Llama', /(?<![\w.-])llama[- ]?\d|(?<![\w.-])llama(?![\w]|[.-]?cpp|-server)/i],
  ['Qwen', /\bqwen/i],
  ['Mistral (a named model)', /\bmistral[- ](?:\d|small|medium|large|nemo|tiny)|\bmixtral\b/i],
  ['FLUX', /\bflux\b/i],
  ['MOSS', /\bmoss[- ]?(?:image|tts)\b/i],
  ['MiniCPM', /\bminicpm/i],
  ['LTX', /\bltx(?!2-text-to-video)/i],
  ['Stable Diffusion', /\bstable[- ]diffusion\b|\bsdxl\b|\bsd\s?[1-3]\.\d\b/i],
  ['Whisper', /\bwhisper(?![.-]?cpp)\b/i],
  ['GLM', /\bglm\b/i],
  ['Kokoro', /\bkokoro\b/i],
  ['Phi', /\bphi-?\d/i],
  ['DeepSeek (a named model)', /\bdeepseek-/i],
  // ("Wan2GP" is a server program, like ComfyUI; "Wan 2.1" is a model)
  ['Wan', /\bwan[ -]?\d(?!gp)/i],
  ['Hunyuan', /\bhunyuan/i],
  ['Z-Image', /\bz-image\b/i],
  ['LLaVA', /\bllava\b/i],
  // (the cut-out and the upscale nodes named the models OAIY's engine runs them on)
  ['BiRefNet', /\bbirefnet/i],
  ['ESRGAN', /esrgan/i],
  ['a placeholder model', /\blocal-model\b/i],
];

/**
 * Strings that hold a model's name because the code cannot work without telling that model apart. `file` is relative
 * to platform/ui; `match` is what the string must contain; `why` is what breaks without it.
 */
const ALLOWED = [
  {
    file: 'src/utils/NodeDefaults.ts',
    match: 'llama3.2',
    why: 'the model earlier versions wrote into every AI node they added (SEEDED_MODEL_BEFORE): recognised so that it can be taken out of a flow saved back then (withoutProviderSeeds, held by tests/in-oaiy.mjs); no node is given it now',
  },
];

/**
 * Names that are still there, known and counted: [file, family, how many, why it is left]. The check fails when a
 * file holds MORE of a family than is written here (a new name), and when it holds fewer (the entry is stale: lower
 * it or take it out), so the list can only shrink.
 *
 * Every one of them is an id that a flow saved before today carries, read by code and never offered as a choice:
 * nothing here is a label, a placeholder, an example, a default or an option. A node whose flow carries one shows it
 * as "<id> (saved in this flow)". They stay because a saved flow names them: renaming an id would break that flow, or
 * need every saved flow rewritten (on this computer, on OAIY Desktop, and in files people keep).
 */
const LEFT = [
  ['src/bundled-modules/core-image/runtime.ts', 'Gemini (a named model)', 3, 'the three picture formats a flow saved before may name (`case` labels): each is run with the address saved beside it. The node offers one "Google (Gemini API)" format now, whose model is the node\'s Model field'],
  ['src/bundled-modules/core-audio/nodes/text_to_speech.json', 'Qwen', 1, 'the saved id of the speech preset for a server on port 8772 (an option\'s value; its label is "Speech server on this computer (port 8772)")'],
  ['src/bundled-modules/core-audio/ui/TextToSpeechNode.tsx', 'Qwen', 1, 'the same preset\'s id, in the node (its label names no model)'],
  ['src/bundled-modules/core-audio/compiler.ts', 'Qwen', 1, 'the same id, where a flow saved with it is given that server\'s address'],
  ['src/lib/nodeAvailability.ts', 'Qwen', 1, 'the same id, in the list of values that are a node\'s own request and not a service'],
  ['src/bundled-modules/core-ai/runtime.ts', 'Claude', 1, 'an address is taken for Anthropic\'s API where it contains this word (a proxy\'s path): what decides the shape of the request, not a model anyone is offered'],
];

const SKIP_DIRS = new Set(['node_modules', 'vendor', 'tests', 'dist', 'public', '.vite']);
function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (!SKIP_DIRS.has(entry.name)) walk(path.join(dir, entry.name), out);
    } else out.push(path.join(dir, entry.name));
  }
  return out;
}

/** Every string a .ts/.tsx file holds outside its comments: literals, template pieces and JSX text, with their lines. */
function stringsOfSource(file) {
  const text = fs.readFileSync(file, 'utf8');
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, file.endsWith('x') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  const found = [];
  const visit = (node) => {
    // (an import's path is a file's name, not words)
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier) {
      for (const child of node.getChildren(source)) if (child !== node.moduleSpecifier) visit(child);
      return;
    }
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node) || ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node) || ts.isJsxText(node)) {
      const value = ts.isJsxText(node) ? node.getText(source) : node.text;
      if (value.trim()) found.push({ value, line: source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1 });
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return found;
}

/** Every string value in a JSON document, with the path to it. */
function stringsOfJson(value, at = '', out = []) {
  if (typeof value === 'string') out.push({ value, line: at });
  else if (Array.isArray(value)) value.forEach((item, i) => stringsOfJson(item, `${at}[${i}]`, out));
  else if (value && typeof value === 'object') for (const [key, item] of Object.entries(value)) stringsOfJson(item, at ? `${at}.${key}` : key, out);
  return out;
}

/** A page's words: its text and attribute values, without its scripts, styles and tags. */
function stringsOfHtml(file) {
  const text = fs.readFileSync(file, 'utf8').replace(/<script[\s\S]*?<\/script>/gi, ' ').replace(/<style[\s\S]*?<\/style>/gi, ' ').replace(/<!--[\s\S]*?-->/g, ' ');
  return text.split('\n').map((value, i) => ({ value, line: i + 1 })).filter((s) => s.value.trim());
}

const sources = [];
for (const file of walk(path.join(UI, 'src'))) {
  if (/\.d\.ts$/.test(file) || /\.generated\./.test(file)) continue;
  if (/\.tsx?$/.test(file)) sources.push({ file: rel(file), strings: stringsOfSource(file) });
  else if (/\.json$/.test(file)) sources.push({ file: rel(file), strings: stringsOfJson(JSON.parse(fs.readFileSync(file, 'utf8'))) });
}
for (const page of ['index.html', 'app.html', 'desktop.html']) {
  const file = path.join(UI, page);
  if (fs.existsSync(file)) sources.push({ file: page, strings: stringsOfHtml(file) });
}
const EXAMPLES = path.resolve(UI, '..', 'examples', 'flows');
const exampleFlows = fs.existsSync(EXAMPLES) ? fs.readdirSync(EXAMPLES).filter((f) => f.endsWith('.flow.json')).sort() : [];
for (const name of exampleFlows) {
  const flow = JSON.parse(fs.readFileSync(path.join(EXAMPLES, name), 'utf8'));
  const labels = (flow.nodes ?? []).map((node) => ({ value: String(node?.data?.label ?? ''), line: `node ${node?.id}: label` })).filter((s) => s.value);
  sources.push({ file: `../examples/flows/${name}`, strings: [{ value: String(flow.name ?? ''), line: 'name' }, { value: String(flow.description ?? ''), line: 'description' }, ...labels] });
}

const problems = [];
const allowed = [];
/** `${file}\n${family}` → the hits there, for the places LEFT counts. */
const counted = new Map();
const leftKey = (file, family) => `${file}\n${family}`;
const leftFor = new Map(LEFT.map(([file, family, count, why]) => [leftKey(file, family), { count, why }]));
let looked = 0;
for (const { file, strings } of sources) {
  for (const { value, line } of strings) {
    looked++;
    for (const [family, pattern] of MODEL_NAMES) {
      const hit = pattern.exec(value);
      if (!hit) continue;
      const pass = ALLOWED.find((a) => a.file === file && value.includes(a.match));
      const said = `${file}:${line}  ${family}: "${hit[0]}" in ${JSON.stringify(value.replace(/\s+/g, ' ').trim().slice(0, 150))}`;
      if (pass) allowed.push(`${said}\n      allowed: ${pass.why}`);
      else if (leftFor.has(leftKey(file, family))) counted.set(leftKey(file, family), [...(counted.get(leftKey(file, family)) ?? []), said]);
      else problems.push(said);
      break;
    }
  }
}
// What is left is counted: more than written is a new name, fewer is a stale entry.
let left = 0;
for (const [file, family, count, why] of LEFT) {
  const hits = counted.get(leftKey(file, family)) ?? [];
  left += hits.length;
  if (hits.length > count) problems.push(`${file} holds ${hits.length} of ${family}, ${hits.length - count} more than the ${count} left there (${why}):\n      ${hits.join('\n      ')}`);
  else if (hits.length < count) problems.push(`LEFT says ${file} holds ${count} of ${family}; it holds ${hits.length} now: lower the count${hits.length ? '' : ' (or take the entry out)'}`);
  if (process.env.LIST) for (const hit of hits) console.log(`  · ${hit}\n      left: ${why}`);
}
// The places held to none at all, whatever LEFT says of the rest.
const NONE_AT_ALL = ['src/components/wizards/', 'src/lib/servicePresets.ts', 'src/bundled-modules/core-ai/ui/', 'src/bundled-modules/core-ai/nodes/', 'src/bundled-modules/core-ai/module.json', 'src/utils/', '../examples/flows/'];
for (const [file, family] of LEFT) {
  if (NONE_AT_ALL.some((held) => file.startsWith(held))) problems.push(`LEFT may not list ${file} (${family}): the wizard, the presets, the AI node, the defaults and the examples hold no model name`);
}

// The example flows' descriptions are one line each (the rail shows them): no steps, counts or settings.
for (const name of exampleFlows) {
  const flow = JSON.parse(fs.readFileSync(path.join(EXAMPLES, name), 'utf8'));
  const description = String(flow.description ?? '');
  if (!description.trim()) problems.push(`../examples/flows/${name}: no description for the rail`);
  else if (description.length > 90 || /\n/.test(description)) problems.push(`../examples/flows/${name}: its description is ${description.length} characters; the rail shows one line (90 at most)`);
}

// Every allowance still has something to allow (a stale one hides the next real name).
for (const a of ALLOWED) {
  if (!allowed.some((line) => line.startsWith(`${a.file}:`) && line.includes(a.match))) problems.push(`ALLOWED lists ${a.file} "${a.match}", and nothing there matches any more: remove the entry`);
}

if (process.env.LIST) for (const line of allowed) console.log(`  ~ ${line}`);
console.log(`\n${'-'.repeat(60)}`);
if (problems.length) {
  console.log(`model names: ${problems.length} in the editor's words (of ${looked} strings in ${sources.length} files)\n`);
  for (const p of problems) console.log(`  - ${p}`);
  process.exit(1);
}
console.log(`model names: no new one in the editor's words (${looked} strings in ${sources.length} files); ${left} known ones left in ${LEFT.length} places, counted and each with its reason (LIST=1 prints them); ${allowed.length} allowed`);
