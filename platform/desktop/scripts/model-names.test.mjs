// The product rule: the dashboard writes no AI model's name, and no model family's, into its pages: not as a
// default, not in help, not as an example ("for example: …"). A model's name on the page is one the engine or a
// provider reported (the catalogue's, a file's, a provider's own list). What a model can do is said from what the
// engine reports of it (`agentTools` in the catalogue; `tools` for a file the person added), in words that name none
// (src/SetupParts.tsx `addedFileLine`, held by src/EngineModelCard.test.tsx).
//
// Scope: every source file of the dashboard that is not a test (`src/**/*.ts`, `.tsx` and `.css`, and `index.html`),
// read whole, comments included, so that an example in a comment cannot become one on the page later; and the
// descriptions the installers show (`src-tauri/tauri.conf.json`). Tests are left out: their fixtures stand in for
// what an engine or a provider would report, and may carry any name.
//
// Four things that are not a model's name on the page are let through, each by its exact words:
//   "Ollama" and "llama.cpp"   programs that serve models, named as providers are
//   ".opus"                    a kind of sound file a voice sample may be
//   id: 'gemini'               the id a provider preset is saved under (people's settings hold it); its name on the
//                              page is "Google AI Studio"
//
// Here, beside the other plain-JS checks, because it reads files: the TypeScript build covers `src` only and has no
// declarations for Node's modules.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const src = path.join(desktop, 'src');

/** Model families and well-known models, as they would be written in a sentence, an id or a file name. */
const FAMILIES = [
  ['Qwen', /qwen/i],
  ['Llama', /(?<![a-z])llama/i],
  ['Mistral', /mi[sx]tral/i],
  ['Gemma', /gemma/i],
  ['GLM', /\bglm\b/i],
  ['FLUX', /flux/i],
  ['GPT', /\bgpt\b|\bgpt-?\d/i],
  ['Gemini', /gemini/i],
  ['Claude', /claude|\bsonnet\b|\bopus\b|\bhaiku\b/i],
  ['DeepSeek', /deepseek/i],
  ['Phi', /\bphi-?\d/i],
  ['MiniCPM', /minicpm/i],
  ['MOSS', /\bmoss\b/i],
  ['LTX', /\bltx\b/i],
  ['Stable Diffusion', /sdxl|stable diffusion/i],
  ['Whisper', /whisper/i],
  ['Kokoro', /kokoro/i],
  ['MiVOLO', /mivolo/i],
  ['YuNet', /yunet/i],
  ['Real-ESRGAN', /esrgan/i],
  ['BiRefNet', /birefnet/i],
  ['Pixal3D', /pixal3d/i],
  ['DINOv3', /dinov\d/i],
  ['Flash-Next', /flash-next/i],
  ['Grok', /\bgrok\b/i],
  ['Codestral', /codestral/i],
  ['Hunyuan', /hunyuan/i],
];

/** Words that hold a family's letters and are not a model's name on the page (see the list at the top). */
const NOT_MODELS = /ollama|llama\.cpp|\.opus\b|id: 'gemini'/gi;

function sources(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...sources(file));
    else if (/\.(tsx?|css)$/.test(entry.name) && !/\.test\.tsx?$|\.testTransport\.ts$|^test-setup\.ts$/.test(entry.name)) out.push(file);
  }
  return out;
}

/** Every place a family is named in `text`, as "line 12: Qwen in «…»". */
function namedModels(text) {
  const found = [];
  text.split(/\r?\n/).forEach((line, i) => {
    const clean = line.replace(NOT_MODELS, '');
    for (const [family, pattern] of FAMILIES) if (pattern.test(clean)) found.push(`line ${i + 1}: ${family} in «${line.trim().slice(0, 140)}»`);
  });
  return found;
}

describe('no model is named in the dashboard', () => {
  const files = [...sources(src), path.join(desktop, 'index.html')];
  const name = (file) => path.relative(desktop, file).replace(/\\/g, '/');

  it('reads the dashboard’s own source (and finds it)', () => {
    const names = files.map(name);
    expect(names.length).toBeGreaterThan(60);
    for (const must of ['src/App.tsx', 'src/SetupParts.tsx', 'src/AgentProviderPicker.tsx', 'src/SettingsPanel.tsx', 'src/api.ts', 'src/styles.css', 'index.html']) expect(names).toContain(must);
    expect(names.some((n) => n.includes('.test.'))).toBe(false);
  });

  it('names no model and no model family in any source file', () => {
    const named = files.flatMap((file) => namedModels(fs.readFileSync(file, 'utf8')).map((where) => `${name(file)} ${where}`));
    expect(named).toEqual([]);
  });

  it('names none in what the installers say OAIY is', () => {
    const { bundle } = JSON.parse(fs.readFileSync(path.join(desktop, 'src-tauri', 'tauri.conf.json'), 'utf8'));
    expect(namedModels(`${bundle.shortDescription}\n${bundle.longDescription}`)).toEqual([]);
    // (and they describe the product as it is: not a tool for developers, nor a helper for another app)
    expect(bundle.category).not.toBe('DeveloperTool');
    expect(`${bundle.shortDescription} ${bundle.longDescription}`).not.toMatch(/oaiy-web|ComfyUI/);
  });

  it('would catch one (the check is not blind), and lets through the words that are not models', () => {
    expect(namedModels('help: "for example: ollama pull qwen3.5"')).toHaveLength(1);
    expect(namedModels('The engine runs Llama, Mistral and Gemma 3 models')).toHaveLength(3);
    expect(namedModels('<option>gpt-5.4-mini</option>')).toHaveLength(1);
    expect(namedModels("{ id: 'openai', name: 'Google Gemini' }")).toHaveLength(1);
    expect(namedModels('LM Studio, Ollama, or a llama.cpp server; accept=".mp3,.opus"; sign in with ChatGPT')).toEqual([]);
    expect(namedModels("{ id: 'gemini', name: 'Google AI Studio' }")).toEqual([]);
  });
});
