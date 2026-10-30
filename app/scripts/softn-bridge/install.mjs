/**
 * Put OAIY's bridge (bot-bridge.js, beside this file) and its
 * screenshot library (modern-screenshot's browser build, as bot-capture.js)
 * into a preview frame's folder: the installed SoftN runtime, and the web page
 * preview (public/webpage/). In the SoftN runtime they are also loaded from
 * its index.html, before the runtime's own scripts. Idempotent; run by
 * scripts/fetch-softn.mjs after an install and by vite.config.ts on every dev
 * server and build, so an existing install gets them too.
 */
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const TAGS = [
  ['data-bot-computer-bridge', '<script src="./bot-bridge.js" data-bot-computer-bridge></script>'],
  ['data-bot-computer-capture', '<script src="./bot-capture.js" data-bot-computer-capture></script>'],
];

/** modern-screenshot's UMD build (window.modernScreenshot), from node_modules. */
function captureLibrary() {
  const require = createRequire(import.meta.url);
  return join(dirname(require.resolve('modern-screenshot/package.json')), 'dist', 'index.js');
}

/** Writes `to` only when it would change: a dev server reloads every open page for a rewritten file. */
function update(to, content) {
  if (existsSync(to) && readFileSync(to).equals(content)) return;
  writeFileSync(to, content);
}

/** Copies the bridge and the capture library into `dir` (no index.html is touched). */
export function copyBridge(dir) {
  update(join(dir, 'bot-bridge.js'), readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'bot-bridge.js')));
  try {
    update(join(dir, 'bot-capture.js'), readFileSync(captureLibrary()));
  } catch {
    /* not installed yet (npm install): screenshots say so */
  }
}

/**
 * The hosted runtime makes asset() URLs only for images, sounds and fonts, as
 * data: URLs, so asset("assets/x.glb") comes back empty and a Scene3D model
 * never loads (and Scene3D refuses a model's data: URL anyway). In the file
 * that receives the app, these edits add glTF models to the runtime's list of
 * asset types and make their URLs blob: URLs, which Scene3D accepts. Each is
 * [the runtime's code, as a pattern that takes any of its minifier's names, what
 * it becomes, and a piece of the edit that says it is made]; the code must be
 * found once, and a release that words it otherwise is left as it is, and said so
 * (v0.0.20 named a variable `s` that v0.0.19 named `c`).
 *
 * From SoftN v0.0.28 the runtime serves glTF, GLB, OBJ, STL and FBX itself, and
 * from v0.0.30 Scene3D loads the data: URLs it gives them, so a runtime that
 * names its own glTF type is left as it is: these edits are for older ones.
 */
const ID = '[\\w$]+';
const MODEL_ASSETS = [
  [/woff2:`font\/woff2`\}/g, () => 'woff2:`font/woff2`,glb:`model/gltf-binary`,gltf:`model/gltf+json`}', 'glb:`model/gltf-binary`'],
  [
    new RegExp(`(${ID})\\.set\\((${ID}),\`data:\\$\\{(${ID})\\};base64,\\$\\{(${ID})\\}\`\\)`, 'g'),
    (_, map, key, type, b64) => `${map}.set(${key},/^model\\//.test(${type})?URL.createObjectURL(new Blob([Uint8Array.from(atob(${b64}),c=>c.charCodeAt(0))],{type:${type}})):\`data:\${${type}};base64,\${${b64}}\`)`,
    'URL.createObjectURL(new Blob([Uint8Array.from(atob(',
  ],
];

/** Lets a SoftN app's Scene3D load a 3D model from its assets/ folder (see MODEL_ASSETS). */
function allowModelAssets(dir) {
  const assets = join(dir, 'assets');
  if (!existsSync(assets)) return;
  for (const name of readdirSync(assets)) {
    if (!name.endsWith('.js')) continue;
    const file = join(assets, name);
    const code = readFileSync(file, 'utf8');
    if (!code.includes('formlogic:init') || MODEL_ASSETS.every(([, , made]) => code.includes(made))) continue;
    // The runtime serves models itself (SoftN v0.0.28 on; see MODEL_ASSETS).
    if (/glb:\s*["'`]model\/gltf-binary["'`]/.test(code)) continue;
    if (!MODEL_ASSETS.every(([from]) => (code.match(from) ?? []).length === 1)) {
      console.warn(`softn-bridge: ${name} does not read as the edits for 3D models expect: a SoftN app's Scene3D cannot load a model from its assets in the preview`);
      continue;
    }
    writeFileSync(file, MODEL_ASSETS.reduce((text, [from, to]) => text.replace(from, to), code));
  }
}

/** Returns false when there is no runtime installed in `dir`. */
export function installBridge(dir) {
  const index = join(dir, 'index.html');
  if (!existsSync(index)) return false;
  copyBridge(dir);
  allowModelAssets(dir);
  const original = readFileSync(index, 'utf8');
  let html = original;
  for (const [mark, tag] of TAGS) {
    if (html.includes(mark)) continue;
    // A classic script in <head> runs before the runtime's module scripts (and
    // before the CSP the runtime adds for the app).
    const at = html.indexOf('<script');
    html = at < 0 ? html.replace('</head>', `${tag}</head>`) : `${html.slice(0, at)}${tag}\n  ${html.slice(at)}`;
  }
  if (html !== original) writeFileSync(index, html);
  return true;
}
