/**
 * Put bot.computer's bridge (bot-bridge.js, beside this file) and its
 * screenshot library (modern-screenshot's browser build, as bot-capture.js)
 * into a preview frame's folder: the installed SoftN runtime, and the web page
 * preview (public/webpage/). In the SoftN runtime they are also loaded from
 * its index.html, before the runtime's own scripts. Idempotent; run by
 * scripts/fetch-softn.mjs after an install and by vite.config.ts on every dev
 * server and build, so an existing install gets them too.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
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

/** Returns false when there is no runtime installed in `dir`. */
export function installBridge(dir) {
  const index = join(dir, 'index.html');
  if (!existsSync(index)) return false;
  copyBridge(dir);
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
