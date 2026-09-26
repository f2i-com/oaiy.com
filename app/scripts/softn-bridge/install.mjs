/**
 * Put bot.computer's bridge (bot-bridge.js, beside this file) into an
 * installed SoftN runtime: copy it next to the runtime's index.html and load
 * it there before the runtime's own scripts. Idempotent; run by
 * scripts/fetch-softn.mjs after an install and by vite.config.ts on every dev
 * server and build, so an existing install gets it too.
 */
import { copyFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const TAG = '<script src="./bot-bridge.js" data-bot-computer-bridge></script>';

/** Returns false when there is no runtime installed in `dir`. */
export function installBridge(dir) {
  const index = join(dir, 'index.html');
  if (!existsSync(index)) return false;
  copyFileSync(join(dirname(fileURLToPath(import.meta.url)), 'bot-bridge.js'), join(dir, 'bot-bridge.js'));
  const html = readFileSync(index, 'utf8');
  if (html.includes('data-bot-computer-bridge')) return true;
  // A classic script in <head> runs before the runtime's module scripts (and
  // before the CSP the runtime adds for the app).
  const at = html.indexOf('<script');
  const patched = at < 0 ? html.replace('</head>', `${TAG}</head>`) : `${html.slice(0, at)}${TAG}\n  ${html.slice(at)}`;
  writeFileSync(index, patched);
  return true;
}
