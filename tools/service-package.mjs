#!/usr/bin/env node
/**
 * service-package.mjs — edit/test the code inside a self-contained service
 * template (a `templates/<id>.json` whose `files` map holds its scripts inline).
 *
 * The self-contained JSON is the shipped/importable artifact, but editing code
 * as escaped JSON strings is miserable. This is the sanctioned workflow:
 *
 *   node tools/service-package.mjs extract <template.json> <srcdir>
 *       → writes each files{} entry to <srcdir>/<name> as a real, editable file,
 *         plus the rest of the template to <srcdir>/_template.json
 *
 *   …edit the real files in <srcdir>…
 *
 *   node tools/service-package.mjs embed <srcdir> <template.json>
 *       → re-bundles <srcdir>/_template.json + the sibling files back into the
 *         self-contained <template.json> (.sh normalized to LF)
 *
 * extract→embed with no edits is byte-stable (no spurious diff). Tests use the
 * same extract step to load the shipped code (tests/krea2/conftest.py).
 */
import fs from 'node:fs';
import path from 'node:path';

const [, , cmd, a, b] = process.argv;

function die(msg) {
  process.stderr.write(msg + '\n');
  process.exit(2);
}
function usage() {
  die('usage:\n  service-package.mjs extract <template.json> <srcdir>\n  service-package.mjs embed   <srcdir> <template.json>');
}

if (cmd === 'extract') {
  if (!a || !b) usage();
  const tpl = JSON.parse(fs.readFileSync(a, 'utf8'));
  const files = tpl.files || {};
  fs.mkdirSync(b, { recursive: true });
  for (const [name, body] of Object.entries(files)) {
    if (name.includes('/') || name.includes('\\')) die(`refusing to extract path-y file name: ${name}`);
    fs.writeFileSync(path.join(b, name), body);
  }
  const meta = { ...tpl };
  delete meta.files;
  fs.writeFileSync(path.join(b, '_template.json'), JSON.stringify(meta, null, 2) + '\n');
  process.stdout.write(`extracted ${Object.keys(files).length} files + _template.json → ${b}\n`);
} else if (cmd === 'embed') {
  if (!a || !b) usage();
  const meta = JSON.parse(fs.readFileSync(path.join(a, '_template.json'), 'utf8'));
  const files = {};
  for (const name of fs.readdirSync(a).sort()) {
    if (name.startsWith('_')) continue; // _template.json + any scratch
    const full = path.join(a, name);
    if (!fs.statSync(full).isFile()) continue;
    let body = fs.readFileSync(full, 'utf8');
    if (name.endsWith('.sh')) body = body.replace(/\r\n/g, '\n'); // POSIX scripts stay LF
    files[name] = body;
  }
  const tpl = { ...meta, files }; // files last, matching the on-disk template
  fs.writeFileSync(b, JSON.stringify(tpl, null, 2) + '\n');
  process.stdout.write(`embedded ${Object.keys(files).length} files → ${b}\n`);
} else {
  usage();
}
