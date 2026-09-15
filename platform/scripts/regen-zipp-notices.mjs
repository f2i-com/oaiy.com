#!/usr/bin/env node
/**
 * Regenerate the curated ZIPP notices in ui/vendor/zipp-notices/ from zipp.org's
 * sources at a release tag: THIRD_PARTY_LICENSES.txt, for what the web-python
 * engine compiles in from elsewhere, and SOURCE.json, which records that tag
 * and the file's SHA-256 for scripts/fetch-zipp-release.mjs.
 *
 *   node scripts/regen-zipp-notices.mjs <zipp.org checkout> vX.Y.Z           write both
 *   node scripts/regen-zipp-notices.mjs <zipp.org checkout> vX.Y.Z --check   compare them with the ones here
 *
 * The sources are read from the tag with `git show`, never from the checkout's
 * working tree. The recipe is the one softn.com's
 * packages/@softn/core/scripts/build-zipp-wasm.mjs writes a Python build's
 * notices with, so OAIY's copy and Softn's are the same bytes.
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { fileURLToPath } from 'node:url';
import { CURATED_NOTICES, NOTICES, REPOSITORY, isReleaseTag, sha256 } from './fetch-zipp-release.mjs';

/** The zipp.org files the notices are made of, by their paths from the repository root. */
export const NOTICE_SOURCES = { rustpython: 'crates/rustpython-parser-fork/LICENSE', unicode: 'LICENSE-UNICODE' };

/** The notices from those files' texts, each as it is, its own final newline included. */
export const noticesText = ({ rustpython, unicode }) => `${[
  'ZIPP engine: Apache-2.0. See the source repository for its complete notices.',
  `RustPython parser (MIT):\n${rustpython}`,
  `Unicode data:\n${unicode}`,
].join('\n\n').trimEnd()}\n`;

/** What SOURCE.json records for notices generated from `release`. */
export const noticesRecord = (release, text) => ({
  repository: REPOSITORY,
  release,
  sources: Object.values(NOTICE_SOURCES),
  file: NOTICES,
  sha256: sha256(Buffer.from(text)),
});

/**
 * Write the notices of `release` into `dir`, `read(file)` giving a source's
 * text at that tag; with `check`, write nothing and return what differs.
 */
export function regenerate({ read, release, dir = path.dirname(CURATED_NOTICES), check = false }) {
  const text = noticesText({ rustpython: read(NOTICE_SOURCES.rustpython), unicode: read(NOTICE_SOURCES.unicode) });
  const record = noticesRecord(release, text);
  const noticesFile = path.join(dir, NOTICES);
  const recordFile = path.join(dir, 'SOURCE.json');
  if (!check) {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(noticesFile, text);
    fs.writeFileSync(recordFile, `${JSON.stringify(record, null, 2)}\n`);
    return [];
  }
  const problems = [];
  if (!(fs.existsSync(noticesFile) && fs.readFileSync(noticesFile).equals(Buffer.from(text)))) problems.push(`${NOTICES} is not what the ${release} sources give`);
  let recorded;
  try {
    recorded = JSON.parse(fs.readFileSync(recordFile, 'utf8'));
  } catch {}
  if (!isDeepStrictEqual(recorded, record)) problems.push(`SOURCE.json does not record ${release} and that file's SHA-256`);
  return problems;
}

function main(args) {
  const [checkout, release, ...rest] = args;
  if (!checkout || !isReleaseTag(release) || rest.some((a) => a !== '--check')) {
    console.error('Usage: node scripts/regen-zipp-notices.mjs <zipp.org checkout> vX.Y.Z [--check]');
    return 1;
  }
  const read = (file) => execFileSync('git', ['-C', checkout, 'show', `${release}:${file}`], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  let problems;
  try {
    problems = regenerate({ read, release, check: rest.includes('--check') });
  } catch (error) {
    console.error(`regen-zipp-notices: cannot read ${release} in ${checkout}: ${String(error.stderr || error.message).trim()}`);
    return 1;
  }
  const where = (path.relative(process.cwd(), path.dirname(CURATED_NOTICES)) || '.').replaceAll('\\', '/');
  if (problems.length) {
    console.error(`regen-zipp-notices: ${where} is not the ${release} notices:\n  - ${problems.join('\n  - ')}`);
    return 1;
  }
  console.log(`${where}: the ${release} notices${rest.length ? ' check' : ' written'}.`);
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = main(process.argv.slice(2));
