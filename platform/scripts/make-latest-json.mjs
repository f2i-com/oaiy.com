#!/usr/bin/env node
// Write the update feed of a release: latest.json, in the format the Tauri updater reads.
//
//   node platform/scripts/make-latest-json.mjs --dir <release files> --version 0.1.0 \
//        [--tag v0.1.0] [--repo f2i-com/oaiy.com] [--notes-file <file>] [--pub-date <RFC 3339>] [--out <file>] \
//        [--conf platform/desktop/src-tauri/tauri.conf.json]
//
// The desktop asks https://github.com/<repo>/releases/latest/download/latest.json
// whether a newer OAIY exists. Only the installers a desktop can replace itself
// with are in it, each with the signature the release build made beside it:
//
//   windows-x86_64   oaiy-desktop-<v>-windows-x64-setup.exe      (the NSIS installer; never the MSI)
//   linux-x86_64     oaiy-desktop-<v>-linux-x86_64.AppImage
//
// each next to its `<name>.sig` in --dir. The signature in the feed is the CONTENT of
// that file (the updater checks the download against it with the public key in
// tauri.conf.json), and the url is where the release publishes the asset:
// https://github.com/<repo>/releases/download/<tag>/<name>.
//
// A missing installer, a missing or empty or unreadable signature is an error and
// nothing is written: a feed that leaves a platform out would tell that platform's
// desktops there is no update, and the release that carries it is the one FormLogic's
// CI and every desktop read as "latest". The release job runs this before it writes
// SHA256SUMS.txt, so the feed and the signatures are covered by it.
//
// With --conf (the release job passes it) each installer is also CHECKED against its signature
// with the public key in that tauri.conf.json (plugins.updater.pubkey), the key every installed
// OAIY carries. The Tauri CLI only warns when the private key in the Actions secrets is not that
// key's pair, and a release signed with the wrong one installs on nobody: this stops it before
// it is published.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MinisignError, verifyMinisign } from './minisign.mjs';

export const DEFAULT_REPO = 'f2i-com/oaiy.com';

/** N.N.N with no leading zeros, as release.yml's meta job takes it. */
const VERSION = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;
const REPO = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const TAG = /^v?[0-9]+\.[0-9]+\.[0-9]+$/;
const RFC3339 = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;

export class FeedError extends Error {}

/** The platforms the feed carries, by the feed's own key, and the release asset that updates each. */
export function platformAssets(version) {
  return [
    { key: 'windows-x86_64', asset: `oaiy-desktop-${version}-windows-x64-setup.exe`, what: 'the Windows NSIS installer' },
    { key: 'linux-x86_64', asset: `oaiy-desktop-${version}-linux-x86_64.AppImage`, what: 'the Linux AppImage' },
  ];
}

/**
 * A minisign signature file as the Tauri CLI writes it beside an installer: the base64 of
 * the text of the .sig file ("untrusted comment: ...", the signature, "trusted comment: ...",
 * the global signature). Only its shape is checked here; the desktop checks the signature.
 */
export function readSignature(file, what) {
  let raw;
  try {
    raw = fs.readFileSync(file, 'utf8');
  } catch {
    throw new FeedError(`there is no signature for ${what}: ${path.basename(file)} is missing`);
  }
  const content = raw.trim();
  if (!content) throw new FeedError(`the signature for ${what} is empty: ${path.basename(file)}`);
  if (!/^[A-Za-z0-9+/]+={0,2}$/.test(content)) throw new FeedError(`the signature for ${what} is not base64: ${path.basename(file)}`);
  const text = Buffer.from(content, 'base64').toString('utf8');
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!text.startsWith('untrusted comment:') || lines.length < 4) {
    throw new FeedError(`the signature for ${what} is not a minisign signature (${path.basename(file)}): was the installer built with the updater artifacts on?`);
  }
  return content;
}

/** The installer against its signature, with the public key the desktop carries. */
function checkAgainstKey(installer, asset, signature, pubkey) {
  let result;
  try {
    result = verifyMinisign(fs.readFileSync(installer), signature, pubkey);
  } catch (error) {
    if (!(error instanceof MinisignError)) throw error;
    throw new FeedError(`${asset} cannot be checked: ${error.message}`);
  }
  if (!result.ok) {
    throw new FeedError(`the signature of ${asset} does not verify against the public key in tauri.conf.json (${result.reason}): the desktop would refuse this release. Were the Actions secrets TAURI_SIGNING_PRIVATE_KEY and TAURI_SIGNING_PRIVATE_KEY_PASSWORD made for that public key?`);
  }
}

/** The updater's public key in a tauri.conf.json. */
export function readPubkey(confPath) {
  let conf;
  try {
    conf = JSON.parse(fs.readFileSync(confPath, 'utf8'));
  } catch {
    throw new FeedError(`${confPath} cannot be read as tauri.conf.json`);
  }
  const pubkey = conf?.plugins?.updater?.pubkey;
  if (typeof pubkey !== 'string' || !pubkey.trim()) throw new FeedError(`${confPath} has no plugins.updater.pubkey`);
  return pubkey.trim();
}

/**
 * The feed for a release whose files are in `dir`. Throws a FeedError, with a message that says
 * what to fix, when anything is missing; returns the feed object otherwise.
 */
export function buildFeed({ dir, version, tag, repo = DEFAULT_REPO, notes, pubDate, pubkey }) {
  if (!VERSION.test(String(version ?? ''))) throw new FeedError(`"${version ?? ''}" is not a version of the form 0.1.0 (three numbers, none with a leading zero)`);
  const releaseTag = tag ?? `v${version}`;
  if (!TAG.test(releaseTag) || releaseTag.replace(/^v/, '') !== version) throw new FeedError(`the tag "${releaseTag}" is not the version ${version} (or v${version})`);
  if (!REPO.test(repo)) throw new FeedError(`"${repo}" is not a repository of the form owner/name`);
  if (!dir || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) throw new FeedError(`the release folder ${dir ?? ''} does not exist`);
  const date = pubDate ?? new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
  if (!RFC3339.test(date) || Number.isNaN(Date.parse(date))) throw new FeedError(`"${date}" is not an RFC 3339 date`);

  const platforms = {};
  for (const { key, asset, what } of platformAssets(version)) {
    const installer = path.join(dir, asset);
    if (!fs.existsSync(installer) || fs.statSync(installer).size === 0) {
      throw new FeedError(`${what} is missing: ${asset} is not in ${dir}, so ${key} would have no update`);
    }
    const signature = readSignature(`${installer}.sig`, `${asset}`);
    if (pubkey) checkAgainstKey(installer, asset, signature, pubkey);
    platforms[key] = { signature, url: `https://github.com/${repo}/releases/download/${releaseTag}/${asset}` };
  }
  return {
    version,
    notes: notes ?? `OAIY ${version}\n\nWhat changed: https://github.com/${repo}/releases/tag/${releaseTag}`,
    pub_date: date,
    platforms,
  };
}

/** Write the feed; nothing is written unless it is complete. Returns the path written. */
export function writeFeed(options) {
  const feed = buildFeed(options);
  const out = options.out ?? path.join(options.dir, 'latest.json');
  fs.writeFileSync(out, `${JSON.stringify(feed, null, 2)}\n`);
  return out;
}

function parseArgs(argv) {
  const names = new Set(['dir', 'version', 'tag', 'repo', 'notes-file', 'pub-date', 'out', 'conf']);
  const options = {};
  for (let i = 0; i < argv.length; i += 2) {
    const flag = argv[i];
    const name = flag?.startsWith('--') ? flag.slice(2) : '';
    if (!names.has(name) || argv[i + 1] === undefined) throw new FeedError(`unexpected argument "${flag ?? ''}"; ${usage()}`);
    options[name] = argv[i + 1];
  }
  return options;
}

const usage = () => 'usage: make-latest-json.mjs --dir <release files> --version <N.N.N> [--tag <tag>] [--repo <owner/name>] [--notes-file <file>] [--pub-date <RFC 3339>] [--out <file>] [--conf <tauri.conf.json>]';

function main() {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (!args.dir || !args.version) throw new FeedError(usage());
    let notes;
    if (args['notes-file']) {
      try {
        notes = fs.readFileSync(args['notes-file'], 'utf8').trim();
      } catch {
        throw new FeedError(`the notes file ${args['notes-file']} cannot be read`);
      }
    }
    const out = writeFeed({
      dir: path.resolve(args.dir),
      version: args.version,
      tag: args.tag,
      repo: args.repo,
      notes,
      pubDate: args['pub-date'],
      pubkey: args.conf ? readPubkey(path.resolve(args.conf)) : undefined,
      out: args.out ? path.resolve(args.out) : undefined,
    });
    console.log(`wrote ${out}`);
  } catch (error) {
    if (!(error instanceof FeedError)) throw error;
    console.error(`::error::update feed (latest.json) NOT written: ${error.message}`);
    process.exit(1);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
