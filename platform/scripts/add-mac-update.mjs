#!/usr/bin/env node
// Add the Mac to a published release's update feed: latest.json with a darwin-aarch64 entry, and SHA256SUMS.txt made true again.
//
//   node platform/scripts/add-mac-update.mjs --feed <latest.json> --sums <SHA256SUMS.txt> --dir <the Mac's files> \
//        --version 0.1.9 [--tag v0.1.9] [--repo f2i-com/oaiy.com] --image-sha256 <the image's digest> --out <folder> \
//        [--conf platform/desktop/src-tauri/tauri.conf.json]
//
// The release job writes the feed (make-latest-json.mjs) before a Mac image exists; the Mac update workflow
// (.github/workflows/mac-update.yml) packs the app out of the notarised image afterwards, signs it, and runs this on the
// feed and the checksums it read back from the release. In --dir: the packed app, oaiy-desktop-<v>-macos-arm64.app.tar.gz,
// and its .sig. Into --out: latest.json and SHA256SUMS.txt, which the workflow puts on the release in place of the two.
//
// The feed is the one published, with one entry added and nothing else changed: its version, notes, date and every other
// platform's entry are what they were (an installed OAIY on Windows or Linux reads the same feed after it as before). The
// Mac's entry is held to what the release job holds the others to (make-latest-json.mjs): the signature verifies against
// the public key in tauri.conf.json, and was made for a file of the Mac's kind, of this version, by its name
// (OAIY_<v>_aarch64.app.tar.gz); the file starts as a gzip file does; the url is this release's asset on GitHub. A feed for
// another version, one that has a Mac entry already, or a release whose checksums do not list latest.json is refused, and
// nothing is written.
//
// The checksums: latest.json's line gets its new digest, and the packed app, its signature and the disk image (with the
// digest the workflow held it to) are added, in the order `sha256sum *` lists files.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MinisignError, parseSignature } from './minisign.mjs';
import { checkAgainstKey, checkSignedFile, DEFAULT_REPO, FeedError, macAsset, readPinnedRepo, readPubkey, readSignature } from './make-latest-json.mjs';

const VERSION = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;
const REPO = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const TAG = /^v?[0-9]+\.[0-9]+\.[0-9]+$/;
const SHA256 = /^[0-9a-f]{64}$/;
const SUMS_LINE = /^([0-9a-f]{64}) {2}(\S(?:.*\S)?)$/;
const GZIP = Buffer.from([0x1f, 0x8b]);

/** The disk image of a release, which the Mac's update is made from. */
export const imageName = (version) => `oaiy-desktop-${version}-macos-arm64.dmg`;

/**
 * `feed` (the published feed, parsed) with the Mac's entry added, from the packed app and its .sig in `dir`. Throws a FeedError
 * that says what is wrong.
 */
export function addMacEntry({ feed, dir, version, tag, repo = DEFAULT_REPO, pubkey, pinnedRepo }) {
  if (!VERSION.test(String(version ?? ''))) throw new FeedError(`"${version ?? ''}" is not a version of the form 0.1.0 (three numbers, none with a leading zero)`);
  const releaseTag = tag ?? `v${version}`;
  if (!TAG.test(releaseTag) || releaseTag.replace(/^v/, '') !== version) throw new FeedError(`the tag "${releaseTag}" is not the version ${version} (or v${version})`);
  if (!REPO.test(repo)) throw new FeedError(`"${repo}" is not a repository of the form owner/name`);
  if (pinnedRepo !== undefined && repo !== pinnedRepo) {
    throw new FeedError(`the repository "${repo}" is not the one every installed OAIY reads its updates from ("${pinnedRepo}", the updater endpoint in tauri.conf.json)`);
  }
  if (!feed || typeof feed !== 'object' || Array.isArray(feed)) throw new FeedError('the published feed is not a JSON object');
  if (feed.version !== version) throw new FeedError(`the published feed is for version ${JSON.stringify(feed.version)}, not ${version}: it is not this release's`);
  const platforms = feed.platforms;
  if (!platforms || typeof platforms !== 'object' || Array.isArray(platforms) || Object.keys(platforms).length === 0) throw new FeedError('the published feed lists no platforms');
  const mac = macAsset(version);
  if (Object.hasOwn(platforms, mac.key)) throw new FeedError(`the published feed has a ${mac.key} entry already: nothing is replaced`);

  const file = path.join(dir ?? '', mac.asset);
  if (!dir || !fs.existsSync(file) || fs.statSync(file).size === 0) throw new FeedError(`${mac.what} is missing: ${mac.asset} is not in ${dir ?? ''}`);
  const signature = readSignature(`${file}.sig`, mac.asset);
  if (pubkey) checkAgainstKey(file, mac.asset, signature, pubkey);
  let trustedComment;
  try {
    trustedComment = parseSignature(signature).trustedComment;
  } catch (error) {
    if (!(error instanceof MinisignError)) throw error;
    throw new FeedError(`the signature of ${mac.asset} cannot be read: ${error.message}`);
  }
  checkSignedFile(trustedComment, { asset: mac.asset, version, suffix: mac.suffix });
  const head = Buffer.alloc(2);
  const fd = fs.openSync(file, 'r');
  try {
    fs.readSync(fd, head, 0, 2, 0);
  } finally {
    fs.closeSync(fd);
  }
  if (!head.equals(GZIP)) throw new FeedError(`${mac.asset} does not start as a gzip file does: it is not the packed app, and the desktop would refuse it`);

  const url = `https://github.com/${repo}/releases/download/${releaseTag}/${mac.asset}`;
  return { ...feed, platforms: { ...platforms, [mac.key]: { signature, url } } };
}

/**
 * The checksums `text` (`sha256sum` lines) with each of `digests` ({name: hex}) put in: a name listed already gets the new
 * digest, a new one is added, and the lines are in the order `sha256sum *` lists files. `mustHave` names files that must be
 * listed already (latest.json: the release job's checksums cover it).
 */
export function updateSums(text, digests, { mustHave = [] } = {}) {
  const entries = new Map();
  for (const line of String(text ?? '').split(/\r?\n/)) {
    if (line === '') continue;
    const found = SUMS_LINE.exec(line);
    if (!found) throw new FeedError(`SHA256SUMS.txt has a line that is not "<sha256>  <file>": ${JSON.stringify(line)}`);
    if (entries.has(found[2])) throw new FeedError(`SHA256SUMS.txt lists ${found[2]} twice`);
    entries.set(found[2], found[1]);
  }
  for (const name of mustHave) {
    if (!entries.has(name)) throw new FeedError(`SHA256SUMS.txt does not list ${name}: it is not the checksums the release job wrote`);
  }
  for (const [name, digest] of Object.entries(digests)) {
    if (!SHA256.test(digest)) throw new FeedError(`"${digest}" is not a sha256 digest (64 lowercase hex digits) for ${name}`);
    if (/[\s/\\]/.test(name)) throw new FeedError(`"${name}" is not a file name`);
    entries.set(name, digest);
  }
  // `sha256sum *` in the release job: the shell's sort of names, which on the runner (C.UTF-8) is by code point.
  const names = [...entries.keys()].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  return `${names.map((name) => `${entries.get(name)}  ${name}`).join('\n')}\n`;
}

const sha256 = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

/** Both files, into `out`; nothing is written unless both can be. Returns their paths. */
export function writeMacUpdate({ feedFile, sumsFile, dir, version, tag, repo, pubkey, pinnedRepo, imageSha256, out }) {
  let feed;
  try {
    feed = JSON.parse(fs.readFileSync(feedFile, 'utf8'));
  } catch {
    throw new FeedError(`the published feed ${feedFile} cannot be read as JSON`);
  }
  let sums;
  try {
    sums = fs.readFileSync(sumsFile, 'utf8');
  } catch {
    throw new FeedError(`the release's checksums ${sumsFile} cannot be read`);
  }
  if (!SHA256.test(String(imageSha256 ?? ''))) throw new FeedError(`"${imageSha256 ?? ''}" is not the disk image's sha256 digest`);
  const next = addMacEntry({ feed, dir, version, tag, repo, pubkey, pinnedRepo });
  const feedText = `${JSON.stringify(next, null, 2)}\n`;
  const mac = macAsset(version);
  const file = path.join(dir, mac.asset);
  const sumsText = updateSums(
    sums,
    {
      'latest.json': crypto.createHash('sha256').update(feedText).digest('hex'),
      [mac.asset]: sha256(file),
      [`${mac.asset}.sig`]: sha256(`${file}.sig`),
      [imageName(version)]: imageSha256,
    },
    { mustHave: ['latest.json'] },
  );
  fs.mkdirSync(out, { recursive: true });
  const written = { feed: path.join(out, 'latest.json'), sums: path.join(out, 'SHA256SUMS.txt') };
  fs.writeFileSync(written.feed, feedText);
  fs.writeFileSync(written.sums, sumsText);
  return written;
}

function parseArgs(argv) {
  const names = new Set(['feed', 'sums', 'dir', 'version', 'tag', 'repo', 'conf', 'image-sha256', 'out']);
  const options = {};
  for (let i = 0; i < argv.length; i += 2) {
    const flag = argv[i];
    const name = flag?.startsWith('--') ? flag.slice(2) : '';
    if (!names.has(name) || argv[i + 1] === undefined) throw new FeedError(`unexpected argument "${flag ?? ''}"; ${usage()}`);
    options[name] = argv[i + 1];
  }
  return options;
}

const usage = () => 'usage: add-mac-update.mjs --feed <latest.json> --sums <SHA256SUMS.txt> --dir <the Mac\'s files> --version <N.N.N> --image-sha256 <hex> --out <folder> [--tag <tag>] [--repo <owner/name>] [--conf <tauri.conf.json>]';

function main() {
  try {
    const args = parseArgs(process.argv.slice(2));
    for (const required of ['feed', 'sums', 'dir', 'version', 'image-sha256', 'out']) {
      if (!args[required]) throw new FeedError(usage());
    }
    const written = writeMacUpdate({
      feedFile: path.resolve(args.feed),
      sumsFile: path.resolve(args.sums),
      dir: path.resolve(args.dir),
      version: args.version,
      tag: args.tag,
      repo: args.repo,
      pubkey: args.conf ? readPubkey(path.resolve(args.conf)) : undefined,
      pinnedRepo: args.conf ? readPinnedRepo(path.resolve(args.conf)) : undefined,
      imageSha256: args['image-sha256'],
      out: path.resolve(args.out),
    });
    console.log(`wrote ${written.feed} and ${written.sums}`);
  } catch (error) {
    if (!(error instanceof FeedError)) throw error;
    console.error(`::error::the Mac's update was NOT added to the feed: ${error.message}`);
    process.exit(1);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
