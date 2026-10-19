#!/usr/bin/env node
// Add the Mac to a published release's update feed: latest.json with a darwin-aarch64 entry, and SHA256SUMS.txt made true again.
//
//   node platform/scripts/add-mac-update.mjs --feed <latest.json> --sums <SHA256SUMS.txt> --dir <the Mac's files> \
//        --version 0.1.9 [--tag v0.1.9] [--repo f2i-com/oaiy.com] --image-sha256 <the image's digest> --out <folder> \
//        [--conf platform/desktop/src-tauri/tauri.conf.json]
//   node platform/scripts/add-mac-update.mjs --plan --version 0.1.9 --assets <the release's file names, one a line> \
//        [--feed <its latest.json>] [--pending-feed <its latest.pending.json>]
//   node platform/scripts/add-mac-update.mjs --check --dir <the Mac's files> --version 0.1.9 [--conf <tauri.conf.json>]
//
// --plan says what a run of the workflow does to the release as it is now, as lines for $GITHUB_OUTPUT (mode, reuse_app,
// reuse_sig, swap): `make` the Mac's update, reusing a packed app and its .sig that an earlier run put on the release
// before it failed (so that a run that failed after uploading them can finish: they cannot be replaced, and a new pack is
// not the same bytes), or `finish` an earlier run that was cut short while it swapped the new feed and checksums in. It
// refuses a release whose feed has the Mac already (nothing is replaced), and one in a state no run leaves (a .sig with no
// packed app beside it). --check holds a packed app and its .sig to the key and to what they were made for, with no feed:
// the workflow's check of the two it reuses, before it trusts them.
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
// digest the workflow held it to) are added, in the order `sha256sum *` lists files. The .sha256 files beside the Mac's files
// are not listed: each holds one of these digests and nothing else, so a line for it would check nothing that its file's
// line does not.
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
 * The names the workflow puts the new feed and checksums on the release under first, with the extension of the name each
 * replaces (so each keeps its content type). The release's own two are then removed and these renamed to them, one at a
 * time, so that the feed is missing for the moment between two calls and not for a whole upload, and never for good when an
 * upload fails.
 */
export const PENDING = { 'latest.json': 'latest.pending.json', 'SHA256SUMS.txt': 'SHA256SUMS.pending.txt' };

const checkVersion = (version) => {
  if (!VERSION.test(String(version ?? ''))) throw new FeedError(`"${version ?? ''}" is not a version of the form 0.1.0 (three numbers, none with a leading zero)`);
};

/**
 * What a run does to the release, from what the release has now: `assets`, the names of its files; `feed`, its latest.json
 * (parsed; undefined when it has none); `pendingFeed`, its latest.pending.json (the same). Returns { mode, reuseApp, reuseSig,
 * swap }: `make` the Mac's update (reusing the packed app and its .sig an earlier run uploaded, if it did: `reuseApp`,
 * `reuseSig`), or `finish` an earlier run's swap, renaming the pending files `swap` names. Throws a FeedError that says why
 * nothing is to be done.
 */
export function planMacUpdate({ version, assets, feed, pendingFeed }) {
  checkVersion(version);
  const names = new Set(assets ?? []);
  const mac = macAsset(version);
  const read = (value, what) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new FeedError(`${what} is not a JSON object (or was not read)`);
    if (value.version !== version) throw new FeedError(`${what} is for version ${JSON.stringify(value.version)}, not ${version}: it is not this release's`);
    const platforms = value.platforms;
    return Boolean(platforms && typeof platforms === 'object' && Object.hasOwn(platforms, mac.key));
  };
  const pending = Object.keys(PENDING).filter((name) => names.has(PENDING[name]));
  // An earlier run renamed nothing yet, or the feed already: the feed the release serves (or, cut short between removing it and
  // renaming the new one, the new one) says which.
  const served = names.has('latest.json') ? read(feed, "the release's latest.json") : undefined;
  const inPlace = served ?? (names.has(PENDING['latest.json']) ? read(pendingFeed, `the release's ${PENDING['latest.json']}`) : undefined);
  if (inPlace === undefined) throw new FeedError("the release has no latest.json: it is not one the release job published");
  if (inPlace) {
    if (pending.length === 0) throw new FeedError(`the release's feed lists ${mac.key} already: ${version} has its Mac update, and nothing is replaced`);
    return { mode: 'finish', reuseApp: false, reuseSig: false, swap: pending };
  }
  if (served === undefined) throw new FeedError(`the release has no latest.json, and its ${PENDING['latest.json']} has no ${mac.key} entry: put the release job's latest.json back`);
  const image = imageName(version);
  if (!names.has(image) || !names.has(`${image}.sha256`)) throw new FeedError(`the release has no ${image} and ${image}.sha256 yet: run mac-build.yml with release=v${version} first`);
  if (!names.has('SHA256SUMS.txt')) throw new FeedError("the release has no SHA256SUMS.txt: it is not one the release job published");
  const reuseApp = names.has(mac.asset);
  for (const beside of [`${mac.asset}.sig`, `${mac.asset}.sha256`]) {
    if (names.has(beside) && !reuseApp) throw new FeedError(`the release has ${beside} but not ${mac.asset}, which no run leaves: delete it, then run this again`);
  }
  return { mode: 'make', reuseApp, reuseSig: names.has(`${mac.asset}.sig`), swap: Object.keys(PENDING) };
}

/**
 * The packed app and its .sig in `dir`, held to what the release job holds the others to (make-latest-json.mjs): the signature
 * verifies against `pubkey` (when given), it was made for a file of the Mac's kind, of this version, by its name, and the file
 * starts as a gzip file does. Returns the signature (the .sig's content). Throws a FeedError that says what is wrong.
 */
export function checkMacFiles({ dir, version, pubkey }) {
  checkVersion(version);
  const mac = macAsset(version);
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
  return signature;
}

/**
 * `feed` (the published feed, parsed) with the Mac's entry added, from the packed app and its .sig in `dir`. Throws a FeedError
 * that says what is wrong.
 */
export function addMacEntry({ feed, dir, version, tag, repo = DEFAULT_REPO, pubkey, pinnedRepo }) {
  checkVersion(version);
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
  const signature = checkMacFiles({ dir, version, pubkey });

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
  const names = new Set(['feed', 'sums', 'dir', 'version', 'tag', 'repo', 'conf', 'image-sha256', 'out', 'assets', 'pending-feed']);
  const switches = new Set(['plan', 'check']);
  const options = {};
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const name = flag?.startsWith('--') ? flag.slice(2) : '';
    if (switches.has(name)) {
      options[name] = true;
      continue;
    }
    if (!names.has(name) || argv[i + 1] === undefined) throw new FeedError(`unexpected argument "${flag ?? ''}"; ${usage()}`);
    options[name] = argv[i + 1];
    i += 1;
  }
  if (options.plan && options.check) throw new FeedError(`--plan and --check are two runs; ${usage()}`);
  return options;
}

const usage = () =>
  'usage: add-mac-update.mjs --feed <latest.json> --sums <SHA256SUMS.txt> --dir <the Mac\'s files> --version <N.N.N> --image-sha256 <hex> --out <folder> [--tag <tag>] [--repo <owner/name>] [--conf <tauri.conf.json>]' +
  ' | --plan --version <N.N.N> --assets <names> [--feed <latest.json>] [--pending-feed <latest.pending.json>]' +
  ' | --check --dir <the Mac\'s files> --version <N.N.N> [--conf <tauri.conf.json>]';

const readJson = (file, what) => {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    throw new FeedError(`${what} ${file} cannot be read as JSON`);
  }
};

function requireArgs(args, names) {
  for (const required of names) {
    if (!args[required]) throw new FeedError(usage());
  }
}

/** --plan: what this run does, as `name=value` lines for $GITHUB_OUTPUT. */
function plan(args) {
  requireArgs(args, ['version', 'assets']);
  let assets;
  try {
    assets = fs.readFileSync(path.resolve(args.assets), 'utf8').split(/\r?\n/).filter((name) => name !== '');
  } catch {
    throw new FeedError(`the release's file names ${args.assets} cannot be read`);
  }
  const feed = args.feed ? readJson(path.resolve(args.feed), "the release's latest.json") : undefined;
  const pendingFeed = args['pending-feed'] ? readJson(path.resolve(args['pending-feed']), `the release's ${PENDING['latest.json']}`) : undefined;
  const { mode, reuseApp, reuseSig, swap } = planMacUpdate({ version: args.version, assets, feed, pendingFeed });
  console.log(`mode=${mode}\nreuse_app=${reuseApp}\nreuse_sig=${reuseSig}\nswap=${swap.join(' ')}`);
}

/** --check: the packed app and its .sig an earlier run put on the release, before this one reuses them. */
function check(args) {
  requireArgs(args, ['dir', 'version']);
  checkMacFiles({ dir: path.resolve(args.dir), version: args.version, pubkey: args.conf ? readPubkey(path.resolve(args.conf)) : undefined });
  console.log(`OK: ${macAsset(args.version).asset} and its .sig in ${args.dir} are the Mac's update of ${args.version}${args.conf ? ', signed with the key in tauri.conf.json' : ''}`);
}

function write(args) {
  requireArgs(args, ['feed', 'sums', 'dir', 'version', 'image-sha256', 'out']);
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
}

function main() {
  let args = {};
  try {
    args = parseArgs(process.argv.slice(2));
    if (args.plan) plan(args);
    else if (args.check) check(args);
    else write(args);
  } catch (error) {
    if (!(error instanceof FeedError)) throw error;
    const what = args.plan ? 'nothing was done to the release' : args.check ? 'the Mac files on the release are NOT reused' : "the Mac's update was NOT added to the feed";
    console.error(`::error::${what}: ${error.message}`);
    process.exit(1);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
