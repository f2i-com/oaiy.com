#!/usr/bin/env node
// Check a file against its updater signature and the public key every installed OAIY carries.
//
//   node platform/scripts/verify-signature.mjs <file> [--sig <file.sig>] [--conf <tauri.conf.json>] [--pubkey-file <key.pub>]
//
// The signature is `<file>.sig` unless --sig says other (what `tauri signer sign <file>` writes beside a file), and
// the key is the updater public key in platform/desktop/src-tauri/tauri.conf.json (plugins.updater.pubkey) unless
// --conf names another tauri.conf.json or --pubkey-file a `.pub` file (a throwaway key's, in a test).
//
// It is what a maintainer runs BEFORE the first release to prove that the private key they hold is the pair of the
// public key inside every installer (docs/RELEASING.md, "Before the first release"): sign a probe file with the
// production key, then run this on it. It is the same check the release job makes (minisign.mjs), with the same
// node:crypto, and it prints the outcome and nothing secret: not the signature, not a key, not a password. The
// key id and the trusted comment (a time and the signed file's name) are public and are shown.
//
// Exit status: 0 verifies, 1 does not verify, 2 it could not be checked (a file, a signature or a key that is
// missing or that is not one).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MinisignError, parsePublicKey, parseSignature, verifyMinisign } from './minisign.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_CONF = path.resolve(here, '..', 'desktop', 'src-tauri', 'tauri.conf.json');

export class CannotCheck extends Error {}

const read = (file, what) => {
  try {
    return fs.readFileSync(file);
  } catch {
    throw new CannotCheck(`${what} ${file} cannot be read`);
  }
};

/** The public key from a tauri.conf.json (plugins.updater.pubkey). */
export function pubkeyFromConf(confPath) {
  let conf;
  try {
    conf = JSON.parse(read(confPath, 'the configuration').toString('utf8'));
  } catch (error) {
    if (error instanceof CannotCheck) throw error;
    throw new CannotCheck(`${confPath} is not a tauri.conf.json`);
  }
  const pubkey = conf?.plugins?.updater?.pubkey;
  if (typeof pubkey !== 'string' || !pubkey.trim()) throw new CannotCheck(`${confPath} has no plugins.updater.pubkey`);
  return pubkey.trim();
}

/**
 * Whether `signature` (the content of a .sig file) is a signature of `data` by the key `pubkey` (the content of the public key as
 * tauri.conf.json holds it). Returns \`{ ok, reason?, keyId, trustedComment? }\`; throws CannotCheck when a key or a signature is not one.
 */
export function verifyFile({ data, signature, pubkey }) {
  try {
    const keyId = parsePublicKey(pubkey).keyId.toString('hex').toUpperCase();
    parseSignature(signature);
    const result = verifyMinisign(data, signature, pubkey);
    return { ...result, keyId };
  } catch (error) {
    if (error instanceof MinisignError) throw new CannotCheck(error.message);
    throw error;
  }
}

function parseArgs(argv) {
  const options = { file: undefined, sig: undefined, conf: undefined, pubkeyFile: undefined };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--sig' || arg === '--conf' || arg === '--pubkey-file') {
      if (argv[i + 1] === undefined) throw new CannotCheck(`${arg} needs a value`);
      options[arg === '--pubkey-file' ? 'pubkeyFile' : arg.slice(2)] = argv[++i];
    } else if (arg.startsWith('--') || options.file !== undefined) {
      throw new CannotCheck(`unexpected argument "${arg}"; ${usage}`);
    } else {
      options.file = arg;
    }
  }
  if (options.file === undefined) throw new CannotCheck(usage);
  if (options.conf !== undefined && options.pubkeyFile !== undefined) throw new CannotCheck('give --conf or --pubkey-file, not both');
  return options;
}

const usage = 'usage: verify-signature.mjs <file> [--sig <file.sig>] [--conf <tauri.conf.json>] [--pubkey-file <key.pub>]';

function main() {
  try {
    const options = parseArgs(process.argv.slice(2));
    const file = path.resolve(options.file);
    const sigFile = path.resolve(options.sig ?? `${options.file}.sig`);
    const source = options.pubkeyFile !== undefined ? path.resolve(options.pubkeyFile) : path.resolve(options.conf ?? DEFAULT_CONF);
    const pubkey = options.pubkeyFile !== undefined ? read(source, 'the public key file').toString('utf8').trim() : pubkeyFromConf(source);
    const data = read(file, 'the file');
    const signature = read(sigFile, 'the signature').toString('utf8').trim();
    const result = verifyFile({ data, signature, pubkey });
    const where = options.pubkeyFile !== undefined ? `the public key in ${path.basename(source)}` : `the public key in ${path.basename(source)} (plugins.updater.pubkey)`;
    if (result.ok) {
      console.log(`OK: ${path.basename(file)} verifies against ${where}; key id ${result.keyId}`);
      console.log(`    signed as: ${result.trustedComment}`);
      process.exit(0);
    }
    console.error(`NOT VERIFIED: ${path.basename(file)} does not verify against ${where} (key id ${result.keyId}): ${result.reason}`);
    process.exit(1);
  } catch (error) {
    if (!(error instanceof CannotCheck)) throw error;
    console.error(`CANNOT CHECK: ${error.message}`);
    process.exit(2);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
