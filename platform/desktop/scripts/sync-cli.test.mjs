// @vitest-environment node
//
// The staging check `sync-cli.mjs` runs after copying the CLI into
// src-tauri/resources/cli. Driven against a fixture here, never the real
// folder: what this pins is that a staged engine that is missing, tampered
// with or built for another bundle is a FAILED build naming the file — not a
// desktop that ships, resolves its CLI, and fails every run `engine_unavailable`
// on the user's machine.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  STAGED_FILES,
  StagingError,
  missingBuildOutputs,
  stageEngine,
  verifyStaged,
} from './sync-cli-lib.mjs';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

let root;
let dist;
let dest;

/** A dist/ shaped like cli/esbuild.mjs's output: the identity is whatever the fixture's bytes hash to. */
function writeDist() {
  const wasm = Buffer.from('not really wasm, but the digest is what matters ' + Math.random());
  const notices = 'Third-party notices for the fixture\n';
  const source = {
    release: 'vX.Y.Z-fixture',
    artifact: 'zipp_wasm_bg.wasm',
    sha256: sha256(wasm),
    notices: { file: 'THIRD_PARTY_LICENSES.txt', source: 'oaiy-curated', sha256: sha256(notices) },
  };
  fs.mkdirSync(path.join(dist, 'zipp'), { recursive: true });
  fs.writeFileSync(path.join(dist, 'zipp', 'zipp_wasm_bg.wasm'), wasm);
  fs.writeFileSync(path.join(dist, 'zipp', 'THIRD_PARTY_LICENSES.txt'), notices);
  fs.writeFileSync(path.join(dist, 'zipp', 'SOURCE.json'), JSON.stringify(source, null, 2));
  fs.writeFileSync(path.join(dist, 'zipp', 'PROFILE.json'), '{"limits":{}}');
  fs.writeFileSync(path.join(dist, 'zipp', 'LICENSE-APACHE'), 'Apache-2.0\n');
  fs.writeFileSync(path.join(dist, 'oaiy-zipp-worker.mjs'), 'export {};\n');
  // The bundle carries the wasm digest as the baked define does.
  fs.writeFileSync(path.join(dist, 'oaiy.mjs'), `var __ZIPP_WASM_SHA256__ = "${source.sha256}";\n`);
  return source;
}

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-sync-cli-test-'));
  dist = path.join(root, 'dist');
  dest = path.join(root, 'staged');
  fs.mkdirSync(dest);
});

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true });
});

describe('staging the CLI and its engine', () => {
  it('names every build output a run needs, and rebuilds when any is missing', () => {
    expect(missingBuildOutputs(dist).length).toBeGreaterThan(0);
    writeDist();
    expect(missingBuildOutputs(dist)).toEqual([]);
    fs.rmSync(path.join(dist, 'oaiy-zipp-worker.mjs'));
    expect(missingBuildOutputs(dist)).toEqual(['oaiy-zipp-worker.mjs']);
  });

  it('stages the bundle, the worker and the five engine files, and they verify against dist', () => {
    const source = writeDist();
    stageEngine(dist, dest);
    for (const rel of STAGED_FILES) expect(fs.existsSync(path.join(dest, rel)), rel).toBe(true);
    expect(STAGED_FILES).toHaveLength(7);
    expect(verifyStaged(dest, dist)).toEqual({ release: source.release, wasmSha256: source.sha256 });
  });

  it('a missing staged file is a failure that names it', () => {
    writeDist();
    stageEngine(dist, dest);
    fs.rmSync(path.join(dest, 'oaiy-zipp-worker.mjs'));
    expect(() => verifyStaged(dest, dist)).toThrow(StagingError);
    expect(() => verifyStaged(dest, dist)).toThrow(/missing oaiy-zipp-worker\.mjs/);
  });

  it('a tampered wasm is a failure naming the file and both digests', () => {
    const source = writeDist();
    stageEngine(dist, dest);
    const wasm = path.join(dest, 'zipp', 'zipp_wasm_bg.wasm');
    const bytes = fs.readFileSync(wasm);
    bytes[0] ^= 0xff;
    fs.writeFileSync(wasm, bytes);
    let message = '';
    try {
      verifyStaged(dest, dist);
    } catch (error) {
      message = error.message;
    }
    expect(message).toMatch(/zipp_wasm_bg\.wasm/);
    expect(message).toContain(source.sha256);
    expect(message).toContain(sha256(bytes));
  });

  it('tampered notices are a failure naming the file', () => {
    writeDist();
    stageEngine(dist, dest);
    fs.appendFileSync(path.join(dest, 'zipp', 'THIRD_PARTY_LICENSES.txt'), 'one more line\n');
    expect(() => verifyStaged(dest, dist)).toThrow(/THIRD_PARTY_LICENSES\.txt/);
  });

  it('a bundle built for another engine is a failure even though every file is present', () => {
    const source = writeDist();
    stageEngine(dist, dest);
    fs.writeFileSync(path.join(dest, 'oaiy.mjs'), 'var __ZIPP_WASM_SHA256__ = "' + 'f'.repeat(64) + '";\n');
    // Only against the staged folder: the dist comparison would catch this
    // first, and the point is that the bundle-to-bytes tie stands on its own.
    expect(() => verifyStaged(dest)).toThrow(/oaiy\.mjs was not built for the staged engine/);
    expect(() => verifyStaged(dest)).toThrow(source.sha256);
  });

  it('a staged file that differs from its dist source is a failure naming it', () => {
    writeDist();
    stageEngine(dist, dest);
    // Same digest story stays intact — SOURCE.json itself is rewritten with
    // the same content but a different byte layout.
    const sourcePath = path.join(dest, 'zipp', 'SOURCE.json');
    fs.writeFileSync(sourcePath, JSON.stringify(JSON.parse(fs.readFileSync(sourcePath, 'utf8'))));
    expect(verifyStaged(dest)).toBeTruthy();
    expect(() => verifyStaged(dest, dist)).toThrow(/SOURCE\.json differs from its source/);
  });

  it('a SOURCE.json without digests is refused rather than trusted', () => {
    writeDist();
    stageEngine(dist, dest);
    fs.writeFileSync(path.join(dest, 'zipp', 'SOURCE.json'), '{"release":"vX"}');
    expect(() => verifyStaged(dest)).toThrow(/records no wasm sha256/);
  });
});
