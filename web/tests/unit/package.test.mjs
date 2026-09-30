/**
 * web/package.json pins every dependency to one exact version (the review's F11).
 *
 * The web app holds people's provider keys and is built and tested from what `npm` fetches, so the version of a tool that builds the
 * page a key is typed into, or that runs its tests, is not left to whatever the registry calls newest that a range allows. A range
 * (`^`, `~`, `>=`, `*`, `latest`, an address) is a version chosen by someone else on the day of the install; an exact one changes only
 * when a commit says so.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';
import { ROOT } from '../support/load.mjs';

const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'web', 'package.json'), 'utf8'));
const SECTIONS = ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies'];

describe('web/package.json pins every dependency to one exact version', () => {
  it('has dependencies to pin (a package with none would pass this for nothing)', () => {
    const names = SECTIONS.flatMap((section) => Object.keys(manifest[section] ?? {}));
    for (const tool of ['esbuild', 'playwright', 'typescript', 'vite']) assert.ok(names.includes(tool), tool);
  });

  for (const section of SECTIONS) {
    it(`${section}: each is a version, exactly (no ^, ~, range, tag or address)`, () => {
      for (const [name, spec] of Object.entries(manifest[section] ?? {})) {
        assert.match(spec, /^\d+\.\d+\.\d+$/, `${name}@${spec} in ${section} is not one exact version`);
      }
    });
  }

  it('says nothing that lets a tool move a version: no overrides, no resolutions', () => {
    assert.equal(manifest.overrides, undefined);
    assert.equal(manifest.resolutions, undefined);
  });
});
