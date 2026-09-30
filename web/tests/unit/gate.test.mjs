/**
 * What runs these tests, and what they run on (the review's F10): a suite nobody runs protects nothing.
 *
 *   - the CI gate has a lane for web/ (typecheck and the unit tests; the browser tests stay a manual run for now, and the lane says so);
 *   - the lane installs from a lockfile (`npm ci`), and the lockfile is the one web/package.json describes, every package from the
 *     registry with an integrity hash;
 *   - the purity check of the download helper, which moved with the helper to shared/, reads the shared files, not the editor's
 *     re-export (a check of a re-export cannot fail).
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';
import { parseYaml, workflowSteps } from '../../../platform/scripts/workflow-yaml.mjs';
import { ROOT } from '../support/load.mjs';

const text = (...parts) => fs.readFileSync(path.join(ROOT, ...parts), 'utf8').replace(/\r\n/g, '\n');
const manifest = JSON.parse(text('web', 'package.json'));

describe('the lockfile is the one web/package.json describes', () => {
  const lock = JSON.parse(text('web', 'package-lock.json'));

  it('is there, in the current format, for this package', () => {
    assert.equal(lock.lockfileVersion, 3);
    assert.equal(lock.name, manifest.name);
    assert.equal(lock.packages[''].name, manifest.name);
  });

  it('lists the same dependencies as the manifest, each locked at the version the manifest pins', () => {
    for (const section of ['dependencies', 'devDependencies', 'optionalDependencies']) {
      assert.deepEqual(lock.packages[''][section], manifest[section], section);
    }
    for (const [name, version] of Object.entries({ ...manifest.dependencies, ...manifest.devDependencies })) {
      assert.equal(lock.packages[`node_modules/${name}`]?.version, version, name);
    }
  });

  it('holds every package from the registry, with an integrity hash: none from an address that could change', () => {
    const entries = Object.entries(lock.packages).filter(([key]) => key !== '');
    assert.ok(entries.length > 20, `${entries.length} packages`);
    for (const [key, entry] of entries) {
      assert.match(entry.resolved ?? '', /^https:\/\/registry\.npmjs\.org\//, `${key} resolves to ${entry.resolved}`);
      assert.match(entry.integrity ?? '', /^sha(256|384|512)-/, `${key} has no integrity hash`);
    }
  });
});

describe('the gate runs the web app\'s tests', () => {
  const ci = parseYaml(text('.github', 'workflows', 'ci.yml'));
  const steps = workflowSteps(ci).filter((step) => step.job === 'webapp');

  it('has a lane for web/ that installs from the lockfile and runs the typecheck and the unit tests', () => {
    assert.ok(ci.jobs.webapp, 'no lane named webapp');
    assert.ok(steps.length >= 4, `${steps.length} steps`);
    const install = steps.find((step) => step.run === 'npm ci');
    assert.equal(install?.workingDirectory, 'web', 'npm ci runs in web/');
    const cache = steps.find((step) => step.with?.['cache-dependency-path']);
    assert.equal(cache?.with['cache-dependency-path'], 'web/package-lock.json');
    const test = steps.find((step) => step.run === 'npm test');
    assert.equal(test?.workingDirectory, 'web', 'npm test runs in web/');
    assert.ok(steps.indexOf(install) < steps.indexOf(test), 'the install is before the tests');
  });

  it('does not download browsers (the browser tests are not part of this lane), and audits like the others', () => {
    assert.equal(ci.jobs.webapp.env?.PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD, '1');
    assert.ok(steps.some((step) => /npm audit --audit-level=high/.test(step.run ?? '') && step.workingDirectory === 'web'), 'a dependency audit runs in web/');
  });

  it('runs what `npm test` runs: the typecheck and every unit test', () => {
    assert.equal(manifest.scripts.test, 'npm run typecheck && npm run test:unit');
    assert.match(manifest.scripts['test:unit'], /tests\/unit\/\*\*\/\*\.test\.mjs/);
  });
});

describe('the download helper\'s purity check reads the file that has the code', () => {
  const suite = text('platform', 'ui', 'tests', 'downloads.mjs');

  it('platform/ui/tests/downloads.mjs reads shared/downloads.ts and shared/repoLinks.ts for the purity check', () => {
    assert.match(suite, /path\.join\(shared, 'downloads\.ts'\)/);
    assert.match(suite, /path\.join\(shared, 'repoLinks\.ts'\)/);
    assert.doesNotMatch(suite, /readFileSync\(path\.join\(UI, 'src', 'lib', 'downloads\.ts'\), 'utf8'\);\s*\n\s*\/\/ Comments may talk about them/, 'the code checked is not the re-export');
  });

  it('and the editor\'s own file only re-exports it', () => {
    assert.match(text('platform', 'ui', 'src', 'lib', 'downloads.ts'), /export \* from '@oaiy\/shared\/downloads';/);
  });
});
