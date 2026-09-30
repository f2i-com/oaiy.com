/**
 * The download offer that moved to shared/ (shared/downloads.ts, shared/repoLinks.ts).
 *
 * platform/ui/tests/downloads.mjs holds the behaviour to what it always was (it loads the file through the editor's
 * re-export, src/lib/downloads.ts). What is checked here is the move: the editor's file re-exports every name, and the shared
 * file is still pure (its own purity check in that suite now reads the re-export, which cannot fail).
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';
import { loadTs, ROOT, SHARED } from '../support/load.mjs';

const shared = await loadTs('shared/downloads.ts');

describe('the download offer in shared/', () => {
  it('is pure: no fetch, no XMLHttpRequest, no navigator, no import.meta, no request of any kind', () => {
    const source = fs.readFileSync(path.join(SHARED, 'downloads.ts'), 'utf8');
    const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
    for (const forbidden of [/\bfetch\s*\(/, /XMLHttpRequest/, /\bnavigator\b/, /import\.meta/, /api\.github\.com/, /sendBeacon/, /new WebSocket/]) assert.doesNotMatch(code, forbidden, String(forbidden));
    const links = fs.readFileSync(path.join(SHARED, 'repoLinks.ts'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    assert.doesNotMatch(links, /fetch\s*\(|XMLHttpRequest|import\.meta/);
  });

  it('is re-exported whole by the editor\'s own file, and the landing page\'s links come from the same constants', async () => {
    const shim = fs.readFileSync(path.join(ROOT, 'platform/ui/src/lib/downloads.ts'), 'utf8');
    assert.match(shim, /export \* from '@oaiy\/shared\/downloads';/);
    const landing = fs.readFileSync(path.join(ROOT, 'platform/ui/src/landing/repoLinks.ts'), 'utf8');
    assert.match(landing, /from '\.\.\/\.\.\/\.\.\/\.\.\/shared\/repoLinks'/);
    assert.doesNotMatch(landing.replace(/\/\/.*$/gm, ''), /github\.com/, 'the address is written once, in shared/repoLinks.ts');
    const links = await loadTs('shared/repoLinks.ts');
    const plan = shared.downloadPlan({ os: 'windows', arch: 'x64' }, 'v1.2.3');
    assert.ok(plan.primary.href.startsWith(`${links.REPO_URL}/releases/download/v1.2.3/`), plan.primary.href);
    assert.equal(plan.allDownloads, links.RELEASES_ALL_URL);
  });

  it('still says what it said: a Mac or a phone is told what OAIY Desktop is for, and gets no button', () => {
    for (const os of ['mac', 'ios', 'android']) {
      const plan = shared.downloadPlan({ os, arch: 'unknown' }, 'v1.2.3');
      assert.equal(plan.primary, null, os);
      assert.equal(plan.note, 'OAIY Desktop is for Windows and Linux. The web app works in your browser.', os);
    }
  });
});
