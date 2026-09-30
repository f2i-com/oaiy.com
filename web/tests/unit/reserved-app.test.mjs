/**
 * `modal` is not an app name (the review's low 8): the embedded modal's Test and Load-models buttons spend an hour of their own, kept under
 * the name `modal`, and an app a deployment called `modal` would share that hour (and the modal's default of 60 an hour, not an app's 600).
 * The list of apps refuses the name where it is read, and the assembler refuses it before it writes a page.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';
import { assembleProviders } from '../../scripts/assemble.mjs';

const P = await loadTs('shared/broker/protocol.ts');

describe('the name modal is reserved', () => {
  it('the list of apps refuses it, alone or among others, so a deployment that used it lists nobody', () => {
    assert.equal(P.parseAppOrigins('modal=https://modal.example.org').size, 0);
    assert.equal(P.parseAppOrigins('agent=https://agent.example.org modal=https://modal.example.org').size, 0, 'the whole list fails closed');
    assert.equal(P.parseAppOrigins('modal=https://modal.example.org agent=https://agent.example.org').size, 0);
    assert.deepEqual([...P.RESERVED_APP_NAMES], ['modal']);
    assert.equal(P.MODAL_APP, 'modal');
  });

  it('a name that only looks like it is an app name, as before', () => {
    const map = P.parseAppOrigins('modal2=https://a.example.org model=https://b.example.org mod-al=https://c.example.org');
    assert.equal(map.size, 3);
  });

  it('the assembler refuses it before it writes anything, and says why; another name assembles', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-reserved-'));
    try {
      const dist = path.join(dir, 'dist');
      fs.mkdirSync(dist);
      for (const page of ['index.html', 'embed.html', 'broker.html']) fs.writeFileSync(path.join(dist, page), '<!doctype html><meta name="oaiy-apps" content="" />\n');
      const out = path.join(dir, 'out');
      await assert.rejects(assembleProviders({ outDir: out, distDir: dist, providers: 'https://providers.example.org', apps: { agent: 'https://agent.example.org', modal: 'https://m.example.org' } }), /"modal" is reserved/);
      assert.equal(fs.existsSync(out), false, 'nothing was written');
      await assembleProviders({ outDir: out, distDir: dist, providers: 'https://providers.example.org', apps: { agent: 'https://agent.example.org', flows: 'https://flows.example.org' } });
      const html = fs.readFileSync(path.join(out, 'broker.html'), 'utf8');
      assert.match(html, /content="agent=https:\/\/agent\.example\.org flows=https:\/\/flows\.example\.org"/);
      assert.ok(!/modal/.test(html));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
