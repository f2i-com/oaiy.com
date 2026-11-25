import { expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { cpSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { strToU8, zipSync } from 'fflate';

it('keeps the working preview after a missing asset, a bridge failure or an extraction error', () => {
  // A sibling of scripts/ resolves the same npm packages without touching public/softn/.
  const app = process.cwd();
  const scratch = mkdtempSync(join(app, '.softn-fetch-test-'));
  try {
    cpSync(join(app, 'scripts'), join(scratch, 'scripts'), { recursive: true });
    const script = join(scratch, 'scripts', 'fetch-softn.mjs');
    const release = readFileSync(script, 'utf8').match(/const RELEASE = '(v[\d.]+)'/)[1];
    const asset = 'assets/app.js';
    const sha = bytes => createHash('sha256').update(bytes).digest('hex');
    const runtime = join(scratch, 'public', 'softn');
    const snapshot = () => Object.fromEntries(['index.html', asset, 'SOURCE.json', 'bot-bridge.js', 'bot-capture.js']
      .map(path => [path, sha(readFileSync(join(runtime, path)))]));
    function install({ html = '<html><head></head></html>', source = 'console.log(1)', missing = false, collision = false, separator = '/' } = {}) {
      const files = { 'index.html': html, [asset]: source };
      const manifest = { files: Object.fromEntries(Object.entries(files).map(([name, text]) => [name, sha(text)])) };
      if (missing) delete files[asset];
      const entries = { ...(collision ? { assets: 'not a directory' } : {}), ...files, 'runtime-manifest.json': JSON.stringify(manifest) };
      const bytes = zipSync(Object.fromEntries(Object.entries(entries).map(([name, text]) =>
        [('hosted-runtime/' + name).replaceAll('/', separator), strToU8(text)])));
      writeFileSync(join(scratch, 'softn-formlogic-runtime-' + release + '.zip'), bytes);
      return spawnSync(process.execPath, [script], { encoding: 'utf8', env: { ...process.env, SOFTN_RELEASE_DIR: scratch, SOFTN_ZIP_SHA256: sha(bytes) } });
    }

    const initial = install();
    expect(initial.status, initial.stderr).toBe(0);
    expect(readFileSync(join(runtime, 'index.html'), 'utf8')).toContain('data-oaiy-agent-bridge');
    const working = snapshot();
    for (const failure of [{ missing: true }, { html: '<body>Incomplete page</body>' }, { collision: true }]) {
      expect(install(failure).status).toBe(1);
      expect(snapshot()).toEqual(working);
      expect(readdirSync(scratch).filter(name => name.startsWith('.softn-install-'))).toEqual([]);
    }
    const replacement = install({ source: 'console.log(2)', separator: '\\' });
    expect(replacement.status, replacement.stderr).toBe(0);
    expect(readFileSync(join(runtime, asset), 'utf8')).toBe('console.log(2)');

    // A bridge update reuses the cached runtime when the archive is unavailable.
    const stamp = JSON.parse(readFileSync(join(runtime, 'SOURCE.json'), 'utf8'));
    writeFileSync(join(runtime, 'bot-bridge.js'), 'old bridge');
    const ensure = spawnSync(process.execPath, [script, '--ensure'], { encoding: 'utf8',
      env: { ...process.env, SOFTN_RELEASE_DIR: join(scratch, 'offline'), SOFTN_ZIP_SHA256: stamp.zip_sha256 } });
    expect(ensure.status, ensure.stderr).toBe(0);
    expect(snapshot()['bot-bridge.js']).toBe(working['bot-bridge.js']);
  } finally {
    rmSync(scratch, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  }
});
