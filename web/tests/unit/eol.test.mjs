/**
 * Line ends (the review's F9): a machine with core.autocrlf=true checks the web app's files out with CRLF. A header template then ends
 * every value in `\r`, and the mutation runner and the tests that match on exact text stop matching, each failing in a way that looks like
 * a bug in the code. The web app's and shared/'s files are LF on every checkout (.gitattributes), and the code that reads a template does
 * not depend on it.
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import { ROOT } from '../support/load.mjs';
import { parseHeaders, headersFor } from '../e2e/hosts.mjs';
import { readTemplate, renderHeaders } from '../../scripts/headers.mjs';

const git = (cwd, ...args) => spawnSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@example.invalid', '-c', 'commit.gpgsign=false', ...args], { cwd, encoding: 'utf8' });

describe('the web app\'s files are LF on every checkout', () => {
  it('.gitattributes says eol=lf for web/ and shared/, and leaves the rest of the repository as it was', () => {
    const asked = ['web/hosting/headers/providers.headers', 'web/providers/src/store.ts', 'web/tests/e2e/mutations.mjs', 'shared/downloads.ts', 'shared/broker/protocol.ts', 'web/package.json'];
    const result = git(ROOT, 'check-attr', 'eol', '--', ...asked, 'platform/ui/package.json', 'app/package.json');
    assert.equal(result.status, 0, result.stderr);
    const attrs = Object.fromEntries(result.stdout.trim().split('\n').map((line) => [line.split(':')[0], line.split(': ').slice(2).join(': ')]));
    for (const file of asked) assert.equal(attrs[file], 'lf', file);
    assert.equal(attrs['platform/ui/package.json'], 'unspecified', 'nothing outside web/ and shared/ is touched');
    assert.equal(attrs['app/package.json'], 'unspecified');
  });

  it('a checkout on a machine that converts line ends (core.autocrlf=true) leaves web/ and shared/ as LF, and converts a file elsewhere (the control)', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-eol-'));
    try {
      const origin = path.join(dir, 'origin');
      const clone = path.join(dir, 'clone');
      const files = { 'web/hosting/headers/x.headers': '/*\n  A: 1\n', 'shared/y.ts': 'export const y = 1;\n', 'web/tests/z.mjs': 'const a = 1;\n', 'other/w.txt': 'one\ntwo\n' };
      fs.mkdirSync(origin);
      assert.equal(git(origin, 'init', '-q').status, 0);
      fs.copyFileSync(path.join(ROOT, '.gitattributes'), path.join(origin, '.gitattributes'));
      for (const [file, text] of Object.entries(files)) {
        fs.mkdirSync(path.dirname(path.join(origin, file)), { recursive: true });
        fs.writeFileSync(path.join(origin, file), text);
      }
      assert.equal(git(origin, 'add', '-A').status, 0);
      const committed = git(origin, 'commit', '-q', '-m', 'files');
      assert.equal(committed.status, 0, committed.stderr);
      const cloned = git(dir, 'clone', '-q', '--config', 'core.autocrlf=true', origin, clone);
      assert.equal(cloned.status, 0, cloned.stderr);
      const read = (file) => fs.readFileSync(path.join(clone, file), 'utf8');
      assert.ok(read('other/w.txt').includes('\r\n'), 'the control: this checkout converts line ends, so the experiment can tell');
      for (const file of ['web/hosting/headers/x.headers', 'shared/y.ts', 'web/tests/z.mjs']) assert.ok(!read(file).includes('\r'), `${file} has no CR`);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('a template with CRLF is read as LF', () => {
  const crlf = '/*\r\n  Cache-Control: no-store\r\n  Content-Security-Policy: frame-ancestors {{APP_ORIGINS}}\r\n';

  it('readTemplate returns no CR, whatever the file has', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-tpl-'));
    try {
      fs.writeFileSync(path.join(dir, 'x.headers'), crlf);
      const text = readTemplate('x', dir);
      assert.ok(!text.includes('\r'));
      assert.equal(text, crlf.replace(/\r\n/g, '\n'));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('renderHeaders takes a CRLF template too, and no header value ends in CR', () => {
    const rendered = renderHeaders(crlf, { apps: ['https://agent.example', 'https://flows.example'] });
    assert.ok(!rendered.includes('\r'));
    const headers = headersFor(parseHeaders(rendered), '/anything');
    assert.equal(headers['cache-control'], 'no-store');
    assert.equal(headers['content-security-policy'], 'frame-ancestors https://agent.example https://flows.example');
  });

  it('the real templates have none', () => {
    for (const host of ['agent', 'flows', 'providers']) assert.ok(!readTemplate(host).includes('\r'), host);
  });
});
