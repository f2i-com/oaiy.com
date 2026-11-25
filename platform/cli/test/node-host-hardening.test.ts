/**
 * Node-host hardening unit tests for FFmpeg confinement (P0-1) and SQLite
 * lexer-aware placeholder normalization (P2-6). Run via `npm test`.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Create an isolated sandbox dir BEFORE dynamically importing ffmpeg/sql so
// TMP_ROOT/DATA_DIR/OAIY_FFMPEG/OAIY_FFPROBE pick up our test paths.
const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-cli-hardening-'));
const tmpRoot = path.join(sandbox, 'tmp');
const dataRoot = path.join(sandbox, 'data');
fs.mkdirSync(tmpRoot, { recursive: true });
fs.mkdirSync(dataRoot, { recursive: true });

process.env.OAIY_DATA_DIR = dataRoot;
process.env.OAIY_TMP_DIR = tmpRoot;
// Point OAIY_FFMPEG / OAIY_FFPROBE at the Node binary itself (works cross-platform
// without .cmd spawn restrictions on Windows) so runCommand can execute `-e ... -- ...`.
process.env.OAIY_FFMPEG = process.execPath;
process.env.OAIY_FFPROBE = process.execPath;
process.env.OAIY_FS_CONFINE = '1';
delete process.env.OAIY_FS_ALLOW_ABSOLUTE;

let pass = 0;
let fail = 0;
const check = (name: string, cond: boolean, detail?: string) => {
  if (cond) pass++;
  else {
    fail++;
    console.log('  FAIL:', name, detail ?? '');
  }
};

async function expectThrows(name: string, fn: () => Promise<unknown>, pattern: RegExp): Promise<void> {
  try {
    await fn();
    fail++;
    console.log('  FAIL:', name, '(did not throw)');
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (pattern.test(msg)) pass++;
    else {
      fail++;
      console.log('  FAIL:', name, `(unexpected error: ${msg})`);
    }
  }
}

async function main(): Promise<void> {
  try {
    const ffmpeg = await import('../src/node-host/ffmpeg');
    const { default: Database, normalizeSql } = await import('../src/node-host/sql');
    const realTmpRoot = fs.realpathSync(tmpRoot);

    // -----------------------------------------------------------------------
    // 1. FFmpeg confinement: disallowed flags, auto-probed manifests, CWD, protocol_whitelist
    // -----------------------------------------------------------------------
    const dummyMp4 = path.join(tmpRoot, 'clip.mp4');
    fs.writeFileSync(dummyMp4, Buffer.from([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70]));

    for (const badArgs of [
      ['-i', dummyMp4, '-filter_script', 'script.txt', 'out.mp4'],
      ['-i', dummyMp4, '-filter_script:v', 'script.txt', 'out.mp4'],
      ['-i', dummyMp4, '-filter_complex_script', 'script.txt', 'out.mp4'],
      ['-i', dummyMp4, '-filter_complex_script:0', 'script.txt', 'out.mp4'],
      ['-i', dummyMp4, '-fpre', 'p.ffpreset', 'out.mp4'],
      ['-i', dummyMp4, '-vpre', 'default', 'out.mp4'],
      ['-i', dummyMp4, '-apre', 'default', 'out.mp4'],
      ['-i', dummyMp4, '-spre', 'default', 'out.mp4'],
      ['-i', dummyMp4, '-preset', 'fast', 'out.mp4'],
      ['-i', dummyMp4, '-preset:v', 'fast', 'out.mp4'],
      ['-i', dummyMp4, '-sdp_file', 'out.sdp', 'out.mp4'],
      ['-i', dummyMp4, '-vstats_file', 'stats.log', 'out.mp4'],
      ['-i', dummyMp4, '-passlogfile', 'pass', 'out.mp4'],
      ['-i', dummyMp4, '-/vf', 'script.txt', 'out.mp4'],
      ['-safe', '0', '-i', dummyMp4, 'out.mp4'],
      ['-safe', 'false', '-i', dummyMp4, 'out.mp4'],
      ['-safe'],
    ]) {
      await expectThrows(
        `rejects confined ffmpeg flag ${badArgs.join(' ')}`,
        () => ffmpeg.runCommand({ command: 'ffmpeg', args: badArgs }),
        /not allowed in server mode/i,
      );
    }

    // Auto-probed ffconcat WITHOUT `-f concat` MUST still be inspected by checkConcatList.
    const evilConcat = path.join(tmpRoot, 'auto-concat.txt');
    fs.writeFileSync(evilConcat, "ffconcat version 1.0\nfile '/etc/passwd'\n");
    await expectThrows(
      'auto-probed ffconcat without -f concat rejects out-of-sandbox file entry',
      () => ffmpeg.runCommand({ command: 'ffmpeg', args: ['-i', evilConcat, 'out.mp4'] }),
      /outside the server fs sandbox/i,
    );

    const evilDirectiveConcat = path.join(tmpRoot, 'auto-concat-dir.txt');
    fs.writeFileSync(evilDirectiveConcat, "ffconcat version 1.0\noption safe 0\nfile 'clip.mp4'\n");
    await expectThrows(
      'auto-probed ffconcat rejects non-allowlisted directive',
      () => ffmpeg.runCommand({ command: 'ffmpeg', args: ['-i', evilDirectiveConcat, 'out.mp4'] }),
      /directive 'option' is not allowed/i,
    );

    // HLS / DASH playlists MUST be rejected for runCommand, getVideoInfo, and extractVideoFrames.
    const hlsFile = path.join(tmpRoot, 'evil.m3u8');
    fs.writeFileSync(hlsFile, '#EXTM3U\n#EXT-X-VERSION:3\nhttp://169.254.169.254/latest/meta-data/\n');
    const hlsNoHeader = path.join(tmpRoot, 'evil2.m3u8');
    fs.writeFileSync(hlsNoHeader, '#EXT-X-TARGETDURATION:10\n/etc/passwd\n');
    const dashFile = path.join(tmpRoot, 'evil.mpd');
    fs.writeFileSync(dashFile, '<?xml version="1.0"?>\n<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"></MPD>\n');

    for (const [label, f] of [
      ['#EXTM3U', hlsFile],
      ['#EXT-X-', hlsNoHeader],
      ['<MPD', dashFile],
    ] as const) {
      await expectThrows(
        `runCommand rejects ${label} playlist`,
        () => ffmpeg.runCommand({ command: 'ffmpeg', args: ['-i', f, 'out.mp4'] }),
        /playlist\/HLS|DASH\/MPD/i,
      );
      await expectThrows(
        `extractVideoFrames rejects ${label} playlist`,
        () => ffmpeg.extractVideoFrames(f, {}),
        /playlist\/HLS|DASH\/MPD/i,
      );
      await expectThrows(
        `getVideoInfo rejects ${label} playlist`,
        () => ffmpeg.getVideoInfo({ path: f }),
        /playlist\/HLS|DASH\/MPD/i,
      );
    }

    // Valid `-safe 1` + valid auto-probed concat: verify child CWD is pinned to TMP_ROOT
    // and `-protocol_whitelist file,pipe` is injected before `-i` (and caller's whitelist replaced).
    const goodConcat = path.join(tmpRoot, 'good-concat.txt');
    fs.writeFileSync(goodConcat, "ffconcat version 1.0\nfile 'clip.mp4'\n");
    const r1 = await ffmpeg.runCommand({
      command: 'ffmpeg',
      args: [
        '-e',
        'process.stdout.write(JSON.stringify({cwd:process.cwd(),argv:process.argv.slice(1)}))',
        '--',
        '-f',
        'concat',
        '-safe',
        '1',
        '-protocol_whitelist',
        'file,http,https,tcp,tls',
        '-i',
        goodConcat,
      ],
    });
    check('valid confined concat with -safe 1 spawns child cleanly', r1.code === 0, r1.stderr);
    const seen = JSON.parse(r1.stdout) as { cwd: string; argv: string[] };
    check(
      'runCommand pins child cwd to TMP_ROOT',
      fs.realpathSync(seen.cwd) === realTmpRoot,
      `cwd=${seen.cwd} expected=${realTmpRoot}`,
    );
    const wlPos = seen.argv.indexOf('-protocol_whitelist');
    const iPos = seen.argv.indexOf('-i');
    check(
      'runCommand replaces caller -protocol_whitelist with file,pipe before -i',
      wlPos !== -1 && seen.argv[wlPos + 1] === 'file,pipe' && iPos === wlPos + 2 && !seen.argv.includes('file,http,https,tcp,tls'),
      `argv=${JSON.stringify(seen.argv)}`,
    );

    // extractVideoFrames arg builder inserts `-protocol_whitelist file,pipe` before `-i` when confined.
    const { args: confinedArgs } = ffmpeg.buildExtractFrameArgs(dummyMp4, { maxFrames: 2 }, tmpRoot, true);
    const exWl = confinedArgs.indexOf('-protocol_whitelist');
    const exI = confinedArgs.indexOf('-i');
    check(
      'extractVideoFrames passes -protocol_whitelist file,pipe before -i when confined',
      exWl !== -1 && confinedArgs[exWl + 1] === 'file,pipe' && exI === exWl + 2,
      `args=${JSON.stringify(confinedArgs)}`,
    );
    const { args: unconfinedArgs } = ffmpeg.buildExtractFrameArgs(dummyMp4, { maxFrames: 2 }, tmpRoot, false);
    check(
      'extractVideoFrames omits -protocol_whitelist when unconfined',
      !unconfinedArgs.includes('-protocol_whitelist'),
    );

    // -----------------------------------------------------------------------
    // 2. Lexer-aware normalizeSql() (P2-6)
    // -----------------------------------------------------------------------
    const cases: Array<[string, string, string]> = [
      ['basic positional', 'SELECT * FROM t WHERE a = $1 AND b = $2', 'SELECT * FROM t WHERE a = ?1 AND b = ?2'],
      ['multi-digit positional', 'SELECT $10, $2', 'SELECT ?10, ?2'],
      ['single-quoted literal untouched', "SELECT '$1', 'cost $100', $1", "SELECT '$1', 'cost $100', ?1"],
      ['escaped single quote inside string', "SELECT 'it''s $1 here', $2", "SELECT 'it''s $1 here', ?2"],
      ['double-quoted identifier untouched', 'SELECT "$1", "a""$2" FROM t WHERE x = $1', 'SELECT "$1", "a""$2" FROM t WHERE x = ?1'],
      ['backtick-quoted identifier untouched', 'SELECT `$1`, `a``$2` FROM t WHERE x = $1', 'SELECT `$1`, `a``$2` FROM t WHERE x = ?1'],
      ['bracket-quoted identifier untouched', 'SELECT [$1], [col $2] FROM t WHERE x = $1', 'SELECT [$1], [col $2] FROM t WHERE x = ?1'],
      ['line comment untouched', 'SELECT $1 -- ignore $2\n + $3', 'SELECT ?1 -- ignore $2\n + ?3'],
      ['block comment untouched', 'SELECT /* ignore $1 */ $2 /* and $3 */', 'SELECT /* ignore $1 */ ?2 /* and $3 */'],
      ['trailing identifier boundary preserved', 'SELECT $1name, $2_id, $3$x, $4', 'SELECT $1name, $2_id, $3$x, ?4'],
      ['leading identifier boundary preserved', 'SELECT col$1, v$2 FROM t WHERE id = $1', 'SELECT col$1, v$2 FROM t WHERE id = ?1'],
      ['json path preserved', "SELECT json_extract(doc, '$.items[0]'), $.x FROM t WHERE id = $1", "SELECT json_extract(doc, '$.items[0]'), $.x FROM t WHERE id = ?1"],
    ];
    for (const [label, input, expected] of cases) {
      const got = normalizeSql(input);
      check(`normalizeSql: ${label}`, got === expected, `got=${JSON.stringify(got)} expected=${JSON.stringify(expected)}`);
    }

    // End-to-end Database test with $1 inside a string literal + real $1 bind parameter
    const db = await Database.load(':memory:');
    try {
      await db.execute('CREATE TABLE notes (id INTEGER PRIMARY KEY, label TEXT, body TEXT)');
      await db.execute("INSERT INTO notes (label, body) VALUES ('$1 literal', $1)", ['hello']);
      const rows = await db.select<Array<{ label: string; body: string }>>(
        "SELECT label, body FROM notes WHERE label = '$1 literal' /* $2 */ AND body = $1 -- $3",
        ['hello'],
      );
      check('Database.select preserves $1 string literal and binds positional $1', rows.length === 1 && rows[0].label === '$1 literal' && rows[0].body === 'hello');
    } finally {
      await db.close();
    }

    console.log(`node-host-hardening: ${pass} passed, ${fail} failed`);
    assert.equal(fail, 0);
  } finally {
    fs.rmSync(sandbox, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error('node-host-hardening crashed:', e);
  process.exit(1);
});
