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
    // 1b. Wave-3 FFmpeg & ffprobe bypass regressions (P0 #1 & #2)
    // -----------------------------------------------------------------------
    // ffprobe unconditionally prepends `-protocol_whitelist file,pipe` before all args,
    // strips caller's `-protocol_whitelist`, and inspects positional inputs.
    const ffprobeGoodArgs = ffmpeg.buildRunCommandArgs('ffprobe', [
      '-v',
      'error',
      '-protocol_whitelist',
      'file,http,https',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      dummyMp4,
    ]);
    check(
      'ffprobe prepends -protocol_whitelist file,pipe before all args and strips caller whitelist',
      ffprobeGoodArgs[0] === '-protocol_whitelist' &&
        ffprobeGoodArgs[1] === 'file,pipe' &&
        !ffprobeGoodArgs.includes('file,http,https') &&
        ffprobeGoodArgs[ffprobeGoodArgs.length - 1] === fs.realpathSync(dummyMp4),
      `args=${JSON.stringify(ffprobeGoodArgs)}`,
    );

    // ffprobe positional inputs pointing to playlists or auto-probed/explicit concat lists MUST be rejected.
    await expectThrows(
      'ffprobe rejects positional HLS playlist input',
      () => ffmpeg.runCommand({ command: 'ffprobe', args: ['-v', 'error', '-show_format', hlsFile] }),
      /playlist\/HLS/i,
    );
    await expectThrows(
      'ffprobe rejects bare positional DASH playlist input',
      () => ffmpeg.runCommand({ command: 'ffprobe', args: [dashFile] }),
      /DASH\/MPD/i,
    );
    await expectThrows(
      'ffprobe rejects positional auto-probed ffconcat input',
      () => ffmpeg.runCommand({ command: 'ffprobe', args: ['-show_format', evilConcat] }),
      /outside the server fs sandbox/i,
    );
    await expectThrows(
      'ffprobe rejects positional explicit -f concat input even when -f concat comes after input',
      () => ffmpeg.runCommand({ command: 'ffprobe', args: [evilConcat, '-f', 'concat', '-safe', '1'] }),
      /outside the server fs sandbox/i,
    );
    await expectThrows(
      'ffprobe rejects positional playlist after --',
      () => ffmpeg.runCommand({ command: 'ffprobe', args: ['--', hlsFile] }),
      /playlist\/HLS/i,
    );

    // Filtergraph bypasses: leading `=` delimiter, `@instance` suffixes, and file-path filter options.
    for (const [label, fg] of [
      ['drawtext=textfile=...', 'drawtext=textfile=/etc/passwd:fontsize=24'],
      ['drawtext=fontfile=...', 'drawtext=fontfile=/etc/passwd:text=hi'],
      ['movie@x=...', 'movie@x=/etc/passwd'],
      ['[0:v]movie@id=...', '[0:v]movie@id=/etc/passwd[out]'],
      ['amovie@a1=...', 'scale=640:480,amovie@a1=/etc/passwd'],
      ['subtitles@s=...', 'subtitles@s=/etc/passwd'],
      ['lut3d@l=...', 'lut3d@l=/etc/passwd'],
      ['metadata=mode=print:file=...', 'metadata=mode=print:file=/tmp/leak.txt'],
      ['ametadata=file=...', 'ametadata=file=/tmp/leak.txt'],
      ['psnr=stats_file=...', 'psnr=stats_file=/tmp/stats.log'],
      ['libvmaf=log_path=...', 'libvmaf=log_path=/tmp/vmaf.json'],
      ['movie=filename=...', 'foo=filename=/etc/passwd'],
      ['dnn_processing=model_path=...', 'dnn_processing=model_path=/etc/passwd'],
      ['sofalizer=sofa=...', 'sofalizer=sofa=/etc/passwd'],
    ] as const) {
      await expectThrows(
        `rejects confined filtergraph ${label}`,
        () => ffmpeg.runCommand({ command: 'ffmpeg', args: ['-i', dummyMp4, '-vf', fg, 'out.mp4'] }),
        /filtergraph file-reading expression is not allowed/i,
      );
    }

    // Benign filtergraphs still pass buildRunCommandArgs cleanly.
    const benignArgs = ffmpeg.buildRunCommandArgs('ffmpeg', [
      '-i',
      dummyMp4,
      '-vf',
      'fps=1/2,scale=640:480,drawtext=text=hello:fontsize=24',
      'out.mp4',
    ]);
    check('benign filtergraph passes confined arg check', benignArgs.includes('fps=1/2,scale=640:480,drawtext=text=hello:fontsize=24'));

    // Concat list parser hardening: backslash/quote escapes & ENOENT fail-closed.
    const concatEscapes: Array<[string, string, RegExp]> = [
      ['backslash protocol escape', "ffconcat version 1.0\nfile http\\://169.254.169.254/\n", /disallowed quotes or backslashes/i],
      ['split single-quote escape', "ffconcat version 1.0\nfile 'http'://169.254.169.254/\n", /malformed quoting/i],
      ['single-quote backslash splice', "ffconcat version 1.0\nfile 'clip'\\''..'/etc/passwd'\n", /disallowed quotes or backslashes/i],
      ['backslash inside single quotes', "ffconcat version 1.0\nfile 'clip\\evil.mp4'\n", /disallowed quotes or backslashes/i],
      ['double-quoted entry', 'ffconcat version 1.0\nfile "clip.mp4"\n', /disallowed quotes or backslashes/i],
      ['unquoted whitespace', 'ffconcat version 1.0\nfile clip.mp4 extra\n', /disallowed quotes or backslashes/i],
      ['missing concat entry fails closed on ENOENT', "ffconcat version 1.0\nfile 'does-not-exist.mp4'\n", /could not be inspected/i],
    ];
    for (const [idx, [label, content, errPat]] of concatEscapes.entries()) {
      const f = path.join(tmpRoot, `concat-escape-${idx}.txt`);
      fs.writeFileSync(f, content);
      await expectThrows(
        `checkConcatList rejects ${label}`,
        () => ffmpeg.runCommand({ command: 'ffmpeg', args: ['-f', 'concat', '-safe', '1', '-i', f, 'out.mp4'] }),
        errPat,
      );
    }

    // >4 KB padded `.mpd` (leading XML comment) MUST still be detected by inspectConfinedInput.
    const paddedMpd = path.join(tmpRoot, 'padded.mpd');
    const xmlPadding = `<!-- ${'A'.repeat(16 * 1024)} -->\n`;
    fs.writeFileSync(paddedMpd, `<?xml version="1.0"?>\n${xmlPadding}<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"></MPD>\n`);
    await expectThrows(
      'runCommand rejects >4 KB comment-padded .mpd playlist',
      () => ffmpeg.runCommand({ command: 'ffmpeg', args: ['-i', paddedMpd, 'out.mp4'] }),
      /DASH\/MPD/i,
    );
    await expectThrows(
      'ffprobe rejects >4 KB comment-padded positional .mpd playlist',
      () => ffmpeg.runCommand({ command: 'ffprobe', args: [paddedMpd] }),
      /DASH\/MPD/i,
    );

    // -----------------------------------------------------------------------
    // 1c. ScriptHost early-death respawn cap & backoff (P1 #7)
    // -----------------------------------------------------------------------
    const { ScriptHost } = await import('../src/zipp/script-host');
    const crashWorker = path.join(sandbox, 'crash-worker.mjs');
    fs.writeFileSync(crashWorker, 'process.exit(2);\n');
    const fakeArtifact = {
      module: {} as WebAssembly.Module,
      scriptWorkerEntry: crashWorker,
      workflowWorkerEntry: crashWorker,
      identity: {
        name: 'zipp' as const,
        release: 'test',
        version: '0.0.0',
        revision: '0',
        variant: 'test',
        bundle: 'test',
        wasmSha256: '0'.repeat(64),
        glueSha256: '0'.repeat(64),
        languages: ['javascript'],
      },
    };
    const crashHost = new ScriptHost(fakeArtifact, {
      keepWarm: true,
      maxRespawnFailures: 3,
      respawnBackoffMs: 15,
    });
    try {
      await crashHost.warm().catch(() => undefined);
      // Wait long enough for 2 backoff retries (15 ms + 30 ms) plus margin;
      // if respawn storm were uncapped, instance would keep climbing indefinitely.
      await new Promise((r) => setTimeout(r, 350));
      const instancesAtCap = crashHost.instance;
      check(
        'ScriptHost stops keepWarm auto-respawning after maxRespawnFailures early deaths',
        instancesAtCap === 3,
        `instance=${instancesAtCap}`,
      );
      await new Promise((r) => setTimeout(r, 150));
      check(
        'ScriptHost does not spawn further workers after stopping auto-respawn',
        crashHost.instance === instancesAtCap,
        `instance=${crashHost.instance} expected=${instancesAtCap}`,
      );
      // Next job lazily attempts a spawn and reports host error instead of hanging.
      const lazyResp = await crashHost.run({ v: 1, jobs: [{ id: 'j1', mode: 'program', source: '1' }] });
      check(
        'ScriptHost lazily retries spawn on next job after auto-respawn pause',
        'results' in lazyResp && lazyResp.results[0]?.ok === false && lazyResp.results[0]?.errorKind === 'host' && crashHost.instance === 4,
        `resp=${JSON.stringify(lazyResp)} instance=${crashHost.instance}`,
      );
    } finally {
      await crashHost.shutdown();
    }

    // -----------------------------------------------------------------------
    // 1d. flow-io.ts zip-bomb & entry-count caps (P2 #14)
    // -----------------------------------------------------------------------
    const { zipSync, strToU8 } = await import('fflate');
    const { loadFlowFile } = await import('../src/flow-io');

    // Too many entries (> 512) rejected.
    const manyEntries: Record<string, Uint8Array> = {
      'manifest.json': strToU8(JSON.stringify({ name: 'many', entryFlow: 'flow.json' })),
      'flow.json': strToU8(JSON.stringify({ nodes: [], edges: [] })),
    };
    for (let i = 0; i < 520; i++) manyEntries[`extra-${i}.txt`] = strToU8('x');
    const manyZipPath = path.join(sandbox, 'many.oaiy');
    fs.writeFileSync(manyZipPath, zipSync(manyEntries));
    await expectThrows(
      'loadFlowFile rejects .oaiy package with >512 entries',
      async () => loadFlowFile(manyZipPath),
      /maximum entry count/i,
    );

    // Highly-compressible >32 MiB JSON zip bomb rejected without OOM.
    const bombJson = `"${'A'.repeat(33 * 1024 * 1024)}"`;
    const bombZipPath = path.join(sandbox, 'bomb.oaiy');
    fs.writeFileSync(
      bombZipPath,
      zipSync({
        'manifest.json': strToU8(JSON.stringify({ name: 'bomb', entryFlow: 'flow.json' })),
        'flow.json': strToU8(bombJson),
      }),
    );
    await expectThrows(
      'loadFlowFile rejects .oaiy package exceeding 32 MiB uncompressed JSON cap',
      async () => loadFlowFile(bombZipPath),
      /maximum uncompressed JSON size/i,
    );

    // Non-.json entries are skipped by the filter while valid JSON package loads cleanly.
    const goodPkgPath = path.join(sandbox, 'good.oaiy');
    fs.writeFileSync(
      goodPkgPath,
      zipSync({
        'manifest.json': strToU8(JSON.stringify({ id: 'pkg-1', name: 'Good', entryFlow: 'flow.json' })),
        'flow.json': strToU8(JSON.stringify({ nodes: [], edges: [] })),
        'assets/huge.bin': new Uint8Array(64 * 1024),
      }),
    );
    const loadedPkg = loadFlowFile(goodPkgPath);
    check('loadFlowFile loads valid .oaiy package while skipping non-JSON entries', loadedPkg.name === 'Good' && loadedPkg.package?.id === 'pkg-1');

    // -----------------------------------------------------------------------
    // 1e. zipp/profile.ts block-hoisted var & Annex-B function detection (P2 #15)
    // -----------------------------------------------------------------------
    const { scanTopLevelDeclarations, parseScriptProfile, sha256Hex } = await import('../src/zipp/profile');
    const hoistedCases: Array<[string, string, string]> = [
      ['BlockStatement var', '{ var host = 1; }', 'host'],
      ['IfStatement consequent var', 'if (true) { var __emit = 1; }', '__emit'],
      ['IfStatement alternate Annex-B function', 'if (false) {} else { function __replies() {} }', '__replies'],
      ['ForStatement init var', 'for (var host = 0; host < 1; host++) {}', 'host'],
      ['ForInStatement left var', 'for (var __oaiyConsole in {}) {}', '__oaiyConsole'],
      ['ForOfStatement left destructuring var', 'for (var { host } of []) {}', 'host'],
      ['WhileStatement body var', 'while (false) var host = 1;', 'host'],
      ['DoWhileStatement body var', 'do { var __emit = 1; } while (false);', '__emit'],
      ['SwitchStatement case var', 'switch (1) { case 1: var host = 1; break; }', 'host'],
      ['TryStatement catch var', 'try {} catch (e) { var __replies = 1; }', '__replies'],
      ['TryStatement finally Annex-B function', 'try {} finally { function host() {} }', 'host'],
      ['LabeledStatement var', 'lbl: var host = 1;', 'host'],
      ['WithStatement var', 'with ({}) { var __emit = 1; }', '__emit'],
    ];
    for (const [label, preamble, forbidden] of hoistedCases) {
      const decls = scanTopLevelDeclarations(preamble);
      check(`scanTopLevelDeclarations collects ${label}`, decls.names.includes(forbidden) && !decls.lexical.includes(forbidden), `decls=${JSON.stringify(decls)}`);
      await expectThrows(
        `parseScriptProfile rejects ${label}`,
        async () => parseScriptProfile({ v: 1, preamble, preambleSha256: sha256Hex(preamble) }),
        new RegExp(`declares "${forbidden}"`),
      );
    }

    // Block-scoped let/const/class and vars inside functions/classes MUST NOT be treated as top-level.
    const scopedDecls = scanTopLevelDeclarations(
      '{ let host = 1; const __emit = 2; class __replies {} } function ok() { var host = 1; function __emit() {} } class C { m() { var __oaiyConsole = 1; } }',
    );
    check(
      'scanTopLevelDeclarations ignores block-scoped lexicals and function/class-scoped vars',
      scopedDecls.names.join(',') === 'ok,C' && scopedDecls.lexical.join(',') === 'C',
      `decls=${JSON.stringify(scopedDecls)}`,
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
