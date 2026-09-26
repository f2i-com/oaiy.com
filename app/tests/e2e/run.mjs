// End-to-end: the sandbox in a real headless Chrome, served by Vite with the
// cross-origin isolation headers. `npm run test:e2e`
// CHROME=<path> picks the browser; otherwise the usual install locations.
import { existsSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const candidates = [
  process.env.CHROME,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean);
const executablePath = candidates.find((p) => existsSync(p));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}

const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const address = server.httpServer.address();
const base = `http://127.0.0.1:${address.port}`;
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-first-run'] });
let failures = 0;

async function check(name, fn) {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures++;
    console.log(`FAIL ${name}\n     ${error.message.split('\n').join('\n     ')}`);
  }
}

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

try {
  const page = await browser.newPage();
  page.on('console', (m) => process.env.VERBOSE && console.log('  [page]', m.text()));
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.goto(`${base}/tests/e2e/harness.html`);
  await page.waitForFunction(() => window.__bot !== undefined, { timeout: 30_000 });

  await check('the page is cross-origin isolated', async () => {
    expect(await page.evaluate(() => window.__bot.isolated), 'crossOriginIsolated is false');
  });

  await check('the shell runs, pipes and redirects into the virtual filesystem', async () => {
    const r = await page.evaluate(async () => {
      window.__bot.vfs.writeFile('/data.csv', 'city,temp\nParis,18\nRome,24\nOslo,9\n');
      return window.__bot.shell("echo alive; tail -n +2 data.csv | sort -t, -k2 -n | head -1 > coldest.txt; cat coldest.txt; mkdir -p out && echo hi > out/x.txt && ls out");
    });
    expect(r.report, `no shell report: ${JSON.stringify(r.outcome)}`);
    expect(r.report.stdout === 'alive\nOslo,9\nx.txt\n', `stdout: ${JSON.stringify(r.report.stdout)} stderr: ${r.report.stderr}`);
    expect(r.changes.includes('coldest.txt') && r.changes.includes('out/x.txt'), `changes: ${r.changes}`);
  });

  await check('the shell cannot leave the project', async () => {
    const r = await page.evaluate(() => window.__bot.shell('cd ../../.. && pwd && cat ../../etc/passwd; ls /'));
    expect(r.report.stdout.startsWith('/\n'), `stdout: ${r.report.stdout}`);
    expect(!r.report.stdout.includes('root:'), 'escaped');
  });

  await check('JavaScript: fs round trip, completion value, top-level await', async () => {
    const r = await page.evaluate(() =>
      window.__bot.run('js', "const fs = require('fs'); const rows = fs.readFileSync('data.csv','utf8').trim().split('\\n').slice(1); fs.writeFileSync('n.txt', String(rows.length)); await Promise.resolve(1); rows.map(r => r.split(',')[0]).join('|')"),
    );
    expect(!r.outcome.result?.error, `error: ${r.outcome.result?.error} ${r.outcome.failure ?? ''}`);
    expect(r.outcome.result?.value === undefined || r.outcome.result.value === 'Paris|Rome|Oslo', `value: ${r.outcome.result?.value}`);
    const n = await page.evaluate(() => window.__bot.vfs.readText('/n.txt'));
    expect(n === '3', `n.txt = ${n}`);
  });

  await check('JavaScript: a synchronous script reports its completion value', async () => {
    const r = await page.evaluate(() => window.__bot.run('js', '[1,2,3].map(x => x * x).join("+")'));
    expect(r.outcome.result?.value === '1+4+9', `value: ${JSON.stringify(r.outcome.result)}`);
  });

  await check('Python: open() over the project, writes saved back', async () => {
    const r = await page.evaluate(() =>
      window.__bot.run('python', "import statistics\nwith open('data.csv') as f:\n    temps=[int(l.split(',')[1]) for l in f.read().splitlines()[1:]]\nprint(statistics.mean(temps))\nwith open('mean.txt','w') as f:\n    f.write(str(round(statistics.mean(temps),2)))\n"),
    );
    expect(r.summary.exitCode === 0, `exit ${r.summary.exitCode}: ${r.summary.stderr}`);
    expect(r.summary.stdout.trim().startsWith('17'), `stdout: ${r.summary.stdout}`);
    const mean = await page.evaluate(() => window.__bot.vfs.readText('/mean.txt'));
    expect(mean === '17.0' || mean === '17', `mean.txt = ${mean}`);
  });

  await check('the network gate refuses a closed host from fetch and curl', async () => {
    const r = await page.evaluate(async () => {
      window.__bot.gate.setMode('blocked');
      const js = await window.__bot.run('js', "const r = await fetch('https://example.com'); r.status");
      const sh = await window.__bot.shell('curl -sS https://example.com');
      window.__bot.gate.setMode('open');
      return { js: js.outcome.result, sh: sh.report, status: window.__bot.gate.status() };
    });
    expect(/network gate is closed/.test(r.js?.error ?? ''), `js: ${JSON.stringify(r.js)}`);
    expect(r.sh.exit_code !== 0 && /closed/.test(r.sh.stderr), `shell: ${JSON.stringify(r.sh)}`);
    expect(r.status.blocked >= 2, `status: ${JSON.stringify(r.status)}`);
  });

  await check('a runaway program is stopped at its deadline', async () => {
    const t0 = Date.now();
    const r = await page.evaluate(() => window.__bot.run('js', 'for (;;) {}', 1500));
    expect(r.outcome.timedOut || r.outcome.result?.limit, `outcome: ${JSON.stringify(r.outcome)}`);
    expect(Date.now() - t0 < 10_000, 'took too long');
  });

  await check('the instruction budget stops a runaway program and says so', async () => {
    const r = await page.evaluate(() => window.__bot.budgetRun('for (;;) {}'));
    expect(!r.timedOut && r.result?.limit, `outcome: ${JSON.stringify(r)}`);
  });

  await check('node and python run nested inside the shell', async () => {
    const r = await page.evaluate(async () => {
      window.__bot.vfs.writeFile('/hello.js', "console.log('js says', process.argv.slice(2).join(' '))");
      window.__bot.vfs.writeFile('/hello.py', "import sys\nprint('py says', len(sys.argv))");
      return window.__bot.shell('node hello.js a b && python hello.py x && echo "print(6*7)" | python');
    });
    expect(r.report.stdout === 'js says a b\npy says 2\n42\n', `stdout: ${JSON.stringify(r.report.stdout)} stderr: ${r.report.stderr}`);
  });

  await check('python runs project scripts like CPython: sibling imports, the working folder, stdin, exit codes, -m', async () => {
    const r = await page.evaluate(async () => {
      const v = window.__bot.vfs;
      v.writeFile('/tools/helper.py', 'def double(n):\n    return n * 2\n', { parents: true });
      v.writeFile('/tools/main.py', 'import sys\nimport helper\nprint(helper.double(int(sys.argv[1])))\nif sys.argv[1] == "0":\n    sys.exit(4)\n', { parents: true });
      v.writeFile('/tools/data/n.txt', '5', { parents: true });
      v.writeFile('/tools/read.py', 'print(open("data/n.txt").read(), open("/tools/data/n.txt").read())\n', { parents: true });
      v.writeFile('/cfg.json', '{"b":1,"a":2}');
      const out = {};
      out.root = (await window.__bot.shell('python tools/main.py 21')).report;
      out.fromDir = (await window.__bot.shell('python main.py 5', '/tools')).report;
      out.exit = (await window.__bot.shell('python tools/main.py 0; echo "code=$?"')).report;
      out.relative = (await window.__bot.shell('python read.py', '/tools')).report;
      out.stdin = (await window.__bot.shell('printf "Ann" | python -c "name = input(); print(name.upper())"')).report;
      out.module = (await window.__bot.shell('python -m json.tool --sort-keys cfg.json')).report;
      out.stdlib = (await window.__bot.shell('python -c "import csv, datetime, glob, shutil, urllib.parse, difflib, logging, base64; print(datetime.date(2024, 3, 1) - datetime.date(2024, 2, 1))"')).report;
      out.hidden = (await window.__bot.shell('python -c "import os; print(sorted(os.listdir(chr(46))))"', '/tools')).report;
      return out;
    });
    const say = (k) => `${k}: ${JSON.stringify(r[k])}`;
    expect(r.root.stdout === '42\n', say('root'));
    expect(r.fromDir.stdout === '10\n', say('fromDir'));
    expect(r.exit.stdout === '0\ncode=4\n' && !r.exit.stderr, say('exit'));
    expect(r.relative.stdout === '5 5\n', say('relative'));
    expect(r.stdin.stdout === 'ANN\n', say('stdin'));
    expect(r.module.stdout.startsWith('{\n    "a": 2,'), say('module'));
    expect(r.stdlib.stdout === '29 days, 0:00:00\n', say('stdlib'));
    expect(r.hidden.stdout === "['data', 'helper.py', 'main.py', 'read.py']\n", say('hidden'));
  });

  await check('node requires project modules, JSON and node_modules packages, and runs ES modules', async () => {
    const r = await page.evaluate(async () => {
      const v = window.__bot.vfs;
      v.writeFile('/web/lib/math.js', 'module.exports = { twice: (n) => n * 2 };', { parents: true });
      v.writeFile('/web/config.json', '{"name":"demo"}', { parents: true });
      v.writeFile('/web/app.js', 'const m = require("./lib/math"); const c = require("./config.json"); console.log(m.twice(21), c.name, process.env.GREETING);', { parents: true });
      v.writeFile('/web/esm/util.mjs', 'export const k = 3;\nexport default function greet(n) { return "hi " + n; }\n', { parents: true });
      v.writeFile('/web/esm/main.mjs', 'import greet, { k } from "./util.mjs";\nconsole.log(greet("there"), k);\n', { parents: true });
      v.writeFile('/node_modules/pad/package.json', '{"main":"lib.js"}', { parents: true });
      v.writeFile('/node_modules/pad/lib.js', 'module.exports = (s, n) => String(s).padStart(n, "0");', { parents: true });
      const out = {};
      out.cjs = (await window.__bot.shell('GREETING=hello node web/app.js')).report;
      out.fromDir = (await window.__bot.shell('node app.js', '/web')).report;
      out.esm = (await window.__bot.shell('node web/esm/main.mjs')).report;
      out.pkg = (await window.__bot.shell('node -e "console.log(require(String.fromCharCode(112,97,100))(7, 3))"')).report;
      return out;
    });
    expect(r.cjs.stdout === '42 demo hello\n', `cjs: ${JSON.stringify(r.cjs)}`);
    expect(r.fromDir.stdout === '42 demo undefined\n', `fromDir: ${JSON.stringify(r.fromDir)}`);
    expect(r.esm.stdout === 'hi there 3\n', `esm: ${JSON.stringify(r.esm)}`);
    expect(r.pkg.stdout === '007\n', `pkg: ${JSON.stringify(r.pkg)}`);
  });

  await check('the shell has bash syntax: arrays, brace expansion, arithmetic, C-style for, "$@", <(...), getopts, trap', async () => {
    const script = [
      'arr=(a "b c" d); echo ${#arr[@]} "${arr[1]}" ${arr[-1]}',
      'declare -A m; m[x]=1; m[y]=2; for k in "${!m[@]}"; do printf "%s=%s " "$k" "${m[$k]}"; done; echo',
      'f() { for a in "$@"; do printf "<%s>" "$a"; done; echo; }; f one "two three"',
      'echo {1..3} {a,b}x',
      'i=2; ((i++)); ((j = i * 4)); echo $i $j $((j % 5))',
      'for ((n = 0; n < 3; n++)); do printf "%d" $n; done; echo',
      'diff <(echo same; echo old) <(echo same; echo new) | head -1',
      'opts() { OPTIND=1; while getopts "vo:" c; do case $c in v) printf "v ";; o) printf "o=%s " "$OPTARG";; esac; done; echo; }; opts -v -o out',
      'trap "echo bye" EXIT',
      'echo end',
    ].join('\n');
    const r = await page.evaluate((src) => window.__bot.shell(src), script);
    const want = '3 b c d\nx=1 y=2 \n<one><two three>\n1 2 3 ax bx\n3 12 2\n012\n2c2\nv o=out \nend\nbye\n';
    expect(r.report.stdout === want, `stdout: ${JSON.stringify(r.report.stdout)} stderr: ${r.report.stderr}`);
  });

  await check('the shell tools: checksums, jq, diff | patch, tar/zip/gzip round trips', async () => {
    const script = [
      'printf abc | sha256sum',
      'printf "" | md5sum',
      `echo '{"items":[{"n":"a","v":2},{"n":"b","v":5}]}' > data.json`,
      `jq -r '.items | map(select(.v > 3)) | .[].n' data.json`,
      `jq -c '.items |= map(.v * 10)' data.json`,
      'printf "one\\ntwo\\nthree\\n" > p1.txt; printf "one\\n2\\nthree\\n" > p2.txt',
      'diff -u p1.txt p2.txt > ch.diff; patch -s p1.txt < ch.diff; cmp p1.txt p2.txt && echo patched',
      'mkdir -p pack/in && echo deep > pack/in/f.txt && tar -czf pack.tgz pack && rm -r pack && tar -xzf pack.tgz && cat pack/in/f.txt',
      'zip -qr pack.zip pack && rm -r pack && unzip -q pack.zip && cat pack/in/f.txt',
      'echo zipped | gzip -c > z.gz && zcat z.gz',
      'echo "scale=2; 7/4" | bc',
    ].join('\n');
    const r = await page.evaluate((src) => window.__bot.shell(src), script);
    const want = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  -\nd41d8cd98f00b204e9800998ecf8427e  -\nb\n{"items":[20,50]}\npatched\ndeep\ndeep\nzipped\n1.75\n';
    expect(r.report.stdout === want, `stdout: ${JSON.stringify(r.report.stdout)} stderr: ${r.report.stderr}`);
  });

  await check('git keeps a local repository: commit, branch, merge, conflict, stash, log', async () => {
    const script = [
      'mkdir -p proj && cd proj && git init -q',
      'printf "a\\nb\\nc\\n" > f.txt && git add . && git commit -qm first',
      'git switch -qc side && sed -i "s/^a$/A/" f.txt && git commit -qam side',
      'git switch -q main && sed -i "s/^c$/C/" f.txt && git commit -qam main',
      'git merge -q side >/dev/null && tr "\\n" " " < f.txt && echo',
      'git log --format=%s | tr "\\n" " " && echo',
      'git switch -qc x && echo X > f.txt && git commit -qam x && git switch -q main && echo M > f.txt && git commit -qam m',
      'git merge x > /dev/null; echo "conflict=$?"; git status --short',
      'echo resolved > f.txt && git add f.txt && git commit -qm resolved && git status --short && echo clean',
      'echo wip >> f.txt && git stash -q && cat f.txt && git stash pop -q >/dev/null && tail -1 f.txt',
      'git diff --stat | tail -1',
      'cd / && git status 2>&1 | head -1',
    ].join('\n');
    const r = await page.evaluate((src) => window.__bot.shell(src), script);
    const lines = r.report.stdout.split('\n');
    const expected = [
      'A b C ',
      'Merge branch \'side\' main side first ',
      'conflict=1',
      'UU f.txt',
      'clean',
      'resolved',
      'wip',
      ' 1 file changed, 1 insertion(+)',
      'fatal: not a git repository (or any of the parent directories): .git',
    ];
    expect(JSON.stringify(lines.slice(0, expected.length)) === JSON.stringify(expected), `stdout: ${JSON.stringify(r.report.stdout)} stderr: ${r.report.stderr}`);
  });

  await check('media tools: compose pictures into a video, read it, take frames, split it, rejoin it with music, and mix sound alone', async () => {
    const r = await page.evaluate(async () => {
      const bot = window.__bot;
      // Two coloured pictures and a tone, made here: no fixtures.
      const picture = async (color) => {
        const c = new OffscreenCanvas(320, 240);
        const g = c.getContext('2d');
        g.fillStyle = color;
        g.fillRect(0, 0, 320, 240);
        return new Uint8Array(await (await c.convertToBlob({ type: 'image/png' })).arrayBuffer());
      };
      bot.vfs.writeFile('/media/red.png', await picture('#ff0000'), { parents: true });
      bot.vfs.writeFile('/media/blue.png', await picture('#0000ff'), { parents: true });
      const rate = 48000;
      const n = rate * 3;
      const wav = new DataView(new ArrayBuffer(44 + n * 2));
      const text = (at, s) => [...s].forEach((ch, i) => wav.setUint8(at + i, ch.charCodeAt(0)));
      text(0, 'RIFF'); wav.setUint32(4, 36 + n * 2, true); text(8, 'WAVE'); text(12, 'fmt '); wav.setUint32(16, 16, true);
      wav.setUint16(20, 1, true); wav.setUint16(22, 1, true); wav.setUint32(24, rate, true); wav.setUint32(28, rate * 2, true);
      wav.setUint16(32, 2, true); wav.setUint16(34, 16, true); text(36, 'data'); wav.setUint32(40, n * 2, true);
      for (let i = 0; i < n; i++) wav.setInt16(44 + i * 2, Math.sin((2 * Math.PI * 440 * i) / rate) * 12000, true);
      bot.vfs.writeFile('/media/tone.wav', new Uint8Array(wav.buffer));
      // The colour at the middle of a saved frame.
      const colour = async (path) => {
        const bitmap = await createImageBitmap(new Blob([bot.vfs.readBytes(path)]));
        const c = new OffscreenCanvas(bitmap.width, bitmap.height);
        const g = c.getContext('2d');
        g.drawImage(bitmap, 0, 0);
        const [red, , blue] = g.getImageData(bitmap.width / 2, bitmap.height / 2, 1, 1).data;
        return { size: `${bitmap.width}x${bitmap.height}`, is: red > 150 && blue < 100 ? 'red' : blue > 150 && red < 100 ? 'blue' : `rgb ${red},${blue}` };
      };
      const out = {};
      out.compose = await bot.tool('media_compose', { output: 'media/slides.mp4', fps: 30, clips: [{ path: 'media/red.png', duration: 2 }, { path: 'media/blue.png', duration: 2 }] });
      out.info = await bot.tool('media_info', { path: 'media/slides.mp4' });
      out.frames = await bot.tool('video_frames', { path: 'media/slides.mp4', times: [0.5, 3.5], last: true });
      const saved = bot.vfs.walk('/media/slides-frames').entries.filter((e) => e.type === 'file').map((e) => `/${e.path}`).sort();
      out.colours = [];
      for (const f of saved) out.colours.push(await colour(f));
      out.split = await bot.tool('video_split', { path: 'media/slides.mp4', at: [2] });
      out.part2 = await bot.tool('media_info', { path: 'media/slides-part-2.mp4' });
      out.rejoin = await bot.tool('media_compose', {
        output: 'media/rejoined.mp4',
        clips: [{ path: 'media/slides-part-2.mp4' }, { path: 'media/slides-part-1.mp4', fade_out: 0.5 }],
        audio: [{ path: 'media/tone.wav', at: 0.5, volume: 0.5, fade_out: 1, loop: true }],
      });
      out.rejoinInfo = await bot.tool('media_info', { path: 'media/rejoined.mp4' });
      await bot.tool('video_frames', { path: 'media/rejoined.mp4', frames: [15], output_dir: 'media/check' });
      out.firstColour = await colour('/media/check/frame-00015.png');
      out.mix = await bot.tool('media_compose', { output: 'media/mix.wav', audio: [{ path: 'media/tone.wav', volume: 0.8 }, { path: 'media/tone.wav', at: 1, volume: 1, duck: 0.2 }] });
      out.mixInfo = await bot.tool('media_info', { path: 'media/mix.wav' });
      out.bad = await bot.tool('media_compose', { output: 'media/x.mp3', audio: [{ path: 'media/tone.wav' }] });
      // The sound, measured: level (RMS) of a stretch of a WAV the tool wrote.
      const level = async (path, from, to) => {
        const ctx = new OfflineAudioContext(1, 1, 48000);
        const buffer = await ctx.decodeAudioData(bot.vfs.readBytes(path).slice().buffer);
        const data = buffer.getChannelData(0).subarray(Math.floor(from * buffer.sampleRate), Math.floor(to * buffer.sampleRate));
        let sum = 0;
        for (const v of data) sum += v * v;
        return Math.sqrt(sum / Math.max(1, data.length));
      };
      await bot.tool('media_compose', { output: 'media/placed.wav', duration: 2, audio: [{ path: 'media/tone.wav', at: 0.5 }] });
      out.placed = { before: await level('/media/placed.wav', 0, 0.45), after: await level('/media/placed.wav', 0.55, 1.5) };
      // A silent voice that ducks: the music drops to 20% while it plays (1 s to 2 s), and comes back.
      await bot.tool('media_compose', { output: 'media/ducked.wav', duration: 3, audio: [{ path: 'media/tone.wav' }, { path: 'media/tone.wav', at: 1, end: 1, volume: 0, duck: 0.2 }] });
      out.ducked = { open: await level('/media/ducked.wav', 0.1, 0.7), ducked: await level('/media/ducked.wav', 1.3, 1.7), back: await level('/media/ducked.wav', 2.4, 2.9) };
      // Four loud tones at once would clip: the mix is turned down to fit.
      out.loud = await bot.tool('media_compose', { output: 'media/loud.wav', audio: [1, 2, 3, 4].map(() => ({ path: 'media/tone.wav', volume: 2 })) });
      return out;
    });
    const all = JSON.stringify(r, null, 1).slice(0, 3000);
    expect(!r.compose.isError && r.compose.content.includes('4 s') && r.compose.content.includes('320×240 at 30 fps'), all);
    expect(r.info.content.includes('duration 4 s') && r.info.content.includes('video 320×240, 30 fps, 120 frames') && r.info.content.includes('no sound'), all);
    expect(!r.frames.isError && JSON.stringify(r.colours.map((c) => c.is)) === '["red","blue","blue"]' && r.colours.every((c) => c.size === '320x240'), all);
    expect(r.frames.content.includes('frame 119'), all);
    expect(!r.split.isError && r.split.content.includes('into 2 parts') && r.part2.content.includes('duration 2 s'), all);
    // AAC's encoder delay adds a few milliseconds to the container's duration.
    expect(!r.rejoin.isError && /duration 4(\.0[0-4]\d*)? s/.test(r.rejoinInfo.content) && /audio (aac|opus)/.test(r.rejoinInfo.content), all);
    expect(r.firstColour.is === 'blue', all);
    expect(!r.mix.isError && r.mix.content.includes('4 s') && r.mixInfo.content.includes('audio') && r.mixInfo.content.includes('2 channels'), all);
    expect(r.bad.isError && r.bad.content.includes('.wav or .m4a'), all);
    expect(r.placed.before < 0.001 && r.placed.after > 0.2, `placement: ${JSON.stringify(r.placed)}`);
    const ratio = r.ducked.ducked / r.ducked.open;
    expect(ratio > 0.17 && ratio < 0.23 && Math.abs(r.ducked.back / r.ducked.open - 1) < 0.05, `ducking: ${JSON.stringify(r.ducked)} ratio ${ratio}`);
    expect(!r.loud.isError && r.loud.content.includes('turned down'), `limiting: ${r.loud.content}`);
  });
} finally {
  await browser.close();
  await server.close();
}

console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
