// The built desktop app, driven through WebView2's remote debugging port
// (Windows): it is cross-origin isolated, runs code in the sandbox, loads the
// SoftN runtime into a sandboxed frame, saves an export as a download, goes to
// the tray when closed and keeps working there, comes back when started again,
// and saves everything when it quits. The portable build keeps all its data
// beside itself, and runs next to the installed one.
//   npm run desktop:build && npm run desktop:portable && node tests/e2e/desktop.mjs
import { execFileSync, spawn } from 'node:child_process';
import { copyFileSync, cpSync, existsSync, mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import puppeteer from 'puppeteer-core';

if (process.platform !== 'win32') {
  console.log('skipped: the desktop check drives WebView2, on Windows');
  process.exit(0);
}
const exe = 'src-tauri/target/release/bot-computer.exe';
if (!existsSync(exe)) {
  console.error(`no ${exe}; run npm run desktop:build first`);
  process.exit(2);
}
const PORT = 9333;
// A data folder of its own, so the check neither sees nor touches the person's projects.
const data = mkdtempSync(join(tmpdir(), 'bot-computer-desktop-'));
const downloads = mkdtempSync(join(tmpdir(), 'bot-computer-downloads-'));
const env = {
  ...process.env,
  // The environment's arguments replace the app's own, so repeat those (no throttling in the tray).
  WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${PORT} --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows`,
  WEBVIEW2_USER_DATA_FOLDER: data,
};

let app = null;
async function launch(path = exe, withEnv = env, port = PORT) {
  app = spawn(path, [], { env: withEnv, stdio: 'ignore' });
  for (let i = 0; i < 120; i++) {
    try {
      const browser = await puppeteer.connect({ browserURL: `http://127.0.0.1:${port}`, defaultViewport: null });
      const pages = await browser.pages();
      const page = pages.find((p) => p.url().includes('botcomputer'));
      if (page) {
        await page.waitForSelector('.tree-row', { timeout: 60_000 });
        return { browser, page };
      }
      browser.disconnect();
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error('the app did not open a page with the debugging port');
}

/** Quit as the tray menu does: the page saves everything, then the app exits. */
async function quit(browser, page, proc = app) {
  const exited = new Promise((r) => proc.once('exit', r));
  await page.evaluate(() => { void window.__botComputerBeforeQuit(); }).catch(() => {});
  const timer = setTimeout(() => proc.kill(), 15_000);
  await exited;
  clearTimeout(timer);
  browser.disconnect();
  // WebView2 lets go of its data folder a moment after the app exits.
  await new Promise((r) => setTimeout(r, 1500));
}

let failures = 0;
const check = async (name, fn) => {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures++;
    console.log(`FAIL ${name}\n     ${String(error.message).split('\n').join('\n     ')}`);
  }
};
const expect = (c, m) => {
  if (!c) throw new Error(m);
};
const ps = (command) => execFileSync('powershell', ['-NoProfile', '-Command', command]).toString().trim();
/** Whether a process's main window shows (a window in the tray does not). */
const windowOf = (pid) => {
  const handle = ps(`(Get-Process -Id ${pid}).MainWindowHandle`);
  return () => ps(`Add-Type -Name W -Namespace U -MemberDefinition '[DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);'; [U.W]::IsWindowVisible([IntPtr]${handle})`) === 'True';
};
/** Wait until `test()` holds (a window shown or hidden), for up to 10 s. */
async function until(test) {
  for (let i = 0; i < 40; i++) {
    if (test()) return true;
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}
/** WebView2's helper processes let go of a data folder a little after the app exits. */
async function removeSoon(dir) {
  for (let i = 0; i < 20; i++) {
    try {
      rmSync(dir, { recursive: true, force: true });
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 500));
    }
  }
}
async function terminal(page, command, waitFor) {
  await page.click('.term-input');
  await page.type('.term-input', command);
  await page.keyboard.press('Enter');
  await page.waitForFunction((w) => document.querySelector('.term-out')?.textContent.includes(w), { timeout: 60_000 }, waitFor).catch(async () => {
    throw new Error(`the terminal never showed "${waitFor}":\n${(await page.$eval('.term-out', (el) => el.textContent)).slice(-600)}`);
  });
}

try {
  let { browser, page } = await launch();

  await check('the app is served from its own origin, cross-origin isolated', async () => {
    const info = await page.evaluate(() => ({ href: location.href, isolated: crossOriginIsolated, sab: typeof SharedArrayBuffer, sw: navigator.serviceWorker?.controller ?? null }));
    expect(info.href.startsWith('http://botcomputer.localhost/'), JSON.stringify(info));
    expect(info.isolated && info.sab === 'function', JSON.stringify(info));
    expect(info.sw === null, 'a service worker controls the desktop page');
  });

  await check('the sandbox runs Python and the shell', async () => {
    await terminal(page, 'python -c "print(6*7)"', '42');
    await terminal(page, 'echo kept > keep.txt && git init -q && git add . && git commit -qm first && git log --oneline | wc -l', '1\n');
  });

  await check('the SoftN runtime loads in a sandboxed, opaque-origin frame', async () => {
    const ready = await page.evaluate(() => new Promise((resolve) => {
      const frame = document.createElement('iframe');
      frame.setAttribute('sandbox', 'allow-scripts');
      frame.setAttribute('credentialless', '');
      frame.style.display = 'none';
      const done = (v) => {
        window.removeEventListener('message', onMessage);
        frame.remove();
        resolve(v);
      };
      const onMessage = (e) => {
        if (e.source === frame.contentWindow && e.data?.type === 'formlogic:ready') done('ready');
      };
      window.addEventListener('message', onMessage);
      setTimeout(() => done('timed out'), 20_000);
      frame.src = '/softn/index.html?v=check';
      document.body.append(frame);
    }));
    expect(ready === 'ready', `the runtime ${ready}`);
  });

  await check('WebView2 edits video: H.264 and AAC encode, a picture and a tone composed into an MP4', async () => {
    const r = await page.evaluate(async () => {
      // The editor as the app loads it: the chunk its own script imports.
      const main = document.querySelector('script[type="module"][src]').src;
      const chunk = (await (await fetch(main)).text()).match(/\.\/(edit-[\w-]+\.js)/)?.[1];
      if (!chunk) return { error: 'no editor chunk in the app' };
      const edit = await import(new URL(chunk, main).href);
      const c = new OffscreenCanvas(320, 240);
      const g = c.getContext('2d');
      g.fillStyle = '#00ff00';
      g.fillRect(0, 0, 320, 240);
      const png = new Uint8Array(await (await c.convertToBlob({ type: 'image/png' })).arrayBuffer());
      const rate = 48000;
      const n = rate;
      const wav = new DataView(new ArrayBuffer(44 + n * 2));
      const text = (at, t) => [...t].forEach((ch, i) => wav.setUint8(at + i, ch.charCodeAt(0)));
      text(0, 'RIFF'); wav.setUint32(4, 36 + n * 2, true); text(8, 'WAVE'); text(12, 'fmt '); wav.setUint32(16, 16, true);
      wav.setUint16(20, 1, true); wav.setUint16(22, 1, true); wav.setUint32(24, rate, true); wav.setUint32(28, rate * 2, true);
      wav.setUint16(32, 2, true); wav.setUint16(34, 16, true); text(36, 'data'); wav.setUint32(40, n * 2, true);
      for (let i = 0; i < n; i++) wav.setInt16(44 + i * 2, Math.sin((2 * Math.PI * 440 * i) / rate) * 12000, true);
      const made = await edit.compose({ clips: [{ bytes: png, name: 'g.png', duration: 1 }], audio: [{ bytes: new Uint8Array(wav.buffer), name: 't.wav' }], output: 'mp4', fps: 30 });
      const info = await edit.mediaInfo(made.bytes, 'out.mp4');
      return { videoCodec: made.videoCodec, audioCodec: made.audioCodec, info };
    });
    expect(!r.error && r.videoCodec === 'avc' && r.audioCodec === 'aac', JSON.stringify(r));
    expect(r.info.video.frames === 30 && r.info.video.width === 320 && r.info.audio?.channels === 2, JSON.stringify(r));
  });

  await check('Export .zip saves a download', async () => {
    const cdp = await page.createCDPSession();
    await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: downloads });
    await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent === 'Export .zip').click());
    for (let i = 0; i < 40 && !readdirSync(downloads).some((f) => f.endsWith('.zip')); i++) await new Promise((r) => setTimeout(r, 250));
    const files = readdirSync(downloads);
    expect(files.some((f) => f.endsWith('.zip')), `downloads: ${files}`);
  });

  await check('closing or minimizing the window leaves it running in the tray, still working; starting it again brings it back', async () => {
    const handle = ps(`(Get-Process -Id ${app.pid}).MainWindowHandle`);
    const visible = () => ps(`Add-Type -Name W -Namespace U -MemberDefinition '[DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);'; [U.W]::IsWindowVisible([IntPtr]${handle})`);
    expect(visible() === 'True', 'the window is not visible to begin with');
    // WM_CLOSE, as the window's close button sends it.
    ps(`(Get-Process -Id ${app.pid}).CloseMainWindow() | Out-Null`);
    expect(await until(() => visible() === 'False'), 'the window is still showing after close');
    expect(app.exitCode === null, 'the app exited on close');
    // Hidden, timers still run at speed: the agent keeps working in the background.
    const ticks = await page.evaluate(() => new Promise((resolve) => {
      let n = 0;
      const t = setInterval(() => n++, 100);
      setTimeout(() => {
        clearInterval(t);
        resolve(n);
      }, 3000);
    }));
    expect(ticks >= 20, `only ${ticks} ticks of a 100 ms timer in 3 s while hidden`);
    await terminal(page, 'echo "background $((6*7))"', 'background 42');
    // A second start hands over to the running app, which shows its window.
    const second = spawn(exe, [], { env, stdio: 'ignore' });
    const code = await new Promise((r) => second.once('exit', r));
    expect(await until(() => visible() === 'True'), 'the window did not come back');
    expect(code === 0 && app.exitCode === null, `second start exited ${code}; first ${app.exitCode}`);
    // Minimizing puts it in the tray too.
    ps(`Add-Type -Name S -Namespace U -MemberDefinition '[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);'; [U.S]::ShowWindow([IntPtr]${handle}, 6) | Out-Null`);
    expect(await until(() => visible() === 'False'), 'the window is still showing after minimize');
    const third = spawn(exe, [], { env, stdio: 'ignore' });
    await new Promise((r) => third.once('exit', r));
    expect(await until(() => visible() === 'True'), 'the minimized window did not come back');
  });

  await check('Quit saves everything: projects and history are there after a restart', async () => {
    await quit(browser, page);
    ({ browser, page } = await launch());
    await terminal(page, 'cat keep.txt && git log --format=%s', 'kept\nfirst');
  });

  const portableExe = readdirSync('src-tauri/target/release/bundle/portable').find((f) => f.endsWith('-portable.exe'));
  if (!portableExe) console.log('skip portable: run npm run desktop:portable first');
  else {
    await check('portable: one exe, its data beside it, in the tray when closed, carried to another folder with it, and running next to the installed app', async () => {
      const usb = mkdtempSync(join(tmpdir(), 'bot-computer-usb-'));
      const moved = mkdtempSync(join(tmpdir(), 'bot-computer-moved-'));
      try {
        const copy = join(usb, portableExe);
        copyFileSync(join('src-tauri/target/release/bundle/portable', portableExe), copy);
        // No data folder from outside: the portable app picks its own.
        const own = { ...process.env, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${PORT + 1}` };
        const installed = app;
        let p = await launch(copy, own, PORT + 1);
        const portable = app;
        expect(installed.exitCode === null && portable.exitCode === null, 'the portable app handed over to the installed one');
        await terminal(p.page, 'echo "on the stick" > carried.txt && cat carried.txt', 'on the stick');
        // The tray works the same without an install: closing hides it, it keeps running, a second start shows it.
        const shown = windowOf(portable.pid);
        ps(`(Get-Process -Id ${portable.pid}).CloseMainWindow() | Out-Null`);
        expect((await until(() => !shown())) && portable.exitCode === null, 'the portable app did not go to the tray on close');
        await terminal(p.page, 'echo "tray $((6*7))"', 'tray 42');
        const again = spawn(copy, [], { env: own, stdio: 'ignore' });
        await new Promise((r) => again.once('exit', r));
        expect(await until(shown), 'the portable app did not come back from the tray');
        await quit(p.browser, p.page, portable);
        expect(existsSync(join(usb, 'bot.computer-data', 'webview')), `nothing beside the exe: ${readdirSync(usb)}`);
        // Carry the exe and its folder somewhere else: the projects come along.
        cpSync(usb, moved, { recursive: true });
        p = await launch(join(moved, portableExe), own, PORT + 1);
        await terminal(p.page, 'cat carried.txt', 'on the stick');
        await quit(p.browser, p.page, app);
        app = installed;
      } finally {
        for (const dir of [usb, moved]) await removeSoon(dir);
      }
    });
  }
  await quit(browser, page);
} finally {
  if (app && app.exitCode === null) app.kill();
  for (const dir of [data, downloads]) {
    try {
      rmSync(dir, { recursive: true, force: true });
    } catch {
      /* WebView2 may still hold it */
    }
  }
}

console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
