// The built desktop app, driven through WebView2's remote debugging port
// (Windows): it is cross-origin isolated, runs code in the sandbox, loads the
// SoftN runtime into a sandboxed frame, saves an export as a download, goes to
// the tray when closed and keeps working there, comes back when started again,
// and saves everything when it quits.
//   npm run desktop:build && node tests/e2e/desktop.mjs
import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, rmSync } from 'node:fs';
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
async function launch() {
  app = spawn(exe, [], { env, stdio: 'ignore' });
  for (let i = 0; i < 120; i++) {
    try {
      const browser = await puppeteer.connect({ browserURL: `http://127.0.0.1:${PORT}`, defaultViewport: null });
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
async function quit(browser, page) {
  const exited = new Promise((r) => app.once('exit', r));
  await page.evaluate(() => { void window.__botComputerBeforeQuit(); }).catch(() => {});
  const timer = setTimeout(() => app.kill(), 15_000);
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

  await check('Export .zip saves a download', async () => {
    const cdp = await page.createCDPSession();
    await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: downloads });
    await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent === 'Export .zip').click());
    for (let i = 0; i < 40 && !readdirSync(downloads).some((f) => f.endsWith('.zip')); i++) await new Promise((r) => setTimeout(r, 250));
    const files = readdirSync(downloads);
    expect(files.some((f) => f.endsWith('.zip')), `downloads: ${files}`);
  });

  await check('closing or minimizing the window leaves it running in the tray, still working; starting it again brings it back', async () => {
    const ps = (command) => execFileSync('powershell', ['-NoProfile', '-Command', command]).toString().trim();
    const handle = ps(`(Get-Process -Id ${app.pid}).MainWindowHandle`);
    const visible = () => ps(`Add-Type -Name W -Namespace U -MemberDefinition '[DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);'; [U.W]::IsWindowVisible([IntPtr]${handle})`);
    expect(visible() === 'True', 'the window is not visible to begin with');
    // WM_CLOSE, as the window's close button sends it.
    ps(`(Get-Process -Id ${app.pid}).CloseMainWindow() | Out-Null`);
    await new Promise((r) => setTimeout(r, 1500));
    expect(visible() === 'False', 'the window is still showing after close');
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
    await new Promise((r) => setTimeout(r, 1500));
    expect(visible() === 'True', 'the window did not come back');
    expect(code === 0 && app.exitCode === null, `second start exited ${code}; first ${app.exitCode}`);
    // Minimizing puts it in the tray too.
    ps(`Add-Type -Name S -Namespace U -MemberDefinition '[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);'; [U.S]::ShowWindow([IntPtr]${handle}, 6) | Out-Null`);
    await new Promise((r) => setTimeout(r, 1500));
    expect(visible() === 'False', 'the window is still showing after minimize');
    const third = spawn(exe, [], { env, stdio: 'ignore' });
    await new Promise((r) => third.once('exit', r));
    await new Promise((r) => setTimeout(r, 1500));
    expect(visible() === 'True', 'the minimized window did not come back');
  });

  await check('Quit saves everything: projects and history are there after a restart', async () => {
    await quit(browser, page);
    ({ browser, page } = await launch());
    await terminal(page, 'cat keep.txt && git log --format=%s', 'kept\nfirst');
  });
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
