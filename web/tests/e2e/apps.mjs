/**
 * The two apps as they are deployed: the REAL builds of the Agent (app/) and the flow editor (platform/ui), not the fixture shells the
 * other cases use as stand-ins, so a claim about what a fresh load sends is a claim about the code a visitor gets.
 *
 * `vite build --outDir <folder>` in each package, into a temporary folder that is removed afterwards. It is plain `vite build`, not
 * the package's `npm run build`: that one fetches the Zipp engine from GitHub first, and nothing here needs the engine (no test runs
 * the sandbox). Nothing is written inside the repository. The build of the flow editor takes about half a minute, so
 * `WEB_E2E_APPS=<folder holding agent/ and flows/>` reuses builds that are already there.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

function build(cwd, outDir) {
  const vite = path.join(cwd, 'node_modules', 'vite', 'bin', 'vite.js');
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [vite, 'build', '--outDir', outDir, '--emptyOutDir'], { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    let output = '';
    child.stdout.on('data', (d) => (output += d));
    child.stderr.on('data', (d) => (output += d));
    child.once('error', reject);
    child.once('close', (code) => (code === 0 ? resolve() : reject(new Error(`vite build failed in ${cwd} (exit ${code}):\n${output.slice(-2000)}`))));
  });
}

/** Build the Agent and the flow editor. Returns their folders and `remove()`. */
export async function buildApps() {
  const given = process.env.WEB_E2E_APPS;
  if (given) {
    const dirs = { agent: path.join(given, 'agent'), flows: path.join(given, 'flows') };
    for (const dir of Object.values(dirs)) if (!fs.existsSync(path.join(dir, 'index.html'))) throw new Error(`WEB_E2E_APPS: ${dir} holds no build`);
    return { ...dirs, remove() {} };
  }
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-web-e2e-apps-'));
  const dirs = { agent: path.join(base, 'agent'), flows: path.join(base, 'flows') };
  await Promise.all([build(path.join(ROOT, 'app'), dirs.agent), build(path.join(ROOT, 'platform', 'ui'), dirs.flows)]);
  return { ...dirs, remove: () => fs.rmSync(base, { recursive: true, force: true }) };
}
