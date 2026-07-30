/**
 * `oaiy` management commands — drive oaiy-server (Python runtime, venvs, services,
 * models) from the CLI. The headless equivalent of the desktop dashboard.
 *
 *   oaiy python install | python status
 *   oaiy venv create NAME [--req pkg ...] | venv rm NAME
 *   oaiy service list | service start|stop|install|rm ID | service add FILE | service logs ID
 *   oaiy model download URL [--name N --subdir S] | model list | model rm NAME | model catalog
 *   oaiy server config | server health
 *
 * Async ops (python install, venv create, service install, model download) poll
 * to completion with progress on stderr; the final summary goes to stdout.
 */
import type { Command } from 'commander';
import fs from 'node:fs';
import { server, ServerError, BASE_URL } from './server-client';

/* eslint-disable @typescript-eslint/no-explicit-any */
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const out = (s: string) => process.stdout.write(s + '\n');
const note = (s: string) => process.stderr.write(s + '\n');

function fmtBytes(n?: number): string {
  if (!n || n <= 0) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
}

const lastLogLine = (logs: any): string => {
  // The /logs endpoints return a bare JSON array of LogLine; tolerate a wrapped
  // { logs: [...] } shape too in case the wire format ever changes.
  const arr = Array.isArray(logs) ? logs : logs?.logs;
  if (!Array.isArray(arr) || !arr.length) return '';
  const l = arr[arr.length - 1];
  return String(l?.text ?? l ?? '').slice(0, 100);
};

/** Poll a python runtime/venv job (by fresh startedAt) to completion. */
async function runPythonJob(
  label: string,
  trigger: () => Promise<any>,
  opts: { untilInstalled?: boolean } = {},
): Promise<void> {
  const prevStart = (await server.python.status())?.currentJob?.startedAt;
  await trigger();
  let last = '';
  for (let i = 0; i < 2000; i++) {
    await sleep(2000);
    const st = await server.python.status();
    if (opts.untilInstalled && st.installed) return;
    const job = st.currentJob;
    if (job && job.startedAt !== prevStart) {
      const line = lastLogLine(await server.python.logs(2).catch(() => null));
      if (line && line !== last) {
        note('  … ' + line);
        last = line;
      }
      if (job.finishedAt) {
        if (job.error || (job.exitCode != null && job.exitCode !== 0)) {
          throw new ServerError(
            `${label} failed${job.error ? ': ' + job.error : ` (exit ${job.exitCode})`}`,
          );
        }
        return;
      }
    }
  }
  throw new ServerError(`${label} timed out`);
}

async function waitServiceInstall(id: string): Promise<void> {
  let last = '';
  for (let i = 0; i < 2000; i++) {
    await sleep(2000);
    const svc = ((await server.services.list()).services || []).find((s: any) => s.id === id);
    if (!svc) throw new ServerError(`service '${id}' not found`);
    const line = lastLogLine(await server.services.logs(id, 2).catch(() => null));
    if (line && line !== last) {
      note('  … ' + line);
      last = line;
    }
    if (svc.status !== 'installing') {
      if (svc.status === 'errored') {
        throw new ServerError(`install of '${id}' errored${svc.error ? ': ' + svc.error : ''}`);
      }
      return;
    }
  }
  throw new ServerError(`install of '${id}' timed out`);
}

async function waitDownload(id: string): Promise<void> {
  let lastPct = -1;
  for (let i = 0; i < 14400; i++) {
    await sleep(2000);
    const raw = await server.models.downloads();
    const list = Array.isArray(raw) ? raw : raw?.downloads || [];
    const dl = list.find((d: any) => d.id === id);
    if (!dl) continue;
    if (dl.bytesTotal) {
      const pct = Math.floor((dl.bytesDownloaded / dl.bytesTotal) * 100);
      if (pct !== lastPct) {
        note(
          `  … ${pct}% (${fmtBytes(dl.bytesDownloaded)}/${fmtBytes(dl.bytesTotal)}${dl.speedBps ? ' · ' + fmtBytes(dl.speedBps) + '/s' : ''})`,
        );
        lastPct = pct;
      }
    }
    if (dl.status === 'completed') return;
    if (dl.status === 'failed') throw new ServerError(`download failed${dl.error ? ': ' + dl.error : ''}`);
    if (dl.status === 'cancelled') throw new ServerError('download cancelled');
  }
  throw new ServerError('download timed out');
}

/** Wrap an action so ServerError prints cleanly and sets exit 1. */
const action = (fn: (...a: any[]) => Promise<void>) =>
  async (...args: any[]) => {
    try {
      await fn(...args);
      process.exit(0);
    } catch (e) {
      note(`oaiy: ${e instanceof Error ? e.message : String(e)}`);
      process.exit(1);
    }
  };

export function registerManageCommands(program: Command): void {
  // --- python ---------------------------------------------------------------
  const python = program.command('python').description('Manage the portable Python runtime (via oaiy-server)');
  python
    .command('install')
    .description('Download + install the portable Python runtime')
    .action(
      action(async () => {
        note(`oaiy: installing Python runtime on ${BASE_URL} …`);
        await runPythonJob('python install', () => server.python.install(), { untilInstalled: true });
        const st = await server.python.status();
        out(`python installed: ${st.interpreterPath}`);
      }),
    );
  python
    .command('status')
    .description('Show Python runtime + venv status')
    .action(
      action(async () => {
        const st = await server.python.status();
        out(JSON.stringify(
          { installed: st.installed, interpreterPath: st.interpreterPath, venvsDir: st.venvsDir, venvs: (st.venvs || []).map((v: any) => ({ name: v.name, sizeBytes: v.sizeBytes, boundServices: v.boundServices })) },
          null,
          2,
        ));
      }),
    );

  // --- venv -----------------------------------------------------------------
  const venv = program.command('venv').description('Manage reusable Python venvs (via oaiy-server)');
  venv
    .command('create')
    .description('Create (or reuse) a venv and pip-install requirements')
    .argument('<name>', 'venv name')
    .option('--req <pkg...>', 'a pip requirement (repeatable), e.g. --req torch --req diffusers')
    .action(
      action(async (name: string, opts: any) => {
        const reqs: string[] = opts.req ?? [];
        note(`oaiy: creating venv '${name}'${reqs.length ? ' with ' + reqs.join(', ') : ''} …`);
        await runPythonJob(`venv '${name}'`, () => server.python.createVenv(name, reqs));
        out(`venv ready: ${name}`);
      }),
    );
  venv
    .command('rm')
    .description('Delete a venv')
    .argument('<name>', 'venv name')
    .action(
      action(async (name: string) => {
        await server.python.deleteVenv(name);
        out(`removed venv: ${name}`);
      }),
    );

  // --- service --------------------------------------------------------------
  const service = program.command('service').description('Manage AI/local services (via oaiy-server)');
  service
    .command('list')
    .description('List services and their status')
    .action(
      action(async () => {
        const snap = await server.services.list();
        for (const s of snap.services || []) {
          out(`${(s.status as string).padEnd(10)} ${String(s.id).padEnd(20)} :${s.port ?? s.defaultPort ?? '-'}  ${s.name}`);
        }
      }),
    );
  service
    .command('install')
    .description('Run a service install script (streams progress)')
    .argument('<id>', 'service id')
    .action(
      action(async (id: string) => {
        note(`oaiy: installing service '${id}' …`);
        await server.services.install(id);
        await waitServiceInstall(id);
        out(`installed: ${id}`);
      }),
    );
  service
    .command('start')
    .description('Start a service')
    .argument('<id>', 'service id')
    .action(
      action(async (id: string) => {
        await server.services.start(id);
        out(`started: ${id}`);
      }),
    );
  service
    .command('stop')
    .description('Stop a service')
    .argument('<id>', 'service id')
    .action(
      action(async (id: string) => {
        await server.services.stop(id);
        out(`stopped: ${id}`);
      }),
    );
  service
    .command('add')
    .description('Add/replace a service from a template JSON file — the same self-contained format the desktop app uses (install/run/health + any bundled `files`); privileged')
    .argument('<file>', 'path to a ServiceTemplate .json')
    .action(
      action(async (file: string) => {
        let template: any;
        try {
          template = JSON.parse(fs.readFileSync(file, 'utf-8'));
        } catch (e) {
          throw new ServerError(`cannot read template '${file}': ${e instanceof Error ? e.message : String(e)}`);
        }
        if (!template || typeof template !== 'object' || !template.id || !template.run) {
          throw new ServerError(`'${file}' is not a valid service template (needs at least an "id" and a "run" block)`);
        }
        await server.services.add(template);
        const n = Object.keys(template.files || {}).length;
        out(`added service: ${template.id}${n ? ` (${n} bundled file${n === 1 ? '' : 's'})` : ''}`);
      }),
    );
  service
    .command('export')
    .description('Export a service to a self-contained template JSON (same format as `add`; bundles its scripts inline so it installs anywhere)')
    .argument('<id>', 'service id')
    .argument('[file]', 'output path (default: ./<id>.json)')
    .action(
      action(async (id: string, file?: string) => {
        const template: any = await server.services.export(id);
        const outPath = file || `${id}.json`;
        fs.writeFileSync(outPath, JSON.stringify(template, null, 2) + '\n');
        const n = Object.keys(template?.files || {}).length;
        out(`exported '${id}' -> ${outPath}${n ? ` (${n} bundled file${n === 1 ? '' : 's'})` : ''}`);
      }),
    );
  service
    .command('rm')
    .description('Remove a service template (privileged)')
    .argument('<id>', 'service id')
    .action(
      action(async (id: string) => {
        await server.services.remove(id);
        out(`removed service: ${id}`);
      }),
    );
  service
    .command('uninstall')
    .description("Remove a service's installed files — venv/binaries (must be stopped; privileged)")
    .argument('<id>', 'service id')
    .action(
      action(async (id: string) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const res: any = await server.services.uninstall(id);
        const n = res?.removed;
        out(`uninstalled '${id}'${typeof n === 'number' ? ` (${n} item${n === 1 ? '' : 's'} removed)` : ''}`);
      }),
    );
  service
    .command('logs')
    .description('Show a service log tail')
    .argument('<id>', 'service id')
    .option('--tail <n>', 'lines', (v) => parseInt(v, 10), 50)
    .action(
      action(async (id: string, opts: any) => {
        const logs = await server.services.logs(id, opts.tail);
        const arr = Array.isArray(logs) ? logs : logs?.logs || [];
        for (const l of arr) out(typeof l === 'string' ? l : String(l.text ?? JSON.stringify(l)));
      }),
    );

  // --- model ----------------------------------------------------------------
  const model = program.command('model').description('Manage downloaded models (via oaiy-server)');
  model
    .command('download')
    .description('Download a model (HuggingFace or direct URL), streaming progress')
    .argument('<url>', 'model URL')
    .option('--name <filename>', 'destination filename')
    .option('--subdir <dir>', 'subdirectory under the models dir')
    .action(
      action(async (url: string, opts: any) => {
        note(`oaiy: downloading ${url} …`);
        const res = await server.models.download(url, opts.name, opts.subdir);
        const id = res?.downloadId ?? res?.id;
        if (!id) throw new ServerError('server did not return a download id');
        await waitDownload(id);
        out(`downloaded: ${opts.name ?? url.split('/').pop()}`);
      }),
    );
  model
    .command('list')
    .description('List downloaded model files')
    .action(
      action(async () => {
        const snap = await server.models.list();
        out(`root: ${snap.rootDir}${snap.freeBytes ? `  (free: ${fmtBytes(snap.freeBytes)})` : ''}`);
        for (const m of snap.models || []) out(`  ${fmtBytes(m.sizeBytes).padStart(9)}  ${m.name}`);
      }),
    );
  model
    .command('rm')
    .description('Delete a model file (privileged)')
    .argument('<name>', 'model file name')
    .action(
      action(async (name: string) => {
        await server.models.remove(name);
        out(`removed model: ${name}`);
      }),
    );
  model
    .command('catalog')
    .description('Show the curated quick-add model catalog')
    .action(
      action(async () => {
        const snap = await server.models.catalog();
        for (const cat of snap?.catalog?.categories || []) {
          out(`# ${cat.name}`);
          for (const m of cat.models || []) out(`  ${m.id}  (${fmtBytes(m.sizeBytes)})  ${m.url}`);
        }
      }),
    );

  // --- server ---------------------------------------------------------------
  const srv = program.command('server').description('Query the oaiy-server itself');
  srv.command('health').action(action(async () => { out(JSON.stringify(await server.health())); }));
  srv.command('config').action(action(async () => { out(JSON.stringify(await server.config(), null, 2)); }));
}
