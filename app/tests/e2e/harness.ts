// Exposes the sandbox to the end-to-end runner (tests/e2e/run.mjs).
import { NetGate } from '../../src/gate/netgate';
import { SandboxHost, summarize } from '../../src/sandbox/host';
import { runInSandbox, sandboxAvailable } from '../../src/sandbox/runner';
import { Vfs } from '../../src/vfs/vfs';

const vfs = new Vfs();
const gate = new NetGate();

async function shell(command: string, cwd = '/', timeoutMs = 20_000) {
  const host = new SandboxHost(vfs, gate, 'sandbox_shell', timeoutMs);
  const outcome = await runInSandbox({ lang: 'shell', source: command, cwd, limits: { maxSteps: 2_000_000_000 } }, host, { timeoutMs });
  return { outcome, report: outcome.result?.shell, changes: [...host.changes.written] };
}

async function run(lang: 'js' | 'python', source: string, timeoutMs = 20_000) {
  const host = new SandboxHost(vfs, gate, 'code_run', timeoutMs);
  const outcome = await host.runProgram({ lang, source });
  return { outcome, summary: summarize(outcome), changes: [...host.changes.written] };
}

Object.assign(window, { __bot: { vfs, gate, shell, run, sandboxAvailable, isolated: globalThis.crossOriginIsolated } });
document.getElementById('status')!.textContent = `ready isolated=${globalThis.crossOriginIsolated}`;
