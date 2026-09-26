import './styles.css';
import { Agent } from './agent/agent';
import type { ProviderConfig } from './agent/providers/types';
import { HELP, internetCommand, internetStatus } from './commands';
import { NetGate } from './gate/netgate';
import { sandboxAvailable, zippModule } from './sandbox/runner';
import { loadSettings, saveGate, saveLastProject, saveProviders } from './settings';
import { ChatPane } from './ui/chat';
import { clear, h } from './ui/dom';
import { EditorPane } from './ui/editor';
import { openSettings } from './ui/settings';
import { TerminalPane } from './ui/terminal';
import { FileTree } from './ui/tree';
import { OpenProject, createProject, deleteProject, listProjects, renameProject, type ProjectMeta } from './vfs/projects';
import { canPickFolder, downloadZip, importFileList, importFolder, importZip, type Imported } from './vfs/transfer';
import type { Vfs } from './vfs/vfs';

const WELCOME: Array<[string, string]> = [
  [
    'README.md',
    `# Welcome to bot.computer

A coding agent that lives entirely in this browser tab.

- **Files** are stored in this browser (OPFS). Open a folder from disk, import a .zip, or start here.
- **Code runs on the Zipp VM** in a Web Worker: JavaScript and Python, with no access to anything
  but this project and what the network gate allows.
- **The terminal** below is an emulated shell on the same sandbox — try \`ls\`, \`cat data/weather.csv | sort -t, -k2 -n\`,
  \`python hello.py\`, or \`help\`.
- **The agent** (right) uses any model: a local server such as Ollama or LM Studio, or an API. Set one up in ⚙ Settings.
- **/internet** controls what the sandbox may reach: \`/internet off\`, \`/internet allow docs.python.org\`, \`/net status\`.
`,
  ],
  ['data/weather.csv', 'city,temp_c\nParis,18\nRome,24\nOslo,9\nCairo,31\nLima,17\n'],
  ['hello.py', 'import statistics\n\nwith open("data/weather.csv") as f:\n    rows = [line.split(",") for line in f.read().splitlines()[1:]]\n\ntemps = [int(t) for _, t in rows]\nprint("cities:", len(rows))\nprint("mean temperature:", statistics.mean(temps))\n'],
  ['hello.js', "const fs = require('fs');\nconst rows = fs.readFileSync('data/weather.csv', 'utf8').trim().split('\\n').slice(1);\nconst warmest = rows.map((r) => r.split(',')).sort((a, b) => b[1] - a[1])[0];\nconsole.log('warmest:', warmest[0], warmest[1] + '°C');\n"],
];

function registerServiceWorker(): void {
  if (!('serviceWorker' in navigator) || import.meta.env.DEV) return;
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).then(() => {
    navigator.serviceWorker.ready.then((registration) => {
      const urls = performance.getEntriesByType('resource').map((e) => e.name);
      registration.active?.postMessage({ type: 'cache', urls: [location.href, ...urls] });
    });
    // The first visit is served without isolation headers; once the service
    // worker controls the page, one reload gives it SharedArrayBuffer.
    if (!crossOriginIsolated && !sessionStorage.getItem('bot.computer.reloaded')) {
      navigator.serviceWorker.ready.then(() => {
        sessionStorage.setItem('bot.computer.reloaded', '1');
        location.reload();
      });
    }
  });
}

function summarizeProject(meta: ProjectMeta, vfs: Vfs, gate: NetGate): string {
  const walked = vfs.walk('/', { limit: 400 });
  const shallow = walked.entries.filter((e) => e.path.split('/').length <= 3).slice(0, 150);
  const files = walked.entries.filter((e) => e.type === 'file').length;
  const tree = shallow.map((e) => (e.type === 'dir' ? `${e.path}/` : e.path)).join('\n');
  return `Project "${meta.name}": ${files}${walked.truncated ? '+' : ''} files. Network gate: ${gate.mode}.\n${tree}`;
}

async function main(): Promise<void> {
  registerServiceWorker();
  const app = document.getElementById('app')!;
  const settings = await loadSettings();
  const gate = new NetGate(settings.gate);
  let saveGateTimer: ReturnType<typeof setTimeout> | null = null;
  let providers: ProviderConfig[] = settings.providers;
  let activeId = settings.activeProviderId;
  const activeProvider = () => providers.find((p) => p.id === activeId) ?? null;

  let project!: OpenProject;
  let agent!: Agent;
  let controller: AbortController | null = null;
  let unsubscribe: (() => void) | null = null;

  const chat = new ChatPane({ submit: (text) => void submit(text), stop: () => controller?.abort() });
  const notice = (message: string) => chat.system(message, 'error');
  const tree = new FileTree(null as unknown as Vfs, (path) => {
    editor.open(path);
    tree.select(path);
  }, notice);
  const editor = new EditorPane(null as unknown as Vfs);
  const terminal = new TerminalPane(null as unknown as Vfs, gate);

  const projectSelect = h('select.project-select', { title: 'Project' });
  const gateChip = h('button.chip', { title: 'Network gate (/internet)', onclick: () => chat.system(internetStatus(gate)) });
  const providerChip = h('button.chip', { title: 'AI provider', onclick: () => void editSettings() });
  const renderChips = () => {
    gateChip.textContent = `internet: ${gate.mode === 'open' ? 'on' : gate.mode === 'blocked' ? 'off' : 'allowlist'}`;
    gateChip.dataset.mode = gate.mode;
    const p = activeProvider();
    providerChip.textContent = p ? `${p.name}${p.modelId ? ` · ${p.modelId}` : ' · no model'}` : 'Set up AI…';
  };
  gate.onChange(() => {
    renderChips();
    if (saveGateTimer) clearTimeout(saveGateTimer);
    saveGateTimer = setTimeout(() => void saveGate(gate.getSettings()), 200);
  });

  const renderProjects = async () => {
    const list = await listProjects();
    clear(projectSelect);
    for (const meta of list) projectSelect.append(h('option', { value: meta.id, selected: meta.id === project?.meta.id }, meta.name));
  };

  const openProject = async (meta: ProjectMeta) => {
    if (controller) controller.abort();
    if (project) {
      editor.flush();
      await project.close();
      await project.saveChat(agent.turns);
      unsubscribe?.();
    }
    project = await OpenProject.open(meta);
    project.onError = notice;
    agent = new Agent({ vfs: project.vfs, gate, provider: activeProvider, projectSummary: () => summarizeProject(project.meta, project.vfs, gate) });
    agent.turns = await project.loadChat();
    tree.setVfs(project.vfs);
    editor.setVfs(project.vfs);
    terminal.setVfs(project.vfs);
    unsubscribe = project.vfs.onChange((change) => {
      tree.refresh();
      editor.externalChange('path' in change ? change.path : null);
    });
    chat.replay(agent.turns);
    await saveLastProject(meta.id);
    await renderProjects();
    document.title = `${meta.name} — bot.computer`;
  };

  const newProjectFrom = async (imported: Imported | null) => {
    if (!imported) return;
    const meta = await createProject(imported.name);
    await openProject(meta);
    project.vfs.load(imported.files);
    await project.flush();
    chat.system(`Opened "${imported.name}": ${imported.files.length} files.${imported.skipped.length ? ` Skipped ${imported.skipped.length} (dependency/build folders or over the size limit): ${imported.skipped.slice(0, 8).join(', ')}${imported.skipped.length > 8 ? ', …' : ''}` : ''}`);
  };

  const fileInput = (accept: string, directory: boolean, onFiles: (files: FileList) => void) => {
    const input = h('input', { type: 'file', accept, style: 'display:none' });
    if (directory) input.setAttribute('webkitdirectory', '');
    input.addEventListener('change', () => input.files && onFiles(input.files));
    document.body.append(input);
    input.click();
    setTimeout(() => input.remove(), 60_000);
  };

  const editSettings = async () => {
    const result = await openSettings({ providers, activeId });
    if (!result) return;
    providers = result.providers;
    activeId = result.activeId;
    await saveProviders(providers, activeId);
    renderChips();
  };

  async function submit(text: string): Promise<void> {
    if (text.startsWith('/')) {
      const [command, ...args] = text.slice(1).trim().split(/\s+/);
      switch (command.toLowerCase()) {
        case 'internet': case 'net':
          chat.system(internetCommand(gate, args));
          return;
        case 'clear': case 'new': case 'reset':
          agent.reset();
          chat.clearLog();
          await project.saveChat([]);
          chat.system('New conversation. The project\'s files are unchanged.');
          return;
        case 'help': case '?':
          chat.system(HELP);
          return;
        default:
          chat.system(`Unknown command /${command}. ${HELP}`, 'error');
          return;
      }
    }
    editor.flush();
    chat.user(text);
    chat.setBusy(true);
    controller = new AbortController();
    try {
      await agent.run(text, (event) => chat.event(event), controller.signal);
    } finally {
      controller = null;
      chat.setBusy(false);
      await project.flush();
      await project.saveChat(agent.turns);
    }
  }

  const header = h(
    'header.topbar',
    h('div.brand', h('span.logo', '◆'), ' bot.computer'),
    projectSelect,
    h('button', { title: 'New empty project', onclick: async () => {
      const name = prompt('Project name', 'untitled');
      if (name) await openProject(await createProject(name));
    } }, 'New'),
    h('button', { title: 'Open a folder from this computer (a copy is kept in the browser)', onclick: async () => {
      if (canPickFolder()) await newProjectFrom(await importFolder());
      else fileInput('', true, async (files) => newProjectFrom(await importFileList(files)));
    } }, 'Open folder…'),
    h('button', { title: 'Import a .zip as a new project', onclick: () => fileInput('.zip,application/zip', false, async (files) => newProjectFrom(await importZip(files[0]))) }, 'Import .zip'),
    h('button', { title: 'Download this project as a .zip', onclick: async () => {
      editor.flush();
      downloadZip(project.meta.name.replace(/[^\w.-]+/g, '-'), project.vfs.files());
    } }, 'Export .zip'),
    h('button', { title: 'Rename this project', onclick: async () => {
      const name = prompt('Rename project', project.meta.name);
      if (name) {
        project.meta = await renameProject(project.meta, name);
        await renderProjects();
      }
    } }, 'Rename'),
    h('button.danger', { title: 'Delete this project from the browser', onclick: async () => {
      if (!confirm(`Delete "${project.meta.name}" and all its files from this browser?`)) return;
      const doomed = project.meta.id;
      const others = (await listProjects()).filter((m) => m.id !== doomed);
      const next = others[0] ?? (await createProject('untitled'));
      await openProject(next);
      await deleteProject(doomed);
      await renderProjects();
    } }, 'Delete'),
    h('div.spacer'),
    gateChip,
    providerChip,
    h('button', { title: 'AI providers', onclick: () => void editSettings() }, '⚙ Settings'),
  );
  projectSelect.addEventListener('change', async () => {
    const meta = (await listProjects()).find((m) => m.id === projectSelect.value);
    if (meta) await openProject(meta);
  });

  clear(app);
  app.append(
    header,
    h('main.workspace', tree.element, h('div.center', editor.element, terminal.element), chat.element),
  );
  renderChips();

  // Open the last project, or make the welcome one.
  const all = await listProjects();
  let meta = all.find((m) => m.id === settings.lastProjectId) ?? all[0];
  if (!meta) {
    meta = await createProject('Welcome');
    await openProject(meta);
    for (const [path, text] of WELCOME) project.vfs.writeFile(`/${path}`, text, { parents: true });
    editor.open('/README.md');
    tree.select('/README.md');
  } else {
    await openProject(meta);
  }

  const sandbox = sandboxAvailable();
  if (!sandbox.ok) chat.system(`The code sandbox is unavailable: ${sandbox.reason}.`, 'error');
  else void zippModule().catch((error: unknown) => chat.system(`Could not load the Zipp engine: ${(error as Error).message}`, 'error'));
  if (!activeProvider()) chat.system('Welcome! Set up an AI provider in ⚙ Settings to talk to the agent — a local server (Ollama, LM Studio) keeps everything on this computer. The editor and terminal work without one.');
  window.addEventListener('beforeunload', () => {
    editor.flush();
    void project.flush();
  });
  chat.focus();
}

main().catch((error: unknown) => {
  document.body.textContent = `bot.computer failed to start: ${(error as Error).message}`;
});
