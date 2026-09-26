import './styles.css';
import { Agent } from './agent/agent';
import type { ProviderConfig } from './agent/providers/types';
import { HELP, internetCommand } from './commands';
import { NetGate } from './gate/netgate';
import { sandboxAvailable, zippModule } from './sandbox/runner';
import { loadSettings, saveGate, saveLastProject, saveProviders } from './settings';
import { ChatPane } from './ui/chat';
import { clear, h } from './ui/dom';
import { EditorPane } from './ui/editor';
import { openSettings } from './ui/settings';
import { TerminalPane } from './ui/terminal';
import { SoftnPreview } from './softn/preview';
import { exampleBundle, exampleCatalogue, warmKnowledge } from './softn/knowledge';
import { askText, confirmAction } from './ui/modal';
import { newAppDialog, type NewAppChoice } from './ui/newApp';
import { SOFTN_BLANK, SOFTN_STARTER, readManifest, appKey, appLabel, checkProject, describeApp, downloadSoftn, findApps, formatFindings, importSoftn, isSoftnProject, logicSyntax, resolveApp } from './softn/softn';
import { imageForMessage, imageMimeFor, type ImagePart } from './agent/images';
import type { Attachment } from './agent/protocol';
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

/**
 * Register the service worker. Resolves true when the page is about to reload
 * itself: a first visit on a host that does not send the isolation headers
 * gets them from the service worker, once it controls the page. The app does
 * not start on that visit, so nothing half-done is left behind.
 */
async function registerServiceWorker(): Promise<boolean> {
  if (!('serviceWorker' in navigator) || import.meta.env.DEV) return false;
  try {
    await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`);
  } catch {
    return false;
  }
  const registration = await navigator.serviceWorker.ready;
  const urls = performance.getEntriesByType('resource').map((e) => e.name);
  registration.active?.postMessage({ type: 'cache', urls: [location.href, ...urls] });
  if (crossOriginIsolated) return false;
  // At most one automatic reload a minute: if isolation still does not come
  // (a browser that refuses it), the app starts anyway and says why the
  // sandbox is unavailable. localStorage, not sessionStorage: switching into
  // a cross-origin isolated context can start a fresh session store.
  const KEY = 'bot.computer.isolation-reload';
  let last = 0;
  try {
    last = Number(localStorage.getItem(KEY)) || 0;
    if (Date.now() - last < 60_000) return false;
    localStorage.setItem(KEY, String(Date.now()));
  } catch {
    return false;
  }
  location.reload();
  return true;
}

function summarizeProject(meta: ProjectMeta, vfs: Vfs, gate: NetGate): string {
  const walked = vfs.walk('/', { limit: 400 });
  const shallow = walked.entries.filter((e) => e.path.split('/').length <= 3).slice(0, 150);
  const files = walked.entries.filter((e) => e.type === 'file').length;
  const tree = shallow.map((e) => (e.type === 'dir' ? `${e.path}/` : e.path)).join('\n');
  return `Project "${meta.name}": ${files}${walked.truncated ? '+' : ''} files. Network gate: ${gate.mode}.\n${tree}`;
}

async function main(): Promise<void> {
  const app = document.getElementById('app')!;
  if (!crossOriginIsolated && (await registerServiceWorker())) {
    app.textContent = 'Setting up bot.computer for offline use…';
    return;
  }
  if (crossOriginIsolated) void registerServiceWorker();
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

  const chat = new ChatPane({
    submit: (text, files) => void submit(text, files),
    stop: () => controller?.abort(),
    file: (path) => {
      try {
        return project.vfs.readBytes(`/${path}`);
      } catch {
        return null;
      }
    },
    open: (path, app) => openFromChat(path, app),
  });
  const notice = (message: string) => chat.system(message, 'error');
  const tree = new FileTree(null as unknown as Vfs, (path) => {
    editor.open(path);
    tree.select(path);
    if (window.matchMedia('(max-width: 900px)').matches) showView('editor');
  }, notice);
  const editor = new EditorPane(null as unknown as Vfs);
  const terminal = new TerminalPane(null as unknown as Vfs, gate);
  const preview = new SoftnPreview(null as unknown as Vfs, () => project?.meta.id ?? 'none');
  // An error while the person uses the app: one click asks the agent to fix it.
  preview.onFix = (root, problems) => {
    if (controller) {
      chat.system('The agent is busy; ask again when it has finished.', 'error');
      return false;
    }
    const list = problems.slice(-8).map((p) => `- ${p.message}`).join('\n');
    void submit(`The SoftN app in ${appLabel(root)} reported ${problems.length === 1 ? 'this error' : 'these errors'} while I was using it (from the running app):\n${list}\n\nFind the cause in its files and fix it, then check the app again (softn_interact can repeat what I was doing).`);
    showView('agent');
    return true;
  };

  const projectSelect = h('select.project-select', { title: 'Project' });
  // The chip is the gate's switch: on (open) ⇄ off (blocked). Allowlists and
  // per-host rules stay with /internet.
  const gateChip = h('button.chip.toggle', {
    title: 'Network gate: click to turn the sandbox\'s internet access on or off (/internet for allow/deny lists and status)',
    role: 'switch',
    onclick: () => {
      gate.setMode(gate.mode === 'blocked' ? 'open' : 'blocked');
      chat.system(gate.mode === 'blocked'
        ? 'Internet off: sandboxed code and web_fetch cannot reach any host. Your AI provider still works.'
        : 'Internet on: sandboxed code and web_fetch may reach any host not on the deny list.');
    },
  });
  const providerChip = h('button.chip', { title: 'AI provider', onclick: () => void editSettings() });
  const renderChips = () => {
    gateChip.textContent = `internet: ${gate.mode === 'open' ? 'on' : gate.mode === 'blocked' ? 'off' : 'allowlist'}`;
    gateChip.dataset.mode = gate.mode;
    gateChip.setAttribute('aria-checked', String(gate.mode !== 'blocked'));
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
    agent = new Agent({ vfs: project.vfs, gate, provider: activeProvider, projectSummary: () => summarizeProject(project.meta, project.vfs, gate), softn: preview });
    agent.turns = await project.loadChat();
    tree.setVfs(project.vfs);
    editor.setVfs(project.vfs);
    terminal.setVfs(project.vfs);
    preview.setVfs(project.vfs);
    unsubscribe = project.vfs.onChange((change) => {
      tree.refresh();
      editor.externalChange('path' in change ? change.path : null);
      preview.changed('path' in change ? change.path : null);
    });
    // A SoftN app opens on its preview, so the person sees it being built.
    if (isSoftnProject(project.vfs)) showPane('preview');
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

  /** A free path for an upload: uploads/name, uploads/name-2, … */
  function freePath(dir: string, name: string): string {
    const clean = name.replace(/[\\/:*?"<>|\u0000-\u001f]+/g, '-').replace(/^\.+/, '') || 'file';
    const dot = clean.lastIndexOf('.');
    const stem = dot > 0 ? clean.slice(0, dot) : clean;
    const ext = dot > 0 ? clean.slice(dot) : '';
    let path = `/${dir}/${clean}`;
    for (let n = 2; project.vfs.exists(path); n++) path = `/${dir}/${stem}-${n}${ext}`;
    return path;
  }

  /**
   * Files attached to a message go into the project (uploads/, or an app
   * folder for a .softn), so the agent can open them with its tools; images
   * also go to the model with the message, sized for its eyes.
   */
  async function receiveFiles(files: File[]): Promise<{ notes: string[]; images: ImagePart[]; attachments: Attachment[] }> {
    const notes: string[] = [];
    const images: ImagePart[] = [];
    const attachments: Attachment[] = [];
    for (const file of files) {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const path = freePath('uploads', file.name);
      project.vfs.writeFile(path, bytes, { parents: true });
      const attachment: Attachment = { name: file.name, path: path.slice(1) };
      attachments.push(attachment);
      // A .softn is unpacked to work on; the original stays in uploads/ to
      // compare with or unpack again (softn_import).
      if (/\.softn$/i.test(file.name)) {
        try {
          const imported = importSoftn(project.vfs, bytes, file.name);
          attachment.app = imported.root;
          notes.push(`${path.slice(1)}: unpacked into ${imported.root}/. ${describeApp(project.vfs, imported.root)}\nTo change this app, edit it in ${imported.root}/; to recreate it (or build something like it), read it and write the new app in another folder, leaving this one as it is.`);
          preview.setApp(imported.root);
        } catch (error) {
          notes.push(`${path.slice(1)} (could not be unpacked as a SoftN app: ${(error as Error).message})`);
        }
        continue;
      }
      const mime = imageMimeFor(file.name) ?? (file.type.startsWith('image/') ? file.type : null);
      if (mime) {
        try {
          const { part, width, height } = await imageForMessage(bytes, mime, path.slice(1));
          images.push(part);
          notes.push(`${path.slice(1)} (image, ${width}×${height} px; attached so you can see it — view_image zooms into details)`);
          continue;
        } catch {
          /* not decodable: treat as a plain file */
        }
      }
      notes.push(`${path.slice(1)} (${bytes.byteLength.toLocaleString()} bytes)`);
    }
    return { notes, images, attachments };
  }

  /** A file (or an unpacked .softn's app) clicked in the chat. */
  function openFromChat(path: string, app?: string): void {
    if (app !== undefined && project.vfs.exists(`/${app}`)) {
      preview.setApp(app);
      showPane('preview');
      if (window.matchMedia('(max-width: 900px)').matches) showView('preview');
      return;
    }
    if (!project.vfs.exists(`/${path}`)) {
      chat.system(`${path} is no longer in the project.`, 'error');
      return;
    }
    showPane('editor');
    editor.open(`/${path}`);
    tree.select(`/${path}`);
    if (window.matchMedia('(max-width: 900px)').matches) showView('editor');
  }

  async function submit(text: string, files: File[] = []): Promise<void> {
    if (text.startsWith('/') && !files.length) {
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
        case 'softn':
          await softnCommand(args);
          return;
        default:
          chat.system(`Unknown command /${command}. ${HELP}`, 'error');
          return;
      }
    }
    editor.flush();
    chat.setBusy(true);
    controller = new AbortController();
    try {
      let prompt = text;
      let images: ImagePart[] = [];
      let attachments: Attachment[] = [];
      if (files.length) {
        const received = await receiveFiles(files);
        images = received.images;
        attachments = received.attachments;
        await project.flush();
        prompt = `${text || 'I attached some files.'}\n\n[Attached and saved in the project: ${received.notes.join('; ')}]`;
      }
      chat.user(text, attachments);
      // The first check of a run brings the preview forward, so the person sees the app being built.
      let previewShown = false;
      await agent.run(prompt, (event) => {
        chat.event(event);
        if (event.type === 'check' && event.state === 'running' && !previewShown) {
          previewShown = true;
          if (!window.matchMedia('(max-width: 900px)').matches) showPane('preview');
        }
      }, controller.signal, images, attachments);
    } finally {
      controller = null;
      chat.setBusy(false);
      await project.flush();
      await project.saveChat(agent.turns);
    }
  }

  const EXAMPLE_ICONS: Record<string, string> = { notes: '📝', twenty48: '🔢', showcase: '📊', 'three-demo': '🧊', 'gpu-demo': '⚡', 'device-kit': '📷' };

  /**
   * A new SoftN app, from the task-list starter, a blank page or one of the
   * example apps, in a project of its own or a folder of this one. With a
   * folder (/softn new <folder>) it skips the dialog and uses the starter.
   */
  async function newSoftnApp(folder?: string): Promise<void> {
    let choice: NewAppChoice | null;
    const folderProblem = (dir: string) => {
      const key = appKey(dir);
      if (project.vfs.exists(`/${key}/manifest.json`)) return `${dir}/ already holds an app (it has a manifest.json).`;
      if (project.vfs.stat(`/${key}`)?.type === 'file') return `${dir} is a file.`;
      return null;
    };
    if (folder) {
      choice = { name: folder.split('/').filter(Boolean).pop() ?? 'app', start: 'starter', where: 'folder', folder };
    } else {
      const catalogue = await exampleCatalogue().catch(() => []);
      choice = await newAppDialog({
        projectName: project.meta.name,
        folderProblem,
        templates: [
          { id: 'starter', name: 'Task list', description: 'A small working app: add, tick off and delete tasks.', icon: '☑' },
          { id: 'blank', name: 'Blank', description: 'One page and an empty logic file.', icon: '◻' },
        ],
        examples: catalogue.map((e) => ({ id: e.slug, name: e.name, description: e.description, icon: EXAMPLE_ICONS[e.slug] ?? '◆' })),
      });
    }
    if (!choice) return;
    let root = '';
    if (choice.where === 'folder') {
      const problem = folderProblem(choice.folder);
      if (problem) {
        chat.system(problem, 'error');
        return;
      }
      root = appKey(choice.folder);
    } else {
      await openProject(await createProject(choice.name));
    }
    const files: Array<[string, string | Uint8Array]> = choice.start === 'starter' ? SOFTN_STARTER : choice.start === 'blank' ? SOFTN_BLANK : await exampleBundle(choice.start);
    const prefix = root ? `${root}/` : '';
    for (const [path, data] of files) {
      let content = data;
      if (path === 'manifest.json') {
        try {
          const manifest = JSON.parse(typeof data === 'string' ? data : new TextDecoder().decode(data)) as Record<string, unknown>;
          manifest.name = choice.name;
          content = `${JSON.stringify(manifest, null, 2)}\n`;
        } catch {
          /* keep it as it is */
        }
      }
      project.vfs.writeFile(`/${prefix}${path}`, content, { parents: true });
    }
    await project.flush();
    const main = readManifest(project.vfs, root)?.main;
    tree.select(`/${prefix}${typeof main === 'string' ? main : 'ui/main.ui'}`);
    preview.setApp(root);
    showPane('preview');
    const from = choice.start === 'starter' ? 'a small task list to start from' : choice.start === 'blank' ? 'a blank page to start from' : `a copy of the ${choice.start} example`;
    chat.system(`New SoftN app "${choice.name}" in ${appLabel(root)}: ${from}. Ask the agent to change it into what you want; the preview updates as it works. Export it with /softn export${root ? ` ${root}` : ''}.`);
  }

  function pickApp(folder?: string): string | null {
    const target = resolveApp(project.vfs, folder ?? (findApps(project.vfs).length > 1 ? preview.app : undefined));
    if (!target.ok) {
      chat.system(`${target.reason}. /softn new starts one; /softn import unpacks a .softn file.`, 'error');
      return null;
    }
    return target.root;
  }

  async function exportSoftn(folder?: string): Promise<void> {
    editor.flush();
    const root = pickApp(folder);
    if (root === null) return;
    const errors = [...checkProject(project.vfs, root), ...(await logicSyntax(project.vfs, root).catch(() => []))].filter((f) => f.level === 'error');
    const name = downloadSoftn(project.vfs, root, project.meta.name);
    chat.system(errors.length ? `Exported ${appLabel(root)} as ${name}, but it has problems that will stop it loading:\n${formatFindings(errors)}` : `Exported ${appLabel(root)} as ${name}.`, errors.length ? 'error' : 'info');
  }

  function importSoftnFile(): void {
    fileInput('.softn,.zip,application/zip', false, async (files) => {
      const file = files[0];
      try {
        const imported = importSoftn(project.vfs, new Uint8Array(await file.arrayBuffer()), file.name);
        await project.flush();
        preview.setApp(imported.root);
        showPane('preview');
        tree.select(`/${imported.root}/manifest.json`);
        chat.system(`Imported "${imported.name}" into ${imported.root}/ (${imported.files} files). It is shown in the preview; the agent can read it, change it, or build a new app from it in another folder.`);
      } catch (error) {
        chat.system((error as Error).message, 'error');
      }
    });
  }

  async function softnCommand(args: string[]): Promise<void> {
    const folder = args[1];
    switch ((args[0] ?? '').toLowerCase()) {
      case 'new':
        await newSoftnApp(folder);
        return;
      case 'import':
        importSoftnFile();
        return;
      case 'export':
        await exportSoftn(folder);
        return;
      case 'apps': case 'list': {
        const apps = findApps(project.vfs);
        chat.system(apps.length ? `SoftN apps in this project:\n${apps.map((r) => `  ${appLabel(r)}`).join('\n')}` : 'No SoftN apps in this project yet.');
        return;
      }
      case 'check': {
        const root = pickApp(folder);
        if (root === null) return;
        const result = await preview.check(root);
        const files = formatFindings([...checkProject(project.vfs, root), ...(await logicSyntax(project.vfs, root).catch(() => []))]);
        chat.system(`App: ${appLabel(root)}\nFiles: ${files}\nRender: ${result.ok ? 'ok' : result.errors.join('; ')}`, result.ok ? 'info' : 'error');
        return;
      }
      case 'preview': case 'show': {
        const root = pickApp(folder);
        if (root === null) return;
        preview.setApp(root);
        showPane('preview');
        return;
      }
      default:
        chat.system(`usage: /softn new [folder] | import | export [folder] | check [folder] | preview [folder] | apps
  new            start a SoftN app in a new project; with a folder, in that folder of this one
  import         unpack a .softn file into a folder of this project
  export         download an app as a .softn file (the one in the preview, or the folder named)
  check          check an app's files and render it
  preview        show an app in the live preview
  apps           list the SoftN apps in this project
A project can hold several apps, each in its own folder (any folder whose manifest.json has a .ui "main").`);
    }
  }

  // Project actions: a row of buttons on a wide screen, a ☰ menu on a phone.
  const closeMenu = () => header.classList.remove('menu-open');
  const actions = h(
    'div.actions',
    { onclick: (e: Event) => { if ((e.target as HTMLElement).closest('button')) closeMenu(); } },
    h('button', { title: 'New empty project', onclick: async () => {
      const name = await askText({ title: 'New project', message: 'An empty project, kept in this browser.', label: 'Project name', value: 'untitled', ok: 'Create' });
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
    h('button', { title: 'Start a SoftN app: from a starter, a blank page or an example; in a new project or a folder of this one', onclick: () => newSoftnApp().catch((error: unknown) => chat.system(`Could not start the app: ${(error as Error).message}`, 'error')) }, 'New SoftN app'),
    h('button', { title: 'Unpack a .softn file into a folder of this project', onclick: () => importSoftnFile() }, 'Import .softn…'),
    h('button', { title: 'Download the SoftN app in the preview as a .softn file', onclick: () => void exportSoftn() }, 'Export .softn'),
    h('button', { title: 'Rename this project', onclick: async () => {
      const name = await askText({ title: 'Rename project', label: 'Project name', value: project.meta.name, ok: 'Rename' });
      if (name && name !== project.meta.name) {
        project.meta = await renameProject(project.meta, name);
        await renderProjects();
      }
    } }, 'Rename'),
    h('button.danger', { title: 'Delete this project from the browser', onclick: async () => {
      if (!(await confirmAction({ title: 'Delete project', message: `Delete "${project.meta.name}" and all its files from this browser? This cannot be undone (export it as a .zip first to keep a copy).`, ok: 'Delete project', danger: true }))) return;
      const doomed = project.meta.id;
      const others = (await listProjects()).filter((m) => m.id !== doomed);
      const next = others[0] ?? (await createProject('untitled'));
      await openProject(next);
      await deleteProject(doomed);
      await renderProjects();
    } }, 'Delete'),
  );
  const header = h(
    'header.topbar',
    h('button.menu-toggle', { title: 'Project menu', 'aria-label': 'Project menu', onclick: () => header.classList.toggle('menu-open') }, '☰'),
    h('div.brand', h('span.logo', '◆'), h('span.brand-name', ' bot.computer')),
    projectSelect,
    actions,
    h('div.spacer'),
    gateChip,
    providerChip,
    h('button.settings-button', { title: 'AI providers', 'aria-label': 'Settings', onclick: () => void editSettings() }, '⚙', h('span.label', ' Settings')),
  );
  projectSelect.addEventListener('change', async () => {
    const meta = (await listProjects()).find((m) => m.id === projectSelect.value);
    if (meta) await openProject(meta);
  });

  // The center shows the editor or the app preview, with the terminal below.
  const centerTabs = h(
    'div.center-tabs',
    h('button', { 'data-pane': 'editor', onclick: () => showPane('editor') }, 'Editor'),
    h('button', { 'data-pane': 'preview', onclick: () => showPane('preview') }, 'App preview'),
  );
  const center = h('div.center', centerTabs, h('div.center-main', editor.element, preview.element), terminal.element);
  function showPane(pane: 'editor' | 'preview'): void {
    center.dataset.pane = pane;
    for (const b of centerTabs.querySelectorAll('button')) b.classList.toggle('active', b.dataset.pane === pane);
    preview.setVisible(pane === 'preview' || workspace?.dataset.view === 'preview');
    if (window.matchMedia('(max-width: 900px)').matches && workspace && workspace.dataset.view !== 'agent') showView(pane);
  }

  // On a narrow screen one pane shows at a time, chosen from a tab bar.
  const workspace = h('main.workspace', tree.element, center, chat.element);
  const views = [['files', 'Files'], ['editor', 'Editor'], ['preview', 'Preview'], ['terminal', 'Terminal'], ['agent', 'Agent']] as const;
  const tabs = h('nav.tabs', ...views.map(([view, label]) => h('button', { 'data-view': view, onclick: () => showView(view) }, label)));
  function showView(view: (typeof views)[number][0]): void {
    workspace.dataset.view = view;
    for (const b of tabs.querySelectorAll('button')) b.classList.toggle('active', b.dataset.view === view);
    if (view === 'editor' || view === 'preview') {
      center.dataset.pane = view;
      for (const b of centerTabs.querySelectorAll('button')) b.classList.toggle('active', b.dataset.pane === view);
    }
    preview.setVisible(center.dataset.pane === 'preview' && (view === 'preview' || !window.matchMedia('(max-width: 900px)').matches));
    if (view === 'agent') chat.focus();
  }
  showPane('editor');
  showView('agent');
  // A tap anywhere outside the open project menu closes it.
  document.addEventListener('pointerdown', (e) => {
    if (header.classList.contains('menu-open') && !header.contains(e.target as Node)) closeMenu();
  });
  clear(app);
  app.append(header, workspace, tabs);
  renderChips();

  // Open the last project, or make the welcome one.
  const all = await listProjects();
  let meta = all.find((m) => m.id === settings.lastProjectId) ?? all[0];
  if (!meta) {
    meta = await createProject('Welcome');
    await openProject(meta);
    for (const [path, text] of WELCOME) project.vfs.writeFile(`/${path}`, text, { parents: true });
    await project.flush();
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
  // The SoftN reference loads on first use; fetch it now so it is cached for offline use.
  (window.requestIdleCallback ?? ((f: () => void) => setTimeout(f, 3000)))(() => warmKnowledge());
}

main().catch((error: unknown) => {
  document.body.textContent = `bot.computer failed to start: ${(error as Error).message}`;
});
