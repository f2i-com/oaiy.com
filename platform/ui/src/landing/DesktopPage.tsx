/**
 * DesktopPage — the dedicated page at `/desktop.html`.
 *
 * What OAIY Desktop is (a window and a tray icon, with the Agent, the flow editor, OAIY's engines,
 * plugins and the receptionist), how it and the flow editor in a browser find each other, how to
 * install it on Windows and Linux and what the installer does not carry, the headless server, the
 * service JSON format, and a live service library of example JSON files you can download (served by
 * the PHP API from a folder — drop a file in and it shows up here).
 *
 * The download button is the shared one (components/DownloadDesktop.tsx): the file for the visitor's
 * system, named at build time, with no request. This page does not look for a desktop either: only the
 * flow editor does that.
 *
 * Same design language as the landing page — the app's tokens, one grotesque
 * for display and text, JetBrains Mono for machine facts, sections as a
 * heading rail beside their content — and the same `lp-*` styles, so the two
 * pages read as one site. Self-contained so it doesn't couple to
 * LandingPage's internal helpers.
 */
import { useEffect, useState } from 'react';
import SiteNav from './SiteNav';
import DownloadDesktop from '../components/DownloadDesktop';
import { REPO_URL, RELEASES_URL, repoFolderUrl } from './repoLinks';
import { loadServiceLibrary, type LibraryItem } from './serviceLibrary';

const APP_URL = 'app.html';
const LANDING_URL = '/';
// Where the PHP API lives. Default: same-origin `/api`. For split hosting (or
// local dev where the api runs on its own port) set VITE_API_BASE, e.g.
// VITE_API_BASE=http://127.0.0.1:8099.
const API_BASE = ((import.meta.env.VITE_API_BASE as string | undefined) || '').replace(/\/$/, '');

export default function DesktopPage() {
  return (
    <div
      className="oaiy-site min-h-screen w-full"
      style={{ backgroundColor: 'rgb(var(--color-bg-primary))', color: 'rgb(var(--color-text-primary))' }}
    >
      <SiteNav
        page="desktop"
        sections={[
          { id: 'how', label: 'How it works' },
          { id: 'capabilities', label: 'Capabilities' },
          { id: 'install', label: 'Install' },
          { id: 'format', label: 'Service format' },
          { id: 'library', label: 'Service library' },
        ]}
      />
      <main id="main-content">
        <Hero />
        <HowItWorks />
        <Capabilities />
        <Install />
        <ServiceFormat />
        <Library />
      </main>
      <Footer />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Hero                                                                */
/* ------------------------------------------------------------------ */

function Hero() {
  return (
    <section id="top" className="oaiy-hero oaiy-desktop-hero">
      <div className="mx-auto max-w-6xl px-5 pb-14 pt-12 sm:px-8 sm:pb-20 sm:pt-20">
        <div className="oaiy-hero-copy">
          <p className="oaiy-kicker">OAIY DESKTOP / THE LOCAL RUNTIME</p>
          <h1 className="lp-reveal lp-h1" style={{ animationDelay: '60ms' }}>
            Your machine.<br /><em>Put to work.</em>
          </h1>
          <p className="lp-reveal lp-lede mt-6" style={{ animationDelay: '140ms' }}>
            OAIY Desktop is OAIY as an app: a window and a tray icon, with the Agent, the flow editor, OAIY&apos;s own
            models and a receptionist for your business phone, all on your own computer.
          </p>
          <div className="lp-reveal mt-8 flex flex-wrap items-start gap-3" style={{ animationDelay: '220ms' }}>
            <DownloadDesktop variant="primary" fallback={{ label: 'Open the web app', href: APP_URL }} />
            <a href="#install" className="btn btn-secondary btn-lg">
              Install steps
              <ArrowRight />
            </a>
          </div>
          <p
            className="lp-reveal mt-5 text-sm"
            style={{ animationDelay: '300ms', color: 'rgb(var(--color-text-tertiary))' }}
          >
            For Windows and Linux (Linux: AppImage and .deb, newer and less tested than Windows). The flow editor in your browser works on any device.
          </p>
        </div>
        <div className="oaiy-runtime-strip" aria-label="What OAIY Desktop does">
          <div><span>01 / THE AGENT</span><h2>Let it do the work.</h2><p>An agent that plans, runs code and shows a live preview, in projects on your computer.</p></div>
          <div><span>02 / YOUR MODELS</span><h2>Run them here.</h2><p>Language, image, video and speech models on your own hardware, and flows that use them.</p></div>
          <div><span>03 / YOUR PHONE</span><h2>Answer the line.</h2><p>An AI receptionist for calls and texts, with a calendar and contacts you keep.</p></div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* How it works                                                        */
/* ------------------------------------------------------------------ */

function HowItWorks() {
  const features = [
    { title: 'A window', body: 'Overview, the Agent, Flows, the AI Receptionist, Engines, Services and Connections. The Agent and the flow editor are pages of the window itself.' },
    { title: 'A tray icon', body: 'Closing the window hides it. OAIY keeps running from the tray, so calls are still answered and services stay up, until you choose Quit.' },
    { title: 'A local API', body: 'Its API is served on 127.0.0.1:17972, and only there unless you turn network access on. It is how the flow editor in your browser finds the desktop.' },
  ];
  return (
    <Section
      id="how"
      title="A window, a tray icon and a local API"
      sub="OAIY Desktop shows its pages in a window and keeps running from the tray. The same local API serves the flow editor in your browser."
      tone="var(--accent-secondary)"
    >
      <div className="grid gap-8 lg:grid-cols-[1fr_1fr] lg:items-start">
        <div className="lp-window">
          <div className="lp-window-bar">
            <span className="lp-window-file">how the two halves talk</span>
          </div>
          <div className="lp-canvas lp-procs bg-dotgrid">
            <ProcessCard tone="var(--accent-primary)" where="In your browser" title="OAIY web app" body="The flow editor: the canvas, the palette, ffmpeg.wasm media, HTTP service calls, the Zipp sandbox." />
            <div className="lp-proc-link" aria-hidden="true">
              <span className="lp-proc-wire" />
              <span className="lp-proc-addr">http://127.0.0.1:17972</span>
              <span className="lp-proc-wire" />
            </div>
            <ProcessCard tone="var(--accent-secondary)" where="On your computer" title="OAIY Desktop" body="The Agent, flows, OAIY's engines, plugins and the receptionist, in a window with a tray icon." />
            <p className="lp-proc-note">
              The flow editor looks for <code>/api/health</code> on your own computer when it opens. This page does not. No desktop? Everything a browser can do still works.
            </p>
          </div>
        </div>
        <dl className="lp-defs lp-defs-tight">
          {features.map((f) => (
            <div key={f.title} className="lp-def" style={{ ['--tone' as string]: 'var(--accent-secondary)' }}>
              <dt>{f.title}</dt>
              <dd>{f.body}</dd>
            </div>
          ))}
        </dl>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Capabilities                                                        */
/* ------------------------------------------------------------------ */

function Capabilities() {
  const caps = [
    { title: 'The Agent', body: "Projects with a file tree, an editor, a terminal and a live preview for web pages, apps and 3D models. It plans the work, does it one step at a time and reviews each step. It thinks with a model chosen in OAIY's engines, or with your ChatGPT account." },
    { title: 'Flows', body: 'The flow editor in the same window, with the history of every run. A flow can be a tool for the Agent, and a flow can hand a task to the Agent. Flows also run without a window, through the oaiy command-line tool.' },
    { title: "OAIY's engines", body: 'Written in Rust: language models, and pictures, video, speech, music, sound effects, 3D models, background removal and upscaling. Most need an NVIDIA GPU; language models also run on WebGPU or the CPU. The models download on first use, and the engines are not in the installer yet.' },
    { title: 'The AI Receptionist', body: 'Calls, texts, a calendar, contacts and your opening hours for a business phone line. A booking is a request you confirm. It needs the Aokie plugin and your phone connected over Bluetooth; the plugin is for Windows and is not published for download yet.' },
    { title: 'Plugins and connections', body: 'Plugins are programs OAIY runs and supervises, like Aokie for the phone. Connections pair apps such as FormLogic with this computer. The Agent can set them up with you.' },
    { title: 'Services and Python', body: "Install, start, stop and tail the logs of your own local servers and Python rigs, with a portable Python and reusable virtual environments. Download models from Hugging Face with pause and resume. A service is one JSON file (below)." },
  ];
  return (
    <Section
      id="capabilities"
      title="What OAIY does on your machine"
      sub="The desktop app is where the work that needs a real computer happens: models on the GPU, real files, the phone."
      tone="var(--signal-amber)"
    >
      <dl className="lp-defs">
        {caps.map((c) => (
          <div key={c.title} className="lp-def" style={{ ['--tone' as string]: 'var(--signal-amber)' }}>
            <dt>{c.title}</dt>
            <dd>{c.body}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Install                                                             */
/* ------------------------------------------------------------------ */

function Install() {
  return (
    <Section
      id="install"
      title="Installing OAIY Desktop"
      sub="For Windows and Linux; the Linux packages are newer and less tested. The installers are not code-signed yet, so Windows will warn you; how to go on is below."
      tone="var(--signal-green)"
    >
      <div className="mb-4 flex flex-wrap items-start gap-3">
        <DownloadDesktop variant="primary" fallback={{ label: 'Open the web app', href: APP_URL }} />
        <a className="btn btn-secondary btn-lg" href={repoFolderUrl('desktop')}>Installation documentation</a>
      </div>
      <p className="mb-10 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.55 }}>
        Every file has its SHA-256 in <code>SHA256SUMS.txt</code> on the <a href={RELEASES_URL} target="_blank" rel="noopener noreferrer" className="lp-star-inline">latest release</a>.
      </p>

      <div className="lp-install">
        <div className="lp-install-os">
          <h3 className="lp-mini-h">Windows</h3>
          <ol>
            <li>Download the setup file (<code>.exe</code>) and run it. It installs for your user account and asks for no administrator rights.</li>
            <li>Windows will say “Windows protected your PC”, because the installer is not code-signed yet: choose “More info”, then “Run anyway”.</li>
            <li>Open OAIY. Setup asks for the essentials (your AI, and what the Agent may change), then “Continue with the Agent” sets up the rest with you.</li>
          </ol>
        </div>
        <div className="lp-install-os">
          <h3 className="lp-mini-h">Linux</h3>
          <p className="mb-3 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.55 }}>The AppImage and the .deb are newer and less tested than the Windows installer.</p>
          <ol>
            <li>Download the AppImage, make it executable and run it: <code>chmod +x oaiy-desktop-*.AppImage &amp;&amp; ./oaiy-desktop-*.AppImage</code>.</li>
            <li>Or install the <code>.deb</code>: <code>sudo apt install ./oaiy-desktop-*.deb</code>.</li>
            <li>Open OAIY and follow setup, as on Windows.</li>
          </ol>
        </div>
      </div>

      <div className="lp-note mt-10">
        <p>The installer carries the dashboard, the Agent, the flow editor and the flow runner. OAIY&apos;s engines, the models and the Aokie plugin are not in it.</p>
        <p>An install without the engines has no models of its own; the Agent can still use ChatGPT or an API provider.</p>
      </div>

      <h3 className="lp-mini-h mt-12">On a Linux host: the headless server</h3>
      <p className="mb-5 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.55, maxWidth: '40rem' }}>
        <code>oaiy-server</code> is the same local API and services with no window, no tray icon and no webview, so nothing graphical needs installing, for a server that the <code>oaiy</code> command line or a web app drives. It has no Agent or flow editor to show. To listen on the network it needs a token.
      </p>
      <div className="lp-window">
        <div className="lp-window-bar">
          <span className="lp-window-file">oaiy-server-&lt;version&gt;-linux-x86_64.tar.gz</span>
        </div>
        <pre className="lp-code" style={{ ['--tone' as string]: 'var(--signal-green)' }}>
<span className="lp-code-c"># unpack it and start it with a token</span>{'\n'}
mkdir oaiy-server &amp;&amp; tar -xzf oaiy-server-*-linux-x86_64.tar.gz -C oaiy-server{'\n'}
cd oaiy-server{'\n'}
OAIY_SERVER_TOKEN=change-me ./oaiy-server{'\n\n'}
<span className="lp-code-c"># in another shell</span>{'\n'}
curl http://127.0.0.1:17972/api/health
        </pre>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Service JSON format                                                 */
/* ------------------------------------------------------------------ */

const FORMAT_FIELDS: { field: string; req?: boolean; desc: string }[] = [
  { field: 'id', req: true, desc: 'Unique id. Also the on-disk filename for the template.' },
  { field: 'name', req: true, desc: 'Display name on the service card.' },
  { field: 'description', desc: 'One-line summary, shown in the library and the Services panel.' },
  { field: 'category', desc: 'Groups it in the library, for example "LLM" or "Image".' },
  { field: 'defaultPort', desc: 'Port the service listens on. Available everywhere as ${port}.' },
  { field: 'install', desc: '{ "kind": "none" } or { "kind": "script", "windows", "unix" }: a per-OS install script, run once.' },
  { field: 'run', desc: '{ command, args[], env, cwd }: how to launch it. A bare command resolves against the desktop app\'s bin folder, then PATH.' },
  { field: 'health', desc: '{ url, timeoutSecs }: the readiness probe. The url may use ${port}.' },
  { field: 'docsUrl', desc: 'Optional link shown on the service card.' },
  { field: 'files', desc: '{ filename: contents }: bundled scripts written to the scripts folder, so a template is self-contained.' },
];

const EXAMPLE_JSON = `{
  "id": "comfyui",
  "name": "ComfyUI",
  "description": "Local image server with its own node graph.",
  "category": "Image",
  "defaultPort": 8188,

  "install": {
    "kind": "script",
    "windows": "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI %OAIY_DATA_DIR%\\comfyui && py -m venv %OAIY_VENVS_DIR%\\comfyui && %OAIY_VENVS_DIR%\\comfyui\\Scripts\\pip install -r %OAIY_DATA_DIR%\\comfyui\\requirements.txt"
  },

  "run": {
    "command": "\${dataDir}/venvs/comfyui/Scripts/python.exe",
    "args": ["\${dataDir}/comfyui/main.py", "--port", "\${port}"],
    "env": {},
    "cwd": "\${dataDir}/comfyui"
  },

  "health": { "url": "http://127.0.0.1:\${port}/", "timeoutSecs": 120 }
}`;

function ServiceFormat() {
  return (
    <Section
      id="format"
      title="A service is an install step and a run command"
      sub="Each local service is one JSON template: how to install it, how to launch it, and how to know it is healthy. Put a binary, a Python venv or a whole repo behind it; the desktop app handles the lifecycle."
      tone="var(--signal-cyan)"
    >
      <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr] lg:items-start">
        <div className="lp-window">
          <div className="lp-window-bar">
            <span className="lp-window-file">comfyui.json</span>
          </div>
          <pre className="lp-code"><code>{EXAMPLE_JSON}</code></pre>
        </div>
        <div className="flex flex-col gap-5">
          <div>
            <h3 className="lp-mini-h">Placeholders, in run and health</h3>
            <ul className="lp-nodes">
              {['${port}', '${dataDir}', '${binDir}', '${modelsDir}', '${modelDirs}'].map((p) => <li key={p}>{p}</li>)}
            </ul>
          </div>
          <div>
            <h3 className="lp-mini-h">Install-script environment variables</h3>
            <ul className="lp-nodes">
              {['OAIY_DATA_DIR', 'OAIY_VENVS_DIR', 'OAIY_BIN_DIR', 'OAIY_MODELS_DIR', 'OAIY_SCRIPTS_DIR'].map((v) => <li key={v}>{v}</li>)}
            </ul>
          </div>
          <p className="text-sm" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.55 }}>
            Import a downloaded template from the desktop app&apos;s Services panel. A template only runs commands you can read first, so read it before importing.
          </p>
        </div>
      </div>

      <h3 className="lp-mini-h mt-12">Every field</h3>
      <dl className="lp-fields">
        {FORMAT_FIELDS.map((f) => (
          <div key={f.field} className="lp-field">
            <dt>
              <code>{f.field}</code>
              {f.req && <span className="lp-field-req">required</span>}
            </dt>
            <dd>{f.desc}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Live service library (fetched from the PHP API)                     */
/* ------------------------------------------------------------------ */

function Library() {
  // 'unavailable': this copy of the site has no library (the standalone release build serves no /api).
  const [state, setState] = useState<'loading' | 'ok' | 'unavailable' | 'error'>('loading');
  const [items, setItems] = useState<LibraryItem[]>([]);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 12000);
    setState('loading');
    void loadServiceLibrary(API_BASE, controller.signal).then((result) => {
      window.clearTimeout(timeout);
      if (cancelled) return;
      if (result.state === 'ok') setItems(result.items);
      setState(result.state);
    });
    return () => { cancelled = true; controller.abort(); window.clearTimeout(timeout); };
  }, [attempt]);

  return (
    <Section
      id="library"
      title="Ready-made services"
      sub="A starting point for your local setup. Download a service template, review its commands and import it into OAIY Desktop."
      tone="var(--accent-primary)"
    >
      {state === 'loading' && (
        <p role="status" className="text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>Loading the library…</p>
      )}

      {state === 'unavailable' && (
        <div className="lp-note" role="status">
          <p>This copy of the site does not serve the service library. The same templates are in the OAIY repository on GitHub.</p>
          <div className="mt-4 flex flex-wrap gap-3">
            <a className="btn btn-secondary btn-sm" href={repoFolderUrl('serviceLibrary')}>Browse templates</a>
          </div>
        </div>
      )}

      {state === 'error' && (
        <div className="lp-note" role="status">
          <p>We couldn’t load the service library. Try again, or browse the templates on GitHub.</p>
          <div className="mt-4 flex flex-wrap gap-3">
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => setAttempt((value) => value + 1)}>Try again</button>
            <a className="btn btn-secondary btn-sm" href={repoFolderUrl('serviceLibrary')}>Browse templates</a>
          </div>
        </div>
      )}

      {state === 'ok' && items.length === 0 && (
        <p className="text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
          No templates are available yet. You can create a service using the example above.
        </p>
      )}

      {state === 'ok' && items.length > 0 && (
        <ul className="lp-library">
          {items.map((it) => (
            <li key={it.file} className="lp-library-item">
              <div className="min-w-0">
                <h3>{it.name}</h3>
                {it.category && <span className="lp-library-cat">{it.category}</span>}
                <p>{it.description}</p>
              </div>
              <div className="lp-library-foot">
                <span className="lp-library-file">{it.file}</span>
                <a href={`${API_BASE}${it.downloadUrl}`} download={it.file} className="btn btn-secondary btn-sm">
                  <DownloadIcon /> Download
                </a>
              </div>
            </li>
          ))}
        </ul>
      )}

      <p className="mt-8 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.55 }}>
        Downloaded a template? In the desktop app open Services and choose Import, then pick the .json. It arrives with its install and run steps wired up. The desktop app only runs what is in the file, so read it first.
      </p>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Footer + layout                                                     */
/* ------------------------------------------------------------------ */

function Footer() {
  return (
    <footer className="border-t" style={{ borderColor: 'rgb(var(--color-border-secondary))' }}>
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-5 py-8 sm:flex-row sm:px-8">
        <a href="index.html" className="flex items-baseline gap-3" aria-label="OAIY home">
          <span className="lp-wordmark">OAIY</span>
          <span className="lp-tagline hidden sm:inline">Orchestrate AI Yourself</span>
        </a>
        <nav className="flex items-center gap-5 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }} aria-label="Footer">
          <a href={LANDING_URL}>Home</a>
          <a href={APP_URL}>Open app</a>
          <a href="#library">Library</a>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">GitHub</a>
        </nav>
        <p className="text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>© 2026 oaiy.com, Apache-2.0</p>
      </div>
    </footer>
  );
}

function Section({
  id, title, sub, tone, children,
}: {
  id: string;
  title: string;
  sub?: string;
  tone?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="lp-section scroll-mt-28" style={{ ['--tone' as string]: tone ?? 'var(--accent-primary)' }}>
      <div className="mx-auto grid max-w-6xl gap-10 px-5 sm:px-8 lg:grid-cols-[minmax(0,17rem)_minmax(0,1fr)] lg:gap-16">
        <div className="lp-rail">
          <h2 className="lp-h2">{title}</h2>
          {sub && <p className="lp-rail-sub">{sub}</p>}
        </div>
        <div className="min-w-0">{children}</div>
      </div>
    </section>
  );
}

function ProcessCard({ tone, where, title, body }: { tone: string; where: string; title: string; body: string }) {
  return (
    <div className="lp-proc" style={{ ['--tone' as string]: tone }}>
      <span className="lp-proc-where">{where}</span>
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Icons                                                               */
/* ------------------------------------------------------------------ */

function ArrowRight() {
  return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-6-6l6 6-6 6" /></svg>;
}
function DownloadIcon() {
  return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" /></svg>;
}
