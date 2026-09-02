/**
 * DesktopPage — the dedicated page at `/desktop.html`.
 *
 * Expands the landing page's "Desktop" section into a full page: how OAIY
 * Desktop works, how to run it, the service JSON format, and a live service
 * library of example JSON files you can download (served by the PHP API from
 * a folder — drop a file in and it shows up here).
 *
 * Same design language as the landing page — the app's tokens, one grotesque
 * for display and text, JetBrains Mono for machine facts, sections as a
 * heading rail beside their content — and the same `lp-*` styles, so the two
 * pages read as one site. Self-contained so it doesn't couple to
 * LandingPage's internal helpers.
 */
import { useEffect, useState } from 'react';
import SiteNav, { REPO_URL } from './SiteNav';

const APP_URL = 'app.html';
const LANDING_URL = '/';
// Where the PHP API lives. Default: same-origin `/api`. For split hosting (or
// local dev where the api runs on its own port) set VITE_API_BASE, e.g.
// VITE_API_BASE=http://127.0.0.1:8099.
const API_BASE = ((import.meta.env.VITE_API_BASE as string | undefined) || '').replace(/\/$/, '');

export default function DesktopPage() {
  return (
    <div
      className="min-h-screen w-full overflow-x-hidden"
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
      <main>
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
    <section className="bg-dotgrid">
      <div className="mx-auto max-w-6xl px-5 pb-14 pt-12 sm:px-8 sm:pb-20 sm:pt-20">
        <div className="max-w-3xl">
          <h1 className="lp-reveal lp-h1" style={{ animationDelay: '60ms', fontSize: 'clamp(2.4rem, 5.5vw, 4rem)' }}>
            The desktop app does the heavy local work.
          </h1>
          <p className="lp-reveal lp-lede mt-6" style={{ animationDelay: '140ms' }}>
            A small system-tray app that does what a browser cannot: starts model
            servers, downloads weights, runs Python, hosts the headless browser.
            It hands all of that to the web app over localhost.
          </p>
          <div className="lp-reveal mt-8 flex flex-wrap items-center gap-3" style={{ animationDelay: '220ms' }}>
            <a href="#install" className="btn btn-primary btn-lg">
              Set it up
              <ArrowRight />
            </a>
            <a href="#library" className="btn btn-secondary btn-lg">Browse the service library</a>
          </div>
          <p
            className="lp-reveal mt-5 inline-flex items-center gap-1.5 text-sm"
            style={{ animationDelay: '300ms', color: 'rgb(var(--color-text-tertiary))' }}
          >
            <WindowsIcon /> Windows. Optional: the web app works fully on its own.
          </p>
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
    { title: 'Manages local services', body: 'Install, start, stop and tail logs for Ollama, llama.cpp and custom Python rigs.' },
    { title: 'Downloads models', body: 'Pull GGUF and safetensors weights from Hugging Face, with pause and resume and a curated quick-add list.' },
    { title: 'Bundles Python', body: 'A portable runtime with reusable virtual environments, so two services can share one heavy install.' },
    { title: 'Runs the browser nodes', body: 'Hosts the headless browser that the web app\'s browser-automation nodes drive.' },
  ];
  return (
    <Section
      id="how"
      title="Two processes, one localhost handshake"
      sub="The web app runs in your browser. OAIY Desktop runs in your tray. When the app loads it probes localhost; if the desktop app answers, its services appear in the palette."
      tone="var(--accent-secondary)"
    >
      <div className="grid gap-8 lg:grid-cols-[1fr_1fr] lg:items-start">
        <div className="lp-window">
          <div className="lp-window-bar">
            <span className="lp-window-file">how the two halves talk</span>
          </div>
          <div className="lp-canvas lp-procs bg-dotgrid">
            <ProcessCard tone="var(--accent-primary)" where="In your browser" title="OAIY web app" body="The canvas, the palette, ffmpeg.wasm media, HTTP service calls, the Zipp sandbox." />
            <div className="lp-proc-link" aria-hidden="true">
              <span className="lp-proc-wire" />
              <span className="lp-proc-addr">http://127.0.0.1:17972</span>
              <span className="lp-proc-wire" />
            </div>
            <ProcessCard tone="var(--accent-secondary)" where="In your system tray" title="OAIY Desktop" body="Model servers, Hugging Face downloads, portable Python, the browser sidecar." />
            <p className="lp-proc-note">
              The web app probes <code>/api/health</code> when it loads. No desktop app? Everything a browser can do still works.
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
    { title: 'Bundled Python and virtual environments', body: 'A portable Python runtime with reusable venvs. Run Python services without touching your system Python, and let two services share one heavy install.' },
    { title: 'Runs any local process', body: 'A service is an install script and a run command, so the desktop app can launch a model server, your own script or a whole cloned repo, and own its start, stop and restart.' },
    { title: 'Uses your GPU directly', body: 'Whatever you install — CUDA, Metal, ROCm, llama.cpp, ComfyUI — talks to the GPU itself. Nothing round-trips through a cloud.' },
    { title: 'Hugging Face downloads', body: 'Pull GGUF and safetensors files with pause and resume into a shared models folder that every service can read as ${modelsDir}.' },
    { title: 'Logs, health and lifecycle', body: 'Tail each service\'s output live, watch its status, and let a health probe confirm it is up before the web app starts using it.' },
    { title: 'Localhost only', body: 'Every service is exposed on 127.0.0.1 and nowhere else. The web app discovers them there; your data and compute never leave the machine.' },
  ];
  return (
    <Section
      id="capabilities"
      title="Local compute, on tap"
      sub="The desktop app is the bridge between the browser sandbox and the machine's real muscle: Python, arbitrary processes and the GPU."
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
  const steps = [
    { title: 'Download and launch', body: 'Get OAIY Desktop for Windows and run it. It lives in the system tray, so there is no window to keep open.' },
    { title: 'It binds localhost', body: 'The desktop app serves a small HTTP API on 127.0.0.1:17972 and exposes its services there.' },
    { title: 'Open the web app', body: 'On load the app probes /api/health. If the desktop app answers, its services appear in the palette.' },
  ];
  return (
    <Section
      id="install"
      title="Running OAIY Desktop"
      sub="Optional and local only. Without it the app still works; a node that needs a desktop service says so and tells you what to start."
      tone="var(--signal-green)"
    >
      <ol className="lp-steps">
        {steps.map((s, i) => (
          <li key={s.title} className="lp-step">
            <span className="lp-step-n" aria-hidden="true">{i + 1}</span>
            <h3>{s.title}</h3>
            <p>{s.body}</p>
          </li>
        ))}
      </ol>
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
  "id": "llama-cpp-server",
  "name": "llama.cpp server",
  "description": "OpenAI-compatible local LLM server.",
  "category": "LLM",
  "defaultPort": 8080,

  "install": {
    "kind": "script",
    "windows": "py -m venv %OAIY_VENVS_DIR%\\llamacpp && %OAIY_VENVS_DIR%\\llamacpp\\Scripts\\pip install llama-cpp-python[server]"
  },

  "run": {
    "command": "\${dataDir}/venvs/llamacpp/Scripts/python.exe",
    "args": ["-m", "llama_cpp.server", "--port", "\${port}",
             "--model", "\${modelsDir}/model.gguf"],
    "env": {},
    "cwd": null
  },

  "health": { "url": "http://127.0.0.1:\${port}/v1/models", "timeoutSecs": 90 }
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
            <span className="lp-window-file">llama-cpp-server.json</span>
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

interface LibraryItem {
  file: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  count: number;
  size: number;
  downloadUrl: string;
}

function Library() {
  const [state, setState] = useState<'loading' | 'ok' | 'error'>('loading');
  const [items, setItems] = useState<LibraryItem[]>([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/service-library`, { headers: { Accept: 'application/json' } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        setItems(Array.isArray(data.services) ? data.services : []);
        setState('ok');
      } catch {
        if (!cancelled) setState('error');
      }
    })();
    return () => { cancelled = true; };
  }, []);

  return (
    <Section
      id="library"
      title="Ready-made services"
      sub="Example service templates you can download and import into OAIY Desktop. Each is a plain .json file served live from a folder on the API; new examples are just dropped in."
      tone="var(--accent-primary)"
    >
      {state === 'loading' && (
        <p className="text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>Loading the library…</p>
      )}

      {state === 'error' && (
        <div className="lp-note">
          <p>The library is served by the OAIY API, which is not reachable from here.</p>
          <p>
            Run the API in <code>api/</code> and set <code>VITE_API_BASE</code> to its URL, or browse the example files directly in <code>api/service-library/</code>.
          </p>
        </div>
      )}

      {state === 'ok' && items.length === 0 && (
        <p className="text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
          No examples in the library yet. Drop a .json into <code className="font-mono">api/service-library/</code>.
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
function WindowsIcon() {
  return <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5.5l8-1.1v7.2H3V5.5zm0 13l8 1.1v-7.1H3v6zm9 1.2l9 1.3v-8.4h-9v7.1zm0-16.4v7.4h9V3.1L12 3.3z" /></svg>;
}
