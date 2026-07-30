/**
 * DesktopPage — the dedicated page at `/desktop.html`.
 *
 * Expands the landing page's "Desktop" section into a full page: how the OAIY
 * Companion works, how to run it, the CustomService JSON format, and a live
 * "service library" of example JSON files you can download (served by the PHP
 * API from a folder — drop a file in and it shows up here).
 *
 * Shares the landing's design language (Fraunces display + Public Sans body,
 * token-based theme, indigo/violet accent, `btn` classes). Self-contained so
 * it doesn't couple to LandingPage's internal helpers.
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
/* Nav + footer                                                         */
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
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">GitHub</a>
          <a href={APP_URL}>Open app</a>
          <a href="#library">Library</a>
        </nav>
        <p className="font-mono text-[11px]" style={{ color: 'rgb(var(--color-text-tertiary))' }}>© 2026 oaiy.com · Apache-2.0</p>
      </div>
    </footer>
  );
}



/* ------------------------------------------------------------------ */
/* Sections                                                             */
/* ------------------------------------------------------------------ */

function Hero() {
  return (
    <section className="mx-auto max-w-6xl px-5 pt-16 pb-4 sm:px-8 sm:pt-24">
      <div className="mx-auto max-w-3xl text-center">
        <h1 className="lp-h1" style={{ fontSize: 'clamp(2.2rem, 4.6vw, 3.4rem)' }}>
          The OAIY Companion
        </h1>
        <p className="mx-auto mt-5 max-w-xl text-base sm:text-lg" style={{ color: 'rgb(var(--color-text-secondary))', lineHeight: 1.6 }}>
          A small system-tray app that does the heavy local work the browser can't —
          spawning model servers, downloading weights, and running Python — and exposes
          them to the web app over <code className="font-mono text-[0.92em]">localhost</code>.
        </p>
        <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
          <a href="#how" className="btn btn-primary btn-md">How it works <ArrowRight /></a>
          <a href="#library" className="btn btn-secondary btn-md">Browse the service library</a>
        </div>
        <p className="mt-4 inline-flex items-center gap-1.5 text-xs" style={{ color: 'rgb(var(--color-text-muted))' }}>
          <WindowsIcon /> Windows · optional · the web app works fully on its own
        </p>
      </div>
    </section>
  );
}

function HowItWorks() {
  const features = [
    { icon: <ServerIcon />, title: 'Manages local services', body: 'Install, start, stop and tail logs for Ollama, llama.cpp and custom Python rigs — from a tidy dashboard.' },
    { icon: <DownloadIcon />, title: 'Downloads models', body: 'Pull GGUFs and safetensors straight from Hugging Face with pause / resume and a quick-add catalogue.' },
    { icon: <PythonIcon />, title: 'Bundles Python', body: 'A portable Python runtime with reusable virtual envs, so two services can share one heavy install.' },
    { icon: <GlobeIcon />, title: 'Powers browser nodes', body: "Runs the headless browser sidecar that the web app's browser-automation nodes drive." },
  ];
  return (
    <Section id="how">
      <SectionHeading
        eyebrow="How it works"
        title="Two processes, one localhost handshake"
        sub="The web app runs in your browser. The companion runs in your tray. When the app loads it probes the companion on localhost — find it, and companion-managed services light up the palette automatically."
      />
      <div className="grid gap-6 lg:grid-cols-[1fr_0.85fr] lg:items-center">
        <Reveal>
          <div className="rounded-2xl p-6 sm:p-8" style={{ backgroundColor: 'rgb(var(--color-bg-elevated))', border: '1px solid rgb(var(--color-border-primary))', boxShadow: 'var(--shadow-md)' }}>
            <div className="flex flex-col items-stretch gap-3">
              <ProcessCard tone="99 102 241" badge="In your browser" title="OAIY web app" body="Flow building, palette UX, ffmpeg.wasm media, HTTP service calls." />
              <div className="flex items-center justify-center py-1" style={{ color: 'rgb(var(--color-text-muted))' }}>
                <span className="font-mono text-[10px] uppercase tracking-[0.18em]">localhost · 127.0.0.1:17972</span>
              </div>
              <div className="flex justify-center" aria-hidden="true"><LinkVertical /></div>
              <ProcessCard tone="124 58 237" badge="In your system tray" title="OAIY desktop companion" body="Model servers, Hugging Face downloads, portable Python venvs, browser sidecar." />
            </div>
            <p className="mt-5 text-center text-xs" style={{ color: 'rgb(var(--color-text-muted))' }}>
              The web app probes <code className="font-mono">/api/health</code> on load. No companion? Everything a browser can do still works.
            </p>
          </div>
        </Reveal>
        <div className="grid gap-3">
          {features.map((f, i) => (
            <Reveal key={f.title} delay={i * 70}>
              <div className="flex gap-4 rounded-xl p-4" style={{ backgroundColor: 'rgb(var(--color-bg-secondary) / 0.5)', border: '1px solid rgb(var(--color-border-secondary))' }}>
                <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg" style={{ backgroundColor: 'rgb(var(--accent-secondary) / 0.12)', color: 'rgb(var(--accent-secondary))' }}>{f.icon}</span>
                <div>
                  <h3 className="text-sm font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{f.title}</h3>
                  <p className="mt-0.5 text-[13px]" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.5 }}>{f.body}</p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </Section>
  );
}

function Capabilities() {
  const caps = [
    { icon: <PythonIcon />, title: 'Bundled Python + venvs', body: 'Ships a portable Python runtime with reusable virtual environments. Run Python services without touching your system Python — and two services can share one heavy install.' },
    { icon: <CodeIcon />, title: 'Runs custom code', body: 'A service is just an install script + a run command, so the companion can launch any local process — a model server, your own script, a whole cloned repo — and own its start / stop / restart.' },
    { icon: <ChipIcon />, title: 'Uses your local GPU', body: 'Inference runs on your own hardware. Whatever you install — CUDA, Metal, ROCm, llama.cpp, ComfyUI — uses the GPU directly; nothing round-trips to a cloud.' },
    { icon: <DownloadIcon />, title: 'Hugging Face downloads', body: 'Pull GGUFs and safetensors with pause / resume into a shared models dir that every service can read via ${modelsDir}.' },
    { icon: <TerminalIcon />, title: 'Logs, health & lifecycle', body: 'Tail each service’s stdout/stderr live, watch status, and a health probe confirms it’s actually up before the web app starts using it.' },
    { icon: <PlugIcon />, title: 'Localhost bridge', body: 'Every service is exposed on 127.0.0.1 only. The web app discovers them and lights up the palette — your data and compute never leave the machine.' },
  ];
  return (
    <Section id="capabilities">
      <SectionHeading
        eyebrow="What it can do"
        title="Local compute, on tap"
        sub="The companion is the bridge between the browser sandbox and your machine's real muscle — Python, arbitrary processes, and the GPU."
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {caps.map((c, i) => (
          <Reveal key={c.title} delay={i * 50}>
            <div className="h-full rounded-xl p-5" style={{ backgroundColor: 'rgb(var(--color-bg-secondary) / 0.5)', border: '1px solid rgb(var(--color-border-secondary))' }}>
              <span className="flex h-10 w-10 items-center justify-center rounded-lg" style={{ backgroundColor: 'rgb(var(--accent-primary) / 0.10)', color: 'rgb(var(--accent-primary))' }}>{c.icon}</span>
              <h3 className="mt-3 text-sm font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{c.title}</h3>
              <p className="mt-1 text-[13px]" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.55 }}>{c.body}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

function Install() {
  const steps = [
    { n: 1, title: 'Download & launch', body: 'Grab the companion for Windows and run it. It lives in the system tray — no window to babysit.' },
    { n: 2, title: 'It binds localhost', body: 'The companion serves a small HTTP API on 127.0.0.1:17972 and exposes its services there.' },
    { n: 3, title: 'Open the web app', body: 'On load the app probes /api/health. Found it? Companion services appear in the palette automatically.' },
  ];
  return (
    <Section id="install">
      <SectionHeading eyebrow="Getting set up" title="Running the companion" sub="It's optional and local-only. The app degrades gracefully without it — nodes that need a companion service show an actionable hint instead of silently failing." />
      <div className="grid gap-4 sm:grid-cols-3">
        {steps.map((s, i) => (
          <Reveal key={s.n} delay={i * 70}>
            <div className="h-full rounded-xl p-5" style={{ backgroundColor: 'rgb(var(--color-bg-secondary) / 0.5)', border: '1px solid rgb(var(--color-border-secondary))' }}>
              <span className="flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold" style={{ backgroundColor: 'rgb(var(--accent-primary) / 0.12)', color: 'rgb(var(--accent-primary))' }}>{s.n}</span>
              <h3 className="mt-3 text-sm font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{s.title}</h3>
              <p className="mt-1 text-[13px]" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.5 }}>{s.body}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Service JSON format                                                  */
/* ------------------------------------------------------------------ */

const FORMAT_FIELDS: { field: string; req?: boolean; desc: string }[] = [
  { field: 'id', req: true, desc: 'Unique id — also the on-disk filename for the template.' },
  { field: 'name', req: true, desc: 'Display name on the service card.' },
  { field: 'description', desc: 'One-line summary, shown in the library + Services panel.' },
  { field: 'category', desc: 'Groups it in the library, e.g. "LLM" / "Image".' },
  { field: 'defaultPort', desc: 'Port the service listens on. Available everywhere as ${port}.' },
  { field: 'install', desc: '{ "kind": "none" } or { "kind": "script", "windows", "unix" } — a per-OS install script run once.' },
  { field: 'run', desc: '{ command, args[], env, cwd } — how to launch it. A bare command resolves against the companion bin dir, then PATH.' },
  { field: 'health', desc: '{ url, timeoutSecs } — readiness probe; url supports ${port}.' },
  { field: 'docsUrl', desc: 'Optional link shown on the service card.' },
  { field: 'files', desc: '{ filename: contents } — bundled scripts written to the scripts dir, so a template is fully self-contained.' },
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
    <Section id="format">
      <SectionHeading
        eyebrow="The companion service format"
        title="A service = an install step + a run command"
        sub="Each local service you add to the companion is one JSON template: how to install it, how to launch it, and how to know it's healthy. Put a binary, a Python venv, or a whole repo behind it — the companion handles the lifecycle."
      />
      {/* Example + substitution reference, side by side */}
      <div className="grid gap-6 lg:grid-cols-2 lg:items-start">
        <Reveal>
          <div className="overflow-hidden rounded-2xl" style={{ border: '1px solid rgb(var(--color-border-primary))', boxShadow: 'var(--shadow-md)' }}>
            <div className="flex items-center gap-1.5 px-4 py-2.5" style={{ backgroundColor: 'rgb(var(--color-bg-tertiary))', borderBottom: '1px solid rgb(var(--color-border-secondary))' }}>
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(var(--color-text-muted) / 0.5)' }} />
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(var(--color-text-muted) / 0.5)' }} />
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(var(--color-text-muted) / 0.5)' }} />
              <span className="ml-2 font-mono text-[11px]" style={{ color: 'rgb(var(--color-text-tertiary))' }}>llama-cpp-server.json</span>
            </div>
            <pre className="overflow-x-auto p-4 text-[12px] leading-relaxed" style={{ backgroundColor: 'rgb(var(--color-bg-canvas))', color: 'rgb(var(--color-text-secondary))', margin: 0 }}>
              <code className="font-mono">{EXAMPLE_JSON}</code>
            </pre>
          </div>
        </Reveal>
        <Reveal delay={70}>
          <div className="flex flex-col gap-4">
            <CalloutBox label="Placeholders" hint="in run + health">
              {['${port}', '${dataDir}', '${binDir}', '${modelsDir}', '${modelDirs}'].map((p) => <Chip key={p}>{p}</Chip>)}
            </CalloutBox>
            <CalloutBox label="Install-script env vars" hint="point at the companion's dirs">
              {['OAIY_DATA_DIR', 'OAIY_VENVS_DIR', 'OAIY_BIN_DIR', 'OAIY_MODELS_DIR', 'OAIY_SCRIPTS_DIR'].map((v) => <Chip key={v}>{v}</Chip>)}
            </CalloutBox>
            <p className="text-xs" style={{ color: 'rgb(var(--color-text-muted))', lineHeight: 1.6 }}>
              Import a downloaded template via the companion's <strong style={{ color: 'rgb(var(--color-text-tertiary))' }}>Services → Add / Import</strong>. A template only runs commands you can read first — review before importing.
            </p>
          </div>
        </Reveal>
      </div>

      {/* Field reference — full width so descriptions have room to breathe */}
      <div className="mt-12">
        <h3 className="mb-4 text-center font-mono text-[11px] uppercase tracking-[0.2em]" style={{ color: 'rgb(var(--color-text-tertiary))' }}>Every field</h3>
        <div className="grid gap-px overflow-hidden rounded-2xl sm:grid-cols-2" style={{ backgroundColor: 'rgb(var(--color-border-secondary))', border: '1px solid rgb(var(--color-border-secondary))', boxShadow: 'var(--shadow-md)' }}>
          {FORMAT_FIELDS.map((f) => (
            <div key={f.field} className="flex flex-col gap-1 p-4 sm:flex-row sm:gap-4" style={{ backgroundColor: 'rgb(var(--color-bg-elevated))' }}>
              <div className="sm:w-28 sm:flex-shrink-0">
                <code className="font-mono text-[13px]" style={{ color: 'rgb(var(--accent-primary))' }}>{f.field}</code>
                {f.req && <span className="ml-1.5 text-[9px] uppercase tracking-wide" style={{ color: 'rgb(var(--color-text-muted))' }}>req</span>}
              </div>
              <p className="text-[13px]" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.5 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Live service library (fetched from the PHP API)                      */
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

// Companion service templates have no `icon` field — fall back to a per-category
// emoji so the cards still read at a glance.
const CATEGORY_EMOJI: Record<string, string> = {
  LLM: '🤖', Image: '🎨', Audio: '🎙️', Video: '🎬', Example: '🧩',
};

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
    <Section id="library">
      <SectionHeading
        eyebrow="Service library"
        title="Grab a ready-made companion service"
        sub="Example service templates you can download and import into the companion. Each is a plain .json file served live from a folder on the API — new examples just get dropped in."
      />

      {state === 'loading' && (
        <p className="text-center text-sm" style={{ color: 'rgb(var(--color-text-muted))' }}>Loading the library…</p>
      )}

      {state === 'error' && (
        <Reveal>
          <div className="mx-auto max-w-2xl rounded-xl p-5 text-center" style={{ backgroundColor: 'rgb(var(--color-bg-secondary) / 0.5)', border: '1px solid rgb(var(--color-border-secondary))' }}>
            <p className="text-sm" style={{ color: 'rgb(var(--color-text-secondary))' }}>The library is served by the OAIY API, which isn't reachable from here.</p>
            <p className="mt-2 text-xs" style={{ color: 'rgb(var(--color-text-muted))', lineHeight: 1.6 }}>
              Run the API (<code className="font-mono">api/</code>) and set <code className="font-mono">VITE_API_BASE</code> to its URL, or browse the example files directly in <code className="font-mono">api/service-library/</code>.
            </p>
          </div>
        </Reveal>
      )}

      {state === 'ok' && items.length === 0 && (
        <p className="text-center text-sm" style={{ color: 'rgb(var(--color-text-muted))' }}>No examples in the library yet — drop a .json into <code className="font-mono">api/service-library/</code>.</p>
      )}

      {state === 'ok' && items.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((it, i) => (
            <Reveal key={it.file} delay={i * 50}>
              <div className="flex h-full flex-col rounded-xl p-5" style={{ backgroundColor: 'rgb(var(--color-bg-secondary) / 0.5)', border: '1px solid rgb(var(--color-border-secondary))' }}>
                <div className="flex items-start gap-3">
                  <span className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg text-lg" style={{ backgroundColor: 'rgb(var(--accent-primary) / 0.10)' }}>{it.icon || CATEGORY_EMOJI[it.category] || '🔌'}</span>
                  <div className="min-w-0 flex-1">
                    <h3 className="truncate text-sm font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{it.name}</h3>
                    {it.category && <span className="font-mono text-[10px] uppercase tracking-wide" style={{ color: 'rgb(var(--accent-primary))' }}>{it.category}</span>}
                  </div>
                </div>
                <p className="mt-3 flex-1 text-[13px]" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.55 }}>{it.description}</p>
                <div className="mt-4 flex items-center justify-between">
                  <span className="font-mono text-[10px]" style={{ color: 'rgb(var(--color-text-muted))' }}>{it.file}</span>
                  <a
                    href={`${API_BASE}${it.downloadUrl}`}
                    download={it.file}
                    className="btn btn-secondary btn-sm"
                  >
                    <DownloadIcon /> Download
                  </a>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      )}

      <p className="mt-10 text-center text-xs" style={{ color: 'rgb(var(--color-text-muted))', lineHeight: 1.6 }}>
        Downloaded a template? In the <strong>companion</strong> open <strong>Services → Add / Import</strong> and pick the .json — it lands with its install + run wired up. Review the script first; the companion only runs what's in the file.
      </p>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Layout primitives + icons                                            */
/* ------------------------------------------------------------------ */

function Section({ id, children }: { id?: string; children: React.ReactNode }) {
  return <section id={id} className="mx-auto max-w-6xl px-5 py-16 sm:px-8 sm:py-24 scroll-mt-20">{children}</section>;
}

function SectionHeading({ eyebrow, title, sub }: { eyebrow: string; title: string; sub?: string }) {
  return (
    <div className="mx-auto mb-10 max-w-2xl text-center sm:mb-14">
      <span className="lp-kicker">{eyebrow}</span>
      <h2 className="lp-h2">{title}</h2>
      {sub && (
        <p
          className="mx-auto mt-4 max-w-xl text-sm sm:text-base"
          style={{ color: 'rgb(var(--color-text-secondary))', lineHeight: 1.6 }}
        >
          {sub}
        </p>
      )}
    </div>
  );
}

function Reveal({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  return <div className="lp-reveal h-full" style={{ animationDelay: `${delay}ms` }}>{children}</div>;
}

function CalloutBox({ label, hint, children }: { label: string; hint: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl p-4" style={{ backgroundColor: 'rgb(var(--color-bg-secondary) / 0.5)', border: '1px solid rgb(var(--color-border-secondary))' }}>
      <div className="mb-2 flex items-baseline gap-2">
        <span className="text-sm font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{label}</span>
        <span className="text-[11px]" style={{ color: 'rgb(var(--color-text-muted))' }}>{hint}</span>
      </div>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded px-1.5 py-0.5 font-mono text-[11px]" style={{ backgroundColor: 'rgb(var(--accent-primary) / 0.10)', color: 'rgb(var(--accent-primary))' }}>
      {children}
    </code>
  );
}

function ProcessCard({ tone, badge, title, body }: { tone: string; badge: string; title: string; body: string }) {
  return (
    <div className="rounded-xl p-4" style={{ backgroundColor: 'rgb(var(--color-bg-canvas))', border: `1px solid rgb(${tone} / 0.3)` }}>
      <span className="font-mono text-[10px] uppercase tracking-[0.16em]" style={{ color: `rgb(${tone})` }}>{badge}</span>
      <h3 className="mt-1 text-base font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{title}</h3>
      <p className="mt-0.5 text-[13px]" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.5 }}>{body}</p>
    </div>
  );
}

function LinkVertical() {
  return (
    <svg width="20" height="28" viewBox="0 0 20 28" fill="none" aria-hidden="true">
      <path d="M10 0v28" stroke="rgb(var(--color-border-primary))" strokeWidth="2" strokeDasharray="3 3" />
      <circle cx="10" cy="14" r="3" fill="rgb(var(--accent-primary))" />
    </svg>
  );
}

function ArrowRight() {
  return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-6-6l6 6-6 6" /></svg>;
}
function DownloadIcon() {
  return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" /></svg>;
}
function ServerIcon() {
  return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="7" rx="2" strokeWidth="2" /><rect x="3" y="13" width="18" height="7" rx="2" strokeWidth="2" /><path strokeLinecap="round" strokeWidth="2" d="M7 7.5h.01M7 16.5h.01" /></svg>;
}
function PythonIcon() {
  return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3c-3 0-4 1.5-4 3v2h5m-1 13c3 0 4-1.5 4-3v-2h-5m-3-3h8a3 3 0 003-3V7a3 3 0 00-3-3M11 16H7a3 3 0 01-3-3v-2a3 3 0 013-3" /></svg>;
}
function GlobeIcon() {
  return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" strokeWidth="2" /><path strokeWidth="2" d="M3 12h18M12 3a14 14 0 010 18M12 3a14 14 0 000 18" /></svg>;
}
function WindowsIcon() {
  return <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5.5l8-1.1v7.2H3V5.5zm0 13l8 1.1v-7.1H3v6zm9 1.2l9 1.3v-8.4h-9v7.1zm0-16.4v7.4h9V3.1L12 3.3z" /></svg>;
}
function CodeIcon() {
  return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 9l-4 3 4 3m8-6l4 3-4 3M14 5l-4 14" /></svg>;
}
function ChipIcon() {
  return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" strokeWidth="2" /><path strokeLinecap="round" strokeWidth="2" d="M10 3v2m4-2v2m-4 14v2m4-2v2M3 10h2m-2 4h2m14-4h2m-2 4h2" /></svg>;
}
function TerminalIcon() {
  return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2" strokeWidth="2" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 9l3 3-3 3m6 0h4" /></svg>;
}
function PlugIcon() {
  return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 7V3m6 4V3M7 7h10v4a5 5 0 01-10 0V7zm5 9v5" /></svg>;
}
