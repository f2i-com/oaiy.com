/**
 * LandingPage — the marketing root at `/`.
 *
 * The page is the app's own design system worn as a front door: Prism Lab dark
 * / Paper Circuit light, one grotesque carrying the display voice through
 * weight and tracking, JetBrains Mono reserved for machine facts (node ids,
 * ports, endpoints, run status), and the dot-grid canvas as the ground. Colour
 * is keyed the way the app keys it — by node family — so a section about
 * image nodes is magenta because image nodes ARE magenta on the canvas.
 *
 * The one thing allowed to move is the hero flow, and it moves once: on load
 * the graph runs exactly like a flow does in the app — each node lights up in
 * turn, the Condition takes its False branch and the graph regenerates, then
 * the second pass passes. That is what the product does, so it is the thing
 * the page shows first. Everything else on the page holds still.
 *
 * Theme/accent are shared with the app via ThemeContext (same localStorage
 * keys), so a returning user sees the landing in whatever theme they last
 * chose. No app boot runs here (see landing/main.tsx).
 */
import SiteNav, { REPO_URL } from './SiteNav';

const APP_URL = 'app.html';

export default function LandingPage() {
  return (
    <div
      className="min-h-screen w-full overflow-x-hidden"
      style={{ backgroundColor: 'rgb(var(--color-bg-primary))', color: 'rgb(var(--color-text-primary))' }}
    >
      <SiteNav page="overview" />
      <main>
        <Hero />
        <Engines />
        <Why />
        <Palette />
        <HowItWorks />
        <Privacy />
        <Desktop />
        <RemoteAI />
        <FinalCta />
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
    <section id="top" className="relative overflow-hidden bg-dotgrid">
      <div className="relative mx-auto max-w-6xl px-5 pt-12 sm:px-8 sm:pt-20 lg:pt-24">
        <div className="max-w-3xl">
          <h1 className="lp-reveal lp-h1" style={{ animationDelay: '60ms' }}>
            Build AI workflows you can see.
          </h1>
          <p
            className="lp-reveal lp-lede mt-6"
            style={{ animationDelay: '140ms' }}
          >
            OAIY is a node canvas for local AI. Chain language models, browser
            automation and image, audio and video generation by drawing the flow,
            and run it in your browser against the engines on your own machine.
          </p>
          <div className="lp-reveal mt-8 flex flex-wrap items-center gap-3" style={{ animationDelay: '220ms' }}>
            <a href={APP_URL} className="btn btn-primary btn-lg">
              Open the app
              <ArrowRight />
            </a>
            <a href="#desktop" className="btn btn-secondary btn-lg">
              Get OAIY Desktop
            </a>
          </div>
          <ul
            className="lp-reveal mt-5 flex flex-wrap gap-x-5 gap-y-1.5 text-sm"
            style={{ animationDelay: '300ms', color: 'rgb(var(--color-text-tertiary))' }}
          >
            <li className="inline-flex items-center gap-1.5"><CheckIcon /> Nothing to install</li>
            <li className="inline-flex items-center gap-1.5"><CheckIcon /> No sign-up</li>
            <li className="inline-flex items-center gap-1.5"><CheckIcon /> Keys stay on your device</li>
          </ul>
        </div>

        <div className="lp-reveal mt-12 sm:mt-16" style={{ animationDelay: '380ms' }}>
          <HeroGraph />
        </div>
      </div>
    </section>
  );
}

/**
 * HeroGraph — the "image review loop", drawn with the app's real nodes, and run.
 *
 * Generate an image, have the vision model review it, branch on the verdict.
 * True goes to Output; False feeds straight back into image_gen's Prompt
 * handle, so the graph regenerates. On load it does exactly that, once: every
 * node carries a status light and the wires carry the data, on one shared
 * 6.4-second timeline written as CSS keyframes (see `lp-run-*` in index.css).
 * Under prefers-reduced-motion the timeline collapses to its final frame — a
 * completed run — so nothing is lost, it simply arrives finished.
 *
 * Every id, label and port comes from the bundled-module definitions, so what
 * the page shows is what the palette gives you:
 *
 *   image_gen   Image Gen   Prompt: string      ⇒  Image: image
 *   ai_llm      AI LLM      Prompt/Image/Video  ⇒  Response: string
 *   condition   Condition   Value: any          ⇒  True / False
 *   output      Output      Result: any
 *
 * Accents follow each module's own colour, so a node here is the colour it
 * will be on the canvas.
 */
function HeroGraph() {
  const W = 168;
  const H = 74;
  const Y = 34;
  const cy = Y + H / 2; // the row's handle height
  const xs = [56, 264, 472, 680];

  const nodes = [
    { x: xs[0], id: 'image_gen', label: 'Image Gen', port: 'Image · image', icon: 'image', accent: 'var(--signal-magenta)', run: 'lp-run-a' },
    { x: xs[1], id: 'ai_llm', label: 'AI LLM', port: 'Response · string', icon: 'brain', accent: 'var(--accent-primary)', run: 'lp-run-b' },
    { x: xs[2], id: 'condition', label: 'Condition', port: 'True / False', icon: 'branch', accent: 'var(--signal-cyan)', run: 'lp-run-c' },
    { x: xs[3], id: 'output', label: 'Output', port: 'Result · any', icon: 'check', accent: 'var(--signal-green)', run: 'lp-run-d' },
  ] as const;

  const trueY = cy - 12;
  const falseY = cy + 12;
  const condRight = xs[2] + W;
  const loopBottom = 206;

  return (
    <div className="lp-window">
      <div className="lp-window-bar">
        <span className="lp-window-file">image-review-loop.oaiy</span>
        <span className="lp-run-chip" aria-hidden="true">
          <span className="lp-run-dot" />
          <span className="lp-run-label lp-run-label-running">Running</span>
          <span className="lp-run-label lp-run-label-done">Done · passed on the second try</span>
        </span>
      </div>

      <div className="lp-canvas bg-dotgrid">
        <svg
          viewBox="0 0 904 236"
          className="lp-canvas-svg"
          role="img"
          aria-label="An example OAIY flow, shown running: an Image Gen node feeds an AI LLM node that reviews the image, and a Condition node either sends the result to Output or loops back to regenerate the image. The first pass fails the review and regenerates; the second pass passes."
        >
          <defs>
            <marker id="lp-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0 1l5 3-5 3z" fill="rgb(var(--color-text-tertiary))" />
            </marker>
            <marker id="lp-arrow-ok" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0 1l5 3-5 3z" fill="rgb(var(--signal-green))" />
            </marker>
            <marker id="lp-arrow-loop" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0 1l5 3-5 3z" fill="rgb(var(--signal-amber))" />
            </marker>
          </defs>

          {/* forward wires */}
          <path className="lp-wire lp-wire-1" d={`M${xs[0] + W} ${cy} L${xs[1] - 8} ${cy}`} markerEnd="url(#lp-arrow)" />
          <path className="lp-wire lp-wire-2" d={`M${xs[1] + W} ${cy} L${xs[2] - 8} ${cy}`} markerEnd="url(#lp-arrow)" />

          {/* Condition · True → Output */}
          <g className="lp-wire-true">
            <path
              d={`M${condRight} ${trueY} C${condRight + 18} ${trueY} ${condRight + 6} ${cy} ${condRight + 24} ${cy} L${xs[3] - 8} ${cy}`}
              fill="none"
              stroke="rgb(var(--signal-green))"
              strokeWidth="2"
              markerEnd="url(#lp-arrow-ok)"
            />
            <text x={condRight + 20} y={trueY - 9} textAnchor="middle" className="lp-wire-label" fill="rgb(var(--signal-green))">
              True
            </text>
          </g>

          {/* Condition · False → back into image_gen's Prompt handle. */}
          <g className="lp-wire-false">
            <path
              d={`M${condRight} ${falseY} H${condRight + 12} Q${condRight + 28} ${falseY} ${condRight + 28} ${falseY + 16} V${loopBottom - 16} Q${condRight + 28} ${loopBottom} ${condRight + 12} ${loopBottom} H${xs[0] - 16} Q${xs[0] - 32} ${loopBottom} ${xs[0] - 32} ${loopBottom - 16} V${cy + 16} Q${xs[0] - 32} ${cy} ${xs[0] - 16} ${cy} H${xs[0] - 9}`}
              fill="none"
              stroke="rgb(var(--signal-amber))"
              strokeWidth="2"
              strokeDasharray="7 6"
              markerEnd="url(#lp-arrow-loop)"
              className="lp-edge-flow"
            />
            <text x={(xs[0] + condRight) / 2} y={loopBottom + 20} textAnchor="middle" className="lp-wire-label" fill="rgb(var(--signal-amber))">
              False · regenerate
            </text>
          </g>

          {nodes.map((n) => (
            <GraphNode key={n.id} {...n} y={Y} w={W} h={H} outputs={n.id === 'condition' ? 2 : 1} />
          ))}
        </svg>
      </div>
    </div>
  );
}

function GraphNode({
  x, y, w, h, accent, label, id, port, icon, outputs, run,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  accent: string;
  label: string;
  /** The real node id from the module definition — shown as the node's kicker. */
  id: string;
  /** The output port + its type, as the palette declares it. */
  port: string;
  icon: 'image' | 'brain' | 'branch' | 'check';
  /** Condition has two right-edge handles; everything else has one. */
  outputs: 1 | 2;
  /** Which lane of the shared run timeline this node plays. */
  run: string;
}) {
  const hasOutput = id !== 'output';
  return (
    <g>
      <rect
        className={`lp-graph-card lp-card ${run}`}
        x={x}
        y={y}
        width={w}
        height={h}
        rx={14}
        fill="rgb(var(--color-bg-tertiary))"
      />
      <rect x={x + 13} y={y + 13} width={30} height={30} rx={9} fill={`rgb(${accent} / 0.16)`} />
      <g
        transform={`translate(${x + 21.5} ${y + 21.5})`}
        stroke={`rgb(${accent})`}
        strokeWidth="1.5"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <NodeGlyph icon={icon} />
      </g>
      <text x={x + 53} y={y + 23} fontSize="9" fill={`rgb(${accent})`} style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.1em' }}>
        {id.toUpperCase()}
      </text>
      <text x={x + 53} y={y + 40} fontSize="16" fontWeight="620" fill="rgb(var(--color-text-primary))" style={{ fontFamily: 'var(--font-body)', letterSpacing: '-0.01em' }}>
        {label}
      </text>
      <text x={x + 14} y={y + h - 12} fontSize="10" fill="rgb(var(--color-text-tertiary))" style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.02em' }}>
        {port}
      </text>

      {/* status light — the same idle / running / done the canvas shows */}
      <circle className={`lp-status ${run}`} cx={x + w - 16} cy={y + 16} r={4.5} />

      {/* typed handles — left in, right out, matching the canvas */}
      <circle cx={x} cy={y + h / 2} r={5.5} fill="rgb(var(--color-bg-canvas))" stroke={`rgb(${accent})`} strokeWidth="2.4" />
      {hasOutput && outputs === 1 && (
        <circle cx={x + w} cy={y + h / 2} r={5.5} fill="rgb(var(--color-bg-canvas))" stroke={`rgb(${accent})`} strokeWidth="2.4" />
      )}
      {hasOutput && outputs === 2 && (
        <>
          <circle cx={x + w} cy={y + h / 2 - 12} r={5.5} fill="rgb(var(--color-bg-canvas))" stroke="rgb(var(--signal-green))" strokeWidth="2.4" />
          <circle cx={x + w} cy={y + h / 2 + 12} r={5.5} fill="rgb(var(--color-bg-canvas))" stroke="rgb(var(--signal-amber))" strokeWidth="2.4" />
        </>
      )}
    </g>
  );
}

/** Glyphs mirroring each module's declared lucide icon, on a 13x13 box. */
function NodeGlyph({ icon }: { icon: 'image' | 'brain' | 'branch' | 'check' }) {
  switch (icon) {
    case 'image':
      return (
        <>
          <rect x="0" y="0.5" width="13" height="12" rx="2.5" />
          <circle cx="3.8" cy="4.2" r="1.3" />
          <path d="M0.5 10l3.8-3.3 3.3 2.7 2.1-1.6 2.8 2.7" />
        </>
      );
    case 'brain':
      return (
        <>
          <path d="M6.5 1.6a2.3 2.3 0 00-4.2 1.4 2.2 2.2 0 00-.7 3.5 2.2 2.2 0 001.9 3.4 2.3 2.3 0 003 0.7z" />
          <path d="M6.5 1.6a2.3 2.3 0 014.2 1.4 2.2 2.2 0 01.7 3.5 2.2 2.2 0 01-1.9 3.4 2.3 2.3 0 01-3 0.7z" />
        </>
      );
    case 'branch':
      return (
        <>
          <path d="M2.4 2.6v7.8" />
          <circle cx="2.4" cy="1.2" r="1.3" />
          <circle cx="2.4" cy="11.8" r="1.3" />
          <circle cx="10.6" cy="3.5" r="1.3" />
          <path d="M10.6 4.8v1.1a3 3 0 01-3 3H2.4" />
        </>
      );
    case 'check':
      return (
        <>
          <circle cx="6.5" cy="6.5" r="6" />
          <path d="M3.8 6.7l2.1 2.1 4-4.2" />
        </>
      );
  }
}

/* ------------------------------------------------------------------ */
/* Engines                                                             */
/* ------------------------------------------------------------------ */

const ENGINES = [
  'Ollama', 'LM Studio', 'llama.cpp', 'vLLM', 'ComfyUI', 'Stable Diffusion',
  'OpenAI', 'Anthropic', 'Any OpenAI-compatible server', 'Any HTTP endpoint',
];

function Engines() {
  return (
    <section className="lp-engines" aria-label="Supported AI engines">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-5 sm:px-8 lg:flex-row lg:items-center lg:gap-10">
        <p className="lp-engines-lead">Talks to the engines you already run</p>
        <ul className="flex flex-wrap gap-2">
          {ENGINES.map((name) => (
            <li key={name} className="lp-engine">{name}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Why                                                                 */
/* ------------------------------------------------------------------ */

function Why() {
  const items = [
    {
      tone: 'var(--accent-primary)',
      title: 'The logic is the diagram',
      body: 'Drag nodes, wire typed handles, and watch data move through. Conditions, loops, macros and subflows are all things you draw, not things you configure.',
    },
    {
      tone: 'var(--signal-green)',
      title: 'Your machine does the work',
      body: 'The flow lives in your browser and inference runs on your engines. OAIY orchestrates; nothing is proxied through a server.',
    },
    {
      tone: 'var(--signal-cyan)',
      title: 'Any AI can press Run',
      body: 'Share a flow as a link and let ChatGPT, Claude or any HTTP client queue runs that execute on your hardware, end-to-end encrypted if you want.',
    },
    {
      tone: 'var(--signal-amber)',
      title: 'Build once, reuse everywhere',
      body: 'Capture a subgraph as a macro, nest flows as subflows, and pass whole flows around as JSON or a private link.',
    },
  ];
  return (
    <Section
      id="why"
      title="The power of code. The clarity of a diagram."
      sub="Everything you would otherwise script — prompts, branching, retries, media pipelines — laid out as a graph you can read at a glance."
    >
      <dl className="lp-defs">
        {items.map((it) => (
          <div key={it.title} className="lp-def" style={{ ['--tone' as string]: it.tone }}>
            <dt>{it.title}</dt>
            <dd>{it.body}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Palette — the app's node catalogue, drawn the way the app draws it  */
/* ------------------------------------------------------------------ */

/**
 * Families and node ids come straight from `src/bundled-modules/*` — each
 * family's swatch is the module's declared colour, mapped onto the design
 * system's signal palette exactly as the canvas maps it.
 */
const FAMILIES: { name: string; tone: string; body: string; nodes: string[] }[] = [
  { name: 'AI', tone: 'var(--accent-primary)', body: 'Chat, vision and raw requests against local or cloud models. Stream tokens, template prompts, fan out across providers.', nodes: ['ai_llm'] },
  { name: 'Browser', tone: 'var(--signal-cyan)', body: 'Fetch pages, drive sessions, and extract text, links, contacts, images, JSON-LD and metadata.', nodes: ['browser_page', 'browser_extract', 'browser_action', 'browser_request', 'browser_session'] },
  { name: 'Image', tone: 'var(--signal-magenta)', body: 'Generate, resize, view and save images, including whole ComfyUI graphs driven from one node.', nodes: ['image_gen', 'image_resize', 'image_view', 'image_save'] },
  { name: 'Audio', tone: 'var(--signal-green)', body: 'Speech in both directions, music generation, and append and fade edits.', nodes: ['text_to_speech', 'speech_to_text', 'music_gen', 'audio_append', 'audio_fade'] },
  { name: 'Video', tone: 'var(--signal-amber)', body: 'Generate, caption, cut, stack and mix video in the browser with ffmpeg.wasm. No uploads.', nodes: ['video_gen', 'video_captions', 'video_append', 'video_pip', 'audio_mixer'] },
  { name: 'Flow control', tone: 'var(--signal-amber)', body: 'Branch on real data, loop over lists, and package subgraphs as macros and subflows.', nodes: ['condition', 'loop_start', 'loop_end', 'macro', 'subflow', 'output'] },
  { name: 'Files and data', tone: 'var(--signal-green)', body: 'Read and write files, chunk long text, and query a built-in SQLite database.', nodes: ['file_read', 'file_write', 'text_chunker', 'database_query'] },
  { name: 'Inputs, services and code', tone: 'var(--signal-cyan)', body: 'Take text, files and media in; call any registered endpoint; drop into JavaScript when a node does not exist yet.', nodes: ['input_text', 'input_file', 'service_call', 'logic_block', 'template'] },
];

function Palette() {
  return (
    <Section
      id="capabilities"
      title="One palette, the whole pipeline"
      sub="Language models are the start. The palette holds enough nodes for a single flow to browse, generate, listen, decide and write."
    >
      <div className="lp-palette">
        {FAMILIES.map((f) => (
          <div key={f.name} className="lp-family" style={{ ['--tone' as string]: f.tone }}>
            <div className="lp-family-head">
              <span className="lp-swatch" aria-hidden="true" />
              <h3>{f.name}</h3>
            </div>
            <p>{f.body}</p>
            <ul className="lp-nodes" aria-label={`${f.name} nodes`}>
              {f.nodes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* How it works                                                        */
/* ------------------------------------------------------------------ */

function HowItWorks() {
  const steps = [
    { title: 'Open the canvas', body: 'It runs in your browser, so there is nothing to install. The first-run wizard wires a starter chat flow in three clicks.' },
    { title: 'Point it at your engines', body: 'Add Ollama, LM Studio, ComfyUI, OpenAI or any HTTP endpoint as a Service and test the connection in place. Keys stay on your device.' },
    { title: 'Run, then share', body: 'Press Run and watch each node light up. Share the flow as a link, or hand the link to an AI and let it trigger runs on your hardware.' },
  ];
  return (
    <Section id="how" title="From a blank canvas to a running flow">
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
/* Privacy                                                             */
/* ------------------------------------------------------------------ */

function Privacy() {
  const points = [
    { title: 'The flow stays in your browser', body: 'It lives in local storage and the JSON you export. It reaches a server only if you share it.' },
    { title: 'Inference runs on your machine', body: 'OAIY orchestrates and your engines compute. No prompt, image or result passes through oaiy.com.' },
    { title: 'Keys never leave your device', body: 'Secrets resolve against your local constants at run time. The sharing backend cannot see them.' },
    { title: 'Flow code runs in a sandbox', body: 'Code nodes execute inside Zipp, a JavaScript engine compiled to WebAssembly with no network, no storage and hard limits on CPU and memory.' },
    { title: 'Shares can be sealed', body: 'Set a password and a shared flow is encrypted with AES-GCM before it leaves the browser. Lose the password, lose the flow — by design.' },
  ];
  return (
    <Section
      id="privacy"
      title="Built to keep your data yours"
      sub="Local-first on purpose. The optional backend is a meeting point for sharing, never a place your prompts, keys or results pass through in the clear."
      tone="var(--signal-green)"
    >
      <ul className="lp-checks">
        {points.map((p) => (
          <li key={p.title}>
            <span className="lp-check" aria-hidden="true"><CheckIcon /></span>
            <div>
              <h3>{p.title}</h3>
              <p>{p.body}</p>
            </div>
          </li>
        ))}
      </ul>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* OAIY Desktop                                                        */
/* ------------------------------------------------------------------ */

function Desktop() {
  const features = [
    { title: 'Manages local services', body: 'Install, start, stop and tail logs for Ollama, llama.cpp and custom Python rigs.' },
    { title: 'Downloads models', body: 'Pull GGUF and safetensors weights from Hugging Face, with pause and resume and a curated quick-add list.' },
    { title: 'Bundles Python', body: 'A portable runtime with reusable virtual environments, so two services can share one heavy install.' },
    { title: 'Runs the browser nodes', body: 'Hosts the headless browser that the web app\'s browser-automation nodes drive.' },
  ];
  return (
    <Section
      id="desktop"
      title="Optional muscle for the heavy local work"
      sub="The web app does everything a browser can. For the rest — starting model servers, downloading weights, running Python — there is a small tray companion."
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
              The web app probes <code>/api/health</code> when it loads. If the desktop app answers, its services appear in the palette.
            </p>
          </div>
        </div>

        <div>
          <dl className="lp-defs lp-defs-tight">
            {features.map((f) => (
              <div key={f.title} className="lp-def" style={{ ['--tone' as string]: 'var(--accent-secondary)' }}>
                <dt>{f.title}</dt>
                <dd>{f.body}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <a href="desktop.html" className="btn btn-primary btn-md">
              About OAIY Desktop
            </a>
            <span className="inline-flex items-center gap-1.5 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
              <WindowsIcon /> Windows, optional
            </span>
          </div>
        </div>
      </div>
    </Section>
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
/* Remote AI                                                           */
/* ------------------------------------------------------------------ */

function RemoteAI() {
  return (
    <Section
      id="remote"
      title="Let an AI run your flows"
      sub="Share a flow and you get two links: one that can view it and one that can queue runs. Hand the second to ChatGPT or Claude with a one-line prompt. It reads the manifest, fills the inputs and triggers a run; your browser picks it up, runs it on your machine and posts the result back."
      tone="var(--signal-cyan)"
    >
      <div className="lp-window">
        <div className="lp-window-bar">
          <span className="lp-window-file">what the AI sends</span>
        </div>
        <pre className="lp-code">
<span className="lp-code-c"># read what the flow accepts</span>{'\n'}
<span className="lp-code-m">GET</span>  /api/flows/<span className="lp-code-v">{'{hash_edit}'}</span>/manifest{'\n\n'}
<span className="lp-code-c"># queue a run with inputs</span>{'\n'}
<span className="lp-code-m">POST</span> /api/flows/<span className="lp-code-v">{'{hash_edit}'}</span>/runs{'\n'}
{'     '}{'{ "inputs": { "prompt": "summarise this" } }'}{'\n\n'}
<span className="lp-code-c"># poll until the status is done</span>{'\n'}
<span className="lp-code-m">GET</span>  /api/flows/<span className="lp-code-v">{'{hash_edit}'}</span>/runs/<span className="lp-code-v">{'{run_id}'}</span>
        </pre>
      </div>
      <ul className="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
        <li className="inline-flex items-center gap-2"><CheckIcon /> 110-bit link secrets</li>
        <li className="inline-flex items-center gap-2"><CheckIcon /> View and edit access kept apart</li>
        <li className="inline-flex items-center gap-2"><CheckIcon /> Optional end-to-end encryption</li>
      </ul>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Final CTA + footer                                                  */
/* ------------------------------------------------------------------ */

function FinalCta() {
  return (
    <section className="mx-auto max-w-6xl px-5 pb-20 pt-4 sm:px-8 sm:pb-28">
      <div className="lp-cta bg-dotgrid">
        <h2 className="lp-h2">Wire up your first flow.</h2>
        <p className="lp-lede mt-4 max-w-lg">
          No account, no install, no cloud bill. Open the canvas and have something running in the next five minutes.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <a href={APP_URL} className="btn btn-primary btn-lg">
            Open the app
            <ArrowRight />
          </a>
          <a href="#desktop" className="btn btn-secondary btn-lg">About the desktop app</a>
        </div>
        <p className="mt-8 text-sm" style={{ color: 'rgb(var(--color-text-secondary))' }}>
          OAIY is open source under Apache-2.0.{' '}
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="lp-star-inline">
            Star it on GitHub
          </a>{' '}
          if it is useful to you.
        </p>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t" style={{ borderColor: 'rgb(var(--color-border-secondary))' }}>
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-5 py-8 sm:flex-row sm:px-8">
        <a href="index.html" className="flex items-baseline gap-3" aria-label="OAIY home">
          <span className="lp-wordmark">OAIY</span>
          <span className="lp-tagline hidden sm:inline">Orchestrate AI Yourself</span>
        </a>
        <nav className="flex items-center gap-5 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }} aria-label="Footer">
          <a href={APP_URL}>Open app</a>
          <a href="#capabilities">Palette</a>
          <a href="#desktop">Desktop</a>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">GitHub</a>
        </nav>
        <p className="text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
          © 2026 oaiy.com, Apache-2.0
        </p>
      </div>
    </footer>
  );
}

/* ------------------------------------------------------------------ */
/* Layout                                                              */
/* ------------------------------------------------------------------ */

/**
 * A section is a heading rail beside its content: the heading and its one
 * paragraph sit left and stay put while the content scrolls on wide screens,
 * and stack above it on narrow ones. `tone` colours the rail's rule, which is
 * how a section says which part of the app it is about.
 */
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
    <section id={id} className="lp-section scroll-mt-20" style={{ ['--tone' as string]: tone ?? 'var(--accent-primary)' }}>
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

/* ------------------------------------------------------------------ */
/* Icons                                                               */
/* ------------------------------------------------------------------ */

function ArrowRight() {
  return (
    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-6-6l6 6-6 6" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
    </svg>
  );
}
function WindowsIcon() {
  return (
    <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 5.5L10.5 4.4v7.1H3V5.5zM10.5 12.5v7.1L3 18.5v-6h7.5zM11.5 4.2L21 3v8.5h-9.5V4.2zM21 12.5V21l-9.5-1.3v-7.2H21z" />
    </svg>
  );
}
