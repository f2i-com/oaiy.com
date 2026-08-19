/**
 * LandingPage — the marketing root at `/`.
 *
 * A self-contained, scroll-friendly page that introduces OAIY and routes
 * people into the app (/app.html). Deliberately matches the app's own
 * design language — the OAIY design system's two themes (Prism Lab dark /
 * Paper Circuit light), one neutral grotesque carrying the display voice
 * through weight and negative tracking, the violet/blue primary with the
 * cyan-magenta-green-amber signal accents, the dot-grid canvas motif and
 * the node-graph visual vocabulary — so clicking "Open the app" feels like
 * walking through a door, not jumping to a different product.
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
      {/* No `sections` here: the overview's own sections (Capabilities, How it
          works) are already top-nav destinations, so a sub-nav would just
          repeat them. desktop.html passes its sections, which the top nav
          doesn't cover. */}
      <SiteNav page="overview" />
      <main>
        <Hero />
        <EnginesStrip />
        <ValueProps />
        <Capabilities />
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
/* Navigation                                                          */
/* ------------------------------------------------------------------ */




/* ------------------------------------------------------------------ */
/* Hero                                                                */
/* ------------------------------------------------------------------ */

function Hero() {
  return (
    <section id="top" className="relative overflow-hidden bg-dotgrid">
      {/* Ambient halo, echoing the app splash. */}
      <div
        className="lp-halo pointer-events-none absolute left-1/2 top-[38%] h-[760px] w-[760px] rounded-full blur-3xl"
        style={{
          background:
            'radial-gradient(closest-side, rgb(var(--accent-primary) / 0.5), rgb(var(--accent-secondary) / 0.28), transparent 72%)',
        }}
        aria-hidden="true"
      />

      <div className="relative mx-auto grid max-w-6xl items-center gap-7 px-5 pb-14 pt-8 sm:gap-12 sm:pb-20 sm:pt-16 sm:px-8 lg:grid-cols-[1fr_1.2fr] lg:gap-10 lg:pb-28 lg:pt-24">
        {/* Copy column */}
        <div>
          <div className="lp-reveal lp-pill mb-5" style={{ animationDelay: '40ms' }}>
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-pulse rounded-full" style={{ backgroundColor: 'rgb(var(--accent-primary))' }} />
            </span>
            Local-first AI workflow builder
          </div>

          <h1 className="lp-reveal lp-h1" style={{ animationDelay: '110ms' }}>
            Build AI workflows
            <br />
            <span style={{ color: 'rgb(var(--accent-primary))' }}>you can see.</span>
          </h1>

          <p
            className="lp-reveal mt-6 max-w-xl text-base sm:text-lg"
            style={{ animationDelay: '190ms', color: 'rgb(var(--color-text-secondary))', lineHeight: 1.6 }}
          >
            OAIY is a visual, node-based canvas for chaining LLMs, browser automation,
            and image, audio &amp; video generation — running right in your browser,
            against the AI engines on <em>your</em> machine. No boilerplate, no cloud
            lock-in, no code required.
          </p>

          <div className="lp-reveal mt-8 flex flex-wrap items-center gap-3" style={{ animationDelay: '270ms' }}>
            <a href={APP_URL} className="btn btn-primary btn-lg">
              Open the app
              <ArrowRight />
            </a>
            <a href="#desktop" className="btn btn-secondary btn-lg">
              <DownloadIcon />
              Get the OAIY Desktop
            </a>
          </div>

          <p
            className="lp-reveal mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs"
            style={{ animationDelay: '340ms', color: 'rgb(var(--color-text-tertiary))' }}
          >
            <span className="inline-flex items-center gap-1.5"><CheckIcon /> Runs in your browser</span>
            <span className="inline-flex items-center gap-1.5"><CheckIcon /> No sign-up to start</span>
            <span className="inline-flex items-center gap-1.5"><CheckIcon /> Your keys never leave your device</span>
          </p>
        </div>

        {/* Graph column */}
        <div className="lp-reveal" style={{ animationDelay: '240ms' }}>
          <HeroGraph />
        </div>
      </div>
    </section>
  );
}

/**
 * HeroGraph — the "image review loop", drawn with the app's REAL nodes.
 *
 * Four nodes, because four is the whole story and five made every card too
 * small to read: generate an image, have the vision model review it, branch on
 * the verdict. True goes to Output; False feeds straight back into image_gen's
 * Prompt handle, so the graph literally regenerates. That feedback edge is the
 * point of the picture, so it's the only labelled wire besides True.
 *
 * Every id, label and port below comes from the actual bundled-module
 * definitions, so what the landing page promises is what the palette gives you:
 *
 *   image_gen   Image Gen   Prompt: string      ⇒  Image: image
 *   ai_llm      AI LLM      Prompt/Image/Video  ⇒  Response: string
 *   condition   Condition   Value: any          ⇒  True / False
 *   output      Output      Result: any
 *
 * Accents follow each module's own colour (pink / purple / cyan / emerald)
 * expressed in the design system's signal palette, so a node here is the colour
 * it will be on the canvas.
 */
function HeroGraph() {
  const W = 136;
  const H = 62;
  const Y = 28;
  const cy = Y + H / 2; // 59 — the row's handle height

  const nodes = [
    { x: 60, id: 'image_gen', label: 'Image Gen', port: 'Image · image', icon: 'image', accent: 'var(--signal-magenta)', delay: '0.10s' },
    { x: 228, id: 'ai_llm', label: 'AI LLM', port: 'Response · string', icon: 'brain', accent: 'var(--accent-primary)', delay: '0.28s' },
    { x: 396, id: 'condition', label: 'Condition', port: 'True / False', icon: 'branch', accent: 'var(--signal-cyan)', delay: '0.46s' },
    { x: 564, id: 'output', label: 'Output', port: 'Result · any', icon: 'check', accent: 'var(--signal-green)', delay: '0.64s' },
  ] as const;

  // Plain forward wires — the port names are already printed on the cards, so
  // labelling these too was just noise at this size.
  const wires = [
    { d: `M202 ${cy} L219 ${cy}`, delay: '0.20s' },
    { d: `M370 ${cy} L387 ${cy}`, delay: '0.38s' },
  ];

  // Condition's two outputs sit at different heights on its right edge, exactly
  // as a multi-output node renders on the canvas.
  const trueY = cy - 11;
  const falseY = cy + 11;

  return (
    <div
      className="relative rounded-2xl p-3 sm:p-4"
      style={{
        backgroundColor: 'rgb(var(--color-bg-elevated) / 0.7)',
        border: '1px solid rgb(var(--color-border-primary))',
        boxShadow: 'var(--shadow-lg)',
        backdropFilter: 'blur(4px)',
      }}
    >
      {/* fake window chrome */}
      <div className="mb-3 flex items-center gap-1.5 px-1">
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(248 113 113)' }} />
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(250 204 21)' }} />
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(74 222 128)' }} />
        <span className="ml-2 font-mono text-[10px]" style={{ color: 'rgb(var(--color-text-muted))' }}>
          image-review-loop.oaiy
        </span>
      </div>

      <div
        className="rounded-xl bg-dotgrid"
        style={{
          backgroundColor: 'rgb(var(--color-bg-canvas))',
          border: '1px solid rgb(var(--color-border-secondary))',
        }}
      >
        <svg
          viewBox="0 0 740 210"
          className="w-full"
          role="img"
          aria-label="An example OAIY flow: an Image Gen node feeds an AI LLM node that reviews the image, and a Condition node either sends the result to Output or loops back to regenerate the image."
        >
          <defs>
            <marker id="lp-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0 1l5 3-5 3z" fill="rgb(var(--color-text-secondary))" />
            </marker>
            <marker id="lp-arrow-ok" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0 1l5 3-5 3z" fill="rgb(var(--signal-green))" />
            </marker>
            <marker id="lp-arrow-loop" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0 1l5 3-5 3z" fill="rgb(var(--signal-cyan))" />
            </marker>
          </defs>

          {wires.map((w) => (
            <path
              key={w.d}
              className="lp-reveal"
              style={{ animationDelay: w.delay }}
              d={w.d}
              fill="none"
              stroke="rgb(var(--color-text-secondary) / 0.8)"
              strokeWidth="2"
              markerEnd="url(#lp-arrow)"
            />
          ))}

          {/* Condition · True → Output */}
          <g className="lp-reveal" style={{ animationDelay: '0.56s' }}>
            <path
              d={`M538 ${trueY} C552 ${trueY} 542 ${cy} 555 ${cy}`}
              fill="none"
              stroke="rgb(var(--signal-green))"
              strokeWidth="2"
              markerEnd="url(#lp-arrow-ok)"
            />
            <text x="548" y={trueY - 8} textAnchor="middle" fontSize="11" fill="rgb(var(--signal-green))" style={{ fontFamily: 'var(--font-mono)' }}>
              True
            </text>
          </g>

          {/* Condition · False → back into image_gen's Prompt handle. Dashed and
              marching so "it regenerates" reads without a caption. The left
              vertical sits at x=28, clear of the first card at x=60. */}
          <g className="lp-reveal" style={{ animationDelay: '0.72s' }}>
            <path
              d={`M538 ${falseY} H548 Q564 ${falseY} 564 ${falseY + 16} V152 Q564 168 548 168 H44 Q28 168 28 152 V${cy + 16} Q28 ${cy} 44 ${cy} H51`}
              fill="none"
              stroke="rgb(var(--signal-cyan))"
              strokeWidth="2"
              strokeDasharray="7 6"
              markerEnd="url(#lp-arrow-loop)"
              className="lp-edge-flow"
            />
            <text x="300" y="184" textAnchor="middle" fontSize="12" fill="rgb(var(--signal-cyan))" style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>
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
  x,
  y,
  w,
  h,
  accent,
  label,
  id,
  port,
  delay,
  icon,
  outputs,
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
  delay: string;
  icon: 'image' | 'brain' | 'branch' | 'check';
  /** Condition has two right-edge handles; everything else has one. */
  outputs: 1 | 2;
}) {
  // Two nested groups so the opacity reveal and the bob transform don't
  // collide on one element's `animation` shorthand (which would override the
  // reveal and leave the node stuck at opacity 0). Outer = fade-in, inner =
  // perpetual bob.
  const hasOutput = id !== 'output';
  return (
    <g className="lp-reveal" style={{ animationDelay: delay }}>
      <g
        className="lp-node-bob"
        style={{ animationDelay: `${parseFloat(delay) + 1}s`, transformBox: 'fill-box', transformOrigin: 'center' } as React.CSSProperties}
      >
        <rect
          className="lp-graph-card"
          x={x}
          y={y}
          width={w}
          height={h}
          rx={13}
          fill="rgb(var(--color-bg-tertiary))"
          stroke="rgb(var(--color-border-strong))"
          strokeWidth="1.2"
        />
        {/* icon chip */}
        <rect x={x + 11} y={y + 11} width={26} height={26} rx={8} fill={`rgb(${accent} / 0.16)`} />
        <g
          transform={`translate(${x + 17.5} ${y + 17.5})`}
          stroke={`rgb(${accent})`}
          strokeWidth="1.4"
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <NodeGlyph icon={icon} />
        </g>
        {/* the node id, in the overline slot the real card uses */}
        <text x={x + 45} y={y + 20} fontSize="8" fill={`rgb(${accent})`} style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.1em' }}>
          {id.toUpperCase()}
        </text>
        <text x={x + 45} y={y + 34} fontSize="15" fontWeight="620" fill="rgb(var(--color-text-primary))" style={{ fontFamily: 'var(--font-body)', letterSpacing: '-0.01em' }}>
          {label}
        </text>
        <text x={x + 12} y={y + h - 10} fontSize="9.5" fill="rgb(var(--color-text-tertiary))" style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.02em' }}>
          {port}
        </text>

        {/* typed handles — left in, right out, matching the canvas */}
        <circle cx={x} cy={y + h / 2} r={5} fill="rgb(var(--color-bg-canvas))" stroke={`rgb(${accent})`} strokeWidth="2.4" />
        {hasOutput && outputs === 1 && (
          <circle cx={x + w} cy={y + h / 2} r={5} fill="rgb(var(--color-bg-canvas))" stroke={`rgb(${accent})`} strokeWidth="2.4" />
        )}
        {hasOutput && outputs === 2 && (
          <>
            <circle cx={x + w} cy={y + h / 2 - 11} r={5} fill="rgb(var(--color-bg-canvas))" stroke="rgb(var(--signal-green))" strokeWidth="2.4" />
            <circle cx={x + w} cy={y + h / 2 + 11} r={5} fill="rgb(var(--color-bg-canvas))" stroke="rgb(var(--signal-cyan))" strokeWidth="2.4" />
          </>
        )}
      </g>
    </g>
  );
}

/** Glyphs mirroring each module's declared lucide icon, on a 13x13 box. */
function NodeGlyph({ icon }: { icon: 'image' | 'brain' | 'branch' | 'check' }) {
  switch (icon) {
    case 'image': // image_gen → "image"
      return (
        <>
          <rect x="0" y="0.5" width="13" height="12" rx="2.5" />
          <circle cx="3.8" cy="4.2" r="1.3" />
          <path d="M0.5 10l3.8-3.3 3.3 2.7 2.1-1.6 2.8 2.7" />
        </>
      );
    case 'brain': // ai_llm → "brain"
      return (
        <>
          <path d="M6.5 1.6a2.3 2.3 0 00-4.2 1.4 2.2 2.2 0 00-.7 3.5 2.2 2.2 0 001.9 3.4 2.3 2.3 0 003 0.7z" />
          <path d="M6.5 1.6a2.3 2.3 0 014.2 1.4 2.2 2.2 0 01.7 3.5 2.2 2.2 0 01-1.9 3.4 2.3 2.3 0 01-3 0.7z" />
        </>
      );
    case 'branch': // condition → "git-branch"
      return (
        <>
          <path d="M2.4 2.6v7.8" />
          <circle cx="2.4" cy="1.2" r="1.3" />
          <circle cx="2.4" cy="11.8" r="1.3" />
          <circle cx="10.6" cy="3.5" r="1.3" />
          <path d="M10.6 4.8v1.1a3 3 0 01-3 3H2.4" />
        </>
      );
    case 'check': // output → "check-circle"
      return (
        <>
          <circle cx="6.5" cy="6.5" r="6" />
          <path d="M3.8 6.7l2.1 2.1 4-4.2" />
        </>
      );
  }
}



/* ------------------------------------------------------------------ */
/* Engines strip                                                       */
/* ------------------------------------------------------------------ */

const ENGINES = [
  'Ollama', 'LM Studio', 'llama.cpp', 'vLLM', 'ComfyUI', 'OpenAI',
  'Anthropic', 'OpenAI-compatible', 'Stable Diffusion', 'Any HTTP endpoint',
];

function EnginesStrip() {
  return (
    <section className="border-y py-6" style={{ borderColor: 'rgb(var(--color-border-secondary))', backgroundColor: 'rgb(var(--color-bg-secondary) / 0.4)' }} aria-label="Supported AI engines">
      <p className="mb-4 text-center font-mono text-[10px] uppercase tracking-[0.28em]" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
        Talks to the engines you already run
      </p>
      {/* One accessible copy for screen readers; the animated track below is
          decorative + duplicated, so it's aria-hidden. */}
      <ul className="sr-only">
        {ENGINES.map((name) => (<li key={name}>{name}</li>))}
      </ul>
      <div className="lp-marquee-mask overflow-hidden">
        <div className="lp-marquee-track flex w-max items-center gap-3" aria-hidden="true">
          {[...ENGINES, ...ENGINES].map((name, i) => (
            <span
              key={i}
              className="whitespace-nowrap rounded-full px-4 py-1.5 text-sm font-medium"
              style={{
                backgroundColor: 'rgb(var(--color-bg-elevated))',
                border: '1px solid rgb(var(--color-border-primary))',
                color: 'rgb(var(--color-text-secondary))',
              }}
            >
              {name}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Value props                                                         */
/* ------------------------------------------------------------------ */

function ValueProps() {
  const items = [
    {
      icon: <CanvasIcon />,
      title: 'A canvas, not a config file',
      body: 'Drag nodes, wire handles, watch data flow through. Typed connections, loops, conditions, macros and subflows — the logic is the diagram.',
    },
    {
      icon: <LockIcon />,
      title: 'Local-first & private',
      body: 'Your flow lives in your browser; inference happens on your machine against your engines. API keys are stored locally and never sent to oaiy.com.',
    },
    {
      icon: <PlugIcon />,
      title: 'Drive it from any AI',
      body: 'Share a flow as a URL and let ChatGPT, Claude, or any HTTP client trigger runs that execute on your hardware — with optional end-to-end encryption.',
    },
    {
      icon: <BoxIcon />,
      title: 'Reusable & shareable',
      body: 'Capture any subgraph as a reusable macro, nest flows as subflows, and share whole flows as portable JSON or a private link — grow your toolkit without copy-paste.',
    },
  ];
  return (
    <Section>
      <SectionHeading
        eyebrow="Why OAIY"
        title="The power of code. The clarity of a diagram."
        sub="Everything you'd script by hand — prompts, branching, retries, media pipelines — laid out as a graph you can actually read."
      />
      <div className="grid gap-4 sm:grid-cols-2">
        {items.map((it, i) => (
          <Reveal key={it.title} delay={i * 80}>
            <article className="card h-full" style={{ borderColor: 'rgb(var(--color-border-primary))' }}>
              <div
                className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg"
                style={{ backgroundColor: 'rgb(var(--accent-primary) / 0.12)', color: 'rgb(var(--accent-primary))' }}
              >
                {it.icon}
              </div>
              <h3 className="mb-1.5 text-base font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{it.title}</h3>
              <p className="text-sm" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.6 }}>{it.body}</p>
            </article>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Capabilities                                                        */
/* ------------------------------------------------------------------ */

function Capabilities() {
  const cats = [
    { c: '99 102 241', icon: <SparkIcon />, title: 'AI & LLMs', body: 'Chat, completion and tool-style calls against local or cloud models. Stream responses, template prompts, fan out across providers.' },
    { c: '6 182 212', icon: <GlobeIcon />, title: 'Browser automation', body: 'Fetch pages, run sessions and actions, and extract text, links, emails, phones, images, logos, JSON-LD and metadata.' },
    { c: '168 85 247', icon: <ImageIcon />, title: 'Image generation', body: 'Generate, resize, view and save images — including full ComfyUI workflow graphs analysed and driven from a node.' },
    { c: '236 72 153', icon: <WaveIcon />, title: 'Audio & speech', body: 'Text-to-speech, speech-to-text, music generation, and audio append/fade — wire sound straight into your flow.' },
    { c: '249 115 22', icon: <FilmIcon />, title: 'Video', body: 'Assemble and process video in-browser with ffmpeg.wasm — no server round-trips, no uploads.' },
    { c: '34 197 94', icon: <BranchIcon />, title: 'Logic & control flow', body: 'Conditions, loops, macros, subflows and outputs — compose reusable building blocks and branch on real data.' },
    { c: '245 158 11', icon: <DbIcon />, title: 'Files & data', body: 'Read and write files, chunk text, and query a built-in SQLite database with a visual Data Viewer.' },
    { c: '20 184 166', icon: <PlugIcon />, title: 'HTTP & services', body: 'Register any endpoint as a Service and call it from the graph — OpenAI, Ollama, Anthropic or your own API.' },
  ];
  return (
    <Section id="capabilities">
      <SectionHeading
        eyebrow="What you can build"
        title="One palette, the whole pipeline"
        sub="LLMs are just the start. OAIY ships a deep catalogue of nodes so a single flow can think, browse, generate and decide."
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cats.map((cat, i) => (
          <Reveal key={cat.title} delay={(i % 4) * 70}>
            <article
              className="card card-interactive h-full"
              style={{ borderColor: 'rgb(var(--color-border-primary))' }}
            >
              <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg" style={{ backgroundColor: `rgb(${cat.c} / 0.14)`, color: `rgb(${cat.c})` }}>
                {cat.icon}
              </div>
              <h3 className="mb-1.5 text-[15px] font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{cat.title}</h3>
              <p className="text-[13px]" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.55 }}>{cat.body}</p>
            </article>
          </Reveal>
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
    { n: '01', title: 'Open the canvas', body: 'Launch the app in your browser — nothing to install. The first-run wizard wires a starter chat flow in three clicks.' },
    { n: '02', title: 'Connect your engines', body: 'Point a Service at Ollama, LM Studio, ComfyUI, OpenAI or any HTTP endpoint. Test the connection inline; keys stay on your device.' },
    { n: '03', title: 'Run, share & automate', body: 'Hit Run and watch each node light up. Share the flow as a link, or let an external AI trigger runs on your hardware over HTTP.' },
  ];
  return (
    <Section id="how">
      <SectionHeading eyebrow="How it works" title="From blank canvas to running flow" />
      <div className="grid gap-5 md:grid-cols-3">
        {steps.map((s, i) => (
          <Reveal key={s.n} delay={i * 90}>
            <div className="relative h-full rounded-xl p-6" style={{ backgroundColor: 'rgb(var(--color-bg-elevated))', border: '1px solid rgb(var(--color-border-primary))' }}>
              <span className="font-display text-4xl" style={{ color: 'rgb(var(--accent-primary) / 0.75)', fontWeight: 500 }}>{s.n}</span>
              <h3 className="mb-1.5 mt-3 text-lg font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{s.title}</h3>
              <p className="text-sm" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.6 }}>{s.body}</p>
              {i < steps.length - 1 && (
                <div className="absolute -right-3 top-1/2 hidden -translate-y-1/2 md:block" style={{ color: 'rgb(var(--color-text-muted))' }} aria-hidden="true">
                  <ArrowRight />
                </div>
              )}
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Privacy / local-first band                                          */
/* ------------------------------------------------------------------ */

function Privacy() {
  const points = [
    { title: 'Your flow stays in your browser', body: 'It lives in localStorage and your exported JSON — never on a server unless you choose to share it.' },
    { title: 'Inference runs on your machine', body: 'OAIY orchestrates; your engines compute. Nothing is proxied through us.' },
    { title: 'Keys never reach oaiy.com', body: 'Secrets resolve against your local constants at execution time. The backend can\'t see them.' },
    { title: 'Shares can be end-to-end encrypted', body: 'Set a password and flows are sealed with AES-GCM + PBKDF2 before they ever leave the browser.' },
  ];
  return (
    <Section>
      <div
        className="relative overflow-hidden rounded-2xl px-6 py-10 sm:px-10 sm:py-14"
        style={{
          backgroundColor: 'rgb(var(--color-bg-elevated))',
          border: '1px solid rgb(var(--color-border-primary))',
          boxShadow: 'var(--shadow-md)',
        }}
      >
        <div
          className="pointer-events-none absolute -right-20 -top-20 h-72 w-72 rounded-full blur-3xl"
          style={{ background: 'radial-gradient(closest-side, rgb(var(--accent-secondary) / 0.18), transparent)' }}
          aria-hidden="true"
        />
        <div className="relative grid gap-10 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
          <div>
            <span className="badge badge-green mb-4">Private by default</span>
            <h2 className="font-display text-3xl sm:text-4xl" style={{ fontWeight: 400, letterSpacing: '-0.02em', color: 'rgb(var(--color-text-primary))' }}>
              Built to keep your data yours.
            </h2>
            <p className="mt-4 max-w-md text-sm sm:text-base" style={{ color: 'rgb(var(--color-text-secondary))', lineHeight: 1.6 }}>
              OAIY is local-first on purpose. The optional backend is a rendezvous
              point for sharing — never a place your prompts, keys or results pass
              through unencrypted.
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {points.map((p, i) => (
              <Reveal key={p.title} delay={i * 70}>
                <div className="flex gap-3">
                  <span className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full" style={{ backgroundColor: 'rgb(34 197 94 / 0.16)', color: 'rgb(22 163 74)' }}>
                    <CheckIcon />
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{p.title}</h3>
                    <p className="mt-0.5 text-[13px]" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.5 }}>{p.body}</p>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* OAIY Desktop                                                   */
/* ------------------------------------------------------------------ */

function Desktop() {
  const features = [
    { icon: <ServerIcon />, title: 'Manages local services', body: 'Install, start, stop and tail logs for Ollama, llama.cpp and custom Python rigs — from a tidy dashboard.' },
    { icon: <DownloadIcon />, title: 'Downloads models', body: 'Pull GGUFs and safetensors straight from Hugging Face with pause / resume and a curated quick-add catalogue.' },
    { icon: <PythonIcon />, title: 'Bundles Python', body: 'A portable Python runtime with reusable virtual envs, so two services can share one heavy install.' },
    { icon: <GlobeIcon />, title: 'Powers browser nodes', body: 'Runs the headless browser sidecar that the web app\'s browser-automation nodes drive.' },
  ];
  return (
    <Section id="desktop">
      <SectionHeading
        eyebrow="The OAIY Desktop"
        title="Optional muscle for the heavy local work"
        sub="The web app does everything a browser can. For the rest — spawning model servers, downloading weights, running Python — there’s a small tray companion."
      />

      <div className="grid gap-6 lg:grid-cols-[1fr_0.85fr] lg:items-center">
        {/* Two-process diagram */}
        <Reveal>
          <div className="rounded-2xl p-6 sm:p-8" style={{ backgroundColor: 'rgb(var(--color-bg-elevated))', border: '1px solid rgb(var(--color-border-primary))', boxShadow: 'var(--shadow-md)' }}>
            <div className="flex flex-col items-stretch gap-3">
              <ProcessCard
                tone="99 102 241"
                badge="In your browser"
                title="OAIY web app"
                body="Flow building, palette UX, ffmpeg.wasm media, HTTP service calls."
              />
              <div className="flex items-center justify-center gap-2 py-1" style={{ color: 'rgb(var(--color-text-muted))' }}>
                <span className="font-mono text-[10px] uppercase tracking-[0.18em]">localhost · 127.0.0.1:17972</span>
              </div>
              <div className="flex justify-center" aria-hidden="true">
                <LinkVertical />
              </div>
              <ProcessCard
                tone="124 58 237"
                badge="In your system tray"
                title="OAIY OAIY Desktop"
                body="Model servers, Hugging Face downloads, portable Python venvs, browser sidecar."
              />
            </div>
            <p className="mt-5 text-center text-xs" style={{ color: 'rgb(var(--color-text-muted))' }}>
              The web app probes <code className="font-mono">/api/health</code> on load. Found it?
              The palette lights up with OAIY-Desktop-managed services automatically.
            </p>
          </div>
        </Reveal>

        {/* Feature list */}
        <div className="grid gap-3">
          {features.map((f, i) => (
            <Reveal key={f.title} delay={i * 70}>
              <div className="flex gap-4 rounded-xl p-4" style={{ backgroundColor: 'rgb(var(--color-bg-secondary) / 0.5)', border: '1px solid rgb(var(--color-border-secondary))' }}>
                <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg" style={{ backgroundColor: 'rgb(var(--accent-secondary) / 0.12)', color: 'rgb(var(--accent-secondary))' }}>
                  {f.icon}
                </span>
                <div>
                  <h3 className="text-sm font-semibold" style={{ color: 'rgb(var(--color-text-primary))' }}>{f.title}</h3>
                  <p className="mt-0.5 text-[13px]" style={{ color: 'rgb(var(--color-text-tertiary))', lineHeight: 1.5 }}>{f.body}</p>
                </div>
              </div>
            </Reveal>
          ))}
          <div className="mt-1 flex flex-wrap items-center gap-3">
            <a href="desktop.html" className="btn btn-primary btn-md">
              Learn about OAIY Desktop
              <ArrowRight />
            </a>
            <span className="inline-flex items-center gap-1.5 text-xs" style={{ color: 'rgb(var(--color-text-muted))' }}>
              <WindowsIcon /> OAIY Desktop for Windows · optional
            </span>
          </div>
        </div>
      </div>
    </Section>
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

/* ------------------------------------------------------------------ */
/* Remote-AI band                                                      */
/* ------------------------------------------------------------------ */

function RemoteAI() {
  return (
    <Section>
      <div className="grid gap-8 lg:grid-cols-[1fr_1fr] lg:items-center">
        <div>
          <span className="badge badge-blue mb-4">Remote control</span>
          <h2 className="font-display text-3xl sm:text-4xl" style={{ fontWeight: 400, letterSpacing: '-0.02em', color: 'rgb(var(--color-text-primary))' }}>
            Let an AI run your flows.
          </h2>
          <p className="mt-4 max-w-md text-sm sm:text-base" style={{ color: 'rgb(var(--color-text-secondary))', lineHeight: 1.6 }}>
            Share any flow and you get two URLs — a read-only view link and an edit
            link that can queue runs. Hand the edit link to ChatGPT or Claude with a
            one-line prompt and it can read the flow's manifest, fill the inputs, and
            trigger a run that executes on <em>your</em> machine. Your browser picks it
            up, runs it, and posts the result back.
          </p>
          <div className="mt-6 flex flex-wrap gap-x-6 gap-y-2 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
            <span className="inline-flex items-center gap-2"><CheckIcon /> 110-bit hash links</span>
            <span className="inline-flex items-center gap-2"><CheckIcon /> View vs. edit access</span>
            <span className="inline-flex items-center gap-2"><CheckIcon /> Optional encryption</span>
          </div>
        </div>

        {/* Terminal-style snippet */}
        <Reveal>
          <div className="overflow-hidden rounded-xl" style={{ border: '1px solid rgb(var(--color-border-primary))', backgroundColor: 'rgb(var(--color-bg-canvas))', boxShadow: 'var(--shadow-md)' }}>
            <div className="flex items-center gap-1.5 px-4 py-2.5" style={{ borderBottom: '1px solid rgb(var(--color-border-secondary))', backgroundColor: 'rgb(var(--color-bg-tertiary) / 0.5)' }}>
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(248 113 113)' }} />
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(250 204 21)' }} />
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(74 222 128)' }} />
              <span className="ml-2 font-mono text-[10px]" style={{ color: 'rgb(var(--color-text-muted))' }}>drive-from-claude.sh</span>
            </div>
            <pre className="overflow-x-auto p-4 font-mono text-[12px] leading-relaxed" style={{ color: 'rgb(var(--color-text-secondary))' }}>
<span style={{ color: 'rgb(var(--color-text-muted))' }}># 1 · read the flow's manifest</span>{'\n'}
<span style={{ color: 'rgb(var(--accent-primary))' }}>GET</span>  /api/flows/<span style={{ color: 'rgb(34 197 94)' }}>{'{hash_edit}'}</span>/manifest{'\n\n'}
<span style={{ color: 'rgb(var(--color-text-muted))' }}># 2 · queue a run with your inputs</span>{'\n'}
<span style={{ color: 'rgb(var(--accent-primary))' }}>POST</span> /api/flows/<span style={{ color: 'rgb(34 197 94)' }}>{'{hash_edit}'}</span>/runs{'\n'}
{'     '}{'{ "inputs": { "prompt": "summarise this" } }'}{'\n\n'}
<span style={{ color: 'rgb(var(--color-text-muted))' }}># 3 · poll until it’s done</span>{'\n'}
<span style={{ color: 'rgb(var(--accent-primary))' }}>GET</span>  /api/flows/<span style={{ color: 'rgb(34 197 94)' }}>{'{hash_edit}'}</span>/runs/<span style={{ color: 'rgb(34 197 94)' }}>{'{run_id}'}</span>  <span style={{ color: 'rgb(var(--color-text-muted))' }}>→ done</span>
            </pre>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Final CTA + footer                                                  */
/* ------------------------------------------------------------------ */

function FinalCta() {
  return (
    <Section>
      <div
        className="relative overflow-hidden rounded-3xl px-6 py-16 text-center sm:px-12 sm:py-20 bg-dotgrid"
        style={{ backgroundColor: 'rgb(var(--color-bg-elevated))', border: '1px solid rgb(var(--color-border-primary))' }}
      >
        <div
          className="lp-halo pointer-events-none absolute left-1/2 top-1/2 h-[520px] w-[520px] rounded-full blur-3xl"
          style={{ background: 'radial-gradient(closest-side, rgb(var(--accent-primary) / 0.4), rgb(var(--accent-secondary) / 0.2), transparent 70%)' }}
          aria-hidden="true"
        />
        <div className="relative">
          <h2 className="font-display" style={{ fontWeight: 400, letterSpacing: '-0.025em', lineHeight: 1.08, fontSize: 'clamp(2rem, 4.5vw, 3.2rem)', color: 'rgb(var(--color-text-primary))' }}>
            Wire up your first flow.
          </h2>
          <p className="mx-auto mt-4 max-w-md text-sm sm:text-base" style={{ color: 'rgb(var(--color-text-secondary))' }}>
            No account, no install, no cloud bill. Open the canvas and build something
            in the next five minutes.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <a href={APP_URL} className="btn btn-primary btn-lg">
              Open the app
              <ArrowRight />
            </a>
            <a href="#desktop" className="btn btn-secondary btn-lg">Learn about the desktop app</a>
          </div>
          <p className="mt-7 text-sm" style={{ color: 'rgb(var(--color-text-secondary))' }}>
            OAIY is open source under Apache-2.0.{' '}
            <a
              href={REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="lp-star-inline"
            >
              Star us on GitHub
            </a>{' '}
            if it's useful to you.
          </p>
        </div>
      </div>
    </Section>
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
          <a href="#capabilities">Capabilities</a>
          <a href="#desktop">Desktop</a>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">GitHub</a>
        </nav>
        <p className="font-mono text-[11px]" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
          © 2026 oaiy.com · Apache-2.0
        </p>
      </div>
    </footer>
  );
}

/* ------------------------------------------------------------------ */
/* Layout primitives                                                   */
/* ------------------------------------------------------------------ */

function Section({ id, children }: { id?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mx-auto max-w-6xl px-5 py-16 sm:px-8 sm:py-24 scroll-mt-20">
      {children}
    </section>
  );
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

/** Lightweight entrance wrapper — CSS-only, no scroll observers. */
function Reveal({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  return (
    <div className="lp-reveal h-full" style={{ animationDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Icons (stroke, currentColor — matches the app's icon language)      */
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
function DownloadIcon() {
  return (
    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" />
    </svg>
  );
}
function CanvasIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="4" width="6" height="6" rx="1.5" strokeWidth="2" />
      <rect x="15" y="14" width="6" height="6" rx="1.5" strokeWidth="2" />
      <path strokeLinecap="round" strokeWidth="2" d="M9 7h4a2 2 0 012 2v6" />
    </svg>
  );
}
function LockIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="4" y="11" width="16" height="9" rx="2" strokeWidth="2" />
      <path strokeLinecap="round" strokeWidth="2" d="M8 11V8a4 4 0 018 0v3" />
    </svg>
  );
}
function PlugIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 7V3m6 4V3M7 7h10v4a5 5 0 01-10 0V7zm5 9v5" />
    </svg>
  );
}
function BoxIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
    </svg>
  );
}
function SparkIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3l1.8 5L19 9.8 14 12l-2 5-2-5-5-2.2L10 8z" />
    </svg>
  );
}
function GlobeIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="9" strokeWidth="2" />
      <path strokeWidth="2" d="M3 12h18M12 3c2.5 2.5 2.5 15 0 18M12 3c-2.5 2.5-2.5 15 0 18" />
    </svg>
  );
}
function ImageIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="4" width="18" height="16" rx="2" strokeWidth="2" />
      <circle cx="8.5" cy="9.5" r="1.5" strokeWidth="2" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 16l-5-5-7 7" />
    </svg>
  );
}
function WaveIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeWidth="2" d="M3 12h2l2-6 3 14 3-18 3 14 2-4h3" />
    </svg>
  );
}
function FilmIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="4" width="18" height="16" rx="2" strokeWidth="2" />
      <path strokeWidth="2" d="M7 4v16M17 4v16M3 9h4m10 0h4M3 15h4m10 0h4" />
    </svg>
  );
}
function BranchIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="6" cy="6" r="2.5" strokeWidth="2" />
      <circle cx="6" cy="18" r="2.5" strokeWidth="2" />
      <circle cx="18" cy="12" r="2.5" strokeWidth="2" />
      <path strokeLinecap="round" strokeWidth="2" d="M6 8.5v7M8.5 6.5h4a3 3 0 013 3v.5M8.5 17.5h4a3 3 0 003-3V14" />
    </svg>
  );
}
function DbIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <ellipse cx="12" cy="6" rx="8" ry="3" strokeWidth="2" />
      <path strokeWidth="2" d="M4 6v12c0 1.66 3.58 3 8 3s8-1.34 8-3V6M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3" />
    </svg>
  );
}
function ServerIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="4" width="18" height="7" rx="2" strokeWidth="2" />
      <rect x="3" y="13" width="18" height="7" rx="2" strokeWidth="2" />
      <path strokeLinecap="round" strokeWidth="2" d="M7 7.5h.01M7 16.5h.01" />
    </svg>
  );
}
function PythonIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3c-3 0-4 1.5-4 3v2h6v1H6c-2 0-3 1.5-3 4s1 4 3 4h2v-3c0-1.5 1-3 3-3h4c1.5 0 3-1 3-3V6c0-1.5-1-3-4-3z" />
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
function LinkVertical() {
  return (
    <svg width="24" height="40" viewBox="0 0 24 40" fill="none" aria-hidden="true">
      <path d="M12 0v40" stroke="rgb(var(--accent-primary) / 0.5)" strokeWidth="2" strokeDasharray="4 4" />
      <circle cx="12" cy="20" r="4" fill="rgb(var(--color-bg-elevated))" stroke="rgb(var(--accent-primary))" strokeWidth="2" />
    </svg>
  );
}
