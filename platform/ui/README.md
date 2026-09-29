# The flow editor and the site (`platform/ui`)

One Vite project, three pages, built to plain static files:

| Page | What it is |
|---|---|
| `/` (`index.html`) | The landing page: what OAIY is, what works in a browser and what needs OAIY Desktop, screenshots |
| `/app.html` | The flow editor. OAIY Desktop shows this same build as its Flows page (told apart by `window.__OAIY_DESKTOP__`) |
| `/desktop.html` | The OAIY Desktop page: what it is, installing it on Windows and Linux, the headless server, the service format and library |

The landing and desktop pages run none of the editor's start-up (`src/landing/main.tsx`): they make no request beyond their own
site (the desktop page's service library asks the site's own `/api`, or the API base the site was built for) and never look for a
desktop. Only the editor does that, once it is open.

```bash
npm install
npm run dev          # http://localhost:5173
npm run build        # dist/, the whole site
npm test             # types, CSS tokens, node contracts, the ZIPP engines, and the suites named in ../TESTING.md
```

## The flow editor as an installable app

`/app.html` registers a service worker (`/sw.js`, scope `/app.html`, `src/pwa/`). The landing and desktop pages never do,
OAIY's own window never does, and neither does the dev server (there is no `/sw.js` there).

- **What it keeps.** The shell, when it installs: `app.html` and the scripts, preloads and stylesheets it names, read from the
  build's own output by `scripts/serviceWorkerPlugin.ts` (a shell file the build did not emit, or one over the limit, fails the build).
  Everything else the page fetches from its own origin is kept as it goes by: HTML network-first, the rest stale-while-revalidate.
- **What it never touches.** Other origins (every engine on `127.0.0.1` or `localhost`, every provider), `/api/`, `?flow=` share
  links, range requests, and its own script. It adds no headers: the editor runs without COOP and COEP on purpose.
- **The size limit.** Nothing over 4 MiB is kept, counted on the bytes the page gets (not `Content-Length`, which a compressing
  host makes small): `MAX_CACHED_BYTES` in `src/pwa/swCore.ts`, read by the worker and the build. The ffmpeg core, the esbuild and
  ZIPP engines and the language workers are over it, so the editor opens offline after one visit but a flow that needs one of them
  needs the network. The code editor (Monaco) is split into two chunks in `vite.config.ts` (`manualChunks`) to stay under it; if a
  shell chunk grows past the limit the build says which one.
- **Updates.** A new build installs beside the old one and waits. The editor says "A new version is ready" with Reload; only then
  does the new worker take over, the page reload and the old build's caches go. The first install says nothing.
- **Install.** "Install app" appears in the sidebar when the browser offers it (never in OAIY's own window or the installed app).
- **Icons.** `scripts/make-icons.py` makes the 192, 512, maskable and apple-touch icons from the desktop's icon
  (`python scripts/make-icons.py`, `--check` to verify).

A static host has to serve `/sw.js` from the site's root, and `/app.html` must be served as itself, not redirected (Cloudflare
Pages redirects `/app.html` to `/app` unless its pretty-URL behaviour is off). The page asks for the worker only when it is at
`/app.html`, so on a host that moves it the editor registers no worker and works as it did, online only. Only the page at exactly
`/app.html` is answered from the shell: another address under the scope (`/app.html/x`) is left to the network, and a redirected or
wrongly typed answer is never kept. `public/_headers` (Netlify, Cloudflare Pages) says `/sw.js` is never cached, and sets the CSP of
the compiler sandbox worker.

## The download buttons

`components/DownloadDesktop.tsx` (on the landing and desktop pages, and in the editor's sidebar and Settings while no desktop
answers) offers the OAIY Desktop file for the visitor's system: the Windows installer, or the Linux AppImage with `.deb`, `.rpm`
and the headless server under "other downloads". A Mac or a phone is told OAIY Desktop is for Windows and Linux. The system comes
from what the browser already says of itself (`lib/downloads.ts`); there is no request and no probe of a desktop.

The links are **made when the site is built**: the files' names carry the release's version, so `VITE_OAIY_RELEASE_TAG` (the
tag the release was pushed as, `github.ref_name`, set by the release workflow's web job) is baked into the build. The release is
published under that tag, `0.1.0` or `v0.1.0`, so the links are
`https://github.com/f2i-com/oaiy.com/releases/download/<tag>/<file>` with the tag as pushed and the version, without the `v`, in
the file's name. Built without a version tag (a local build, a run on a branch, where `github.ref_name` is the branch's name) the
button links to the latest release and says "Download OAIY Desktop". **The site has to be redeployed with each release**, or it
keeps offering the previous one ([`docs/RELEASING.md`](../../docs/RELEASING.md#the-web-site)).

## Pictures

`scripts/make-site-images.py` makes the landing page's four screenshots (`public/images/*.webp`, from `docs/images`, the README's
demo setup) and the social card (`public/og-image.png`, 1200 x 630). Sizes and descriptions are in `src/landing/screenshots.ts`.

## Tests

`npm test` covers all of the above without a browser (`tests/sw-core.mjs`, `sw-build.mjs`, `pwa-controllers.mjs`, `pwa-assets.mjs`,
`downloads.mjs`, `site-assets.mjs`). The browser suites need a build or a server: `tests/e2e.mjs` and `tests/pwa-e2e.mjs`
(see [`../TESTING.md`](../TESTING.md)).

## What the landing page says about privacy

It depends on the build (`src/landing/privacy.ts`). The release is built with no sharing service (`VITE_API_BASE` unset): nothing a
person builds or runs is uploaded. A build with a service has sharing on by default, so the page says a flow reaches the service only
when the person presses Share, that a run someone queues on it passes through the service, and that sharing can be turned off. Keys are
"sealed where the browser supports it": the editor keeps them in plain storage where it cannot seal them.
