# Releasing OAIY

A release is one tag. GitHub Actions builds everything at the tagged commit, the
verification gate (`ci.yml`) has to pass at that same commit, and only then is one GitHub
Release published. Both workflows are in [`.github/workflows`](../.github/workflows) at the
repository root: `release.yml` starts on a tag, and `ci.yml` is the gate it calls (started
by hand on its own, because automatic CI is paused).

## What a release contains

| File | What it is |
|---|---|
| `oaiy-desktop-<v>-windows-x64-setup.exe`, `oaiy-desktop-<v>-windows-x64.msi` | OAIY Desktop for Windows. The setup.exe is what OAIY updates itself with; the MSI is a manual download. |
| `oaiy-desktop-<v>-linux-x86_64.AppImage`, `-linux-amd64.deb`, and `-linux-x86_64.rpm` when the build made one (the workflow copies it only if it is there, so a release may have none) | OAIY Desktop for Linux. The AppImage can update itself; the deb and rpm are manual downloads. |
| `oaiy-desktop-<v>-windows-x64-setup.exe.sig`, `oaiy-desktop-<v>-linux-x86_64.AppImage.sig` | The signature of each installer an update can install, made with the updater key. |
| `latest.json` | The update feed: the version, the notes, the date, and for `windows-x86_64` and `linux-x86_64` the installer's URL and signature. OAIY Desktop reads it from `releases/latest/download/latest.json`. |
| `oaiy-server-<v>-windows-x64.zip`, `oaiy-server-<v>-linux-x86_64.tar.gz` | The headless server: the same local API with no window, for a host the CLI or a web app drives. It has no Agent or flow editor to show. |
| `oaiy-cli-<v>.tar.gz` | The CLI alone, for a product that embeds it |
| `oaiy-web-<v>.zip`, `oaiy-web-<v>.tar.gz` | The flow editor's site (landing page, `/app.html`, `/desktop.html`), for any static host, built for this release: its download buttons name this release's files (see [The web site](#the-web-site)) |
| `SHA256SUMS.txt` | The checksum of every file above, `latest.json` and the `.sig` files included |
| `release-evidence-web.json`, `-linux.json`, `-windows.json` | For each build: the revision, target, toolchain, digests, and the verification run that passed before anything was published |

An OAIY Desktop installer is complete by itself for what it shows. Besides the dashboard
it carries the CLI that runs flows (`resources/cli`), the Agent (`resources/app`, opening
on `index.html`) and the flow editor (`resources/flows`, opening on `app.html`), which the
desktop serves into its own window. Building an installer stops if either page is missing,
empty or incomplete (`platform/desktop/scripts/stage-pages.mjs`), so there is no installer
that opens on "the page is not built". What OAIY needs at run time (language and other
models, the portable Python and the Node runtime) is downloaded on first use, not shipped.

## The updater key

OAIY Desktop updates itself from these releases, and refuses any installer whose signature
does not match the public key inside it (`plugins.updater.pubkey` in
`platform/desktop/src-tauri/tauri.conf.json`). The release workflow signs the two
installers an update can install with the matching private key, which lives in two GitHub
Actions secrets on this repository (Settings, Secrets and variables, Actions):

| Secret | Value |
|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | the content of the private key file |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | its password |

They are read by the desktop build step and by the `meta` job's check, and by nothing else.
A run on a **tag** without either stops in `meta`, before anything is built, with a message
naming the missing secret: a release that installed copies of OAIY could not verify would be
worse than none. A run on a branch (the trial run below) without the key builds without
signatures and says so.

The key was made once. Its file and its password are kept outside every repository, with a
README that says this again (`C:\Users\<you>\.oaiy-signing\README.txt` on the machine that
made it). Keep an offline copy of both in a password manager, and delete the local ones.
**Losing the private key means no installed OAIY can be updated again**: they trust only the
old public key. The way out is a new key with its public half in `tauri.conf.json`, a new
release, and everyone installing that release by hand from the releases page; see
[UPDATES.md](UPDATES.md#the-key-and-its-custody). Never build a test with this key: make a
throwaway one (`npx tauri signer generate -w <a folder outside the repository>`).

The release job checks each signed installer against the public key in `tauri.conf.json`
before it writes `latest.json` and stops if one does not verify: the Tauri CLI only *warns*
when the secrets hold a different key from the one in the build, and a release signed with the
wrong key would install on nobody. If that step fails, the two secrets do not belong to the
public key in `tauri.conf.json`.

The base `tauri.conf.json` does not turn the signatures on (`bundle.createUpdaterArtifacts`).
The release workflow passes that as an override of its own, so `npm run tauri:build` on a
machine without the key still builds an unsigned installer.

## Cutting a release

1. **Choose the version.** `0.1.0` or later, and newer than the last release (see
   [The version](#the-version)). The tag is the version: it is stamped into
   `tauri.conf.json` and `Cargo.toml` while the installers are built, so no release commit
   is needed for it. The `0.1.0` committed in those files is only what a build from a
   checkout reports.
2. **Tag a commit whose message does not contain `[skip ci]`.** GitHub starts no workflow
   for a push whose commit says so (`[skip ci]`, `[ci skip]`, `[no ci]`, `[skip actions]`
   and `[actions skip]`), and a tag push is a push: the tag on such a commit starts
   nothing. The commit at the head of a push to this repository has carried `[skip ci]`
   while CI is paused, so tag a commit that does not, for instance a release commit made
   for the purpose:

   ```sh
   git commit --allow-empty -m "OAIY 0.1.0"
   ```

   The commit must also hold `.github/workflows/release.yml`, which the commits before the
   workflows moved to the root (and so `v0.0.1` to `v0.0.6`) do not.
3. **Tag it and push the tag.**

   ```sh
   git tag -a v0.1.0 -m "OAIY 0.1.0"
   git push origin v0.1.0
   ```

   The workflow takes `0.1.0` or `v0.1.0` for the same version; the earlier releases were
   tagged with the `v` and annotated, so do the same, and push only one of the two forms.
   Push the commit first if it is not on GitHub yet.
4. **Watch the run** (Actions, "Release", or `gh run watch`). `meta` fixes the version and
   the revision, `verify` runs `ci.yml` at that revision, and the web and desktop builds
   run beside it. `release` publishes only when all of them passed. The GitHub Release then
   holds the files above, and every `release-evidence-*.json` says `verified`. The
   release job writes `latest.json` (`platform/scripts/make-latest-json.mjs`) from the
   installers and their `.sig` files before it writes the checksums, and stops if an
   installer or a signature is missing.

   The build stops when a page is missing, but nothing lists a finished package, so look
   inside one the first time. `7z l oaiy-desktop-<v>-windows-x64-setup.exe` shows
   `resources\app\index.html` and `resources\flows\app.html`, and
   `dpkg -c oaiy-desktop-<v>-linux-amd64.deb | grep -E 'resources/(app/index|flows/app)\.html'`
   should show both in the Debian package. The Windows installers were looked at this way
   when the pages were added; the Linux packages come from the same configuration but had
   not been built then, so a first release is their first look.
5. **If the push started nothing**, run the workflow by hand on the tag's ref. A run on a
   tag publishes, the same as the push would have:

   ```sh
   gh workflow run release.yml --ref v0.1.0 -f version=0.1.0
   ```

   The `version` input is required but the tag decides the version. GitHub takes manual
   runs only of workflows that are on the default branch, so the workflows have to be
   there. `gh run list --workflow release.yml` shows the run.
6. **Redeploy the site** from the release's `oaiy-web-<v>.zip`. Nothing in the workflow puts it
   on a host, and until it is replaced the site's download buttons still offer the previous
   release (see the next section).

To try the build without publishing, run the same workflow on a branch instead:
`gh workflow run release.yml --ref <branch> -f version=0.1.0`. It builds everything and
keeps the files as the run's artifacts, and creates no release.

Do not move, delete or re-create a published tag. **A release that is published as the
latest must always be complete**, because two things read the latest release: every
installed OAIY asks it for `latest.json`, and FormLogic's CI takes files from it. A later
release without `latest.json`, or with one that leaves a platform out, breaks the feed or
tells those desktops there is no update, and one without the CLI files breaks FormLogic.
Do not publish a partial release (a web-only or desktop-only hotfix) as the latest, and do
not make an older release the latest again: cut a complete new version instead.
Other products take files from the latest release (FormLogic's CI takes `oaiy-cli-<v>.tar.gz`, `SHA256SUMS.txt` and
`release-evidence-linux.json`) and check that the evidence names the tag's commit; make a
new version instead.

## The web site

`oaiy-web-<v>.zip` is the whole site (the landing page, the flow editor at `/app.html`, the
desktop page), built by the `web` job with `npm run build` in `platform/ui`. How the site is
built and what is in it is [`platform/ui/README.md`](../platform/ui/README.md). Three things about
a release matter here:

- **The download links are baked in, so the site must be redeployed with each release.** The
  installers' names carry the version, so the landing and desktop pages and the flow editor
  offer `https://github.com/f2i-com/oaiy.com/releases/download/<tag>/oaiy-desktop-<v>-windows-x64-setup.exe`
  (and the AppImage, `.deb` and the headless server under "other downloads"; the site never links the `.rpm`, which a release may not have). The `web`
  job gives the build `VITE_OAIY_RELEASE_TAG`, the tag as it was pushed (`github.ref_name`), and the
  page picks the visitor's system in the browser with no request. Nothing asks GitHub what is
  newest, so a site that is not redeployed keeps offering the release it was built for.
- **Either form of the tag works.** The workflow accepts `0.1.0` or `v0.1.0` and publishes the
  release under the tag as pushed, so the site's addresses use the tag as it is
  (`.../releases/download/0.1.0/...` or `.../download/v0.1.0/...`) and the file names take the
  version without the `v`. A build that has no version tag (a local one, or a run on a branch,
  where the tag is the branch's name) links to the latest release and says only "Download OAIY
  Desktop". The earlier releases were tagged with the `v`; keep to that so there is one form.
- **A host must serve `/sw.js` from the root, and `/app.html` as itself, not redirected.** The
  flow editor installs a service worker (scope `/app.html`) so it can open offline and be
  installed as an app; it keeps the shell of this build under a cache named for the build and
  removes the old build's when the person reloads onto the new one. The page asks for the worker
  only when it is at `/app.html`, so on a host that redirects `/app.html` to `/app` (Cloudflare
  Pages does, unless its pretty URLs are off) the editor registers none and stays online-only.
  A host that caches `/sw.js` hides a new release from returning visitors (`public/_headers` says
  `no-cache` for hosts that read it). The `HOSTING.md` the release's web job writes into the zip
  does not say any of this yet.

## What is not in a release

- **The engines.** The programs that run models: `oaiy-llm-server` and `oaiy-media`, built
  with CUDA 12.8 and the Visual Studio 2022 C++ tools; `oaiy-llm-server-webgpu`, the same
  server without CUDA, for a computer that lacks it; the `oaiy-studio` host and its tray;
  and `oaiy-voice`, the speech server for calls. `tools/qwen-image/build.ps1` builds all of
  them on Windows but `oaiy-voice`, which is a crate of its own (its `cuda` and
  `flash-attn` features put it on the GPU). They are built by hand: they are a separate
  channel, and no workflow here builds them. The desktop finds them beside itself, in an
  `engines` folder there, or where `OAIY_ENGINES_DIR` points. An install without them has
  no models of its own on the computer; the Agent can still use ChatGPT or a provider.
- **Aokie**, the phone plugin. It is a separate product: OAIY installs it from a folder or
  an archive, and no release of OAIY contains it.
- **Windows code signing.** The installers are not signed with a certificate
  (Authenticode), so Windows SmartScreen warns on a download (below). The update signature
  above is another thing: OAIY checks it, Windows does not.
- **Updates of the engines, the plugins and the headless server.** OAIY Desktop updates
  itself, when its owner presses the button. The engines are a separate channel, plugins
  are installed from a folder or an archive, and `oaiy-server` only tells you that a newer
  release exists. See [UPDATES.md](UPDATES.md).
- **macOS.** Only Windows and Linux are built.

## Installing a release

- **Windows SmartScreen will warn.** The installers are unsigned, so Windows shows
  "Windows protected your PC" for a downloaded one. Check the file against
  `SHA256SUMS.txt` (`certutil -hashfile <file> SHA256`), then choose "More info" and "Run
  anyway". The NSIS installer installs for the current user (`%LOCALAPPDATA%\OAIY`) and
  asks for no administrator rights.
- **An install of 0.0.x has to be uninstalled first.** The app's identifier is now
  `com.oaiy.app` (it was `com.oaiy`, and the product was called "OAIY Desktop"), and its
  data lives in `%APPDATA%\com.oaiy.app`, where the old one kept it in
  `%APPDATA%\com.oaiy`. Nothing is carried across: a new install starts empty, beside the
  old one if that is still installed, and both want port 17972. Uninstall "OAIY Desktop"
  first, and keep its data folder if there is anything in it you want to copy over.

## The version

A release is `N.N.N` (a tag of `N.N.N` or `vN.N.N`, no suffix such as `-beta`, and no
leading zero in a number: Cargo refuses `0.09.0`, so `meta` does too), and not
below `0.1.0`. The desktop's version is the one stamped from the tag, and it refuses a
plugin whose manifest asks for a newer desktop. Aokie's manifest asks for 0.1.0, so an
installer stamped `0.0.x` could not run Aokie; the `meta` job stops a tag below 0.1.0, and
a manual run of a lower version, with that reason. `0.0.1` to `0.0.6` were the releases of
the previous OAIY, before the merge.

## Building an installer on your machine

The same build, unsigned, without the workflow (the commands are in
[the desktop's README](../platform/desktop/README.md#production-build)): build the Agent
(`npm run build:desktop` in `app/`), the flow editor and the CLI, then `npm run tauri:build`
in `platform/desktop`. It writes the installers to
`platform/desktop/src-tauri/target/release/bundle/`, and stops if a page is not built. Do not
run one beside another OAIY: they listen on the same ports.
