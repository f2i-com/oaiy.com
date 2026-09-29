# Releasing OAIY

A release is one tag. GitHub Actions builds everything at the tagged commit, the
verification gate (`ci.yml`) has to pass at that same commit, and only then is one GitHub
Release published. Both workflows are in [`.github/workflows`](../.github/workflows) at the
repository root: `release.yml` starts on a tag, and `ci.yml` is the gate it calls (started
by hand on its own, because automatic CI is paused).

## What a release contains

| File | What it is |
|---|---|
| `oaiy-desktop-<v>-windows-x64-setup.exe`, `oaiy-desktop-<v>-windows-x64.msi` | OAIY Desktop for Windows |
| `oaiy-desktop-<v>-linux-x86_64.AppImage`, `-linux-amd64.deb`, `-linux-x86_64.rpm` | OAIY Desktop for Linux |
| `oaiy-server-<v>-windows-x64.zip`, `oaiy-server-<v>-linux-x86_64.tar.gz` | The headless server: the same local API with no window, for a host the CLI or a web app drives. It has no Agent or flow editor to show. |
| `oaiy-cli-<v>.tar.gz` | The CLI alone, for a product that embeds it |
| `oaiy-web-<v>.zip`, `oaiy-web-<v>.tar.gz` | The flow editor's site (landing page, `/app.html`, `/desktop.html`), for any static host |
| `SHA256SUMS.txt` | The checksum of every file above |
| `release-evidence-web.json`, `-linux.json`, `-windows.json` | For each build: the revision, target, toolchain, digests, and the verification run that passed before anything was published |

An OAIY Desktop installer is complete by itself for what it shows. Besides the dashboard
it carries the CLI that runs flows (`resources/cli`), the Agent (`resources/app`, opening
on `index.html`) and the flow editor (`resources/flows`, opening on `app.html`), which the
desktop serves into its own window. Building an installer stops if either page is missing,
empty or incomplete (`platform/desktop/scripts/stage-pages.mjs`), so there is no installer
that opens on "the page is not built". What OAIY needs at run time (language and other
models, the portable Python and the Node runtime) is downloaded on first use, not shipped.

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
   holds the files above, and every `release-evidence-*.json` says `verified`.
5. **If the push started nothing**, run the workflow by hand on the tag's ref. A run on a
   tag publishes, the same as the push would have:

   ```sh
   gh workflow run release.yml --ref v0.1.0 -f version=0.1.0
   ```

   The `version` input is required but the tag decides the version. GitHub takes manual
   runs only of workflows that are on the default branch, so the workflows have to be
   there. `gh run list --workflow release.yml` shows the run.

To try the build without publishing, run the same workflow on a branch instead:
`gh workflow run release.yml --ref <branch> -f version=0.1.0`. It builds everything and
keeps the files as the run's artifacts, and creates no release.

Do not move, delete or re-create a published tag. Other products take files from the
latest release (FormLogic's CI takes `oaiy-cli-<v>.tar.gz`, `SHA256SUMS.txt` and
`release-evidence-linux.json`) and check that the evidence names the tag's commit; make a
new version instead.

## What is not in a release

- **The engines.** `oaiy-llm-server` (and its WebGPU build), `oaiy-media` and `oaiy-voice`
  need CUDA 12.8 and the Visual Studio 2022 C++ tools to build
  (`tools/qwen-image/build.ps1`) and are built by hand: they are a separate channel, and
  no workflow here builds them. The desktop finds them beside itself, in an `engines`
  folder there, or where `OAIY_ENGINES_DIR` points. An install without them has no models
  of its own on the computer; the Agent can still use ChatGPT or a provider.
- **Aokie**, the phone plugin. It is a separate product: OAIY installs it from a folder or
  an archive, and no release of OAIY contains it.
- **Signed installers, and updates.** The installers are not signed, and OAIY does not
  update itself: a new version is a new download.
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

A release is `N.N.N` (a tag of `N.N.N` or `vN.N.N`, no suffix such as `-beta`), and not
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
