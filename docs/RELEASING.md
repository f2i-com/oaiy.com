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
installers an update can install (the NSIS setup.exe and the AppImage) with the matching
private key, in ONE job, `sign`, and nowhere else. The key is two secrets:

| Secret | Value |
|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | the content of the private key file |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | its password |

What keeps the key from everything that does not need it:

- **Nothing that builds has it.** The `desktop` job runs the project's npm packages, its Rust
  crates and the builds of the Agent and the flow editor. It is given no secret at all and
  builds the installers UNSIGNED, on a tag and on a branch alike. The `sign` job is the only
  place the two secrets are named, in one step ("Sign the setup.exe and the AppImage"), and
  the job does nothing but `tauri signer sign` on the two installers the builds made: the Tauri
  CLI is installed from the desktop app's lockfile with no install scripts (`npm ci
  --ignore-scripts`: `@tauri-apps/cli` has none, and its Linux binding is an optional package
  the lockfile pins; only esbuild's and fsevents' scripts are skipped, and this job needs
  neither), and nothing of the project is built or run there.
- **It runs on a tag only.** `sign` has `if: needs.meta.outputs.is_tag == 'true'`. A run on a
  branch (the trial run below) never starts it, and says in its log that its installers are
  unsigned. The job also waits for the verification gate and for every desktop build, so
  nothing is signed that a test failed on.
- **It runs in the `release` environment**, whose rules decide who and what may have the
  secrets (next section).
- **The workflow may not write to the repository** by default (`permissions: contents: read`
  at the top): only the job that publishes the release has `contents: write`.

Each installer is signed under the name the Tauri bundler gives it (`OAIY_<v>_x64-setup.exe`,
`OAIY_<v>_amd64.AppImage`), because the desktop holds a signature to its version by the name of
the file it was made for (UPDATES.md, "What the signature was made for"). The signatures are
uploaded as the artifact `signatures` and reach the release next to the installers.

### The `release` environment

Set this up ONCE, before the first tag. **If the environment does not exist when the first tag is
pushed, GitHub creates it empty: no reviewer and no tag rule, and the job would run unprotected.**

What GitHub gives depends on the repository and its plan. This repository is public (checked with
`gh repo view`), where environments, their secrets, their deployment rules and required reviewers
are all available. In a PRIVATE repository they need a paid plan, and required reviewers a higher
one (check GitHub's current plan table): on a plan without them the `sign` job is handed no
secrets and stops, naming them. Do not put the secrets under the repository's Actions secrets to
get round that: any workflow on any branch can read those. Move the repository to a plan that has
the environment, or sign the release by hand (the commands in the next section sign a file the
way the job does, and `platform/scripts/verify-signature.mjs` checks it) until it is.

1. Settings, Environments, New environment: `release`.
2. **Deployment branches and tags**: choose "Selected branches and tags" and add two TAG
   rules, `[0-9]*.[0-9]*.[0-9]*` and `v[0-9]*.[0-9]*.[0-9]*`. A branch, a pull request or any
   other ref then cannot start a job in this environment, whatever a workflow file on it
   says, so it can never be handed the secrets.
3. **Required reviewers**: add the owner and anyone else who may cut a release. The `sign` job
   waits for one of them to approve after the builds and the gate have passed: that click is
   when someone looks at the run before the key is used. Look at what the tagged commit
   changed in `.github/workflows`: the workflow that runs is the one at the tag, and it decides
   what the job does with the key. (Turn on "Prevent self-review" only when there is a second
   reviewer.)
4. **Environment secrets**: add the two secrets HERE, under the environment's own secrets, and
   delete any copy of them from the repository's Actions secrets. A repository secret can be
   read by a workflow on any branch, whatever the environment's rules; an environment secret
   only by a job that has passed them. From a shell that has `<` (Git Bash, say), the values
   are read from their files and never appear on a command line or in the history:

   ```sh
   gh secret set TAURI_SIGNING_PRIVATE_KEY --env release --repo f2i-com/oaiy.com < ~/.oaiy-signing/oaiy-updater.key
   gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD --env release --repo f2i-com/oaiy.com < ~/.oaiy-signing/oaiy-updater.key.password.txt
   ```

   (The sign step drops a line ending from the end of both values, which a value read from a
   file can carry: the Tauri CLI refuses a key that ends in one, "Invalid symbol 10", and a
   password that does, "Wrong password for that key".)
5. **A tag ruleset** (Settings, Rules, Rulesets, New tag ruleset): target the same two tag
   patterns; restrict who may create them to the people who cut releases, and block updates
   and deletions (a published tag is never moved: see below). Whoever can create a matching
   tag can start a run that reaches the environment, and its reviewer is the last check.

None of this can be enforced from the workflow file: a rule that is missing is a weaker setup,
not a failing build. A tag run whose environment lacks the secrets fails in the `sign` job,
after the builds, naming the secret that is missing; try the key once before the first release
(next section) so that is not where it is found.

### Before the first release: try the key

Do this ONCE before the first tag, and again whenever the key or the secrets change. It proves
that the private key you hold is the pair of the public key inside every installer. The Tauri
CLI does not check that (it only warns, and a release signed with the wrong key installs on
nobody), and no installed OAIY can be fixed afterwards. The key is used on a probe file and
nothing of it is printed: the CLI reads the key from its file and the password from the
environment, and the check prints only whether the signature verifies.

From the repository's root, in PowerShell (the key and its password are in
`%USERPROFILE%\.oaiy-signing`, as above):

```powershell
$dir = Join-Path $env:USERPROFILE '.oaiy-signing'
$probe = Join-Path $env:TEMP 'oaiy-signing-probe.txt'
Set-Content -LiteralPath $probe -Value 'probe' -NoNewline
$env:TAURI_SIGNING_PRIVATE_KEY_PATH = Join-Path $dir 'oaiy-updater.key'
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = (Get-Content -LiteralPath (Join-Path $dir 'oaiy-updater.key.password.txt') -Raw).Trim()
Push-Location platform\desktop
npx tauri signer sign $probe | Out-Null
Pop-Location
Remove-Item Env:\TAURI_SIGNING_PRIVATE_KEY_PATH, Env:\TAURI_SIGNING_PRIVATE_KEY_PASSWORD
node platform\scripts\verify-signature.mjs $probe
Remove-Item -LiteralPath $probe, "$probe.sig"
```

Or in a POSIX shell (on Linux or macOS; on Windows use the PowerShell version, since a path
of Git Bash is not one Node understands):

```sh
probe="$(mktemp)" && printf probe > "$probe"
(cd platform/desktop && TAURI_SIGNING_PRIVATE_KEY_PATH="$HOME/.oaiy-signing/oaiy-updater.key" \
  TAURI_SIGNING_PRIVATE_KEY_PASSWORD="$(tr -d '\r\n' < "$HOME/.oaiy-signing/oaiy-updater.key.password.txt")" \
  npx tauri signer sign "$probe" > /dev/null)
node platform/scripts/verify-signature.mjs "$probe"
rm -f "$probe" "$probe.sig"
```

It must say `OK: ... verifies against the public key in tauri.conf.json (plugins.updater.pubkey);
key id ...`. `NOT VERIFIED ... the key ids differ` (or `the file does not match its signature`)
means the key you hold is not the pair of the public key in `tauri.conf.json`: do NOT release;
find the right key, or make a new pair and follow "If the private key is lost" in
[UPDATES.md](UPDATES.md#the-key-and-its-custody). `CANNOT CHECK` means something was missing
(the key file, the signature the CLI should have written, a wrong path). The password goes
through the environment and never through `-p`/`--password`, which would put it in the shell's
history and in the process list; `| Out-Null` and `> /dev/null` discard the CLI's own output
(it holds the signature, which is public, but nothing here needs it). The script is the same
check the release job makes before it writes `latest.json` (`platform/scripts/minisign.mjs`).

The key was made once. Until you delete them, its file and its password sit on the machine that
made it, in `C:\Users\<you>\.oaiy-signing` (outside every repository, with a README that says
this again). Keep an offline copy of both in a password manager, and then delete the local ones:
from then on the private key is in that offline copy and in the `release` environment's two
secrets, and nowhere else (UPDATES.md, "The key and its custody").
**Losing the private key means no installed OAIY can be updated again**: they trust only the
old public key. The way out is a new key with its public half in `tauri.conf.json`, a new
release, and everyone installing that release by hand from the releases page; see
[UPDATES.md](UPDATES.md#the-key-and-its-custody). Never build a test with this key: make a
throwaway one (`npx tauri signer generate -w <a folder outside the repository>`).

The release job also holds the feed to the repository the desktop is pinned to: the repository
it is run in must be the one `tauri.conf.json`'s updater endpoint names (`f2i-com/oaiy.com`), or
`make-latest-json.mjs` stops the release. The desktop takes installers only from that repository's
releases (`update::REPO`), so a feed for another one, from a fork's run, say, would be read by no
installed OAIY. A fork that wants updates of its own changes the endpoint, the public key and
`REPO`.

The release job checks each signed installer against the public key in `tauri.conf.json`
before it writes `latest.json` and stops if one does not verify: the Tauri CLI only *warns*
when the secrets hold a different key from the one in the build, and a release signed with the
wrong key would install on nobody. If that step fails, the two secrets do not belong to the
public key in `tauri.conf.json`.

The base `tauri.conf.json` does not turn the updater artifacts on (`bundle.createUpdaterArtifacts`),
and the release workflow does not either: `npm run tauri:build` builds an unsigned installer
everywhere, on a developer's machine and in the workflow, and the `sign` job signs it afterwards.

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
   installer or a signature is missing, or does not verify against the key in
   `tauri.conf.json`, or was made for another version or another kind of file. The `sign` job
   comes after the gate and the desktop builds and waits for the `release` environment's
   reviewer: approve it when `verify` and both desktop builds are green.

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
keeps the files as the run's artifacts, and creates no release. Its installers are
UNSIGNED (the `sign` job starts on a tag only, and no other job has the key), and the
build's log says so.

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
- **The first release with the updater is installed by hand by everyone.** Only a copy that
  already contains the updater can update itself, and no earlier copy does: 0.1.0 is a download
  from this page for every owner of OAIY, and from that copy on, updates are offered inside OAIY
  (UPDATES.md). Say so in the notes of that release.
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
