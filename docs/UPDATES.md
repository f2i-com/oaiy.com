# Updates

How OAIY Desktop finds a newer release, gets it, checks it and installs it, and what it
never does. For how a release is made, see [RELEASING.md](RELEASING.md).

## The policy

**Notify, download when asked, install when idle and only when the owner presses the
button.** In order:

1. OAIY **looks** for a newer release: a little after it starts (90 seconds), then once a
   day, when Settings or the tray asks ("Check for updates"), and never more than once every
   30 seconds however it is asked. The daily look can be switched off in Settings; a check
   asked for by hand works either way. It is one small request to
   `github.com/f2i-com/oaiy.com/releases/latest/download/latest.json`, which says which
   version is current. The request carries no identifier of this computer or its owner
   (only a user agent that names OAIY); like any request it shows GitHub the computer's
   network address.
2. If there is a newer version, OAIY **says so**: a banner on the Overview (dismissible until
   the version after that one) and Settings → About and updates, with its notes.
3. **Download** is a button. OAIY fetches the installer and checks its signature (below). A
   download that does not verify is thrown away and never becomes installable.
4. **Restart to update** is a button, and it is off while anything is in the way (below). It
   names each reason beside it. Pressed, OAIY saves the Agent's work, stops what it runs,
   hands the installer the verified bytes, and opens again by itself.

Nothing installs by itself, at any hour. A restart takes a minute or so (the model and the
voice load again), and OAIY answers no call or text meanwhile: the decision to press the
button, and when, is the owner's. A download waits in memory until then; restarting OAIY
first forgets it and it is downloaded again.

## What can update itself

| Install | Updates from inside OAIY? |
|---|---|
| Windows, the NSIS `setup.exe` (per user, `%LOCALAPPDATA%\OAIY`) | Yes. Installed passively (a progress bar, no prompts, no administrator rights), and OAIY opens again with the arguments it had (a start in the tray stays in the tray). |
| Linux, the AppImage | Yes. The AppImage file is replaced and OAIY restarts. |
| Windows, the MSI (machine-wide) | **No: a manual download.** The MSI is another kind of install of the same product; the update is the NSIS installer, and running it over an MSI install would leave two. Settings says so and links the releases page. |
| Linux, `.deb` or `.rpm` | No: a manual download, for the same reason (the feed carries the AppImage). |
| A development build (`tauri dev`, `cargo run`) | No: there is no installed copy to replace. |
| macOS | No release is built. |
| The headless `oaiy-server` | No: it **only reports** that a newer release exists (see below). |

Only what OAIY itself ships is updated. The engines (`oaiy-llm-server`, `oaiy-media`,
`oaiy-voice`) are a separate channel, plugins are installed from a folder or an archive, and
models are downloaded on use; an update of the desktop leaves all of them, and OAIY's data
(its data folder, `%APPDATA%\com.oaiy.app` by default, and the Agent's projects and chats in
the window's own storage), where they are.

## What blocks an install

Each of these keeps "Restart to update" off, by itself, and is shown in words next to it:

- **A live phone call**, from two sources, both asked afresh when the button is pressed:
  - *OAIY's own line*: a call that reaches OAIY's realtime stream (what the call hub counts).
  - *The phone plugin*: OAIY asks whichever plugin provides the phone module (the module
    registry says which; nothing in this check knows a plugin by name), by a read-only
    connector command (`call.switchboard`, else `call.current`), whether a call is ringing, on
    the line, waiting or on hold. That is what sees a call the plugin runs through its own
    speech pipeline, screens or holds, which never reaches OAIY's own line. The plugin has 3
    seconds to answer. **A phone plugin that is running and does not answer, answers with an
    error, or answers something OAIY cannot read blocks the install** ("can't tell whether a
    call is live"), and so does one that declares neither command; stopping that plugin
    (Connections, Plugins) lets it through, as does a plugin that is not running at all. A plugin that is
    running without its radio attached, or paused, is such a case: it answers with an error, so
    the install waits until the plugin is stopped or the radio is back.
  - Not checked: a call the phone plugin does not report through those commands, and a call
    on a phone or line OAIY has no plugin for. OAIY cannot wait for what it cannot see.
- **A task the Agent is working on for a flow** (one given to it and not yet answered).
- **A model or file download** (queued or running, OAIY's or the engines'); a paused one
  resumes after the update.
- **The engines making media** (a picture, video, song or 3D model).
- **Something installing**: a service, Python, Node or a plugin.
- **The data folder being moved.**
- **OAIY having started less than two minutes ago**, or not yet knowing what it is doing.

The blockers are looked at when the button is pressed, and again right before anything is
stopped (saving the Agent's work takes a few seconds, and a call may have begun). The
second look asks every source afresh, the phone plugin included: what the status showed a few
seconds ago is never the answer to "is it safe now".

What OAIY cannot see, it cannot wait for: a conversation the person is having in the Agent
page right now is not a task the desktop knows about. "Restart to update" asks the Agent to
save, and ends its turn, as quitting does.

## The install, in order

1. The update and the blockers (nothing is touched if either is wrong).
2. The Agent page is asked to save its work (it saves its projects and chat, and reports back
   to the local API); it has five seconds, and the install goes on if it does not answer.
3. The blockers again.
4. Everything OAIY runs is stopped, in the order quitting stops it (the same code: quitting
   and updating share one list): the engines OAIY started, the script host, the plugins, the
   services. The services that were running are written down so the OAIY that opens after
   the update starts them again.
5. The installer takes the verified bytes. On Windows that starts the installer and ends
   this process; the installer replaces OAIY and opens it again (the old tray icon may stay
   on screen until the mouse passes over it: OAIY does not remove it before the installer is
   started, so that a failed start leaves OAIY as it was). On Linux the AppImage is replaced
   and OAIY restarts through its normal exit.

If a part will not stop, or the installer cannot be started, everything that was stopped is
started again (last stopped, first started) and the update is shown as **failed**, with the
reason in words. OAIY is then running as before. If something could not be started again,
the message says which and to quit OAIY from its tray icon and open it again.

The desktop process is never put in a job object that kills its children when it ends:
the installer it starts would inherit that and be killed with it.

## Trust: what is checked

- **The signature.** Each installer an update can install is signed by the release workflow
  with the updater key (minisign, made by the Tauri CLI). The matching public key is inside
  every OAIY (`plugins.updater.pubkey` in `tauri.conf.json`). The updater plugin checks the
  download against it, and OAIY checks the same bytes again (`update::verify`) before it
  keeps them: only bytes that pass become a `VerifiedPackage`, which is the only thing the
  install step accepts. A signature that does not match, a signature of another key, a
  changed byte, or a signature that is not a signature: the download is discarded and shown
  as failed.
- **What the signature was made for.** A signature proves that the key signed those bytes. It
  does not say which release they belong to, and the version in the feed is only text: a feed
  that announced 9.9.9 with the address and the genuine signature of an OLD installer would
  install an old OAIY as if it were an update. What ties the bytes to a release is the name of
  the file the signature was made for. The Tauri CLI writes it into the signature
  (`file:OAIY_0.1.0_x64-setup.exe`) and the key's signature covers it, so it cannot be edited.
  OAIY refuses the download unless that name (1) ends the way this platform's installer does,
  `-setup.exe` on Windows and `.AppImage` on Linux, and (2) has the announced version as one
  whole part of it: the parts are what the underscores of the bundler's names separate, so
  `0.1.0` is not found in `10.1.0`, `0.1.05` or `0.1.0-rc.1`. Nothing else of the name is
  looked at, on either platform: not the product name, not the architecture (the AppImage's
  name has changed between Tauri versions, and neither adds to what stops a downgrade). The
  release job signs the installers under the bundler's names (`OAIY_<v>_x64-setup.exe`,
  `OAIY_<v>_amd64.AppImage`) and makes the same check from each `.sig` before it publishes.
- **The address.** The installer must be this project's release asset on `github.com`, over
  https, under the tag `v<version>` (or `<version>`) and named `oaiy-desktop-<version>-...`,
  and that version is the one the feed announced. That keeps a feed from sending OAIY
  elsewhere, and it is not what ties the bytes to their version: a release can hold any file
  under any name, so the version is proven by the signature (above), not by the address.
- **The feed.** At most 256 KiB, counted as it arrives; https only, redirects too (GitHub
  sends the request through two); every field checked; a version that is lower or equal is
  never an update, and neither is one that is not a version. A platform the feed has nothing
  for is "no update for this platform", not an error.
- **Who may ask.** Installing, downloading and checking are commands of the dashboard's own
  window (its webview is labelled `main`), and nothing else. The Agent, the flow editor and
  the engines' page are webviews of the same window and are refused; a web page is not a
  webview of OAIY at all. None of the three takes an address, a path or a version, and no
  webview is granted the updater plugin's own commands. `POST /api/update/check` is a
  privileged route (OAIY's own window or the token) and only ever reads the feed OAIY was built
  with. The feed's address cannot be changed by a request or a setting. A debug build (never a
  release build) reads one from the environment variable `OAIY_UPDATE_FEED`, for testing
  against a local stub.

What the installers are **not** is signed for Windows: they carry no Authenticode
certificate, so SmartScreen warns on a first download (RELEASING.md). That is separate from
the update signature above, which OAIY checks and Windows does not.

## The headless server

`oaiy-server` never downloads or replaces itself: it runs as an unprivileged user under a
hardened unit and has nothing to replace itself with. It **reports** a newer release:

```sh
curl -s http://127.0.0.1:17972/api/update/status            # open here, like /api/health (the desktop keeps it to OAIY's own pages: it says whether a call is live)
curl -s -X POST -H "Authorization: Bearer $OAIY_SERVER_TOKEN" http://127.0.0.1:17972/api/update/check
```

The status says `currentVersion`, `latestVersion`, `notes`, when it last looked, and
`canAutoUpdate: false` with the reason. The server looks only when asked (there is no daily
check); the check is limited to one in 30 seconds.

To upgrade one, by hand, keeping each version in a directory of its own so that going back
is one command (the archive has `resources/` beside the server, and needs them there):

```sh
new=0.2.0
cd /tmp
curl -fLO https://github.com/f2i-com/oaiy.com/releases/download/v$new/oaiy-server-$new-linux-x86_64.tar.gz
curl -fLO https://github.com/f2i-com/oaiy.com/releases/download/v$new/SHA256SUMS.txt
grep " oaiy-server-$new-linux-x86_64.tar.gz\$" SHA256SUMS.txt | sha256sum -c -
sudo mkdir /opt/oaiy-server/$new
sudo tar -C /opt/oaiy-server/$new -xzf oaiy-server-$new-linux-x86_64.tar.gz
sudo ln -sfn /opt/oaiy-server/$new /opt/oaiy-server/current
sudo systemctl restart oaiy-server
curl -s http://127.0.0.1:17972/api/health                    # "version" is the new one
```

The first time, point the unit at the link instead of `/usr/local/bin`
(`sudo systemctl edit oaiy-server`, then `ExecStart=` on one line to clear it and
`ExecStart=/opt/oaiy-server/current/oaiy-server` on the next). To go back:
`sudo ln -sfn /opt/oaiy-server/<the old version> /opt/oaiy-server/current && sudo systemctl restart oaiy-server`.
`SHA256SUMS.txt` is the release's own list; `latest.json` and the `.sig` files are in it too.
The data (`/var/lib/oaiy`) is not in the version directory and is untouched. Restarting stops
the server's services and plugins (SIGTERM stops them cleanly); do it when nothing is running
that should not be interrupted.

A Windows headless install (`oaiy-server-<v>-windows-x64.zip`) is upgraded the same way:
stop it, unzip the new one into a new folder beside the old, start that.

## The key and its custody

Every update is signed with one minisign key. Its **public** half is in `tauri.conf.json`
and in every installed OAIY. Its **private** half and password are two GitHub Actions
secrets (`TAURI_SIGNING_PRIVATE_KEY`, `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`) and an offline
copy in the owner's password manager, and nowhere else: not in a repository, not in an
artifact, not in a log. A release without the secrets stops before anything is built, and a
release whose signatures do not verify against the public key in `tauri.conf.json` (the
secrets belong to another key) stops before it is published.
[RELEASING.md](RELEASING.md#the-updater-key) says how the secrets are set.

**If the private key is lost**, no one can sign an update that installed copies accept, and
they cannot be told to trust another key: they trust only the key they carry. The way out is
a new key: make a keypair (`npx tauri signer generate`), put the new public key in
`tauri.conf.json`, set the two secrets to the new private key and password, and release a
new version. **Installed copies cannot verify that release, so people install it by hand**
from the releases page (the installer runs over the old install and keeps their data); from
then on the copies update themselves again. If the old key is still held, a "bridge" release
signed with it, carrying the new public key, moves installed copies over without anyone
doing anything: publish that first, then a release signed with the new key.

**If the key leaks**, do the same at once: whoever holds it can sign code every installed
OAIY would accept. Rotate it, and tell people to install the new release by hand.

## Not yet

- **Rollback.** A failed install is put back (above), but an install that finishes and then
  does not work is not: the previous installer is not kept. The design: keep the last
  installer under `<data>/updates`, mark the first launch after an update, and offer
  "Reinstall the previous version" when that launch does not come up healthy.
- **A quiet-hours window** to install in. The owner presses the button; nothing picks a time.
- **`requireSignedVersion`** (the updater plugin can insist that a signature carries a
  `version:` field of its own): the Tauri CLI in this repository (2.11) writes none, only
  `timestamp:` and `file:`, so it is off: turned on, every release would be refused. The
  version is tied to the signature by the name of the file it was made for (see "What the
  signature was made for"). OAIY already refuses a signature whose `version:` field, if it has
  one, is not the announced version. Should a later CLI write the field, check a built
  signature for it, then turn this on.
- **A beta channel**, **Authenticode signing**, **macOS**, updates of the engines and of
  plugins.
