# Isolated native plugin qualification

`oaiy-desktop --isolated-root <absolute-local-directory>` opts into a Windows-only native Desktop mode for local plugin qualification. It provides separate host storage, a separate WebView profile and process identity, and a reserved loopback API endpoint. The dashboard opens on Plugins; contributed plugin screens remain available.

This mode is intended for trusted native plugins. It is **not an OS sandbox**: an explicitly started plugin executable retains the operating-system permissions of its user. Package verification, signature and trust decisions remain necessary. Root validation and locking do not defend against a malicious process with the same user's filesystem rights, or against the owner replacing files while the process runs.

## Launch and storage contract

Use a dedicated absolute drive path such as `C:\oaiy-qualification\run-01`. Both `--isolated-root <path>` and `--isolated-root=<path>` are accepted. `--hidden` keeps the native window hidden in the tray at startup; it does not make this a headless-server launch.

```powershell
.\src-tauri\target\release\oaiy-desktop.exe --isolated-root 'C:\oaiy-qualification\run-01' --hidden
```

The executable parses and prepares isolation before Tauri initialization or normal desktop-path resolution. Missing, repeated or misspelled isolation arguments fail closed. Other platforms currently refuse isolated launch because equivalent local-volume checks have not been qualified.

Before creating the root or WebView, a read-only preflight refuses nonempty `WEBVIEW2_*` environment overrides and configured WebView2 folder/browser/channel policies in either Windows registry hive/view. An unreadable policy also refuses launch. This is deliberately conservative, including policies that might target another app; it does not change environment or machine policy settings. These overrides can supersede an explicitly selected WebView profile, as described in [Microsoft's WebView2 environment contract](https://learn.microsoft.com/en-us/microsoft-edge/webview2/reference/win32/webview2-idl?view=webview2-1.0.3537.50#createcorewebview2environmentwithoptions). Refused isolated launches exit with code 2.

The root must be an ordinary directory on a local writable drive. Relative paths, drive roots, parent traversal, UNC/network/device paths, ambiguous Windows names, symlinks, junctions and other reparse points are refused. Ancestors are checked, and existing trees are checked before reuse; hard-linked files and nested isolated roots are also refused.

A new root must be empty, apart from an empty lock file left by an interrupted initialization. A reusable root must carry the exact supported `.oaiy-isolated.json` marker for its canonical path and identifier. A foreign, malformed or moved marker is refused. The marker is not an authorization token.

| Root entry | Purpose |
| --- | --- |
| `.oaiy-isolated.json` | Versioned root identity and ownership marker |
| `.oaiy-isolated.lock` | Exclusive lock held for the process lifetime |
| `config/` | Isolated `desktop-config.json` and token-path namespace |
| `data/` | Host stores, plugin packages and plugin state, logs and bridge state |
| `models/` | Fixed isolated model directory |
| `webview/` | Main native WebView user-data profile |

The normal desktop configuration, data directory, model search roots and HuggingFace token are not adopted. Data/model relocation, additional model directories and credential writes are unavailable in this mode. The host does not repurpose `HOME`, `APPDATA` or a user's existing credential directory, and it does not fall back to shared storage when isolation preparation fails.

The canonical root determines a `com.oaiy.isolated.<hash>` runtime identifier, installed before Tauri builds the app. The ordinary single-instance plugin remains enabled. Different roots have different identities; a second process using the same root is refused by its held root lock. The normal identifier remains `com.oaiy.app`.

Before the WebView starts, the host binds `127.0.0.1:0` and retains the resulting listener. Ports below 1024 and the normal desktop/development ports 17972 and 17973 are excluded. The same held listener is transferred once to the HTTP server; there is no probe-and-rebind gap or fallback to the normal endpoint. A restart may select a new port.

Native startup installs this context before dashboard modules run:

```javascript
Object.defineProperty(window, '__OAIY_DESKTOP_LAUNCH__', {
  value: Object.freeze({ version: 1, isolated: true, apiPort: 48123 }),
  writable: false,
  configurable: false,
  enumerable: false,
});
```

The example port is illustrative. The frontend requires a callable Tauri bridge, the immutable marker, exactly the three own scalar fields shown above, and an integer port from 1024 through 65535 excluding 17972/17973. It constructs `http://127.0.0.1:<port>` locally. A malformed native marker prevents API initialization. URL parameters, local storage, arbitrary URLs and bearer credentials cannot select the endpoint. An ordinary browser ignores the marker; an unmarked normal launch retains `http://127.0.0.1:17972`.

The isolation flag does not change normal-launch initialization, single-instance identity, storage or default endpoint. The typed plugin error behavior described below applies to plugin screens in both launch modes.

Source: [root preparation](../src-tauri/src/isolated.rs), [native startup and paths](../src-tauri/src/lib.rs), [HTTP listener handoff](../src-tauri/src/http.rs), [frontend launch validation](../src/nativeLaunch.ts).

## Available capabilities

The restricted HTTP allowlist permits health/config/module/update-status reads; plugin listing, installation, explicit start/stop, trust and enablement; plugin logs and screen assets; and connector command requests. The normal authentication, origin and CORS guards still surround this allowlist. Merely knowing the selected port does not authorize a plugin installation or command.

Everything outside the allowlist is refused, including future routes unless explicitly added. Refused HTTP operations return status 403 with code `isolated_capability_unavailable`. Tauri IPC has its own small read-only allowlist plus `hide_embedded`; it does not provide a settings, execution, external-navigation or update bypass. The dashboard restricts navigation to Plugins and contributed screens, and does not enter the first-run/setup flow.

The isolated Tauri authority grants only local main-WebView event listen/unlisten operations; application commands pass through the separate refusal policy. Optional shell/dialog/notification/updater plugins and Agent/Flow URI protocols are not registered. Main-document navigation stays on the bundled app origin. See [native ACL](../src-tauri/src/isolated_acl.rs) and [WebView preflight](../src-tauri/src/isolated_webview.rs).

Isolated startup skips engine attachment/start, GPU enumeration, service and plugin boot autostart, the flow worker, account/link/relay workers, calendar synchronization, the voice gateway, the update scheduler and hidden Agent preload. AI providers, Codex, MCP, engines, voice/phone, accounts, flows, setup, service operations, downloads and settings mutations are unavailable through the host's qualification surface. Plugins must be started explicitly after the normal trust checks. Plugin supervision and the local connector/dispatch machinery still operate for those explicitly started plugins.

These are restrictions on OAIY's host capabilities, not a network or filesystem sandbox around a native plugin. No live-model, account, MCP or unrelated integration qualification is implied by a plugin-only run.

Source: [HTTP and IPC policy](../src-tauri/src/isolated_policy.rs), [startup gates](../src-tauri/src/lib.rs), [HTTP background-service gates](../src-tauri/src/http.rs), [dashboard gates](../src/App.tsx).

## Build an executable for this qualification scope

Run these PowerShell commands from `platform/desktop` in the source checkout being reviewed, with the Windows Node/Rust/MSVC toolchains and dependency caches available. Offline commands fail if their required packages are not cached. Check each command's exit status before proceeding.

```powershell
npm ci --offline --ignore-scripts
if ($LASTEXITCODE -ne 0) { throw 'Dependency preparation failed.' }
node node_modules/typescript/bin/tsc --noEmit
if ($LASTEXITCODE -ne 0) { throw 'TypeScript checking failed.' }
node node_modules/vite/bin/vite.js build
if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }

$qualificationPreviousTauriConfig = $env:TAURI_CONFIG
Push-Location src-tauri
try {
  $env:TAURI_CONFIG = '{"bundle":{"resources":[]}}'
  cargo build --offline --locked --release --features custom-protocol --bin oaiy-desktop
  if ($LASTEXITCODE -ne 0) { throw 'Native qualification build failed.' }
} finally {
  $env:TAURI_CONFIG = $qualificationPreviousTauriConfig
  Pop-Location
}
```

The `custom-protocol` feature embeds the matching built dashboard. Isolated launch refuses a build without it and removes the development URL at native startup, preventing an isolated window from loading a different process's development frontend.

The process-scoped `TAURI_CONFIG` override deliberately clears `bundle.resources` for this bounded qualification build. The CLI, Agent and Flows resource bundles are **not built or included** by these commands. Direct TypeScript/Vite invocation also skips the normal CLI synchronization and page-staging scripts. This is an executable qualification procedure, **not shipping-release or installer qualification**. It does not replace the normal bundle build or establish that those omitted resources work. The override is restored after Cargo finishes.

With Cargo's default target location, the executable is `src-tauri/target/release/oaiy-desktop.exe`; an explicitly configured Cargo target directory changes that location. Launch the newly built executable against its own dedicated root. Reusing an older binary does not qualify the current source.

## Plugin command errors

The host unwraps the gateway response and the plugin's SDK response separately. An explicit outer or inner `ok: false` always rejects `PluginHost.command(...)`, even when error metadata is missing or malformed. The frame still receives the legacy `error: string`; newer frames may also receive a closed `errorDetails` record.

Typed metadata contains only own scalar `code`, `message` and optional `version` fields. A code is 1 through 64 ASCII characters matching `[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}`. Messages are limited to 1024 characters, with control characters replaced by spaces and surrounding whitespace removed. A version, when present, must be a positive safe integer. Unknown well-formed codes remain plugin-owned. Extra fields, symbols, accessors, custom prototypes, invalid versions and conflicting error/envelope versions prevent typed promotion; the refusal still rejects as a legacy `Error`.

For example, this gateway reply resolves to `{ saved: true }`:

```json
{ "ok": true, "result": { "ok": true, "data": { "saved": true } } }
```

This older plugin reply rejects with `Error("Not connected.")` and retains a string-only error wire response:

```json
{ "ok": true, "result": { "ok": false, "error": "Not connected." } }
```

This typed inner refusal rejects with an `Error` named `PluginCommandError`, carrying `code: "revision_conflict"` and `version: 1`:

```json
{
  "ok": true,
  "result": {
    "ok": false,
    "data": {
      "version": 1,
      "error": { "code": "revision_conflict", "message": "The saved revision has changed." }
    }
  }
}
```

Its frame response contains `ok: false`, the same bounded message in `error`, and `errorDetails: { code, message, version }`. A closed typed object in an outer or inner `error` field is supported too. The bootstrap validates the metadata again and constructs a local Error; it never copies arbitrary plugin properties, causes or prototypes into that Error.

Successful scalar/array results and SDK `data` remain compatible. In particular, an inner `ok: true` with `data.error` stays resolved domain data, allowing a plugin to handle that envelope itself. This transport change does not reinterpret a plugin's successful data as a failure.

The host services only requests from its mounted iframe, preserves scalar request correlation, and drops completions after that screen is replaced. The bootstrap ignores replies from other windows or unknown request IDs and resolves only literal `ok: true`. A rejected call clears its deadline. A timed-out call reports that the outcome may be unknown; a late reply cannot complete a subsequent request. There is no automatic retry. A caller's explicit retry gets a new request ID and the host creates a fresh idempotency key for that action.

Source: [bounded envelope/error helpers](../src/pluginRpc.ts), [injected bootstrap and frame message pump](../src/PluginScreenPage.tsx).

## Evidence and remaining gate

The focused evidence recorded during implementation is:

| Check | Recorded result | Scope |
| --- | --- | --- |
| Typed plugin RPC and existing screen regression suites | 112 tests passed in 4 files | Pure envelopes, production bootstrap and mounted React message pump, legacy/success compatibility, forged/correlated replies, rejection/retry, timeout and replaced-screen cancellation |
| Frontend launch/navigation group | 64 tests passed | Immutable launch marker, API endpoint selection and isolated dashboard behavior, with related frontend regression coverage |
| Full frontend and shell regression suites | 694 frontend tests and 21 shell tests passed | Includes the focused frontend groups above |
| Native isolation, ACL and WebView preflight | 19 tests passed; one unrelated GPU test ignored | Root parsing/validation, identity/lock/listener ownership, junction/hardlink checks, native authority and override refusal |
| Existing HTTP regression group | 37 tests passed | Ordinary auth/origin/CORS behavior and routes |
| Headless server compilation | Passed | `cargo check --offline --locked --no-default-features --bin oaiy-server` |
| TypeScript checking | Passed | `tsc --noEmit` on the reviewed frontend source |

The frontend commands, run from `platform/desktop`, are:

```powershell
node node_modules/vitest/vitest.mjs run src/pluginRpc.test.ts src/PluginScreenPage.rpc.test.tsx src/PluginScreenPage.test.tsx src/PluginScreenPage.status.test.tsx
node node_modules/vitest/vitest.mjs run src/nativeLaunch.test.ts src/App.isolated.test.tsx src/App.nav.test.tsx
```

The RPC tests execute the production `HOST_BOOTSTRAP` source and mounted message listener. Their API responses and jsdom message transports are mocked. The isolated UI tests mock the launch snapshot and panel boundaries; the native-launch tests directly exercise the immutable marker and imported API URL routing. Those component tests do not establish native WebView rendering or real plugin execution.

The [retained executable smoke report](isolated-native-smoke.json) separately records two real hidden native instances, distinct roots/identifiers/listeners, populated owned WebView profiles, and the bundled dashboard reaching each owned API before the harness sends a request. Same-root duplication and relative/network/foreign roots were refused. Seven actual restricted API calls returned the typed 403 refusal; an installation request without a permitted origin was refused by the normal authentication guard. All held processes stopped and their API listeners disappeared; the pre-existing normal listener owners remained unchanged. No protected normal port was contacted. The generic harness is [qualify-isolated-desktop.py](../scripts/qualify-isolated-desktop.py).

The tested Windows executable is 39,275,520 bytes with SHA-256 `2a5c4b8a6e848229476c83c37f9cf292fe9ca1783a3113abdd5b2e8942355213`. It uses the qualification build described above, with optional CLI/Agent/Flow resource bundles omitted. The build and run kept all 458 recorded source/asset input hashes unchanged. It was compiled with Rust/Cargo 1.92.0 and the frontend built with Node 24.19.0. Existing compiler and Vite chunk-size warnings remain; no new compilation or test failure is outstanding.

Run the native tests from `src-tauri` with the same process-scoped qualification `TAURI_CONFIG` as the build:

```powershell
cargo test --offline --locked --release --features custom-protocol --lib isolated
cargo test --offline --locked --release --features custom-protocol --lib http::tests
```

From `platform/desktop`, give the smoke harness an absolute executable and a **new**, dedicated output directory. It retains its test roots and report and stops only its own process handles:

```powershell
python scripts/qualify-isolated-desktop.py --exe 'C:\oaiy-build\oaiy-desktop.exe' --output 'C:\oaiy-qualification\smoke-01'
```

Automatic CI is paused in this repository's workflow for cost control. Local evidence does not imply a remote CI or release gate passed.

**Pending acceptance gate:** supported native visual-control tooling was unavailable during this qualification. Native window inspection and the interactive plugin open/save/close/reopen workflow remain unverified. A successful build, hidden launch, HTTP check or jsdom test does not satisfy that visual and interaction gate. No installer, deployment, live integration or full-release acceptance is claimed here.
