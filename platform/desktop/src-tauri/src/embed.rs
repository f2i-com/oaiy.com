//! The pages OAIY shows in its window beside the sidebar: the agent (the app in
//! `app/`), the flow editor (`platform/ui`), and the engines' own control pages
//! (served by the engines at their address; see `engines.rs`).
//!
//! Each is a webview of its own laid over the dashboard's content area, not a
//! frame inside the dashboard: as a top-level page it can be cross-origin
//! isolated, which the agent's code sandbox needs (a Worker blocking on shared
//! memory), and a frame of the dashboard could not be. Each is served from a
//! scheme of its own (`oaiy`, `oaiyflows`; `http://<scheme>.localhost` on
//! Windows) with the headers that isolation takes, and starts knowing the
//! desktop's address and a token for it, so neither has to pair. On a Mac and
//! on Linux the agent's comes from a port of its own instead ([`AGENT_PORT`]):
//! WebKit gives a page on a scheme of its own no shared memory, and WebKitGTK no
//! cross-origin isolation. WebKitGTK also leaves off the storage the agent keeps
//! its projects in, which each page's webview has turned on ([`webkit_features`]).
//!
//! All three follow the dashboard's light or dark (`set_theme`): each starts
//! in it and is told when it changes, and the ground under each (what shows
//! before the page has drawn) is the dashboard's own for that theme ([`ground`]).
//!
//! A page is not laid over the dashboard before it has something to show: it is
//! made, or brought to a new size, out of sight, and brought in once it has
//! loaded and had a moment to draw (or a short while has passed), so a person
//! sees the dashboard's "Opening…" and then the page, never a blank one
//! ([`show_embedded`]).
//!
//! The keyboard is the dashboard's until a page is opened: making a webview
//! takes it from nobody, opening a page gives it to the page, putting a page
//! away gives it back, and Tab goes on from the dashboard's last stop into the
//! page and from the page's last stop back to the dashboard's first (Shift+Tab
//! the other way), where WebView2 would keep it going round inside one webview
//! ([`keyboard`], Windows).

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::SeqCst};
use std::time::{Duration, Instant};

use percent_encoding::percent_decode_str;
use tauri::http::{header, Request, Response, StatusCode};
use tauri::webview::{Color, NewWindowResponse, PageLoadEvent};
use tauri::{AppHandle, LogicalPosition, LogicalSize, Manager, Runtime, Url, WebviewBuilder, WebviewUrl};

pub const AGENT_SCHEME: &str = "oaiy";
pub const FLOWS_SCHEME: &str = "oaiyflows";

/// The port a Mac and Linux serve the agent's page from, on this computer alone ([`serve_agent_over_http`]).
///
/// The agent's code sandbox blocks on shared memory. WebKit calls a page on a scheme of its own, served with the
/// headers that isolate a page, cross-origin isolated, and still gives it no `SharedArrayBuffer`; the same page from
/// `http://127.0.0.1` has both (`tools/mac/webview-probe.swift` on the first Mac, macOS 27). WebKitGTK (2.52) does not
/// call such a page isolated at all, and the page from `http://127.0.0.1` it does: with `SharedArrayBuffer` turned on
/// ([`linux_shared_memory`]), the sandbox has what it needs. The port is fixed because a page's storage is its
/// origin's: another port would be an agent with none of its projects. The engines' gateway lets this origin in
/// (`oaiy_studio::AGENT_HTTP_ORIGIN`, which a test holds to this port).
pub const AGENT_PORT: u16 = 17974;

/// Whether this process holds [`AGENT_PORT`] and serves the agent's page there.
static AGENT_PORT_HELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The agent page's origin while this process serves it over http (a Mac, Linux; its port held), and only then: the desktop's
/// API takes it for the agent's window ([`crate::http::is_embedded_origin`]). A program that had taken the port would
/// serve its own pages from that origin to any browser here; while OAIY holds it, nothing else answers there.
pub fn agent_http_origin() -> Option<&'static str> {
    AGENT_PORT_HELD.load(std::sync::atomic::Ordering::Acquire).then_some(oaiy_studio::AGENT_HTTP_ORIGIN)
}

/// On a Mac and Linux: hold [`AGENT_PORT`] and serve the agent's page from it, with the headers that isolate it
/// ([`serve_path`]), so that its webview opens it there. Before the page is made. Where the port is taken (or this is
/// Windows, whose WebView2 isolates the page on its scheme) the page stays on its own scheme, and its code sandbox says
/// it is unavailable.
pub fn serve_agent_over_http<R: Runtime>(app: &AppHandle<R>) {
    use std::sync::atomic::Ordering;
    if !cfg!(any(target_os = "macos", target_os = "linux")) {
        return;
    }
    let Some(root) = dist(app, Page::Agent) else { return };
    let listener = match std::net::TcpListener::bind(("127.0.0.1", AGENT_PORT)).and_then(|l| l.set_nonblocking(true).map(|()| l)) {
        Ok(listener) => listener,
        Err(e) => {
            log::warn!("embed: the agent's page stays on {AGENT_SCHEME}://localhost, where it has no code sandbox: port {AGENT_PORT} is not free ({e})");
            return;
        }
    };
    AGENT_PORT_HELD.store(true, Ordering::Release);
    log::info!("embed: the agent's page is served from {} (its code sandbox needs shared memory)", oaiy_studio::AGENT_HTTP_ORIGIN);
    tauri::async_runtime::spawn(async move {
        let served = async {
            let listener = tokio::net::TcpListener::from_std(listener)?;
            // (the page's files and nothing else: no route of the desktop's API is here, and the route check lists
            // these two as another listener's, `auth::route_coverage::NOT_ON_MAIN_ROUTER`)
            let file = move |uri: axum::http::Uri| {
                let root = root.clone();
                async move { serve_path(Some(&root), Page::Agent, uri.path()).map(axum::body::Body::from) }
            };
            let pages = axum::Router::new().route("/", axum::routing::get(file.clone())).route("/*path", axum::routing::get(file));
            axum::serve(listener, pages).await
        };
        if let Err(e) = served.await {
            log::warn!("embed: the agent's page server on port {AGENT_PORT} stopped: {e}");
        }
        AGENT_PORT_HELD.store(false, Ordering::Release);
    });
}

/// The browser arguments of every webview in the window: WebView2 refuses a
/// second webview whose arguments differ from the first's. A hidden window
/// must keep working (the agent answers text messages in the background), so
/// its timers are not throttled.
pub const BROWSER_ARGS: &str = "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Page {
    Agent,
    Flows,
    Engines,
}

impl Page {
    pub const ALL: [Page; 3] = [Page::Agent, Page::Flows, Page::Engines];
    /// The pages this desktop serves itself (the engines serve their own).
    pub const SERVED: [Page; 2] = [Page::Agent, Page::Flows];

    pub fn parse(name: &str) -> Option<Page> {
        match name {
            "agent" => Some(Page::Agent),
            "flows" => Some(Page::Flows),
            "engines" => Some(Page::Engines),
            _ => None,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Page::Agent => "embed-agent",
            Page::Flows => "embed-flows",
            Page::Engines => "embed-engines",
        }
    }

    /// Its place in [`Page::ALL`]: where what is kept for each page is kept.
    fn slot(self) -> usize {
        match self {
            Page::Agent => 0,
            Page::Flows => 1,
            Page::Engines => 2,
        }
    }

    /// The page as one number, for [`WANTED`] and [`SHOWN`]: one more than its place, so that 0 is no page.
    fn mark(self) -> u8 {
        self.slot() as u8 + 1
    }

    pub fn scheme(self) -> &'static str {
        match self {
            Page::Agent => AGENT_SCHEME,
            Page::Flows => FLOWS_SCHEME,
            Page::Engines => "http",
        }
    }

    /// The page's folder among the bundle's resources.
    fn folder(self) -> &'static str {
        match self {
            Page::Agent => "app",
            Page::Flows => "flows",
            Page::Engines => "engines",
        }
    }

    /// The document it opens on.
    fn start(self) -> &'static str {
        match self {
            Page::Agent => "/index.html",
            Page::Flows => "/app.html",
            Page::Engines => "/",
        }
    }

    /// Its address in the webview: WebView2 (Windows) maps a custom scheme to `http://<scheme>.localhost`; a Mac and
    /// Linux serve the agent's from a port of its own where they hold one ([`agent_http_origin`]).
    fn url(self) -> Url {
        if self == Page::Engines {
            let base = crate::engines::ui_url().unwrap_or_else(|| "http://127.0.0.1:7860".into());
            return base.parse().unwrap_or_else(|_| "http://127.0.0.1:7860/".parse().expect("a valid URL"));
        }
        if let (Page::Agent, Some(origin)) = (self, agent_http_origin()) {
            return format!("{origin}{}", self.start()).parse().expect("the agent's page's URL is valid");
        }
        let text = if cfg!(windows) {
            format!("http://{}.localhost{}", self.scheme(), self.start())
        } else {
            format!("{}://localhost{}", self.scheme(), self.start())
        };
        text.parse().expect("an embedded page's URL is valid")
    }
}

/// Where a page's built files are: `OAIY_APP_DIST` / `OAIY_FLOWS_DIST`, then
/// the bundle's resources, then (a build from the repository) its own build
/// folders (`app/dist`, `platform/ui/dist`).
fn dist<R: Runtime>(app: &AppHandle<R>, page: Page) -> Option<PathBuf> {
    let var = match page {
        Page::Agent => "OAIY_APP_DIST",
        Page::Flows => "OAIY_FLOWS_DIST",
        Page::Engines => return None,
    };
    let candidates = dist_candidates(
        std::env::var(var).ok().map(PathBuf::from),
        app.path().resource_dir().ok(),
        Path::new(env!("CARGO_MANIFEST_DIR")),
        page,
        cfg!(debug_assertions),
    );
    with_start_document(candidates, page)
}

/// The folders a page's built files may be in, best first: the variable
/// (`OAIY_APP_DIST`, `OAIY_FLOWS_DIST`), the bundle's copy, the pages' own build
/// folders under the repository at `repo` (`platform/desktop/src-tauri`).
///
/// The bundle's copy is `<resource dir>/resources/<folder>`: tauri.conf.json
/// lists `resources/app` and `resources/flows` (staged by
/// `scripts/stage-pages.mjs`), and Tauri keeps the path of a resource, in the
/// installed program's folder on Windows and in `/usr/lib/<name>` on Linux, and
/// beside the program in `target/<profile>` for a build from the repository.
///
/// A debug build (`tauri dev`) is a developer's own, and takes the build folders
/// they are working on before a copy an earlier staging left among the resources.
fn dist_candidates(var: Option<PathBuf>, resource_dir: Option<PathBuf>, repo: &Path, page: Page, debug: bool) -> Vec<PathBuf> {
    let built = match page {
        Page::Agent => repo.join("../../../app/dist"),
        Page::Flows => repo.join("../../ui/dist"),
        Page::Engines => return Vec::new(),
    };
    let bundled = resource_dir.map(|dir| dir.join("resources").join(page.folder()));
    let mut candidates: Vec<PathBuf> = var.into_iter().collect();
    if debug {
        candidates.push(built);
        candidates.extend(bundled);
    } else {
        candidates.extend(bundled);
        candidates.push(built);
    }
    candidates
}

/// The first of the folders that holds the page's start document: an empty
/// folder (`build.rs` makes `resources/app` and `resources/flows` for a build
/// that stages no pages) is not the page, and does not hide one further on.
fn with_start_document(candidates: Vec<PathBuf>, page: Page) -> Option<PathBuf> {
    candidates.into_iter().find(|d| d.join(page.start().trim_start_matches('/')).is_file())
}

fn mime(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "js" | "mjs" | "cjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "webmanifest" => "application/manifest+json",
        "wasm" => "application/wasm",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "txt" | "md" => "text/plain; charset=utf-8",
        "glb" => "model/gltf-binary",
        "zip" => "application/zip",
        _ => "application/octet-stream",
    }
}

fn not_found(what: &str) -> Response<Vec<u8>> {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(format!("{what} is not part of OAIY").into_bytes())
        .expect("a response")
}

/// The embedder policy an embedded page is served with, which with the opener policy is what makes it cross-origin
/// isolated (the Agent's code sandbox blocks on shared memory, which a page has only then).
///
/// `credentialless` is Chromium's, so Windows' webview's: it lets a page load what other sites serve without their
/// saying so, with no credentials. WebKit, which is the webview of a Mac and of Linux, does not know that value and
/// takes a page served with it as not isolated at all: the Agent said "The code sandbox is unavailable: the page is
/// not cross-origin isolated" on the first Mac that ran it. WebKit isolates a page with `require-corp`, under which
/// a file from elsewhere loads only if its server says it may (the page's own files do, just below; what the page
/// asks of the desktop and of the engines it asks with CORS, which that policy lets through).
fn embedder_policy(windows: bool) -> &'static str {
    if windows {
        "credentialless"
    } else {
        "require-corp"
    }
}

/// A file of an embedded page, with the headers that make it cross-origin isolated.
pub fn serve<R: Runtime>(app: &AppHandle<R>, page: Page, request: &Request<Vec<u8>>) -> Response<Vec<u8>> {
    serve_path(dist(app, page).as_deref(), page, request.uri().path())
}

/// The file at `uri_path` of an embedded page whose built files are in `root`, with the headers that make it
/// cross-origin isolated: for its scheme ([`serve`]) and, on a Mac, the agent's port ([`serve_agent_over_http`]).
fn serve_path(root: Option<&Path>, page: Page, uri_path: &str) -> Response<Vec<u8>> {
    let raw = percent_decode_str(uri_path).decode_utf8_lossy().into_owned();
    let path = if raw == "/" || raw.is_empty() { page.start().to_string() } else { raw };
    // Only plain names inside the page's folder.
    let relative = Path::new(path.trim_start_matches('/'));
    if relative.components().any(|c| !matches!(c, Component::Normal(_))) {
        return not_found(&path);
    }
    let Some(root) = root else {
        return not_found(&format!("{path} (the {} page is not built)", page.folder()));
    };
    let mut file = root.join(relative);
    if file.is_dir() {
        file = file.join("index.html");
    }
    let Ok(bytes) = std::fs::read(&file) else {
        return not_found(&path);
    };
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime(&file.to_string_lossy()))
        .header("Cross-Origin-Opener-Policy", "same-origin")
        .header("Cross-Origin-Embedder-Policy", embedder_policy(cfg!(windows)))
        .header("Cross-Origin-Resource-Policy", "cross-origin")
        .header(header::X_CONTENT_TYPE_OPTIONS, "nosniff")
        .header(header::CACHE_CONTROL, "no-cache");
    // The SoftN preview frame has an opaque origin: its module scripts are cross-origin loads.
    if path.starts_with("/softn/") {
        response = response.header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*");
    }
    response.body(bytes).expect("a response")
}

/// What each embedded page knows from its first line: where the desktop is, a token for it, and the theme.
/// The Agent's page also gets the secret its restore hand-over asks for (`backupToken`): no other page
/// has it, and no route returns it.
///
/// Only the page at its own address is told: the script runs in whatever the webview loads, and a popup the page opens
/// (a blob or a blank page of its own, let through by `on_new_window`) shares it but has no navigation guard, so a link
/// followed there would hand the token to that site.
fn desktop_script(theme: &str, page: Page) -> String {
    let backup = if page == Page::Agent { format!(", backupToken: {:?}", crate::backup::agent::page_token()) } else { String::new() };
    let url = page.url();
    let own = format!("{}://{}", url.scheme(), url.host_str().unwrap_or_default()) + &url.port().map(|p| format!(":{p}")).unwrap_or_default();
    format!(
        "if (location.protocol + '//' + location.host === {own:?}) window.__OAIY_DESKTOP__ = Object.freeze({{ origin: {origin:?}, token: {token:?}, theme: {theme:?}{backup} }});",
        origin = format!("http://127.0.0.1:{}", crate::api_port()),
        token = crate::internal_token(),
    )
}

/// The dashboard's light or dark, which the embedded pages follow. Kept in the
/// config folder, so a page made at startup (before the dashboard has said) starts in it.
static THEME: std::sync::Mutex<Option<&'static str>> = std::sync::Mutex::new(None);

fn theme_file<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|dir| dir.join("theme"))
}

fn theme<R: Runtime>(app: &AppHandle<R>) -> &'static str {
    let mut theme = THEME.lock().unwrap_or_else(|e| e.into_inner());
    theme.get_or_insert_with(|| {
        let saved = theme_file(app).and_then(|file| std::fs::read_to_string(file).ok());
        parse_theme(saved.as_deref().unwrap_or("").trim()).unwrap_or("dark")
    })
}

fn parse_theme(mode: &str) -> Option<&'static str> {
    match mode {
        "light" => Some("light"),
        "dark" => Some("dark"),
        _ => None,
    }
}

/// Puts a page in a theme: its `__oaiySetTheme`, or, while it is still loading, where it looks at start.
fn theme_script(mode: &str) -> String {
    format!("window.__oaiySetTheme ? window.__oaiySetTheme({mode:?}) : (window.__OAIY_THEME__ = {mode:?});")
}

/// The ground under a page, dark and light: the dashboard's own `--canvas`, the colour of the box a page is laid over
/// (`.embedded-page` in `platform/desktop/src/styles.css`, where the two are said; a test holds these to that file).
const GROUND_DARK: [u8; 3] = [0x08, 0x0d, 0x18];
const GROUND_LIGHT: [u8; 3] = [0xf8, 0xf5, 0xee];

/// What a page's webview shows where the page has drawn nothing yet (as it loads, or loads again): the dashboard's
/// ground for the theme, not WebView2's white, which was a white window in a dark one for a frame or two.
fn ground(mode: &str) -> Color {
    let [r, g, b] = if mode == "light" { GROUND_LIGHT } else { GROUND_DARK };
    Color(r, g, b, 255)
}

/// The dashboard's theme changed (or it has just started): the embedded pages follow.
#[tauri::command]
pub async fn set_theme<R: Runtime>(app: AppHandle<R>, mode: String) -> Result<(), String> {
    let mode = parse_theme(&mode).ok_or_else(|| format!("no theme called {mode}"))?;
    *THEME.lock().unwrap_or_else(|e| e.into_inner()) = Some(mode);
    if let Some(file) = theme_file(&app) {
        if let Some(dir) = file.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(file, mode);
    }
    for page in Page::ALL {
        if let Some(webview) = app.get_webview(page.label()) {
            let _ = webview.set_background_color(Some(ground(mode)));
            let _ = webview.eval(theme_script(mode));
        }
    }
    Ok(())
}

/// What the dashboard may ask the agent's page to do, by name only: the setup
/// wizard's "Answer calls and texts with OAIY" (the agent's own settings live
/// in its page's storage, which only the page can change), and the first-run
/// wizard's "Continue with the Agent", which opens a "Set up OAIY" conversation
/// (until the page handles it, the dashboard showing the Agent is all it does).
const AGENT_INTENTS: [&str; 2] = ["answerWithOaiy", "setupWithAgent"];

/// Hands an intent to the agent's page: its `__oaiyIntent`, or, while the
/// page is still starting, where it looks when it starts.
fn intent_script(intent: &str) -> String {
    format!("window.__oaiyIntent ? window.__oaiyIntent({intent:?}) : (window.__OAIY_INTENTS__ = (window.__OAIY_INTENTS__ || []).concat([{intent:?}]));")
}

/// Ask the agent's page (made at startup, hidden) to do one of `AGENT_INTENTS`.
#[tauri::command]
pub async fn agent_intent<R: Runtime>(app: AppHandle<R>, webview: tauri::Webview<R>, intent: String) -> Result<(), String> {
    // (the dashboard's alone: the flow editor runs plugins' screens, and `answerWithOaiy` has the Agent answer calls and texts)
    if !crate::update::gui::is_dashboard(webview.label()) {
        log::warn!("refused an Agent intent from the webview \"{}\"", webview.label());
        return Err("Only OAIY's own window can do that.".to_string());
    }
    if !AGENT_INTENTS.contains(&intent.as_str()) {
        return Err(format!("no agent intent called {intent}"));
    }
    let agent = app
        .get_webview(Page::Agent.label())
        .ok_or("the Agent is not open: open it once from the sidebar, then try again")?;
    agent.eval(intent_script(&intent)).map_err(|e| e.to_string())
}

/// What the dashboard may ask the flow editor's page to do, by name: Run history's "Open the editor" on a run opens
/// that run's flow (`openFlow`, with its id), not only the editor.
const FLOWS_INTENTS: [&str; 1] = ["openFlow"];

/// Whether the flow editor's page has loaded (its webview made, its document in): an intent is handed to it then;
/// before, it waits in [`PENDING_FLOWS_INTENT`] and is handed over as the page finishes loading.
static FLOWS_LOADED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The intent asked for while the editor's page was being made or loading (the latest only: one flow is opened).
static PENDING_FLOWS_INTENT: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Hands an intent and its flow to the editor's page: its `__oaiyFlowsIntent`, or, while the page is starting, where it
/// looks when it starts. The id is a flow id ([`crate::bridge::FlowStore::valid_id`]: letters, digits, `-` and `_`).
fn flows_intent_script(intent: &str, flow_id: &str) -> String {
    format!("window.__oaiyFlowsIntent ? window.__oaiyFlowsIntent({intent:?}, {flow_id:?}) : (window.__OAIY_FLOWS_INTENTS__ = (window.__OAIY_FLOWS_INTENTS__ || []).concat([[{intent:?}, {flow_id:?}]]));")
}

/// Ask the flow editor's page to do one of `FLOWS_INTENTS` with the flow `flow_id`. The dashboard shows the editor
/// as it asks; a page not made or not loaded yet is handed the intent as it finishes loading.
#[tauri::command]
pub async fn flows_intent<R: Runtime>(app: AppHandle<R>, webview: tauri::Webview<R>, intent: String, flow_id: String) -> Result<(), String> {
    // (the dashboard's alone, as the Agent's intents are)
    if !crate::update::gui::is_dashboard(webview.label()) {
        log::warn!("refused a flow editor intent from the webview \"{}\"", webview.label());
        return Err("Only OAIY's own window can do that.".to_string());
    }
    if !FLOWS_INTENTS.contains(&intent.as_str()) {
        return Err(format!("no flow editor intent called {intent}"));
    }
    if !crate::bridge::FlowStore::valid_id(&flow_id) {
        return Err(format!("{flow_id:?} is not a flow id"));
    }
    let script = flows_intent_script(&intent, &flow_id);
    let editor = app.get_webview(Page::Flows.label());
    match editor {
        Some(editor) if FLOWS_LOADED.load(std::sync::atomic::Ordering::SeqCst) => editor.eval(script).map_err(|e| e.to_string()),
        _ => {
            *PENDING_FLOWS_INTENT.lock().unwrap_or_else(|e| e.into_inner()) = Some(script);
            Ok(())
        }
    }
}

/// The Agent's page, when it exists (hidden or shown): what the desktop asks to save its work before an update.
pub fn agent_webview<R: Runtime>(app: &AppHandle<R>) -> Option<tauri::Webview<R>> {
    app.get_webview(Page::Agent.label())
}

/// A link that leaves an embedded page: to the system browser. (On Linux through `xdg-open` with the computer's
/// environment, where OAIY runs from an AppImage: the shell plugin's opener hands it the AppImage's libraries.)
fn open_outside<R: Runtime>(app: &AppHandle<R>, url: &Url) {
    if matches!(url.scheme(), "http" | "https" | "mailto") {
        #[cfg(target_os = "linux")]
        {
            let _ = app;
            let _ = oaiy_host_env::command("xdg-open").arg(url.as_str()).spawn();
        }
        #[cfg(not(target_os = "linux"))]
        {
            use tauri_plugin_shell::ShellExt;
            #[allow(deprecated)]
            let _ = app.shell().open(url.as_str(), None);
        }
    }
}

fn own(page: Page, url: &Url) -> bool {
    own_at(&page.url(), url)
}

/// Where the Engines page goes to ask the window for one of the dashboard's pages: this path at its own address,
/// then the page's name (`/oaiy-window/show/storage`, its link to Settings → Storage, where the Hugging Face token
/// is kept). The engines' page has none of the desktop's API to ask with, and is given none for this.
const SHOW_VIEW: &str = "/oaiy-window/show/";

/// The name of the dashboard's page `url` asks the window to show, where it is such an asking: the Engines page's
/// own address (`mine`: its scheme, host and port) and [`SHOW_VIEW`]. An asking is never loaded, whatever it names;
/// what is shown for it is `control::show_view`'s to say (a page `ui_open` shows, by that name, and nothing else of
/// the address is passed on).
fn asked_view<'a>(page: Page, mine: &Url, url: &'a Url) -> Option<&'a str> {
    let at_its_own = url.scheme() == mine.scheme() && url.host_str() == mine.host_str() && url.port_or_known_default() == mine.port_or_known_default();
    if page != Page::Engines || !at_its_own {
        return None;
    }
    url.path().strip_prefix(SHOW_VIEW)
}

/// Whether `url` is at the page's own address `mine`: its scheme, host and port (another port of 127.0.0.1 is another
/// program's, which the window's script would hand the desktop's token), or a blob, data or blank page of its own.
fn own_at(mine: &Url, url: &Url) -> bool {
    (url.scheme() == mine.scheme() && url.host_str() == mine.host_str() && url.port_or_known_default() == mine.port_or_known_default())
        || matches!(url.scheme(), "blob" | "data" | "about")
}

/// Show an embedded page over the dashboard's content area (logical pixels
/// within the window), made on first use; the other embedded page is hidden.
///
/// Async, so it runs off the main thread: making a webview from a synchronous
/// command (which runs on the main thread) waits on that thread's own event
/// loop, and on Windows never returns.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn show_embedded<R: Runtime>(app: AppHandle<R>, window: tauri::Window<R>, page: String, x: f64, y: f64, width: f64, height: f64, seen: Option<Seen>) -> Result<(), String> {
    let page = Page::parse(&page).ok_or_else(|| format!("no embedded page called {page}"))?;
    let rect = [x.max(0.0), y.max(0.0), width.max(1.0), height.max(1.0)];
    let mine = ASKS.fetch_add(1, SeqCst) + 1;
    WANTED.store(page.mark(), SeqCst);
    kept(&PLACE)[page.slot()] = Some(rect);
    let opening = {
        let _one = placing();
        lay(&app, &window, page, rect)?
    };
    if opening {
        // Out of sight for now: the dashboard's "Opening…" shows, in the theme, until the page has loaded and had a
        // moment to draw (or the while it is given is up). Nothing is held meanwhile: whatever is asked next goes ahead.
        let until = Instant::now() + OPENING_DEADLINE;
        let mut loaded = None;
        while ASKS.load(SeqCst) == mine {
            let now = Instant::now();
            if loaded.is_none() && LOADED[page.slot()].load(SeqCst) {
                loaded = Some(now);
            }
            if now >= until || loaded.is_some_and(|at| now.duration_since(at) >= FIRST_PICTURE) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
        let _one = placing();
        if ASKS.load(SeqCst) == mine {
            bring_in(&app, page)?;
        }
    }
    // Asked otherwise since (the person left the page while it was being made, or asked for another): what was asked
    // last holds. A page nobody wants now is put away; the same page asked for again is that asking's to show.
    if ASKS.load(SeqCst) != mine {
        let _one = placing();
        if WANTED.load(SeqCst) != page.mark() {
            put_away(&app, page);
        }
        return Ok(());
    }
    note_placement(&app, &window, page, [x, y, width, height], seen);
    Ok(())
}

/// Each asking of the pages, counted, a show or a hide: a page's first showing makes its webview and waits for its
/// first picture, which takes a while, and what is asked meanwhile is what holds. A hide then found nothing to hide
/// (the editor was left over Run history), and a show answered late would lay one page over another: a showing goes on
/// only while it is still the last thing asked.
static ASKS: AtomicU64 = AtomicU64::new(0);
/// The page asked for last, and the page laid over the dashboard now ([`Page::mark`]; 0 for none).
static WANTED: AtomicU8 = AtomicU8::new(0);
static SHOWN: AtomicU8 = AtomicU8::new(0);
/// Whether each page's document has loaded: once is enough (a page that loads again is not taken away meanwhile).
static LOADED: [AtomicBool; 3] = [const { AtomicBool::new(false) }; 3];
/// Where each page was asked for last (x, y, width, height), and the size each last had while it could draw.
static PLACE: std::sync::Mutex<[Option<[f64; 4]>; 3]> = std::sync::Mutex::new([None; 3]);
static DRAWN: std::sync::Mutex<[Option<(u32, u32)>; 3]> = std::sync::Mutex::new([None; 3]);

/// How long a page that is opening may leave the dashboard's "Opening…" showing: after it, the page is laid over the
/// dashboard loaded or not (on its ground, [`ground`]).
const OPENING_DEADLINE: Duration = Duration::from_millis(1500);
/// From a page's load, or its new size, to its first picture: what it is given to draw before it is brought in.
const FIRST_PICTURE: Duration = Duration::from_millis(80);
/// Whether a page opens out of sight and is brought in drawn. On Windows, where a webview made in place showed black
/// and then WebView2's white before its page (measured: 60 to 150 ms). Elsewhere a page is laid in place as before:
/// Linux packs a page into the window by its margins, which cannot put it out of sight, and neither has been looked at.
const OPENS_OUT_OF_SIGHT: bool = cfg!(windows);

fn kept<T>(of: &'static std::sync::Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    of.lock().unwrap_or_else(|e| e.into_inner())
}

/// A size in whole pixels, to tell whether a page was last drawn at it.
fn whole(rect: [f64; 4]) -> (u32, u32) {
    (rect[2].round() as u32, rect[3].round() as u32)
}

/// Where a page of `rect`'s size is out of sight: left of the window by more than its own width (a child window is
/// cut off at its window's edge). It is shown there, so it draws; a hidden webview draws nothing.
fn out_of_sight(rect: [f64; 4]) -> LogicalPosition<f64> {
    LogicalPosition::new(-(rect[2] + 64.0), rect[1])
}

/// What the dashboard's page saw as it measured the box (`EmbeddedPage.tsx`), for the log: its viewport in CSS
/// pixels, how far it was scrolled, and the pixels of the screen to one of its own.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Seen {
    #[serde(default)]
    width: f64,
    #[serde(default)]
    height: f64,
    #[serde(default)]
    scroll_x: f64,
    #[serde(default)]
    scroll_y: f64,
    #[serde(default)]
    ratio: f64,
}

/// How many placements of each page are written to the log in a run. A page is placed again at every change of
/// the window's size, so not all of them: the first few say what there is to say.
const PLACEMENTS_LOGGED: usize = 4;

/// One line in the log for a page's first placements: the box the dashboard asked for, what the dashboard saw as
/// it measured, and where the window says the page and the dashboard's own page then are. The page is laid by
/// the window's toolkit from numbers measured in the dashboard, two things that agree on Windows and were seen
/// not to on a Mac (the page over half of a section's tabs); this is what tells which of them is off, on a
/// system nobody here can look at.
fn note_placement<R: Runtime>(app: &AppHandle<R>, window: &tauri::Window<R>, page: Page, asked: [f64; 4], seen: Option<Seen>) {
    static LOGGED: [std::sync::atomic::AtomicUsize; 3] = [const { std::sync::atomic::AtomicUsize::new(0) }; 3];
    let slot = Page::ALL.iter().position(|p| *p == page).unwrap_or(0);
    if LOGGED[slot].fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= PLACEMENTS_LOGGED {
        return;
    }
    let scale = window.scale_factor().unwrap_or(1.0);
    let of = |label: &str| -> String {
        let Some(webview) = app.get_webview(label) else { return "none".into() };
        match (webview.position(), webview.size()) {
            (Ok(at), Ok(size)) => {
                let (at, size) = (at.to_logical::<f64>(scale), size.to_logical::<f64>(scale));
                format!("{:.1},{:.1} {:.1}x{:.1}", at.x, at.y, size.width, size.height)
            }
            _ => "not told".into(),
        }
    };
    let size = |size: tauri::Result<tauri::PhysicalSize<u32>>| size.map_or_else(|_| "not told".to_string(), |s| {
        let s = s.to_logical::<f64>(scale);
        format!("{:.1}x{:.1}", s.width, s.height)
    });
    let seen = seen.map_or_else(|| "not said".to_string(), |s| format!("viewport {:.1}x{:.1}, scrolled {:.1},{:.1}, {} screen pixels to one of its own", s.width, s.height, s.scroll_x, s.scroll_y, s.ratio));
    log::info!(
        "embed: {} asked for at {:.1},{:.1} {:.1}x{:.1} (the dashboard saw: {seen}); it is at {}; the dashboard's page is at {}; the window is {} inside, {} outside, scale {scale}",
        page.label(),
        asked[0],
        asked[1],
        asked[2],
        asked[3],
        of(page.label()),
        of(window.label()),
        size(window.inner_size()),
        size(window.outer_size()),
    );
}

/// Lay `page` over the dashboard at `rect` (x, y, width, height), the other pages put away. Says whether it has still
/// to be brought in ([`bring_in`]): a page that is being made, is still loading, or was never drawn at this size is
/// put out of sight first, at its size and shown, so that it draws there ([`OPENS_OUT_OF_SIGHT`]).
///
/// The keyboard goes to the page as it comes to be shown, and not each time it is placed again: the dashboard says
/// where its box is at every change of the window's size, and whenever a notice moves it, and each of those took the
/// keyboard back from wherever the person had put it.
fn lay<R: Runtime>(app: &AppHandle<R>, window: &tauri::Window<R>, page: Page, rect: [f64; 4]) -> Result<bool, String> {
    for other in Page::ALL {
        if other != page {
            put_away(app, other);
        }
    }
    keyboard_joins(app, crate::update::gui::DASHBOARD_LABEL, 0);
    let (position, size) = (LogicalPosition::new(rect[0], rect[1]), LogicalSize::new(rect[2], rect[3]));
    if let Some(webview) = app.get_webview(page.label()) {
        let here = SHOWN.load(SeqCst) == page.mark();
        let as_it_was = LOADED[page.slot()].load(SeqCst) && kept(&DRAWN)[page.slot()] == Some(whole(rect));
        if here || as_it_was || !OPENS_OUT_OF_SIGHT {
            place(&webview, position, size)?;
            kept(&DRAWN)[page.slot()] = Some(whole(rect));
            if !here {
                webview.show().map_err(|e| e.to_string())?;
                SHOWN.store(page.mark(), SeqCst);
                let _ = webview.set_focus();
                let _ = webview.eval(theme_script(theme(app)));
            }
            return Ok(false);
        }
        place(&webview, out_of_sight(rect), size)?;
        webview.show().map_err(|e| e.to_string())?;
        let _ = webview.eval(theme_script(theme(app)));
        // (the page put away just now may have had the keyboard: the dashboard's until this one is brought in)
        keyboard_home(app);
        return Ok(true);
    }
    if page == Page::Engines {
        if crate::engines::ui_url().is_none() {
            return Err("the engines are not running yet: they start with OAIY, or run oaiy-studio".into());
        }
    } else if dist(app, page).is_none() {
        return Err(format!("the {} page is not built: run `npm run build` in {}", page.folder(), match page {
            Page::Agent => "app/",
            _ => "platform/ui/",
        }));
    }
    let at = if OPENS_OUT_OF_SIGHT { out_of_sight(rect) } else { position };
    let webview = window.add_child(builder(app, page), at, size).map_err(|e| e.to_string())?;
    keyboard_joins(app, page.label(), page.mark() as usize);
    // (made where it was asked for, but on Linux, where it is only packed into the window)
    if cfg!(target_os = "linux") {
        place(&webview, position, size)?;
    }
    if page == Page::Agent {
        webkit_features(&webview, page);
    }
    if OPENS_OUT_OF_SIGHT {
        keyboard_home(app);
        return Ok(true);
    }
    // (in place at once: a webview takes no keyboard by being made, [`builder`], so it is given it here)
    kept(&DRAWN)[page.slot()] = Some(whole(rect));
    SHOWN.store(page.mark(), SeqCst);
    let _ = webview.set_focus();
    Ok(false)
}

/// Bring a page that opened out of sight to where it was asked for last, drawn, and give it the keyboard: the person
/// opened it.
fn bring_in<R: Runtime>(app: &AppHandle<R>, page: Page) -> Result<(), String> {
    let (Some(webview), Some(rect)) = (app.get_webview(page.label()), kept(&PLACE)[page.slot()]) else { return Ok(()) };
    place(&webview, LogicalPosition::new(rect[0], rect[1]), LogicalSize::new(rect[2], rect[3]))?;
    kept(&DRAWN)[page.slot()] = Some(whole(rect));
    SHOWN.store(page.mark(), SeqCst);
    let _ = webview.set_focus();
    Ok(())
}

/// Put a page away, where it is made (it goes on working: the agent answers texts and calls hidden).
fn put_away<R: Runtime>(app: &AppHandle<R>, page: Page) {
    if let Some(w) = app.get_webview(page.label()) {
        let _ = w.hide();
    }
    let _ = SHOWN.compare_exchange(page.mark(), 0, SeqCst, SeqCst);
}

/// The keyboard back with the dashboard, where it is in a page that is not showing (a page put away keeps it
/// otherwise, and every key goes to a page nobody can see). Asked on the main thread, after what was asked of the
/// pages before it. On Windows, where it was seen; a Mac and Linux are as they were.
fn keyboard_home<R: Runtime>(app: &AppHandle<R>) {
    #[cfg(windows)]
    {
        let _ = app.run_on_main_thread(keyboard::home);
    }
    #[cfg(not(windows))]
    let _ = app;
}

/// A webview takes its place in the round Tab makes ([`keyboard`]): the dashboard's is place 0, a page's its
/// [`Page::mark`]. Once each; on Windows.
fn keyboard_joins<R: Runtime>(app: &AppHandle<R>, label: &str, place: usize) {
    #[cfg(windows)]
    {
        let Some(webview) = app.get_webview(label) else { return };
        if !keyboard::claim(place) {
            return;
        }
        if let Err(e) = webview.with_webview(move |w| keyboard::join(w.controller(), place)) {
            log::warn!("embed: {label} is not in Tab's round (its webview was not reached: {e})");
        }
    }
    #[cfg(not(windows))]
    let _ = (app, label, place);
}

/// A page's webview put at `position`, `size` over the dashboard. On Linux Tauri packs a window's webviews into its
/// GTK box, one under another, where a webview's position and size are not its own to set (wry moves only one in a
/// `GtkFixed`): shown, the agent's took the whole window and left the dashboard a pixel high, its sidebar gone. There
/// the page is laid over the dashboard instead ([`overlay::place`]).
fn place<R: Runtime>(webview: &tauri::Webview<R>, position: LogicalPosition<f64>, size: LogicalSize<f64>) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        overlay::place(webview, position, size)
    }
    #[cfg(not(target_os = "linux"))]
    {
        webview.set_position(position).map_err(|e| e.to_string())?;
        webview.set_size(size).map_err(|e| e.to_string())
    }
}

/// Linux's placing of the pages: the dashboard's webview in a `GtkOverlay` where it was in the window's box (made the
/// first time a page is placed), and each page an overlay child of it, at its place by its margins and as large as
/// its size request.
#[cfg(target_os = "linux")]
mod overlay {
    use gtk::prelude::*;
    use tauri::{LogicalPosition, LogicalSize, Runtime};

    pub fn place<R: Runtime>(webview: &tauri::Webview<R>, position: LogicalPosition<f64>, size: LogicalSize<f64>) -> Result<(), String> {
        // (GTK's own pixels are logical ones)
        let (x, y) = (position.x.round() as i32, position.y.round() as i32);
        let (width, height) = (size.width.round().max(1.0) as i32, size.height.round().max(1.0) as i32);
        webview
            .with_webview(move |w| {
                let page: gtk::Widget = w.inner().upcast();
                if let Err(e) = over_the_dashboard(&page) {
                    log::warn!("embed: a page could not be laid over the dashboard: {e}");
                    return;
                }
                page.set_halign(gtk::Align::Start);
                page.set_valign(gtk::Align::Start);
                page.set_margin_start(x);
                page.set_margin_top(y);
                page.set_size_request(width, height);
            })
            .map_err(|e| e.to_string())
    }

    /// `page` an overlay child over the dashboard's webview, the overlay made where the dashboard was in the window's
    /// box the first time. (Each widget moved is held here while it is moved, so it is not let go.)
    fn over_the_dashboard(page: &gtk::Widget) -> Result<(), String> {
        let parent = page.parent().ok_or("it has no parent")?;
        if parent.is::<gtk::Overlay>() {
            return Ok(());
        }
        let vbox = parent.downcast::<gtk::Box>().map_err(|p| format!("it is in a {}, not the window's box", p.type_().name()))?;
        let children = vbox.children();
        let overlay = match children.iter().find_map(|c| c.clone().downcast::<gtk::Overlay>().ok()) {
            Some(overlay) => overlay,
            None => {
                let at = children.iter().position(|c| c != page && c.is::<webkit2gtk::WebView>()).ok_or("the window's box has no dashboard")?;
                let dashboard = &children[at];
                let overlay = gtk::Overlay::new();
                vbox.remove(dashboard);
                overlay.add(dashboard);
                vbox.pack_start(&overlay, true, true, 0);
                vbox.reorder_child(&overlay, at as i32);
                overlay.show();
                overlay
            }
        };
        vbox.remove(page);
        overlay.add_overlay(page);
        Ok(())
    }
}

/// Make the agent's page at startup, hidden: it answers texts and calls in
/// the background, before (or without) anyone opening it. Made on its own
/// thread (a webview is made on the main thread, which setup is still on).
pub fn preload<R: Runtime>(app: &AppHandle<R>) {
    let app = app.clone();
    std::thread::spawn(move || {
        let Some(window) = app.get_window("main") else { return };
        if app.get_webview(Page::Agent.label()).is_some() || dist(&app, Page::Agent).is_none() {
            return;
        }
        let _one = placing();
        keyboard_joins(&app, crate::update::gui::DASHBOARD_LABEL, 0);
        // (a pixel, and out of sight where that can be: it is hidden a moment after it is made)
        let rect = [0.0, 0.0, 1.0, 1.0];
        let at = if OPENS_OUT_OF_SIGHT { out_of_sight(rect) } else { LogicalPosition::new(0.0, 0.0) };
        match window.add_child(builder(&app, Page::Agent), at, LogicalSize::new(1.0, 1.0)) {
            Ok(webview) => {
                let _ = webview.hide();
                if cfg!(target_os = "linux") {
                    let _ = place(&webview, LogicalPosition::new(0.0, 0.0), LogicalSize::new(1.0, 1.0));
                }
                webkit_features(&webview, Page::Agent);
                keyboard_joins(&app, Page::Agent.label(), Page::Agent.mark() as usize);
                // The keyboard is the dashboard's as OAIY starts: a webview made here takes none ([`builder`]), and
                // should WebView2 have given it one all the same, it goes back.
                keyboard_home(&app);
            }
            Err(e) => log::warn!("the agent's page was not made at startup: {e}"),
        }
    });
}

/// What WebKitGTK leaves off and the agent's page needs: `navigator.storage`, and the origin-private file system its
/// projects and settings are kept in, with its files' writable streams and access handles. Without them the agent
/// stopped as it started ("undefined is not an object (evaluating 'navigator.storage.getDirectory')", WebKitGTK 2.52).
#[cfg(target_os = "linux")]
const WEBKIT_FEATURES: [&str; 4] = ["StorageAPI", "FileSystem", "FileSystemWritableStream", "AccessHandle"];

/// On Linux: [`WEBKIT_FEATURES`] turned on in a page's webview, and the page opened again to start with them (a page
/// has them from the document made after them, and its webview was made with the page loading). WebKitGTK names its
/// features from 2.42 and the bindings stop at 2.40, so the calls are found as OAIY runs ([`webkit::enable`]); with
/// an older WebKitGTK the page stays as it was, and the log says why. Elsewhere, nothing.
fn webkit_features<R: Runtime>(webview: &tauri::Webview<R>, page: Page) {
    #[cfg(target_os = "linux")]
    {
        let url = page.url();
        let made = webview.with_webview(move |w| {
            use webkit2gtk::glib::prelude::ObjectType;
            use webkit2gtk::WebViewExt;
            let view = w.inner();
            let Some(settings) = view.settings() else { return };
            // SAFETY: a `WebKitSettings` of this webview, alive while `settings` is, on the main thread.
            match unsafe { webkit::enable(settings.as_ptr().cast(), &WEBKIT_FEATURES) } {
                Ok(on) => {
                    log::info!("embed: {} has WebKitGTK's {} on", page.label(), on.join(", "));
                    view.load_uri(url.as_str());
                }
                Err(e) => log::warn!("embed: {} keeps WebKitGTK's defaults ({e}): the agent cannot keep its projects", page.label()),
            }
        });
        if let Err(e) = made {
            log::warn!("embed: {}'s WebKitGTK settings were not reached: {e}", page.label());
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (webview, page);
}

/// WebKitGTK's features by name (2.42 and later), its calls looked up in this process (it has WebKitGTK loaded).
#[cfg(target_os = "linux")]
mod webkit {
    use std::ffi::{c_char, c_int, c_void, CStr};

    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    /// The function `name` among those this process has loaded (glibc's `RTLD_DEFAULT` is a null handle), as an `F`.
    ///
    /// # Safety
    /// `F` must be the function's own type.
    unsafe fn find<F: Copy>(name: &CStr) -> Option<F> {
        let p = dlsym(std::ptr::null_mut(), name.as_ptr());
        (!p.is_null()).then(|| std::mem::transmute_copy::<*mut c_void, F>(&p))
    }

    /// `wanted` (features' identifiers) turned on in `settings`: those it had, or why none could be.
    ///
    /// # Safety
    /// `settings` must be a live `WebKitSettings*`, and this on the thread that owns it.
    pub unsafe fn enable(settings: *mut c_void, wanted: &[&str]) -> Result<Vec<String>, String> {
        type All = unsafe extern "C" fn() -> *mut c_void;
        type Len = unsafe extern "C" fn(*mut c_void) -> usize;
        type Get = unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void;
        type Identifier = unsafe extern "C" fn(*mut c_void) -> *const c_char;
        type Set = unsafe extern "C" fn(*mut c_void, *mut c_void, c_int);
        type Unref = unsafe extern "C" fn(*mut c_void);
        let (Some(all), Some(len), Some(get), Some(identifier), Some(set), Some(unref)) = (
            find::<All>(c"webkit_settings_get_all_features"),
            find::<Len>(c"webkit_feature_list_get_length"),
            find::<Get>(c"webkit_feature_list_get"),
            find::<Identifier>(c"webkit_feature_get_identifier"),
            find::<Set>(c"webkit_settings_set_feature_enabled"),
            find::<Unref>(c"webkit_feature_list_unref"),
        ) else {
            return Err("this WebKitGTK is older than 2.42, which names its features".into());
        };
        let list = all();
        let mut on = Vec::new();
        for i in 0..len(list) {
            let feature = get(list, i);
            let id = CStr::from_ptr(identifier(feature)).to_string_lossy();
            if wanted.contains(&id.as_ref()) {
                set(settings, feature, 1);
                on.push(id.into_owned());
            }
        }
        unref(list);
        if on.len() < wanted.len() {
            return Err(format!("it has {} of {}", on.join(", "), wanted.join(", ")));
        }
        Ok(on)
    }
}

/// On Linux, before the first webview is made: `SharedArrayBuffer` for the agent's page, which its code sandbox blocks
/// on. WebKitGTK gives a page one only where its JavaScript engine is told to (`JSC_useSharedArrayBuffer`, read by
/// each web process it starts), even a page it calls cross-origin isolated (the agent's from [`AGENT_PORT`]). Every
/// page in the window has it then, all of them this computer's own. A value already set is left alone.
pub fn linux_shared_memory() {
    #[cfg(target_os = "linux")]
    if std::env::var_os("JSC_useSharedArrayBuffer").is_none() {
        std::env::set_var("JSC_useSharedArrayBuffer", "1");
    }
}

/// One page made or placed at a time: two quick resizes must not both make a webview.
fn placing() -> std::sync::MutexGuard<'static, ()> {
    static PLACING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    PLACING.lock().unwrap_or_else(|e| e.into_inner())
}

/// A page's webview, as `show_embedded` and `preload` make it.
fn builder<R: Runtime>(app: &AppHandle<R>, page: Page) -> WebviewBuilder<R> {
    let (for_navigation, for_windows) = (app.clone(), app.clone());
    let first = theme(app);
    // The engines' pages are theirs, at their own address (told the theme in it); the desktop's two know its address and token.
    let made = if page == Page::Engines {
        let mut url = page.url();
        url.query_pairs_mut().append_pair("in", "oaiy").append_pair("theme", first);
        WebviewBuilder::new(page.label(), WebviewUrl::External(url))
    } else if page == Page::Agent && agent_http_origin().is_some() {
        // (a Mac's or Linux's agent, from its own port: an address, not a scheme the window serves)
        WebviewBuilder::new(page.label(), WebviewUrl::External(page.url())).initialization_script(&desktop_script(first, page))
    } else {
        WebviewBuilder::new(page.label(), WebviewUrl::CustomProtocol(page.url())).initialization_script(&desktop_script(first, page))
    };
    // The agent answers calls and texts while it is hidden (`preload`), or while the window is behind another or
    // minimized. WebKit slows such a page down and then suspends it (BROWSER_ARGS are WebView2's alone): on a Mac a
    // whole call came to it a minute after the call had ended, in one burst, as the person looked at it. On macOS 14
    // and later this keeps it running as if it were shown. (The other pages have nothing to do while hidden.)
    let made = if page == Page::Agent { made.background_throttling(tauri::utils::config::BackgroundThrottlingPolicy::Disabled) } else { made };
    // The engines' page offers files to save (its models file, a line of speech, a song, a 3D model): WebKit, with no
    // one to hand a download to, cancels it, and the button did nothing. WebView2 saves them by itself.
    #[cfg(not(windows))]
    let made = if page == Page::Engines { made.on_download(engines_download) } else { made };
    made
        .additional_browser_args(BROWSER_ARGS)
        // Being made takes the keyboard from nobody (wry gives a new webview the keyboard unless told not to: the
        // agent's, made hidden as OAIY starts, had it, and no key reached the dashboard until someone clicked in it).
        // A page is given it when it is shown ([`lay`], [`bring_in`]).
        .focused(false)
        // Where the page has drawn nothing yet, the dashboard's ground for the theme, kept in step by `set_theme`.
        .background_color(ground(first))
        // A page that (re)loads starts in the theme it was made with: tell it the one now. The flow editor's is handed
        // the intent asked for while it loaded (a run's flow to open), and is told one at once after.
        .on_page_load(move |webview, payload| {
            if page == Page::Flows {
                FLOWS_LOADED.store(payload.event() == PageLoadEvent::Finished, std::sync::atomic::Ordering::SeqCst);
            }
            if payload.event() == PageLoadEvent::Finished {
                // (loaded: it may be brought in once it has had a moment to draw, `show_embedded`)
                LOADED[page.slot()].store(true, SeqCst);
                let _ = webview.eval(theme_script(theme(webview.app_handle())));
                if page == Page::Flows {
                    if let Some(script) = PENDING_FLOWS_INTENT.lock().unwrap_or_else(|e| e.into_inner()).take() {
                        let _ = webview.eval(script);
                    }
                }
            }
        })
        .on_navigation(move |url| {
            // (the Engines page's link to one of the dashboard's pages: shown there, and nothing is loaded here)
            if let Some(view) = asked_view(page, &page.url(), url) {
                if !crate::control::show_view(view) {
                    log::warn!("embed: {} asked the window for a page it does not have", page.label());
                }
                return false;
            }
            if own(page, url) {
                return true;
            }
            // WebView2 asks this for the page itself alone, so a link followed in it opens in the browser. WebKit asks it for
            // every frame too, and wry does not say which: a frame that loads another site (a model's reply, a plugin's screen
            // in the flow editor's sandbox setting its `location`) must not open the browser with nobody's click. There the
            // navigation is only stopped; the pages' links open outside through `on_new_window` (they are `target="_blank"`).
            if cfg!(windows) {
                open_outside(&for_navigation, url);
            }
            false
        })
        .on_new_window(move |url, _features| {
            // (the Engines page's link to one of the dashboard's pages, opened with Shift, Ctrl or the middle button:
            // shown in the window as a plain click shows it, never a window of its own on an address nothing serves)
            if let Some(view) = asked_view(page, &page.url(), &url) {
                if !crate::control::show_view(view) {
                    log::warn!("embed: {} asked the window for a page it does not have", page.label());
                }
                return NewWindowResponse::Deny;
            }
            // (a paired computer's result the engines' Playground opens, `/api/play/node/…`, is shown by the system's
            // browser on this computer, as this computer's own outputs are through /api/open-file: never a bare window
            // of the app's own with no way back)
            if own(page, &url) && !url.path().starts_with("/api/play/node/") {
                return NewWindowResponse::Allow;
            }
            open_outside(&for_windows, &url);
            NewWindowResponse::Deny
        })
}

/// The engines' page's downloads, saved where a browser saves them: the Downloads folder, under the name the page
/// gives, numbered rather than over a file there (wry's choice). Its own files alone. As each ends the page is told
/// where it went, or that it failed (`__oaiyDownloaded`).
#[cfg(not(windows))]
fn engines_download<R: Runtime>(webview: tauri::Webview<R>, event: tauri::webview::DownloadEvent<'_>) -> bool {
    use tauri::webview::DownloadEvent;
    // (where each is going: a Mac says no path as one ends)
    static SAVING: std::sync::Mutex<Vec<(String, PathBuf)>> = std::sync::Mutex::new(Vec::new());
    let saving = || SAVING.lock().unwrap_or_else(|e| e.into_inner());
    match event {
        DownloadEvent::Requested { url, destination } => {
            if !own(Page::Engines, &url) {
                log::warn!("embed: the engines' page asked to save {url}, which is not its own: not saved");
                return false;
            }
            log::info!("embed: saving {url} to {}", destination.display());
            saving().push((url.to_string(), destination.clone()));
            true
        }
        DownloadEvent::Finished { url, path, success } => {
            let went = {
                let mut s = saving();
                let at = s.iter().position(|(u, _)| u == url.as_str());
                at.map(|i| s.remove(i).1)
            };
            let path = path.or(went);
            if !success {
                log::warn!("embed: saving {url} failed");
            }
            let script = format!(
                "window.__oaiyDownloaded && window.__oaiyDownloaded({}, {success})",
                serde_json::Value::from(path.map(|p| p.display().to_string()))
            );
            let _ = webview.eval(script);
            true
        }
        _ => true,
    }
}

/// Hide the embedded pages (the dashboard shows one of its own).
#[tauri::command]
pub async fn hide_embedded<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    // One placing at a time, a hiding too: between a showing's last look at what is asked and the keyboard it gives the
    // page, a hiding left the keyboard in a page out of sight; and one that stalled after it had forgotten which page
    // shows could hide a page asked for again meanwhile, which then read as shown and was never shown.
    let _one = placing();
    ASKS.fetch_add(1, SeqCst);
    WANTED.store(0, SeqCst);
    SHOWN.store(0, SeqCst);
    for page in Page::ALL {
        if let Some(w) = app.get_webview(page.label()) {
            w.hide().map_err(|e| e.to_string())?;
        }
    }
    // A page that had the keyboard keeps it hidden: back to the dashboard, to what was last used in it.
    keyboard_home(&app);
    Ok(())
}

/// Tab's round through the window's webviews, and the keyboard's way home, on Windows.
///
/// Each webview is a WebView2 of its own, and WebView2 keeps Tab inside the one that has the keyboard: past its last
/// stop it asks its host where to go next (`MoveFocusRequested`), and where the host says nothing it starts again at
/// its own first stop. So Tab went round the dashboard and never into the page beside it, and from inside a page no
/// key came back. Here the host answers: from the dashboard, on into the page that is showing (at its first stop for
/// Tab, its last for Shift+Tab); from a page, on into the dashboard the same way. The order a person gets is the
/// sidebar, the header, the page, the sidebar again.
///
/// What is not handled, and why:
/// - Leaving OAIY for another program and coming back (Alt+Tab, the taskbar) lands the keyboard in the dashboard,
///   also where it was in a page. That is wry's doing: it watches the window for the webview made first (the
///   dashboard's), and gives that webview the keyboard whenever the window gets it back (wry 0.55.1,
///   `src/webview2/mod.rs`, `parent_subclass_proc` on `WM_SETFOCUS`). Left as it is: undoing it means remembering
///   which webview had the keyboard and moving it back after wry has moved it.
/// - A Mac and Linux have none of this module, and no opening out of sight (`OPENS_OUT_OF_SIGHT`). They have what is
///   not Windows' alone: a webview being made takes the keyboard from nobody (`focused(false)`), a page is given
///   the keyboard when it comes to be shown and only then, and the ground under a page is the dashboard's
///   (`ground`). Tab's round there is WKWebView's and WebKitGTK's own, and neither was run.
/// - A dialog of a page's own keeps Tab inside it, as a dialog should; the round is back when it closes.
///
/// The calls are WebView2's own COM interfaces through `webview2-com`, the crate Tauri and wry are built on (named
/// in Cargo.toml for this; the one copy, since Tauri hands out the controller as that crate's type). Everything here
/// runs on the main thread, which is where a controller lives: `join` inside `with_webview`, `home` through
/// `run_on_main_thread`, and the answer where WebView2 asks for it.
#[cfg(windows)]
mod keyboard {
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::mem::ManuallyDrop;
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

    // Why the keyboard moves: by the program, by Tab, by Shift+Tab.
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2Controller, ICoreWebView2MoveFocusRequestedEventHandler, COREWEBVIEW2_MOVE_FOCUS_REASON as Reason,
        COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT as NEXT, COREWEBVIEW2_MOVE_FOCUS_REASON_PREVIOUS as PREVIOUS,
        COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC as PROGRAMMATIC,
    };
    use webview2_com::MoveFocusRequestedEventHandler;

    // Two of user32's, by name (a window's handle is a pointer): which of this thread's windows has the keyboard, and
    // whether one window is inside another.
    #[link(name = "user32")]
    extern "system" {
        fn GetFocus() -> *mut c_void;
        fn IsChild(parent: *mut c_void, window: *mut c_void) -> i32;
    }

    /// The places in the round: the dashboard's (0), then each page's (its `Page::mark`).
    const PLACES: usize = 4;

    thread_local! {
        /// The webviews' controllers by place, on the main thread: the only one a controller is used on, and any
        /// other finds none here and does nothing. Each is a reference held for as long as OAIY runs (a webview here
        /// is never destroyed), and not let go as the thread ends, when WebView2 itself is on its way out.
        static CONTROLLERS: RefCell<[Option<ManuallyDrop<ICoreWebView2Controller>>; PLACES]> =
            const { RefCell::new([None, None, None, None]) };
    }
    static CLAIMED: [AtomicBool; PLACES] = [const { AtomicBool::new(false) }; PLACES];

    /// Take `place` for a webview that is about to join: false where one has it already.
    pub fn claim(place: usize) -> bool {
        CLAIMED.get(place).is_some_and(|c| !c.swap(true, SeqCst))
    }

    /// The controller of the webview at `place`, where one has joined: a reference of the caller's own, let go as it
    /// is dropped, so nothing of the table is held while WebView2 is called.
    fn at(place: usize) -> Option<ICoreWebView2Controller> {
        CONTROLLERS.with_borrow(|all| all.get(place)?.as_deref().cloned())
    }

    /// The webview whose controller this is joins the round at `place`: it is told who answers when Tab leaves it,
    /// and its controller is kept. `controller` is the one Tauri hands out (`with_webview`), a reference of its own.
    pub fn join(controller: ICoreWebView2Controller, place: usize) {
        if place >= PLACES {
            return;
        }
        let mut token = 0i64;
        // SAFETY: a live controller on its own (the main) thread; WebView2 takes a reference of its own to the answer.
        if let Err(e) = unsafe { controller.add_MoveFocusRequested(&answer(place), &mut token) } {
            log::warn!("embed: WebView2 did not take the answer to Tab leaving a webview ({e}): Tab stays inside it");
            return;
        }
        CONTROLLERS.with_borrow_mut(|all| all[place] = Some(ManuallyDrop::new(controller)));
    }

    /// Whether the window with the keyboard is `controller`'s webview: its frame (a child of OAIY's window), or
    /// anything inside it.
    fn has_keyboard(controller: &ICoreWebView2Controller, focus: *mut c_void) -> bool {
        let mut frame = Default::default();
        // SAFETY: a live controller, on its own thread, writes its window's handle.
        if unsafe { controller.ParentWindow(&mut frame) }.is_err() || frame.0.is_null() {
            return false;
        }
        // SAFETY: IsChild only reads.
        focus == frame.0 || unsafe { IsChild(frame.0, focus) } != 0
    }

    /// The keyboard to the dashboard, where it is neither there nor in the page that is showing: in a page that was
    /// put away, or in OAIY's window itself. Nothing where no window of OAIY's has the keyboard (another program
    /// does: it is not taken from it). On the main thread.
    pub fn home() {
        let Some(dashboard) = at(0) else { return };
        // SAFETY: GetFocus only reads.
        let focus = unsafe { GetFocus() };
        if focus.is_null() || has_keyboard(&dashboard, focus) {
            return;
        }
        let shown = super::SHOWN.load(SeqCst) as usize;
        if shown != 0 && at(shown).is_some_and(|page| has_keyboard(&page, focus)) {
            return;
        }
        // SAFETY: a live controller, on its own thread.
        let _ = unsafe { dashboard.MoveFocus(PROGRAMMATIC) };
    }

    /// Tab (or Shift+Tab) has left the webview at `place`: where the keyboard goes, by `reason`. From the dashboard,
    /// into the page that is showing, where one is (else the dashboard keeps it, going round as before); from a
    /// page, into the dashboard. None where it stays.
    fn onward(place: usize, shown: usize, reason: Reason) -> Option<usize> {
        if reason != NEXT && reason != PREVIOUS {
            return None;
        }
        match place {
            0 => (shown != 0).then_some(shown),
            _ => Some(0),
        }
    }

    /// What WebView2 calls when Tab leaves the webview at `place` (0 the dashboard, else a page): the keyboard goes
    /// [`onward`], and WebView2 is told it was answered. Where it goes nowhere, nothing is said, and WebView2 starts
    /// again at the webview's own first stop, as it always did.
    fn answer(place: usize) -> ICoreWebView2MoveFocusRequestedEventHandler {
        MoveFocusRequestedEventHandler::create(Box::new(move |_webview, args| {
            let Some(args) = args else { return Ok(()) };
            let mut reason = PROGRAMMATIC;
            // SAFETY: WebView2's own arguments, in its own call, on its own thread.
            if unsafe { args.Reason(&mut reason) }.is_err() {
                return Ok(());
            }
            let Some(to) = onward(place, super::SHOWN.load(SeqCst) as usize, reason) else { return Ok(()) };
            let Some(next) = at(to) else { return Ok(()) };
            // SAFETY: as above, and `next` is a live controller on the same thread.
            unsafe {
                // (answered: WebView2 does not start again at this webview's own first stop)
                let _ = args.SetHandled(true);
                let _ = next.MoveFocus(reason);
            }
            Ok(())
        }))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2MoveFocusRequestedEventArgs;

        #[test]
        fn tab_goes_from_the_dashboard_into_the_page_showing_and_from_a_page_back() {
            // From the dashboard (place 0): into the page that is showing, by Tab or Shift+Tab; nowhere with none showing.
            assert_eq!(onward(0, 2, NEXT), Some(2));
            assert_eq!(onward(0, 3, PREVIOUS), Some(3));
            assert_eq!(onward(0, 0, NEXT), None, "no page showing: the dashboard keeps the keyboard, as before");
            // From a page: into the dashboard, whichever page is showing.
            for page in 1..=3 {
                assert_eq!(onward(page, page, NEXT), Some(0));
                assert_eq!(onward(page, 0, PREVIOUS), Some(0), "a page that was put away with the keyboard gives it up too");
            }
            // The keyboard moved by the program is nobody's Tab.
            assert_eq!(onward(1, 1, PROGRAMMATIC), None);
        }

        #[test]
        fn a_call_with_nothing_to_read_is_answered_and_does_nothing() {
            // The answer as WebView2 holds it, called through its own interface with no arguments at all.
            let answer = answer(0);
            // SAFETY: a COM object made here, called as WebView2 calls one, with nothing for it to read.
            let called = unsafe { answer.Invoke(None::<&ICoreWebView2Controller>, None::<&ICoreWebView2MoveFocusRequestedEventArgs>) };
            assert!(called.is_ok());
            // There are four places in the round and no fifth, and no webview has joined in a test.
            assert!(!claim(9), "there is no such place");
            assert!(at(9).is_none() && at(0).is_none());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ground_under_a_page_is_the_dashboards_own_for_the_theme() {
        // The dashboard says its ground once for each theme, `--canvas` in its stylesheet: under `:root` for the dark
        // theme and under `:root[data-theme='light']` for the light. A page's webview is given the same.
        let css = include_str!("../../src/styles.css").replace("\r\n", "\n");
        let canvas = |opens: &str| -> String {
            let from = css.find(&format!("\n{opens} {{\n")).unwrap_or_else(|| panic!("styles.css has a block that opens {opens:?}"));
            let block = &css[from..from + css[from..].find("\n}\n").expect("the block closes")];
            let at = block.find("--canvas:").unwrap_or_else(|| panic!("{opens:?} says --canvas"));
            block[at + "--canvas:".len()..].split(';').next().unwrap_or("").trim().to_ascii_lowercase()
        };
        let hex = |c: Color| format!("#{:02x}{:02x}{:02x}", c.0, c.1, c.2);
        assert_eq!(hex(ground("dark")), canvas(":root"));
        assert_eq!(hex(ground("light")), canvas(":root[data-theme='light']"));
        // Opaque (WebView2 takes a colour whole or not at all), and dark where the theme is not said.
        assert_eq!((ground("dark").3, ground("light").3), (255, 255));
        assert_eq!(hex(ground("")), hex(ground("dark")));
    }

    #[test]
    fn the_engines_page_asks_the_window_for_a_dashboard_page_at_an_address_that_is_never_loaded() {
        let mine: Url = "http://127.0.0.1:17860/?in=oaiy&theme=light".parse().unwrap();
        let asked = |page: Page, url: &str| asked_view(page, &mine, &url.parse::<Url>().unwrap()).map(str::to_string);
        // Its link to Settings → Storage: an asking, by the page's name.
        assert_eq!(asked(Page::Engines, "http://127.0.0.1:17860/oaiy-window/show/storage").as_deref(), Some("storage"));
        // Anything under the path is an asking, and so is not loaded; which names are shown is the list `ui_open` uses.
        assert_eq!(asked(Page::Engines, "http://127.0.0.1:17860/oaiy-window/show/plugin:demo:home?x=1#y").as_deref(), Some("plugin:demo:home"));
        assert!(crate::control::own_view("storage") && crate::control::own_view("settings"));
        for not_a_page in ["plugin:demo:home", "agent-settings", "", "storage/", "Storage", "../../etc"] {
            assert!(!crate::control::own_view(not_a_page), "{not_a_page:?}");
            assert!(!crate::control::show_view(not_a_page), "{not_a_page:?}");
        }
        // The page's own other addresses load as before; another program's port, another site and another page of
        // the window are not askings.
        assert_eq!(asked(Page::Engines, "http://127.0.0.1:17860/?in=oaiy#get"), None);
        assert_eq!(asked(Page::Engines, "http://127.0.0.1:17860/api/oaiy-window/show/storage"), None);
        assert_eq!(asked(Page::Engines, "http://127.0.0.1:17861/oaiy-window/show/storage"), None);
        assert_eq!(asked(Page::Engines, "https://127.0.0.1:17860/oaiy-window/show/storage"), None);
        assert_eq!(asked(Page::Engines, "http://example.com/oaiy-window/show/storage"), None);
        assert_eq!(asked(Page::Agent, "http://127.0.0.1:17860/oaiy-window/show/storage"), None);
    }

    #[test]
    fn a_page_opens_out_of_sight_at_its_own_size() {
        let rect = [222.0, 48.0, 1058.4, 771.6];
        assert_eq!(whole(rect), (1058, 772));
        let at = out_of_sight(rect);
        assert!(at.x + rect[2] < 0.0, "wholly left of the window's edge, where a child window is cut off");
        assert_eq!(at.y, rect[1]);
        // A page is one number where it is kept as one, and no page is none of them.
        assert_eq!(Page::ALL.map(Page::mark), [1, 2, 3]);
        assert_eq!(Page::ALL.map(Page::slot), [0, 1, 2]);
        // It is given a while to load and a moment to draw, and the moment is the shorter.
        assert!(FIRST_PICTURE < OPENING_DEADLINE && OPENING_DEADLINE <= Duration::from_secs(2));
    }

    #[test]
    fn each_webview_is_given_the_embedder_policy_it_isolates_a_page_for() {
        // Chromium's (Windows) own value, which lets the page load other sites' files without credentials.
        assert_eq!(embedder_policy(true), "credentialless");
        // WebKit (a Mac, Linux) does not know that one, and with it the page is not isolated: no code sandbox.
        assert_eq!(embedder_policy(false), "require-corp");
    }

    #[test]
    fn the_agents_port_is_the_one_the_gateway_lets_in() {
        // The engines' gateway names the origin in its own crate (it lets the page in); it is this port's.
        assert_eq!(oaiy_studio::AGENT_HTTP_ORIGIN, format!("http://127.0.0.1:{AGENT_PORT}"));
        // Not held (nothing here served it), the page is at its scheme and the origin is nobody's.
        assert_eq!(agent_http_origin(), None);
        assert!(!crate::http::is_embedded_origin(oaiy_studio::AGENT_HTTP_ORIGIN));
        assert!(!crate::backup::routes::is_agent_origin(oaiy_studio::AGENT_HTTP_ORIGIN));
    }

    #[test]
    fn a_page_on_127_0_0_1_owns_its_port_alone() {
        let mine: Url = "http://127.0.0.1:17974/index.html".parse().unwrap();
        assert!(own_at(&mine, &"http://127.0.0.1:17974/softn/x.js".parse().unwrap()));
        // Another program's port on this computer would be handed the desktop's token by the window's script.
        for other in ["http://127.0.0.1:7860/", "http://127.0.0.1/", "http://localhost:17974/", "https://127.0.0.1:17974/"] {
            assert!(!own_at(&mine, &other.parse().unwrap()), "{other}");
        }
        let scheme: Url = "oaiy://localhost/index.html".parse().unwrap();
        assert!(own_at(&scheme, &"oaiy://localhost/assets/a.js".parse().unwrap()));
        assert!(!own_at(&scheme, &"oaiyflows://localhost/app.html".parse().unwrap()));
        assert!(own_at(&scheme, &"blob:oaiy://localhost/1".parse().unwrap()));
    }

    #[test]
    fn a_page_file_is_served_isolated_from_its_folder_alone() {
        let root = std::env::temp_dir().join(format!("oaiy-embed-serve-{}", std::process::id()));
        std::fs::create_dir_all(root.join("assets")).unwrap();
        std::fs::write(root.join("index.html"), "<!doctype html>").unwrap();
        std::fs::write(root.join("assets/a.js"), "1").unwrap();
        let start = serve_path(Some(&root), Page::Agent, "/");
        assert_eq!(start.status(), StatusCode::OK);
        assert_eq!(start.headers()["Cross-Origin-Opener-Policy"], "same-origin");
        assert_eq!(start.headers()["Cross-Origin-Embedder-Policy"], embedder_policy(cfg!(windows)));
        assert_eq!(start.body(), b"<!doctype html>");
        assert_eq!(serve_path(Some(&root), Page::Agent, "/assets/a.js").headers()[header::CONTENT_TYPE], "text/javascript; charset=utf-8");
        for outside in ["/../x", "/assets/../../x", "/%2e%2e/x", "/nope.js"] {
            assert_eq!(serve_path(Some(&root), Page::Agent, outside).status(), StatusCode::NOT_FOUND, "{outside}");
        }
        assert_eq!(serve_path(None, Page::Agent, "/").status(), StatusCode::NOT_FOUND);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_embedded_pages_are_named_and_have_their_schemes() {
        use crate::http::is_embedded_origin;
        for page in Page::SERVED {
            let url = page.url();
            // `Url::origin()` is opaque for a custom scheme (the pages' address off Windows); scheme and host are what a browser sends.
            let origin = format!("{}://{}", url.scheme(), url.host_str().expect("a host"));
            assert!(is_embedded_origin(&origin), "{url}");
        }
        assert!(!is_embedded_origin("http://evil.localhost"));
        assert!(!is_embedded_origin("https://oaiy.com"));
        assert_eq!(Page::parse("agent"), Some(Page::Agent));
        assert_eq!(Page::parse("nope"), None);
    }

    #[test]
    fn the_browser_service_is_told_the_origins_these_windows_really_have() {
        // The Playwright server lets in the origins the registry gives it (`http::embedded_window_origins`),
        // and only these windows make requests from them. They must be the very addresses the windows
        // are opened at, or browser nodes silently stop working in them (a page opened at an https
        // address, a scheme renamed…). `Url::origin()` is opaque for a custom scheme; scheme and host are what a browser sends.
        assert_eq!(crate::http::EMBEDDED_SCHEMES, [AGENT_SCHEME, FLOWS_SCHEME]);
        let told = crate::http::embedded_window_origins(cfg!(windows));
        let opened: Vec<String> = Page::SERVED
            .iter()
            .map(|page| {
                let url = page.url();
                format!("{}://{}", url.scheme(), url.host_str().expect("a host"))
            })
            .collect();
        assert_eq!(told, opened);
    }

    #[test]
    fn a_page_is_told_its_theme() {
        assert_eq!(parse_theme("light"), Some("light"));
        assert_eq!(parse_theme("sepia"), None);
        assert!(desktop_script("light", Page::Flows).contains(r#"theme: "light""#));
        assert_eq!(theme_script("dark"), r#"window.__oaiySetTheme ? window.__oaiySetTheme("dark") : (window.__OAIY_THEME__ = "dark");"#);
    }

    #[test]
    fn only_the_page_at_its_own_address_is_told_where_the_desktop_is() {
        for page in [Page::Agent, Page::Flows] {
            let url = page.url();
            let own = format!("{}://{}", url.scheme(), url.host_str().unwrap()) + &url.port().map(|p| format!(":{p}")).unwrap_or_default();
            let script = desktop_script("dark", page);
            assert!(script.starts_with(&format!("if (location.protocol + '//' + location.host === {own:?}) window.__OAIY_DESKTOP__ =")), "{script}");
        }
    }

    #[test]
    fn only_the_agents_page_is_given_the_restore_secret() {
        let secret = crate::backup::agent::page_token();
        assert!(secret.len() >= 32, "a real secret");
        assert!(desktop_script("dark", Page::Agent).contains(&format!("backupToken: {secret:?}")), "the Agent's page has it");
        assert!(!desktop_script("dark", Page::Flows).contains("backupToken") && !desktop_script("dark", Page::Flows).contains(secret));
        assert!(!desktop_script("dark", Page::Engines).contains(secret));
    }

    #[test]
    fn the_flow_editor_is_handed_a_flow_to_open_by_its_id() {
        assert_eq!(
            flows_intent_script("openFlow", "morning-greeting_2"),
            r#"window.__oaiyFlowsIntent ? window.__oaiyFlowsIntent("openFlow", "morning-greeting_2") : (window.__OAIY_FLOWS_INTENTS__ = (window.__OAIY_FLOWS_INTENTS__ || []).concat([["openFlow", "morning-greeting_2"]]));"#
        );
        // Only an id goes in the script: anything else is refused before one is made.
        for bad in ["a\"); alert(1); (\"", "../x", "", "a b"] {
            assert!(!crate::bridge::FlowStore::valid_id(bad), "{bad}");
        }
        assert_eq!(FLOWS_INTENTS, ["openFlow"]);
    }

    #[test]
    fn the_agent_is_handed_an_intent_by_name_only() {
        assert_eq!(
            intent_script("answerWithOaiy"),
            r#"window.__oaiyIntent ? window.__oaiyIntent("answerWithOaiy") : (window.__OAIY_INTENTS__ = (window.__OAIY_INTENTS__ || []).concat(["answerWithOaiy"]));"#
        );
        assert_eq!(
            intent_script("setupWithAgent"),
            r#"window.__oaiyIntent ? window.__oaiyIntent("setupWithAgent") : (window.__OAIY_INTENTS__ = (window.__OAIY_INTENTS__ || []).concat(["setupWithAgent"]));"#
        );
        assert!(AGENT_INTENTS.contains(&"answerWithOaiy"));
        assert!(AGENT_INTENTS.contains(&"setupWithAgent"));
        assert!(!AGENT_INTENTS.contains(&"alert(1)"));
        assert!(!AGENT_INTENTS.contains(&"setupwithagent"));
    }

    #[test]
    fn files_are_typed() {
        assert_eq!(mime("/zipp/engine.wasm"), "application/wasm");
        assert_eq!(mime("/index.html"), "text/html; charset=utf-8");
        assert_eq!(mime("/assets/main.js"), "text/javascript; charset=utf-8");
    }

    /// A folder under the temp folder for one test, removed when it is dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let dir = std::env::temp_dir().join(format!("oaiy-embed-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("a scratch folder");
            Scratch(dir)
        }

        fn path(&self, relative: &str) -> PathBuf {
            self.0.join(relative)
        }

        fn folder(&self, relative: &str) -> PathBuf {
            let dir = self.path(relative);
            std::fs::create_dir_all(&dir).expect("a folder");
            dir
        }

        fn file(&self, relative: &str) {
            let file = self.path(relative);
            std::fs::create_dir_all(file.parent().expect("a parent")).expect("a folder");
            std::fs::write(file, "<!doctype html>").expect("a file");
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A stand-in for `platform/desktop/src-tauri` in a scratch folder, so the pages' build folders (`../../../app/dist`, `../../ui/dist`) exist to be found.
    fn repo(scratch: &Scratch) -> PathBuf {
        scratch.folder("repo/platform/desktop/src-tauri")
    }

    fn start_of(page: Page) -> &'static str {
        page.start().trim_start_matches('/')
    }

    #[test]
    fn the_bundle_lists_the_folders_the_pages_are_looked_for_in() {
        // tauri.conf.json's `bundle.resources` is where the installer's pages come from: `resources/<folder>` for each page this desktop serves.
        let config: serde_json::Value = serde_json::from_str(include_str!("../tauri.conf.json")).expect("tauri.conf.json is JSON");
        let listed: Vec<&str> = config["bundle"]["resources"].as_array().expect("a list of resources").iter().filter_map(|r| r.as_str()).collect();
        for page in Page::SERVED {
            let entry = format!("resources/{}", page.folder());
            assert!(listed.contains(&entry.as_str()), "bundle.resources lists {entry}, which the {} page is looked for in: {listed:?}", page.folder());
        }
    }

    #[test]
    fn a_page_in_the_bundle_is_found_where_tauri_puts_resources() {
        let scratch = Scratch::new("bundle");
        let install = scratch.folder("install");
        let repo = repo(&scratch);
        for page in Page::SERVED {
            scratch.file(&format!("install/resources/{}/{}", page.folder(), start_of(page)));
        }
        for page in Page::SERVED {
            let found = with_start_document(dist_candidates(None, Some(install.clone()), &repo, page, false), page);
            assert_eq!(found, Some(install.join("resources").join(page.folder())), "the {} page", page.folder());
        }
    }

    #[test]
    fn a_folder_beside_the_resources_is_not_where_the_pages_are() {
        // The folder among the resources (`<resource dir>/app`) is not what `resources/app` in tauri.conf.json makes: Tauri keeps the `resources/`.
        let scratch = Scratch::new("flat");
        let install = scratch.folder("install");
        let repo = repo(&scratch);
        for page in Page::SERVED {
            scratch.file(&format!("install/{}/{}", page.folder(), start_of(page)));
            assert_eq!(with_start_document(dist_candidates(None, Some(install.clone()), &repo, page, false), page), None);
        }
    }

    #[test]
    fn an_empty_folder_among_the_resources_does_not_hide_the_pages_build() {
        // What `build.rs` leaves when nothing was staged: the folders, empty.
        let scratch = Scratch::new("empty");
        let install = scratch.folder("install");
        let repo = repo(&scratch);
        scratch.folder("install/resources/app");
        scratch.folder("install/resources/flows");
        scratch.file("repo/app/dist/index.html");
        scratch.file("repo/platform/ui/dist/app.html");
        for (page, built) in [(Page::Agent, "../../../app/dist"), (Page::Flows, "../../ui/dist")] {
            for debug in [false, true] {
                let found = with_start_document(dist_candidates(None, Some(install.clone()), &repo, page, debug), page);
                assert_eq!(found, Some(repo.join(built)), "the {} page, debug {debug}", page.folder());
            }
        }
    }

    #[test]
    fn an_installed_build_takes_the_bundle_and_a_debug_build_its_own_build_folders() {
        let scratch = Scratch::new("order");
        let install = scratch.folder("install");
        let repo = repo(&scratch);
        for page in Page::SERVED {
            scratch.file(&format!("install/resources/{}/{}", page.folder(), start_of(page)));
        }
        scratch.file("repo/app/dist/index.html");
        scratch.file("repo/platform/ui/dist/app.html");
        for (page, built) in [(Page::Agent, "../../../app/dist"), (Page::Flows, "../../ui/dist")] {
            let bundled = install.join("resources").join(page.folder());
            assert_eq!(with_start_document(dist_candidates(None, Some(install.clone()), &repo, page, false), page), Some(bundled));
            assert_eq!(with_start_document(dist_candidates(None, Some(install.clone()), &repo, page, true), page), Some(repo.join(built)));
        }
    }

    #[test]
    fn the_variable_comes_first_when_it_names_a_page() {
        let scratch = Scratch::new("variable");
        let install = scratch.folder("install");
        let repo = repo(&scratch);
        scratch.file("install/resources/app/index.html");
        scratch.file("mine/index.html");
        let mine = scratch.path("mine");
        for debug in [false, true] {
            let found = with_start_document(dist_candidates(Some(mine.clone()), Some(install.clone()), &repo, Page::Agent, debug), Page::Agent);
            assert_eq!(found, Some(mine.clone()), "debug {debug}");
        }
        // A variable that names a folder without the page is passed over, not obeyed.
        let nothing = scratch.folder("nothing");
        let found = with_start_document(dist_candidates(Some(nothing), Some(install.clone()), &repo, Page::Agent, false), Page::Agent);
        assert_eq!(found, Some(install.join("resources").join("app")));
    }

    #[test]
    fn the_engines_page_has_no_files_of_ours() {
        let scratch = Scratch::new("engines");
        assert!(dist_candidates(None, Some(scratch.path("install")), &repo(&scratch), Page::Engines, false).is_empty());
    }
}
