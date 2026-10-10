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
//! desktop's address and a token for it, so neither has to pair. On a Mac the
//! agent's comes from a port of its own instead ([`AGENT_PORT`]): WebKit gives a
//! page on a scheme of its own no shared memory.
//!
//! All three follow the dashboard's light or dark (`set_theme`): each starts
//! in it and is told when it changes.

use std::path::{Component, Path, PathBuf};

use percent_encoding::percent_decode_str;
use tauri::http::{header, Request, Response, StatusCode};
use tauri::webview::{NewWindowResponse, PageLoadEvent};
use tauri::{AppHandle, LogicalPosition, LogicalSize, Manager, Runtime, Url, WebviewBuilder, WebviewUrl};

pub const AGENT_SCHEME: &str = "oaiy";
pub const FLOWS_SCHEME: &str = "oaiyflows";

/// The port a Mac serves the agent's page from, on this computer alone ([`serve_agent_over_http`]).
///
/// The agent's code sandbox blocks on shared memory. WebKit calls a page on a scheme of its own, served with the
/// headers that isolate a page, cross-origin isolated, and still gives it no `SharedArrayBuffer`; the same page from
/// `http://127.0.0.1` has both (`tools/mac/webview-probe.swift` on the first Mac, macOS 27). The port is fixed because a
/// page's storage is its origin's: another port would be an agent with none of its projects. The engines' gateway lets
/// this origin in (`oaiy_studio::MAC_AGENT_ORIGIN`, which a test holds to this port).
pub const AGENT_PORT: u16 = 17974;

/// Whether this process holds [`AGENT_PORT`] and serves the agent's page there.
static AGENT_PORT_HELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The agent page's origin while this process serves it over http (a Mac, its port held), and only then: the desktop's
/// API takes it for the agent's window ([`crate::http::is_embedded_origin`]). A program that had taken the port would
/// serve its own pages from that origin to any browser here; while OAIY holds it, nothing else answers there.
pub fn agent_http_origin() -> Option<&'static str> {
    AGENT_PORT_HELD.load(std::sync::atomic::Ordering::Acquire).then_some(oaiy_studio::MAC_AGENT_ORIGIN)
}

/// On a Mac: hold [`AGENT_PORT`] and serve the agent's page from it, with the headers that isolate it ([`serve_path`]),
/// so that its webview opens it there. Before the page is made. Where the port is taken (or this is not a Mac) the page
/// stays on its own scheme, and its code sandbox says it is unavailable.
pub fn serve_agent_over_http<R: Runtime>(app: &AppHandle<R>) {
    use std::sync::atomic::Ordering;
    if !cfg!(target_os = "macos") {
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
    log::info!("embed: the agent's page is served from {} (its code sandbox needs shared memory)", oaiy_studio::MAC_AGENT_ORIGIN);
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

    /// Its address in the webview: WebView2 (Windows) maps a custom scheme to `http://<scheme>.localhost`; a Mac
    /// serves the agent's from a port of its own where it holds one ([`agent_http_origin`]).
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
fn desktop_script(theme: &str, page: Page) -> String {
    let backup = if page == Page::Agent { format!(", backupToken: {:?}", crate::backup::agent::page_token()) } else { String::new() };
    format!(
        "window.__OAIY_DESKTOP__ = Object.freeze({{ origin: {origin:?}, token: {token:?}, theme: {theme:?}{backup} }});",
        origin = format!("http://127.0.0.1:{}", crate::DESKTOP_PORT),
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
pub async fn agent_intent<R: Runtime>(app: AppHandle<R>, intent: String) -> Result<(), String> {
    if !AGENT_INTENTS.contains(&intent.as_str()) {
        return Err(format!("no agent intent called {intent}"));
    }
    let webview = app
        .get_webview(Page::Agent.label())
        .ok_or("the Agent is not open: open it once from the sidebar, then try again")?;
    webview.eval(intent_script(&intent)).map_err(|e| e.to_string())
}

/// The Agent's page, when it exists (hidden or shown): what the desktop asks to save its work before an update.
pub fn agent_webview<R: Runtime>(app: &AppHandle<R>) -> Option<tauri::Webview<R>> {
    app.get_webview(Page::Agent.label())
}

/// A link that leaves an embedded page: to the system browser.
fn open_outside<R: Runtime>(app: &AppHandle<R>, url: &Url) {
    use tauri_plugin_shell::ShellExt;
    if matches!(url.scheme(), "http" | "https" | "mailto") {
        #[allow(deprecated)]
        let _ = app.shell().open(url.as_str(), None);
    }
}

fn own(page: Page, url: &Url) -> bool {
    own_at(&page.url(), url)
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
    let _one = placing();
    let placed = show_at(&app, &window, page, x, y, width, height);
    if placed.is_ok() {
        note_placement(&app, &window, page, [x, y, width, height], seen);
    }
    placed
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

fn show_at<R: Runtime>(app: &AppHandle<R>, window: &tauri::Window<R>, page: Page, x: f64, y: f64, width: f64, height: f64) -> Result<(), String> {
    for other in Page::ALL {
        if other != page {
            if let Some(w) = app.get_webview(other.label()) {
                let _ = w.hide();
            }
        }
    }
    let (position, size) = (LogicalPosition::new(x.max(0.0), y.max(0.0)), LogicalSize::new(width.max(1.0), height.max(1.0)));
    if let Some(webview) = app.get_webview(page.label()) {
        webview.set_position(position).map_err(|e| e.to_string())?;
        webview.set_size(size).map_err(|e| e.to_string())?;
        webview.show().map_err(|e| e.to_string())?;
        let _ = webview.set_focus();
        let _ = webview.eval(theme_script(theme(app)));
        return Ok(());
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
    window.add_child(builder(app, page), position, size).map_err(|e| e.to_string())?;
    Ok(())
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
        match window.add_child(builder(&app, Page::Agent), LogicalPosition::new(0.0, 0.0), LogicalSize::new(1.0, 1.0)) {
            Ok(webview) => {
                let _ = webview.hide();
            }
            Err(e) => log::warn!("the agent's page was not made at startup: {e}"),
        }
    });
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
        // (a Mac's agent, from its own port: an address, not a scheme the window serves)
        WebviewBuilder::new(page.label(), WebviewUrl::External(page.url())).initialization_script(&desktop_script(first, page))
    } else {
        WebviewBuilder::new(page.label(), WebviewUrl::CustomProtocol(page.url())).initialization_script(&desktop_script(first, page))
    };
    made
        .additional_browser_args(BROWSER_ARGS)
        // A page that (re)loads starts in the theme it was made with: tell it the one now.
        .on_page_load(|webview, payload| {
            if payload.event() == PageLoadEvent::Finished {
                let _ = webview.eval(theme_script(theme(webview.app_handle())));
            }
        })
        .on_navigation(move |url| {
            if own(page, url) {
                return true;
            }
            open_outside(&for_navigation, url);
            false
        })
        .on_new_window(move |url, _features| {
            if own(page, &url) {
                return NewWindowResponse::Allow;
            }
            open_outside(&for_windows, &url);
            NewWindowResponse::Deny
        })
}

/// Hide the embedded pages (the dashboard shows one of its own).
#[tauri::command]
pub async fn hide_embedded<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    for page in Page::ALL {
        if let Some(w) = app.get_webview(page.label()) {
            w.hide().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_webview_is_given_the_embedder_policy_it_isolates_a_page_for() {
        // Chromium's (Windows) own value, which lets the page load other sites' files without credentials.
        assert_eq!(embedder_policy(true), "credentialless");
        // WebKit (a Mac, Linux) does not know that one, and with it the page is not isolated: no code sandbox.
        assert_eq!(embedder_policy(false), "require-corp");
    }

    #[test]
    fn a_macs_agent_is_served_from_the_port_the_gateway_lets_in() {
        // The engines' gateway names the origin in its own crate (it lets the page in); it is this port's.
        assert_eq!(oaiy_studio::MAC_AGENT_ORIGIN, format!("http://127.0.0.1:{AGENT_PORT}"));
        // Not held (nothing here served it), the page is at its scheme and the origin is nobody's.
        assert_eq!(agent_http_origin(), None);
        assert!(!crate::http::is_embedded_origin(oaiy_studio::MAC_AGENT_ORIGIN));
        assert!(!crate::backup::routes::is_agent_origin(oaiy_studio::MAC_AGENT_ORIGIN));
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
    fn only_the_agents_page_is_given_the_restore_secret() {
        let secret = crate::backup::agent::page_token();
        assert!(secret.len() >= 32, "a real secret");
        assert!(desktop_script("dark", Page::Agent).contains(&format!("backupToken: {secret:?}")), "the Agent's page has it");
        assert!(!desktop_script("dark", Page::Flows).contains("backupToken") && !desktop_script("dark", Page::Flows).contains(secret));
        assert!(!desktop_script("dark", Page::Engines).contains(secret));
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
