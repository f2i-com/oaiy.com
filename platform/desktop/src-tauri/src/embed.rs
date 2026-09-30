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
//! desktop's address and a token for it, so neither has to pair.
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

    /// Its address in the webview: WebView2 (Windows) maps a custom scheme to `http://<scheme>.localhost`.
    fn url(self) -> Url {
        if self == Page::Engines {
            let base = crate::engines::ui_url().unwrap_or_else(|| "http://127.0.0.1:7860".into());
            return base.parse().unwrap_or_else(|_| "http://127.0.0.1:7860/".parse().expect("a valid URL"));
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

/// A file of an embedded page, with the headers that make it cross-origin isolated.
pub fn serve<R: Runtime>(app: &AppHandle<R>, page: Page, request: &Request<Vec<u8>>) -> Response<Vec<u8>> {
    let raw = percent_decode_str(request.uri().path()).decode_utf8_lossy().into_owned();
    let path = if raw == "/" || raw.is_empty() { page.start().to_string() } else { raw };
    // Only plain names inside the page's folder.
    let relative = Path::new(path.trim_start_matches('/'));
    if relative.components().any(|c| !matches!(c, Component::Normal(_))) {
        return not_found(&path);
    }
    let Some(root) = dist(app, page) else {
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
        .header("Cross-Origin-Embedder-Policy", "credentialless")
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
fn desktop_script(theme: &str) -> String {
    format!(
        "window.__OAIY_DESKTOP__ = Object.freeze({{ origin: {origin:?}, token: {token:?}, theme: {theme:?} }});",
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
    let mine = page.url();
    (url.scheme() == mine.scheme() && url.host_str() == mine.host_str()) || matches!(url.scheme(), "blob" | "data" | "about")
}

/// Show an embedded page over the dashboard's content area (logical pixels
/// within the window), made on first use; the other embedded page is hidden.
///
/// Async, so it runs off the main thread: making a webview from a synchronous
/// command (which runs on the main thread) waits on that thread's own event
/// loop, and on Windows never returns.
#[tauri::command]
pub async fn show_embedded<R: Runtime>(app: AppHandle<R>, window: tauri::Window<R>, page: String, x: f64, y: f64, width: f64, height: f64) -> Result<(), String> {
    let page = Page::parse(&page).ok_or_else(|| format!("no embedded page called {page}"))?;
    let _one = placing();
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
        let _ = webview.eval(theme_script(theme(&app)));
        return Ok(());
    }
    if page == Page::Engines {
        if crate::engines::ui_url().is_none() {
            return Err("the engines are not running yet: they start with OAIY, or run oaiy-studio".into());
        }
    } else if dist(&app, page).is_none() {
        return Err(format!("the {} page is not built: run `npm run build` in {}", page.folder(), match page {
            Page::Agent => "app/",
            _ => "platform/ui/",
        }));
    }
    window.add_child(builder(&app, page), position, size).map_err(|e| e.to_string())?;
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
    } else {
        WebviewBuilder::new(page.label(), WebviewUrl::CustomProtocol(page.url())).initialization_script(&desktop_script(first))
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
    fn the_embedded_pages_are_named_and_have_their_schemes() {
        use crate::http::is_embedded_origin;
        for page in Page::SERVED {
            let url = page.url();
            assert!(is_embedded_origin(&url.origin().ascii_serialization()), "{url}");
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
        assert!(desktop_script("light").contains(r#"theme: "light""#));
        assert_eq!(theme_script("dark"), r#"window.__oaiySetTheme ? window.__oaiySetTheme("dark") : (window.__OAIY_THEME__ = "dark");"#);
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
