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

use std::path::{Component, Path, PathBuf};

use percent_encoding::percent_decode_str;
use tauri::http::{header, Request, Response, StatusCode};
use tauri::webview::NewWindowResponse;
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

    fn label(self) -> &'static str {
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
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var(var) {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(dir) = app.path().resource_dir() {
        candidates.push(dir.join(page.folder()));
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    candidates.push(match page {
        Page::Agent => repo.join("../../../app/dist"),
        Page::Flows => repo.join("../../ui/dist"),
        Page::Engines => return None,
    });
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

/// What each embedded page knows from its first line: where the desktop is, and a token for it.
fn desktop_script() -> String {
    format!(
        "window.__OAIY_DESKTOP__ = Object.freeze({{ origin: {origin:?}, token: {token:?} }});",
        origin = format!("http://127.0.0.1:{}", crate::DESKTOP_PORT),
        token = crate::internal_token(),
    )
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
    // The engines' pages are theirs, at their own address; the desktop's two know its address and token.
    let made = if page == Page::Engines {
        WebviewBuilder::new(page.label(), WebviewUrl::External(page.url()))
    } else {
        WebviewBuilder::new(page.label(), WebviewUrl::CustomProtocol(page.url())).initialization_script(&desktop_script())
    };
    made
        .additional_browser_args(BROWSER_ARGS)
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
    fn files_are_typed() {
        assert_eq!(mime("/zipp/engine.wasm"), "application/wasm");
        assert_eq!(mime("/index.html"), "text/html; charset=utf-8");
        assert_eq!(mime("/assets/main.js"), "text/javascript; charset=utf-8");
    }
}
