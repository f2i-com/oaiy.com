//! bot.computer as a desktop app: the same web app (the built `dist/`) in the
//! system's webview, served from the app's own `botcomputer` scheme.
//!
//! Serving it here, rather than from Tauri's default scheme, is for the
//! headers the app needs:
//! - COOP and COEP make the page cross-origin isolated, so the code sandbox can
//!   block a Worker on shared memory;
//! - CORP and a wildcard CORS header on `/softn/` let the SoftN preview (an
//!   opaque-origin, sandboxed frame) load its runtime.
//!
//! It also gives the app an origin of its own (`http://botcomputer.localhost`
//! on Windows, `botcomputer://localhost` elsewhere) that local servers such as
//! nrob can allow by name. Links that leave the app open in the system browser.
//!
//! The app lives in the system tray: minimizing hides the window there, and
//! (unless switched off in the tray menu) so does closing it, so the agent can
//! keep working in the background. Quit saves the page's work before exiting.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use percent_encoding::percent_decode_str;
use tauri::http::{header, Request, Response, StatusCode};
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::webview::NewWindowResponse;
use tauri::{AppHandle, Manager, Runtime, Url, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_opener::OpenerExt;

const SCHEME: &str = "botcomputer";
const MAIN: &str = "main";

/// Where the app's pages live: Windows (WebView2) maps a custom scheme to `http://<scheme>.localhost`.
fn app_url() -> Url {
    let url = if cfg!(any(windows, target_os = "android")) { "http://botcomputer.localhost/" } else { "botcomputer://localhost/" };
    url.parse().expect("the app URL is valid")
}

fn same_site(a: &Url, b: &Url) -> bool {
    a.scheme() == b.scheme() && a.host_str() == b.host_str() && a.port_or_known_default() == b.port_or_known_default()
}

/// A URL the app shows itself (its own pages, in-page data), rather than the system browser.
fn belongs_to_app(url: &Url, dev: Option<&Url>) -> bool {
    same_site(url, &app_url()) || dev.is_some_and(|d| same_site(url, d)) || matches!(url.scheme(), "blob" | "data" | "about")
}

/// A link that leaves the app: to the system browser (or mail client).
fn open_outside<R: Runtime>(app: &AppHandle<R>, url: &Url) {
    if matches!(url.scheme(), "http" | "https" | "mailto") {
        let _ = app.opener().open_url(url.as_str(), None::<&str>);
    }
}

/// A file of the built app, with the headers the page needs.
fn serve<R: Runtime>(app: &AppHandle<R>, request: &Request<Vec<u8>>) -> Response<Vec<u8>> {
    let raw = request.uri().path();
    let mut path = percent_decode_str(raw).decode_utf8_lossy().into_owned();
    if path.ends_with('/') {
        path.push_str("index.html");
    }
    let Some(asset) = app.asset_resolver().get(path.clone()) else {
        return Response::builder().status(StatusCode::NOT_FOUND).header(header::CONTENT_TYPE, "text/plain").body(format!("{path} is not part of bot.computer").into_bytes()).unwrap();
    };
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, asset.mime_type.as_str())
        .header("Cross-Origin-Opener-Policy", "same-origin")
        .header("Cross-Origin-Embedder-Policy", "credentialless")
        .header("Cross-Origin-Resource-Policy", "cross-origin")
        .header(header::X_CONTENT_TYPE_OPTIONS, "nosniff");
    // The SoftN preview frame has an opaque origin: its module scripts are cross-origin loads.
    if path.starts_with("/softn/") {
        response = response.header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*");
    }
    response.body(asset.bytes).unwrap()
}

// --- Living in the tray ------------------------------------------------------

/// Whether closing the window leaves the app running in the tray (the tray menu's check box).
struct KeepRunning(AtomicBool);

fn settings_file<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|dir| dir.join("desktop.json"))
}

fn load_keep_running<R: Runtime>(app: &AppHandle<R>) -> bool {
    settings_file(app)
        .and_then(|path| std::fs::read(path).ok())
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|v| v.get("keepRunningWhenClosed").and_then(serde_json::Value::as_bool))
        .unwrap_or(true)
}

fn save_keep_running<R: Runtime>(app: &AppHandle<R>, keep: bool) {
    if let Some(path) = settings_file(app) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, serde_json::json!({ "keepRunningWhenClosed": keep }).to_string());
    }
}

fn show_main<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window(MAIN) {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
    if let Some(tray) = app.tray_by_id(MAIN) {
        let _ = tray.set_tooltip(Some("bot.computer"));
    }
}

fn hide_main<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window(MAIN) {
        let _ = window.hide();
    }
    if let Some(tray) = app.tray_by_id(MAIN) {
        let _ = tray.set_tooltip(Some("bot.computer: running in the background"));
    }
}

static QUITTING: AtomicBool = AtomicBool::new(false);

/// Save the page's work (files, chat), then exit. The page says when it is done
/// (`ready_to_quit`); a page that cannot is given a few seconds.
fn quit<R: Runtime>(app: &AppHandle<R>) {
    if QUITTING.swap(true, Ordering::SeqCst) {
        return;
    }
    match app.get_webview_window(MAIN) {
        Some(window) => {
            let _ = window.eval("window.__botComputerBeforeQuit ? window.__botComputerBeforeQuit() : window.__TAURI_INTERNALS__.invoke('ready_to_quit')");
            let handle = app.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(8));
                handle.exit(0);
            });
        }
        None => app.exit(0),
    }
}

/// The page has saved everything: exit now.
#[tauri::command]
fn ready_to_quit<R: Runtime>(app: AppHandle<R>) {
    app.exit(0);
}

fn tray<R: Runtime>(app: &tauri::App<R>) -> tauri::Result<()> {
    let keep = app.state::<KeepRunning>().0.load(Ordering::SeqCst);
    let open = MenuItem::with_id(app, "open", "Open bot.computer", true, None::<&str>)?;
    let keep_item = CheckMenuItem::with_id(app, "keep", "Keep running when the window is closed", true, keep, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit bot.computer", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &PredefinedMenuItem::separator(app)?, &keep_item, &PredefinedMenuItem::separator(app)?, &quit_item])?;
    let mut builder = TrayIconBuilder::with_id(MAIN)
        .tooltip("bot.computer")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "open" => show_main(app),
            "keep" => {
                let keep = keep_item.is_checked().unwrap_or(true);
                app.state::<KeepRunning>().0.store(keep, Ordering::SeqCst);
                save_keep_running(app, keep);
            }
            "quit" => quit(app),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                show_main(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder.build(app)?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // A second start (a shortcut clicked while the app is in the tray) shows the running one.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| show_main(app)))
        .plugin(tauri_plugin_opener::init())
        .register_uri_scheme_protocol(SCHEME, |ctx, request| serve(ctx.app_handle(), &request))
        .invoke_handler(tauri::generate_handler![ready_to_quit])
        .setup(|app| {
            app.manage(KeepRunning(AtomicBool::new(load_keep_running(app.handle()))));
            let dev = app.config().build.dev_url.clone().filter(|_| tauri::is_dev());
            // `tauri dev` shows the Vite dev server (it sends the same headers); a build, the app's own scheme.
            let url = if dev.is_some() { WebviewUrl::App("index.html".into()) } else { WebviewUrl::CustomProtocol(app_url()) };
            let for_navigation = app.handle().clone();
            let for_windows = app.handle().clone();
            let dev_nav = dev.clone();
            let dev_win = dev.clone();
            let builder = WebviewWindowBuilder::new(app, MAIN, url)
                .title("bot.computer")
                .inner_size(1400.0, 900.0)
                .min_inner_size(720.0, 520.0)
                .center()
                .on_navigation(move |url| {
                    if belongs_to_app(url, dev_nav.as_ref()) {
                        return true;
                    }
                    open_outside(&for_navigation, url);
                    false
                })
                .on_new_window(move |url, _features| {
                    if belongs_to_app(&url, dev_win.as_ref()) {
                        return NewWindowResponse::Allow;
                    }
                    open_outside(&for_windows, &url);
                    NewWindowResponse::Deny
                });
            // In the tray the window is hidden, and a hidden Chromium page has its timers
            // throttled and may be frozen: the agent has to keep working there.
            #[cfg(windows)]
            let builder = builder.additional_browser_args(
                "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows",
            );
            #[cfg(target_os = "macos")]
            let builder = builder.background_throttling(tauri::utils::config::BackgroundThrottlingPolicy::Disabled);
            builder.build()?;
            tray(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() != MAIN {
                return;
            }
            match event {
                WindowEvent::CloseRequested { api, .. } => {
                    api.prevent_close();
                    if window.app_handle().state::<KeepRunning>().0.load(Ordering::SeqCst) {
                        hide_main(window.app_handle());
                    } else {
                        quit(window.app_handle());
                    }
                }
                // Minimizing puts the app in the tray, off the taskbar.
                WindowEvent::Resized(_) if window.is_minimized().unwrap_or(false) => hide_main(window.app_handle()),
                _ => {}
            }
        })
        .run(tauri::generate_context!())
        .expect("bot.computer could not start");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_app_keeps_its_own_pages_and_sends_links_out() {
        let app = app_url();
        assert!(belongs_to_app(&app.join("index.html").unwrap(), None));
        assert!(belongs_to_app(&app.join("softn/index.html?v=3").unwrap(), None));
        assert!(belongs_to_app(&"blob:http://botcomputer.localhost/1234".parse().unwrap(), None));
        assert!(!belongs_to_app(&"https://github.com/f2i-com".parse().unwrap(), None));
        assert!(!belongs_to_app(&"http://tauri.localhost/".parse().unwrap(), None));
        let dev: Url = "http://localhost:5317".parse().unwrap();
        assert!(belongs_to_app(&"http://localhost:5317/src/main.ts".parse().unwrap(), Some(&dev)));
        assert!(!belongs_to_app(&"http://localhost:8080/".parse().unwrap(), Some(&dev)));
    }
}
