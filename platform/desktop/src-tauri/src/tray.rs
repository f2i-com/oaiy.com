//! Tray icon + context menu setup.
//!
//! OAIY Desktop is designed to live in the system tray. The main window
//! is hidden by default (see tauri.conf.json `visible: false`); the user
//! opens it from the tray menu when they want to see service status.

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    App, Manager,
};

/// Bring the main window up: shown, restored if minimized, and focused.
///
/// As a `Window`, not a `WebviewWindow`: the agent's and the flow editor's
/// pages are webviews of their own inside it, and a window with several
/// webviews is no `WebviewWindow` (`get_webview_window` finds nothing).
pub fn show_main(app: &tauri::AppHandle) {
    match app.get_window("main") {
        Some(window) => {
            let _ = window.show();
            let _ = window.unminimize(); // show() alone won't restore a minimized window
            let _ = window.set_focus();
        }
        None => log::warn!("there is no main window to show"),
    }
}

/// Bring the main window up for a ring, without asking for the keyboard: shown, restored if minimized, and its taskbar button flashed, and no more.
/// [`show_main`] focuses the window, which would send the keys the owner is typing in another program to this one at the moment a caller rings. (Whether
/// Windows makes a window that was hidden in the tray the active one when it is shown is Windows's to decide, and has not been checked on every machine.)
pub fn show_main_for_a_ring(app: &tauri::AppHandle) {
    match app.get_window("main") {
        Some(window) => {
            let _ = window.show();
            let _ = window.unminimize();
            let _ = window.request_user_attention(Some(tauri::UserAttentionType::Informational));
        }
        None => log::warn!("there is no main window to show"),
    }
}

/// "Check for updates": show OAIY on its Settings page, where the answer is, and look. (The check is limited to one
/// in 30 seconds, like every other way of asking; a refusal only means the page already shows the last answer.)
fn check_for_updates(app: &tauri::AppHandle) {
    use tauri::Emitter as _;
    show_main(app);
    let _ = app.emit(crate::control::NAVIGATE_EVENT, serde_json::json!({ "view": "settings" }));
    if let Some(updater) = app.try_state::<crate::update::UpdaterHandle>() {
        let updater = updater.inner().clone();
        tauri::async_runtime::spawn(async move {
            if let Err(refusal) = updater.check().await {
                log::info!("update: the check from the tray did not run: {refusal}");
            }
        });
    }
}

pub fn setup(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    let handle = app.handle();

    let open_item = MenuItem::with_id(handle, "open", "Open OAIY", true, None::<&str>)?;
    let update_item = MenuItem::with_id(handle, "update-check", "Check for updates", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(handle, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(handle, &[&open_item, &update_item, &quit_item])?;

    let mut builder = TrayIconBuilder::with_id("oaiy-desktop")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip("OAIY Desktop");
    // Tauri 2 does NOT auto-assign a tray icon — without this the tray
    // entry shows blank. Reuse the app's bundled window icon.
    if let Some(icon) = handle.default_window_icon().cloned() {
        builder = builder.icon(icon);
    }
    builder
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => {
                show_main(app);
            }
            "update-check" => check_for_updates(app),
            "quit" => {
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            // Left-click the tray icon = open/focus the window. This is
            // the most natural action and matches Windows/macOS tray
            // expectations (the menu shows on right-click only).
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main(tray.app_handle());
            }
        })
        .build(handle)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    /// The window comes up for a ring without a call to focus it: focusing would send the keys the owner is typing in another program to OAIY at the
    /// moment a caller rings. (There is no window to make here, so the functions' own text is what is read.)
    #[test]
    fn a_ring_shows_the_window_and_flashes_its_button_and_never_focuses_it() {
        // (Read with its line endings as LF: a checkout on Windows, where git converts them, has CRLF, and the search below is for "\n}\n".)
        let source = include_str!("tray.rs").replace("\r\n", "\n");
        let start = source.find("pub fn show_main_for_a_ring").expect("the function");
        let end = source[start..].find("\n}\n").expect("its end") + start;
        let body = &source[start..end];
        assert!(!body.contains("set_focus"), "{body}");
        assert!(body.contains("request_user_attention") && body.contains(".show()") && body.contains("unminimize"), "{body}");
        // What the ring's notifier calls is that function, and not the one that focuses.
        let notify = include_str!("notify.rs").replace("\r\n", "\n");
        let of = notify.find("pub fn of(app: AppHandle) -> Self").expect("GuiRing::of");
        let of_end = notify[of..].find("\n    }\n").expect("its end") + of;
        let of_body = &notify[of..of_end];
        assert!(of_body.contains("show_main_for_a_ring") && !of_body.contains("show_main(") && !of_body.contains("set_focus"), "{of_body}");
    }
}