//! Tray icon + context menu setup.
//!
//! The companion is designed to live in the system tray. The main window
//! is hidden by default (see tauri.conf.json `visible: false`); the user
//! opens it from the tray menu when they want to see service status.

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    App, Manager,
};

pub fn setup(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    let handle = app.handle();

    let open_item = MenuItem::with_id(handle, "open", "Open OAIY", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(handle, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(handle, &[&open_item, &quit_item])?;

    let mut builder = TrayIconBuilder::with_id("oaiy-companion")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip("OAIY");
    // Tauri 2 does NOT auto-assign a tray icon — without this the tray
    // entry shows blank. Reuse the app's bundled window icon.
    if let Some(icon) = handle.default_window_icon().cloned() {
        builder = builder.icon(icon);
    }
    builder
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.unminimize(); // show() alone won't restore a minimized window
                    let _ = window.set_focus();
                }
            }
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
                if let Some(window) = tray.app_handle().get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.unminimize(); // show() alone won't restore a minimized window
                    let _ = window.set_focus();
                }
            }
        })
        .build(handle)?;

    Ok(())
}
