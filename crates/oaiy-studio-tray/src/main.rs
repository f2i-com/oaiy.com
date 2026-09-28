//! `oaiy-studio-tray`: OAIY as a Windows notification-area app.
//!
//! The same studio as `oaiy-studio`, without a console window: it starts in the
//! background, puts an icon by the clock, and opens the UI on a click. Closing
//! the UI window leaves it serving; Quit (in the icon's menu) stops the engines
//! and exits. It takes `oaiy-studio`'s options; `--open none` starts it silently
//! (what "Start with Windows" registers).

#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(windows)]
mod tray;

/// Added to the "Start with Windows" command line: a sign-in start stays
/// quiet, with no balloon. Not an `oaiy-studio` option, so it is taken out
/// before the rest are parsed.
pub const AUTOSTART_FLAG: &str = "--autostarted";

fn main() {
    #[cfg(windows)]
    tray::dpi_aware();
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    let autostarted = argv.iter().any(|a| a == AUTOSTART_FLAG);
    argv.retain(|a| a != AUTOSTART_FLAG);
    let args = match oaiy_studio::parse_args_from(argv.into_iter()) {
        Ok(a) => a,
        Err(e) => fail(&e),
    };
    // A second launch shows the running studio rather than failing on its ports.
    if let Some(url) = oaiy_studio::running_instance(&args) {
        #[cfg(windows)]
        tray::show_ui(&url);
        #[cfg(not(windows))]
        oaiy_studio::open_ui(&url, "app");
        return;
    }
    let running = match oaiy_studio::launch(&args, true) {
        Ok(r) => r,
        Err(e) => fail(&e),
    };
    let silent = running.open == "none";
    if !silent {
        #[cfg(windows)]
        if running.open == "app" {
            tray::show_ui(&running.ui_url);
        } else {
            oaiy_studio::open_ui(&running.ui_url, &running.open);
        }
        #[cfg(not(windows))]
        oaiy_studio::open_ui(&running.ui_url, &running.open);
    }
    #[cfg(windows)]
    tray::run(running, silent && !autostarted);
    #[cfg(not(windows))]
    let _ = autostarted;
    #[cfg(not(windows))]
    {
        eprintln!("oaiy-studio-tray: the notification-area icon is Windows-only; serving as oaiy-studio does");
        oaiy_studio::console(&running.studio, &running.ui_url);
    }
}

fn fail(message: &str) -> ! {
    #[cfg(windows)]
    tray::message_box(&format!("OAIY could not start:\n\n{message}"));
    #[cfg(not(windows))]
    eprintln!("oaiy-studio-tray: {message}");
    std::process::exit(1);
}
