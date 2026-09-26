//! `nrob-studio-tray`: NROB Studio as a Windows notification-area app.
//!
//! The same studio as `nrob-studio`, without a console window: it starts in the
//! background, puts an icon by the clock, and opens the UI on a click. Closing
//! the UI window leaves it serving; Quit (in the icon's menu) stops the engines
//! and exits. It takes `nrob-studio`'s options; `--open none` starts it silently
//! (what "Start with Windows" registers).

#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(windows)]
mod tray;

fn main() {
    let args = match nrob_studio::parse_args_from(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => fail(&e),
    };
    // A second launch shows the running studio rather than failing on its ports.
    if let Some(url) = nrob_studio::running_instance(&args) {
        nrob_studio::open_ui(&url, "app");
        return;
    }
    let running = match nrob_studio::launch(&args, true) {
        Ok(r) => r,
        Err(e) => fail(&e),
    };
    let silent = running.open == "none";
    if !silent {
        nrob_studio::open_ui(&running.ui_url, &running.open);
    }
    #[cfg(windows)]
    tray::run(running, silent);
    #[cfg(not(windows))]
    {
        eprintln!("nrob-studio-tray: the notification-area icon is Windows-only; serving as nrob-studio does");
        nrob_studio::console(&running.studio, &running.ui_url);
    }
}

fn fail(message: &str) -> ! {
    #[cfg(windows)]
    tray::message_box(&format!("NROB Studio could not start:\n\n{message}"));
    #[cfg(not(windows))]
    eprintln!("nrob-studio-tray: {message}");
    std::process::exit(1);
}
