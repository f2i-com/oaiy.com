// Prevents additional console window on Windows in release. OAIY Desktop
// is a tray-resident app — there's no terminal to show.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    #[cfg(feature = "gui")]
    oaiy_desktop_lib::run();
    #[cfg(not(feature = "gui"))]
    eprintln!("oaiy-desktop requires the 'gui' feature; use the oaiy-server binary for headless.");
}
