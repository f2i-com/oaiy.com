fn main() {
    // Only the GUI build needs tauri's build-time codegen. The headless
    // oaiy-server (--no-default-features) skips it entirely.
    #[cfg(feature = "gui")]
    tauri_build::build();
}
