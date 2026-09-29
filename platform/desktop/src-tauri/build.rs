fn main() {
    // Only the GUI build needs tauri's build-time codegen. The headless
    // oaiy-server (--no-default-features) skips it entirely.
    #[cfg(feature = "gui")]
    {
        // tauri.conf.json lists the folders the installer carries the Agent and
        // the flow editor in (`bundle.resources`: resources/app and
        // resources/flows, which scripts/stage-pages.mjs fills from their
        // builds). Tauri's build fails on a listed folder that does not exist
        // and skips one that is empty, so make sure both are there: a build that
        // is not making an installer (tauri dev, the tests) needs no pages,
        // because embed.rs serves them from the pages' own build folders.
        // `tauri build` stages them first (beforeBuildCommand), and stops if it
        // cannot, so an installer never goes without.
        for folder in ["resources/app", "resources/flows"] {
            let _ = std::fs::create_dir_all(folder);
        }
        tauri_build::build();
    }
}
