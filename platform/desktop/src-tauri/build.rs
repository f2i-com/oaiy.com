fn main() {
    version_from_git();
    // Only the GUI build needs tauri's build-time codegen. The headless
    // oaiy-server (--no-default-features) skips it entirely.
    #[cfg(feature = "gui")]
    {
        // tauri.conf.json lists the folders the installer carries the Agent and
        // the flow editor in (`bundle.resources`: resources/app and
        // resources/flows, which scripts/stage-pages.mjs fills from their
        // builds), and the portable language-model engine (resources/engines,
        // which scripts/stage-engines.mjs fills). Tauri's build fails on a listed
        // folder that does not exist and skips one that is empty, so make sure
        // they are there: a build that is not making an installer (tauri dev,
        // the tests) needs none of them, because embed.rs serves the pages from
        // their own build folders and engines.rs finds the repository's build.
        // `tauri build` stages them first (beforeBuildCommand), and stops if it
        // cannot, so an installer never goes without.
        for folder in ["resources/app", "resources/flows", "resources/engines"] {
            let _ = std::fs::create_dir_all(folder);
        }
        tauri_build::build();
    }
}

/// The version a build from a checkout says it is (`CARGO_PKG_VERSION`, which everything that says or checks this
/// desktop's version reads). A release stamps its tag's version into Cargo.toml and tauri.conf.json as it builds
/// (release.yml) and never commits it, so a checkout keeps the placeholder, and a build from one said "OAIY 0.1.0" in
/// Settings beside a latest release of 0.1.4. Where Cargo.toml has the placeholder and the checkout has a release
/// tag behind it, the version is that tag's, with the commits since it as build metadata (`0.1.4+9.g5950d331`, at
/// the tag itself `0.1.4`), which no check counts: the updater and a plugin's minDesktopVersion see 0.1.4. Without
/// git, or a tag (a shallow clone), it stays as Cargo.toml has it.
fn version_from_git() {
    const PLACEHOLDER: &str = "0.1.0";
    if std::env::var("CARGO_PKG_VERSION").as_deref() != Ok(PLACEHOLDER) {
        return;
    }
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git").args(args).output().ok().filter(|o| o.status.success())?;
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    // (made again when HEAD moves, its branch has a new commit or a tag comes; only paths that are there, as one that
    // is not would have every build made again)
    if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        let branch = git(&["symbolic-ref", "-q", "HEAD"]);
        for path in ["HEAD", "packed-refs", "refs/tags"].into_iter().map(String::from).chain(branch) {
            let path = std::path::Path::new(&dir).join(path);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    // `v0.1.4-9-g5950d331`: the tag, the commits since it, and this commit
    let Some(described) = git(&["describe", "--tags", "--long", "--match", "v[0-9]*"]) else { return };
    let Some((rest, commit)) = described.rsplit_once('-') else { return };
    let Some((tag, since)) = rest.rsplit_once('-') else { return };
    let version = tag.trim_start_matches('v');
    let numbers: Vec<&str> = version.split('.').collect();
    if numbers.len() != 3 || numbers.iter().any(|n| n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit())) {
        return;
    }
    let version = if since == "0" { version.to_string() } else { format!("{version}+{since}.{commit}") };
    println!("cargo:rustc-env=CARGO_PKG_VERSION={version}");
}
