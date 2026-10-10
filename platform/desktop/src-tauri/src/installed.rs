//! Where an installed OAIY finds the files its installer carries (`resources/…`: the CLI, the pages, the engines,
//! and in the headless archive a Node runtime).
//!
//! Each system's package puts them somewhere else, seen from the program:
//!
//! | the install | the program | its `resources/` |
//! |---|---|---|
//! | Windows' installer, the headless archive, a build's `target/<profile>` | `<dir>/oaiy-desktop` | `<dir>/resources/` |
//! | Linux: the `.deb`, the `.rpm`, and the same tree inside the AppImage | `<prefix>/bin/oaiy-desktop` | `<prefix>/lib/OAIY/resources/` |
//! | macOS: the app | `OAIY.app/Contents/MacOS/oaiy-desktop` | `OAIY.app/Contents/Resources/resources/` |
//! | `cargo run` in the crate (nothing is copied) | `target/<profile>/oaiy-desktop` | `src-tauri/resources/` |
//!
//! The window app is told its resource folder by Tauri as well ([`set_resource_dir`]); the table is for the programs
//! with no Tauri to ask (`oaiy-server`, which a Linux package installs beside the app) and for anything that runs
//! before the app has said. A lookup that knew only the first row is why no flow ran on an installed Linux app: the
//! CLI was looked for beside the program, and was in `/usr/lib/OAIY`.

use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

/// The folder Tauri's Linux packages keep resources in is named after the product: `productName` in
/// tauri.conf.json (a test holds the two together).
const PRODUCT: &str = "OAIY";

static RESOURCE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The resource folder as Tauri resolved it for this install. The app says so once, at start.
pub fn set_resource_dir(dir: PathBuf) {
    let _ = RESOURCE_DIR.set(dir);
}

/// The first `resources/<relative>` this install carries for which `is` holds (a file, a folder with a program in
/// it), or `None`: a build that staged none, or a layout nobody has described.
pub fn resource(relative: &str, is: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    resource_from(exe.parent()?, RESOURCE_DIR.get().map(PathBuf::as_path), relative, is)
}

/// [`resource`] for a program in `exe_dir`, with the folder Tauri named (`told`), if it has.
fn resource_from(exe_dir: &Path, told: Option<&Path>, relative: &str, is: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    roots(exe_dir, told).into_iter().map(|root| root.join("resources").join(relative)).find(|p| is(p))
}

/// The folders that may hold `resources/`, best first: beside the program (a copy put there is one's own build, and
/// is where Windows and the headless archive have it), where Tauri said, then each package's place (module docs).
fn roots(exe_dir: &Path, told: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = vec![exe_dir.to_path_buf()];
    roots.extend(told.map(Path::to_path_buf));
    for relative in [format!("../lib/{PRODUCT}"), "../Resources".into(), "..".into(), "../..".into()] {
        roots.push(tidy(&exe_dir.join(relative)));
    }
    roots
}

/// `path` with its `..` steps taken, by name alone. NOT `canonicalize`: on Windows that makes a verbatim path
/// (`\\?\C:\…`), which Node cannot take as its main module (`EISDIR: illegal operation on a directory, lstat 'C:'`).
/// The program's own path has no link in it worth following, and what is found is checked to be there.
fn tidy(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A folder of this test's own, with `files` (empty) in it.
    fn tree(name: &str, files: &[&str]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("oaiy-installed-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for file in files {
            let path = root.join(file);
            std::fs::create_dir_all(path.parent().expect("a file has a folder")).expect("the folder is made");
            std::fs::write(&path, b"").expect("the file is written");
        }
        root
    }

    fn cli(exe_dir: &Path, told: Option<&Path>) -> Option<PathBuf> {
        resource_from(exe_dir, told, "cli/oaiy.mjs", |p| p.is_file())
    }

    #[test]
    fn the_product_name_is_the_one_tauri_names_the_linux_folder_after() {
        let conf: serde_json::Value = serde_json::from_str(include_str!("../tauri.conf.json")).expect("tauri.conf.json is JSON");
        assert_eq!(conf["productName"], PRODUCT, "a Linux package keeps resources in /usr/lib/<productName>");
    }

    #[test]
    fn each_systems_package_is_found_from_its_program() {
        // Windows' installer and the headless archive: beside the program.
        let windows = tree("windows", &["OAIY/oaiy-desktop.exe", "OAIY/resources/cli/oaiy.mjs"]);
        assert_eq!(cli(&windows.join("OAIY"), None), Some(windows.join("OAIY/resources/cli/oaiy.mjs")));

        // A .deb or .rpm, and an AppImage's mounted tree: <prefix>/bin and <prefix>/lib/OAIY.
        let linux = tree("linux", &["usr/bin/oaiy-desktop", "usr/lib/OAIY/resources/cli/oaiy.mjs"]);
        assert_eq!(cli(&linux.join("usr/bin"), None), Some(linux.join("usr/lib/OAIY/resources/cli/oaiy.mjs")));

        // A Mac's app.
        let mac = tree("mac", &["OAIY.app/Contents/MacOS/oaiy-desktop", "OAIY.app/Contents/Resources/resources/cli/oaiy.mjs"]);
        assert_eq!(
            cli(&mac.join("OAIY.app/Contents/MacOS"), None),
            Some(mac.join("OAIY.app/Contents/Resources/resources/cli/oaiy.mjs")),
        );

        // `cargo run`: the crate's own resources, two folders up from target/<profile>.
        let dev = tree("dev", &["src-tauri/target/debug/oaiy-desktop", "src-tauri/resources/cli/oaiy.mjs"]);
        assert_eq!(cli(&dev.join("src-tauri/target/debug"), None), Some(dev.join("src-tauri/resources/cli/oaiy.mjs")));

        for root in [windows, linux, mac, dev] {
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn where_tauri_says_is_used_and_a_copy_beside_the_program_comes_first() {
        let root = tree("told", &["opt/app/oaiy-desktop", "elsewhere/share/resources/cli/oaiy.mjs"]);
        let told = root.join("elsewhere/share");
        assert_eq!(cli(&root.join("opt/app"), None), None, "no layout this module knows");
        assert_eq!(cli(&root.join("opt/app"), Some(&told)), Some(told.join("resources/cli/oaiy.mjs")));

        std::fs::create_dir_all(root.join("opt/app/resources/cli")).expect("the folder is made");
        std::fs::write(root.join("opt/app/resources/cli/oaiy.mjs"), b"").expect("the file is written");
        assert_eq!(cli(&root.join("opt/app"), Some(&told)), Some(root.join("opt/app/resources/cli/oaiy.mjs")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn what_is_found_is_a_plain_path_with_no_step_back_in_it() {
        let linux = tree("plain", &["usr/bin/oaiy-desktop", "usr/lib/OAIY/resources/node/bin/node"]);
        let found = resource_from(&linux.join("usr/bin"), None, "node", |p| p.is_dir()).expect("the folder is found");
        assert!(found.components().all(|c| !matches!(c, Component::ParentDir | Component::CurDir)), "{}", found.display());
        assert_eq!(tidy(Path::new("a/b/../../../c")), PathBuf::from("../c"), "a step past the start is kept");
        let _ = std::fs::remove_dir_all(linux);
    }
}
