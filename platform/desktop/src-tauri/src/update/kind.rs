//! Whether this copy of OAIY can replace itself, and why not when it cannot.
//!
//! Three installs can: the Windows NSIS installer (`setup.exe`), which installs for the current
//! user and is what the feed's `windows-x86_64` entry is, the Linux AppImage (`linux-x86_64`), and
//! the app on a Mac with Apple silicon (`darwin-aarch64`), dragged from the disk image into a folder
//! of the Mac's own disk. Every other kind of install is a manual download from the releases page,
//! however the feed reads:
//!
//! - the MSI is machine-wide, a second kind of install of the same product; the updater plugin,
//!   given an MSI install, falls back from `windows-x86_64-msi` to the `windows-x86_64` entry and
//!   would run the NSIS installer over it, leaving two installs;
//! - a `.deb` or `.rpm` install would be handed the AppImage the feed names for `linux-x86_64`;
//! - a build that was not installed at all (`tauri dev`, `cargo run`) has nothing to replace;
//! - an Intel Mac has no release;
//! - a Mac app still in its disk image, on another disk, or in the copy macOS opens an app from when it
//!   was opened where it was downloaded (App Translocation): the updater moves the app through the
//!   Mac's temporary folder, which only works on the disk that folder is on, and the other two are
//!   read-only. It would fail half way through, so it is not offered;
//! - a Mac app at a path with a quote, a backslash or a control character in it: where the updater cannot move the app
//!   itself, it asks macOS for an administrator's password by an AppleScript it writes the path into unescaped, which such a
//!   character breaks (a `"` or a `\` stops the script from compiling, and the plugin then panics; a `'` changes the shell
//!   command inside it, which removes the app as root).
//!
//! On a Mac two more are looked at on disk, as the status is read and so before anything is stopped
//! ([`mac_install_problem`]): the app must still be where it was opened (tauri keeps the path the program started from,
//! which the updater replaces), and this Mac user must be able to move it (it and the folder it is in). The updater can
//! replace an app the user cannot move only by that AppleScript: a password dialog that waits for as long as nobody answers,
//! with OAIY stopped and its calls down meanwhile, and then `rm -rf` of the old app before the new one is moved in, as root,
//! so a move that fails leaves no app at all. OAIY does not go that way: it says so up front and points to the disk image,
//! where Finder asks for the password itself while OAIY keeps running.
//!
//! The kind of install is the bundle type the bundler stamps into the program (`tauri::utils::platform::bundle_type`);
//! it is read in `update::gui` and given here by name. On a Mac the bundler stamps nothing and tauri says `app` for every
//! build, `cargo run`'s too, so there the program's own path decides ([`bundle_name`]): an installed app's program is
//! `<name>.app/Contents/MacOS/<program>`, which is also how the updater plugin finds the app it replaces. Given anything
//! else, the plugin would replace the folder the program is in.

use std::path::{Component, Path};

use super::feed::platform_key;
use super::updater::AutoUpdate;

/// Whether an install on `os` and `arch` (`std::env::consts` spellings) of bundle type `bundle`
/// (`nsis`, `msi`, `appimage`, `deb`, `rpm`, `app`; None: not installed from a package) can update itself.
/// `exe` is the program's path, which says on a Mac where the app is (see [`mac_app_problem`]).
pub fn auto_update_for(os: &str, arch: &str, bundle: Option<&str>, exe: Option<&Path>) -> AutoUpdate {
    let manual = |why: &str| AutoUpdate::No(why.to_string());
    if platform_key(os, arch).is_none() {
        return AutoUpdate::NoRelease;
    }
    match (os, bundle) {
        ("windows", Some("nsis")) | ("linux", Some("appimage")) => AutoUpdate::Yes,
        ("macos", Some("app")) => match exe.and_then(app_bundle_of) {
            Some(app) => match mac_app_problem(app) {
                Some(why) => AutoUpdate::No(why),
                None => AutoUpdate::Yes,
            },
            None => manual("This copy of OAIY was not installed from a release (a development build, say), so it cannot update itself."),
        },
        ("windows", Some("msi")) => manual("This copy of OAIY was installed from the MSI package, which cannot update itself. Download the newer setup.exe from the releases page and run it (uninstall this one first)."),
        ("linux", Some("deb")) => manual("This copy of OAIY was installed from a .deb package, which cannot update itself. Download the newer package from the releases page."),
        ("linux", Some("rpm")) => manual("This copy of OAIY was installed from an .rpm package, which cannot update itself. Download the newer package from the releases page."),
        (_, None) => manual("This copy of OAIY was not installed from a release (a development build, say), so it cannot update itself."),
        _ => manual("This kind of install cannot update itself. Download the newer version from the releases page."),
    }
}

/// The bundle type by the name [`auto_update_for`] takes. On a Mac tauri says `app` of every build, so `app` stands only
/// when the program at `exe` is in an app bundle; any other is not installed (None).
#[cfg(feature = "gui")]
pub fn bundle_name(bundle: Option<tauri::utils::config::BundleType>, exe: Option<&Path>) -> Option<&'static str> {
    use tauri::utils::config::BundleType;
    bundle.and_then(|b| match b {
        BundleType::Nsis => Some("nsis"),
        BundleType::Msi => Some("msi"),
        BundleType::AppImage => Some("appimage"),
        BundleType::Deb => Some("deb"),
        BundleType::Rpm => Some("rpm"),
        BundleType::App | BundleType::Dmg => exe.and_then(app_bundle_of).map(|_| "app"),
    })
}

/// The app bundle a Mac program is the program of: `/Applications/OAIY.app` for `/Applications/OAIY.app/Contents/MacOS/oaiy-desktop`,
/// and None for a program anywhere else (`target/release/oaiy-desktop`). The updater plugin takes the same two folders up
/// from the program to find what it replaces, so this is what it would replace.
pub fn app_bundle_of(exe: &Path) -> Option<&Path> {
    let macos = exe.parent()?;
    let contents = macos.parent()?;
    let app = contents.parent()?;
    let named = |dir: &Path, name: &str| dir.file_name().is_some_and(|n| n == name);
    let is_app = app.extension().is_some_and(|e| e == "app") && app.file_stem().is_some_and(|s| !s.is_empty());
    (exe.is_absolute() && named(macos, "MacOS") && named(contents, "Contents") && is_app).then_some(app)
}

/// Why the Mac app at `app` cannot be replaced where it is, or None when it can: it is in a folder of the Mac's own disk,
/// where the updater can move it aside and the new one in (through the temporary folder, which is on that disk). A path
/// under `/Volumes` is a disk image or another disk; one with an `AppTranslocation` folder is the read-only copy macOS
/// opens an app from when it was opened where it was downloaded.
pub fn mac_app_problem(app: &Path) -> Option<String> {
    if !path_fits_the_updaters_script(app) {
        return Some(PATH_WITH_QUOTES.to_string());
    }
    let parts: Vec<_> = app.components().collect();
    if parts.iter().any(|c| matches!(c, Component::Normal(n) if *n == "AppTranslocation")) {
        return Some("This copy of OAIY was opened where it was downloaded, which macOS does from a read-only copy, so it cannot update itself. Drag OAIY to Applications, open it from there, and check again.".to_string());
    }
    if matches!(parts.get(1), Some(Component::Normal(n)) if *n == "Volumes") {
        return Some("This copy of OAIY is on a disk image or another disk, so it cannot update itself (an update is put in place through the Mac's own disk). Drag OAIY to Applications, open it from there, and check again.".to_string());
    }
    None
}

/// Why an app at a path with a quote, a backslash or a control character in it is not updated (see the module's docs).
const PATH_WITH_QUOTES: &str = "This copy of OAIY is at a path with a quote, a backslash or a control character in it, which the updater cannot take, so it cannot update itself. Rename it, or move it to a folder whose path has no quote, backslash or control character, open it from there, and check again.";

/// Whether the updater plugin's AppleScript can take `path` as it writes it (between `'` inside a `"` string, unescaped):
/// text with no `'`, `"`, `\` or control character. A path that is not text is refused too (the plugin would write
/// another one in its place).
pub fn path_fits_the_updaters_script(path: &Path) -> bool {
    path.to_str().is_some_and(|text| !text.chars().any(|c| matches!(c, '\'' | '"' | '\\') || c.is_control()))
}

/// Why the Mac app at `app` (whose program is `exe`) cannot be replaced as the disk has it now, or None when it can: it must
/// still be where it was opened, and this user must be able to move it and to write in the folder it is in, which is what
/// the updater does with it (it moves it aside to `temp`, the Mac's temporary folder, whose path is held to the same rule
/// as the app's). Asked as the status is read, so before anything is stopped (see the module's docs for why a folder this
/// user cannot change is refused rather than left to macOS's password dialog).
#[cfg(unix)]
pub fn mac_install_problem(exe: &Path, app: &Path, temp: &Path) -> Option<String> {
    if !exe.is_file() || !app.join("Contents").join("Info.plist").is_file() {
        return Some("OAIY was moved since it was opened: quit it and open it again.".to_string());
    }
    if !path_fits_the_updaters_script(temp) {
        return Some("The Mac's temporary folder has a quote, a backslash or a control character in its path, which the updater cannot take, so OAIY cannot update itself. Install the newer version from the disk image on the releases page.".to_string());
    }
    let folder = app.parent().unwrap_or(Path::new("/"));
    let refuse = |what: String| Some(format!("This Mac user cannot change {what}, so OAIY cannot update itself there: macOS would have to ask for an administrator's password while OAIY is stopped. Install the newer version from the disk image on the releases page, or move OAIY to a folder of your own, open it from there, and check again."));
    if !may_write(folder) {
        return refuse(format!("the folder OAIY is in ({})", folder.display()));
    }
    // Moving a folder to another one rewrites its `..`, so the app itself must be this user's to write as well; and in a
    // folder with the sticky bit (as /tmp has) only the owner of the app or of the folder may move it.
    let own = owned_by_this_user(app);
    if !may_write(app) || (is_sticky(folder) && !own && !owned_by_this_user(folder)) {
        let whose = if own { "" } else { " (it belongs to another user)" };
        return refuse(format!("{} in {}{whose}", app.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(), folder.display()));
    }
    None
}

/// Whether this process may write in `dir` and look into it (access(2): its owner, its group, its access list).
#[cfg(unix)]
fn may_write(dir: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(path) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else { return false };
    // SAFETY: a valid C string, read by access(2) and not kept.
    unsafe { libc::access(path.as_ptr(), libc::W_OK | libc::X_OK) == 0 }
}

#[cfg(unix)]
fn is_sticky(dir: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(dir).is_ok_and(|m| m.permissions().mode() & 0o1000 != 0)
}

#[cfg(unix)]
fn owned_by_this_user(path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    // SAFETY: getuid(2) cannot fail and touches nothing.
    std::fs::metadata(path).is_ok_and(|m| m.uid() == unsafe { libc::getuid() })
}

#[cfg(test)]
mod tests {
    use super::*;

    const INSTALLED: &str = "/Applications/OAIY.app/Contents/MacOS/oaiy-desktop";

    fn manual(os: &str, arch: &str, bundle: Option<&str>) -> String {
        manual_at(os, arch, bundle, None)
    }

    fn manual_at(os: &str, arch: &str, bundle: Option<&str>, exe: Option<&str>) -> String {
        auto_update_for(os, arch, bundle, exe.map(Path::new)).reason().unwrap_or_else(|| panic!("{os} {arch} {bundle:?} {exe:?} should not update itself"))
    }

    #[test]
    fn the_windows_setup_exe_the_linux_appimage_and_an_installed_mac_app_can_update_themselves() {
        assert_eq!(auto_update_for("windows", "x86_64", Some("nsis"), None), AutoUpdate::Yes);
        assert_eq!(auto_update_for("linux", "x86_64", Some("appimage"), None), AutoUpdate::Yes);
        for exe in [INSTALLED, "/Users/sam/Applications/OAIY.app/Contents/MacOS/oaiy-desktop", "/Applications/Utilities/OAIY 2.app/Contents/MacOS/oaiy-desktop"] {
            assert_eq!(auto_update_for("macos", "aarch64", Some("app"), Some(Path::new(exe))), AutoUpdate::Yes, "{exe}");
        }
    }

    #[test]
    fn the_msi_is_a_manual_download_because_the_feeds_windows_entry_is_the_nsis_installer() {
        let why = manual("windows", "x86_64", Some("msi"));
        assert!(why.contains("MSI") && why.contains("setup.exe") && why.contains("releases page"), "{why}");
    }

    #[test]
    fn deb_and_rpm_installs_are_manual_downloads() {
        assert!(manual("linux", "x86_64", Some("deb")).contains(".deb"));
        assert!(manual("linux", "x86_64", Some("rpm")).contains(".rpm"));
    }

    #[test]
    fn a_build_that_was_not_installed_from_a_release_has_nothing_to_replace() {
        assert!(manual("windows", "x86_64", None).contains("not installed from a release"));
        assert!(manual("linux", "x86_64", None).contains("development build"));
        assert!(manual("macos", "aarch64", None).contains("development build"));
        // On a Mac tauri says "app" of every build: a program that is not in an app bundle (cargo run's, tauri dev's) is not
        // installed, and the updater would otherwise replace the folder it is in.
        for exe in [None, Some("/Users/sam/oaiy.com/target/release/oaiy-desktop"), Some("/Users/sam/oaiy.com/target/debug/oaiy-desktop"), Some("Applications/OAIY.app/Contents/MacOS/oaiy-desktop")] {
            assert!(manual_at("macos", "aarch64", Some("app"), exe).contains("not installed from a release"), "{exe:?}");
        }
    }

    #[test]
    fn a_program_is_an_installed_mac_apps_only_inside_its_bundle() {
        assert_eq!(app_bundle_of(Path::new(INSTALLED)), Some(Path::new("/Applications/OAIY.app")));
        assert_eq!(app_bundle_of(Path::new("/OAIY.app/Contents/MacOS/oaiy-server")), Some(Path::new("/OAIY.app")));
        for exe in [
            "/Applications/OAIY.app/Contents/Resources/oaiy-desktop",
            "/Applications/OAIY.app/MacOS/oaiy-desktop",
            "/Applications/OAIY/Contents/MacOS/oaiy-desktop",
            "/Applications/.app/Contents/MacOS/oaiy-desktop",
            "/Applications/OAIY.app/Contents/MacOS",
            "/Contents/MacOS/oaiy-desktop",
            "/usr/local/bin/oaiy-desktop",
            "OAIY.app/Contents/MacOS/oaiy-desktop",
            "",
        ] {
            assert_eq!(app_bundle_of(Path::new(exe)), None, "{exe}");
        }
    }

    #[test]
    fn a_mac_app_in_its_disk_image_on_another_disk_or_opened_where_it_was_downloaded_cannot_update_itself() {
        for exe in ["/Volumes/OAIY/OAIY.app/Contents/MacOS/oaiy-desktop", "/Volumes/T9/Apps/OAIY.app/Contents/MacOS/oaiy-desktop"] {
            let why = manual_at("macos", "aarch64", Some("app"), Some(exe));
            assert!(why.contains("disk image or another disk") && why.contains("Applications"), "{why}");
        }
        let translocated = "/private/var/folders/x1/abc/T/AppTranslocation/4F1E5C2A-0000-4000-8000-000000000000/d/OAIY.app/Contents/MacOS/oaiy-desktop";
        let why = manual_at("macos", "aarch64", Some("app"), Some(translocated));
        assert!(why.contains("where it was downloaded") && why.contains("Applications"), "{why}");
        // A folder merely named like one of them is not one.
        for exe in ["/Applications/Volumes/OAIY.app/Contents/MacOS/oaiy-desktop", "/Users/sam/AppTranslocations/OAIY.app/Contents/MacOS/oaiy-desktop"] {
            assert_eq!(auto_update_for("macos", "aarch64", Some("app"), Some(Path::new(exe))), AutoUpdate::Yes, "{exe}");
        }
    }

    #[test]
    fn a_mac_app_at_a_path_the_updaters_script_cannot_take_cannot_update_itself() {
        for exe in [
            "/Users/sam/Sam's Apps/OAIY.app/Contents/MacOS/oaiy-desktop",
            "/Users/sam/\"Apps\"/OAIY.app/Contents/MacOS/oaiy-desktop",
            "/Users/sam/Apps\\Old/OAIY.app/Contents/MacOS/oaiy-desktop",
            "/Users/sam/Apps\nOld/OAIY.app/Contents/MacOS/oaiy-desktop",
            "/Applications/OAIY's.app/Contents/MacOS/oaiy-desktop",
        ] {
            let why = manual_at("macos", "aarch64", Some("app"), Some(exe));
            assert!(why.contains("a quote, a backslash or a control character") && why.contains("Rename it, or move it to a folder whose path has no quote"), "{exe:?}: {why}");
            // (an app in Applications already, /Applications/OAIY's.app, is not told to go there)
            assert!(!why.contains("to Applications"), "{exe:?}: {why}");
        }
        // Spaces, letters of other languages and the shell's other signs are written between the quotes as they are.
        for exe in ["/Users/sam/My Apps (2)/OAIY.app/Contents/MacOS/oaiy-desktop", "/Users/zoë/Programme/OAIY.app/Contents/MacOS/oaiy-desktop", "/Users/sam/$HOME & `x`/OAIY.app/Contents/MacOS/oaiy-desktop"] {
            assert_eq!(auto_update_for("macos", "aarch64", Some("app"), Some(Path::new(exe))), AutoUpdate::Yes, "{exe}");
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!path_fits_the_updaters_script(Path::new(std::ffi::OsStr::from_bytes(b"/Users/sam/\xff/OAIY.app"))), "a path that is not text");
        }
    }

    /// An app bundle as an install leaves it, in a folder of its own under the temporary folder: (that folder, the app, its program).
    #[cfg(unix)]
    fn installed_app(name: &str) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        let folder = std::env::temp_dir().join(format!("oaiy-kind-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        let app = folder.join("OAIY.app");
        std::fs::create_dir_all(app.join("Contents/MacOS")).unwrap();
        std::fs::write(app.join("Contents/Info.plist"), "<plist/>").unwrap();
        std::fs::write(app.join("Contents/MacOS/oaiy-desktop"), "program").unwrap();
        let exe = app.join("Contents/MacOS/oaiy-desktop");
        (folder, app, exe)
    }

    #[cfg(unix)]
    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn an_app_this_user_can_move_from_where_it_was_opened_can_be_replaced() {
        let (folder, app, exe) = installed_app("fine");
        assert_eq!(mac_install_problem(&exe, &app, &std::env::temp_dir()), None);
        // A folder with the sticky bit is fine when the app is this user's.
        set_mode(&folder, 0o1777);
        assert_eq!(mac_install_problem(&exe, &app, &std::env::temp_dir()), None);
        set_mode(&folder, 0o755);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[cfg(unix)]
    #[test]
    fn an_app_moved_since_it_was_opened_is_said_to_be_before_anything_is_stopped() {
        let (folder, app, exe) = installed_app("moved");
        std::fs::rename(&app, folder.join("Elsewhere.app")).unwrap();
        assert_eq!(mac_install_problem(&exe, &app, &std::env::temp_dir()).as_deref(), Some("OAIY was moved since it was opened: quit it and open it again."));
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[cfg(unix)]
    #[test]
    fn an_app_this_user_cannot_move_is_not_updated_and_the_disk_image_is_named_instead() {
        // (root may write anywhere: access(2) says yes to it whatever the modes)
        if unsafe { libc::getuid() } == 0 {
            return;
        }
        let (folder, app, exe) = installed_app("locked");
        // The folder it is in (/Applications for a user who is not an administrator).
        set_mode(&folder, 0o555);
        let why = mac_install_problem(&exe, &app, &std::env::temp_dir()).expect("a folder this user cannot write in");
        set_mode(&folder, 0o755);
        assert!(why.starts_with(&format!("This Mac user cannot change the folder OAIY is in ({})", folder.display())), "{why}");
        assert!(why.contains("administrator's password") && why.contains("disk image on the releases page"), "{why}");
        // The app itself (moving it to another folder rewrites its `..`).
        set_mode(&app, 0o555);
        let why = mac_install_problem(&exe, &app, &std::env::temp_dir()).expect("an app this user cannot write in");
        set_mode(&app, 0o755);
        assert!(why.starts_with(&format!("This Mac user cannot change OAIY.app in {}, so", folder.display())), "{why}");
        assert!(why.ends_with('.'));
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[cfg(unix)]
    #[test]
    fn a_temporary_folder_the_updaters_script_cannot_take_is_refused_too() {
        let (folder, app, exe) = installed_app("temp");
        let why = mac_install_problem(&exe, &app, Path::new("/private/var/folders/x1/Sam's/T/")).expect("a quote in the temporary folder");
        assert!(why.contains("temporary folder") && why.contains("disk image"), "{why}");
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn a_platform_with_no_release_is_manual_whatever_the_install() {
        for (os, arch, bundle) in [("macos", "x86_64", Some("app")), ("windows", "aarch64", Some("nsis")), ("linux", "aarch64", Some("appimage")), ("freebsd", "x86_64", None)] {
            // Said as a kind of its own (the status's `no_release`), which the window goes by, and in words beside it.
            assert_eq!(auto_update_for(os, arch, bundle, Some(Path::new(INSTALLED))), AutoUpdate::NoRelease, "{os} {arch}");
            let why = manual_at(os, arch, bundle, Some(INSTALLED));
            assert!(why.contains("no release of OAIY for this kind of computer"), "{os} {arch}");
        }
        // Every other reason a copy cannot update itself is one where a release exists.
        for (os, arch, bundle, exe) in [("windows", "x86_64", Some("msi"), None), ("linux", "x86_64", Some("deb"), None), ("macos", "aarch64", Some("app"), None), ("macos", "aarch64", Some("app"), Some("/Volumes/OAIY/OAIY.app/Contents/MacOS/oaiy-desktop"))] {
            assert!(matches!(auto_update_for(os, arch, bundle, exe.map(Path::new)), AutoUpdate::No(_)), "{os} {arch} {bundle:?} {exe:?}");
        }
    }

    #[test]
    fn a_bundle_of_the_wrong_kind_for_the_os_is_manual_too() {
        // (These do not exist in practice; the answer must still be no.)
        let exe = Some(Path::new(INSTALLED));
        assert!(matches!(auto_update_for("windows", "x86_64", Some("appimage"), None), AutoUpdate::No(_)));
        assert!(matches!(auto_update_for("linux", "x86_64", Some("nsis"), None), AutoUpdate::No(_)));
        assert!(matches!(auto_update_for("windows", "x86_64", Some("other"), None), AutoUpdate::No(_)));
        assert!(matches!(auto_update_for("windows", "x86_64", Some("app"), exe), AutoUpdate::No(_)));
        assert!(matches!(auto_update_for("linux", "x86_64", Some("app"), exe), AutoUpdate::No(_)));
        assert!(matches!(auto_update_for("macos", "aarch64", Some("nsis"), exe), AutoUpdate::No(_)));
        assert!(matches!(auto_update_for("macos", "aarch64", Some("appimage"), exe), AutoUpdate::No(_)));
    }

    #[test]
    fn every_reason_is_a_sentence_for_a_person() {
        for (os, arch, bundle, exe) in [
            ("windows", "x86_64", Some("msi"), None),
            ("linux", "x86_64", Some("deb"), None),
            ("linux", "x86_64", Some("rpm"), None),
            ("windows", "x86_64", None, None),
            ("macos", "x86_64", Some("app"), Some(INSTALLED)),
            ("macos", "aarch64", Some("app"), None),
            ("macos", "aarch64", Some("app"), Some("/Volumes/OAIY/OAIY.app/Contents/MacOS/oaiy-desktop")),
            ("macos", "aarch64", Some("app"), Some("/private/var/folders/x/T/AppTranslocation/1/d/OAIY.app/Contents/MacOS/oaiy-desktop")),
            ("macos", "aarch64", Some("app"), Some("/Users/sam/Sam's Apps/OAIY.app/Contents/MacOS/oaiy-desktop")),
            ("windows", "x86_64", Some("other"), None),
        ] {
            let why = manual_at(os, arch, bundle, exe);
            assert!(why.ends_with('.') && why.chars().next().unwrap().is_uppercase(), "{why}");
        }
    }
}
