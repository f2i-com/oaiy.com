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
//!   read-only. It would fail half way through, so it is not offered.
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
        return manual("There is no release of OAIY for this kind of computer, so it cannot update itself. Look on the releases page.");
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
    let parts: Vec<_> = app.components().collect();
    if parts.iter().any(|c| matches!(c, Component::Normal(n) if *n == "AppTranslocation")) {
        return Some("This copy of OAIY was opened where it was downloaded, which macOS does from a read-only copy, so it cannot update itself. Drag OAIY to Applications, open it from there, and check again.".to_string());
    }
    if matches!(parts.get(1), Some(Component::Normal(n)) if *n == "Volumes") {
        return Some("This copy of OAIY is on a disk image or another disk, so it cannot update itself (an update is put in place through the Mac's own disk). Drag OAIY to Applications, open it from there, and check again.".to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const INSTALLED: &str = "/Applications/OAIY.app/Contents/MacOS/oaiy-desktop";

    fn manual(os: &str, arch: &str, bundle: Option<&str>) -> String {
        manual_at(os, arch, bundle, None)
    }

    fn manual_at(os: &str, arch: &str, bundle: Option<&str>, exe: Option<&str>) -> String {
        match auto_update_for(os, arch, bundle, exe.map(Path::new)) {
            AutoUpdate::No(why) => why,
            AutoUpdate::Yes => panic!("{os} {arch} {bundle:?} {exe:?} should not update itself"),
        }
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
    fn a_platform_with_no_release_is_manual_whatever_the_install() {
        for (os, arch, bundle) in [("macos", "x86_64", Some("app")), ("windows", "aarch64", Some("nsis")), ("linux", "aarch64", Some("appimage")), ("freebsd", "x86_64", None)] {
            let why = manual_at(os, arch, bundle, Some("/Applications/OAIY.app/Contents/MacOS/oaiy-desktop"));
            assert!(why.contains("no release of OAIY for this kind of computer"), "{os} {arch}");
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
            ("windows", "x86_64", Some("other"), None),
        ] {
            let why = manual_at(os, arch, bundle, exe);
            assert!(why.ends_with('.') && why.chars().next().unwrap().is_uppercase(), "{why}");
        }
    }
}
