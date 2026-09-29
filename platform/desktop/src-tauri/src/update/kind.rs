//! Whether this copy of OAIY can replace itself, and why not when it cannot.
//!
//! Only two installs can: the Windows NSIS installer (`setup.exe`), which installs for the current
//! user and is what the feed's `windows-x86_64` entry is, and the Linux AppImage (`linux-x86_64`).
//! Every other kind of install is a manual download from the releases page, however the feed reads:
//!
//! - the MSI is machine-wide, a second kind of install of the same product; the updater plugin,
//!   given an MSI install, falls back from `windows-x86_64-msi` to the `windows-x86_64` entry and
//!   would run the NSIS installer over it, leaving two installs;
//! - a `.deb` or `.rpm` install would be handed the AppImage the feed names for `linux-x86_64`;
//! - a build that was not installed at all (`tauri dev`, `cargo run`) has nothing to replace;
//! - macOS is not built.
//!
//! The kind of install is the bundle type the bundler stamps into the program (`tauri::utils::platform::bundle_type`);
//! it is read in `update::gui` and given here by name.

use super::feed::platform_key;
use super::updater::AutoUpdate;

/// Whether an install on `os` and `arch` (`std::env::consts` spellings) of bundle type `bundle`
/// (`nsis`, `msi`, `appimage`, `deb`, `rpm`, `app`; None: not installed from a package) can update itself.
pub fn auto_update_for(os: &str, arch: &str, bundle: Option<&str>) -> AutoUpdate {
    let manual = |why: &str| AutoUpdate::No(why.to_string());
    if platform_key(os, arch).is_none() {
        return manual("There is no release of OAIY for this kind of computer, so it cannot update itself. Look on the releases page.");
    }
    match (os, bundle) {
        ("windows", Some("nsis")) | ("linux", Some("appimage")) => AutoUpdate::Yes,
        ("windows", Some("msi")) => manual("This copy of OAIY was installed from the MSI package, which cannot update itself. Download the newer setup.exe from the releases page and run it (uninstall this one first)."),
        ("linux", Some("deb")) => manual("This copy of OAIY was installed from a .deb package, which cannot update itself. Download the newer package from the releases page."),
        ("linux", Some("rpm")) => manual("This copy of OAIY was installed from an .rpm package, which cannot update itself. Download the newer package from the releases page."),
        (_, None) => manual("This copy of OAIY was not installed from a release (a development build, say), so it cannot update itself."),
        _ => manual("This kind of install cannot update itself. Download the newer version from the releases page."),
    }
}

/// The bundle type by the name [`auto_update_for`] takes.
#[cfg(feature = "gui")]
pub fn bundle_name(bundle: Option<tauri::utils::config::BundleType>) -> Option<&'static str> {
    use tauri::utils::config::BundleType;
    bundle.map(|b| match b {
        BundleType::Nsis => "nsis",
        BundleType::Msi => "msi",
        BundleType::AppImage => "appimage",
        BundleType::Deb => "deb",
        BundleType::Rpm => "rpm",
        BundleType::App | BundleType::Dmg => "app",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manual(os: &str, arch: &str, bundle: Option<&str>) -> String {
        match auto_update_for(os, arch, bundle) {
            AutoUpdate::No(why) => why,
            AutoUpdate::Yes => panic!("{os} {arch} {bundle:?} should not update itself"),
        }
    }

    #[test]
    fn the_windows_setup_exe_and_the_linux_appimage_can_update_themselves() {
        assert_eq!(auto_update_for("windows", "x86_64", Some("nsis")), AutoUpdate::Yes);
        assert_eq!(auto_update_for("linux", "x86_64", Some("appimage")), AutoUpdate::Yes);
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
    }

    #[test]
    fn a_platform_with_no_release_is_manual_whatever_the_install() {
        for (os, arch, bundle) in [("macos", "aarch64", Some("app")), ("macos", "x86_64", Some("app")), ("windows", "aarch64", Some("nsis")), ("linux", "aarch64", Some("appimage")), ("freebsd", "x86_64", None)] {
            assert!(manual(os, arch, bundle).contains("no release of OAIY for this kind of computer"), "{os} {arch}");
        }
    }

    #[test]
    fn a_bundle_of_the_wrong_kind_for_the_os_is_manual_too() {
        // (These do not exist in practice; the answer must still be no.)
        assert!(matches!(auto_update_for("windows", "x86_64", Some("appimage")), AutoUpdate::No(_)));
        assert!(matches!(auto_update_for("linux", "x86_64", Some("nsis")), AutoUpdate::No(_)));
        assert!(matches!(auto_update_for("windows", "x86_64", Some("other")), AutoUpdate::No(_)));
    }

    #[test]
    fn every_reason_is_a_sentence_for_a_person() {
        for (os, arch, bundle) in [("windows", "x86_64", Some("msi")), ("linux", "x86_64", Some("deb")), ("linux", "x86_64", Some("rpm")), ("windows", "x86_64", None), ("macos", "aarch64", Some("app")), ("windows", "x86_64", Some("other"))] {
            let why = manual(os, arch, bundle);
            assert!(why.ends_with('.') && why.chars().next().unwrap().is_uppercase(), "{why}");
        }
    }
}
