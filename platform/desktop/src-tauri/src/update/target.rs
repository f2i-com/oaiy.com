//! What an update installs on each platform: which kind of installer, and which name it may have been signed under.
//!
//! Two installs can update themselves (see [`super::kind`]), and each has exactly one kind of installer:
//!
//! | platform key | installer | the bundler's name for it |
//! |---|---|---|
//! | `windows-x86_64` | the NSIS setup | `OAIY_<v>_x64-setup.exe` |
//! | `linux-x86_64` | the AppImage | `OAIY_<v>_amd64.AppImage` |
//!
//! The Tauri CLI writes the name of the file it signed into the signature's trusted comment (`file:<name>`), and the
//! trusted comment is covered by the signature. That is what ties a signature to a version: the feed's version number is
//! text anyone who can change the feed can change, but the name inside a genuine signature cannot be changed.

/// The two things an update can install.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// The NSIS `setup.exe` (Windows, x86_64).
    WindowsSetup,
    /// The AppImage (Linux, x86_64).
    LinuxAppImage,
}

/// Why a name a signature was made for does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameProblem {
    /// Not a file of this kind (it does not end the way one does, or it is a path, not a name).
    NotThisKind,
    /// It is a file of this kind, but not for the announced version.
    NotThisVersion,
}

impl Target {
    /// The installer for the feed's platform key (`windows-x86_64`, `linux-x86_64`), or None.
    pub fn for_platform_key(key: &str) -> Option<Target> {
        match key {
            "windows-x86_64" => Some(Target::WindowsSetup),
            "linux-x86_64" => Some(Target::LinuxAppImage),
            _ => None,
        }
    }

    /// The installer for the platform this build runs on, or None (a platform no release is built for).
    pub fn current() -> Option<Target> {
        super::feed::current_platform_key().and_then(Target::for_platform_key)
    }

    /// The feed's platform key.
    pub fn platform_key(self) -> &'static str {
        match self {
            Target::WindowsSetup => "windows-x86_64",
            Target::LinuxAppImage => "linux-x86_64",
        }
    }

    /// How the name of such a file ends (the bundler's, and the release's).
    pub fn file_suffix(self) -> &'static str {
        match self {
            Target::WindowsSetup => "-setup.exe",
            Target::LinuxAppImage => ".AppImage",
        }
    }

    /// What it is, in words for a person.
    pub fn what(self) -> &'static str {
        match self {
            Target::WindowsSetup => "the Windows installer (a setup.exe)",
            Target::LinuxAppImage => "the Linux AppImage",
        }
    }

    /// Whether the file name a signature says it was made for (the `file:` of its trusted comment: the Tauri CLI writes the
    /// name of the file it signed, `OAIY_0.2.0_x64-setup.exe`) is a file of this kind for `version`.
    ///
    /// Only two things of the name are looked at, on every platform: it ends the way this kind of file's name does, and the
    /// version is one whole part of it. The parts are what the underscores separate, which is how the bundler names its
    /// files (`OAIY_0.2.0_x64-setup.exe`), and an underscore cannot be in a version, so `0.2.0` is never taken from
    /// `10.2.0`, `0.2.05` or `0.2.0-rc.1`. The product name and the architecture are not looked at: the Windows setup has
    /// one name, but the AppImage's has changed between Tauri versions, and neither adds to what stops a downgrade.
    pub fn signed_name_fits(self, file: &str, version: &str) -> Result<(), NameProblem> {
        if file.contains(['/', '\\']) {
            return Err(NameProblem::NotThisKind);
        }
        let Some(stem) = file.strip_suffix(self.file_suffix()) else { return Err(NameProblem::NotThisKind) };
        if !version.is_empty() && stem.split('_').any(|part| part == version) {
            Ok(())
        } else {
            Err(NameProblem::NotThisVersion)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_platform_key_has_its_own_kind_of_installer_and_nothing_else_has_one() {
        assert_eq!(Target::for_platform_key("windows-x86_64"), Some(Target::WindowsSetup));
        assert_eq!(Target::for_platform_key("linux-x86_64"), Some(Target::LinuxAppImage));
        for key in ["darwin-aarch64", "windows-aarch64", "linux-aarch64", "windows-x86_64-msi", "Windows-x86_64", ""] {
            assert_eq!(Target::for_platform_key(key), None, "{key}");
        }
        for target in [Target::WindowsSetup, Target::LinuxAppImage] {
            assert_eq!(Target::for_platform_key(target.platform_key()), Some(target));
        }
    }

    #[test]
    fn a_name_the_bundler_gave_fits_its_own_version_and_kind() {
        assert_eq!(Target::WindowsSetup.signed_name_fits("OAIY_0.1.0_x64-setup.exe", "0.1.0"), Ok(()));
        assert_eq!(Target::LinuxAppImage.signed_name_fits("OAIY_0.1.0_amd64.AppImage", "0.1.0"), Ok(()));
        // The AppImage's name is not tied to the architecture or the product: the version and the ending are what is looked at.
        for name in ["OAIY_0.1.0_x86_64.AppImage", "oaiy_0.1.0.AppImage", "0.1.0_.AppImage"] {
            assert_eq!(Target::LinuxAppImage.signed_name_fits(name, "0.1.0"), Ok(()), "{name}");
        }
    }

    #[test]
    fn a_name_for_another_version_never_fits_however_close_it_looks() {
        for name in ["OAIY_0.1.0_x64-setup.exe", "OAIY_10.1.0_x64-setup.exe", "OAIY_0.1.05_x64-setup.exe", "OAIY_0.1.0-rc.1_x64-setup.exe", "OAIY_0.1.0.1_x64-setup.exe", "OAIY_x0.1.0_x64-setup.exe", "OAIY_v0.1.0_x64-setup.exe", "OAIY-0.1.0-x64-setup.exe", "OAIY_x64-setup.exe", "-setup.exe"] {
            let wanted = if name == "OAIY_0.1.0_x64-setup.exe" { Ok(()) } else { Err(NameProblem::NotThisVersion) };
            assert_eq!(Target::WindowsSetup.signed_name_fits(name, "0.1.0"), wanted, "{name}");
        }
        assert_eq!(Target::WindowsSetup.signed_name_fits("OAIY_0.1.0_x64-setup.exe", "9.9.9"), Err(NameProblem::NotThisVersion));
        assert_eq!(Target::WindowsSetup.signed_name_fits("OAIY_0.1.0_x64-setup.exe", ""), Err(NameProblem::NotThisVersion));
        // An empty version is in no name, not even in one with an empty part.
        for name in ["OAIY__x64-setup.exe", "-setup.exe", "_-setup.exe"] {
            assert_eq!(Target::WindowsSetup.signed_name_fits(name, ""), Err(NameProblem::NotThisVersion), "{name}");
        }
    }

    #[test]
    fn a_name_of_the_other_kind_or_not_a_name_never_fits() {
        // The Windows setup where the AppImage belongs, and the other way round.
        assert_eq!(Target::LinuxAppImage.signed_name_fits("OAIY_0.1.0_x64-setup.exe", "0.1.0"), Err(NameProblem::NotThisKind));
        assert_eq!(Target::WindowsSetup.signed_name_fits("OAIY_0.1.0_amd64.AppImage", "0.1.0"), Err(NameProblem::NotThisKind));
        for name in ["OAIY_0.1.0_x64.msi", "OAIY_0.1.0_amd64.deb", "OAIY_0.1.0_x64-setup.exe.zip", "OAIY_0.1.0_x64-setup.EXE", "OAIY_0.1.0_amd64.appimage", "OAIY_0.1.0", "", "installer.bin", "../OAIY_0.1.0_x64-setup.exe", "C:\\OAIY_0.1.0_x64-setup.exe", "dir/OAIY_0.1.0_x64-setup.exe"] {
            assert_eq!(Target::WindowsSetup.signed_name_fits(name, "0.1.0"), Err(NameProblem::NotThisKind), "{name}");
        }
    }
}
