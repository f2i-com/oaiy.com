//! What the keyfile provider accepts of a file and its directory: the rule, as a pure function of what the file system reported, so that it is
//! tested on every platform, and the thin glue that asks a Unix file system.
//!
//! The rule (design 4.5.1): a key file is `0600` in a `0700` directory, and "the loader refuses looser modes". Refused: any group or other permission on
//! either, a set-uid, set-gid or sticky bit, an owner-execute bit on a file, a symbolic link in place of either, a directory where a file is expected
//! (and the reverse), and (when the owner is known) a file or directory that belongs to another user. The provider never repairs a mode: a store whose
//! permissions were loosened by someone else may have been read by someone else, and the owner must hear about it.
//!
//! Windows has no POSIX modes; the keyfile provider there refuses symbolic links and relies on the access control the data folder inherits, and says so
//! in its provider information (it is the weakest provider, and the default on Windows is DPAPI).

use std::fs;
use std::io;
use std::path::Path;

/// What is being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The keys directory.
    Dir,
    /// One key file.
    File,
}

/// What the file system reported about one path (without following a symbolic link).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Meta {
    /// A directory.
    pub is_dir: bool,
    /// A symbolic link.
    pub is_symlink: bool,
    /// The permission bits, as `st_mode & 0o7777`.
    pub mode: u32,
    /// The owner.
    pub uid: u32,
}

/// Applies the rule. `expected_uid` is the user this process runs as, when known.
pub fn check(kind: Kind, meta: &Meta, expected_uid: Option<u32>) -> Result<(), String> {
    if meta.is_symlink {
        return Err("is a symbolic link".into());
    }
    match kind {
        Kind::Dir if !meta.is_dir => return Err("is not a directory".into()),
        Kind::File if meta.is_dir => return Err("is a directory".into()),
        _ => {}
    }
    let mode = meta.mode & 0o7777;
    match kind {
        Kind::Dir if mode & 0o7077 != 0 => return Err(format!("mode {mode:04o} is looser than 0700")),
        Kind::File if mode & 0o7177 != 0 => return Err(format!("mode {mode:04o} is looser than 0600")),
        _ => {}
    }
    if let Some(uid) = expected_uid {
        if meta.uid != uid {
            return Err("belongs to another user".into());
        }
    }
    Ok(())
}

/// Asks the file system.
#[cfg(unix)]
pub(crate) fn read_meta(path: &Path) -> io::Result<Meta> {
    use std::os::unix::fs::MetadataExt;
    let m = fs::symlink_metadata(path)?;
    Ok(Meta { is_dir: m.is_dir(), is_symlink: m.file_type().is_symlink(), mode: m.mode() & 0o7777, uid: m.uid() })
}

/// Asks the file system (no modes to read: the values that pass the rule are reported, and only the type is checked).
#[cfg(not(unix))]
pub(crate) fn read_meta(path: &Path) -> io::Result<Meta> {
    let m = fs::symlink_metadata(path)?;
    Ok(Meta { is_dir: m.is_dir(), is_symlink: m.file_type().is_symlink(), mode: if m.is_dir() { 0o700 } else { 0o600 }, uid: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(mode: u32) -> Meta {
        Meta { is_dir: false, is_symlink: false, mode, uid: 1000 }
    }

    fn dir(mode: u32) -> Meta {
        Meta { is_dir: true, is_symlink: false, mode, uid: 1000 }
    }

    /// The whole table of modes: exactly 0600 and 0400 pass for a file, exactly 0700 (and the stricter ones) for a directory.
    #[test]
    fn keyfile_mode_refusal_over_every_permission_bit_pattern() {
        for mode in 0u32..=0o7777 {
            let file_ok = check(Kind::File, &file(mode), Some(1000)).is_ok();
            let dir_ok = check(Kind::Dir, &dir(mode), Some(1000)).is_ok();
            // a file: no group, no other, no owner-execute, no special bits
            assert_eq!(file_ok, mode & 0o7177 == 0, "file {mode:04o}");
            // a directory: no group, no other, no special bits; the owner may have any of rwx
            assert_eq!(dir_ok, mode & 0o7077 == 0, "dir {mode:04o}");
        }
        for ok in [0o600, 0o400, 0o200, 0o000] {
            assert!(check(Kind::File, &file(ok), None).is_ok(), "{ok:04o}");
        }
        for loose in [0o644, 0o640, 0o604, 0o660, 0o666, 0o700, 0o755, 0o777, 0o4600, 0o2600, 0o1600] {
            assert!(check(Kind::File, &file(loose), None).is_err(), "{loose:04o}");
        }
        for ok in [0o700, 0o500, 0o300, 0o100, 0o000] {
            assert!(check(Kind::Dir, &dir(ok), None).is_ok(), "{ok:04o}");
        }
        for loose in [0o755, 0o750, 0o705, 0o770, 0o777, 0o701, 0o710, 0o2700, 0o1700, 0o4700] {
            assert!(check(Kind::Dir, &dir(loose), None).is_err(), "{loose:04o}");
        }
    }

    #[test]
    fn a_symbolic_link_a_wrong_type_and_another_owner_are_refused() {
        let mut link = file(0o600);
        link.is_symlink = true;
        assert!(check(Kind::File, &link, None).unwrap_err().contains("symbolic link"));
        let mut dir_link = dir(0o700);
        dir_link.is_symlink = true;
        assert!(check(Kind::Dir, &dir_link, None).is_err());
        assert!(check(Kind::File, &dir(0o600), None).unwrap_err().contains("is a directory"));
        assert!(check(Kind::Dir, &file(0o700), None).unwrap_err().contains("not a directory"));
        assert!(check(Kind::File, &file(0o600), Some(1001)).unwrap_err().contains("another user"));
        assert!(check(Kind::Dir, &dir(0o700), Some(1001)).unwrap_err().contains("another user"));
        assert!(check(Kind::File, &file(0o600), Some(1000)).is_ok());
        // the message names the mode, and nothing secret is in it
        let message = check(Kind::File, &file(0o644), None).unwrap_err();
        assert!(message.contains("0644") && message.contains("0600"), "{message}");
    }

    #[test]
    fn what_the_file_system_reports_is_read_for_a_real_file_and_a_real_directory() {
        let base = std::env::temp_dir().join(format!("oaiy-keystore-perm-{}-{}", std::process::id(), line!()));
        fs::create_dir_all(&base).unwrap();
        let path = base.join("a");
        fs::write(&path, b"x").unwrap();
        let file_meta = read_meta(&path).unwrap();
        let dir_meta = read_meta(&base).unwrap();
        assert!(!file_meta.is_dir && !file_meta.is_symlink);
        assert!(dir_meta.is_dir && !dir_meta.is_symlink);
        assert!(read_meta(&base.join("missing")).is_err());
        fs::remove_dir_all(&base).unwrap();
    }
}
