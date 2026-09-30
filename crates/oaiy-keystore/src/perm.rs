//! What the keyfile provider accepts of a file, its directory and the directories above it: the rule, as a pure function of what the file system
//! reported, so that it is tested on every platform, and the thin glue that asks a Unix file system.
//!
//! The rule (design 4.5.1): a key file is `0600` in a `0700` directory, and "the loader refuses looser modes". Refused: any group or other permission on
//! either, a set-uid, set-gid or sticky bit, an owner-execute bit on a file, a symbolic link in place of either, a directory where a file is expected
//! (and the reverse), anything that is not a regular file (a FIFO, a socket, a device: opening a FIFO would block for ever), and (when the owner is known)
//! a file or directory that belongs to another user. The provider never repairs a mode: a store whose permissions were loosened by someone else may have
//! been read by someone else, and the owner must hear about it.
//!
//! The directories **above** the keys folder matter too: whoever can rename or remove the folder can put another in its place. [`check_ancestor`] refuses
//! an ancestor that belongs to someone other than this user or root, and one that group or others can write to unless it has the sticky bit (then only the
//! owner of an entry can rename it, as `/tmp` does).
//!
//! Windows has no POSIX modes; the keyfile provider is refused there outside tests (see `provider.rs`), and its files are checked for being regular files that
//! are not reparse points.

/// What is being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The keys directory.
    Dir,
    /// One key file.
    File,
}

/// What kind of thing a path is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    /// A regular file.
    Regular,
    /// A directory.
    Directory,
    /// A symbolic link.
    Symlink,
    /// A FIFO, a socket, a device, or anything else.
    Other,
}

/// What the file system reported about one path or one open descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Meta {
    /// What it is.
    pub kind: FileKind,
    /// The permission bits, as `st_mode & 0o7777`.
    pub mode: u32,
    /// The owner.
    pub uid: u32,
}

/// Applies the rule. `expected_uid` is the user this process runs as, when known.
pub fn check(kind: Kind, meta: &Meta, expected_uid: Option<u32>) -> Result<(), String> {
    match (kind, meta.kind) {
        (_, FileKind::Symlink) => return Err("is a symbolic link".into()),
        (Kind::Dir, FileKind::Directory) | (Kind::File, FileKind::Regular) => {}
        (Kind::Dir, _) => return Err("is not a directory".into()),
        (Kind::File, FileKind::Directory) => return Err("is a directory".into()),
        (Kind::File, _) => return Err("is not a regular file (a FIFO, a socket or a device)".into()),
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

/// The rule for a directory above the keys folder: a directory, owned by this user or root, that group and others cannot write to unless it is sticky.
pub fn check_ancestor(meta: &Meta, uid: u32) -> Result<(), String> {
    if meta.kind != FileKind::Directory {
        return Err("is not a directory".into());
    }
    if meta.uid != 0 && meta.uid != uid {
        return Err(format!("belongs to another user (uid {})", meta.uid));
    }
    let mode = meta.mode & 0o7777;
    if mode & 0o022 != 0 && mode & 0o1000 == 0 {
        return Err(format!("mode {mode:04o} lets others rename what is inside it, and it is not sticky"));
    }
    Ok(())
}

/// What the file system reported in `m` (from an open descriptor, or from `symlink_metadata` of a path), as the rule wants it.
#[cfg(unix)]
pub(crate) fn meta_of(m: &std::fs::Metadata) -> Meta {
    use std::os::unix::fs::MetadataExt;
    let t = m.file_type();
    let kind = if t.is_symlink() {
        FileKind::Symlink
    } else if t.is_dir() {
        FileKind::Directory
    } else if t.is_file() {
        FileKind::Regular
    } else {
        FileKind::Other
    };
    Meta { kind, mode: m.mode() & 0o7777, uid: m.uid() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(mode: u32) -> Meta {
        Meta { kind: FileKind::Regular, mode, uid: 1000 }
    }

    fn dir(mode: u32) -> Meta {
        Meta { kind: FileKind::Directory, mode, uid: 1000 }
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
        link.kind = FileKind::Symlink;
        assert!(check(Kind::File, &link, None).unwrap_err().contains("symbolic link"));
        let mut dir_link = dir(0o700);
        dir_link.kind = FileKind::Symlink;
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

    /// A FIFO, a socket or a device with mode 0600 is not a key file: opening a FIFO for reading blocks until a writer appears, so a `get` on one would
    /// never return.
    #[test]
    fn anything_that_is_not_a_regular_file_is_refused_however_good_its_mode() {
        let other = Meta { kind: FileKind::Other, mode: 0o600, uid: 1000 };
        let message = check(Kind::File, &other, Some(1000)).unwrap_err();
        assert!(message.contains("not a regular file"), "{message}");
        assert!(check(Kind::Dir, &other, Some(1000)).is_err());
    }

    /// The directories above the keys folder (the reviewer's world-writable parent): owned by this user or root, and not writable by others unless sticky.
    #[test]
    fn an_ancestor_that_others_can_rename_in_is_refused_unless_it_is_sticky() {
        let me = 1000;
        for (mode, uid, ok) in [
            (0o755, me, true),
            (0o700, me, true),
            (0o755, 0, true),
            (0o1777, 0, true),  // /tmp
            (0o1777, me, true), // a sticky directory of mine
            (0o777, me, false), // world-writable, not sticky
            (0o777, 0, false),
            (0o775, me, false), // group-writable, not sticky
            (0o757, me, false),
            (0o770, me, false),
            (0o755, 1001, false), // another user's
            (0o1777, 1001, false),
            (0o700, 65534, false),
        ] {
            let meta = Meta { kind: FileKind::Directory, mode, uid };
            assert_eq!(check_ancestor(&meta, me).is_ok(), ok, "mode {mode:04o} uid {uid}");
        }
        let not_dir = Meta { kind: FileKind::Regular, mode: 0o755, uid: me };
        assert!(check_ancestor(&not_dir, me).is_err());
        // the message says which rule was broken
        let writable = Meta { kind: FileKind::Directory, mode: 0o777, uid: me };
        assert!(check_ancestor(&writable, me).unwrap_err().contains("not sticky"));
        let foreign = Meta { kind: FileKind::Directory, mode: 0o755, uid: 1001 };
        assert!(check_ancestor(&foreign, me).unwrap_err().contains("another user"));
    }

    #[cfg(unix)]
    #[test]
    fn what_the_file_system_reports_is_read_for_a_real_file_and_a_real_directory() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("oaiy-keystore-perm-{}-{}", std::process::id(), line!()));
        fs::create_dir_all(&base).unwrap();
        let path = base.join("a");
        fs::write(&path, b"x").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        let read = |p: &std::path::Path| meta_of(&fs::symlink_metadata(p).unwrap());
        let file_meta = read(&path);
        assert_eq!((file_meta.kind, file_meta.mode), (FileKind::Regular, 0o640));
        assert_eq!(read(&base).kind, FileKind::Directory);
        let link = base.join("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert_eq!(read(&link).kind, FileKind::Symlink);
        // a descriptor reports the same as the path does
        let open = fs::File::open(&path).unwrap();
        assert_eq!(meta_of(&open.metadata().unwrap()), file_meta);
        fs::remove_dir_all(&base).unwrap();
    }
}
