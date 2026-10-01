//! Two questions about an open Windows handle that `std` answers only on nightly (`windows_by_handle`) or not at all: **which file is this** (the volume and the
//! file index, what the operating system itself says two handles have in common when they are one file), and **where is it really** (the final path, with every
//! junction and symbolic link on the way resolved). The keystore holds its keys folder open and needs both (review M-1): to compare the folder it holds with the
//! folder its path leads to now, and to work in the folder it holds, by its real path, and not in whatever a path with a junction in it leads to at this moment.
//!
//! This is the second module of the crate with `unsafe`, and the only other one (the first is `dpapi.rs`): three calls into `kernel32`, each on a handle that a
//! `&File` keeps open for the length of the call, each with a buffer that this module owns. Nothing here keeps a pointer after it returns.

use core::ffi::c_void;
use core::mem::size_of;
use std::ffi::OsString;
use std::fs::File;
use std::io;
use std::os::windows::ffi::OsStringExt;
use std::os::windows::io::AsRawHandle;
use std::path::PathBuf;

use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::Storage::FileSystem::{
    FileIdInfo, GetFileInformationByHandle, GetFileInformationByHandleEx, GetFinalPathNameByHandleW, BY_HANDLE_FILE_INFORMATION, FILE_ID_INFO,
    FILE_NAME_NORMALIZED, VOLUME_NAME_DOS, VOLUME_NAME_GUID,
};

/// What makes a file the same file: the volume it is on and its index there. Two handles with equal ids are two ways into one file or folder, whatever path each was
/// opened by (this is how a hard link, a junction and a subst drive are told from a different file).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileId {
    volume: u64,
    index: u128,
}

/// What the file system says about an open file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Info {
    /// Which file.
    pub id: FileId,
    /// How many names the file has (hard links). A folder has one.
    pub links: u32,
    /// The attribute bits (`FILE_ATTRIBUTE_*`).
    pub attributes: u32,
}

#[cfg(test)]
impl FileId {
    /// A made-up identity, for the tests of what is compared: there is no second volume to open in a test without an administrator.
    pub(crate) fn made_up(volume: u64, index: u128) -> FileId {
        FileId { volume, index }
    }

    pub(crate) fn parts(&self) -> (u64, u128) {
        (self.volume, self.index)
    }
}

#[cfg(test)]
impl Info {
    /// A made-up `Info`, for the tests of the rules that judge one: what a file symbolic link, a cloud placeholder or a file with two names look like cannot be made here without privileges.
    pub(crate) fn made_up(attributes: u32, links: u32) -> Info {
        Info { id: FileId { volume: 0, index: 0 }, links, attributes }
    }
}

fn handle_of(file: &File) -> HANDLE {
    file.as_raw_handle() as HANDLE
}

/// The id, the link count and the attributes of an open file or folder. The id is the 128-bit file id where the file system has one (NTFS and ReFS do, from
/// Windows 8 on) and the 64-bit index otherwise.
pub(crate) fn info(file: &File) -> io::Result<Info> {
    let handle = handle_of(file);
    // SAFETY: an all-zero `BY_HANDLE_FILE_INFORMATION` is a valid value (integers and two `FILETIME`s of integers).
    let mut basic: BY_HANDLE_FILE_INFORMATION = unsafe { core::mem::zeroed() };
    // SAFETY: `handle` is open for as long as `file` is borrowed, which covers the call; `basic` is a valid, writable structure of the type the call expects.
    if unsafe { GetFileInformationByHandle(handle, &mut basic) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: an all-zero `FILE_ID_INFO` is a valid value (a 64-bit integer and sixteen bytes).
    let mut wide: FILE_ID_INFO = unsafe { core::mem::zeroed() };
    // SAFETY: as above; the buffer is `wide`, `size_of::<FILE_ID_INFO>()` bytes, which is what the `FileIdInfo` class writes.
    let have_wide = unsafe {
        GetFileInformationByHandleEx(handle, FileIdInfo, (&mut wide as *mut FILE_ID_INFO).cast::<c_void>(), size_of::<FILE_ID_INFO>() as u32)
    } != 0;
    let id = if have_wide {
        FileId { volume: wide.VolumeSerialNumber, index: u128::from_le_bytes(wide.FileId.Identifier) }
    } else {
        FileId { volume: u64::from(basic.dwVolumeSerialNumber), index: (u128::from(basic.nFileIndexHigh) << 32) | u128::from(basic.nFileIndexLow) }
    };
    Ok(Info { id, links: basic.nNumberOfLinks, attributes: basic.dwFileAttributes })
}

/// The real path of an open file or folder: the name the file system knows it by, from the root of its volume, with no junction and no symbolic link in it
/// (`\\?\C:\...`, or `\\?\Volume{...}\...` for a volume that has no drive letter). A folder that is held open cannot be renamed, so this stays true for as long as
/// the handle is open.
pub(crate) fn final_path(file: &File) -> io::Result<PathBuf> {
    let handle = handle_of(file);
    let mut last = io::Error::other("no path");
    for flags in [FILE_NAME_NORMALIZED | VOLUME_NAME_DOS, FILE_NAME_NORMALIZED | VOLUME_NAME_GUID] {
        let mut buffer = vec![0u16; 512];
        loop {
            // SAFETY: `handle` is open for the call; `buffer` is a valid, writable array of `buffer.len()` UTF-16 units, which is the size passed.
            let length = unsafe { GetFinalPathNameByHandleW(handle, buffer.as_mut_ptr(), buffer.len() as u32, flags) } as usize;
            if length == 0 {
                last = io::Error::last_os_error();
                break;
            }
            if length < buffer.len() {
                return Ok(PathBuf::from(OsString::from_wide(&buffer[..length])));
            }
            // too small: `length` is the size that is needed, terminator included
            buffer.resize(length + 1, 0);
        }
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::os::windows::fs::OpenOptionsExt;

    fn scratch(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("oaiy-keystore-winfs-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn folder(path: &std::path::Path) -> File {
        OpenOptions::new().read(true).share_mode(7).custom_flags(0x0200_0000 | 0x0020_0000).open(path).unwrap()
    }

    /// What makes two files one: **the volume and the index**, both. The file index of an NTFS volume is only unique on that volume: the same folder index on another volume (a USB
    /// drive, a second partition, a share) is another folder, and a path that has been re-pointed at it must not pass for the one that is held. (There is no second volume to open in a test
    /// without an administrator, so the identities are made up; the comparison in `KeyDir` is tested with them below.)
    #[test]
    fn two_files_are_one_only_when_the_volume_and_the_index_are_both_the_same() {
        let id = FileId::made_up(0xE200_0B7F, 42);
        assert_eq!(id, FileId::made_up(0xE200_0B7F, 42));
        assert_ne!(id, FileId::made_up(0xE200_0B80, 42), "the same index on another volume is another file");
        assert_ne!(id, FileId::made_up(0xE200_0B7F, 43), "another index on the same volume is another file");
        assert_ne!(id, FileId::made_up(0, 0));
        // the 128-bit index of ReFS: the high half counts
        assert_ne!(FileId::made_up(1, 1), FileId::made_up(1, 1 | (1u128 << 64)));
    }

    /// Two handles to one file have one id, however they were opened; two files have two; a hard link is another name of the same file and says so.
    #[test]
    fn the_id_is_the_file_and_the_link_count_is_its_names() {
        let dir = scratch("id");
        let (a, b) = (dir.join("a"), dir.join("b"));
        std::fs::write(&a, b"x").unwrap();
        std::fs::write(&b, b"x").unwrap();
        let (first, second) = (File::open(&a).unwrap(), File::open(&a).unwrap());
        let other = File::open(&b).unwrap();
        assert_eq!(info(&first).unwrap().id, info(&second).unwrap().id);
        assert_ne!(info(&first).unwrap().id, info(&other).unwrap().id);
        assert_eq!(info(&first).unwrap().links, 1);
        std::fs::hard_link(&a, dir.join("a2")).unwrap();
        let linked = File::open(dir.join("a2")).unwrap();
        assert_eq!(info(&first).unwrap().links, 2, "the count is read when it is asked, not when the handle was opened");
        assert_eq!(info(&linked).unwrap().id, info(&first).unwrap().id, "a hard link is the same file");
        // a folder is a folder: the directory attribute, one name, and its own id
        let held = folder(&dir);
        let dir_info = info(&held).unwrap();
        assert_ne!(dir_info.attributes & 0x10, 0);
        assert_eq!(dir_info.links, 1);
        assert_ne!(dir_info.id, info(&first).unwrap().id);
        drop((first, second, other, linked, held));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The final path is the real one: a junction on the way to the folder is resolved, so that the path and the folder are not two things that can come apart.
    #[test]
    fn the_final_path_has_no_junction_in_it() {
        let dir = scratch("final");
        std::fs::create_dir_all(dir.join("real").join("keys")).unwrap();
        let link = dir.join("link");
        let made = std::process::Command::new("cmd").args(["/C", "mklink", "/J"]).arg(&link).arg(dir.join("real")).output().unwrap();
        assert!(made.status.success(), "mklink /J failed");
        let through_the_junction = folder(&link.join("keys"));
        let direct = folder(&dir.join("real").join("keys"));
        let (via, real) = (final_path(&through_the_junction).unwrap(), final_path(&direct).unwrap());
        assert_eq!(via, real, "one folder, one real path");
        let shown = real.to_string_lossy().to_lowercase();
        assert!(shown.starts_with(r"\\?\") && shown.ends_with(r"\real\keys") && !shown.contains("link"), "{shown}");
        assert_eq!(info(&through_the_junction).unwrap().id, info(&direct).unwrap().id);
        drop((through_the_junction, direct));
        std::fs::remove_dir(&link).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A path longer than the first buffer (512 units) is returned whole.
    #[test]
    fn a_path_longer_than_the_first_buffer_is_returned_whole() {
        let dir = scratch("long");
        let mut deep = dir.clone();
        while deep.to_string_lossy().len() < 600 {
            deep = deep.join("a-folder-with-a-rather-long-name-so-that-the-path-gets-long-quickly");
        }
        let verbatim = PathBuf::from(format!(r"\\?\{}", deep.display()));
        std::fs::create_dir_all(&verbatim).unwrap();
        let held = folder(&verbatim);
        let path = final_path(&held).unwrap();
        assert!(path.to_string_lossy().len() > 512, "{}", path.display());
        assert!(path.to_string_lossy().to_lowercase().ends_with("so-that-the-path-gets-long-quickly"));
        drop(held);
        std::fs::remove_dir_all(PathBuf::from(format!(r"\\?\{}", dir.display()))).unwrap();
    }
}
