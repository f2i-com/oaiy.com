//! Positioned reads, with an optional page-cache bypass — the one place in
//! this crate with platform `cfg`s (CONVENTIONS.md: one place per concern).
//!
//! Direct I/O matters for the expert store: with 192 GB of RAM and a 289 GB
//! expert set, letting the OS page cache hold a second copy of whatever the
//! expert cache already holds wastes the RAM that decides the hit rate
//! (see the `Ecache` module docs). The price is alignment: offsets, lengths
//! and the buffer address must be sector multiples, and safetensors offsets
//! are not. So a direct read fetches the aligned superset into an aligned
//! scratch buffer and copies the wanted range out — at most 2 x 4 KiB of
//! extra transfer per read. All of it is safe Rust: the flag goes through
//! `OpenOptionsExt::custom_flags`, and the alignment comes from offsetting
//! into an over-allocated `Vec`.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

/// Alignment that satisfies direct I/O on every NVMe/SATA device we target.
pub const DIRECT_ALIGN: usize = 4096;

/// Fill `buf` from `file` at `offset`, looping over short reads.
pub fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let n = read_at(file, buf, offset)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

/// Open for reading, bypassing the page cache where the platform allows it
/// from safe Rust. Returns the file and whether the bypass is in effect.
pub fn open_read(path: &Path, direct: bool) -> io::Result<(File, bool)> {
    if direct {
        if let Some(file) = open_direct(path)? {
            return Ok((file, true));
        }
    }
    Ok((File::open(path)?, false))
}

#[cfg(windows)]
fn open_direct(path: &Path) -> io::Result<Option<File>> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING).open(path).map(Some)
}

/// O_DIRECT is not one value across Linux architectures (asm-generic uses
/// 0o40000, arm/aarch64 0o200000), so only architectures whose value is
/// spelled out here get the bypass; others fall back to buffered reads.
#[cfg(target_os = "linux")]
const LINUX_O_DIRECT: Option<i32> = if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
    Some(0o40000)
} else if cfg!(any(target_arch = "aarch64", target_arch = "arm")) {
    Some(0o200000)
} else {
    None
};

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> io::Result<Option<File>> {
    use std::os::unix::fs::OpenOptionsExt;
    match LINUX_O_DIRECT {
        Some(flag) => OpenOptions::new().read(true).custom_flags(flag).open(path).map(Some),
        None => Ok(None),
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn open_direct(_path: &Path) -> io::Result<Option<File>> {
    Ok(None) // macOS needs fcntl(F_NOCACHE), which safe std cannot reach
}

/// Reusable aligned scratch for direct reads.
#[derive(Default)]
pub struct AlignedScratch {
    raw: Vec<u8>,
}

impl AlignedScratch {
    /// An aligned window of at least `len` bytes (contents unspecified).
    fn window(&mut self, len: usize) -> &mut [u8] {
        if self.raw.len() < len + DIRECT_ALIGN {
            self.raw = vec![0u8; len + DIRECT_ALIGN];
        }
        let off = self.raw.as_ptr().align_offset(DIRECT_ALIGN);
        &mut self.raw[off..off + len]
    }
}

/// Read `dst.len()` bytes at `offset` from a file opened with the page-cache
/// bypass: aligned superset into `scratch`, then copy out the wanted range.
pub fn read_direct(file: &File, dst: &mut [u8], offset: u64, scratch: &mut AlignedScratch) -> io::Result<()> {
    let a = DIRECT_ALIGN as u64;
    let start = offset / a * a;
    let end = (offset + dst.len() as u64).div_ceil(a) * a;
    let span = (end - start) as usize;
    let head = (offset - start) as usize;
    // The aligned tail can run past EOF; a short final read is fine as long
    // as the bytes we need arrived. An unbuffered read only stops short at
    // EOF, and it then ends on an unaligned byte: reading on from there would
    // hand the OS a misaligned buffer and offset (Windows: ERROR_INVALID_PARAMETER),
    // so an unaligned total means we are done.
    let window = scratch.window(span);
    let mut got = 0;
    while got < span {
        let n = read_at(file, &mut window[got..], start + got as u64)?;
        got += n;
        if n == 0 || got % DIRECT_ALIGN != 0 {
            break;
        }
    }
    if got < head + dst.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short direct read"));
    }
    dst.copy_from_slice(&window[head..head + dst.len()]);
    Ok(())
}
