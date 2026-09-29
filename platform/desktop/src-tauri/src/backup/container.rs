//! The file: a ZIP encrypted as a whole with age, and the checks made on the way in.
//!
//! Writing streams: the ZIP is built in a private staged file, and the encrypted copy is written
//! from it in 64 KiB pieces, so memory stays small however large the backup is. Reading streams the
//! same way: the age payload is decrypted into a private scratch file (the ZIP format needs random
//! access, and an age file is read front to back), and the ZIP is then read entry by entry, each
//! entry's size and SHA-256 checked against the manifest as it goes.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use age::secrecy::SecretString;
use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use super::manifest::{Entry, Manifest};
use super::{hex, BackupError, Budget, ErrorKind, Limits, Result, MANIFEST_NAME};
use crate::secret_file;

const COPY_BUF: usize = 64 * 1024;

/// How hard the passphrase is made to work (the age default takes about a second).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Cost {
    /// age's own default, chosen for about a second on this computer.
    Default,
    /// A fixed work factor `N = 2^n`: for tests, which would otherwise wait a second for each file.
    #[cfg(test)]
    Fixed(u8),
}

/// The largest scrypt work factor a backup is opened at, `N = 2^20`. scrypt holds `128 * r * N` bytes
/// (r is 8 in age): 1 GiB at 20, 2 GiB at 21, 4 GiB at 22 (where age's own command line stops). 1 GiB
/// is what a computer that OAIY runs on can spare for a moment, and it is the most a backup is
/// written with (below): a file that asks for more is refused before any scrypt runs.
pub(crate) const MAX_READ_WORK_FACTOR: u8 = 20;
/// The least a backup is written with, `N = 2^18` (256 MiB, about a quarter of a second): age
/// picks its work factor by timing this computer, and a busy computer would time a weak one.
pub(crate) const MIN_WRITE_WORK_FACTOR: u8 = 18;
/// The most a backup is written with (see [`MAX_READ_WORK_FACTOR`]).
pub(crate) const MAX_WRITE_WORK_FACTOR: u8 = 20;
/// The most an age header may be: a passphrase header is under 200 bytes.
pub(crate) const MAX_HEADER_BYTES: usize = 2048;

/// The work factor for a computer that runs scrypt at `log_n = 14` in `seconds_at_14`: the one that
/// takes about a second, never below [`MIN_WRITE_WORK_FACTOR`] or above [`MAX_WRITE_WORK_FACTOR`].
/// (scrypt's cost doubles with each step of `log_n`.)
pub(crate) fn pick_work_factor(seconds_at_14: f64) -> u8 {
    let steps = (1.0 / seconds_at_14.max(1e-6)).log2().floor();
    (14.0 + steps).clamp(MIN_WRITE_WORK_FACTOR as f64, MAX_WRITE_WORK_FACTOR as f64) as u8
}

/// Time scrypt here, and pick the work factor for about a second.
pub(crate) fn default_work_factor() -> u8 {
    let started = std::time::Instant::now();
    let mut out = [0u8; 32];
    if let Ok(params) = scrypt::Params::new(14, 8, 1, 32) {
        let _ = scrypt::scrypt(b"oaiy backup timing", b"oaiy backup salt", &params, &mut out);
    }
    pick_work_factor(started.elapsed().as_secs_f64())
}

fn secret(passphrase: &str) -> SecretString {
    SecretString::from(passphrase.to_string())
}

fn damaged() -> BackupError {
    BackupError::new(ErrorKind::Damaged, "This file is not an OAIY backup, or it is damaged or incomplete.")
}

// ---- names ----------------------------------------------------------------------------------------

/// Whether a part of a name (before its first dot) is one Windows keeps for a device: `CON`, `PRN`, `AUX`, `NUL`, the console
/// handles `CONIN$` and `CONOUT$`, and `COM` or `LPT` followed by one digit, `0` to `9`, or by a superscript digit
/// (`COM¹`, `LPT²`, `COM³`), which Windows reads as the same device. A digit of another script or width is refused too.
/// Trailing spaces are ignored, as Windows ignores them.
pub(crate) fn is_reserved_device_name(stem: &str) -> bool {
    let stem = stem.trim_end_matches(' ').to_lowercase();
    if matches!(stem.as_str(), "con" | "prn" | "aux" | "nul" | "conin$" | "conout$") {
        return true;
    }
    stem.strip_prefix("com").or_else(|| stem.strip_prefix("lpt")).is_some_and(|digit| {
        let mut chars = digit.chars();
        matches!((chars.next(), chars.next()), (Some(c), None) if c.is_numeric())
    })
}

/// Whether `name` is a name that stays inside the folder it is put in, on every platform: a
/// relative path of plain segments with forward slashes. Refuses `..` and `.`, a leading slash or
/// backslash, a drive letter (any colon), a NUL, any control character, an empty segment, a
/// segment Windows would trim or reserve (`CON`, a trailing dot or space), and a name that is too long.
pub(crate) fn check_entry_name(name: &str, limits: &Limits) -> Result<()> {
    let unsafe_name = |why: &str| BackupError::new(ErrorKind::Unsafe, format!("This backup is refused: it holds an item with an unsafe name ({why})."));
    if name.is_empty() || name.len() > limits.max_name_len {
        return Err(unsafe_name("empty or too long"));
    }
    if name.contains('\0') {
        return Err(unsafe_name("it has a NUL"));
    }
    if name.contains('\\') || name.starts_with('/') {
        return Err(unsafe_name("it is not a plain relative path"));
    }
    if name.contains(':') {
        return Err(unsafe_name("it names a drive"));
    }
    if name.chars().any(|c| c.is_control() || matches!(c, '<' | '>' | '"' | '|' | '?' | '*')) {
        return Err(unsafe_name("it has a character files cannot have"));
    }
    for segment in name.split('/') {
        if segment.is_empty() {
            return Err(unsafe_name("an empty part or a trailing slash"));
        }
        if segment == "." || segment == ".." {
            return Err(unsafe_name("it leaves its folder"));
        }
        if segment.ends_with('.') || segment.ends_with(' ') {
            return Err(unsafe_name("a part ends with a dot or a space"));
        }
        // NTFS answers to a short name (PAIRIN~1.JSO) for a file with a long one, and a short name
        // passes every check made on the long one (a word rule, an extension rule).
        if segment.as_bytes().windows(2).any(|w| w[0] == b'~' && w[1].is_ascii_digit()) {
            return Err(unsafe_name("it is a short-name alias"));
        }
        if is_reserved_device_name(segment.split('.').next().unwrap_or(segment)) {
            return Err(unsafe_name("a part is a reserved device name"));
        }
    }
    Ok(())
}

/// `root` joined with a checked relative `name`.
pub(crate) fn safe_join(root: &Path, name: &str, limits: &Limits) -> Result<PathBuf> {
    check_entry_name(name, limits)?;
    let mut path = root.to_path_buf();
    for segment in name.split('/') {
        path.push(segment);
    }
    Ok(path)
}

// ---- age ------------------------------------------------------------------------------------------

/// Encrypt the file `plain` to `out` (a new, private file) with `passphrase`, streaming. Returns the
/// SHA-256 and length of the plaintext. The caller removes `out` on failure.
pub(crate) fn encrypt_file(plain: &Path, out: &Path, passphrase: &str, cost: Cost) -> Result<(String, u64)> {
    let mut input = BufReader::with_capacity(COPY_BUF, File::open(plain).map_err(|e| BackupError::io("Could not read the staged backup", &e))?);
    let file = secret_file::create_new_owner_only(out).map_err(|e| BackupError::io("Could not create the backup file", &e))?;
    let encryptor = match cost {
        Cost::Default => encryptor_at(passphrase, default_work_factor()),
        #[cfg(test)]
        Cost::Fixed(n) => encryptor_at(passphrase, n),
    };
    let write_err = |e: io::Error| BackupError::io("Could not write the backup file", &e);
    let mut writer = encryptor.wrap_output(BufWriter::with_capacity(COPY_BUF, file)).map_err(write_err)?;
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut buf = vec![0u8; COPY_BUF];
    loop {
        let n = input.read(&mut buf).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
        writer.write_all(&buf[..n]).map_err(write_err)?;
    }
    let buffered = writer.finish().map_err(write_err)?;
    let file = buffered.into_inner().map_err(|e| write_err(e.into_error()))?;
    file.sync_all().map_err(write_err)?;
    Ok((hex(&hasher.finalize()), total))
}

fn encryptor_at(passphrase: &str, log_n: u8) -> age::Encryptor {
    let mut recipient = age::scrypt::Recipient::new(secret(passphrase));
    recipient.set_work_factor(log_n);
    age::Encryptor::with_recipients(std::iter::once(&recipient as &dyn age::Recipient)).expect("one passphrase recipient")
}

/// Read the age header text and refuse a file whose header is not a short one, BEFORE age sees it.
/// age reads a header line by line and parses everything read again after each line, so a header of
/// a megabyte takes minutes; a real one is under 200 bytes. The bytes read (the header and whatever
/// follows it, up to the cap) are handed back so the caller can put them in front of the rest of the file.
fn read_header_prefix(file: &mut File) -> Result<Vec<u8>> {
    const VERSION_LINE: &[u8] = b"age-encryption.org/v1\n";
    let mut prefix = Vec::with_capacity(MAX_HEADER_BYTES + 1);
    Read::by_ref(file).take(MAX_HEADER_BYTES as u64 + 1).read_to_end(&mut prefix).map_err(|e| BackupError::io("Could not read the backup file", &e))?;
    if !prefix.starts_with(VERSION_LINE) {
        return Err(damaged());
    }
    // The header ends with the line "--- <mac>".
    let mut at = VERSION_LINE.len();
    while let Some(len) = prefix[at..].iter().position(|b| *b == b'\n') {
        if prefix[at..at + len].starts_with(b"--- ") {
            return Ok(prefix);
        }
        at += len + 1;
    }
    Err(BackupError::new(ErrorKind::Damaged, "This file is not an OAIY backup: its header is far longer than a backup's, or never ends."))
}

fn decrypt_error(e: age::DecryptError) -> BackupError {
    use age::DecryptError as D;
    match e {
        D::DecryptionFailed | D::KeyDecryptionFailed | D::NoMatchingKeys => {
            BackupError::new(ErrorKind::WrongPassphrase, "That passphrase did not open the backup, or the file is not an OAIY backup.")
        }
        D::ExcessiveWork { .. } => BackupError::new(ErrorKind::Unsupported, "This backup asks for more computing work to open than OAIY will spend."),
        _ => damaged(),
    }
}

/// Decrypt the age file `enc` with `passphrase` into `out` (a new, private file), streaming, refusing
/// more than `max_plain` bytes. Returns the SHA-256 and length of the plaintext. The caller removes
/// `out` on failure.
pub(crate) fn decrypt_to_file(enc: &Path, passphrase: &str, out: &Path, max_plain: u64, budget: &Budget) -> Result<(String, u64)> {
    let mut input = File::open(enc).map_err(|e| BackupError::io("Could not read the backup file", &e))?;
    let prefix = read_header_prefix(&mut input)?;
    let decryptor = age::Decryptor::new_buffered(BufReader::with_capacity(COPY_BUF, io::Cursor::new(prefix).chain(input))).map_err(decrypt_error)?;
    if !decryptor.is_scrypt() {
        return Err(BackupError::new(ErrorKind::Damaged, "This file is not protected by a passphrase, so it is not an OAIY backup."));
    }
    let mut identity = age::scrypt::Identity::new(secret(passphrase));
    // A file that asks for more work than this is refused before any of it is done: it would take the
    // memory of a small computer, and time.
    identity.set_max_work_factor(MAX_READ_WORK_FACTOR);
    let mut reader = decryptor.decrypt(std::iter::once(&identity as &dyn age::Identity)).map_err(decrypt_error)?;
    let mut file = BufWriter::with_capacity(COPY_BUF, secret_file::create_new_owner_only(out).map_err(|e| BackupError::io("Could not stage the backup", &e))?);
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut buf = vec![0u8; COPY_BUF];
    loop {
        budget.check()?;
        let n = match reader.read(&mut buf) {
            Ok(n) => n,
            // The payload fails its own checks: a changed byte, or a file cut short.
            Err(_) => return Err(damaged()),
        };
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > max_plain {
            return Err(BackupError::new(ErrorKind::TooLarge, "This backup is larger than OAIY will restore."));
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n]).map_err(|e| BackupError::io("Could not stage the backup", &e))?;
    }
    let file = file.into_inner().map_err(|e| BackupError::io("Could not stage the backup", &e.into_error()))?;
    file.sync_all().map_err(|e| BackupError::io("Could not stage the backup", &e))?;
    Ok((hex(&hasher.finalize()), total))
}

// ---- the ZIP --------------------------------------------------------------------------------------

/// Items that are compressed already: storing them is faster and no larger.
fn already_compressed(name: &str) -> bool {
    let lower = name.to_lowercase();
    [".zip", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".ogg", ".mp4", ".glb"].iter().any(|e| lower.ends_with(e))
}

fn zip_err(e: zip::result::ZipError) -> BackupError {
    match e {
        zip::result::ZipError::Io(io) => BackupError::io("Could not write the backup", &io),
        _ => damaged(),
    }
}

/// Write the ZIP: `manifest.json` first, then each of `files` (name inside the ZIP, and where its bytes are).
pub(crate) fn write_zip(path: &Path, manifest_json: &[u8], files: &[(String, PathBuf)]) -> Result<()> {
    let out = secret_file::create_new_owner_only(path).map_err(|e| BackupError::io("Could not stage the backup", &e))?;
    let mut zip = ZipWriter::new(BufWriter::with_capacity(COPY_BUF, out));
    // A level can only be given to a method that has levels.
    let deflated = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated).compression_level(Some(6));
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(MANIFEST_NAME, deflated).map_err(zip_err)?;
    zip.write_all(manifest_json).map_err(|e| BackupError::io("Could not stage the backup", &e))?;
    let mut buf = vec![0u8; COPY_BUF];
    for (name, source) in files {
        let mut input = BufReader::with_capacity(COPY_BUF, File::open(source).map_err(|e| BackupError::io("Could not read a staged item", &e))?);
        let size = input.get_ref().metadata().map(|m| m.len()).unwrap_or(0);
        let options = if already_compressed(name) { stored } else { deflated };
        zip.start_file(name, options.large_file(size >= (u32::MAX as u64 - 1))).map_err(zip_err)?;
        loop {
            let n = input.read(&mut buf).map_err(|e| BackupError::io("Could not read a staged item", &e))?;
            if n == 0 {
                break;
            }
            zip.write_all(&buf[..n]).map_err(|e| BackupError::io("Could not stage the backup", &e))?;
        }
    }
    let buffered = zip.finish().map_err(zip_err)?;
    let file = buffered.into_inner().map_err(|e| BackupError::io("Could not stage the backup", &e.into_error()))?;
    file.sync_all().map_err(|e| BackupError::io("Could not stage the backup", &e))?;
    Ok(())
}

/// A ZIP that has been read whole: its manifest and where its bytes are.
pub(crate) struct Verified {
    pub manifest: Manifest,
    pub plain: PathBuf,
    /// The SHA-256 of the decrypted ZIP, whole: what a restore is bound to. It covers the record and every
    /// item (the Agent's storage archive included), so two files with this hash are the same backup.
    pub plain_sha256: String,
}

/// Read and check the decrypted ZIP at `plain`: the manifest is the first entry and is valid, the
/// entries are exactly the manifest's (no extra, none missing, none twice), and every entry has its
/// size and its SHA-256.
pub(crate) fn verify_zip(plain: &Path, limits: &Limits, budget: &Budget) -> Result<Verified> {
    // How many entries it says it has is read from its end record BEFORE its list of them is parsed: parsing
    // is what costs the memory and the time, and a file can list millions.
    let directory = peek_zip_directory(plain)?;
    if directory.entries > limits.max_entries as u64 || directory.bytes > (limits.max_entries as u64).saturating_mul(ENTRY_RECORD_MAX) {
        return Err(BackupError::new(ErrorKind::TooLarge, "This backup holds more items than OAIY will restore."));
    }
    let file = File::open(plain).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
    let mut archive = ZipArchive::new(BufReader::new(file)).map_err(|_| damaged())?;
    // The reader keeps one entry of a name that is listed twice, so the count its end record gives is the true one.
    if archive.len() as u64 != directory.entries {
        return Err(twice());
    }
    if archive.is_empty() || archive.len() > limits.max_entries {
        return Err(if archive.is_empty() { damaged() } else { BackupError::new(ErrorKind::TooLarge, "This backup holds more items than OAIY will restore.") });
    }
    let manifest: Manifest = {
        let entry = archive.by_index(0).map_err(|_| damaged())?;
        if entry.name() != MANIFEST_NAME || entry.is_dir() {
            return Err(BackupError::new(ErrorKind::Damaged, "This backup does not start with its record."));
        }
        if entry.size() > limits.max_manifest_bytes {
            return Err(BackupError::new(ErrorKind::TooLarge, "This backup's record is larger than OAIY will read."));
        }
        let mut bytes = Vec::new();
        entry.take(limits.max_manifest_bytes + 1).read_to_end(&mut bytes).map_err(|_| damaged())?;
        if bytes.len() as u64 > limits.max_manifest_bytes {
            return Err(BackupError::new(ErrorKind::TooLarge, "This backup's record is larger than OAIY will read."));
        }
        serde_json::from_slice(&bytes).map_err(|_| BackupError::new(ErrorKind::Damaged, "This backup's record is damaged."))?
    };
    manifest.validate(limits)?;

    let mut expected: HashMap<&str, &Entry> = manifest.entries.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut buf = vec![0u8; COPY_BUF];
    for i in 1..archive.len() {
        budget.check()?;
        let entry = archive.by_index(i).map_err(|_| damaged())?;
        if entry.is_dir() {
            return Err(BackupError::new(ErrorKind::Unsafe, "This backup is refused: it holds a folder entry."));
        }
        let name = entry.name().to_string();
        let Some(wanted) = expected.remove(name.as_str()) else {
            return Err(BackupError::new(ErrorKind::Unsafe, "This backup is refused: it holds an item its record does not list, or one twice."));
        };
        if entry.size() != wanted.size {
            return Err(BackupError::new(ErrorKind::Damaged, "An item in this backup does not match its record."));
        }
        let mut reader = entry.take(wanted.size + 1);
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        loop {
            budget.check()?;
            let n = reader.read(&mut buf).map_err(|_| damaged())?;
            if n == 0 {
                break;
            }
            total += n as u64;
            hasher.update(&buf[..n]);
        }
        if total != wanted.size || hex(&hasher.finalize()) != wanted.sha256 {
            return Err(BackupError::new(ErrorKind::Damaged, "An item in this backup does not match its checksum."));
        }
    }
    if !expected.is_empty() {
        return Err(BackupError::new(ErrorKind::Damaged, "This backup is missing items its record lists."));
    }
    Ok(Verified { manifest, plain: plain.to_path_buf(), plain_sha256: String::new() })
}

/// Decrypt `enc`, and check it whole, using `scratch` (a private folder that exists) for the plaintext ZIP.
/// The scratch folder holds the plaintext until the caller removes it.
pub(crate) fn open_backup(enc: &Path, passphrase: &str, scratch: &Path, limits: &Limits, budget: &Budget) -> Result<Verified> {
    let plain = scratch.join("plain.zip");
    // A little more than the entries allow, for the ZIP's own headers and manifest.
    let max_plain = limits.max_total_bytes.saturating_add(limits.max_manifest_bytes).saturating_add(limits.max_entries as u64 * 1024);
    let (plain_sha256, _) = decrypt_to_file(enc, passphrase, &plain, max_plain, budget)?;
    Ok(Verified { plain_sha256, ..verify_zip(&plain, limits, budget)? })
}

/// Unpack every entry of a verified backup, each into the path `dest_for` gives it, streaming and
/// checking each entry's hash again as it is written. The destination files are new and private.
pub(crate) fn extract_all(verified: &Verified, mut dest_for: impl FnMut(&Entry) -> Result<Option<PathBuf>>, budget: &Budget) -> Result<()> {
    let file = File::open(&verified.plain).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
    let mut archive = ZipArchive::new(BufReader::new(file)).map_err(|_| damaged())?;
    let mut buf = vec![0u8; COPY_BUF];
    // Found by name in one step each: a scan of the list for every entry is a quadratic amount of work.
    let listed: HashMap<&str, &Entry> = verified.manifest.entries.iter().map(|e| (e.name.as_str(), e)).collect();
    for i in 1..archive.len() {
        budget.check()?;
        let entry = archive.by_index(i).map_err(|_| damaged())?;
        let name = entry.name().to_string();
        let Some(wanted) = listed.get(name.as_str()).copied() else {
            return Err(damaged());
        };
        // An entry that is not wanted (a class the person did not tick) is not read.
        let Some(dest) = dest_for(wanted)? else { continue };
        if let Some(parent) = dest.parent() {
            secret_file::create_private_dir(parent).map_err(|e| BackupError::io("Could not make a folder to restore into", &e))?;
        }
        let out = secret_file::create_new_owner_only(&dest).map_err(|e| BackupError::io("Could not stage a restored item", &e))?;
        let mut out = BufWriter::with_capacity(COPY_BUF, out);
        let mut reader = entry.take(wanted.size + 1);
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        loop {
            budget.check()?;
            let n = reader.read(&mut buf).map_err(|_| damaged())?;
            if n == 0 {
                break;
            }
            total += n as u64;
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n]).map_err(|e| BackupError::io("Could not stage a restored item", &e))?;
        }
        let out = out.into_inner().map_err(|e| BackupError::io("Could not stage a restored item", &e.into_error()))?;
        out.sync_all().map_err(|e| BackupError::io("Could not stage a restored item", &e))?;
        if total != wanted.size || hex(&hasher.finalize()) != wanted.sha256 {
            return Err(BackupError::new(ErrorKind::Damaged, "An item in this backup does not match its checksum."));
        }
    }
    Ok(())
}

/// The most a record of one entry in a ZIP's directory takes: its fixed part, a name of the longest
/// length and room for its extra fields and comment.
pub(crate) const ENTRY_RECORD_MAX: u64 = 1024;

/// What a ZIP says of its own directory, read from its end record.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ZipDirectory {
    pub entries: u64,
    /// The size of the list of entries.
    pub bytes: u64,
}

/// Read how many entries a ZIP says it holds, and how large its list of them is, from its end record (and its
/// ZIP64 end record when it has one) without parsing the list. Parsing keeps a record in memory for each entry,
/// so this comes before it.
pub(crate) fn peek_zip_directory(path: &Path) -> Result<ZipDirectory> {
    let mut file = File::open(path).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
    let len = file.metadata().map_err(|e| BackupError::io("Could not read the staged backup", &e))?.len();
    if len < 22 {
        return Err(damaged());
    }
    // The end record is the last thing in the file, followed by a comment of at most 65,535 bytes.
    let tail_len = len.min(22 + 65_535 + 20);
    file.seek(SeekFrom::Start(len - tail_len)).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
    let mut tail = vec![0u8; tail_len as usize];
    file.read_exact(&mut tail).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
    let at = (0..=tail.len() - 22).rev().find(|&i| tail[i..i + 4] == [0x50, 0x4b, 0x05, 0x06] && i + 22 + u16::from_le_bytes([tail[i + 20], tail[i + 21]]) as usize <= tail.len()).ok_or_else(damaged)?;
    let u16_at = |i: usize| u16::from_le_bytes([tail[i], tail[i + 1]]) as u64;
    let u32_at = |i: usize| u32::from_le_bytes([tail[i], tail[i + 1], tail[i + 2], tail[i + 3]]) as u64;
    let (mut entries, mut bytes, offset) = (u16_at(at + 10), u32_at(at + 12), u32_at(at + 16));
    if entries == 0xFFFF || bytes == 0xFFFF_FFFF || offset == 0xFFFF_FFFF {
        // ZIP64: a locator just before the end record says where the larger record is.
        if at < 20 || tail[at - 20..at - 16] != [0x50, 0x4b, 0x06, 0x07] {
            return Err(damaged());
        }
        let at64 = u64::from_le_bytes(tail[at - 12..at - 4].try_into().expect("eight bytes"));
        if at64.checked_add(56).is_none_or(|end| end > len) {
            return Err(damaged());
        }
        file.seek(SeekFrom::Start(at64)).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
        let mut record = [0u8; 56];
        file.read_exact(&mut record).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
        if record[..4] != [0x50, 0x4b, 0x06, 0x06] {
            return Err(damaged());
        }
        entries = u64::from_le_bytes(record[32..40].try_into().expect("eight bytes"));
        bytes = u64::from_le_bytes(record[40..48].try_into().expect("eight bytes"));
    }
    Ok(ZipDirectory { entries, bytes })
}

/// A ZIP opened for reading a few of its entries (a verified backup's plaintext).
pub(crate) type Archive = ZipArchive<BufReader<File>>;

pub(crate) fn open_archive(plain: &Path) -> Result<Archive> {
    let directory = peek_zip_directory(plain)?;
    let file = File::open(plain).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
    let archive = ZipArchive::new(BufReader::new(file)).map_err(|_| damaged())?;
    // A name listed twice is refused: the reader keeps one of them, and other programs may read the other.
    if archive.len() as u64 != directory.entries {
        return Err(twice());
    }
    Ok(archive)
}

fn twice() -> BackupError {
    BackupError::new(ErrorKind::Unsafe, "This backup is refused: it lists the same item twice, or its list of items is damaged.")
}

/// One entry's bytes, when it is there and no larger than `max` (else `None`).
pub(crate) fn read_entry(archive: &mut Archive, name: &str, max: u64) -> Result<Option<Vec<u8>>> {
    let entry = match archive.by_name(name) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };
    if entry.size() > max {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    entry.take(max + 1).read_to_end(&mut bytes).map_err(|_| damaged())?;
    Ok((bytes.len() as u64 <= max).then_some(bytes))
}
