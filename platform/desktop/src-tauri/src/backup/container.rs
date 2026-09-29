//! The file: a ZIP encrypted as a whole with age, and the checks made on the way in.
//!
//! Writing streams: the ZIP is built in a private staged file, and the encrypted copy is written
//! from it in 64 KiB pieces, so memory stays small however large the backup is. Reading streams the
//! same way: the age payload is decrypted into a private scratch file (the ZIP format needs random
//! access, and an age file is read front to back), and the ZIP is then read entry by entry, each
//! entry's size and SHA-256 checked against the manifest as it goes.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use age::secrecy::SecretString;
use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use super::manifest::{Entry, Manifest};
use super::{hex, BackupError, ErrorKind, Limits, Result, MANIFEST_NAME};
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

fn secret(passphrase: &str) -> SecretString {
    SecretString::from(passphrase.to_string())
}

fn damaged() -> BackupError {
    BackupError::new(ErrorKind::Damaged, "This file is not an OAIY backup, or it is damaged or incomplete.")
}

// ---- names ----------------------------------------------------------------------------------------

const RESERVED: [&str; 22] = [
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6",
    "lpt7", "lpt8", "lpt9",
];

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
        let stem = segment.split('.').next().unwrap_or(segment).to_lowercase();
        if RESERVED.contains(&stem.as_str()) {
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
        Cost::Default => age::Encryptor::with_user_passphrase(secret(passphrase)),
        #[cfg(test)]
        Cost::Fixed(n) => {
            let mut recipient = age::scrypt::Recipient::new(secret(passphrase));
            recipient.set_work_factor(n);
            age::Encryptor::with_recipients(std::iter::once(&recipient as &dyn age::Recipient)).expect("one passphrase recipient")
        }
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
pub(crate) fn decrypt_to_file(enc: &Path, passphrase: &str, out: &Path, max_plain: u64) -> Result<(String, u64)> {
    let input = File::open(enc).map_err(|e| BackupError::io("Could not read the backup file", &e))?;
    let decryptor = age::Decryptor::new_buffered(BufReader::with_capacity(COPY_BUF, input)).map_err(decrypt_error)?;
    if !decryptor.is_scrypt() {
        return Err(BackupError::new(ErrorKind::Damaged, "This file is not protected by a passphrase, so it is not an OAIY backup."));
    }
    let mut identity = age::scrypt::Identity::new(secret(passphrase));
    // The age command line accepts up to this much work, and so does this: a file that asks for more
    // is a way to make this machine spend hours.
    identity.set_max_work_factor(22);
    let mut reader = decryptor.decrypt(std::iter::once(&identity as &dyn age::Identity)).map_err(decrypt_error)?;
    let mut file = BufWriter::with_capacity(COPY_BUF, secret_file::create_new_owner_only(out).map_err(|e| BackupError::io("Could not stage the backup", &e))?);
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut buf = vec![0u8; COPY_BUF];
    loop {
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
}

/// Read and check the decrypted ZIP at `plain`: the manifest is the first entry and is valid, the
/// entries are exactly the manifest's (no extra, none missing, none twice), and every entry has its
/// size and its SHA-256.
pub(crate) fn verify_zip(plain: &Path, limits: &Limits) -> Result<Verified> {
    let file = File::open(plain).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
    let mut archive = ZipArchive::new(BufReader::new(file)).map_err(|_| damaged())?;
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
    Ok(Verified { manifest, plain: plain.to_path_buf() })
}

/// Decrypt `enc`, and check it whole, using `scratch` (a private folder that exists) for the plaintext ZIP.
/// The scratch folder holds the plaintext until the caller removes it.
pub(crate) fn open_backup(enc: &Path, passphrase: &str, scratch: &Path, limits: &Limits) -> Result<Verified> {
    let plain = scratch.join("plain.zip");
    // A little more than the entries allow, for the ZIP's own headers and manifest.
    let max_plain = limits.max_total_bytes.saturating_add(limits.max_manifest_bytes).saturating_add(limits.max_entries as u64 * 1024);
    decrypt_to_file(enc, passphrase, &plain, max_plain)?;
    verify_zip(&plain, limits)
}

/// Unpack every entry of a verified backup, each into the path `dest_for` gives it, streaming and
/// checking each entry's hash again as it is written. The destination files are new and private.
pub(crate) fn extract_all(verified: &Verified, mut dest_for: impl FnMut(&Entry) -> Result<PathBuf>) -> Result<()> {
    let file = File::open(&verified.plain).map_err(|e| BackupError::io("Could not read the staged backup", &e))?;
    let mut archive = ZipArchive::new(BufReader::new(file)).map_err(|_| damaged())?;
    let mut buf = vec![0u8; COPY_BUF];
    for i in 1..archive.len() {
        let entry = archive.by_index(i).map_err(|_| damaged())?;
        let name = entry.name().to_string();
        let Some(wanted) = verified.manifest.entries.iter().find(|e| e.name == name) else {
            return Err(damaged());
        };
        let dest = dest_for(wanted)?;
        if let Some(parent) = dest.parent() {
            secret_file::create_private_dir(parent).map_err(|e| BackupError::io("Could not make a folder to restore into", &e))?;
        }
        let out = secret_file::create_new_owner_only(&dest).map_err(|e| BackupError::io("Could not stage a restored item", &e))?;
        let mut out = BufWriter::with_capacity(COPY_BUF, out);
        let mut reader = entry.take(wanted.size + 1);
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        loop {
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
