//! K1 (design 4.5.1, work package V-02) through its public API, for every provider this platform has: the keyfile everywhere, and DPAPI on Windows.
//!
//! Every test makes its own scratch folder under the system temp folder with a unique name and removes it when it ends (the folder is removed
//! by a guard, so a failed assertion cleans up too). Nothing here touches the real `<data>/keys`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use oaiy_keystore::{names, open_at, KeyError, KeyStore, Name, ProviderChoice, Strength, ENV_PROVIDER, MAX_VALUE_LEN};

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("oaiy-keystore-test-{tag}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        // the folder above the keys folder is judged too: one that others could rename in is refused, and the umask of some systems makes it 0775
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
        }
        Scratch(dir)
    }

    fn keys(&self) -> PathBuf {
        self.0.join("keys")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap()).collect()
}

/// The keyfile provider, by the name this platform allows: on Windows `keyfile` is refused, and only a test may name the unsafe one.
fn keyfile() -> ProviderChoice {
    if cfg!(windows) {
        ProviderChoice::KeyfileUnsafe
    } else {
        ProviderChoice::Keyfile
    }
}

/// The providers this platform has, with the extension their files carry.
fn providers() -> Vec<(ProviderChoice, &'static str)> {
    let mut list = vec![(keyfile(), "kf")];
    if cfg!(windows) {
        list.push((ProviderChoice::DpapiFile, "ks"));
    }
    list
}

fn store(scratch: &Scratch, choice: ProviderChoice) -> Box<dyn KeyStore> {
    open_at(scratch.keys(), choice).unwrap()
}

fn name(text: &str) -> Name {
    Name::new(text).unwrap()
}

/// A run of bytes that appears nowhere else: every value in these tests is one of them, so a scan for it finds a leak and only a leak.
fn canary(seed: u8, len: usize) -> Vec<u8> {
    let mut state = u64::from(seed).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

fn get(store: &dyn KeyStore, n: &Name) -> Option<Vec<u8>> {
    store.get(n).unwrap().map(|v| v.to_vec())
}

fn all_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(all_files(&path));
        } else {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// The bookkeeping files of the store: they hold no secret and no name, and are not key files.
const BOOKKEEPING: [&str; 2] = [".lock", ".provider"];

/// The files under `dir` that are key files or their debris (everything but the bookkeeping files).
fn key_files(dir: &Path) -> Vec<PathBuf> {
    all_files(dir).into_iter().filter(|p| !BOOKKEEPING.contains(&p.file_name().unwrap().to_str().unwrap())).collect()
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    needle.len() <= haystack.len() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// Files under `dir` (every one, dot files and temporaries included) whose bytes contain `needle`.
fn files_containing(dir: &Path, needle: &[u8]) -> Vec<String> {
    all_files(dir)
        .into_iter()
        .filter(|p| contains(&fs::read(p).unwrap(), needle))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect()
}

fn file_of(scratch: &Scratch, n: &str, ext: &str) -> PathBuf {
    scratch.keys().join(format!("{n}.{ext}"))
}

#[test]
fn round_trip_overwrite_delete_and_list_for_every_provider() {
    for (choice, ext) in providers() {
        let scratch = Scratch::new("roundtrip");
        let store = store(&scratch, choice);
        let info = store.provider();
        match choice {
            ProviderChoice::Keyfile | ProviderChoice::KeyfileUnsafe => assert_eq!((info.id, info.strength), ("keyfile", Strength::FilePermissions)),
            _ => assert_eq!((info.id, info.strength), ("windows-dpapi-file", Strength::OsAccount)),
        }
        let a = name(names::ARCHIVE_WRITER);
        store.put(&a, &canary(1, 32)).unwrap();
        assert_eq!(get(&*store, &a), Some(canary(1, 32)));
        store.put(&a, &canary(2, 48)).unwrap();
        assert_eq!(get(&*store, &a), Some(canary(2, 48)), "{ext}: overwritten");

        for (n, seed) in [("vault.fk.f2", 4u8), ("vault.fk.f1", 5), ("vault.pins", 6), ("backup.sig", 7)] {
            store.put(&name(n), &canary(seed, 32)).unwrap();
        }
        let listed = |prefix: &str| -> Vec<String> { store.list(prefix).unwrap().into_iter().map(|n| n.to_string()).collect() };
        assert_eq!(listed(""), ["archive.writer", "backup.sig", "vault.fk.f1", "vault.fk.f2", "vault.pins"]);
        assert_eq!(listed(names::VAULT_FK_PREFIX), ["vault.fk.f1", "vault.fk.f2"]);
        assert_eq!(listed("vault."), ["vault.fk.f1", "vault.fk.f2", "vault.pins"]);
        assert!(listed("relay.").is_empty());
        assert_eq!(listed("archive.writer"), ["archive.writer"]);

        store.delete(&a).unwrap();
        assert_eq!(get(&*store, &a), None);
        store.delete(&a).unwrap();
        assert_eq!(listed(""), ["backup.sig", "vault.fk.f1", "vault.fk.f2", "vault.pins"]);

        // every size from one byte to the cap, with zero bytes and 0xff in them
        for len in [1usize, 2, 15, 16, 17, 63, 64, 65, 255, 256, 4095, 4096, 65535, MAX_VALUE_LEN] {
            let mut value = canary(9, len);
            value[0] = 0;
            *value.last_mut().unwrap() = 0xff;
            let n = name("size.check");
            store.put(&n, &value).unwrap();
            assert_eq!(get(&*store, &n), Some(value), "{ext}: {len} bytes");
        }
    }
}

#[test]
fn a_name_that_was_never_stored_is_none_not_an_error() {
    for (choice, _) in providers() {
        let scratch = Scratch::new("none");
        let store = store(&scratch, choice);
        assert_eq!(get(&*store, &name("never.stored")), None);
        assert!(store.list("").unwrap().is_empty());
        store.delete(&name("never.stored")).unwrap();
        store.put(&name("other"), b"x").unwrap();
        assert_eq!(get(&*store, &name("never.stored")), None, "another name exists, this one still does not");
        store.delete(&name("other")).unwrap();
        assert_eq!(get(&*store, &name("other")), None, "deleted is the same as never stored");
    }
}

#[test]
fn a_store_that_cannot_read_its_key_is_an_error_and_never_none() {
    for (choice, ext) in providers() {
        let scratch = Scratch::new("errors");
        let store = store(&scratch, choice);
        let n = name("archive.writer");
        let value = canary(3, 40);
        store.put(&n, &value).unwrap();
        let path = file_of(&scratch, "archive.writer", ext);
        let blob = fs::read(&path).unwrap();

        // not a blob at all, and an empty file
        for junk in [&b"not a blob"[..], b"", &[0u8; 200][..], &blob[..8]] {
            fs::write(&path, junk).unwrap();
            let error = store.get(&n).expect_err(&format!("{ext}: a file of {} bytes of junk", junk.len()));
            assert!(matches!(error, KeyError::Corrupt(_) | KeyError::Os(..)), "{error:?}");
        }
        // every truncation of a real blob
        for len in 0..blob.len() {
            fs::write(&path, &blob[..len]).unwrap();
            assert!(store.get(&n).is_err(), "{ext}: truncated to {len} of {} bytes read as a secret or as nothing", blob.len());
        }
        // every single bit of a real blob flipped: an error, or (only where a provider ignores a field) the very same value; never None, never another value
        let mut tolerated = 0;
        for bit in 0..blob.len() * 8 {
            let mut bad = blob.clone();
            bad[bit / 8] ^= 1 << (bit % 8);
            fs::write(&path, &bad).unwrap();
            match store.get(&n) {
                Err(_) => {}
                Ok(Some(v)) => {
                    assert_eq!(&*v, value.as_slice(), "{ext}: bit {bit} changed the value that was read");
                    tolerated += 1;
                }
                Ok(None) => panic!("{ext}: a damaged file (bit {bit}) was read as 'never stored'"),
            }
        }
        if ext == "kf" {
            assert_eq!(tolerated, 0, "the keyfile check covers every byte");
        }
        // an extended blob
        let mut longer = blob.clone();
        longer.push(0);
        fs::write(&path, &longer).unwrap();
        match store.get(&n) {
            Err(_) => {}
            Ok(Some(v)) => assert_eq!(&*v, value.as_slice()),
            Ok(None) => panic!("{ext}: an extended file was read as 'never stored'"),
        }
        // a directory where the file should be, and a put over it
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(matches!(store.get(&n), Err(KeyError::Io { .. } | KeyError::Permissions(_))), "{ext}: a directory in place of a key file");
        assert!(store.put(&n, b"replacement").is_err());
        assert!(key_files(&scratch.keys()).is_empty(), "{ext}: the failed put left a file behind");
        fs::remove_dir(&path).unwrap();
        // the store is fine afterwards
        store.put(&n, &value).unwrap();
        assert_eq!(get(&*store, &n), Some(value));
    }
}

#[test]
fn a_keys_directory_that_has_gone_is_an_error_not_an_empty_store() {
    for (choice, _) in providers() {
        let scratch = Scratch::new("nodir");
        let store = store(&scratch, choice);
        let n = name("vault.pins");
        store.put(&n, b"pins").unwrap();
        if cfg!(windows) {
            // the store holds its folder open without FILE_SHARE_DELETE: nobody can remove or rename it, or the one above it, while the store is open
            assert!(fs::rename(scratch.keys(), scratch.0.join("moved")).is_err(), "a folder held open cannot be renamed");
            assert!(fs::rename(&scratch.0, scratch.0.with_extension("moved")).is_err(), "nor can the folder above it");
            for entry in fs::read_dir(scratch.keys()).unwrap() {
                fs::remove_file(entry.unwrap().path()).unwrap(); // the files can go: the folder cannot
            }
            assert!(fs::remove_dir(scratch.keys()).is_err(), "an empty folder held open cannot be removed");
            assert_eq!(get(&*store, &n), None, "and the store is still usable, and its folder is still its folder");
            store.put(&n, b"pins").unwrap();
            assert_eq!(get(&*store, &n), Some(b"pins".to_vec()));
            drop(store);
            fs::remove_dir_all(scratch.keys()).unwrap();
        } else {
            fs::remove_dir_all(scratch.keys()).unwrap();
            assert!(matches!(store.get(&n), Err(KeyError::Io { .. })), "get");
            assert!(matches!(store.put(&n, b"pins"), Err(KeyError::Io { .. })), "put");
            assert!(matches!(store.list(""), Err(KeyError::Io { .. })), "list");
            assert!(matches!(store.delete(&n), Err(KeyError::Io { .. })), "delete");
        }
        // and a store opened again finds the folder empty: that is a new store, and the caller learns it from `None` only now
        let reopened = open_at(scratch.keys(), choice).unwrap();
        assert_eq!(get(&*reopened, &n), None);
    }
}

#[cfg(windows)]
#[test]
fn a_locked_file_is_an_error_and_a_locked_destination_makes_put_fail_and_keep_the_old_value() {
    use std::fs::OpenOptions;
    use std::os::windows::fs::OpenOptionsExt;
    for (choice, ext) in providers() {
        let scratch = Scratch::new("locked");
        let store = store(&scratch, choice);
        let n = name("relay.token");
        store.put(&n, b"before").unwrap();
        let path = file_of(&scratch, "relay.token", ext);
        {
            // an exclusive handle (share mode 0), as an antivirus scan or another process might hold
            let _lock = OpenOptions::new().read(true).share_mode(0).open(&path).unwrap();
            assert!(matches!(store.get(&n), Err(KeyError::Io { .. })), "{ext}: a sharing violation is an error, not 'never stored'");
            assert!(store.put(&n, b"after").is_err(), "{ext}: the rename over a locked file fails");
            assert!(store.delete(&n).is_err());
        }
        assert_eq!(get(&*store, &n), Some(b"before".to_vec()), "{ext}: the previous value survived");
        assert_eq!(key_files(&scratch.keys()).len(), 1, "{ext}: no temporary file of the failed put");
    }
}

#[test]
fn a_blob_copied_or_moved_to_another_name_fails_and_never_yields_the_other_secret() {
    for (choice, ext) in providers() {
        let scratch = Scratch::new("moved");
        let store = store(&scratch, choice);
        let (a, b) = (name("archive.writer"), name("backup.sig"));
        store.put(&a, &canary(1, 32)).unwrap();
        store.put(&b, &canary(2, 32)).unwrap();

        // a copy under a new name
        fs::copy(file_of(&scratch, "archive.writer", ext), file_of(&scratch, "vault.pins", ext)).unwrap();
        let moved = store.get(&name("vault.pins")).expect_err("a blob under another name");
        match choice {
            ProviderChoice::Keyfile | ProviderChoice::KeyfileUnsafe => assert!(matches!(moved, KeyError::WrongName), "{moved:?}"),
            _ => assert!(matches!(moved, KeyError::Os(..)), "{moved:?}"),
        }
        // one secret's file over another's
        fs::copy(file_of(&scratch, "archive.writer", ext), file_of(&scratch, "backup.sig", ext)).unwrap();
        assert!(store.get(&b).is_err(), "{ext}: the file of another name was read as this name");
        // and the two swapped
        fs::write(file_of(&scratch, "archive.writer", ext), fs::read(file_of(&scratch, "vault.pins", ext)).unwrap()).unwrap();
        assert_eq!(get(&*store, &a), Some(canary(1, 32)), "{ext}: a file under its own name still reads");
        // a rename, the way a person might tidy up
        fs::rename(file_of(&scratch, "vault.pins", ext), file_of(&scratch, "relay.host_identity", ext)).unwrap();
        assert!(store.get(&name("relay.host_identity")).is_err());
    }
}

#[test]
fn no_plaintext_beside_the_blob() {
    for (choice, ext) in providers() {
        let scratch = Scratch::new("plaintext");
        let store = store(&scratch, choice);
        let n = name("archive.writer");
        let (first, second) = (canary(21, 96), canary(22, 96));
        store.put(&n, &first).unwrap();
        let holders = files_containing(&scratch.keys(), &first);
        match choice {
            // the keyfile is a plaintext provider: the value is in its own file and in no other
            ProviderChoice::Keyfile | ProviderChoice::KeyfileUnsafe => assert_eq!(holders, ["archive.writer.kf"]),
            _ => assert!(holders.is_empty(), "{ext}: DPAPI files hold the value in the clear: {holders:?}"),
        }
        assert_eq!(key_files(&scratch.keys()).len(), 1, "{ext}: exactly one file per secret, no temporary");

        // an overwrite leaves no trace of the old value anywhere
        store.put(&n, &second).unwrap();
        assert!(files_containing(&scratch.keys(), &first).is_empty(), "{ext}: the old value is still in a file");
        assert_eq!(key_files(&scratch.keys()).len(), 1);

        // a failed put leaves nothing either (a directory sits where the file would go, so the rename fails after the temporary file was written and verified)
        let blocker = name("blocked.name");
        fs::create_dir(file_of(&scratch, "blocked.name", ext)).unwrap();
        let third = canary(23, 96);
        assert!(store.put(&blocker, &third).is_err());
        assert!(files_containing(&scratch.keys(), &third).is_empty(), "{ext}: the failed put left the value in a file");
        assert!(files_containing(&scratch.keys(), &second).len() <= 1);
        fs::remove_dir(file_of(&scratch, "blocked.name", ext)).unwrap();

        // and a delete removes everything
        store.delete(&n).unwrap();
        assert!(key_files(&scratch.keys()).is_empty(), "{ext}: files remain after the only secret was deleted");
        assert!(files_containing(&scratch.keys(), &second).is_empty());
    }
}

/// KM11, KM12 at the file system: what the store writes is the pinned format, byte for byte (the expected bytes are computed with node's crypto, not with this
/// code), and a file in that format that someone else wrote is read. A later version must read the files of this one.
#[test]
fn a_keyfile_is_the_pinned_format_on_disk_and_a_pinned_file_is_read() {
    let first = "4f4149594b463101195a64d1c2e3a8d0c197bc2ddf1df260424530839be9fda34c33e23fdde2795c0000001a6f616979206b657973746f7265206b6e6f776e20616e73776572d57f32c601b74541a0f629ddf6506393553028cb8598860d71ade7a85424bc2e";
    let second = "4f4149594b46310171a3542662649453a76dc19689a899f8bbff6f57b7dd580805c66c809a7aa7770000000300ff01748ed4d0507837dfd14a1edb9e7296d0a9638617d76efe6359acb88933fb2a46";
    let scratch = Scratch::new("kat");
    let store = store(&scratch, keyfile());
    store.put(&name("archive.writer"), b"oaiy keystore known answer").unwrap();
    assert_eq!(fs::read(file_of(&scratch, "archive.writer", "kf")).unwrap(), unhex(first));
    // a file of the same format that this store did not write
    let path = file_of(&scratch, "vault.fk.f1", "kf");
    fs::write(&path, unhex(second)).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    }
    assert_eq!(get(&*store, &name("vault.fk.f1")), Some(vec![0x00, 0xff, 0x01]));
    assert_eq!(get(&*store, &name("archive.writer")), Some(b"oaiy keystore known answer".to_vec()));
}

#[test]
fn values_are_one_byte_to_sixty_four_kib() {
    for (choice, _) in providers() {
        let scratch = Scratch::new("bounds");
        let store = store(&scratch, choice);
        let n = name("bounds");
        let secret = canary(31, 70_000);
        assert!(matches!(store.put(&n, b""), Err(KeyError::InvalidValue(_))));
        assert!(matches!(store.put(&n, &secret[..MAX_VALUE_LEN + 1]), Err(KeyError::InvalidValue(_))));
        assert!(matches!(store.put(&n, &secret), Err(KeyError::InvalidValue(_))));
        assert_eq!(get(&*store, &n), None, "a refused put stored nothing");
        store.put(&n, &secret[..MAX_VALUE_LEN]).unwrap();
        assert_eq!(get(&*store, &n).unwrap().len(), MAX_VALUE_LEN);
        // an error names the problem and never carries the value
        let error = store.put(&n, &secret).unwrap_err();
        let text = format!("{error} / {error:?}");
        assert!(!contains(text.as_bytes(), &secret[..16]) && !text.contains("70000"), "{text}");
        assert_eq!(error.code(), "key_invalid_value");
    }
}

#[test]
fn a_file_that_is_far_larger_than_any_secret_is_refused_without_being_read_into_memory() {
    for (choice, ext) in providers() {
        let scratch = Scratch::new("huge");
        let store = store(&scratch, choice);
        let n = name("huge.file");
        let file = fs::File::create(file_of(&scratch, "huge.file", ext)).unwrap();
        file.set_len(300 * 1024 * 1024).unwrap(); // sparse: 300 MiB that costs nothing
        drop(file);
        #[cfg(unix)]
        {
            // the mode is checked before the size: make it a permissible file so that it is the size that is refused
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(file_of(&scratch, "huge.file", ext), fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert!(matches!(store.get(&n), Err(KeyError::Corrupt(_))), "{ext}");
    }
}

#[test]
fn the_error_codes_are_stable_and_no_error_prints_a_secret() {
    let all = [
        (KeyError::InvalidName, "key_invalid_name"),
        (KeyError::InvalidValue("empty"), "key_invalid_value"),
        (KeyError::Corrupt("x"), "key_corrupt"),
        (KeyError::WrongName, "key_wrong_name"),
        (KeyError::Os("CryptUnprotectData", 13), "key_os"),
        (KeyError::Permissions("mode".into()), "key_permissions"),
        (KeyError::Verify, "key_verify_failed"),
        (KeyError::ProviderUnavailable("x"), "key_provider_unavailable"),
        (KeyError::InvalidProvider("x".into()), "key_invalid_provider"),
        (KeyError::ProviderMismatch { stored: "windows-dpapi-file".into(), requested: "keyfile" }, "key_provider_mismatch"),
    ];
    for (error, code) in all {
        assert_eq!(error.code(), code);
        assert!(error.to_string().starts_with(code), "{error}");
    }
    assert!(Name::new("Bad Name").is_err());
}

#[test]
fn the_provider_is_chosen_on_purpose_and_a_typo_is_an_error() {
    assert_eq!(ProviderChoice::parse("auto").unwrap(), ProviderChoice::Auto);
    assert_eq!(ProviderChoice::parse("keyfile").unwrap(), ProviderChoice::Keyfile);
    assert_eq!(ProviderChoice::parse("windows-dpapi-file").unwrap(), ProviderChoice::DpapiFile);
    assert_eq!(ProviderChoice::parse("os-keyring").unwrap(), ProviderChoice::OsKeyring);
    for bad in ["", "KEYFILE", "Keyfile", "key-file", "dpapi", "plain", "none", "auto ", " keyfile", "file", "keyring"] {
        assert!(matches!(ProviderChoice::parse(bad), Err(KeyError::InvalidProvider(_))), "{bad:?}");
    }
    // the variable: unset is Auto, a value is parsed, a bad one is an error and never Auto (this is the only test that touches the variable)
    std::env::remove_var(ENV_PROVIDER);
    assert_eq!(ProviderChoice::from_env().unwrap(), ProviderChoice::Auto);
    std::env::set_var(ENV_PROVIDER, "keyfile");
    assert_eq!(ProviderChoice::from_env().unwrap(), ProviderChoice::Keyfile);
    std::env::set_var(ENV_PROVIDER, "keyfil");
    assert!(matches!(ProviderChoice::from_env(), Err(KeyError::InvalidProvider(_))));
    std::env::remove_var(ENV_PROVIDER);

    // Auto is DPAPI on Windows and nothing anywhere else: no silent fallback to a file
    if cfg!(windows) {
        assert_eq!(ProviderChoice::Auto.resolve().unwrap(), ProviderChoice::DpapiFile);
        let scratch = Scratch::new("auto");
        let store = open_at(scratch.keys(), ProviderChoice::Auto).unwrap();
        assert_eq!(store.provider().id, "windows-dpapi-file");
    } else {
        assert!(matches!(ProviderChoice::Auto.resolve(), Err(KeyError::ProviderUnavailable(_))));
        let scratch = Scratch::new("auto");
        assert!(matches!(open_at(scratch.keys(), ProviderChoice::Auto), Err(KeyError::ProviderUnavailable(_))));
        assert!(!scratch.keys().exists(), "a refusal creates nothing");
        assert!(matches!(open_at(scratch.keys(), ProviderChoice::DpapiFile), Err(KeyError::ProviderUnavailable(_))));
    }
    // the Secret Service is named, is not built, and does not fall back
    let scratch = Scratch::new("keyring");
    assert!(matches!(open_at(scratch.keys(), ProviderChoice::OsKeyring), Err(KeyError::ProviderUnavailable(_))));
    assert!(!scratch.keys().exists());
}

#[test]
fn list_shows_only_names_and_only_files_of_its_own_provider() {
    for (choice, ext) in providers() {
        let scratch = Scratch::new("list");
        let store = store(&scratch, choice);
        store.put(&name("b.two"), b"2").unwrap();
        store.put(&name("a.one"), b"1").unwrap();
        let dir = scratch.keys();
        // things that are not names, not files, not of this provider, or temporaries
        for junk in [
            format!("Bad Name.{ext}"),
            format!("UPPER.{ext}"),
            format!(".hidden.{ext}"),
            "notes.txt".into(),
            format!("con.{ext}"),
            format!("x.{ext}.bak"),
            ".a.0123456789abcdef.tmp".into(),
        ] {
            fs::write(dir.join(junk), b"x").unwrap();
        }
        fs::create_dir(dir.join(format!("adir.{ext}"))).unwrap();
        let listed: Vec<String> = store.list("").unwrap().into_iter().map(|n| n.to_string()).collect();
        assert_eq!(listed, ["a.one", "b.two"], "{ext}");
    }
}

/// M-2, the test that used to assert the bad behaviour: a file of the other provider's kind in the folder was not listed, and `get` of its name said `None`,
/// "never stored", so a caller minted a new identity over a secret that exists. It is an error now, in every operation, and nothing changes.
#[test]
fn a_secret_of_another_providers_kind_is_an_error_everywhere_never_none_and_never_a_second_value() {
    for (choice, ext) in providers() {
        let scratch = Scratch::new("otherkind");
        let store = store(&scratch, choice);
        let n = name("other.provider");
        store.put(&name("a.one"), b"1").unwrap();
        let other = if ext == "kf" { "ks" } else { "kf" };
        let foreign = scratch.keys().join(format!("other.provider.{other}"));
        fs::write(&foreign, b"a secret that another provider made").unwrap();
        let mismatch = |error: KeyError| matches!(&error, KeyError::ProviderMismatch { requested, .. } if *requested == store_id(ext));
        assert!(mismatch(store.get(&n).unwrap_err()), "{ext}: get");
        assert!(mismatch(store.put(&n, b"a second value").unwrap_err()), "{ext}: put");
        assert!(mismatch(store.delete(&n).unwrap_err()), "{ext}: delete");
        assert!(mismatch(store.list("").unwrap_err()), "{ext}: list");
        assert_eq!(fs::read(&foreign).unwrap(), b"a secret that another provider made", "{ext}: nothing was changed");
        assert_eq!(key_files(&scratch.keys()).len(), 2, "{ext}: and nothing was added: no second value, no temporary file");
        // the names that have no such file are unaffected
        assert_eq!(get(&*store, &name("a.one")), Some(b"1".to_vec()));
        store.put(&name("b.two"), b"2").unwrap();
        // and once the stray file is gone the store is whole again
        fs::remove_file(&foreign).unwrap();
        assert_eq!(get(&*store, &n), None);
        assert_eq!(store.list("").unwrap().len(), 2);
    }
}

fn store_id(ext: &str) -> &'static str {
    if ext == "kf" {
        "keyfile"
    } else {
        "windows-dpapi-file"
    }
}

/// M-2: the folder remembers its provider. Opened again with another one it is refused before anything is read (the reviewer's DPAPI folder, reopened with
/// `OAIY_KEY_PROVIDER=keyfile`, read every secret as `None`, listed nothing, and after a `put` held two live values under one name, one in the clear).
#[test]
fn a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another() {
    for (choice, ext) in providers() {
        let scratch = Scratch::new("marker");
        let id = store_id(ext);
        {
            let store = store(&scratch, choice);
            store.put(&name("archive.writer"), &canary(70, 32)).unwrap();
        }
        let marker = scratch.keys().join(".provider");
        assert_eq!(fs::read(&marker).unwrap(), format!("{id}\n").into_bytes(), "{ext}: the marker names the provider and nothing else");
        // the same provider opens it again, and the marker is not rewritten
        let before = fs::metadata(&marker).unwrap().modified().unwrap();
        assert_eq!(get(&*store(&scratch, choice), &name("archive.writer")), Some(canary(70, 32)));
        assert_eq!(fs::metadata(&marker).unwrap().modified().unwrap(), before);

        // a marker that names another provider: refused, whatever the files are
        for other in ["windows-dpapi-file", "keyfile", "os-keyring", "something-else"] {
            if other == id {
                continue;
            }
            fs::write(&marker, format!("{other}\n")).unwrap();
            match open_at(scratch.keys(), choice) {
                Err(KeyError::ProviderMismatch { stored, requested }) => assert_eq!((stored.as_str(), requested), (other, id)),
                other => panic!("{ext}: a folder of another provider was opened: {:?}", other.err()),
            }
        }
        // a marker that is not one of ours is an error, not a folder to adopt
        for junk in [&b""[..], b"keyfile", b"keyfile\r\n", b"KEYFILE\n", b"key file\n", &[0xff, 0xfe, b'\n'][..], &[b'a'; 65][..]] {
            fs::write(&marker, junk).unwrap();
            assert!(matches!(open_at(scratch.keys(), choice), Err(KeyError::Corrupt(_))), "{ext}: marker {junk:?}");
        }
        // a marker that is gone, in a folder that holds only this provider's files (one made before markers): adopted, and the marker is written
        fs::remove_file(&marker).unwrap();
        assert_eq!(get(&*store(&scratch, choice), &name("archive.writer")), Some(canary(70, 32)));
        assert_eq!(fs::read(&marker).unwrap(), format!("{id}\n").into_bytes());
        // a folder with no marker that holds another provider's files is refused, and gets no marker
        fs::remove_file(&marker).unwrap();
        let other_ext = if ext == "kf" { "ks" } else { "kf" };
        fs::write(scratch.keys().join(format!("strange.secret.{other_ext}")), b"x").unwrap();
        match open_at(scratch.keys(), choice) {
            Err(KeyError::ProviderMismatch { stored, requested }) => assert_eq!((stored.as_str(), requested), (store_id(other_ext), id)),
            other => panic!("{ext}: a folder that holds another provider's files was adopted: {:?}", other.err()),
        }
        assert!(!marker.exists(), "{ext}: a refused folder is not claimed");
    }
}

/// On Windows the keyfile has no modes and this crate sets no ACL: it is refused unless a test names it as unsafe, and a DPAPI folder cannot be downgraded to it
/// by a setting.
#[test]
fn the_keyfile_is_refused_on_windows_and_the_unsafe_name_is_needed_to_use_it_there() {
    assert_eq!(ProviderChoice::parse("keyfile-unsafe-for-tests").unwrap(), ProviderChoice::KeyfileUnsafe);
    let scratch = Scratch::new("unsafe");
    if cfg!(windows) {
        let error = open_at(scratch.keys(), ProviderChoice::Keyfile).err().expect("the keyfile is refused on Windows");
        assert!(matches!(&error, KeyError::ProviderUnavailable(why) if why.contains("keyfile-unsafe-for-tests")), "{error:?}");
        assert!(!scratch.keys().exists(), "a refusal creates nothing");
        // the downgrade the reviewer found: a DPAPI folder, reopened with the keyfile named on purpose, is refused by the folder itself
        let dpapi = open_at(scratch.keys(), ProviderChoice::DpapiFile).unwrap();
        dpapi.put(&name("archive.writer"), &canary(71, 32)).unwrap();
        drop(dpapi);
        assert!(matches!(open_at(scratch.keys(), ProviderChoice::KeyfileUnsafe), Err(KeyError::ProviderMismatch { .. })));
        assert_eq!(get(&*open_at(scratch.keys(), ProviderChoice::DpapiFile).unwrap(), &name("archive.writer")), Some(canary(71, 32)));
    } else {
        assert!(open_at(scratch.keys(), ProviderChoice::Keyfile).is_ok());
        let scratch = Scratch::new("unsafe2");
        assert!(open_at(scratch.keys(), ProviderChoice::KeyfileUnsafe).is_ok(), "on Unix the unsafe name is the same provider");
    }
}

#[test]
fn many_threads_reading_and_writing_never_see_a_torn_value() {
    for (choice, _) in providers() {
        let scratch = Scratch::new("threads");
        let store: Arc<dyn KeyStore> = Arc::from(store(&scratch, choice));
        let shared = name("shared.value");
        let candidates: Vec<Vec<u8>> = (0..4u8).map(|i| canary(40 + i, 200 + usize::from(i) * 50)).collect();
        store.put(&shared, &candidates[0]).unwrap();
        let mut handles = Vec::new();
        for t in 0..8u8 {
            let store = Arc::clone(&store);
            let candidates = candidates.clone();
            let shared = shared.clone();
            handles.push(thread::spawn(move || {
                let own = name(&format!("thread.{t}"));
                for i in 0..20usize {
                    let value = canary(100 + t, 64 + i);
                    store.put(&own, &value).unwrap();
                    assert_eq!(store.get(&own).unwrap().unwrap().to_vec(), value);
                    if t % 2 == 0 {
                        store.put(&shared, &candidates[(i + usize::from(t)) % 4]).unwrap();
                    }
                    let seen = store.get(&shared).unwrap().expect("the shared name is never absent");
                    assert!(candidates.iter().any(|c| c.as_slice() == *seen), "a torn or foreign value was read");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(store.list("thread.").unwrap().len(), 8);
        assert!(key_files(&scratch.keys()).len() == 9, "no temporary file survives the run");
    }
}

const CHILD_DIR: &str = "OAIY_KS_CHILD_DIR";
const CHILD_PROVIDER: &str = "OAIY_KS_CHILD_PROVIDER";

fn shared_candidates() -> Vec<Vec<u8>> {
    (0..4u8).map(|i| canary(60 + i, 100 + usize::from(i) * 37)).collect()
}

/// The writer of the next test, run by the test binary itself as a second process. It does nothing when the harness runs it on its own.
#[test]
fn child_of_the_two_process_test_rewrites_one_name_over_and_over() {
    let (Some(dir), Some(label)) = (std::env::var_os(CHILD_DIR), std::env::var(CHILD_PROVIDER).ok()) else { return };
    let choice = if label == "dpapi" { ProviderChoice::DpapiFile } else { keyfile() };
    let store = open_at(dir, choice).unwrap();
    let candidates = shared_candidates();
    for i in 0..150usize {
        store.put(&name("shared.value"), &candidates[i % 4]).unwrap();
    }
}

/// H-1 with real processes. Two rewrite one name in a loop (each put is a rename over the existing file); three threads of this process read it in a loop. A
/// reader must see a whole value every time: never `None` (on Windows a reader that opens a name in the moment of a `MoveFileEx` replace finds it missing, and
/// the reviewer's reader was told `None` for a key that existed, 174 times in one run), never an error, never a mix of two values.
#[test]
fn readers_never_see_a_name_that_other_processes_are_rewriting_as_missing_or_torn() {
    for (choice, _) in providers() {
        let label = if choice == ProviderChoice::DpapiFile { "dpapi" } else { "keyfile" };
        let scratch = Scratch::new("twoproc");
        let store: Arc<dyn KeyStore> = Arc::from(store(&scratch, choice));
        let shared = name("shared.value");
        let candidates = shared_candidates();
        store.put(&shared, &candidates[0]).unwrap();
        let writers: Vec<std::process::Child> = (0..2)
            .map(|_| {
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "child_of_the_two_process_test_rewrites_one_name_over_and_over", "--test-threads=1"])
                    .env(CHILD_DIR, scratch.keys())
                    .env(CHILD_PROVIDER, label)
                    .stdout(std::process::Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..3)
            .map(|_| {
                let (store, stop, shared, candidates) = (Arc::clone(&store), Arc::clone(&stop), shared.clone(), candidates.clone());
                thread::spawn(move || {
                    let (mut reads, mut wrong) = (0u64, Vec::new());
                    while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                        match store.get(&shared) {
                            Ok(Some(value)) if candidates.iter().any(|c| c.as_slice() == *value) => {}
                            Ok(Some(_)) => wrong.push("a torn or foreign value".to_string()),
                            Ok(None) => wrong.push("a key that exists was reported as never stored".to_string()),
                            Err(error) => wrong.push(format!("a read failed while other processes were replacing the file: {error}")),
                        }
                        reads += 1;
                    }
                    (reads, wrong)
                })
            })
            .collect();
        for writer in writers {
            let output = writer.wait_with_output().unwrap();
            assert!(output.status.success(), "{label}: a writer process failed: {}\n{}", output.status, String::from_utf8_lossy(&output.stdout));
        }
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        let (mut reads, mut wrong) = (0, Vec::new());
        for reader in readers {
            let (r, w) = reader.join().unwrap();
            reads += r;
            wrong.extend(w);
        }
        assert!(wrong.is_empty(), "{label}: {} of {reads} reads were wrong, for example: {}", wrong.len(), wrong[0]);
        assert!(reads >= 100, "{label}: the readers only got {reads} reads in while the writers ran");
        assert_eq!(key_files(&scratch.keys()).len(), 1, "{label}: no temporary file is left");
    }
}
/// L-6 with threads: one opens the store over and over (as other processes do), the other puts. No put may fail because of an open: the store used to delete
/// the temporary file of a put in progress, and the put failed at its read-back (870 of 874 puts on Linux, about 1,200 of 1,265 on Windows).
#[test]
fn puts_do_not_fail_while_the_store_is_being_opened_again_and_again() {
    for (choice, _) in providers() {
        let scratch = Scratch::new("openloop");
        let store = store(&scratch, choice);
        let n = name("relay.token");
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let opener = {
            let (keys, stop) = (scratch.keys(), Arc::clone(&stop));
            thread::spawn(move || {
                let mut opens = 0u32;
                while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                    drop(open_at(&keys, choice).expect("a second open of the same folder"));
                    opens += 1;
                }
                opens
            })
        };
        for i in 0..60u8 {
            let value = canary(i, 40 + usize::from(i));
            store.put(&n, &value).unwrap_or_else(|e| panic!("put {i} failed while the store was being opened: {e}"));
            assert_eq!(get(&*store, &n), Some(value));
        }
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(opener.join().unwrap() > 5, "the other thread opened the store a few times at least");
        assert_eq!(key_files(&scratch.keys()).len(), 1, "no temporary file is left");
    }
}
/// DPAPI specifics: what is on disk, and what "another user's key" does.
#[cfg(windows)]
mod dpapi {
    use super::*;

    #[test]
    fn the_file_is_a_dpapi_blob_with_the_magic_and_no_plaintext() {
        let scratch = Scratch::new("dpapi-file");
        let store = store(&scratch, ProviderChoice::DpapiFile);
        let value = canary(50, 64);
        store.put(&name("vault.wrapper"), &value).unwrap();
        let bytes = fs::read(file_of(&scratch, "vault.wrapper", "ks")).unwrap();
        assert_eq!(&bytes[..8], b"OAIYKS1\x01");
        assert!(!contains(&bytes, &value));
        // DPAPI blobs start with version 1 and the DPAPI provider GUID df9d8cd0-1501-11d1-8c7a-00c04fc297eb
        assert_eq!(&bytes[8..12], &[1, 0, 0, 0]);
        assert_eq!(&bytes[12..28], &[0xd0, 0x8c, 0x9d, 0xdf, 0x01, 0x15, 0xd1, 0x11, 0x8c, 0x7a, 0x00, 0xc0, 0x4f, 0xc2, 0x97, 0xeb]);
        // KM19: the scope. The blob header after the master key's GUID is dwFlags: zero for a blob protected for the current user, 0x4 (CRYPTPROTECT_LOCAL_MACHINE)
        // for one that any process of the machine could open. A real blob of this provider is user scope; a provider that asked for the machine scope would be a
        // silent weakening that nothing else here would notice, because the same machine opens it.
        assert_eq!(&bytes[48..52], &[0, 0, 0, 0], "the DPAPI blob is not user-scope: flags {:02x?}", &bytes[48..52]);
        // the same value stored twice gives different blobs (DPAPI salts every one)
        store.put(&name("vault.wrapper"), &value).unwrap();
        assert_ne!(fs::read(file_of(&scratch, "vault.wrapper", "ks")).unwrap(), bytes);
    }

    /// The master key that protected a blob is named inside it (its GUID). A blob made under another user's master key names a key this user does not have: the
    /// same failure as another user's blob, produced without a second account by pointing a real blob at a master key that does not exist.
    #[test]
    fn a_blob_that_names_a_master_key_this_user_does_not_hold_is_an_error() {
        let scratch = Scratch::new("dpapi-guid");
        let store = store(&scratch, ProviderChoice::DpapiFile);
        let n = name("archive.writer");
        let value = canary(51, 32);
        store.put(&n, &value).unwrap();
        let path = file_of(&scratch, "archive.writer", "ks");
        let good = fs::read(&path).unwrap();
        // offset: 8 (our magic) + 4 (version) + 16 (provider GUID) + 4 (master key version) = the master key GUID
        for byte in 0..16 {
            let mut foreign = good.clone();
            foreign[8 + 24 + byte] ^= 0xa5;
            fs::write(&path, &foreign).unwrap();
            match store.get(&n) {
                Err(KeyError::Os(_, _)) => {}
                other => panic!("master key GUID byte {byte}: {other:?}"),
            }
        }
        fs::write(&path, &good).unwrap();
        assert_eq!(get(&*store, &n), Some(value));
    }

    /// The real other-user test. It needs a blob that another Windows account made with this crate, so it is `#[ignore]`d and never runs by itself:
    ///
    /// 1. as a *different* Windows user, run `cargo run -p oaiy-keystore --example write_blob -- <folder> foreign.secret` (the example stores a fixed value under that name with the DPAPI provider);
    /// 2. as the normal user, `set OAIY_KS_FOREIGN_BLOB=<folder>\keys\foreign.secret.ks` and `cargo test -p oaiy-keystore -- --ignored another_users_blob`.
    ///
    /// The test copies the blob into a scratch folder under a unique name, expects `get` to fail (an error, not `None` and not a value), and removes the copy.
    #[test]
    #[ignore = "needs a blob written by another Windows user: see the doc comment"]
    fn another_users_blob_fails() {
        let Some(source) = std::env::var_os("OAIY_KS_FOREIGN_BLOB") else {
            panic!("set OAIY_KS_FOREIGN_BLOB to the path of foreign.secret.ks written by another Windows user (see the test's documentation)");
        };
        let scratch = Scratch::new("dpapi-foreign");
        let store = store(&scratch, ProviderChoice::DpapiFile);
        fs::copy(&source, file_of(&scratch, "foreign.secret", "ks")).unwrap();
        match store.get(&name("foreign.secret")) {
            Err(KeyError::Os(_, _)) => {}
            other => panic!("another user's blob: {other:?}"),
        }
        assert_eq!(get(&*store, &name("never.stored")), None);
    }
}

/// The keyfile's permission rules on a real Unix file system. (Compiled everywhere Unix is; run wherever the tests run on Unix.)
#[cfg(unix)]
mod unix {
    use super::*;
    use std::os::unix::fs::{symlink, PermissionsExt};

    fn mode_of(path: &Path) -> u32 {
        fs::metadata(path).unwrap().permissions().mode() & 0o7777
    }

    #[test]
    fn a_new_store_is_0700_and_its_files_are_0600() {
        let scratch = Scratch::new("modes");
        let store = store(&scratch, ProviderChoice::Keyfile);
        assert_eq!(mode_of(&scratch.keys()), 0o700);
        store.put(&name("vault.pins"), b"pins").unwrap();
        assert_eq!(mode_of(&file_of(&scratch, "vault.pins", "kf")), 0o600);
        assert_eq!(get(&*store, &name("vault.pins")), Some(b"pins".to_vec()));
    }

    #[test]
    fn a_directory_or_file_with_any_group_or_other_permission_is_refused_and_never_repaired() {
        let scratch = Scratch::new("loose");
        let store = store(&scratch, ProviderChoice::Keyfile);
        store.put(&name("vault.pins"), b"pins").unwrap();
        let file = file_of(&scratch, "vault.pins", "kf");
        for loose in [0o644, 0o640, 0o604, 0o660, 0o666, 0o700, 0o4600] {
            fs::set_permissions(&file, fs::Permissions::from_mode(loose)).unwrap();
            assert!(matches!(store.get(&name("vault.pins")), Err(KeyError::Permissions(_))), "file {loose:04o}");
            assert_eq!(mode_of(&file), loose, "the provider does not repair a mode");
        }
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(store.get(&name("vault.pins")).unwrap().is_some());
        for loose in [0o755, 0o750, 0o705, 0o777, 0o2700] {
            fs::set_permissions(scratch.keys(), fs::Permissions::from_mode(loose)).unwrap();
            assert!(matches!(store.get(&name("vault.pins")), Err(KeyError::Permissions(_))), "directory {loose:04o}");
            assert!(matches!(store.put(&name("vault.pins"), b"x"), Err(KeyError::Permissions(_))));
            assert!(matches!(store.list(""), Err(KeyError::Permissions(_))));
            assert!(matches!(open_at(scratch.keys(), ProviderChoice::Keyfile), Err(KeyError::Permissions(_))), "opening a loose directory");
        }
        fs::set_permissions(scratch.keys(), fs::Permissions::from_mode(0o700)).unwrap();
        assert!(store.get(&name("vault.pins")).unwrap().is_some());
    }

    #[test]
    fn a_symbolic_link_in_place_of_the_file_or_the_directory_is_refused() {
        let scratch = Scratch::new("symlink");
        let store = store(&scratch, ProviderChoice::Keyfile);
        store.put(&name("real.secret"), b"real").unwrap();
        symlink(file_of(&scratch, "real.secret", "kf"), file_of(&scratch, "linked.secret", "kf")).unwrap();
        assert!(matches!(store.get(&name("linked.secret")), Err(KeyError::Permissions(_))));
        let elsewhere = Scratch::new("symlink-target");
        fs::create_dir(elsewhere.0.join("keys")).unwrap();
        fs::set_permissions(elsewhere.0.join("keys"), fs::Permissions::from_mode(0o700)).unwrap();
        let link_parent = Scratch::new("symlink-dir");
        symlink(elsewhere.0.join("keys"), link_parent.0.join("keys")).unwrap();
        assert!(matches!(open_at(link_parent.0.join("keys"), ProviderChoice::Keyfile), Err(KeyError::Permissions(_))));
    }

    /// KM25: every failure to open a key file other than "there is no such file" is an error. A store that read them all as "never stored" would let a caller
    /// mint a new identity over a key it merely could not open (a file that another account owns, a disk that is failing). A mode of 0000 is the simplest such
    /// failure; root is not stopped by modes, so the test stands aside for root (the locked-file test does the same on Windows).
    #[test]
    fn a_key_file_that_cannot_be_opened_is_an_error_and_never_none() {
        if rustix::process::geteuid().is_root() {
            return;
        }
        let scratch = Scratch::new("noaccess");
        let store = store(&scratch, ProviderChoice::Keyfile);
        store.put(&name("vault.pins"), b"pins").unwrap();
        let file = file_of(&scratch, "vault.pins", "kf");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o000)).unwrap();
        let error = store.get(&name("vault.pins")).expect_err("a file that cannot be opened is not `None`");
        assert!(
            matches!(&error, KeyError::Io { op: "open a key file", source } if source.kind() == std::io::ErrorKind::PermissionDenied),
            "{error:?}"
        );
        assert!(matches!(store.put(&name("vault.pins"), b"x"), Ok(())), "a put replaces the file by renaming, which needs no access to the old one");
        assert_eq!(get(&*store, &name("vault.pins")), Some(b"x".to_vec()));
    }

    /// Whether a result is the refusal of something that another user owns.
    fn is_another_user<T>(result: Result<T, KeyError>) -> bool {
        matches!(result, Err(KeyError::Permissions(why)) if why.contains("another user"))
    }

    fn require_root() {
        assert!(
            rustix::process::geteuid().is_root(),
            "this test needs root (it makes files that another user owns): build the test binary as yourself, then run it as root, for example \
             `wsl -d Ubuntu-24.04 -u root -- <the keystore test binary> --ignored`; see scripts in the keystore README"
        );
    }

    /// KM16 with a real second account (nobody, 65534). Needs root, so it is `#[ignore]`d; it is run by running the test binary as root. A key file, the keys
    /// folder, and the folder above it that another user owns are each refused, in the place where each is judged.
    #[test]
    #[ignore = "needs root: run the test binary as root with --ignored (see the README)"]
    fn a_file_a_folder_and_a_parent_that_a_real_other_user_owns_are_refused() {
        use std::os::unix::fs::chown;
        require_root();
        let scratch = Scratch::new("realowner");
        let store = store(&scratch, ProviderChoice::Keyfile);
        let n = name("a.one");
        store.put(&n, b"1").unwrap();
        let (nobody, root) = (Some(65534), Some(0));
        let file = file_of(&scratch, "a.one", "kf");
        chown(&file, nobody, nobody).unwrap();
        assert!(is_another_user(store.get(&n)), "a key file that another user owns");
        chown(&file, root, root).unwrap();
        assert!(store.get(&n).unwrap().is_some());
        chown(scratch.keys(), nobody, nobody).unwrap();
        assert!(is_another_user(store.get(&n)), "the keys folder, before each operation");
        assert!(
            matches!(open_at(scratch.keys(), ProviderChoice::Keyfile), Err(KeyError::Permissions(why)) if why.contains("another user")),
            "the keys folder, at open"
        );
        chown(scratch.keys(), root, root).unwrap();
        chown(&scratch.0, nobody, nobody).unwrap();
        let error = open_at(scratch.keys(), ProviderChoice::Keyfile).err().expect("a folder above that another user owns");
        assert!(
            matches!(&error, KeyError::Permissions(why) if why.contains("above the keys directory") && why.contains("another user")),
            "{error:?}"
        );
        chown(&scratch.0, root, root).unwrap();
        assert!(open_at(scratch.keys(), ProviderChoice::Keyfile).is_ok());
    }

    const ATTACK_DIR: &str = "OAIY_KS_ATTACK_DIR";
    const ATTACK_SECONDS: u64 = 5;
    /// A keyfile (the pinned format) for the name `archive.writer` holding `oaiy keystore known answer`: what the attacker would like the victim to read.
    const ATTACKER_BLOB: &str = "4f4149594b463101195a64d1c2e3a8d0c197bc2ddf1df260424530839be9fda34c33e23fdde2795c0000001a6f616979206b657973746f7265206b6e6f776e20616e73776572d57f32c601b74541a0f629ddf6506393553028cb8598860d71ade7a85424bc2e";

    /// The attacker of the next test: run by the test binary as another user. It swaps the victim's `keys` folder and its own `evilkeys` folder as fast as it can.
    #[test]
    fn child_of_the_uid_swap_stress_swaps_two_folders_as_fast_as_it_can() {
        let Some(shared) = std::env::var_os(ATTACK_DIR) else { return };
        let shared = PathBuf::from(shared);
        let (keys, evil, parked) = (shared.join("keys"), shared.join("evilkeys"), shared.join("parked"));
        fs::create_dir(&evil).unwrap();
        fs::write(evil.join("archive.writer.kf"), unhex(ATTACKER_BLOB)).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(ATTACK_SECONDS);
        let mut swaps = 0u64;
        while std::time::Instant::now() < deadline {
            if fs::rename(&keys, &parked).is_ok() {
                let _ = fs::rename(&evil, &keys);
                let _ = fs::rename(&parked, &evil);
                swaps += 1;
            }
        }
        println!("swaps={swaps}");
    }

    /// M-3, the reviewer's stress with a real second account, made a test (needs root, so `#[ignore]`d; see the README for the command). The victim is root; the
    /// attacker is uid 65534, running this test binary. The victim's store is opened where it should be (a sticky parent); the parent then becomes world-writable
    /// and not sticky, which is exactly the reviewer's configuration, and the attacker swaps `keys` and `evilkeys` for five seconds while the victim reads in a
    /// loop. The reviewer's store accepted 3,388 of the attacker's values and answered `None` 58,442 times in 40 seconds. This one must never return the
    /// attacker's value and never `None`: it reads its own value, or says that the folder is not where it was (an error). A new `open` in that parent is refused.
    #[test]
    #[ignore = "needs root: run the test binary as root with --ignored (see the README)"]
    fn uid_swap_stress_the_victim_never_reads_the_attackers_value_and_never_gets_none() {
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::process::CommandExt;
        require_root();
        let base = Scratch::new("uidswap");
        fs::set_permissions(&base.0, fs::Permissions::from_mode(0o755)).unwrap();
        let shared = base.0.join("shared");
        fs::create_dir(&shared).unwrap();
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o1777)).unwrap();
        let store = open_at(shared.join("keys"), ProviderChoice::Keyfile).unwrap();
        let n = name("archive.writer");
        let value = canary(80, 32);
        store.put(&n, &value).unwrap();
        // the reviewer's configuration: the parent that anyone can rename in
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o777)).unwrap();
        let refused =
            open_at(shared.join("keys2"), ProviderChoice::Keyfile).err().expect("a new open in a parent that anyone can rename in is refused");
        assert!(matches!(&refused, KeyError::Permissions(why) if why.contains("not sticky")), "{refused:?}");

        let attacker = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "unix::child_of_the_uid_swap_stress_swaps_two_folders_as_fast_as_it_can", "--test-threads=1", "--nocapture"])
            .env(ATTACK_DIR, &shared)
            .uid(65534)
            .gid(65534)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let (mut good, mut errors, mut none, mut attackers) = (0u64, 0u64, 0u64, 0u64);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(ATTACK_SECONDS + 1);
        while std::time::Instant::now() < deadline {
            match store.get(&n) {
                Ok(Some(v)) if *v == value => good += 1,
                Ok(Some(_)) => attackers += 1,
                Ok(None) => none += 1,
                Err(_) => errors += 1,
            }
        }
        let output = attacker.wait_with_output().unwrap();
        let text = String::from_utf8_lossy(&output.stdout);
        // (the harness prints the line after "test <name> ... ", so it is searched for, not matched from the start)
        let swaps: u64 = text
            .split("swaps=")
            .nth(1)
            .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|digits| digits.parse().ok())
            .unwrap_or_else(|| panic!("the attacker said nothing: {text}"));
        println!("uid swap stress: attacker swaps {swaps}; victim reads: {good} good, {errors} errors, {none} none, {attackers} attacker values");
        assert!(swaps > 100, "the attack did not happen ({swaps} swaps): the test proves nothing");
        assert_eq!(attackers, 0, "the victim read the attacker's value");
        assert_eq!(none, 0, "the victim was told that a key that exists was never stored");
        assert!(good > 0, "the victim never read its own value");
    }

    /// A FIFO, made by the system's `mkfifo` (it is on every Unix this crate builds for, macOS included, where `mknodat` is not), with mode 0600.
    fn make_fifo(path: &Path) {
        let status = std::process::Command::new("mkfifo").arg(path).status().expect("mkfifo runs");
        assert!(status.success(), "mkfifo failed");
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    }

    /// A FIFO with a perfectly good mode in place of a key file: opening it for reading blocks until someone opens it for writing, so a store that opens the
    /// path first and looks at what it is afterwards never returns (the reviewer's probe hung for ever). The file is opened without blocking and judged by
    /// `fstat` on the descriptor.
    #[test]
    fn a_fifo_in_place_of_a_key_file_is_refused_at_once_and_never_blocks() {
        let scratch = Scratch::new("fifo");
        let store = store(&scratch, ProviderChoice::Keyfile);
        store.put(&name("other.secret"), b"x").unwrap();
        let fifo = file_of(&scratch, "piped.secret", "kf");
        make_fifo(&fifo);
        assert_eq!(mode_of(&fifo), 0o600);
        let (sender, receiver) = std::sync::mpsc::channel();
        let store: Arc<dyn KeyStore> = Arc::from(store);
        let asked = Arc::clone(&store);
        thread::spawn(move || {
            let _ = sender.send(asked.get(&name("piped.secret")).map(|v| v.map(|_| ())));
        });
        let answer = receiver.recv_timeout(std::time::Duration::from_secs(20)).expect("get on a FIFO blocked");
        let error = answer.expect_err("a FIFO is not a key file");
        assert!(matches!(&error, KeyError::Permissions(why) if why.contains("not a regular file")), "{error:?}");
        // the FIFO is not listed, does not stop the store, and a put replaces it with a file of the store's own
        assert_eq!(store.list("").unwrap().len(), 1);
        store.put(&name("piped.secret"), b"now a file").unwrap();
        assert_eq!(get(&*store, &name("piped.secret")), Some(b"now a file".to_vec()));
    }

    /// A key file has one name: a hard link to it (from the same folder or another) is refused until the extra name is gone, and a put, which renames a new file over
    /// it, repairs it.
    #[test]
    fn a_key_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone() {
        let scratch = Scratch::new("hardlink");
        let store = store(&scratch, ProviderChoice::Keyfile);
        let n = name("vault.pins");
        store.put(&n, b"pins").unwrap();
        let elsewhere = scratch.0.join("another-name");
        fs::hard_link(file_of(&scratch, "vault.pins", "kf"), &elsewhere).unwrap();
        let error = store.get(&n).expect_err("two names for one key file");
        assert!(matches!(&error, KeyError::Permissions(why) if why.contains("hard links")), "{error:?}");
        fs::remove_file(&elsewhere).unwrap();
        assert_eq!(get(&*store, &n), Some(b"pins".to_vec()));
        fs::hard_link(file_of(&scratch, "vault.pins", "kf"), &elsewhere).unwrap();
        store.put(&n, b"pins again").unwrap();
        assert_eq!(get(&*store, &n), Some(b"pins again".to_vec()), "a put replaces the file with one that has one name");
        fs::remove_file(&elsewhere).unwrap();
    }

    /// The directories above the keys folder: whoever can rename the folder can put another in its place (the reviewer's attacker did it 1.1 million times
    /// in 40 seconds against a check-then-open store). World- or group-writable and not sticky, or another user's, is refused; sticky (as `/tmp` is) is not.
    #[test]
    fn a_folder_above_the_keys_folder_that_others_can_rename_in_is_refused_unless_it_is_sticky() {
        let scratch = Scratch::new("ancestors");
        let above = scratch.0.join("above");
        fs::create_dir(&above).unwrap();
        for (mode, ok) in [(0o755, true), (0o700, true), (0o1777, true), (0o777, false), (0o775, false), (0o757, false), (0o770, false)] {
            fs::set_permissions(&above, fs::Permissions::from_mode(mode)).unwrap();
            let result = open_at(above.join(format!("keys-{mode:o}")), ProviderChoice::Keyfile);
            match (ok, result) {
                (true, Ok(_)) => {}
                (false, Err(KeyError::Permissions(why))) => {
                    assert!(why.contains("above the keys directory") && why.contains("sticky"), "{mode:o}: {why}")
                }
                (ok, other) => panic!("mode {mode:o}: expected ok={ok}, got {:?}", other.err()),
            }
        }
        fs::set_permissions(&above, fs::Permissions::from_mode(0o700)).unwrap();
    }

    /// The chain is the one the file system has, not the one the path spells: a symbolic link to a folder under a world-writable one does not hide it.
    #[test]
    fn the_folders_above_are_those_of_the_real_folder_however_the_path_reaches_it() {
        let scratch = Scratch::new("physical");
        let shared = scratch.0.join("shared");
        fs::create_dir(&shared).unwrap();
        fs::create_dir(shared.join("real")).unwrap();
        fs::set_permissions(shared.join("real"), fs::Permissions::from_mode(0o700)).unwrap();
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o777)).unwrap();
        symlink(shared.join("real"), scratch.0.join("link")).unwrap();
        let error = open_at(scratch.0.join("link").join("keys"), ProviderChoice::Keyfile).err().expect("refused");
        assert!(matches!(&error, KeyError::Permissions(why) if why.contains("above the keys directory")), "{error:?}");
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(
            open_at(scratch.0.join("link").join("keys"), ProviderChoice::Keyfile).is_ok(),
            "the same path is fine once the folder above it is private"
        );
    }

    #[test]
    fn a_keys_path_that_is_a_file_is_refused() {
        let scratch = Scratch::new("notdir");
        fs::write(scratch.keys(), b"not a folder").unwrap();
        assert!(matches!(open_at(scratch.keys(), ProviderChoice::Keyfile), Err(KeyError::Permissions(why)) if why.contains("not a directory")));
    }
}

/// What Windows adds: a junction is a reparse point that `std` follows, and the folder is held open so that it cannot be swapped.
#[cfg(windows)]
mod windows {
    use super::*;

    fn junction(link: &Path, target: &Path) {
        let status = std::process::Command::new("cmd").args(["/C", "mklink", "/J"]).arg(link).arg(target).output().unwrap();
        assert!(status.status.success(), "mklink /J failed: {}", String::from_utf8_lossy(&status.stdout));
    }

    /// The reviewer's junction: `keys` is a junction to a folder somewhere else, so every value would be read from and written to a place the owner never chose.
    #[test]
    fn a_junction_in_place_of_the_keys_folder_is_refused_for_every_provider() {
        for (choice, _) in providers() {
            let scratch = Scratch::new("junction");
            let elsewhere = scratch.0.join("elsewhere");
            fs::create_dir(&elsewhere).unwrap();
            junction(&scratch.keys(), &elsewhere);
            let error = open_at(scratch.keys(), choice).err().expect("a junction is refused");
            assert!(matches!(&error, KeyError::Permissions(why) if why.contains("junction")), "{error:?}");
            assert!(fs::read_dir(&elsewhere).unwrap().next().is_none(), "nothing was written through the junction");
            fs::remove_dir(scratch.keys()).unwrap();
        }
    }

    #[test]
    fn a_junction_or_a_file_in_place_of_a_key_file_is_an_error_not_none() {
        for (choice, ext) in providers() {
            let scratch = Scratch::new("junction-file");
            let store = store(&scratch, choice);
            let elsewhere = scratch.0.join("elsewhere");
            fs::create_dir(&elsewhere).unwrap();
            junction(&file_of(&scratch, "linked.secret", ext), &elsewhere);
            assert!(store.get(&name("linked.secret")).is_err(), "{ext}: a junction where a key file should be");
            assert!(store.list("").unwrap().is_empty(), "a junction is not a key file, so it is not listed");
            fs::remove_dir(file_of(&scratch, "linked.secret", ext)).unwrap();
        }
    }

    /// K17: the kind of the refusal, not just that there is one. A `keys` path that is a file must be refused **as that**: without the check a DPAPI store would open
    /// it, and every read would fail later for some other reason or, worse, read as `None`.
    #[test]
    fn a_keys_path_that_is_a_file_is_refused_as_not_a_directory() {
        for (choice, _) in providers() {
            let scratch = Scratch::new("notdir");
            fs::write(scratch.keys(), b"not a folder").unwrap();
            let error = open_at(scratch.keys(), choice).err().expect("refused");
            assert!(matches!(&error, KeyError::Permissions(why) if why.contains("not a directory")), "{error:?}");
            assert_eq!(fs::read(scratch.keys()).unwrap(), b"not a folder", "and the file is left as it was");
        }
    }

    /// M-1, the reviewer's first case. A junction **above** the keys folder (`p`, pointing at a tree of the attacker's or of another place), with the store opened through
    /// it and the junction re-pointed while the store is open: the store answered from the other tree (`None` for a key that exists, a write into the other tree). The
    /// junction is a reparse point on the path, and a path with one on it is refused at open, so the store is never open through it to be re-pointed. Nothing is made
    /// through the junction either: not the keys folder, not the lock, not the marker.
    #[test]
    fn a_junction_above_the_keys_folder_is_refused_for_every_provider_and_nothing_is_made_through_it() {
        for (choice, _) in providers() {
            let scratch = Scratch::new("jabove");
            let (t1, t2) = (scratch.0.join("t1"), scratch.0.join("t2"));
            fs::create_dir_all(&t1).unwrap();
            fs::create_dir_all(t2.join("keys")).unwrap();
            let link = scratch.0.join("p");
            junction(&link, &t1);
            // the folder is not made through the junction (there is no t1\keys yet)
            let error = open_at(link.join("keys"), choice).err().expect("a junction on the way is refused");
            assert!(matches!(&error, KeyError::Permissions(why) if why.contains("junction") && why.contains("real directories")), "{error:?}");
            assert!(fs::read_dir(&t1).unwrap().next().is_none(), "the keys folder was not made in the tree the junction points to");
            // re-pointed at another tree, which has a keys folder of its own: the same refusal, and nothing is added to it
            fs::remove_dir(&link).unwrap();
            junction(&link, &t2);
            assert!(matches!(open_at(link.join("keys"), choice), Err(KeyError::Permissions(_))));
            assert!(fs::read_dir(t2.join("keys")).unwrap().next().is_none(), "no lock file, no marker, nothing in the other tree");
            // and the junction as the data folder itself, with `keys` inside it
            assert!(matches!(open_at(link, choice), Err(KeyError::Permissions(_))));
        }
    }

    /// M-1, the reviewer's rollback. The same user's keys folder exists twice, current and older (a backup or a snapshot), and a junction above `keys` is pointed at the older
    /// one: DPAPI blobs of one user stay valid, so a store that followed the junction would serve the old value as the current one. Here the store is opened at the real
    /// path of the current folder: re-pointing a junction that is not on its path changes nothing, and a store opened through the junction is refused, whichever copy
    /// it points at. `None` is never the answer, and neither is the old value.
    #[test]
    fn a_junction_pointed_at_an_older_copy_of_the_keys_folder_cannot_roll_a_value_back() {
        for (choice, ext) in providers() {
            let scratch = Scratch::new("rollback");
            let (current, old) = (scratch.0.join("current"), scratch.0.join("old"));
            let store = open_at(current.join("keys"), choice).unwrap();
            let pins = name("vault.pins");
            store.put(&pins, b"pins version 1 (old)").unwrap();
            fs::create_dir_all(old.join("keys")).unwrap();
            for entry in fs::read_dir(current.join("keys")).unwrap() {
                let entry = entry.unwrap();
                if !entry.file_name().to_string_lossy().ends_with(".lock") {
                    fs::copy(entry.path(), old.join("keys").join(entry.file_name())).unwrap();
                }
            }
            store.put(&pins, b"pins version 2 (current)").unwrap();
            let link = scratch.0.join("p");
            for target in [&current, &old] {
                junction(&link, target);
                assert!(
                    matches!(open_at(link.join("keys"), choice), Err(KeyError::Permissions(_))),
                    "{ext}: opened through a junction to {}",
                    target.display()
                );
                assert_eq!(
                    get(&*store, &pins),
                    Some(b"pins version 2 (current)".to_vec()),
                    "{ext}: the store that is open is unmoved by a junction that is not on its path"
                );
                fs::remove_dir(&link).unwrap();
            }
            // and the old copy is itself a perfectly good store, which shows what a rollback would have been
            drop(store);
            assert_eq!(get(&*open_at(old.join("keys"), choice).unwrap(), &pins), Some(b"pins version 1 (old)".to_vec()));
        }
    }

    /// Nothing above the keys folder can be replaced while the store is open: the real folders are held by the handle inside them. (This is the half of the guarantee that does
    /// hold for folders, and that a junction does not get.)
    #[test]
    fn the_real_folders_above_the_keys_folder_cannot_be_renamed_or_replaced_while_the_store_is_open() {
        for (choice, _) in providers() {
            let scratch = Scratch::new("held-above");
            let above = scratch.0.join("a").join("b");
            let store = open_at(above.join("keys"), choice).unwrap();
            store.put(&name("a.one"), b"1").unwrap();
            for folder in [above.clone(), scratch.0.join("a"), scratch.0.clone()] {
                assert!(fs::rename(&folder, folder.with_extension("moved")).is_err(), "{} was renamed under the store", folder.display());
                assert!(fs::remove_dir(&folder).is_err(), "{} was removed under the store", folder.display());
            }
            assert_eq!(get(&*store, &name("a.one")), Some(b"1".to_vec()));
        }
    }

    /// A key file has one name. A second name for the same bytes (a hard link) is a way to reach the secret from a place this store does not look after, and what the
    /// store reads is no longer "the file of this name". It is refused, for every provider, until the extra name is gone.
    #[test]
    fn a_key_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone() {
        for (choice, ext) in providers() {
            let scratch = Scratch::new("hardlink");
            let store = store(&scratch, choice);
            let n = name("vault.pins");
            store.put(&n, b"pins").unwrap();
            let elsewhere = scratch.0.join("another-name");
            fs::hard_link(file_of(&scratch, "vault.pins", ext), &elsewhere).unwrap();
            let error = store.get(&n).expect_err("two names for one key file");
            assert!(matches!(&error, KeyError::Permissions(why) if why.contains("hard links")), "{ext}: {error:?}");
            fs::remove_file(&elsewhere).unwrap();
            assert_eq!(get(&*store, &n), Some(b"pins".to_vec()), "{ext}: the same file with one name again");
            // a put replaces the file by renaming a new one over it, which has one name: a put repairs it
            fs::hard_link(file_of(&scratch, "vault.pins", ext), &elsewhere).unwrap();
            store.put(&n, b"pins again").unwrap();
            assert_eq!(get(&*store, &n), Some(b"pins again".to_vec()));
            fs::remove_file(&elsewhere).unwrap();
        }
    }
}
