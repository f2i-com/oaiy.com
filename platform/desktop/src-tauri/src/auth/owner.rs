//! `<data>/auth/owner.json`: the owner's password hash, the login's settings and the known devices (design 4.1).
//!
//! ```json
//! {"v":1,"created_ms":1790000000000,"password_changed_ms":1790000000000,"min_session_epoch":0,
//!  "password":"$argon2id$v=19$m=65536,t=3,p=1$...$...","factors":["password"],"passkeys":[],
//!  "devices":[{"id":"89abcdef01234567","hash":"<sha256 hex of the device secret>","created_ms":...,"last_ms":...,"ip":"203.0.113.9"}]}
//! ```
//!
//! The running server is the only writer of it, and memory is authoritative: it is read once, at start, by the
//! credential store (which refuses a file it cannot understand: an unparsable `owner.json` is a startup error,
//! never "setup-only", and only `ENOENT` means there is no owner yet). Fields this build does not know are kept
//! and written back, so an additive field never needs a new `v`. Nothing here is ever shown by `Debug`: not the
//! password hash (an offline guesser's target) and not the device hashes.
//!
//! It is written atomically and privately through the store's [`FileWriter`], so that the tests can make the disk
//! full, and created without replacing another with [`create_exclusive`].

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::store::{FileWriter, FILE_VERSION};

/// One known device: a browser that has logged in with the password, recognised by its `dev` cookie.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    /// `lowercase_hex(SHA-256(secret))` of the cookie's secret.
    pub hash: String,
    pub created_ms: u64,
    pub last_ms: u64,
    #[serde(default)]
    pub ip: String,
    /// Fields a newer OAIY wrote.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("id", &self.id)
            .field("hash", &"[redacted]")
            .field("created_ms", &self.created_ms)
            .field("last_ms", &self.last_ms)
            .field("ip", &self.ip)
            .finish()
    }
}

fn default_factors() -> Vec<String> {
    vec!["password".to_string()]
}

/// The owner file.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerDoc {
    pub v: u64,
    pub created_ms: u64,
    pub password_changed_ms: u64,
    #[serde(default)]
    pub min_session_epoch: u64,
    /// The PHC string of the password.
    pub password: String,
    #[serde(default = "default_factors")]
    pub factors: Vec<String>,
    #[serde(default)]
    pub passkeys: Vec<Value>,
    #[serde(default)]
    pub devices: Vec<Device>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl std::fmt::Debug for OwnerDoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnerDoc")
            .field("v", &self.v)
            .field("created_ms", &self.created_ms)
            .field("password_changed_ms", &self.password_changed_ms)
            .field("password", &"[redacted]")
            .field("devices", &self.devices.len())
            .finish()
    }
}

impl OwnerDoc {
    /// A new owner with this password hash.
    pub fn new(now_ms: u64, password_phc: String) -> OwnerDoc {
        OwnerDoc {
            v: FILE_VERSION,
            created_ms: now_ms,
            password_changed_ms: now_ms,
            min_session_epoch: 0,
            password: password_phc,
            factors: default_factors(),
            passkeys: Vec::new(),
            devices: Vec::new(),
            extra: Map::new(),
        }
    }

    /// From the document the credential store read at start. The store has already checked the `v`.
    pub fn from_value(doc: &Value) -> Result<OwnerDoc, String> {
        serde_json::from_value(doc.clone()).map_err(|e| e.to_string())
    }

    /// The text of the file.
    pub fn to_text(&self) -> String {
        let mut text = serde_json::to_string_pretty(self).unwrap_or_default();
        text.push('\n');
        text
    }
}

/// `<data>/auth/owner.json`.
pub fn path(auth_dir: &Path) -> PathBuf {
    auth_dir.join("owner.json")
}

/// Write the file, atomically and privately, replacing the old one.
pub fn write(writer: &dyn FileWriter, auth_dir: &Path, doc: &OwnerDoc) -> io::Result<()> {
    writer.write(&path(auth_dir), doc.to_text().as_bytes())
}

/// Why an exclusive create did not happen.
#[derive(Debug)]
pub enum CreateError {
    /// There is an owner already.
    Exists,
    Io(io::Error),
}

/// Create the file only if there is none: the bytes are staged in a file of their own (atomic and private, as
/// every write is) and linked into place, which fails if `owner.json` exists, atomically. Two setups that both
/// hold a valid code cannot both succeed. Where the file system has no hard links the file is written after a
/// check, which is exclusive too because the caller holds the login's setup lock and the data folder's lock is
/// held by this process alone.
pub fn create_exclusive(
    writer: &dyn FileWriter,
    auth_dir: &Path,
    doc: &OwnerDoc,
) -> Result<(), CreateError> {
    let target = path(auth_dir);
    if target.exists() {
        return Err(CreateError::Exists);
    }
    let staging = auth_dir.join("owner.json.creating");
    writer
        .write(&staging, doc.to_text().as_bytes())
        .map_err(CreateError::Io)?;
    let linked = std::fs::hard_link(&staging, &target);
    let _ = std::fs::remove_file(&staging);
    match linked {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Err(CreateError::Exists),
        Err(_) => {
            // No hard links here: check again and write.
            if target.exists() {
                return Err(CreateError::Exists);
            }
            write(writer, auth_dir, doc).map_err(CreateError::Io)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::store::SecureWriter;
    use crate::secret_file::testing::TempDir;
    use serde_json::json;

    const PHC: &str = "$argon2id$v=19$m=65536,t=3,p=1$AAECAwQFBgcICQoLDA0ODw$DRo8ZSPI8G5OCvnFFapbVEjP69aDjy1Sw9i2743cPC4";

    #[test]
    fn the_design_example_parses_and_unknown_fields_are_written_back() {
        let doc = json!({
            "v": 1, "created_ms": 1790000000000u64, "password_changed_ms": 1790000000000u64,
            "min_session_epoch": 0, "password": PHC, "factors": ["password"], "passkeys": [],
            "devices": [{"id": "89abcdef01234567", "hash": "ab", "created_ms": 1, "last_ms": 2, "ip": "203.0.113.9", "note": "kept"}],
            "future_field": {"a": 1}
        });
        let owner = OwnerDoc::from_value(&doc).unwrap();
        assert_eq!(owner.devices.len(), 1);
        assert_eq!(owner.devices[0].extra["note"], "kept");
        let text = owner.to_text();
        let again: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            again, doc,
            "a round trip changes nothing, unknown fields included"
        );
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn a_file_with_only_what_a_new_owner_has_fills_the_defaults() {
        let owner = OwnerDoc::from_value(&json!({
            "v": 1, "created_ms": 5, "password_changed_ms": 5, "password": PHC
        }))
        .unwrap();
        assert_eq!(owner.factors, ["password"]);
        assert!(owner.passkeys.is_empty() && owner.devices.is_empty());
        assert_eq!(owner.min_session_epoch, 0);
        // No password at all is not an owner.
        assert!(OwnerDoc::from_value(
            &json!({ "v": 1, "created_ms": 5, "password_changed_ms": 5 })
        )
        .is_err());
    }

    #[test]
    fn debug_shows_neither_the_password_hash_nor_a_device_hash() {
        let mut owner = OwnerDoc::new(1, PHC.to_string());
        owner.devices.push(Device {
            id: "89abcdef01234567".into(),
            hash: "0123deviceHASH".into(),
            created_ms: 1,
            last_ms: 1,
            ip: "203.0.113.9".into(),
            extra: Map::new(),
        });
        let shown = format!("{owner:?} {:?}", owner.devices[0]);
        assert!(
            !shown.contains("argon2") && !shown.contains("DRo8ZSPI"),
            "{shown}"
        );
        assert!(!shown.contains("deviceHASH"), "{shown}");
        assert!(shown.contains("89abcdef01234567"));
    }

    #[test]
    fn create_exclusive_makes_the_file_once_and_a_second_create_is_refused_without_touching_it() {
        let dir = TempDir::new("owner-exclusive");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        let first = OwnerDoc::new(1, PHC.to_string());
        create_exclusive(&SecureWriter, &auth, &first).unwrap();
        let text = std::fs::read_to_string(path(&auth)).unwrap();
        let mut second = OwnerDoc::new(2, PHC.replace("AAEC", "BBEC"));
        second.created_ms = 99;
        assert!(matches!(
            create_exclusive(&SecureWriter, &auth, &second),
            Err(CreateError::Exists)
        ));
        assert_eq!(
            std::fs::read_to_string(path(&auth)).unwrap(),
            text,
            "untouched"
        );
        // The staging file does not linger.
        assert!(!auth.join("owner.json.creating").exists());
    }

    /// A writer that lets another setup win the race: while this one stages its bytes, the other links its own
    /// `owner.json` into place.
    struct Racer<'a> {
        winner: &'a Path,
        text: &'a str,
        stage: bool,
    }

    impl FileWriter for Racer<'_> {
        fn write(&self, at: &Path, bytes: &[u8]) -> io::Result<()> {
            if at.file_name().is_some_and(|n| n == "owner.json.creating") {
                std::fs::write(self.winner, self.text)?;
                if self.stage {
                    std::fs::write(at, bytes)?;
                }
                return Ok(());
            }
            std::fs::write(at, bytes)
        }
    }

    #[test]
    fn a_setup_that_loses_the_race_for_the_file_is_told_there_is_an_owner_and_the_winners_file_stays(
    ) {
        let dir = TempDir::new("owner-race");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        let winner = path(&auth);
        // The other setup's file appears after this one's check and before its link: the link finds it there.
        let racer = Racer {
            winner: &winner,
            text: "the winner\n",
            stage: true,
        };
        let mine = OwnerDoc::new(3, PHC.replace("AAEC", "CCEC"));
        assert!(matches!(
            create_exclusive(&racer, &auth, &mine),
            Err(CreateError::Exists)
        ));
        assert_eq!(std::fs::read_to_string(&winner).unwrap(), "the winner\n");
        assert!(!auth.join("owner.json.creating").exists());
    }

    #[test]
    fn where_links_are_not_possible_the_file_is_checked_again_and_then_written() {
        // The staged file is not there to link (a file system without hard links looks the same to the caller): the
        // second check finds a winner ...
        let dir = TempDir::new("owner-nolink");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        let winner = path(&auth);
        let racer = Racer {
            winner: &winner,
            text: "the winner\n",
            stage: false,
        };
        assert!(matches!(
            create_exclusive(&racer, &auth, &OwnerDoc::new(4, PHC.to_string())),
            Err(CreateError::Exists)
        ));
        assert_eq!(std::fs::read_to_string(&winner).unwrap(), "the winner\n");
        // ... and with no winner the owner is written in place.
        let dir = TempDir::new("owner-nolink-alone");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        struct Unlinkable;
        impl FileWriter for Unlinkable {
            fn write(&self, at: &Path, bytes: &[u8]) -> io::Result<()> {
                if at.file_name().is_some_and(|n| n == "owner.json.creating") {
                    return Ok(());
                }
                std::fs::write(at, bytes)
            }
        }
        create_exclusive(&Unlinkable, &auth, &OwnerDoc::new(5, PHC.to_string())).unwrap();
        assert!(OwnerDoc::from_value(
            &serde_json::from_str::<Value>(&std::fs::read_to_string(path(&auth)).unwrap()).unwrap()
        )
        .is_ok());
    }

    #[test]
    fn create_exclusive_reports_a_write_that_fails_and_leaves_no_owner() {
        struct Full;
        impl FileWriter for Full {
            fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
                Err(io::Error::other("disk full"))
            }
        }
        let dir = TempDir::new("owner-full");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        assert!(matches!(
            create_exclusive(&Full, &auth, &OwnerDoc::new(1, PHC.to_string())),
            Err(CreateError::Io(_))
        ));
        assert!(!path(&auth).exists());
    }

    #[test]
    fn writing_replaces_the_file_atomically() {
        let dir = TempDir::new("owner-write");
        let auth = dir.0.join("auth");
        let mut owner = OwnerDoc::new(1, PHC.to_string());
        write(&SecureWriter, &auth, &owner).unwrap();
        owner.password_changed_ms = 77;
        write(&SecureWriter, &auth, &owner).unwrap();
        let read: Value =
            serde_json::from_str(&std::fs::read_to_string(path(&auth)).unwrap()).unwrap();
        assert_eq!(read["password_changed_ms"], 77);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path(&auth)).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }
}
