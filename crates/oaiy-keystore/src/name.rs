//! The name of a secret: `^[a-z0-9][a-z0-9._-]{0,79}$` (design 4.5.1), one file per name, so a name is also a file name.
//!
//! Two rules beyond the regular expression, because a name becomes `<name>.ks` on every platform this ships on: it must not be one of
//! the names Windows reserves for devices (`con`, `prn`, `aux`, `nul`, `com0` to `com9`, `lpt0` to `lpt9`, judged on the part before
//! the first `.`, since `nul.ks` is still the NUL device), and the same rule is applied on every platform so that a name that is valid
//! on one is valid on all (a keystore folder is copied between machines by a backup and by hand).

use core::fmt;

use crate::error::KeyError;

/// The longest name.
pub const MAX_NAME_LEN: usize = 80;

/// A checked name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Name(String);

/// Names Windows treats as devices whatever extension follows.
const RESERVED: [&str; 4] = ["con", "prn", "aux", "nul"];

fn is_reserved(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or(name);
    if RESERVED.contains(&stem) {
        return true;
    }
    let bytes = stem.as_bytes();
    bytes.len() == 4 && (stem.starts_with("com") || stem.starts_with("lpt")) && bytes[3].is_ascii_digit()
}

impl Name {
    /// Checks `text` against the rule and returns the name.
    pub fn new(text: &str) -> Result<Name, KeyError> {
        let bytes = text.as_bytes();
        let shape = !bytes.is_empty()
            && bytes.len() <= MAX_NAME_LEN
            && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
            && bytes.iter().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-'));
        if !shape || is_reserved(text) {
            return Err(KeyError::InvalidName);
        }
        Ok(Name(text.to_string()))
    }

    /// The name as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The names of Release 1 (design 4.5.1) and of the consumers that asked for K1 (relay D1, apps V1). A consumer builds the
/// per-item ones (`vault.fk.<formId>`, `endpoint.x25519.<plugin>`) with `Name::new`.
pub mod names {
    /// Signer and X25519 pins, head version.
    pub const VAULT_PINS: &str = "vault.pins";
    /// The public phrase wrapper, for backup headers.
    pub const VAULT_WRAPPER: &str = "vault.wrapper";
    /// Prefix of a Form Key: `vault.fk.<formId>`.
    pub const VAULT_FK_PREFIX: &str = "vault.fk.";
    /// The archive writer seed.
    pub const ARCHIVE_WRITER: &str = "archive.writer";
    /// The archive writer certificate.
    pub const ARCHIVE_WRITER_CERT: &str = "archive.writer.cert";
    /// The archive chain head.
    pub const ARCHIVE_HEAD: &str = "archive.head";
    /// The backup manifest signing seed.
    pub const BACKUP_SIG: &str = "backup.sig";
    /// The pinned backup recipient public key.
    pub const BACKUP_PIN: &str = "backup.pin";
    /// The relay's token (relay design D1).
    pub const RELAY_TOKEN: &str = "relay.token";
    /// The relay host identity (relay design D1).
    pub const RELAY_HOST_IDENTITY: &str = "relay.host_identity";
    /// Prefix of an endpoint key: `endpoint.x25519.<plugin>` (relay design D1).
    pub const ENDPOINT_X25519_PREFIX: &str = "endpoint.x25519.";
    /// The provider link credential (apps design V1).
    pub const LINK_CREDENTIAL: &str = "link.credential";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_name_rule_over_a_table() {
        for good in [
            "a",
            "0",
            "vault.pins",
            "vault.fk.b7d2e4a1-5c3f-4d8e-9a6b-1f0e2d3c4b5a",
            "archive.writer.cert",
            "relay.host_identity",
            "endpoint.x25519.aokie",
            "a-b_c.d",
            "9lives",
            "a.",
            "a..b",
            "a-",
            "a_",
            &"a".repeat(80),
            "console",
            "comm",
            "com",
            "com10",
            "lpt",
            "auxiliary",
            "nulled",
            "con-x",
            "xcon",
            "a.con",
            "prn2",
        ] {
            assert!(Name::new(good).is_ok(), "{good:?} is a name");
        }
        for bad in [
            "",
            ".",
            "..",
            ".a",
            "-a",
            "_a",
            "A",
            "aB",
            "a b",
            "a/b",
            "a\\b",
            "a:b",
            "a*b",
            "a?b",
            "a\"b",
            "a<b",
            "a>b",
            "a|b",
            "a\0b",
            "é",
            "日本",
            "a\n",
            "a%2e",
            &"a".repeat(81),
            "con",
            "prn",
            "aux",
            "nul",
            "com0",
            "com1",
            "com9",
            "lpt0",
            "lpt1",
            "lpt9",
            "con.txt",
            "nul.ks",
            "aux.x.y",
            "com1.pins",
            "lpt3.a",
            "CON",
        ] {
            assert!(matches!(Name::new(bad), Err(KeyError::InvalidName)), "{bad:?} is not a name");
        }
        assert_eq!(Name::new("vault.pins").unwrap().as_str(), "vault.pins");
        assert_eq!(Name::new("vault.pins").unwrap().to_string(), "vault.pins");
    }

    /// The names the design lists are all names, and the per-item ones become names with an id after them.
    #[test]
    fn the_names_of_release_1_and_of_the_consumers_are_valid() {
        for text in [
            names::VAULT_PINS,
            names::VAULT_WRAPPER,
            names::ARCHIVE_WRITER,
            names::ARCHIVE_WRITER_CERT,
            names::ARCHIVE_HEAD,
            names::BACKUP_SIG,
            names::BACKUP_PIN,
            names::RELAY_TOKEN,
            names::RELAY_HOST_IDENTITY,
            names::LINK_CREDENTIAL,
        ] {
            assert!(Name::new(text).is_ok(), "{text}");
        }
        assert!(Name::new(&format!("{}{}", names::VAULT_FK_PREFIX, "b7d2e4a1-5c3f-4d8e-9a6b-1f0e2d3c4b5a")).is_ok());
        assert!(Name::new(&format!("{}{}", names::ENDPOINT_X25519_PREFIX, "aokie")).is_ok());
    }

    /// Nothing that passes can climb out of the directory, name a stream, or be a device: every name is one path component.
    #[test]
    fn a_name_is_always_one_plain_path_component() {
        let mut checked = 0u32;
        for a in 0u8..128 {
            for b in [b'a', b'.', b'-', b'_', b'0'] {
                let text = format!("{}{}", char::from(a), char::from(b));
                if Name::new(&text).is_ok() {
                    checked += 1;
                    assert!(!text.contains(['/', '\\', ':', '\0']), "{text:?}");
                    assert!(text != "." && text != "..");
                    let path = std::path::Path::new(&text);
                    assert_eq!(path.components().count(), 1, "{text:?}");
                    assert!(matches!(path.components().next(), Some(std::path::Component::Normal(_))), "{text:?}");
                }
            }
        }
        assert!(checked > 60);
    }
}
