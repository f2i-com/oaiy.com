//! Canonical strings: the shape every AAD and every signed string of the vault takes (design 4.1.3 rules 1 to 3, 4.1.5).
//!
//! A canonical string is `token|field|field|...` (or `token LF payload` for the two data-node domains), UTF-8, where
//! the token names the domain, every other field matches `[A-Za-z0-9_.:+@=/-]+`, and free text (a label, a name) enters
//! only as `sha256hex(utf8(label))`. The builders refuse an empty field and any field with a character outside that
//! set, which includes `|` and LF, so a field can never shift the others (`x|2027-01-01` as a label would). The domain
//! registry below is append-only, and a test proves that no token is a prefix of another followed by the same separator
//! (rule 1), so a string made for one purpose cannot be read as another.
//!
//! There is deliberately no function here (or anywhere in the crate) that signs or encrypts under a domain chosen by the
//! caller as text: the domain is an enum, and the only path to a signature over arbitrary bytes is `KeyRole::Hazmat`.

use crate::ed25519::KeyRole;
use crate::error::Error;

/// The byte after the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Separator {
    /// `|`: the domain's fields follow, one per `|`.
    Pipe,
    /// LF: a canonical JSON payload follows (the data-node domains).
    Lf,
}

impl Separator {
    /// The separator as a byte.
    pub const fn byte(self) -> u8 {
        match self {
            Separator::Pipe => b'|',
            Separator::Lf => b'\n',
        }
    }
}

/// What a domain is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainUse {
    /// The first component of an AAD.
    Aad,
    /// The start of a signed string, made by a key of this role.
    Signature(KeyRole),
}

/// One row of the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainSpec {
    /// The token, including its version (`flvault-op:1`).
    pub token: &'static str,
    /// What follows the token.
    pub separator: Separator,
    /// What the domain is for.
    pub usage: DomainUse,
}

/// The registry (append-only). AAD domains first, then signature domains, in the order of the design's table 4.1.3.
pub const DOMAINS: &[DomainSpec] = &[
    // AAD domains: wrappers of the vault (FormLogic's vault.ts, 4.2.1) and the response envelope.
    DomainSpec { token: "flenc:1", separator: Separator::Pipe, usage: DomainUse::Aad },
    DomainSpec { token: "flvault-umk:1", separator: Separator::Pipe, usage: DomainUse::Aad },
    DomainSpec { token: "flvault-bundle:1", separator: Separator::Pipe, usage: DomainUse::Aad },
    DomainSpec { token: "flvault-umk-recovery:1", separator: Separator::Pipe, usage: DomainUse::Aad },
    DomainSpec { token: "flvault-wrap:2", separator: Separator::Pipe, usage: DomainUse::Aad },
    DomainSpec { token: "flingest:1", separator: Separator::Pipe, usage: DomainUse::Aad },
    DomainSpec { token: "flvault-fk:1", separator: Separator::Pipe, usage: DomainUse::Aad },
    // Signature domains.
    DomainSpec { token: "flmanifest:1", separator: Separator::Pipe, usage: DomainUse::Signature(KeyRole::Vault) },
    DomainSpec { token: "flgrant:1", separator: Separator::Pipe, usage: DomainUse::Signature(KeyRole::Vault) },
    DomainSpec { token: "flplacement:1", separator: Separator::Lf, usage: DomainUse::Signature(KeyRole::Vault) },
    DomainSpec { token: "flnodecert:1", separator: Separator::Lf, usage: DomainUse::Signature(KeyRole::Vault) },
    DomainSpec { token: "flvault-op:1", separator: Separator::Pipe, usage: DomainUse::Signature(KeyRole::Vault) },
    DomainSpec { token: "flvault-head:1", separator: Separator::Pipe, usage: DomainUse::Signature(KeyRole::Vault) },
    DomainSpec { token: "flwriter:1", separator: Separator::Pipe, usage: DomainUse::Signature(KeyRole::Vault) },
    DomainSpec { token: "flarch:1", separator: Separator::Pipe, usage: DomainUse::Signature(KeyRole::Writer) },
    DomainSpec { token: "flbackup:1", separator: Separator::Pipe, usage: DomainUse::Signature(KeyRole::Backup) },
];

/// Design 4.1.3 rule 1: the first token up to `|` or LF differs in every row, and no row's token followed by its separator
/// is a prefix of another row's token followed by its separator. Returns the first offending pair of rows.
pub fn prefix_free(specs: &[DomainSpec]) -> Result<(), (usize, usize)> {
    for (i, a) in specs.iter().enumerate() {
        for (j, b) in specs.iter().enumerate().skip(i + 1) {
            let mut sa = a.token.as_bytes().to_vec();
            sa.push(a.separator.byte());
            let mut sb = b.token.as_bytes().to_vec();
            sb.push(b.separator.byte());
            let same_token = a.token == b.token;
            if same_token || sa.starts_with(&sb) || sb.starts_with(&sa) || a.token.starts_with(b.token) || b.token.starts_with(a.token) {
                return Err((i, j));
            }
        }
    }
    Ok(())
}

/// Design 4.1.3 rule 3: a field is one or more of `[A-Za-z0-9_.:+@=/-]`. Nothing else: no `|`, no LF, no space.
pub fn is_component(field: &str) -> bool {
    !field.is_empty() && field.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'+' | b'@' | b'=' | b'/' | b'-'))
}

/// Checks one field, naming it in the error.
pub(crate) fn check_component(name: &'static str, field: &str) -> Result<(), Error> {
    if is_component(field) {
        Ok(())
    } else {
        Err(Error::InvalidComponent(name))
    }
}

/// `token|f1|f2|...`, every field checked.
pub(crate) fn join_pipe(token: &str, fields: &[&str]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(token.len() + fields.iter().map(|f| f.len() + 1).sum::<usize>());
    out.extend_from_slice(token.as_bytes());
    for field in fields {
        check_component("field", field)?;
        out.push(b'|');
        out.extend_from_slice(field.as_bytes());
    }
    Ok(out)
}

/// A checked AAD: `domain|field|field|...`, never empty, never with a field that could shift the others.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aad(Vec<u8>);

/// The AAD domains of the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AadDomain {
    /// `flenc:1`: a response envelope's content (FormLogic).
    Enc,
    /// `flvault-umk:1`: the UMK wrapped by the passphrase key (FormLogic).
    VaultUmk,
    /// `flvault-bundle:1`: the key bundle under the UMK (FormLogic).
    VaultBundle,
    /// `flvault-umk-recovery:1`: the UMK wrapped by the recovery kit (FormLogic).
    VaultUmkRecovery,
    /// `flvault-wrap:2`: a wrapper of design 4.2.1 (phrase).
    VaultWrap,
    /// `flingest:1`: an ingestion secret wrapped under a Form Key (4.4.2).
    Ingest,
    /// `flvault-fk:1`: the browser device cache (registered so that no other use takes the name; this crate does not do AES-GCM).
    VaultFk,
}

impl AadDomain {
    /// The token.
    pub const fn token(self) -> &'static str {
        match self {
            AadDomain::Enc => "flenc:1",
            AadDomain::VaultUmk => "flvault-umk:1",
            AadDomain::VaultBundle => "flvault-bundle:1",
            AadDomain::VaultUmkRecovery => "flvault-umk-recovery:1",
            AadDomain::VaultWrap => "flvault-wrap:2",
            AadDomain::Ingest => "flingest:1",
            AadDomain::VaultFk => "flvault-fk:1",
        }
    }
}

impl Aad {
    /// Builds `domain|fields...`. Each field must match `[A-Za-z0-9_.:+@=/-]+` (use `-` for "none", and
    /// `sha256hex(label)` for text).
    pub fn new(domain: AadDomain, fields: &[&str]) -> Result<Aad, Error> {
        Ok(Aad(join_pipe(domain.token(), fields)?))
    }

    /// The bytes to hand to the AEAD.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}
