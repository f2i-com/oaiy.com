//! One error type for the whole crate. Every variant has a stable machine code (`code()`), and none of them
//! carries key material, plaintext, a byte of ciphertext or a position in any input: the message says what was
//! refused, never why a secret did not match. A failed decryption is always the same `DecryptFailed`, whatever
//! part of the input was wrong (no oracle for which field a forger got right).

use core::fmt;

/// What an operation of this crate refused, and nothing more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A number of iterations, an amount of memory or a salt length outside the bounds of the design (4.1.2).
    KdfParamsOutOfRange,
    /// An input of the wrong length; the field is named, the bytes are not.
    InvalidLength(&'static str),
    /// A decryption (XChaCha20-Poly1305, a sealed box) that did not authenticate, for any reason.
    DecryptFailed,
    /// A point of small order, or a Diffie-Hellman result that is all zero (RFC 7748 section 6.1).
    LowOrderPoint,
    /// An Ed25519 public key that is not a point of the curve.
    InvalidKey,
    /// An Ed25519 public key of small order (refused at pin time, 4.1.3 rule 4).
    SmallOrderKey,
    /// An Ed25519 public key that is a point of the curve but not in its one canonical encoding.
    NonCanonicalKey,
    /// An Ed25519 signature that did not verify strictly (any reason: S not below the group order, small-order
    /// R or A, a wrong message, a wrong key).
    SignatureInvalid,
    /// A key of one role asked to sign in a domain of another (4.1.4, R-KEY), or a raw signature asked of a key that
    /// has a role.
    DomainNotAllowed,
    /// A field of a signed string or an AAD that is empty, or holds a character outside `[A-Za-z0-9_.:+@=/-]`
    /// (which includes `|` and LF). The field is named, its content is not.
    InvalidComponent(&'static str),
    /// A recovery phrase that does not have exactly twelve words.
    PhraseLength,
    /// A recovery phrase with a word that is not in the list.
    PhraseWord,
    /// A recovery phrase whose four check bits do not match its entropy.
    PhraseChecksum,
    /// A recovery kit code that does not have the FLRK1 shape (prefix, length, characters, trailing bits).
    KitFormat,
    /// A recovery kit code whose checksum does not match its key.
    KitChecksum,
    /// A KDF context that is not eight characters of `[a-z0-9]`, or a subkey length outside 16 to 64 bytes.
    KdfContext,
    /// HKDF was asked for more than 255 blocks of output.
    HkdfLength,
    /// The operating system's random generator failed.
    Random,
    /// The memory for a key derivation could not be reserved.
    Memory,
}

impl Error {
    /// The stable identifier of the refusal, as the design writes them (`kdf_params_out_of_range`, `phrase_word`...).
    pub const fn code(&self) -> &'static str {
        match self {
            Error::KdfParamsOutOfRange => "kdf_params_out_of_range",
            Error::InvalidLength(_) => "invalid_length",
            Error::DecryptFailed => "decrypt_failed",
            Error::LowOrderPoint => "low_order_point",
            Error::InvalidKey => "invalid_key",
            Error::SmallOrderKey => "small_order_key",
            Error::NonCanonicalKey => "non_canonical_key",
            Error::SignatureInvalid => "signature_invalid",
            Error::DomainNotAllowed => "domain_not_allowed",
            Error::InvalidComponent(_) => "invalid_component",
            Error::PhraseLength => "phrase_length",
            Error::PhraseWord => "phrase_word",
            Error::PhraseChecksum => "phrase_checksum",
            Error::KitFormat => "kit_format",
            Error::KitChecksum => "kit_checksum",
            Error::KdfContext => "kdf_context",
            Error::HkdfLength => "hkdf_length",
            Error::Random => "random_failed",
            Error::Memory => "out_of_memory",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidLength(what) => write!(f, "invalid_length: {what}"),
            Error::InvalidComponent(what) => write!(f, "invalid_component: {what}"),
            other => f.write_str(other.code()),
        }
    }
}

impl std::error::Error for Error {}
