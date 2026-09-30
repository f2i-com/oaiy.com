//! `ed25519` (design 4.1.1, 4.1.3, 4.1.4): detached Ed25519 signatures with strict verification only.
//!
//! **Verification is strict, always.** There is no non-strict path in this crate. A signature verifies only if S is below the
//! group order (a malleated S+L is refused), R and A are not points of small order, A is in its canonical encoding, and the
//! cofactorless equation holds with the R that was sent (so a non-canonical R cannot match either). This is what
//! libsodium does (checked case by case in the tests against libsodium's own verdicts) and what OpenSSL as shipped in Python's
//! `cryptography` 46.0.5 does not: it accepts a small-order key with `R = identity, S = 0` for every message (design finding A5).
//! A [`VerifyingKey`] cannot even be built from a key of small order or a non-canonical encoding: that is the check "at pin time".
//!
//! **A key has one role, and a role has its domains** (R-KEY, 4.1.4): a [`SigningKey`] made for [`KeyRole::Writer`] signs only
//! `flarch:1` strings, a `Backup` key only `flbackup:1`, a `Vault` key the vault's rows of design 4.1.3. The string to sign is a
//! [`SignedString`], built from a [`SigDomain`] (an enum: no caller-chosen prefix) and fields that each match
//! `[A-Za-z0-9_.:+@=/-]+`, so `|` and LF in a field are refused at build time. The one way to sign arbitrary bytes is a key made with
//! [`KeyRole::Hazmat`] and `sign_raw`, which exists for the RFC 8032 vectors and for other specs' messages; a caller that lets a
//! remote party choose the role or the bytes has built a signing oracle, and `grep Hazmat` finds every place that could.

use core::fmt;

use ed25519_dalek::{Signature as DalekSignature, Signer, SigningKey as DalekSigningKey, VerifyingKey as DalekVerifyingKey};
use zeroize::ZeroizeOnDrop;

use crate::canon::{join_pipe, Separator};
use crate::error::Error;
use crate::zeroize::Secret;

/// What a signing key is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyRole {
    /// The vault identity: `flmanifest:1`, `flgrant:1`, `flplacement:1`, `flnodecert:1`, `flvault-op:1`, `flvault-head:1`, `flwriter:1`.
    Vault,
    /// The archive writer key (K1 `archive.writer`): `flarch:1` only.
    Writer,
    /// The backup manifest signing seed (`flbksig1`): `flbackup:1` only.
    Backup,
    /// Any bytes, through `sign_raw`. For known-answer tests and for messages another specification defines.
    Hazmat,
}

/// The signature domains of design 4.1.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigDomain {
    /// `flmanifest:1|...`
    Manifest,
    /// `flgrant:1|...`
    Grant,
    /// `flplacement:1` LF canonical JSON
    Placement,
    /// `flnodecert:1` LF canonical JSON
    NodeCert,
    /// `flvault-op:1|...`
    VaultOp,
    /// `flvault-head:1|...`
    VaultHead,
    /// `flwriter:1|...`
    Writer,
    /// `flarch:1|...`
    Arch,
    /// `flbackup:1|...`
    Backup,
}

impl SigDomain {
    /// The token, version included.
    pub const fn token(self) -> &'static str {
        match self {
            SigDomain::Manifest => "flmanifest:1",
            SigDomain::Grant => "flgrant:1",
            SigDomain::Placement => "flplacement:1",
            SigDomain::NodeCert => "flnodecert:1",
            SigDomain::VaultOp => "flvault-op:1",
            SigDomain::VaultHead => "flvault-head:1",
            SigDomain::Writer => "flwriter:1",
            SigDomain::Arch => "flarch:1",
            SigDomain::Backup => "flbackup:1",
        }
    }

    /// What follows the token.
    pub const fn separator(self) -> Separator {
        match self {
            SigDomain::Placement | SigDomain::NodeCert => Separator::Lf,
            _ => Separator::Pipe,
        }
    }

    /// The role of the key that signs in this domain.
    pub const fn role(self) -> KeyRole {
        match self {
            SigDomain::Arch => KeyRole::Writer,
            SigDomain::Backup => KeyRole::Backup,
            _ => KeyRole::Vault,
        }
    }
}

/// A string ready to be signed or verified: its domain and its bytes, built only through the checked constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedString {
    domain: SigDomain,
    bytes: Vec<u8>,
}

impl SignedString {
    /// `token|f1|f2|...` for a `|` domain. At least one field; each must match `[A-Za-z0-9_.:+@=/-]+` (`Error::InvalidComponent`).
    pub fn pipe(domain: SigDomain, fields: &[&str]) -> Result<SignedString, Error> {
        if domain.separator() != Separator::Pipe || fields.is_empty() {
            return Err(Error::DomainNotAllowed);
        }
        Ok(SignedString { domain, bytes: join_pipe(domain.token(), fields)? })
    }

    /// `token LF payload` for an LF domain; the payload is the canonical JSON (RFC 8785) the caller made, non-empty UTF-8.
    pub fn lf(domain: SigDomain, payload: &[u8]) -> Result<SignedString, Error> {
        if domain.separator() != Separator::Lf {
            return Err(Error::DomainNotAllowed);
        }
        if payload.is_empty() || core::str::from_utf8(payload).is_err() {
            return Err(Error::InvalidComponent("payload"));
        }
        let mut bytes = Vec::with_capacity(domain.token().len() + 1 + payload.len());
        bytes.extend_from_slice(domain.token().as_bytes());
        bytes.push(b'\n');
        bytes.extend_from_slice(payload);
        Ok(SignedString { domain, bytes })
    }

    /// The domain.
    pub const fn domain(&self) -> SigDomain {
        self.domain
    }

    /// The bytes that are signed.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// A 64-byte Ed25519 signature (`R || S`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Signature([u8; 64]);

impl Signature {
    /// From 64 bytes. Whether they are a good signature is decided when it is verified.
    pub const fn from_bytes(bytes: &[u8; 64]) -> Signature {
        Signature(*bytes)
    }

    /// From a slice (`Error::InvalidLength` unless it is 64 bytes).
    pub fn from_slice(bytes: &[u8]) -> Result<Signature, Error> {
        let array: &[u8; 64] = bytes.try_into().map_err(|_| Error::InvalidLength("ed25519 signature"))?;
        Ok(Signature(*array))
    }

    /// The 64 bytes.
    pub const fn to_bytes(&self) -> [u8; 64] {
        self.0
    }
}

impl fmt::Debug for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ed25519::Signature({})", crate::kdf::hex_lower(&self.0))
    }
}

/// An Ed25519 public key that is canonical and not of small order.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct VerifyingKey(DalekVerifyingKey);

impl VerifyingKey {
    /// Parses and checks a public key: on the curve (`InvalidKey`), in its one canonical encoding (`NonCanonicalKey`), not of
    /// small order (`SmallOrderKey`).
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<VerifyingKey, Error> {
        let inner = DalekVerifyingKey::from_bytes(bytes).map_err(|_| Error::InvalidKey)?;
        // dalek keeps the bytes it was given and decodes y modulo p, so a y >= p (or a sign bit on x = 0) parses. libsodium refuses
        // those encodings; re-encoding the point and comparing is the same test.
        if inner.to_edwards().compress().to_bytes() != *bytes {
            return Err(Error::NonCanonicalKey);
        }
        if inner.is_weak() {
            return Err(Error::SmallOrderKey);
        }
        Ok(VerifyingKey(inner))
    }

    /// The same from a slice (`Error::InvalidLength` unless it is 32 bytes).
    pub fn from_slice(bytes: &[u8]) -> Result<VerifyingKey, Error> {
        let array: &[u8; 32] = bytes.try_into().map_err(|_| Error::InvalidLength("ed25519 public key"))?;
        VerifyingKey::from_bytes(array)
    }

    /// The 32 bytes.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    /// Verifies a signature over a domain string, strictly.
    pub fn verify(&self, message: &SignedString, signature: &Signature) -> Result<(), Error> {
        self.verify_raw(message.as_bytes(), signature)
    }

    /// Verifies a signature over any bytes, strictly. (Verifying arbitrary bytes is not dangerous; signing them is.)
    pub fn verify_raw(&self, message: &[u8], signature: &Signature) -> Result<(), Error> {
        self.0.verify_strict(message, &DalekSignature::from_bytes(&signature.0)).map_err(|_| Error::SignatureInvalid)
    }
}

impl fmt::Debug for VerifyingKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ed25519::VerifyingKey({})", crate::kdf::hex_lower(self.0.as_bytes()))
    }
}

/// An Ed25519 signing key with a role, zeroized when dropped.
pub struct SigningKey {
    inner: DalekSigningKey,
    role: KeyRole,
}

impl SigningKey {
    /// From a 32-byte seed (RFC 8032 section 5.1.5; libsodium's `crypto_sign_seed_keypair`).
    pub fn from_seed(role: KeyRole, seed: &Secret<32>) -> SigningKey {
        SigningKey { inner: DalekSigningKey::from_bytes(seed.expose()), role }
    }

    /// A fresh key from the operating system's random generator.
    pub fn generate(role: KeyRole) -> Result<SigningKey, Error> {
        let seed: Secret<32> = Secret::random()?;
        Ok(SigningKey::from_seed(role, &seed))
    }

    /// From libsodium's 64-byte secret key (`seed || public key`, the form FormLogic's key bundle stores). The public half must be
    /// the one the seed gives: a mismatch is `Error::InvalidKey`, so a bundle cannot make this code sign under one public key
    /// with another key's seed.
    pub fn from_libsodium_secret_key(role: KeyRole, secret_key: &Secret<64>) -> Result<SigningKey, Error> {
        let inner = DalekSigningKey::from_keypair_bytes(secret_key.expose()).map_err(|_| Error::InvalidKey)?;
        Ok(SigningKey { inner, role })
    }

    /// The role.
    pub const fn role(&self) -> KeyRole {
        self.role
    }

    /// The seed, copied into a zeroizing container (the K1 item `archive.writer` is this).
    pub fn seed(&self) -> Secret<32> {
        Secret::new(self.inner.to_bytes())
    }

    /// The public key. It is canonical and of large order by construction, so it needs no check.
    pub fn verifying_key(&self) -> VerifyingKey {
        VerifyingKey(self.inner.verifying_key())
    }

    /// Signs a domain string, if this key's role is the domain's role (or `Hazmat`).
    pub fn sign(&self, message: &SignedString) -> Result<Signature, Error> {
        if self.role != KeyRole::Hazmat && self.role != message.domain().role() {
            return Err(Error::DomainNotAllowed);
        }
        Ok(Signature(self.inner.sign(message.as_bytes()).to_bytes()))
    }

    /// Signs any bytes. Only a `Hazmat` key may.
    pub fn sign_raw(&self, message: &[u8]) -> Result<Signature, Error> {
        if self.role != KeyRole::Hazmat {
            return Err(Error::DomainNotAllowed);
        }
        Ok(Signature(self.inner.sign(message).to_bytes()))
    }
}

impl ZeroizeOnDrop for SigningKey {}

impl fmt::Debug for SigningKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ed25519::SigningKey({:?}, redacted)", self.role)
    }
}
