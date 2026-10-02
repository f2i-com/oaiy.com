//! Keys, signatures and the protocol's domain strings (README section 10.7), on top of `oaiy-crypto` (nothing here implements a primitive).
//!
//! **Every signature of the protocol is over a domain string, a zero byte and the text** (`"oaiy/relay/1/cmd" || 0x00 || bytes`), so a value made for one purpose never
//! verifies as another. [`SignDomain`] is an enum, and [`Signer::sign`] takes it, so no caller can sign under a domain of its own or over a text without one. `oaiy-crypto`
//! offers one way to sign arbitrary bytes, a key of role `Hazmat`, and that is what a [`Signer`] holds: this file is the only place in the crate that makes one, and
//! `grep Hazmat` finds it.
//!
//! **Keys of small order, and verification, are refused by `oaiy-crypto`** and are not repeated here: [`VerifyKey::from_bytes`] cannot be built from an Ed25519 key that
//! is not in its canonical encoding or is of small order, every verification is strict (a malleated `S`, a small-order `R`), and [`X25519Public::from_bytes`] cannot be
//! built from one of the fourteen encodings (seven values, each with and without bit 255) whose Diffie-Hellman result is all zero (vector A12).

use oaiy_crypto::ed25519::{KeyRole, Signature, SigningKey, VerifyingKey};
use oaiy_crypto::kdf::sha256;
use oaiy_crypto::sealbox;
use oaiy_crypto::x25519::{PublicKey as DalekPublic, SecretKey as DalekSecret};
use oaiy_crypto::zeroize::{Secret, SecretVec};

use crate::b64;
use crate::error::{Error, Result};

/// The domains under which something is signed (README section 10.7). The MACs and the key derivations of pairing have their own constants in [`crate::pairing`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignDomain {
    /// The relay's static signature of `GET /v1/info` (`X-OAIY-Sig`): made by the relay key.
    Info,
    /// The relay's identity proof: `domain || 0x00 || nonce bytes || SHA-256(body) || ASCII decimal time`, made by the relay key.
    InfoProof,
    /// An enrolment request: made by the key derived from the enrolment secret, over the raw request body.
    Enroll,
    /// A command's signed bytes: made by a provider's signing key.
    Cmd,
    /// A result's signed bytes: made by the host identity.
    Res,
    /// A `sync` item's signed bytes: made by the host identity.
    Sync,
    /// A provider's rotation statement: made by the currently pinned provider key.
    ProviderRotate,
    /// A ring's body text, carried in `hdr.sig`: made by the host identity.
    Ring,
    /// The pairing response's canonical claims: made by the phone's endpoint key.
    PairingResponse,
    /// The approval receipt's canonical document: made by the desktop's endpoint key.
    PairingApproval,
}

impl SignDomain {
    /// The domain string.
    pub const fn as_str(self) -> &'static str {
        match self {
            SignDomain::Info => "oaiy/relay/1/info",
            SignDomain::InfoProof => "oaiy/relay/1/info-proof",
            SignDomain::Enroll => "oaiy/relay/1/enroll",
            SignDomain::Cmd => "oaiy/relay/1/cmd",
            SignDomain::Res => "oaiy/relay/1/res",
            SignDomain::Sync => "oaiy/relay/1/sync",
            SignDomain::ProviderRotate => "oaiy/relay/1/provider-rotate",
            SignDomain::Ring => "oaiy/relay/1/ring",
            SignDomain::PairingResponse => "oaiy/pairing/3/response",
            SignDomain::PairingApproval => "oaiy/pairing/3/approval",
        }
    }

    /// `domain || 0x00 || part || part ...`: the bytes that are signed.
    pub fn message(self, parts: &[&[u8]]) -> Vec<u8> {
        domain_message(self.as_str(), parts)
    }
}

/// `domain || 0x00 || parts` for any domain string (the MACs of pairing are built with it too).
pub fn domain_message(domain: &str, parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(domain.len() + 1 + parts.iter().map(|p| p.len()).sum::<usize>());
    out.extend_from_slice(domain.as_bytes());
    out.push(0);
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

/// The thumbprint of an Ed25519 public key: `b64u(SHA-256(`{"crv":"Ed25519","kty":"OKP","x":"<b64u>"}`))`, 43 characters (README section 2). The same function as the Aokie
/// `endpoint_thumbprint`.
pub fn thumbprint_of(public_key: &[u8; 32]) -> String {
    let jwk = format!("{{\"crv\":\"Ed25519\",\"kty\":\"OKP\",\"x\":\"{}\"}}", b64::encode(public_key));
    b64::encode(&sha256(jwk.as_bytes()))
}

/// An Ed25519 signature as base64url (86 characters).
pub fn signature_to_b64u(signature: &Signature) -> String {
    b64::encode(&signature.to_bytes())
}

/// An Ed25519 signature from its 86 characters of canonical base64url.
pub fn signature_from_b64u(text: &str) -> Result<Signature> {
    let bytes = b64::decode_exact::<64>(text)?;
    Ok(Signature::from_bytes(&bytes))
}

/// An Ed25519 public key that is in its canonical encoding and is not of small order: the only kind this crate holds. Every verification with it is strict.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct VerifyKey(VerifyingKey);

impl VerifyKey {
    /// From 32 bytes (`Error::Crypto(InvalidKey | NonCanonicalKey | SmallOrderKey)` otherwise).
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<VerifyKey> {
        Ok(VerifyKey(VerifyingKey::from_bytes(bytes)?))
    }

    /// From the 43 characters of canonical base64url.
    pub fn from_b64u(text: &str) -> Result<VerifyKey> {
        VerifyKey::from_bytes(&b64::decode_exact::<32>(text)?)
    }

    /// The 32 bytes.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    /// The 43 characters.
    pub fn to_b64u(&self) -> String {
        b64::encode(&self.0.to_bytes())
    }

    /// The thumbprint of this key.
    pub fn thumbprint(&self) -> String {
        thumbprint_of(&self.0.to_bytes())
    }

    /// Verifies `signature` over `domain || 0x00 || parts`, strictly.
    pub fn verify(&self, domain: SignDomain, parts: &[&[u8]], signature: &Signature) -> Result<()> {
        self.0.verify_raw(&domain.message(parts), signature).map_err(|_| Error::BadSignature(domain.as_str()))
    }

    /// [`VerifyKey::verify`] with the signature as its 86 characters of base64url.
    pub fn verify_b64u(&self, domain: SignDomain, parts: &[&[u8]], signature: &str) -> Result<()> {
        self.verify(domain, parts, &signature_from_b64u(signature)?)
    }

    /// Verifies a signature over a message that is already `domain || 0x00 || ...` (a JWS's signing input is not one: see [`crate::ticket`]).
    pub fn verify_raw(&self, message: &[u8], signature: &Signature) -> Result<()> {
        self.0.verify_raw(message, signature).map_err(|_| Error::BadSignature("raw"))
    }
}

impl core::fmt::Debug for VerifyKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "VerifyKey({})", self.thumbprint())
    }
}

/// An Ed25519 private key: signs under the domains of [`SignDomain`] and nothing else. Zeroized when dropped, never printed.
pub struct Signer(SigningKey);

impl Signer {
    /// From a 32-byte seed (RFC 8032; the enrolment secret's derived seed, the phone's and the desktop's endpoint seeds).
    pub fn from_seed(seed: &Secret<32>) -> Signer {
        Signer(SigningKey::from_seed(KeyRole::Hazmat, seed))
    }

    /// A fresh key from the operating system's random generator.
    pub fn generate() -> Result<Signer> {
        Ok(Signer(SigningKey::generate(KeyRole::Hazmat)?))
    }

    /// The public key.
    pub fn verify_key(&self) -> VerifyKey {
        VerifyKey(self.0.verifying_key())
    }

    /// The thumbprint of the public key.
    pub fn thumbprint(&self) -> String {
        self.verify_key().thumbprint()
    }

    /// The seed, in a container that wipes itself (for the keystore).
    pub fn seed(&self) -> Secret<32> {
        self.0.seed()
    }

    /// Signs `domain || 0x00 || parts`.
    pub fn sign(&self, domain: SignDomain, parts: &[&[u8]]) -> Signature {
        // A key of role Hazmat cannot be refused: the only error `sign_raw` has is a key with another role.
        self.0.sign_raw(&domain.message(parts)).unwrap_or_else(|_| Signature::from_bytes(&[0; 64]))
    }

    /// [`Signer::sign`] as the 86 characters of base64url.
    pub fn sign_b64u(&self, domain: SignDomain, parts: &[&[u8]]) -> String {
        signature_to_b64u(&self.sign(domain, parts))
    }
}

impl core::fmt::Debug for Signer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Signer({}, redacted)", self.thumbprint())
    }
}

/// An X25519 public key that is not of small order.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct X25519Public(DalekPublic);

impl X25519Public {
    /// From 32 bytes (`Error::Crypto(LowOrderPoint)` for the fourteen encodings whose shared secret would be all zero).
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<X25519Public> {
        Ok(X25519Public(DalekPublic::from_bytes(bytes)?))
    }

    /// From the 43 characters of canonical base64url.
    pub fn from_b64u(text: &str) -> Result<X25519Public> {
        X25519Public::from_bytes(&b64::decode_exact::<32>(text)?)
    }

    /// The 32 bytes.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    /// The 43 characters.
    pub fn to_b64u(&self) -> String {
        b64::encode(&self.0.to_bytes())
    }

    /// Seals `plaintext` to this key (`crypto_box_seal`).
    pub fn seal(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        Ok(sealbox::seal(&self.0, plaintext)?)
    }
}

impl core::fmt::Debug for X25519Public {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "X25519Public({})", self.to_b64u())
    }
}

/// An X25519 private key: opens sealed boxes. Zeroized when dropped, never printed.
pub struct X25519Secret(DalekSecret);

impl X25519Secret {
    /// From 32 bytes (clamped when used, as libsodium does).
    pub fn from_secret(secret: &Secret<32>) -> X25519Secret {
        X25519Secret(DalekSecret::from_secret(secret))
    }

    /// A fresh key from the operating system's random generator.
    pub fn generate() -> Result<X25519Secret> {
        Ok(X25519Secret(DalekSecret::generate()?))
    }

    /// The public key.
    pub fn public_key(&self) -> X25519Public {
        X25519Public(self.0.public_key())
    }

    /// The 32 bytes, in a container that wipes itself.
    pub fn to_secret(&self) -> Secret<32> {
        self.0.to_secret()
    }

    /// Opens a sealed box (`crypto_box_seal_open`). Any failure, an ephemeral key of small order and a box that is too short included, is the same
    /// `Error::Crypto(DecryptFailed)`.
    pub fn open_sealed(&self, sealed: &[u8]) -> Result<SecretVec> {
        Ok(sealbox::open(&self.0, sealed)?)
    }
}

impl core::fmt::Debug for X25519Secret {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("X25519Secret(redacted)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_domain_strings_are_the_readmes() {
        let all = [
            (SignDomain::Info, "oaiy/relay/1/info"),
            (SignDomain::InfoProof, "oaiy/relay/1/info-proof"),
            (SignDomain::Enroll, "oaiy/relay/1/enroll"),
            (SignDomain::Cmd, "oaiy/relay/1/cmd"),
            (SignDomain::Res, "oaiy/relay/1/res"),
            (SignDomain::Sync, "oaiy/relay/1/sync"),
            (SignDomain::ProviderRotate, "oaiy/relay/1/provider-rotate"),
            (SignDomain::Ring, "oaiy/relay/1/ring"),
            (SignDomain::PairingResponse, "oaiy/pairing/3/response"),
            (SignDomain::PairingApproval, "oaiy/pairing/3/approval"),
        ];
        for (d, s) in all {
            assert_eq!(d.as_str(), s);
        }
        assert_eq!(SignDomain::Cmd.message(&[b"ab", b"c"]), b"oaiy/relay/1/cmd\0abc");
    }

    #[test]
    fn a_signature_made_under_one_domain_does_not_verify_under_another() {
        let signer = Signer::from_seed(&Secret::new([7; 32]));
        let key = signer.verify_key();
        let sig = signer.sign(SignDomain::Cmd, &[b"payload"]);
        assert!(key.verify(SignDomain::Cmd, &[b"payload"], &sig).is_ok());
        assert!(key.verify(SignDomain::Res, &[b"payload"], &sig).is_err());
        assert!(key.verify(SignDomain::Cmd, &[b"payloaD"], &sig).is_err());
        // The domain and the text are joined by a zero byte, and nothing else: moving a byte from one to the other is a different message.
        assert_ne!(SignDomain::Cmd.message(&[b"x"]), domain_message("oaiy/relay/1/cmdx", &[b""]));
    }

    #[test]
    fn debug_never_prints_a_secret() {
        let signer = Signer::from_seed(&Secret::new([7; 32]));
        let x = X25519Secret::from_secret(&Secret::new([9; 32]));
        assert!(format!("{signer:?}").contains("redacted") && format!("{x:?}").contains("redacted"));
        assert!(!format!("{signer:?}").contains("0707"));
    }
}
