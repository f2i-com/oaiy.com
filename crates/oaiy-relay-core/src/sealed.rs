//! `sealed1`: the sealed token of pairing and the signed container of commands, results and `sync` items (README sections 10.1 and 10.3).
//!
//! **The sealed token.** `GET /v1/pair/{pid}` returns `sealedToken`, the base64url of a libsodium `crypto_box_seal` of the 63-character device token, addressed to the
//! phone's X25519 key (`ephemeral key (32) || tag (16) || ciphertext`). [`open_token`] reads it as `fixtures/sealed-token.json` says a reader must: strict base64url, at least
//! 48 bytes, an ephemeral key of small order refused **before** the box is opened (the Rust `crypto_box` crate's `unseal` opens such a box; `oaiy-crypto`'s `open` does not),
//! and only then the plaintext looked at: exactly a token, nothing else.
//!
//! **The container.** `{"k": <thumbprint of the signer's key>, "b": <b64u of the exact signed bytes>, "s": <b64u Ed25519 signature>[, "p": <padding>]}`, sealed to the
//! recipient and posted as the item's `body` (b64u of the box) with `hdr.ct = "sealed1"`. The signature is over `domain || 0x00 || bytes`, where `bytes` is the decoded `b`,
//! never its text; a verifier checks it over those bytes and only then parses them, and ignores `p`. No other member is allowed (`container.schema.json`).

use oaiy_crypto::ed25519::Signature;

use crate::b64;
use crate::error::{Error, Result};
use crate::ids::Token;
use crate::json;
use crate::keys::{signature_from_b64u, signature_to_b64u, SignDomain, Signer, VerifyKey, X25519Public, X25519Secret};

/// The largest sealed token a reader will decode (the real one is 111 bytes).
pub const MAX_SEALED_TOKEN_BYTES: usize = 512;

/// Opens a sealed token addressed to `recipient` and returns the device token it holds.
///
/// Every failure (not base64url, too short, an ephemeral key of small order, a box that does not authenticate, a box for another key, a plaintext that is not exactly a
/// token) is an error, and the box is opened only after those checks that can be made without it.
pub fn open_token(recipient: &X25519Secret, sealed_token: &str) -> Result<Token> {
    if sealed_token.len() > b64::encoded_len(MAX_SEALED_TOKEN_BYTES) {
        return Err(Error::Invalid("sealed token: too long"));
    }
    let sealed = b64::decode(sealed_token)?;
    if sealed.len() < oaiy_crypto::sealbox::SEAL_OVERHEAD {
        return Err(Error::Crypto(oaiy_crypto::Error::DecryptFailed));
    }
    let plaintext = recipient.open_sealed(&sealed)?;
    let text = core::str::from_utf8(plaintext.expose()).map_err(|_| Error::Invalid("sealed token: the plaintext is not a token"))?;
    Token::parse(text)
}

/// The domains a container is signed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerDomain {
    /// A command, signed by a provider.
    Cmd,
    /// A result, signed by the host identity.
    Res,
    /// A `sync` item, signed by the host identity.
    Sync,
}

impl ContainerDomain {
    fn sign_domain(self) -> SignDomain {
        match self {
            ContainerDomain::Cmd => SignDomain::Cmd,
            ContainerDomain::Res => SignDomain::Res,
            ContainerDomain::Sync => SignDomain::Sync,
        }
    }
}

/// A parsed container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Container {
    /// `k`: the thumbprint of the key the signer says it signed with. A verifier looks up a pinned key with it and verifies with that key, never with one that came with
    /// the container.
    pub signer: String,
    /// The decoded `b`: the exact signed bytes.
    pub bytes: Vec<u8>,
    /// `s`.
    pub signature: Signature,
}

impl Container {
    /// Parses a container's text: an object with `k`, `b` and `s` and, optionally, `p`, and no other member.
    pub fn parse(text: &[u8]) -> Result<Container> {
        let doc = json::parse(text)?;
        let members = doc.as_object().ok_or(Error::Invalid("container: not an object"))?;
        if members.iter().any(|(k, _)| !matches!(k.as_str(), "k" | "b" | "s" | "p")) {
            return Err(Error::Invalid("container: an unknown member"));
        }
        let signer = doc.get_str("k").filter(|k| crate::ids::is_thumbprint(k)).ok_or(Error::Invalid("container: k"))?;
        let bytes = b64::decode(doc.get_str("b").ok_or(Error::Invalid("container: b"))?)?;
        let signature = signature_from_b64u(doc.get_str("s").ok_or(Error::Invalid("container: s"))?)?;
        if let Some(p) = doc.get("p") {
            // Padding that verifiers ignore, but it is a string of the alphabet (`common#b64u`).
            let p = p.as_str().ok_or(Error::Invalid("container: p"))?;
            if p.is_empty() || !p.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
                return Err(Error::Invalid("container: p"));
            }
        }
        Ok(Container { signer: signer.to_string(), bytes, signature })
    }

    /// Verifies the signature over the decoded bytes (strictly, under `domain`) with `key`, which the caller took from its pins by [`Container::signer`] and not from
    /// the container, and returns the signed bytes. Only now may they be parsed.
    pub fn verify(&self, domain: ContainerDomain, key: &VerifyKey) -> Result<&[u8]> {
        if key.thumbprint() != self.signer {
            return Err(Error::Mismatch("container: k is not the thumbprint of the key given"));
        }
        key.verify(domain.sign_domain(), &[&self.bytes], &self.signature)?;
        Ok(&self.bytes)
    }
}

/// Builds the container text (`k`, `b`, `s`, then `p` when padding is asked for) that signs `bytes` under `domain`. With `pad` the text is made a multiple of 256
/// bytes long by `p`, which a signer SHOULD do so that a container's length says little about its content (README 10.3); `fill` supplies the padding characters.
pub fn build_container(signer: &Signer, domain: ContainerDomain, bytes: &[u8], pad: bool, mut fill: impl FnMut(&mut [u8])) -> String {
    let signature = signer.sign(domain.sign_domain(), &[bytes]);
    let mut text = format!("{{\"k\":\"{}\",\"b\":\"{}\",\"s\":\"{}\"", signer.thumbprint(), b64::encode(bytes), signature_to_b64u(&signature));
    if pad {
        // Closed as it is, the text is `text` and `}`: one byte more. With padding it is `text`, `,"p":"`, the characters and `"}`: eight bytes besides the characters, of which
        // there is at least one, so that p is never empty.
        let n = match (256 - (text.len() + 8) % 256) % 256 {
            0 => 256,
            n => n,
        };
        if (text.len() + 1) % 256 != 0 {
            let mut raw = vec![0u8; n];
            fill(&mut raw);
            const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
            let chars: String = raw.iter().map(|b| char::from(ALPHABET[usize::from(b & 63)])).collect();
            text.push_str(",\"p\":\"");
            text.push_str(&chars);
            text.push('"');
        }
    }
    text.push('}');
    text
}

/// Seals container text to `recipient` and returns the item `body`: the base64url of the box.
pub fn seal_container(recipient: &X25519Public, container_text: &str) -> Result<String> {
    Ok(b64::encode(&recipient.seal(container_text.as_bytes())?))
}

/// Opens an item `body` (`hdr.ct` is `sealed1`) addressed to `recipient` and parses the container. The caller then picks the pinned key by `container.signer` and calls
/// [`Container::verify`]; nothing in the container is an authority before that.
pub fn open_container(recipient: &X25519Secret, body: &str) -> Result<Container> {
    let sealed = b64::decode(body)?;
    let plaintext = recipient.open_sealed(&sealed)?;
    Container::parse(plaintext.expose())
}
