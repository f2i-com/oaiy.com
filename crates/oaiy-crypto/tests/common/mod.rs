//! Helpers shared by the integration tests: hex, strict base64, JSON, and the fixtures embedded at compile time so that the tests
//! need nothing outside the repository (no FormLogic checkout, no scratchpad, no network).
#![allow(dead_code)]

use oaiy_crypto::zeroize::Secret;

/// FormLogic's committed known-answer files, copied byte for byte (see README.md, "Test vectors").
pub const FL_VECTORS: &str = include_str!("../vectors/formlogic/e2ee-envelope-vectors.json");
pub const FL_SEALED_JS: &str = include_str!("../vectors/formlogic/e2ee-sealed-js.json");
pub const FL_SEALED_PHP: &str = include_str!("../vectors/formlogic/e2ee-sealed-php.json");
/// The design's vectors, computed by two implementations (`design/vault-work/vectors.json`).
pub const DESIGN_VECTORS: &str = include_str!("../vectors/vault-work/vectors.json");
/// RFC 5869, 8032, 7748, 9106, the XChaCha draft and the trezor BIP-39 vectors, extracted from the published texts.
pub const PUBLIC_VECTORS: &str = include_str!("../vectors/public-vectors.json");
/// libsodium's verdicts and known answers for the negative corpus and random-looking inputs.
pub const ORACLE: &str = include_str!("../vectors/libsodium-oracle.json");

pub fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).expect("fixture is valid JSON")
}

pub fn hex(bytes: &[u8]) -> String {
    oaiy_crypto::kdf::hex_lower(bytes)
}

pub fn unhex(text: &str) -> Vec<u8> {
    assert!(text.len().is_multiple_of(2), "odd hex length");
    (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("hex digit")).collect()
}

pub fn arr<const N: usize>(text: &str) -> [u8; N] {
    unhex(text).try_into().unwrap_or_else(|v: Vec<u8>| panic!("expected {N} bytes, got {}", v.len()))
}

pub fn secret<const N: usize>(text: &str) -> Secret<N> {
    Secret::new(arr::<N>(text))
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding (the encoding of every FormLogic wire field).
pub fn b64(bytes: &[u8]) -> String {
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8) | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(char::from(B64[(n >> 18) as usize & 63]));
        out.push(char::from(B64[(n >> 12) as usize & 63]));
        out.push(if chunk.len() > 1 { char::from(B64[(n >> 6) as usize & 63]) } else { '=' });
        out.push(if chunk.len() > 2 { char::from(B64[n as usize & 63]) } else { '=' });
    }
    out
}

/// Strict decode: padding required, canonical (the trailing bits are zero), no other characters. This is the server's
/// `requireBytes` rule.
pub fn unb64(text: &str) -> Vec<u8> {
    assert!(text.len().is_multiple_of(4), "base64 length");
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    for quad in bytes.chunks(4) {
        let mut n = 0u32;
        let mut pad = 0;
        for (i, c) in quad.iter().enumerate() {
            n <<= 6;
            if *c == b'=' {
                assert!(i >= 2, "padding position");
                pad += 1;
            } else {
                assert!(pad == 0, "data after padding");
                n |= B64.iter().position(|b| b == c).expect("base64 character") as u32;
            }
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    assert_eq!(b64(&out), text, "non-canonical base64");
    out
}

/// A deterministic byte generator (SplitMix64) so that every "random" corpus is the same on every run.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }

    pub fn array<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for b in out.iter_mut() {
            *b = self.next() as u8;
        }
        out
    }
}
