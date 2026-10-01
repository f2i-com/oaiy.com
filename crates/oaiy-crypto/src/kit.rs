//! The FLRK1 recovery kit code (FormLogic's `vault.ts`, `encodeRecoveryKey` and `decodeRecoveryKey`): the way a 256-bit recovery key is
//! written on paper, and the key that wraps the UMK (`flrecov1`).
//!
//! The code is `FLRK1-` and 52 unpadded RFC 4648 Base32 characters of the key in groups of four, then a group of four characters that
//! are the first 20 bits of SHA-256 of the key: `FLRK1-AAAA-...-MZUH` for 32 zero bytes. Decoding ignores case, white space and
//! hyphens, and checks the checksum before anything else uses the key. **White space is JavaScript's `\s`** (review L-8), the set FormLogic's
//! `display.trim().toUpperCase().replace(/[\s-]+/g, "")` strips and the one the phrase decoder splits on (`text::is_js_space`): it used to be ASCII only, and refused
//! a code pasted with a no-break space, an ideographic space, a vertical tab or a byte-order mark, which FormLogic accepts. Three things are stricter than the
//! JavaScript, on purpose, and each is a class of entries in `tests/vectors/text-corpus.json` (computed by a port of FormLogic's decoder in Node): only ASCII letters
//! are upper-cased (the JavaScript maps the dotless i, the long s and some ligatures onto `A` to `Z`); the four unused bits of the 52nd character must be zero (the
//! JavaScript ignores them, so two spellings of one key decode there and only one here); and the input is at most [`MAX_INPUT_BYTES`] bytes. So no code that
//! this decoder accepts is refused by the JavaScript, and FormLogic should tighten its decoder to the same (see the crate README).

use zeroize::{Zeroize, Zeroizing};

use crate::error::Error;
use crate::kdf::{self, sha256, Purpose};
use crate::text::is_js_space;
use crate::zeroize::{scrub_stack, Secret, SecretString};

/// The longest text `decode` looks at, in bytes: a code is 75 characters, and the rest is room for the white space a person types. Longer is `KitFormat`. (FormLogic's
/// decoder has no cap; this is one of the places where this decoder is stricter, and `tests/vectors/text-corpus.json` has the boundary: 256 bytes accepted, 257 not.)
pub const MAX_INPUT_BYTES: usize = 256;

const PREFIX: &str = "FLRK1";
const ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

/// The 256-bit recovery key of a kit.
pub struct RecoveryKit(Secret<32>);

fn base32_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity((bytes.len() * 8).div_ceil(5));
    let mut buffer: u32 = 0;
    let mut bits = 0u32;
    for byte in bytes {
        buffer = (buffer << 8) | u32::from(*byte);
        bits += 8;
        while bits >= 5 {
            out.push(char::from(ALPHABET[((buffer >> (bits - 5)) & 31) as usize]));
            bits -= 5;
        }
        buffer &= (1 << bits) - 1;
    }
    if bits > 0 {
        out.push(char::from(ALPHABET[((buffer << (5 - bits)) & 31) as usize]));
    }
    out
}

fn checksum(key: &[u8; 32]) -> String {
    // The first 20 bits of the digest are the first four characters of its Base32.
    let digest = sha256(key);
    base32_encode(&digest[..3])[..4].to_string()
}

impl RecoveryKit {
    /// From the 32 bytes.
    pub const fn from_bytes(key: Secret<32>) -> RecoveryKit {
        RecoveryKit(key)
    }

    /// A fresh key from the operating system's random generator.
    pub fn generate() -> Result<RecoveryKit, Error> {
        let mut key = Secret::<32>::zeroed();
        key.fill_random()?;
        scrub_stack();
        Ok(RecoveryKit(key))
    }

    /// The key.
    pub const fn key(&self) -> &Secret<32> {
        &self.0
    }

    /// The code as printed: `FLRK1-XXXX-...-XXXX` (14 groups after the prefix). The stack below the caller is overwritten when it returns (the key passed through the frames of the
    /// Base32 and of the checksum, and one copy was left in the dead stack in an optimised build: third review, L-1).
    pub fn encode(&self) -> SecretString {
        let text = self.encode_unscrubbed();
        scrub_stack();
        text
    }

    #[inline(never)]
    fn encode_unscrubbed(&self) -> SecretString {
        let body = Zeroizing::new(base32_encode(self.0.expose()));
        // 5 + 14 x 5 characters; reserved up front so that the text never moves to a bigger block and leaves the old one unwiped (review M-4)
        let mut text = String::with_capacity(96);
        text.push_str(PREFIX);
        for group in body.as_bytes().chunks(4) {
            text.push('-');
            text.push_str(core::str::from_utf8(group).unwrap_or(""));
        }
        text.push('-');
        text.push_str(&checksum(self.0.expose()));
        SecretString::new(text)
    }

    /// Reads a code. `Error::KitFormat` for a wrong prefix, length, character or trailing bit; `Error::KitChecksum` when the
    /// checksum does not match the key (typing errors show here, before any derivation).
    ///
    /// The stack below the caller is overwritten when it returns (third review, L-1): the Base32 was unpacked into the key in the frames below this one, and two copies of the key
    /// were left in the dead stack in an optimised build. The kit itself comes back by value, which leaves what any by-value return leaves (see `kdf::derive`).
    pub fn decode(display: &str) -> Result<RecoveryKit, Error> {
        let result = RecoveryKit::decode_unscrubbed(display);
        scrub_stack();
        result
    }

    #[inline(never)]
    fn decode_unscrubbed(display: &str) -> Result<RecoveryKit, Error> {
        if display.len() > MAX_INPUT_BYTES {
            return Err(Error::KitFormat);
        }
        // never longer than the input, so it never grows (review M-4)
        let mut cleaned: Zeroizing<String> = Zeroizing::new(String::with_capacity(display.len()));
        for c in display.chars().filter(|c| !is_js_space(*c) && *c != '-') {
            cleaned.push(c.to_ascii_uppercase());
        }
        let rest = cleaned.strip_prefix(PREFIX).ok_or(Error::KitFormat)?;
        if rest.len() != 56 || !rest.bytes().all(|b| ALPHABET.contains(&b)) {
            return Err(Error::KitFormat);
        }
        let (body, given) = rest.split_at(52);
        let mut key = [0u8; 32];
        let mut buffer: u32 = 0;
        let mut bits = 0u32;
        let mut written = 0usize;
        for ch in body.bytes() {
            let value = ALPHABET.iter().position(|a| *a == ch).unwrap_or(0) as u32;
            buffer = (buffer << 5) | value;
            bits += 5;
            if bits >= 8 {
                key[written] = ((buffer >> (bits - 8)) & 0xff) as u8;
                written += 1;
                bits -= 8;
                buffer &= (1 << bits) - 1;
            }
        }
        // 52 characters carry 260 bits: 32 bytes and four bits that must be zero.
        if written != 32 || buffer != 0 {
            key.zeroize();
            return Err(Error::KitFormat);
        }
        let ok = crate::zeroize::ct_eq(checksum(&key).as_bytes(), given.as_bytes());
        cleaned.zeroize();
        if !ok {
            key.zeroize();
            return Err(Error::KitChecksum);
        }
        let kit = RecoveryKit(Secret::new(key));
        key.zeroize();
        Ok(kit)
    }

    /// `kdf(1, "flrecov1", key)`: the key that wraps the UMK.
    pub fn wrap_key(&self) -> Result<Secret<32>, Error> {
        kdf::derive(&self.0, Purpose::KitWrap)
    }

    /// [`RecoveryKit::wrap_key`], written into `out` in place: no copy of the wrap key is left in the stack below the caller (see `kdf::derive_into`).
    pub fn wrap_key_into(&self, out: &mut Secret<32>) -> Result<(), Error> {
        kdf::derive_into(&self.0, Purpose::KitWrap, out)
    }
}

impl zeroize::ZeroizeOnDrop for RecoveryKit {}

impl core::fmt::Debug for RecoveryKit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("RecoveryKit(redacted)")
    }
}
