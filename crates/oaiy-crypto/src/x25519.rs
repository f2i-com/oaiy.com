//! `x25519` (design 4.1.1): RFC 7748 Diffie-Hellman, with the checks the RFC leaves to the implementation.
//!
//! A public key of small order (the eight points of order dividing 8, and their non-canonical encodings, 14 byte strings once the
//! ignored top bit is counted: [`LOW_ORDER`]) makes the shared secret all zero whatever the private key, so
//! whoever chose that "public key" knows the "shared secret". libsodium refuses them; so does this module, twice: a
//! [`PublicKey`] cannot be built from one ([`PublicKey::from_bytes`], "refused at pin time", 4.1.3 rule 4), and
//! [`SecretKey::diffie_hellman`] refuses an all-zero result (`was_contributory`, the check RFC 7748 section 6.1 recommends).
//!
//! Secret keys are 32 bytes, clamped when used, exactly as libsodium's `crypto_scalarmult` and `crypto_box_keypair`: a key made
//! by FormLogic's JavaScript or PHP (`crypto_box_keypair`, or [`SecretKey::from_libsodium_seed`] for `crypto_box_seed_keypair`,
//! whose secret is the first 32 bytes of SHA-512 of the seed) gives the same public key here.

use core::fmt;

use sha2::{Digest, Sha512};
use x25519_dalek::{PublicKey as DalekPublic, StaticSecret};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::Error;
use crate::zeroize::{scrub_stack, Secret};

/// Key, public key and shared secret length.
pub const KEY_LEN: usize = 32;

/// The u-coordinates of low order (0, 1, the two of order 8, p-1, p and p+1), little-endian, top bit clear. X25519 ignores the top
/// bit of a u-coordinate, so each of these stands for two encodings.
pub const LOW_ORDER: [[u8; 32]; 7] = [
    hex32("0000000000000000000000000000000000000000000000000000000000000000"),
    hex32("0100000000000000000000000000000000000000000000000000000000000000"),
    hex32("e0eb7a7c3b41b8ae1656e3faf19fc46ada098deb9c32b1fd866205165f49b800"),
    hex32("5f9c95bca3508c24b1d0b1559c83ef5b04445cc4581c8e86d8224eddd09f1157"),
    hex32("ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
    hex32("edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
    hex32("eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
];

const fn hex32(text: &str) -> [u8; 32] {
    let bytes = text.as_bytes();
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = (nibble(bytes[2 * i]) << 4) | nibble(bytes[2 * i + 1]);
        i += 1;
    }
    out
}

const fn nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => panic!("hex digit"),
    }
}

/// True if `u` (with its ignored top bit) is one of the [`LOW_ORDER`] points.
pub fn is_low_order(u: &[u8; 32]) -> bool {
    let mut masked = *u;
    masked[31] &= 0x7f;
    LOW_ORDER.iter().fold(false, |found, low| found | (crate::zeroize::ct_eq(&masked, low)))
}

/// An X25519 public key that is not of small order.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PublicKey([u8; 32]);

impl PublicKey {
    /// Refuses a point of small order (`Error::LowOrderPoint`).
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<PublicKey, Error> {
        if is_low_order(bytes) {
            return Err(Error::LowOrderPoint);
        }
        Ok(PublicKey(*bytes))
    }

    /// The same from a slice of any length (`Error::InvalidLength` unless it is 32 bytes).
    pub fn from_slice(bytes: &[u8]) -> Result<PublicKey, Error> {
        let array: &[u8; 32] = bytes.try_into().map_err(|_| Error::InvalidLength("x25519 public key"))?;
        PublicKey::from_bytes(array)
    }

    /// The 32 bytes.
    pub const fn to_bytes(&self) -> [u8; 32] {
        self.0
    }

    /// The 32 bytes, borrowed.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x25519::PublicKey({})", crate::kdf::hex_lower(&self.0))
    }
}

/// An X25519 private key, zeroized when dropped.
pub struct SecretKey(StaticSecret);

impl SecretKey {
    /// From 32 bytes (any 32 bytes are a key; clamping happens when it is used).
    pub fn from_bytes(bytes: [u8; 32]) -> SecretKey {
        SecretKey(StaticSecret::from(bytes))
    }

    /// The same from a slice (`Error::InvalidLength` unless it is 32 bytes).
    pub fn from_slice(bytes: &[u8]) -> Result<SecretKey, Error> {
        let secret: Secret<32> = Secret::from_slice(bytes).map_err(|_| Error::InvalidLength("x25519 secret key"))?;
        Ok(SecretKey::from_bytes(*secret.expose()))
    }

    /// A fresh key from the operating system's random generator.
    pub fn generate() -> Result<SecretKey, Error> {
        let mut secret = Secret::<32>::zeroed();
        secret.fill_random()?;
        let key = SecretKey::from_secret(&secret);
        scrub_stack();
        Ok(key)
    }

    /// From a secret that the caller holds: the key is copied once, inside this function, and not through the caller's frame.
    #[inline(never)]
    pub fn from_secret(secret: &Secret<32>) -> SecretKey {
        SecretKey(StaticSecret::from(*secret.expose()))
    }

    /// libsodium's `crypto_box_seed_keypair`: the private key is the first 32 bytes of SHA-512 of the 32-byte seed.
    /// This is how FormLogic makes its ingestion and test recipient keys.
    pub fn from_libsodium_seed(seed: &Secret<32>) -> SecretKey {
        let mut digest = Sha512::digest(seed.expose());
        let mut key = [0u8; 32];
        key.copy_from_slice(&digest[..32]);
        digest.as_mut_slice().zeroize();
        let secret = SecretKey::from_bytes(key);
        key.zeroize();
        secret
    }

    /// The matching public key (`crypto_scalarmult_base`).
    pub fn public_key(&self) -> PublicKey {
        PublicKey(DalekPublic::from(&self.0).to_bytes())
    }

    /// The private key's 32 bytes, copied into a zeroizing container.
    pub fn to_secret(&self) -> Secret<32> {
        Secret::new(self.0.to_bytes())
    }

    /// X25519 with a peer's public key. An all-zero result is `Error::LowOrderPoint` and no secret is returned. The secret comes back **by value**, which leaves a copy in
    /// the frame that made it (see `kdf::derive`); [`SecretKey::diffie_hellman_into`] leaves none.
    pub fn diffie_hellman(&self, peer: &PublicKey) -> Result<Secret<32>, Error> {
        let mut out = Secret::zeroed();
        self.diffie_hellman_into(peer, &mut out)?;
        Ok(out)
    }

    /// [`SecretKey::diffie_hellman`], written into `out` in place: no copy of the shared secret is left in the stack below the caller. On an error `out` is not written.
    pub fn diffie_hellman_into(&self, peer: &PublicKey, out: &mut Secret<32>) -> Result<(), Error> {
        let result = self.dh_unscrubbed(peer, out);
        scrub_stack();
        result
    }

    #[inline(never)]
    fn dh_unscrubbed(&self, peer: &PublicKey, out: &mut Secret<32>) -> Result<(), Error> {
        let shared = self.0.diffie_hellman(&DalekPublic::from(peer.0));
        if !shared.was_contributory() {
            return Err(Error::LowOrderPoint);
        }
        out.expose_mut().copy_from_slice(shared.as_bytes());
        Ok(())
    }
}

impl ZeroizeOnDrop for SecretKey {}

impl fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("x25519::SecretKey(redacted)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The all-zero check of `diffie_hellman` on its own, with the key check out of the way: a peer key that the constructor would
    /// refuse, forced in, must still give no shared secret. (Two independent barriers; each has a test that does not need the other.)
    #[test]
    fn diffie_hellman_refuses_an_all_zero_result_even_for_a_key_that_bypassed_the_constructor() {
        let secret = SecretKey::from_bytes([0x42; 32]);
        for low in LOW_ORDER {
            for top in [0u8, 0x80] {
                let mut bytes = low;
                bytes[31] |= top;
                let forced = PublicKey(bytes);
                assert_eq!(secret.diffie_hellman(&forced).unwrap_err(), Error::LowOrderPoint, "{bytes:02x?}");
            }
        }
        // and an ordinary key is not refused
        let peer = SecretKey::from_bytes([0x17; 32]).public_key();
        assert!(secret.diffie_hellman(&peer).is_ok());
    }

    #[test]
    fn the_constructor_refuses_the_table_and_only_the_table() {
        for low in LOW_ORDER {
            assert_eq!(PublicKey::from_bytes(&low).unwrap_err(), Error::LowOrderPoint);
            let mut high = low;
            high[31] |= 0x80;
            assert_eq!(PublicKey::from_bytes(&high).unwrap_err(), Error::LowOrderPoint, "the top bit is ignored by X25519");
        }
        assert!(PublicKey::from_bytes(&[9; 32]).is_ok());
        let mut nine = [0u8; 32];
        nine[0] = 9;
        assert!(PublicKey::from_bytes(&nine).is_ok(), "the base point");
    }
}
