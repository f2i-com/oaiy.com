//! Secret types, zeroization and constant-time comparison.
//!
//! Every secret of this crate lives in one of these types (or in a dalek type that zeroizes on drop and is wrapped
//! by one of ours). They have no `Clone`, no `Copy`, no `Display`, and a `Debug` that prints nothing of the value, so a
//! secret cannot reach a log by accident, and they overwrite their bytes when dropped. Comparison of secrets is
//! constant-time (`==` on a [`Secret`] or a [`SecretVec`] is `subtle`'s `ct_eq`), and [`ct_eq`] is the one function the
//! rest of the workspace should use to compare two byte strings when either is a key, a tag or a signature.
//!
//! What this does not do (and no Rust library can): stop the compiler from leaving a copy of a value in a register or a
//! dead stack slot, or stop the operating system from paging memory out. It removes the copies it owns.

use core::fmt;

use subtle::ConstantTimeEq;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::error::Error;

/// `N` secret bytes, zeroized when dropped.
pub struct Secret<const N: usize>([u8; N]);

impl<const N: usize> Secret<N> {
    /// Takes ownership of the bytes.
    pub const fn new(bytes: [u8; N]) -> Self {
        Secret(bytes)
    }

    /// Copies exactly `N` bytes out of a slice, refusing any other length.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, Error> {
        let array: [u8; N] = bytes.try_into().map_err(|_| Error::InvalidLength("secret"))?;
        Ok(Secret(array))
    }

    /// `N` zero bytes: a place for a secret to be written **in place** (see [`Secret::expose_mut`]). A secret that a function returns by value is copied at every move between
    /// the frame that made it and the frame that holds it, and the copies stay in stack that has been given back; a caller that makes the `Secret` itself and hands the function
    /// `&mut` of it (the `*_into` functions) gets the value written once, where it will live.
    pub const fn zeroed() -> Self {
        Secret([0u8; N])
    }

    /// The bytes, writable: for the `*_into` functions of this crate, which fill a secret in place. Write to it and never copy out of it.
    pub fn expose_mut(&mut self) -> &mut [u8; N] {
        &mut self.0
    }

    /// Fills the bytes from the operating system's random generator, in place (the random bytes are written once, where they will live).
    pub fn fill_random(&mut self) -> Result<(), Error> {
        crate::random::fill(&mut self.0)
    }

    /// `N` bytes from the operating system's random generator.
    pub fn random() -> Result<Self, Error> {
        let mut secret = Secret::zeroed();
        secret.fill_random()?;
        Ok(secret)
    }

    /// The bytes, for use as a key or a message. Keep the borrow short; never copy them into a type without zeroization.
    pub const fn expose(&self) -> &[u8; N] {
        &self.0
    }

    /// An explicit second copy (there is no `Clone`, so that a copy is always a decision that can be searched for).
    pub fn duplicate(&self) -> Self {
        Secret(self.0)
    }

    /// Overwrites the bytes now. `Drop` does the same.
    pub fn wipe(&mut self) {
        self.0.zeroize();
    }
}

impl<const N: usize> Drop for Secret<N> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl<const N: usize> ZeroizeOnDrop for Secret<N> {}

impl<const N: usize> fmt::Debug for Secret<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Secret<{N}>(redacted)")
    }
}

impl<const N: usize> PartialEq for Secret<N> {
    fn eq(&self, other: &Self) -> bool {
        self.0.ct_eq(&other.0).into()
    }
}

impl<const N: usize> Eq for Secret<N> {}

/// Secret bytes of a length that varies (a decrypted message, a value in the keystore), zeroized when dropped, capacity
/// included.
pub struct SecretVec(Zeroizing<Vec<u8>>);

impl SecretVec {
    /// Takes ownership of the bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        SecretVec(Zeroizing::new(bytes))
    }

    /// Takes over a buffer that is already zeroizing.
    pub fn from_zeroizing(bytes: Zeroizing<Vec<u8>>) -> Self {
        SecretVec(bytes)
    }

    /// The bytes.
    pub fn expose(&self) -> &[u8] {
        &self.0
    }

    /// The number of bytes (a length is not a secret).
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True for zero bytes.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Hands the buffer to a caller that wants `Zeroizing<Vec<u8>>` (the keystore's `get` does).
    pub fn into_zeroizing(self) -> Zeroizing<Vec<u8>> {
        self.0
    }
}

impl ZeroizeOnDrop for SecretVec {}

impl fmt::Debug for SecretVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretVec(len {}, redacted)", self.0.len())
    }
}

impl PartialEq for SecretVec {
    fn eq(&self, other: &Self) -> bool {
        ct_eq(&self.0, &other.0)
    }
}

impl Eq for SecretVec {}

/// Constant-time equality of two byte strings: the time depends on the length, never on where the first difference is.
/// Strings of different lengths are unequal (a length is not a secret).
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.ct_eq(b).into()
}

/// How much of the stack below the caller is overwritten by [`scrub_stack`]: **one depth, in every build configuration** (review low 1: it was keyed on `debug_assertions`, which is
/// not what decides how deep a computation's frames are, and a build with assertions off and no optimisation left a copy of the master key that the depth of a debug build would
/// have reached). The frames of a computation are largest with no optimisation (the BLAKE2b computation reaches about 80 KiB down; measured: 8 KiB and 64 KiB left a copy, 88 KiB left
/// none) and a few hundred bytes with it; 96 KiB covers every configuration that was measured (dev, release at opt-level 0, 1, 2, 3, s and z, with and without assertions), and what
/// it costs is one `memset` of 96 KiB per derivation, and stack that the computation needed anyway. A thread with less than about 200 KiB of stack left is not safe to derive on.
const SCRUB_DEPTH: usize = 96 * 1024;

/// Overwrites the stack below its caller with zeros. A function that has worked on a key leaves, in the stack space it has given back, the copies that moves made
/// (Rust cannot name them and `zeroize` cannot reach them: a state that is moved into a function is copied, and dropping wipes one of the places). Calling this, from
/// the same frame that called the function, right after it returns, puts its own frame where the frames of the computation were and overwrites them as a whole.
/// It is `#[inline(never)]` so that its frame is a frame of its own; the dead-stack probe in `tests/zeroize_stack.rs` is what shows that it reaches the copies.
#[inline(never)]
pub(crate) fn scrub_stack() {
    let mut pad = [0u8; SCRUB_DEPTH];
    core::hint::black_box(&mut pad);
}

/// A secret `String` (a recovery phrase, a kit code) that overwrites itself when dropped, and prints nothing.
pub struct SecretString(Zeroizing<String>);

impl SecretString {
    /// Takes ownership of the text.
    pub fn new(text: String) -> Self {
        SecretString(Zeroizing::new(text))
    }

    /// The text. Show it to the owner; never log it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl ZeroizeOnDrop for SecretString {}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString(redacted)")
    }
}

impl PartialEq for SecretString {
    fn eq(&self, other: &Self) -> bool {
        ct_eq(self.0.as_bytes(), other.0.as_bytes())
    }
}

impl Eq for SecretString {}
