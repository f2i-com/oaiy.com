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

    /// `N` bytes from the operating system's random generator.
    pub fn random() -> Result<Self, Error> {
        let mut bytes = [0u8; N];
        crate::random::fill(&mut bytes)?;
        Ok(Secret(bytes))
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

/// How much of the stack below the caller is overwritten by [`scrub_stack`]: as much as a computation on a key can have used. The frames of a debug build are very large
/// (the BLAKE2b computation reaches about 80 KiB down; the computation needs that much stack itself, so scrubbing it asks for nothing new), those of an optimised build
/// a few hundred bytes.
const SCRUB_DEPTH: usize = if cfg!(debug_assertions) { 96 * 1024 } else { 4 * 1024 };

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
