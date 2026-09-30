//! The one place this crate asks the operating system for random bytes (`getrandom`: BCryptGenRandom on Windows,
//! getrandom(2) on Linux, getentropy on macOS). Nothing here is seeded, cached or replaceable: a caller that needs
//! deterministic output for a known-answer test uses the `*_with_*` functions that take the randomness as an argument.

use crate::error::Error;

/// Fills `buf` from the operating system. A failure is reported, never papered over with weaker randomness.
pub(crate) fn fill(buf: &mut [u8]) -> Result<(), Error> {
    getrandom::fill(buf).map_err(|_| Error::Random)
}
