//! Windows DPAPI, the only `unsafe` of this crate (design 4.5.1, provider `windows-dpapi-file`).
//!
//! Two functions over `CryptProtectData` and `CryptUnprotectData`, and nothing else: user scope (never `CRYPTPROTECT_LOCAL_MACHINE`),
//! `CRYPTPROTECT_UI_FORBIDDEN` (a service or a headless session never waits on a dialog), the secret's name bound in as optional entropy. The
//! plaintext DPAPI hands back lives in memory the system allocated with `LocalAlloc`; it is copied into a zeroizing buffer, the system's copy
//! is overwritten, and only then is it freed. Every block below says why it is sound.

use core::ffi::c_void;
use core::ptr;

use windows_sys::Win32::Foundation::{GetLastError, LocalFree};
use windows_sys::Win32::Security::Cryptography::{CryptProtectData, CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB};
use zeroize::{Zeroize, Zeroizing};

use crate::error::KeyError;

/// The most this module is ever asked to protect (the keystore caps a value at 64 KiB, far below `u32::MAX`).
const MAX_INPUT: usize = 1 << 20;

/// Protects `plain` for the current user, with `entropy` bound in. Returns DPAPI's opaque blob.
pub(crate) fn protect(plain: &[u8], entropy: &[u8; 32]) -> Result<Vec<u8>, KeyError> {
    if plain.is_empty() || plain.len() > MAX_INPUT {
        return Err(KeyError::InvalidValue("length"));
    }
    let input = CRYPT_INTEGER_BLOB { cbData: plain.len() as u32, pbData: plain.as_ptr().cast_mut() };
    let entropy_blob = CRYPT_INTEGER_BLOB { cbData: entropy.len() as u32, pbData: entropy.as_ptr().cast_mut() };
    let mut out = CRYPT_INTEGER_BLOB { cbData: 0, pbData: ptr::null_mut() };
    // SAFETY: `input` and `entropy_blob` describe memory that outlives the call (`plain` and `entropy` are borrowed for it) and that DPAPI only reads
    // (the `cast_mut` is for the C signature, which takes a non-const pointer in a struct it does not write through). `out` is a valid, writable blob
    // header. The description, reserved and prompt arguments are null, which the API documents as "none". The flags are the constant for "no UI".
    let ok = unsafe { CryptProtectData(&input, ptr::null(), &entropy_blob, ptr::null(), ptr::null(), CRYPTPROTECT_UI_FORBIDDEN, &mut out) };
    if ok == 0 {
        // SAFETY: GetLastError reads the calling thread's last-error value and has no other requirement.
        return Err(KeyError::Os("CryptProtectData", unsafe { GetLastError() }));
    }
    if out.pbData.is_null() {
        return Err(KeyError::Os("CryptProtectData", 0));
    }
    // SAFETY: on success DPAPI set `out` to a `LocalAlloc`ed block of `cbData` bytes that nothing else owns; it is copied out and then freed exactly
    // once. It holds ciphertext, which needs no wiping.
    let blob = unsafe { core::slice::from_raw_parts(out.pbData, out.cbData as usize) }.to_vec();
    unsafe { LocalFree(out.pbData.cast::<c_void>()) };
    Ok(blob)
}

/// Unprotects a DPAPI blob for the current user with the same `entropy`. The plaintext is returned in a zeroizing buffer.
pub(crate) fn unprotect(blob: &[u8], entropy: &[u8; 32]) -> Result<Zeroizing<Vec<u8>>, KeyError> {
    if blob.is_empty() || blob.len() > MAX_INPUT {
        return Err(KeyError::Corrupt("DPAPI blob length"));
    }
    let input = CRYPT_INTEGER_BLOB { cbData: blob.len() as u32, pbData: blob.as_ptr().cast_mut() };
    let entropy_blob = CRYPT_INTEGER_BLOB { cbData: entropy.len() as u32, pbData: entropy.as_ptr().cast_mut() };
    let mut out = CRYPT_INTEGER_BLOB { cbData: 0, pbData: ptr::null_mut() };
    // SAFETY: as in `protect`; the description out-pointer is null, so DPAPI allocates no description string that would need freeing.
    let ok = unsafe { CryptUnprotectData(&input, ptr::null_mut(), &entropy_blob, ptr::null(), ptr::null(), CRYPTPROTECT_UI_FORBIDDEN, &mut out) };
    if ok == 0 {
        // SAFETY: as above.
        return Err(KeyError::Os("CryptUnprotectData", unsafe { GetLastError() }));
    }
    if out.pbData.is_null() {
        return Err(KeyError::Os("CryptUnprotectData", 0));
    }
    // SAFETY: on success `out` is a `LocalAlloc`ed block of `cbData` bytes that this function now owns. It is copied into a zeroizing buffer, then
    // overwritten through a mutable slice over the same block (volatile writes, so the overwrite is not optimised away) and freed exactly once.
    let plain = unsafe {
        let block = core::slice::from_raw_parts_mut(out.pbData, out.cbData as usize);
        let copy = Zeroizing::new(block.to_vec());
        block.zeroize();
        LocalFree(out.pbData.cast::<c_void>());
        copy
    };
    Ok(plain)
}
