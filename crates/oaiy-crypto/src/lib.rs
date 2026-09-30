//! oaiy-crypto: the cryptographic primitives of the OAIY vault (design `vault.final.md`, work package V-01).
//!
//! One crate for every key the vault, the backups, the relay and the apps handle, so that the same low-order checks, the same strict
//! signature verification, the same zeroization and the same constant-time comparison serve them all:
//!
//! - [`kdf`]: `crypto_kdf` with the registry of contexts, HKDF-SHA256, HMAC-SHA256, SHA-256
//! - [`aead`]: XChaCha20-Poly1305 and the 72-byte `wrap` format
//! - [`sealbox`]: libsodium's `crypto_box_seal`
//! - [`ed25519`]: strict Ed25519, keys with roles, signed strings built from a domain registry
//! - [`x25519`]: RFC 7748 with low-order refusal
//! - [`argon`]: Argon2id with the design's bounds, checked before anything is allocated
//! - [`bip39`]: the twelve-word phrase, checksum before any derivation
//! - [`kit`]: the FLRK1 recovery kit code of FormLogic
//! - [`canon`]: canonical AAD and signed strings, and the prefix-free domain registry
//! - [`zeroize`]: secret containers and `ct_eq`
//!
//! It is a library with no state: nothing is cached, nothing is global except the word list. It contains no `unsafe`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod aead;
pub mod argon;
pub mod bip39;
pub mod canon;
pub mod ed25519;
pub mod error;
pub mod kdf;
pub mod kit;
mod random;
pub mod sealbox;
mod text;
pub mod x25519;
pub mod zeroize;

pub use error::Error;
