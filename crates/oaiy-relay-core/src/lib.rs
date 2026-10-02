//! oaiy-relay-core: the OAIY relay client, once, for the desktop and the phone (design `relay.final.md` 4.16.1 and package DK-03; `mobile.final.md` packages MOB-08,
//! MOB-21a and MOB-22a).
//!
//! The relay protocol (`platform/protocol/relay/v1/README.md`) is read here and nowhere else: the desktop's relay client and the phone's both depend on this crate, so
//! there is one reading of a rule and one set of tests for it. The design asked for it ("two implementations of one protocol are the largest risk of divergence").
//! Nothing here knows Tauri, JNI, a runtime, a TLS stack or a file system layout: the platform supplies a few small traits (an HTTP client, a clock, secret and profile
//! stores, a random source) and this crate is the rest.
//!
//! | Layer | Modules | Pure? |
//! |---|---|---|
//! | Wire primitives | [`b64`], [`json`], [`ids`], [`url`] | yes |
//! | Keys and signed objects | [`keys`], [`info`], [`enrol`], [`sealed`], [`ring`], [`ticket`] | yes |
//! | Pairing v3 | [`pairing`] (the math, both parties' state machines) | yes: the parties are driven by the caller |
//! | The poll loop's rules (README 5.1.1, P1 to P9) | [`poll`] | yes: one deterministic function |
//! | I/O behind traits | [`client`] (HTTP, clock, stores, the client, the loop) | the traits are the seam |
//! | Test support (feature `testing`) | [`testing`] (an in-process stub relay, a scripted transport, a fake clock) | |
//!
//! It contains no `unsafe`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod admission;
pub mod b64;
pub mod enrol;
pub mod error;
pub mod ids;
pub mod info;
pub mod json;
pub mod keys;
pub mod pairing;
pub mod ring;
pub mod roster;
pub mod rotation;
pub mod sealed;
pub mod ticket;
pub mod url;

pub use error::{Error, Result};
