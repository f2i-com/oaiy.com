//! OAIY Desktop plugin host.
//!
//! A plugin is a directory under `<data_dir>/plugins/<id>/` containing a
//! `manifest.json` and an executable. It runs as a supervised child process
//! speaking **JSON-RPC 2.0 over newline-delimited stdio**.
//!
//! ```text
//!   manifest  — parse + validate; resolve the capability surface
//!   rpc       — the wire: framing, correlation, caps, non-protocol output
//!   registry  — scan the plugins root, hold state + reasons
//!   runner    — spawn, handshake, health, bounded restart, shutdown
//! ```
//!
//! Contract: `protocol/README.md` (capabilities, events, connector commands) and
//! `protocol/v1/{event-envelope,connector-request,connector-response}.schema.json`.
//! The reference plugin is Aokie (`aokie.com/crates/aokie-plugin`), whose manifest
//! this host reads unmodified apart from branding.
//!
//! # Design commitments
//!
//! **A plugin is untrusted code the user chose to run.** It gets a declared,
//! expanded capability list and nothing else. Three rules follow, and each is
//! enforced in this module rather than left to the plugin's good behaviour:
//!
//! - An **undeclared connector command is rejected before the plugin sees it**.
//!   The manifest is the allow-list, checked at the gateway.
//! - An **undeclared event is dropped**, so a plugin cannot invent event names to
//!   reach triggers it was never granted.
//! - **Wildcards are expanded at load time**, never matched per-call — otherwise
//!   a grant a user reviewed silently widens with every plugin update.
//!
//! **Failure is never silent.** An invalid manifest, an unsupported API version
//! or a crashed process all surface as a state *with a reason attached*. A plugin
//! that quietly fails to appear is indistinguishable from one that was never
//! installed, and the user's next move — reinstalling — does not help.

pub mod manifest;
pub mod rpc;

pub use manifest::{ManifestError, PluginManifest, HOST_CAPABILITIES};
pub use rpc::{RpcError, RpcMessage, RpcPeer, MAX_LINE_BYTES};
