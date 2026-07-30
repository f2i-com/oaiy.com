//! OAIY Bridge Protocol v1 — the desktop runtime.
//!
//! Contract: `protocol/README.md` + `protocol/v1/*.schema.json`. That document is
//! normative; this module implements it. Where they disagree, the schema wins.
//!
//! # Why there is no flow engine in here
//!
//! FormLogic Desktop reimplemented a subset of the flow engine in Rust —
//! `flows/runner.rs` (3.5k lines) plus a QuickJS sandbox — because its desktop
//! could not run the real engine. That reimplementation then had to be kept in
//! lock-step with the browser executor by a parity test, and the two drifted in
//! exactly the way you would expect: a node's declared input handles disagreeing
//! with what its compiler actually read, so flows reported success with wrong
//! output.
//!
//! OAIY does not have that problem, because OAIY *owns* the engine. `cli/` runs
//! flows headlessly through the same `oaiy-core` the browser uses, over a Node
//! host adapter. So this runtime **drives that engine as a sidecar** instead of
//! growing a second one:
//!
//! ```text
//!   consumer ──HTTP──► bridge (this module)
//!                        ├── ledger   : reserve / claim / finalise   [Rust]
//!                        ├── triggers : event -> binding -> run      [Rust]
//!                        └── runner   : spawn `oaiy run`             [Rust]
//!                                          └──► oaiy-core            [TS, shared]
//! ```
//!
//! Desktop and browser runs are then identical *by construction* rather than by
//! a test asserting two implementations still agree. Rust keeps the jobs it is
//! actually better at: process supervision, the durable ledger, and HTTP.
//!
//! The cost is a Node sidecar on the box. That is already true — the CLI ships
//! in the same repo and the desktop already supervises far heavier things
//! (Python venvs, llama.cpp, a Playwright browser).

pub mod ledger;

pub use ledger::{
    ClaimOutcome, Ledger, LedgerHandle, LineageRef, ReserveOutcome, RunError, RunErrorCode,
    RunRecord, RunRequest, RunStatus, Runtime,
};
