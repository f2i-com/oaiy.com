//! The access model.
//!
//! One model for everything that reaches the local API: every non-public route is classified into a
//! scope (or one of a few small special classes) in [`routes`], and a route with no classification is
//! refused. See the route table's own documentation for the classes.
//!
//! - [`token`]: the token grammar, the hash at rest, the CSRF value, strict parsing of a bearer.
//! - [`scopes`], [`presets`]: the 54 exact scopes and the named bundles.
//! - [`store`], [`chain`], [`lock`], [`principal`]: credentials in memory and on disk, the rule that a
//!   credential is valid only while its whole parent chain is, the one-process lock, and who a
//!   request is.
//! - [`audit`], [`scrub`]: the audit and noise logs, and the scrub every log line passes through.
//! - [`guard`], [`mode`], [`host`], [`clientip`], [`exposure_checks`], [`bearer_throttle`], [`cors`]:
//!   the request pipeline, and the access mode that decides which requests it judges.
//! - [`api`], [`runtime`]: the routes that exist (`/api/auth/info`, `whoami`, `derive`) and the assembly of
//!   a guard for a running server.

pub mod api;
pub mod audit;
pub mod bearer_throttle;
pub mod chain;
pub mod clientip;
pub mod clock;
#[cfg(feature = "web")]
pub mod cookie;
pub mod cors;
#[cfg(feature = "web")]
pub mod device;
pub mod export;
pub mod exposure_checks;
pub mod guard;
pub mod host;
pub mod lock;
pub mod mode;
#[cfg(feature = "web")]
pub mod owner;
pub mod presets;
pub mod principal;
pub mod routes;
pub mod runtime;
pub mod scopes;
pub mod scrub;
pub mod store;
pub mod token;
#[cfg(feature = "web")]
pub mod password;
#[cfg(feature = "web")]
pub mod policy;
#[cfg(feature = "web")]
pub mod throttle;
#[cfg(feature = "web")]
pub mod wordlist;

pub use guard::{Guard, GuardConfig, HealthExtras};
pub use mode::{AccessMode, ConfigRefusal, Exposure};
pub use runtime::{build_guard, flush_installed, AccessSettings};

#[cfg(test)]
mod conformance;
#[cfg(test)]
mod guard_tests;
#[cfg(test)]
pub(crate) mod route_coverage;
#[cfg(test)]
mod store_tests;
