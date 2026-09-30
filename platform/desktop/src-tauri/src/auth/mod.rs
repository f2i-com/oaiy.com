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
pub mod audit;
pub mod clock;
pub mod export;
pub mod presets;
pub mod principal;
pub mod routes;
pub mod scopes;
pub mod scrub;
pub mod token;

#[cfg(test)]
mod route_coverage;
