//! Shared plain-value types.

/// Crate version string, semver.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Expert cache eviction policy.
///
/// Both are implemented; there is no third. LFRU is frequency-first with a
/// recency tiebreak and is the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CachePolicy {
    /// Frequency-first, recency tiebreak (default).
    #[default]
    Lfru,
    /// Plain least-recently-used.
    Lru,
}
