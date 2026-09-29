//! Which version is newer.
//!
//! One rule decides whether a release is ever offered, for both binaries and for the updater
//! plugin (which is given [`is_newer_version`] as its comparator, so the plugin's default
//! cannot differ): a release is an update only when its version is HIGHER than the running one.
//! A lower or equal version is never one, so a feed that names an older release (or the same
//! one) offers nothing, and a garbage version offers nothing either.
//!
//! Versions are semver: `0.10.0` is above `0.9.0` (numbers, not text), and a pre-release is
//! below its release (`0.1.0-beta.1` is below `0.1.0`) and ordered as semver orders them
//! (`alpha` < `alpha.1` < `beta` < `beta.2` < `beta.11` < `rc.1` < the release). Build metadata
//! (`+abc`) does not count: `0.1.0+7` is the same version as `0.1.0`. A leading `v` is
//! accepted, because tags carry one.

use semver::Version;

/// A version as written in a feed, a tag or a manifest (`0.2.0`, `v0.2.0`, `0.2.0-rc.1`).
pub fn parse(text: &str) -> Result<Version, String> {
    let trimmed = text.trim();
    let bare = trimmed.strip_prefix('v').unwrap_or(trimmed);
    Version::parse(bare).map_err(|e| format!("\"{trimmed}\" is not a version ({e})"))
}

/// The version with its build metadata dropped: the part precedence is decided by.
fn precedence(version: &Version) -> Version {
    Version { build: semver::BuildMetadata::EMPTY, ..version.clone() }
}

/// Whether `candidate` is a higher version than `current`. False when equal, lower, or either is not a version.
pub fn is_newer(current: &str, candidate: &str) -> bool {
    match (parse(current), parse(candidate)) {
        (Ok(current), Ok(candidate)) => is_newer_version(&current, &candidate),
        _ => false,
    }
}

/// [`is_newer`] for versions already read (the updater plugin's comparator).
pub fn is_newer_version(current: &Version, candidate: &Version) -> bool {
    precedence(candidate) > precedence(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_higher_version_is_an_update() {
        assert!(is_newer("0.1.0", "0.1.1"));
        assert!(is_newer("0.1.0", "0.2.0"));
        assert!(is_newer("0.9.9", "1.0.0"));
        assert!(is_newer("v0.1.0", "v0.1.1"));
    }

    #[test]
    fn numbers_are_compared_as_numbers_not_as_text() {
        assert!(is_newer("0.9.0", "0.10.0"));
        assert!(!is_newer("0.10.0", "0.9.0"));
        assert!(is_newer("1.2.9", "1.2.10"));
        assert!(!is_newer("1.2.10", "1.2.9"));
        assert!(is_newer("9.0.0", "10.0.0"));
    }

    #[test]
    fn a_lower_version_is_never_an_update() {
        assert!(!is_newer("0.1.1", "0.1.0"));
        assert!(!is_newer("1.0.0", "0.9.9"));
        assert!(!is_newer("0.2.0", "0.1.9"));
    }

    #[test]
    fn an_equal_version_is_never_an_update() {
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("v0.1.0", "0.1.0"));
        assert!(!is_newer("0.1.0-beta.1", "0.1.0-beta.1"));
        // Build metadata is not part of the precedence: the same version.
        assert!(!is_newer("0.1.0", "0.1.0+7"));
        assert!(!is_newer("0.1.0+8", "0.1.0+7"));
        assert!(!is_newer("0.1.0+7", "0.1.0+8"));
    }

    #[test]
    fn a_pre_release_is_below_its_release() {
        // The running build is the release: the beta of it is older, not an update.
        assert!(!is_newer("0.1.0", "0.1.0-beta.1"));
        assert!(!is_newer("0.1.0", "0.1.0-rc.1"));
        // A running beta is updated to its release.
        assert!(is_newer("0.1.0-beta.1", "0.1.0"));
        // ...and a pre-release of a HIGHER version is above the lower release.
        assert!(is_newer("0.1.0", "0.2.0-beta.1"));
        assert!(!is_newer("0.2.0", "0.2.0-beta.1"));
    }

    #[test]
    fn pre_releases_are_ordered_as_semver_orders_them() {
        let chain = ["0.1.0-alpha", "0.1.0-alpha.1", "0.1.0-alpha.beta", "0.1.0-beta", "0.1.0-beta.2", "0.1.0-beta.11", "0.1.0-rc.1", "0.1.0"];
        for (i, lower) in chain.iter().enumerate() {
            for (j, higher) in chain.iter().enumerate() {
                assert_eq!(is_newer(lower, higher), j > i, "{lower} -> {higher}");
            }
        }
    }

    #[test]
    fn garbage_is_never_an_update_in_either_place() {
        for bad in ["", "   ", "latest", "1", "1.2", "1.2.3.4", "one.two.three", "0.01.0", "0.1.0-", "🙂", "0.1.0 beta"] {
            assert!(!is_newer("0.1.0", bad), "candidate {bad:?}");
            assert!(!is_newer(bad, "0.2.0"), "current {bad:?}");
        }
        assert!(!is_newer("", ""));
    }

    #[test]
    fn a_version_is_read_with_or_without_its_v_and_spaces() {
        assert_eq!(parse("v1.2.3").unwrap(), parse(" 1.2.3 ").unwrap());
        assert!(parse("V1.2.3").is_err());
        assert!(parse("vv1.2.3").is_err());
    }
}
