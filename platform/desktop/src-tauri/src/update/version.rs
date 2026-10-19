//! Which version is newer.
//!
//! One rule decides whether a release is ever offered, for both binaries and for the updater
//! plugin (which is given [`is_newer_version`] as its comparator, so the plugin's default
//! cannot differ): a release is an update only when its version is HIGHER than the running one.
//! A lower or equal version is never one, so a feed that names an older release (or the same
//! one) offers nothing, and a garbage version offers nothing either.
//!
//! Versions are semver: `0.10.0` is above `0.9.0` (numbers, not text), and a release a feed names
//! that is a pre-release is below its release (`0.1.0-beta.1` is below `0.1.0`) and ordered as semver
//! orders them (`alpha` < `alpha.1` < `beta` < `beta.2` < `beta.11` < `rc.1` < the release). Build
//! metadata (`+abc`) does not count: `0.1.0+7` is the same version as `0.1.0`. A leading `v` is
//! accepted, because tags carry one.
//!
//! The RUNNING version is the one exception to semver. A release is always `N.N.N` (the release
//! workflow refuses a tag with a suffix), so a running version with a pre-release part is never a
//! release: it is a build of one's own, made on top of the release it names (a Mac built here is
//! `0.1.9-mac.3`, from the code after 0.1.9). Semver puts `0.1.9-mac.3` below `0.1.9`, which would
//! offer it 0.1.9 and put the older code in place of the newer. So such a build is offered only a
//! release newer than its base ([`base_of_build`]): `0.1.10`, and not `0.1.9`. The rule is the same
//! on every platform, because it is about the version and not the install.

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

/// The release a running build with a pre-release part was made on top of: `0.1.9` for `0.1.9-mac.3`. None for a
/// release (`0.1.9`), which is its own base.
pub fn base_of_build(running: &Version) -> Option<Version> {
    (!running.pre.is_empty()).then(|| Version::new(running.major, running.minor, running.patch))
}

/// [`is_newer`] for versions already read (the updater plugin's comparator). A running build of one's own is held to its
/// base, so that the release it was made on top of is not newer than it (see the module's docs).
pub fn is_newer_version(current: &Version, candidate: &Version) -> bool {
    let floor = base_of_build(current).unwrap_or_else(|| precedence(current));
    precedence(candidate) > floor
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
    fn a_pre_release_in_the_feed_is_below_its_release() {
        // The running build is the release: the beta of it is older, not an update.
        assert!(!is_newer("0.1.0", "0.1.0-beta.1"));
        assert!(!is_newer("0.1.0", "0.1.0-rc.1"));
        // ...and a pre-release of a HIGHER version is above the lower release.
        assert!(is_newer("0.1.0", "0.2.0-beta.1"));
        assert!(!is_newer("0.2.0", "0.2.0-beta.1"));
    }

    #[test]
    fn pre_releases_in_the_feed_are_ordered_as_semver_orders_them() {
        // Against a running release, a candidate is newer exactly when semver puts it above.
        let chain = ["0.0.9", "0.1.0-alpha", "0.1.0-alpha.1", "0.1.0-alpha.beta", "0.1.0-beta", "0.1.0-beta.2", "0.1.0-beta.11", "0.1.0-rc.1", "0.1.0", "0.1.1-rc.1", "0.1.1"];
        for (i, running) in chain.iter().enumerate().filter(|(_, v)| !v.contains('-')) {
            for (j, candidate) in chain.iter().enumerate() {
                assert_eq!(is_newer(running, candidate), j > i, "{running} -> {candidate}");
            }
        }
    }

    #[test]
    fn a_running_build_of_ones_own_is_offered_only_a_release_newer_than_the_one_it_was_built_on() {
        // A Mac built here after 0.1.9 is 0.1.9-mac.3: 0.1.9 is older code than it, not an update, on any platform.
        for running in ["0.1.9-mac.3", "0.1.9-mac.1", "v0.1.9-mac.44", "0.1.9-beta.1", "0.1.9-mac.3+abc"] {
            assert!(!is_newer(running, "0.1.9"), "{running}");
            assert!(!is_newer(running, "0.1.8"), "{running}");
            assert!(!is_newer(running, "0.1.9-rc.1"), "{running}");
            assert!(!is_newer(running, "0.1.9-mac.9"), "{running}: another build of the same base is not a release either");
            assert!(is_newer(running, "0.1.10"), "{running}");
            assert!(is_newer(running, "0.2.0"), "{running}");
            assert!(is_newer(running, "0.1.10-rc.1"), "{running}: above the base, so above the build");
        }
        assert_eq!(base_of_build(&parse("0.1.9-mac.3").unwrap()), Some(Version::new(0, 1, 9)));
        assert_eq!(base_of_build(&parse("0.1.9+7").unwrap()), None, "build metadata is not a pre-release part");
        assert_eq!(base_of_build(&parse("0.1.9").unwrap()), None);
        // A release is held to semver exactly as before.
        assert!(is_newer("0.1.9", "0.1.10") && !is_newer("0.1.9", "0.1.9") && !is_newer("0.1.10", "0.1.9"));
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
