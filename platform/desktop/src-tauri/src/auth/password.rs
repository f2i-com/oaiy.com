//! The owner's password: Argon2id, as a PHC string (design 4.7.2).
//!
//! Argon2id, version 0x13, `m = 65536` KiB (64 MiB), `t = 3`, `p = 1`, a 16-byte random salt and a 32-byte
//! output, stored as `$argon2id$v=19$m=65536,t=3,p=1$<salt>$<hash>`. That is the second recommended profile of
//! RFC 9106 with one lane (a small VPS has one core, so more lanes would only run one after another), and
//! heavier than the OWASP minimum (19 MiB, `t = 2`).
//!
//! Verification takes a stored string from a file that a person can edit, so it is bounded before any memory is
//! taken: only Argon2id at version 0x13, `m` from 19,456 to 262,144 KiB, `t` from 2 to 10, `p` from 1 to 4 and
//! a 16 to 64 byte output. A string outside that, or one that is not a PHC string at all, is not a reason to
//! answer faster than for a wrong password: it is verified against a hash of the engine's own, and the answer is
//! the same mismatch. After a successful verification the caller re-hashes at the current cost when the stored
//! one is lower.
//!
//! The work is behind [`PasswordEngine`] so that the rules around it (the lanes, the concurrency bound, the
//! answers that must not differ) are tested by counting calls instead of by waiting for 64 MiB passes.
//!
//! What this module never does: log a password, print one, keep one after the call, or compare a hash any other
//! way than the constant-time comparison `password-hash` does.

use std::sync::{Arc, OnceLock};

use argon2::password_hash::{PasswordHash, PasswordVerifier};
use argon2::{Algorithm, Argon2, Params, Version};
use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine as _;

use super::store::Random;
use super::token::MintError;

/// The cost this build hashes at.
pub const MEMORY_KIB: u32 = 65_536;
pub const PASSES: u32 = 3;
pub const LANES: u32 = 1;
pub const OUTPUT_LEN: usize = 32;
pub const SALT_LEN: usize = 16;

/// What a stored hash may ask of the verifier.
pub const MEMORY_KIB_RANGE: std::ops::RangeInclusive<u32> = 19_456..=262_144;
pub const PASSES_RANGE: std::ops::RangeInclusive<u32> = 2..=10;
pub const LANES_RANGE: std::ops::RangeInclusive<u32> = 1..=4;
pub const OUTPUT_LEN_RANGE: std::ops::RangeInclusive<usize> = 16..=64;

/// The parameters of one hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    pub memory_kib: u32,
    pub passes: u32,
    pub lanes: u32,
}

impl Cost {
    /// What a hash made now is made with.
    pub const CURRENT: Cost = Cost {
        memory_kib: MEMORY_KIB,
        passes: PASSES,
        lanes: LANES,
    };

    /// The cheapest a verifier accepts (OWASP's minimum): for the tests, which hash a great deal.
    pub const CHEAPEST: Cost = Cost {
        memory_kib: 19_456,
        passes: 2,
        lanes: 1,
    };

    fn within_bounds(self) -> bool {
        MEMORY_KIB_RANGE.contains(&self.memory_kib)
            && PASSES_RANGE.contains(&self.passes)
            && LANES_RANGE.contains(&self.lanes)
    }

    /// Whether a hash made at `self` should be made again at `now`: a lower memory or fewer passes. More lanes
    /// are not weaker (and the current is one).
    pub fn is_lower_than(self, now: Cost) -> bool {
        self.memory_kib < now.memory_kib || self.passes < now.passes
    }
}

/// What verifying a password against a stored hash found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The password is right. `rehash`: the stored hash is cheaper than the current cost and should be replaced.
    Match { rehash: bool },
    /// The password is wrong, or the stored hash cannot be used (it was verified against a hash of ours all the
    /// same, so the two cost the same).
    Mismatch,
}

/// Why a password could not be hashed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HashError {
    /// The operating system gave no randomness for the salt.
    NoRandomness,
    /// The Argon2 implementation refused (it does not for a cost inside the bounds).
    Argon2(String),
}

impl std::fmt::Display for HashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HashError::NoRandomness => {
                write!(f, "the operating system gave no randomness for a salt")
            }
            HashError::Argon2(e) => write!(f, "the password could not be hashed: {e}"),
        }
    }
}

impl std::error::Error for HashError {}

impl From<MintError> for HashError {
    fn from(_: MintError) -> Self {
        HashError::NoRandomness
    }
}

/// Hashing and verifying, the two operations that take 64 MiB and a moment.
pub trait PasswordEngine: Send + Sync {
    /// A PHC string for `password` (already normalised, as bytes) with a fresh salt at the current cost.
    fn hash(&self, password: &[u8]) -> Result<String, HashError>;

    /// Verify `password` against `stored`. Never fails: a stored string that cannot be used is a mismatch that
    /// took as long as a wrong password.
    fn verify(&self, password: &[u8], stored: &str) -> Verdict;
}

/// The Argon2id implementation.
pub struct Argon2Engine {
    cost: Cost,
    random: Random,
    /// A hash of the engine's own, verified against when the stored one is unusable.
    dummy: OnceLock<String>,
}

impl Argon2Engine {
    /// The cost of this build, salts from the operating system.
    pub fn production() -> Argon2Engine {
        Argon2Engine::with(Cost::CURRENT, Arc::new(super::token::os_random))
    }

    /// Another cost (inside the bounds) and another source of salts: the tests.
    pub fn with(cost: Cost, random: Random) -> Argon2Engine {
        debug_assert!(
            cost.within_bounds(),
            "{cost:?} is outside what verification accepts"
        );
        Argon2Engine {
            cost,
            random,
            dummy: OnceLock::new(),
        }
    }

    pub fn cost(&self) -> Cost {
        self.cost
    }

    fn argon(cost: Cost, output_len: usize) -> Result<Argon2<'static>, HashError> {
        let params = Params::new(cost.memory_kib, cost.passes, cost.lanes, Some(output_len))
            .map_err(|e| HashError::Argon2(e.to_string()))?;
        Ok(Argon2::new(Algorithm::Argon2id, Version::V0x13, params))
    }

    /// The PHC string for `password` and this exact salt. The salt is random in [`PasswordEngine::hash`]; this
    /// exists so that the design's known-answer vector can be reproduced.
    pub fn hash_with_salt(
        &self,
        password: &[u8],
        salt: &[u8; SALT_LEN],
    ) -> Result<String, HashError> {
        let mut out = [0u8; OUTPUT_LEN];
        Self::argon(self.cost, OUTPUT_LEN)?
            .hash_password_into(password, salt, &mut out)
            .map_err(|e| HashError::Argon2(e.to_string()))?;
        Ok(format!(
            "$argon2id$v=19$m={},t={},p={}${}${}",
            self.cost.memory_kib,
            self.cost.passes,
            self.cost.lanes,
            STANDARD_NO_PAD.encode(salt),
            STANDARD_NO_PAD.encode(out)
        ))
    }

    /// The hash verified against when the stored one is unusable.
    fn dummy(&self) -> &str {
        self.dummy.get_or_init(|| {
            // Any password does: nobody knows the salt's password, and the answer is discarded.
            let mut salt = [0u8; SALT_LEN];
            let _ = (self.random)(&mut salt);
            self.hash_with_salt(b"oaiy dummy password", &salt)
                .unwrap_or_default()
        })
    }

    /// Verify against a string the bounds have passed. `None` when it cannot be parsed after all.
    fn verify_parsed(&self, password: &[u8], stored: &str) -> Option<Verdict> {
        let parsed = PasswordHash::new(stored).ok()?;
        if parsed.algorithm.as_str() != "argon2id" {
            return None;
        }
        if parsed.version != Some(0x13) {
            return None;
        }
        let params = Params::try_from(&parsed).ok()?;
        let cost = Cost {
            memory_kib: params.m_cost(),
            passes: params.t_cost(),
            lanes: params.p_cost(),
        };
        let output_len = parsed.hash.map(|h| h.len())?;
        if !cost.within_bounds() || !OUTPUT_LEN_RANGE.contains(&output_len) || salt_len(&parsed) < 8
        {
            return None;
        }
        // `Argon2::default()` verifies with the parameters the string names; they are inside the bounds.
        match Argon2::default().verify_password(password, &parsed) {
            Ok(()) => Some(Verdict::Match {
                rehash: cost.is_lower_than(Cost::CURRENT),
            }),
            Err(_) => Some(Verdict::Mismatch),
        }
    }
}

impl PasswordEngine for Argon2Engine {
    fn hash(&self, password: &[u8]) -> Result<String, HashError> {
        let mut salt = [0u8; SALT_LEN];
        (self.random)(&mut salt)?;
        self.hash_with_salt(password, &salt)
    }

    fn verify(&self, password: &[u8], stored: &str) -> Verdict {
        match self.verify_parsed(password, stored) {
            Some(v) => v,
            None => {
                // The stored string cannot be used: the work of a verification is done all the same.
                let _ = self.verify_parsed(password, self.dummy());
                Verdict::Mismatch
            }
        }
    }
}

/// Whether a stored string is one this build could verify (the bounds and the shape), without verifying.
pub fn is_usable(stored: &str) -> bool {
    let Ok(parsed) = PasswordHash::new(stored) else {
        return false;
    };
    let Ok(params) = Params::try_from(&parsed) else {
        return false;
    };
    let cost = Cost {
        memory_kib: params.m_cost(),
        passes: params.t_cost(),
        lanes: params.p_cost(),
    };
    parsed.algorithm.as_str() == "argon2id"
        && parsed.version == Some(0x13)
        && cost.within_bounds()
        && parsed
            .hash
            .is_some_and(|h| OUTPUT_LEN_RANGE.contains(&h.len()))
        && salt_len(&parsed) >= 8
}

/// The salt's length in bytes (the string holds it in base64).
fn salt_len(parsed: &PasswordHash<'_>) -> usize {
    parsed
        .salt
        .and_then(|s| STANDARD_NO_PAD.decode(s.as_str()).ok())
        .map_or(0, |bytes| bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use argon2::{AssociatedData, ParamsBuilder};

    /// The design's vector (4.7.2), computed with Node's Argon2id after it reproduced RFC 9106.
    const PASSWORD: &str = "correct horse battery staple";
    const SALT: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    const PHC: &str = "$argon2id$v=19$m=65536,t=3,p=1$AAECAwQFBgcICQoLDA0ODw$DRo8ZSPI8G5OCvnFFapbVEjP69aDjy1Sw9i2743cPC4";
    const TAG_HEX: &str = "0d1a3c6523c8f06e4e0af9c515aa5b5448cfebd6838f2d52c3d8b6ef8ddc3c2e";

    fn hex(bytes: &[u8]) -> String {
        crate::auth::token::to_hex(bytes)
    }

    fn engine() -> Argon2Engine {
        Argon2Engine::production()
    }

    /// A cheap engine for the tests that hash many times.
    fn cheap() -> Argon2Engine {
        Argon2Engine::with(Cost::CHEAPEST, Arc::new(crate::auth::token::os_random))
    }

    #[test]
    fn the_known_answer_phc_string_of_the_design_is_reproduced_exactly() {
        let made = engine().hash_with_salt(PASSWORD.as_bytes(), &SALT).unwrap();
        assert_eq!(made, PHC);
        // The tag inside it is the hex of the vector.
        let parsed = PasswordHash::new(&made).unwrap();
        assert_eq!(hex(parsed.hash.unwrap().as_bytes()), TAG_HEX);
        assert_eq!(parsed.salt.unwrap().as_str(), "AAECAwQFBgcICQoLDA0ODw");
    }

    #[test]
    fn the_stored_vector_verifies_and_only_for_its_password() {
        let e = engine();
        assert_eq!(
            e.verify(PASSWORD.as_bytes(), PHC),
            Verdict::Match { rehash: false }
        );
        for wrong in [
            "correct horse battery stapleX",
            "Correct horse battery staple",
            "",
            "correct horse battery",
        ] {
            assert_eq!(
                e.verify(wrong.as_bytes(), PHC),
                Verdict::Mismatch,
                "{wrong:?}"
            );
        }
    }

    /// RFC 9106 section 5.3, the Argon2id test vector: it has a secret and associated data, which a login
    /// never uses, so it is the check that this crate's Argon2id is Argon2id.
    #[test]
    fn rfc_9106_section_5_3_argon2id_vector() {
        let password = [0x01u8; 32];
        let salt = [0x02u8; 16];
        let secret = [0x03u8; 8];
        let data = [0x04u8; 12];
        let params = ParamsBuilder::new()
            .m_cost(32)
            .t_cost(3)
            .p_cost(4)
            .data(AssociatedData::new(&data).unwrap())
            .output_len(32)
            .build()
            .unwrap();
        let argon =
            Argon2::new_with_secret(&secret, Algorithm::Argon2id, Version::V0x13, params).unwrap();
        let mut tag = [0u8; 32];
        argon
            .hash_password_into(&password, &salt, &mut tag)
            .unwrap();
        assert_eq!(
            hex(&tag),
            "0d640df58d78766c08c037a34a8b53c9d01ef0452d75b65eb52520e96b01e659"
        );
    }

    #[test]
    fn a_hash_is_made_with_a_fresh_salt_at_the_current_cost_and_verifies() {
        let e = cheap();
        let a = e.hash(b"a password of some length").unwrap();
        let b = e.hash(b"a password of some length").unwrap();
        assert_ne!(a, b, "two salts");
        assert!(a.starts_with("$argon2id$v=19$m=19456,t=2,p=1$"), "{a}");
        assert_eq!(
            e.verify(b"a password of some length", &a),
            Verdict::Match { rehash: true },
            "cheaper than the build's cost: to be made again"
        );
        assert_eq!(e.verify(b"another password", &a), Verdict::Mismatch);
        // The production engine's cost is the vector's.
        assert_eq!(
            Cost::CURRENT,
            Cost {
                memory_kib: 65_536,
                passes: 3,
                lanes: 1
            }
        );
        assert_eq!(engine().cost(), Cost::CURRENT);
    }

    #[test]
    fn a_failure_of_the_random_source_is_an_error_and_never_a_fixed_salt() {
        let e = Argon2Engine::with(
            Cost::CHEAPEST,
            Arc::new(|_: &mut [u8]| Err(MintError::NoRandomness)),
        );
        assert_eq!(e.hash(b"whatever password"), Err(HashError::NoRandomness));
    }

    fn phc(m: u32, t: u32, p: u32) -> String {
        let salt = [7u8; 16];
        let params = Params::new(m, t, p, Some(32)).unwrap();
        let mut out = [0u8; 32];
        Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
            .hash_password_into(b"a password of some length", &salt, &mut out)
            .unwrap();
        format!(
            "$argon2id$v=19$m={m},t={t},p={p}${}${}",
            STANDARD_NO_PAD.encode(salt),
            STANDARD_NO_PAD.encode(out)
        )
    }

    #[test]
    fn the_bounds_on_a_stored_hash_are_exact_at_both_ends() {
        let e = cheap();
        // Inside, at each edge: accepted.
        for (m, t, p) in [
            (19_456, 2, 1),
            (19_456, 10, 1),
            (19_456, 2, 4),
            (65_536, 3, 1),
        ] {
            let s = phc(m, t, p);
            assert!(is_usable(&s), "{m},{t},{p}");
            assert!(
                matches!(
                    e.verify(b"a password of some length", &s),
                    Verdict::Match { .. }
                ),
                "{m},{t},{p}"
            );
        }
        // One step outside, on each side: refused before any memory is taken (m = 262,145 is not hashed).
        let too_big = "$argon2id$v=19$m=262145,t=3,p=1$AAECAwQFBgcICQoLDA0ODw$DRo8ZSPI8G5OCvnFFapbVEjP69aDjy1Sw9i2743cPC4";
        for s in [
            phc(19_455, 3, 1),
            phc(19_456, 1, 1),
            too_big.to_string(),
            "$argon2id$v=19$m=65536,t=11,p=1$AAECAwQFBgcICQoLDA0ODw$DRo8ZSPI8G5OCvnFFapbVEjP69aDjy1Sw9i2743cPC4".to_string(),
            "$argon2id$v=19$m=65536,t=3,p=5$AAECAwQFBgcICQoLDA0ODw$DRo8ZSPI8G5OCvnFFapbVEjP69aDjy1Sw9i2743cPC4".to_string(),
            "$argon2id$v=19$m=4294967295,t=3,p=1$AAECAwQFBgcICQoLDA0ODw$DRo8ZSPI8G5OCvnFFapbVEjP69aDjy1Sw9i2743cPC4".to_string(),
        ] {
            assert!(!is_usable(&s), "{s}");
            assert_eq!(e.verify(b"a password of some length", &s), Verdict::Mismatch, "{s}");
        }
    }

    #[test]
    fn only_argon2id_at_version_0x13_is_verified() {
        let e = cheap();
        let good = phc(19_456, 2, 1);
        assert!(is_usable(&good));
        for bad in [
            good.replace("argon2id", "argon2i"),
            good.replace("argon2id", "argon2d"),
            good.replace("v=19", "v=16"),
            good.replace("v=19$", ""),
            good.replace("$m=", "$M="),
            "".to_string(),
            "not a hash".to_string(),
            "$".to_string(),
            "$argon2id$".to_string(),
            format!("{good}$extra"),
            // A hash of 8 bytes and one of 65 bytes are outside 16 to 64.
            format!(
                "$argon2id$v=19$m=19456,t=2,p=1$AAECAwQFBgcICQoLDA0ODw${}",
                STANDARD_NO_PAD.encode([1u8; 8])
            ),
            format!(
                "$argon2id$v=19$m=19456,t=2,p=1$AAECAwQFBgcICQoLDA0ODw${}",
                STANDARD_NO_PAD.encode([1u8; 65])
            ),
            // A salt of 4 bytes.
            format!(
                "$argon2id$v=19$m=19456,t=2,p=1${}${}",
                STANDARD_NO_PAD.encode([1u8; 4]),
                STANDARD_NO_PAD.encode([1u8; 32])
            ),
        ] {
            assert!(!is_usable(&bad), "{bad}");
            assert_eq!(
                e.verify(b"a password of some length", &bad),
                Verdict::Mismatch,
                "{bad}"
            );
        }
    }

    #[test]
    fn a_hash_at_a_lower_cost_says_so_and_one_at_a_higher_cost_does_not() {
        let e = cheap();
        assert_eq!(
            e.verify(b"a password of some length", &phc(19_456, 3, 1)),
            Verdict::Match { rehash: true },
            "less memory"
        );
        assert_eq!(
            e.verify(b"a password of some length", &phc(65_536, 2, 1)),
            Verdict::Match { rehash: true },
            "fewer passes"
        );
        assert_eq!(
            e.verify(b"a password of some length", &phc(65_536, 3, 2)),
            Verdict::Match { rehash: false },
            "more lanes is not weaker"
        );
        assert_eq!(
            e.verify(b"a password of some length", &phc(131_072, 4, 1)),
            Verdict::Match { rehash: false },
            "dearer"
        );
        assert!(Cost::CHEAPEST.is_lower_than(Cost::CURRENT));
        assert!(!Cost::CURRENT.is_lower_than(Cost::CURRENT));
    }

    #[test]
    fn an_unusable_stored_hash_costs_a_verification_all_the_same() {
        // The counting stand-in for the Argon2 pass: what a verify does inside is one pass whatever the
        // stored string is. Here the real engine is watched from outside by its own dummy: after an unusable
        // string the dummy hash exists (it was verified against), and after a usable one it may not.
        let e = cheap();
        assert!(e.dummy.get().is_none());
        assert_eq!(e.verify(b"whatever password", "garbage"), Verdict::Mismatch);
        assert!(
            e.dummy
                .get()
                .is_some_and(|d| d.starts_with("$argon2id$v=19$m=19456,t=2,p=1$")),
            "the unusable string was verified against a hash of the engine's own"
        );
    }
}
