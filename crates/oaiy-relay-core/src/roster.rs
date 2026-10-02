//! The roster hash (README 10.6, vector A0) and the rules of a roster (README 5.4, "Roster"; Interpretation 16).
//!
//! `peerRosterHash = b64u(SHA-256("aokie/v2/peer-roster" || 0x00 || canonical({"approvedPeerKeyThumbprints":[sorted],"peerRosterRevision":N})))`: the Aokie construction, and
//! vector A0 checks it against the Aokie README's own example. A roster is the list of phones' key thumbprints a desktop approves: 1 to 16 of them in an admission (a push
//! may be empty), strictly ascending bytewise, none equal to the desktop's own endpoint thumbprint.

use oaiy_crypto::kdf::sha256;

use crate::b64;
use crate::error::{Error, Result};
use crate::ids;
use crate::json::{Json, MAX_SAFE_INT};
use crate::keys::domain_message;

/// The domain of the roster hash.
pub const DOMAIN: &str = "aokie/v2/peer-roster";
/// The most phones a relay lets one desktop and app have (`limits.rosterMax`, never raised past 16).
pub const MAX_PHONES: usize = 16;

/// The roster hash of `thumbprints` at `revision`. The thumbprints are sorted here (bytewise) as the canonical form of the Aokie construction has them; nothing is
/// validated, because the Aokie README's own example uses text that is not a thumbprint (use [`check`] for the rules of a roster).
pub fn hash(revision: u64, thumbprints: &[String]) -> Result<String> {
    let mut sorted: Vec<&String> = thumbprints.iter().collect();
    sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    let text = Json::obj([
        ("approvedPeerKeyThumbprints", Json::Arr(sorted.into_iter().map(|t| Json::str(t.clone())).collect())),
        ("peerRosterRevision", Json::int(revision)),
    ])
    .to_canonical()?;
    Ok(b64::encode(&sha256(&domain_message(DOMAIN, &[text.as_bytes()]))))
}

/// The rules of a roster pushed to the relay or named in an admission: every entry a thumbprint, strictly ascending bytewise, none equal to `own_thumbprint`, at most 16, and a
/// revision of at most 2^53 - 1.
pub fn check(own_thumbprint: &str, revision: u64, thumbprints: &[String]) -> Result<()> {
    if revision > MAX_SAFE_INT {
        return Err(Error::Invalid("roster: revision"));
    }
    if thumbprints.len() > MAX_PHONES {
        return Err(Error::Invalid("roster: more than 16"));
    }
    for t in thumbprints {
        if !ids::is_thumbprint(t) || t == own_thumbprint {
            return Err(Error::Invalid("roster: an entry"));
        }
    }
    if thumbprints.windows(2).any(|w| w[0].as_bytes() >= w[1].as_bytes()) {
        return Err(Error::Invalid("roster: not strictly ascending"));
    }
    Ok(())
}
