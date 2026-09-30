//! The known-device cookie (design 4.7.9): how the owner's usual browsers reach the reserved lane.
//!
//! A successful password login from a browser that does not present a valid `dev` cookie sets one: a
//! `oaiydev_<id>_<secret>` token whose SHA-256 goes into `owner.json`'s `devices` (at most 16: a seventeenth
//! drops the oldest). Later logins from that browser present it and are routed to the known-device lane of 4.7.4:
//! they skip the per-address and global counters and the anonymous queue, use the reserved verification slot, and
//! have a counter of their own, so an attacker who fills the anonymous lane cannot keep the owner out, and a
//! stolen cookie is worth 35 password guesses a day and nothing more.
//!
//! The cookie is never a credential: it does not authenticate a request, it only chooses which lane a password
//! is tried in. Losing a laptop is answered by revoking the device (the Security card,
//! `DELETE /api/auth/sessions/:id`, or `auth sessions revoke-all --devices`); a password change revokes all of them.
//!
//! A cookie is checked in constant time, and an unknown id costs the same comparison as a wrong secret, so the
//! answer does not say which of the two it was (and either way the login is treated as one from a browser with
//! no device: the anonymous lane).

use super::owner::Device;
use super::token::{self, Kind, MintError};

/// Devices kept: the oldest is dropped for the seventeenth.
pub const MAX_DEVICES: usize = 16;

/// A hash no device has, compared against when the id is unknown.
const DUMMY_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// The device a cookie value names, if it is valid.
pub fn verify<'a>(devices: &'a [Device], cookie_value: &str) -> Option<&'a Device> {
    let parsed = token::parse(cookie_value)?;
    if parsed.kind != Kind::Dev {
        return None;
    }
    let device = devices.iter().find(|d| d.id == parsed.id);
    let expected = device.map_or(DUMMY_HASH, |d| d.hash.as_str());
    let equal = token::hashes_equal(&token::secret_hash(parsed.secret), expected);
    device.filter(|_| equal)
}

/// A device made now: its record, and the cookie value (shown once, never stored).
pub struct NewDevice {
    pub device: Device,
    pub token: String,
}

impl std::fmt::Debug for NewDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewDevice")
            .field("device", &self.device)
            .field("token", &"[redacted]")
            .finish()
    }
}

/// Make a device for `ip` from the random source `fill`.
pub fn make(
    now_ms: u64,
    ip: &str,
    fill: &mut dyn FnMut(&mut [u8]) -> Result<(), MintError>,
) -> Result<NewDevice, MintError> {
    let minted = token::mint_with(Kind::Dev, fill)?;
    Ok(NewDevice {
        device: Device {
            id: minted.id,
            hash: minted.hash,
            created_ms: now_ms,
            last_ms: now_ms,
            ip: ip.chars().take(64).collect(),
            extra: serde_json::Map::new(),
        },
        token: minted.token,
    })
}

/// Add `new` to `devices`, dropping the oldest while there are more than [`MAX_DEVICES`]. The ids dropped.
pub fn add(devices: &mut Vec<Device>, new: Device) -> Vec<String> {
    devices.push(new);
    let mut dropped = Vec::new();
    while devices.len() > MAX_DEVICES {
        let oldest = devices
            .iter()
            .enumerate()
            .min_by_key(|(_, d)| (d.created_ms, d.last_ms))
            .map(|(i, _)| i);
        match oldest {
            Some(i) => dropped.push(devices.remove(i).id),
            None => break,
        }
    }
    dropped
}

/// Remove a device. Whether there was one.
pub fn remove(devices: &mut Vec<Device>, id: &str) -> bool {
    let before = devices.len();
    devices.retain(|d| d.id != id);
    devices.len() != before
}

/// Note that a device was used now, from `ip`.
pub fn touch(devices: &mut [Device], id: &str, now_ms: u64, ip: &str) {
    if let Some(d) = devices.iter_mut().find(|d| d.id == id) {
        d.last_ms = now_ms.max(d.created_ms);
        d.ip = ip.chars().take(64).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_from(seed: u8) -> impl FnMut(&mut [u8]) -> Result<(), MintError> {
        let mut n = seed;
        move |buf: &mut [u8]| {
            for b in buf.iter_mut() {
                n = n.wrapping_mul(31).wrapping_add(7);
                *b = n;
            }
            Ok(())
        }
    }

    fn made(now: u64, seed: u8) -> NewDevice {
        make(now, "203.0.113.9", &mut random_from(seed)).unwrap()
    }

    #[test]
    fn a_device_cookie_is_a_dev_token_and_only_its_hash_is_kept() {
        let d = made(5, 1);
        let parsed = token::parse(&d.token).unwrap();
        assert_eq!(parsed.kind, Kind::Dev);
        assert_eq!(parsed.id, d.device.id);
        assert_eq!(d.device.hash, token::secret_hash(parsed.secret));
        assert!(!format!("{d:?}").contains(&d.token));
        assert!(!serde_json::to_string(&d.device)
            .unwrap()
            .contains(parsed.secret));
    }

    #[test]
    fn a_valid_cookie_names_its_device_and_nothing_else_does() {
        let a = made(1, 1);
        let b = made(2, 2);
        let devices = vec![a.device.clone(), b.device.clone()];
        assert_eq!(
            verify(&devices, &a.token).map(|d| d.id.as_str()),
            Some(a.device.id.as_str())
        );
        assert_eq!(
            verify(&devices, &b.token).map(|d| d.id.as_str()),
            Some(b.device.id.as_str())
        );
        // The right id with the other's secret; an unknown id with a real secret; a session token in the
        // cookie; a token of the wrong length; junk; nothing in the list.
        let pa = token::parse(&a.token).unwrap();
        let pb = token::parse(&b.token).unwrap();
        let wrong_secret = format!("oaiydev_{}_{}", pa.id, pb.secret);
        let unknown_id = format!("oaiydev_{}_{}", "f".repeat(16), pa.secret);
        let session = format!("oaiyses_{}_{}", pa.id, pa.secret);
        for bad in [
            wrong_secret.as_str(),
            unknown_id.as_str(),
            session.as_str(),
            &a.token[..67],
            "",
            "garbage",
        ] {
            assert!(verify(&devices, bad).is_none(), "{bad}");
        }
        assert!(verify(&[], &a.token).is_none());
    }

    #[test]
    fn an_unknown_id_and_a_wrong_secret_are_the_same_answer_from_the_same_comparison() {
        // Both go through one constant-time hash comparison: the unknown id against the dummy hash.
        let a = made(1, 1);
        let devices = vec![a.device.clone()];
        let p = token::parse(&a.token).unwrap();
        let unknown = format!("oaiydev_{}_{}", "e".repeat(16), p.secret);
        let wrong = format!("oaiydev_{}_{}", p.id, "A".repeat(43));
        assert_eq!(verify(&devices, &unknown), None);
        assert_eq!(verify(&devices, &wrong), None);
    }

    #[test]
    fn at_most_sixteen_devices_are_kept_and_the_oldest_goes() {
        let mut devices = Vec::new();
        let mut first = None;
        // The number is the design's, written out: the test does not follow the constant.
        assert_eq!(MAX_DEVICES, 16);
        for i in 0..16 {
            let d = made(100 + i as u64, i as u8);
            first.get_or_insert(d.device.id.clone());
            assert!(add(&mut devices, d.device).is_empty());
        }
        assert_eq!(devices.len(), 16);
        let extra = made(500, 99);
        let dropped = add(&mut devices, extra.device.clone());
        assert_eq!(dropped, vec![first.unwrap()]);
        assert_eq!(devices.len(), 16);
        assert!(devices.iter().any(|d| d.id == extra.device.id));
    }

    #[test]
    fn a_device_is_removed_by_id_and_touched_with_its_last_use() {
        let a = made(10, 1);
        let b = made(20, 2);
        let mut devices = vec![a.device.clone(), b.device.clone()];
        touch(&mut devices, &a.device.id, 900, "198.51.100.7");
        assert_eq!(
            (devices[0].last_ms, devices[0].ip.as_str()),
            (900, "198.51.100.7")
        );
        // A use before the device existed (a clock that went back) is not before its creation.
        touch(&mut devices, &b.device.id, 1, "198.51.100.8");
        assert_eq!(devices[1].last_ms, 20);
        assert!(remove(&mut devices, &a.device.id));
        assert!(!remove(&mut devices, &a.device.id));
        assert_eq!(devices.len(), 1);
    }

    #[test]
    fn a_failure_of_the_random_source_makes_no_device() {
        let mut broken = |_: &mut [u8]| Err(MintError::NoRandomness);
        assert_eq!(
            make(1, "x", &mut broken).map(|_| ()),
            Err(MintError::NoRandomness)
        );
    }
}
