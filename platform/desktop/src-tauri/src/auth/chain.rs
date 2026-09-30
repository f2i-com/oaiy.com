//! Chain validity: a credential is valid only while its whole parent chain is.
//!
//! A child (an app-host session of a dashboard session, a derived credential of a `desk` or `pat`)
//! records its parent. At every use the credential itself and every ancestor must be not revoked, not
//! expired and, for a session, not idle-expired and not older than the owner's `min_session_epoch`.
//! Nothing is cached: an ancestor that ends takes every descendant with it at the next request, and a
//! parent that is missing (dropped, or never there) ends its children too. A chain that loops or is
//! longer than [`MAX_DEPTH`] is broken, not followed.

use std::collections::HashMap;

use super::store::Record;
use super::token::Kind;

/// The most ancestors followed. Real chains are two long (a derived credential of a session of a
/// session); this is a bound against a file that was edited into a loop.
pub const MAX_DEPTH: usize = 8;

/// The parent id that means the operator's environment token (`OAIY_SERVER_TOKEN`).
pub const STATIC_PARENT: &str = super::principal::STATIC_ID;

/// Why a credential (or one of its ancestors) is not valid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Broken {
    /// A record with a sender-constraint this build does not understand (`cnf` is reserved).
    Unusable,
    Revoked {
        reason: Option<String>,
    },
    Expired,
    /// A session unused for longer than its idle timeout.
    Idle,
    /// A session older than the owner's `min_session_epoch`.
    Epoch,
    /// The chain names a parent that is not there, loops, or is too long.
    Missing,
}

/// What a check is made against.
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    pub now_ms: u64,
    /// `min_session_epoch` of `owner.json`.
    pub min_epoch: u64,
    /// Whether the environment token is configured (a `static` parent is valid only then).
    pub static_present: bool,
}

/// Check one record on its own.
pub fn check_record(rec: &Record, ctx: &Ctx) -> Result<(), Broken> {
    if rec.cnf.as_ref().is_some_and(|c| !c.is_null()) {
        return Err(Broken::Unusable);
    }
    if rec.revoked_ms.is_some() {
        return Err(Broken::Revoked {
            reason: rec.revoked_reason.clone(),
        });
    }
    // No expiry at all is not "never": it is expired, so a record missing the field fails closed.
    if ctx.now_ms >= rec.expires_ms {
        return Err(Broken::Expired);
    }
    if let Some(idle) = rec.idle_ms {
        // A last use in the future (a clock that went back) counts as now.
        let last = rec.last_used_ms.unwrap_or(rec.created_ms).min(ctx.now_ms);
        if ctx.now_ms - last > idle {
            return Err(Broken::Idle);
        }
    }
    if rec.kind == Kind::Ses && rec.epoch < ctx.min_epoch {
        return Err(Broken::Epoch);
    }
    Ok(())
}

/// Check the record `start` and every ancestor. On failure, the record that failed is named, so a
/// caller can tell "it ended" from "its parent ended".
pub fn check_chain(
    records: &HashMap<String, Record>,
    start: &Record,
    ctx: &Ctx,
) -> Result<(), (String, Broken)> {
    check_record(start, ctx).map_err(|b| (start.id.clone(), b))?;
    let mut seen: Vec<&str> = vec![start.id.as_str()];
    let mut next = start.parent.as_deref();
    while let Some(parent_id) = next {
        if parent_id == STATIC_PARENT {
            return if ctx.static_present {
                Ok(())
            } else {
                Err((parent_id.to_string(), Broken::Missing))
            };
        }
        if seen.len() > MAX_DEPTH || seen.contains(&parent_id) {
            return Err((parent_id.to_string(), Broken::Missing));
        }
        let Some(parent) = records.get(parent_id) else {
            return Err((parent_id.to_string(), Broken::Missing));
        };
        check_record(parent, ctx).map_err(|b| (parent.id.clone(), b))?;
        seen.push(parent.id.as_str());
        next = parent.parent.as_deref();
    }
    Ok(())
}

/// The ids of the ancestors of `start`, nearest first (as far as the records go).
pub fn ancestors(records: &HashMap<String, Record>, start: &Record) -> Vec<String> {
    let mut out = Vec::new();
    let mut next = start.parent.as_deref();
    while let Some(id) = next {
        if out.len() >= MAX_DEPTH || out.iter().any(|o: &String| o == id) {
            break;
        }
        out.push(id.to_string());
        next = records.get(id).and_then(|r| r.parent.as_deref());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::store::Record;

    const NOW: u64 = 1_790_000_000_000;
    const HOUR: u64 = 3_600_000;

    fn ctx(now: u64) -> Ctx {
        Ctx {
            now_ms: now,
            min_epoch: 0,
            static_present: true,
        }
    }

    fn rec(id: &str, kind: Kind, parent: Option<&str>) -> Record {
        let mut r = Record::blank(id, kind);
        r.parent = parent.map(str::to_string);
        r.created_ms = NOW - HOUR;
        r.expires_ms = NOW + 24 * HOUR;
        r
    }

    fn map(records: Vec<Record>) -> HashMap<String, Record> {
        records.into_iter().map(|r| (r.id.clone(), r)).collect()
    }

    fn chain(
        records: &HashMap<String, Record>,
        id: &str,
        now: u64,
    ) -> Result<(), (String, Broken)> {
        check_chain(records, &records[id], &ctx(now))
    }

    #[test]
    fn a_credential_with_no_problem_and_no_parent_is_valid() {
        let m = map(vec![rec("a", Kind::Pat, None)]);
        assert_eq!(chain(&m, "a", NOW), Ok(()));
    }

    #[test]
    fn a_revoked_parent_ends_its_child() {
        let mut parent = rec("p", Kind::Ses, None);
        parent.revoked_ms = Some(NOW - 10);
        parent.revoked_reason = Some("logged_out".into());
        let m = map(vec![parent, rec("c", Kind::Run, Some("p"))]);
        assert_eq!(
            chain(&m, "c", NOW),
            Err((
                "p".into(),
                Broken::Revoked {
                    reason: Some("logged_out".into())
                }
            ))
        );
    }

    #[test]
    fn an_expired_parent_ends_its_child() {
        let mut parent = rec("p", Kind::Pat, None);
        parent.expires_ms = NOW - 1;
        let m = map(vec![parent, rec("c", Kind::Run, Some("p"))]);
        assert_eq!(chain(&m, "c", NOW), Err(("p".into(), Broken::Expired)));
        // The instant of expiry already counts as expired.
        let mut on_the_dot = rec("q", Kind::Pat, None);
        on_the_dot.expires_ms = NOW;
        assert_eq!(check_record(&on_the_dot, &ctx(NOW)), Err(Broken::Expired));
        assert_eq!(check_record(&on_the_dot, &ctx(NOW - 1)), Ok(()));
    }

    #[test]
    fn an_idle_parent_session_ends_its_child_and_using_the_child_would_have_kept_it_alive() {
        let mut parent = rec("p", Kind::Ses, None);
        parent.idle_ms = Some(8 * HOUR);
        parent.last_used_ms = Some(NOW - 9 * HOUR);
        let m = map(vec![parent.clone(), rec("c", Kind::Ses, Some("p"))]);
        assert_eq!(chain(&m, "c", NOW), Err(("p".into(), Broken::Idle)));
        // Used a minute ago: fine.
        let mut fresh = parent;
        fresh.last_used_ms = Some(NOW - 60_000);
        let m = map(vec![fresh, rec("c", Kind::Ses, Some("p"))]);
        assert_eq!(chain(&m, "c", NOW), Ok(()));
    }

    #[test]
    fn a_session_is_idle_only_past_its_timeout_and_a_last_use_in_the_future_counts_as_now() {
        let mut s = rec("s", Kind::Ses, None);
        s.idle_ms = Some(8 * HOUR);
        s.last_used_ms = Some(NOW - 8 * HOUR);
        assert_eq!(
            check_record(&s, &ctx(NOW)),
            Ok(()),
            "exactly at the timeout is still inside it"
        );
        assert_eq!(check_record(&s, &ctx(NOW + 1)), Err(Broken::Idle));
        // A last use in the future (a clock that went back) is clamped to now. The store rewrites it
        // to now when it next looks, so the timeout runs from there (store::clamp tests).
        s.last_used_ms = Some(NOW + 100 * HOUR);
        assert_eq!(check_record(&s, &ctx(NOW)), Ok(()));
        // Never used: idle counts from its creation.
        let mut never = rec("n", Kind::Ses, None);
        never.idle_ms = Some(HOUR);
        never.last_used_ms = None;
        never.created_ms = NOW - 2 * HOUR;
        assert_eq!(check_record(&never, &ctx(NOW)), Err(Broken::Idle));
        // A credential with no idle timeout never idles.
        let mut pat = rec("p", Kind::Pat, None);
        pat.last_used_ms = Some(NOW - 1000 * HOUR);
        assert_eq!(check_record(&pat, &ctx(NOW)), Ok(()));
    }

    #[test]
    fn a_missing_parent_a_loop_and_a_chain_that_is_too_long_are_broken() {
        let m = map(vec![rec("c", Kind::Run, Some("gone"))]);
        assert_eq!(chain(&m, "c", NOW), Err(("gone".into(), Broken::Missing)));
        let m = map(vec![
            rec("a", Kind::Run, Some("b")),
            rec("b", Kind::Run, Some("a")),
        ]);
        assert_eq!(chain(&m, "a", NOW).unwrap_err().1, Broken::Missing);
        let m = map(vec![rec("s", Kind::Run, Some("s"))]);
        assert_eq!(chain(&m, "s", NOW).unwrap_err().1, Broken::Missing);
        // A chain of exactly MAX_DEPTH ancestors is followed; one more is refused.
        let long = |n: usize| {
            let mut v = vec![rec("r0", Kind::Pat, None)];
            for i in 1..=n {
                v.push(rec(
                    &format!("r{i}"),
                    Kind::Run,
                    Some(&format!("r{}", i - 1)),
                ));
            }
            map(v)
        };
        assert_eq!(
            chain(&long(MAX_DEPTH), &format!("r{MAX_DEPTH}"), NOW),
            Ok(())
        );
        assert_eq!(
            chain(&long(MAX_DEPTH + 1), &format!("r{}", MAX_DEPTH + 1), NOW)
                .unwrap_err()
                .1,
            Broken::Missing
        );
    }

    #[test]
    fn the_static_parent_is_valid_only_while_the_environment_token_is_configured() {
        let m = map(vec![rec("c", Kind::Run, Some(STATIC_PARENT))]);
        assert_eq!(check_chain(&m, &m["c"], &ctx(NOW)), Ok(()));
        let none = Ctx {
            static_present: false,
            ..ctx(NOW)
        };
        assert_eq!(
            check_chain(&m, &m["c"], &none),
            Err((STATIC_PARENT.to_string(), Broken::Missing))
        );
    }

    #[test]
    fn a_grandparent_that_ends_ends_the_whole_line() {
        let mut g = rec("g", Kind::Ses, None);
        g.revoked_ms = Some(NOW - 1);
        let m = map(vec![
            g,
            rec("p", Kind::Ses, Some("g")),
            rec("c", Kind::Run, Some("p")),
        ]);
        assert_eq!(chain(&m, "c", NOW).unwrap_err().0, "g");
        assert_eq!(chain(&m, "p", NOW).unwrap_err().0, "g");
    }

    #[test]
    fn a_session_older_than_the_owners_epoch_is_ended_and_a_pat_is_not() {
        let mut s = rec("s", Kind::Ses, None);
        s.epoch = 1;
        let c = Ctx {
            min_epoch: 2,
            ..ctx(NOW)
        };
        assert_eq!(check_record(&s, &c), Err(Broken::Epoch));
        s.epoch = 2;
        assert_eq!(check_record(&s, &c), Ok(()));
        let mut p = rec("p", Kind::Pat, None);
        p.epoch = 0;
        assert_eq!(
            check_record(&p, &c),
            Ok(()),
            "only sessions are ended by the epoch"
        );
    }

    #[test]
    fn a_record_with_a_sender_constraint_this_build_does_not_know_is_unusable() {
        let mut r = rec("r", Kind::Pat, None);
        r.cnf = Some(serde_json::json!({ "jkt": "abc" }));
        assert_eq!(check_record(&r, &ctx(NOW)), Err(Broken::Unusable));
        r.cnf = Some(serde_json::Value::Null);
        assert_eq!(check_record(&r, &ctx(NOW)), Ok(()));
    }

    #[test]
    fn a_record_with_no_expiry_is_expired() {
        let mut r = rec("r", Kind::Pat, None);
        r.expires_ms = 0;
        assert_eq!(check_record(&r, &ctx(NOW)), Err(Broken::Expired));
    }

    #[test]
    fn ancestors_are_listed_nearest_first_and_a_loop_stops() {
        let m = map(vec![
            rec("g", Kind::Ses, None),
            rec("p", Kind::Ses, Some("g")),
            rec("c", Kind::Run, Some("p")),
        ]);
        assert_eq!(ancestors(&m, &m["c"]), ["p", "g"]);
        assert!(ancestors(&m, &m["g"]).is_empty());
        let l = map(vec![
            rec("a", Kind::Run, Some("b")),
            rec("b", Kind::Run, Some("a")),
        ]);
        assert_eq!(ancestors(&l, &l["a"]), ["b", "a"]);
    }
}
