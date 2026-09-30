//! The scopes: exact strings, never patterns.
//!
//! There are 48 core scopes (18 read, 18 act, 12 dangerous) and 6 that other designs reserved.
//! Wildcards and bundles are expanded when a credential is created and never matched at request
//! time: the guard asks whether a credential holds the route's exact scope.
//!
//! Besides these, a `pat` may hold resource scopes, `connector.<plugin>.<command>`, which the
//! connector gate (not the route guard) checks.

use std::collections::BTreeSet;

/// Which family a scope is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Read,
    Act,
    Dangerous,
    /// Named for the relay, mobile and vault designs; in no preset but `owner` (and `vault.kt` in
    /// `ceremony`).
    Reserved,
}

#[derive(Clone, Copy, Debug)]
pub struct ScopeInfo {
    pub name: &'static str,
    pub group: Group,
    /// Needs a step-up (an elevated session) to use on a cookie session.
    pub dangerous: bool,
}

const fn info(name: &'static str, group: Group, dangerous: bool) -> ScopeInfo {
    ScopeInfo {
        name,
        group,
        dangerous,
    }
}

/// Every scope. The position is the scope's bit in a [`ScopeSet`]; appending is safe, reordering
/// is not (nothing persists a bit, but the tests pin the order).
pub const SCOPES: [ScopeInfo; 54] = [
    // Read (18)
    info("system.read", Group::Read, false),
    info("logs.read", Group::Read, false),
    info("services.read", Group::Read, false),
    info("models.read", Group::Read, false),
    info("plugins.read", Group::Read, false),
    info("events.read", Group::Read, false),
    info("flows.read", Group::Read, false),
    info("runs.read", Group::Read, false),
    info("ai.read", Group::Read, false),
    info("calls.read", Group::Read, false),
    info("calendar.read", Group::Read, false),
    info("contacts.read", Group::Read, false),
    info("agent.read", Group::Read, false),
    info("setup.read", Group::Read, false),
    info("link.read", Group::Read, false),
    info("companion.read", Group::Read, false),
    info("auth.read", Group::Read, false),
    info("control.read", Group::Read, false),
    // Act (18)
    info("services.control", Group::Act, false),
    info("models.write", Group::Act, false),
    info("plugins.control", Group::Act, false),
    info("flows.write", Group::Act, false),
    info("runs.write", Group::Act, false),
    info("connectors.use", Group::Act, false),
    info("ai.use", Group::Act, false),
    info("speech.use", Group::Act, false),
    info("calls.write", Group::Act, false),
    info("calendar.write", Group::Act, false),
    info("contacts.write", Group::Act, false),
    info("agent.tasks", Group::Act, false),
    info("agent.serve", Group::Act, false),
    info("agent.settings", Group::Act, false),
    info("setup.write", Group::Act, false),
    info("control.project", Group::Act, false),
    info("ui.events", Group::Act, false),
    info("auth.revoke", Group::Act, false),
    // Dangerous (12)
    info("services.define", Group::Dangerous, true),
    info("runtimes.install", Group::Dangerous, true),
    info("plugins.install", Group::Dangerous, true),
    info("ai.admin", Group::Dangerous, true),
    info("control.admin", Group::Dangerous, true),
    info("link.manage", Group::Dangerous, true),
    info("auth.manage", Group::Dangerous, true),
    info("companion.manage", Group::Dangerous, true),
    info("secrets.write", Group::Dangerous, true),
    info("system.update", Group::Dangerous, true),
    info("system.restart", Group::Dangerous, true),
    info("flows.approve", Group::Dangerous, true),
    // Reserved (6)
    info("relay.read", Group::Reserved, false),
    info("relay.manage", Group::Reserved, true),
    info("vault.read", Group::Reserved, false),
    info("vault.control", Group::Reserved, false),
    info("vault.admin", Group::Reserved, true),
    info("vault.kt", Group::Reserved, false),
];

/// The dangerous scopes a token can never hold: they exist for the dashboard (a cookie session
/// after step-up, or the desktop's own dashboard) and the console.
pub const NEVER_ON_A_TOKEN: [&str; 9] = [
    "control.admin",
    "auth.manage",
    "link.manage",
    "companion.manage",
    "secrets.write",
    "system.update",
    "system.restart",
    "relay.manage",
    "vault.admin",
];

/// The only dangerous scopes a native token may hold (minted with elevation or on the console, and
/// then for at most 24 hours).
pub const NATIVE_TOKEN_DANGEROUS: [&str; 5] = [
    "services.define",
    "runtimes.install",
    "plugins.install",
    "ai.admin",
    "flows.approve",
];

/// The scope's position in [`SCOPES`].
fn index_of(name: &str) -> Option<usize> {
    SCOPES.iter().position(|s| s.name == name)
}

/// The scope named `name`, if it is one of the 54.
pub fn info_of(name: &str) -> Option<&'static ScopeInfo> {
    index_of(name).map(|i| &SCOPES[i])
}

pub fn is_known(name: &str) -> bool {
    index_of(name).is_some()
}

/// A dangerous scope needs a step-up on a cookie session.
pub fn is_dangerous(name: &str) -> bool {
    info_of(name).is_some_and(|s| s.dangerous)
}

pub fn is_reserved(name: &str) -> bool {
    info_of(name).is_some_and(|s| s.group == Group::Reserved)
}

pub fn never_on_a_token(name: &str) -> bool {
    NEVER_ON_A_TOKEN.contains(&name)
}

/// `auth.read`, `auth.revoke`, `auth.manage`: what a derived credential can never hold.
pub fn is_auth_scope(name: &str) -> bool {
    name.starts_with("auth.")
}

/// A resource scope: `connector.<plugin>.<command>`, with `<plugin>` matching `[a-z0-9_-]{1,64}`
/// and `<command>` matching `[A-Za-z0-9_.-]{1,64}`. Checked by the connector gate, held only by a
/// `pat`.
pub fn is_resource_scope(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("connector.") else {
        return false;
    };
    let Some((plugin, command)) = rest.split_once('.') else {
        return false;
    };
    (1..=64).contains(&plugin.len())
        && plugin
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
        && (1..=64).contains(&command.len())
        && command
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// Why a name is not a scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScopeError {
    /// Not one of the 54, and not a resource scope.
    Unknown(String),
}

impl std::fmt::Display for ScopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScopeError::Unknown(name) => write!(f, "{name:?} is not a scope"),
        }
    }
}

impl std::error::Error for ScopeError {}

/// A set of scopes: the 54 by a bit each, everything else (resource scopes, and names a newer OAIY
/// wrote that this one does not know) by name. A name this build does not know never matches a
/// route, so keeping it costs nothing and loses nothing when the file goes back to the newer build.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScopeSet {
    bits: u64,
    extra: BTreeSet<String>,
}

impl ScopeSet {
    pub fn empty() -> Self {
        Self::default()
    }

    /// All 54.
    pub fn all() -> Self {
        Self {
            bits: (1u64 << SCOPES.len()) - 1,
            extra: BTreeSet::new(),
        }
    }

    /// The scopes named, every one of which must be a core scope or a resource scope.
    pub fn parse<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<Self, ScopeError> {
        let mut set = Self::empty();
        for name in names {
            if !is_known(name) && !is_resource_scope(name) {
                return Err(ScopeError::Unknown(name.to_string()));
            }
            set.insert(name);
        }
        Ok(set)
    }

    /// The scopes named, keeping what this build does not know (for reading a store).
    pub fn parse_lenient<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        let mut set = Self::empty();
        for name in names {
            set.insert(name);
        }
        set
    }

    /// The core scopes named; the names are compile-time constants.
    pub fn of(names: &[&str]) -> Self {
        let mut set = Self::empty();
        for name in names {
            assert!(is_known(name), "{name} is not a core scope");
            set.insert(name);
        }
        set
    }

    pub fn insert(&mut self, name: &str) {
        match index_of(name) {
            Some(i) => self.bits |= 1 << i,
            None => {
                self.extra.insert(name.to_string());
            }
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        match index_of(name) {
            Some(i) => self.bits & (1 << i) != 0,
            None => self.extra.contains(name),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bits == 0 && self.extra.is_empty()
    }

    pub fn len(&self) -> usize {
        self.bits.count_ones() as usize + self.extra.len()
    }

    /// Every scope of `self` is in `other`.
    pub fn is_subset_of(&self, other: &ScopeSet) -> bool {
        self.bits & !other.bits == 0 && self.extra.is_subset(&other.extra)
    }

    pub fn intersection(&self, other: &ScopeSet) -> ScopeSet {
        ScopeSet {
            bits: self.bits & other.bits,
            extra: self.extra.intersection(&other.extra).cloned().collect(),
        }
    }

    pub fn union(&self, other: &ScopeSet) -> ScopeSet {
        ScopeSet {
            bits: self.bits | other.bits,
            extra: self.extra.union(&other.extra).cloned().collect(),
        }
    }

    /// Only the scopes for which `keep` says so.
    pub fn filtered(&self, mut keep: impl FnMut(&str) -> bool) -> ScopeSet {
        let mut out = ScopeSet::empty();
        for name in self.names() {
            if keep(&name) {
                out.insert(&name);
            }
        }
        out
    }

    pub fn without_dangerous(&self) -> ScopeSet {
        self.filtered(|n| !is_dangerous(n))
    }

    pub fn has_dangerous(&self) -> bool {
        self.names().iter().any(|n| is_dangerous(n))
    }

    /// The names: core scopes in table order, then the rest sorted.
    pub fn names(&self) -> Vec<String> {
        let mut out: Vec<String> = SCOPES
            .iter()
            .enumerate()
            .filter(|(i, _)| self.bits & (1 << i) != 0)
            .map(|(_, s)| s.name.to_string())
            .collect();
        out.extend(self.extra.iter().cloned());
        out
    }

    /// Only the core scopes (no resource scopes, no unknown names).
    pub fn core_names(&self) -> Vec<&'static str> {
        SCOPES
            .iter()
            .enumerate()
            .filter(|(i, _)| self.bits & (1 << i) != 0)
            .map(|(_, s)| s.name)
            .collect()
    }

    /// The resource scopes (`connector.<plugin>.<command>`).
    pub fn resource_names(&self) -> Vec<&str> {
        self.extra
            .iter()
            .map(String::as_str)
            .filter(|n| is_resource_scope(n))
            .collect()
    }
}

impl serde::Serialize for ScopeSet {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.names().serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for ScopeSet {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let names = Vec::<String>::deserialize(d)?;
        Ok(ScopeSet::parse_lenient(names.iter().map(String::as_str)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn there_are_54_scopes_of_the_kinds_the_design_counts() {
        assert_eq!(SCOPES.len(), 54);
        let count = |g: Group| SCOPES.iter().filter(|s| s.group == g).count();
        assert_eq!(
            (
                count(Group::Read),
                count(Group::Act),
                count(Group::Dangerous),
                count(Group::Reserved)
            ),
            (18, 18, 12, 6)
        );
        assert_eq!(
            SCOPES.iter().filter(|s| s.dangerous).count(),
            14,
            "12 core and the two reserved that are dangerous"
        );
        let mut names: Vec<_> = SCOPES.iter().map(|s| s.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 54, "no duplicate scope");
        for s in &SCOPES {
            assert!(
                s.name.split('.').count() == 2
                    && s.name.bytes().all(|b| b.is_ascii_lowercase() || b == b'.'),
                "{}",
                s.name
            );
            assert_eq!(
                s.group == Group::Dangerous,
                s.dangerous && s.group != Group::Reserved,
                "{}",
                s.name
            );
        }
    }

    #[test]
    fn the_dangerous_core_scopes_are_the_twelve_the_design_names() {
        let mut d: Vec<_> = SCOPES
            .iter()
            .filter(|s| s.group == Group::Dangerous)
            .map(|s| s.name)
            .collect();
        d.sort_unstable();
        let mut want = vec![
            "services.define",
            "runtimes.install",
            "plugins.install",
            "ai.admin",
            "control.admin",
            "link.manage",
            "auth.manage",
            "companion.manage",
            "secrets.write",
            "system.update",
            "system.restart",
            "flows.approve",
        ];
        want.sort_unstable();
        assert_eq!(d, want);
        assert!(
            is_dangerous("relay.manage") && is_dangerous("vault.admin"),
            "the reserved dangerous ones"
        );
        assert!(
            !is_dangerous("auth.revoke"),
            "revoking never needs a password"
        );
        assert!(!is_dangerous("vault.kt"));
    }

    #[test]
    fn what_a_token_can_never_hold_is_a_dangerous_subset_and_native_tokens_hold_the_rest() {
        for s in NEVER_ON_A_TOKEN {
            assert!(is_dangerous(s), "{s}");
            assert!(never_on_a_token(s));
        }
        for s in NATIVE_TOKEN_DANGEROUS {
            assert!(is_dangerous(s), "{s}");
            assert!(!never_on_a_token(s), "{s}");
        }
        // Every dangerous scope is in exactly one of the two lists.
        for s in SCOPES.iter().filter(|s| s.dangerous) {
            let a = NEVER_ON_A_TOKEN.contains(&s.name);
            let b = NATIVE_TOKEN_DANGEROUS.contains(&s.name);
            assert!(a ^ b, "{} must be in exactly one list", s.name);
        }
    }

    #[test]
    fn a_resource_scope_has_the_shape_the_design_gives() {
        for ok in [
            "connector.aokie.call.answer",
            "connector.a.b",
            "connector.my-plugin_2.settings.get",
            "connector.x.A_b-c.d",
        ] {
            assert!(is_resource_scope(ok), "{ok}");
        }
        let long_plugin = format!("connector.{}.x", "a".repeat(65));
        let long_command = format!("connector.p.{}", "c".repeat(65));
        for bad in [
            "connector.",
            "connector.aokie",
            "connector..x",
            "connector.aokie.",
            "connector.Aokie.x",
            "connector.a b.x",
            "connector.a.b c",
            "connector.a/b.c",
            "connectors.use",
            "Connector.a.b",
            &long_plugin,
            &long_command,
        ] {
            assert!(!is_resource_scope(bad), "{bad}");
        }
        assert!(
            is_resource_scope(&format!("connector.{}.{}", "a".repeat(64), "c".repeat(64))),
            "64 is the longest"
        );
    }

    #[test]
    fn a_set_holds_exact_names_and_nothing_by_pattern() {
        let s = ScopeSet::parse(["ai.read", "connector.aokie.call.answer"]).unwrap();
        assert!(s.contains("ai.read"));
        assert!(!s.contains("ai.use"));
        assert!(!s.contains("ai.*"));
        assert!(!s.contains("ai"));
        assert!(!s.contains("connector.aokie.call.hangup"));
        assert!(!s.contains("connector.*"));
        assert!(s.contains("connector.aokie.call.answer"));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn parsing_refuses_a_name_that_is_not_a_scope_and_lenient_parsing_keeps_it_without_granting() {
        assert_eq!(
            ScopeSet::parse(["ai.read", "ai.everything"]),
            Err(ScopeError::Unknown("ai.everything".into()))
        );
        assert!(ScopeSet::parse(["*"]).is_err());
        assert!(ScopeSet::parse([""]).is_err());
        let s = ScopeSet::parse_lenient(["ai.read", "future.scope"]);
        assert!(s.contains("future.scope"));
        assert!(s.contains("ai.read"));
        // It survives a round trip (the file goes back to the newer build), and grants nothing here.
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, r#"["ai.read","future.scope"]"#);
        let back: ScopeSet = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
        assert!(back.core_names() == ["ai.read"]);
    }

    #[test]
    fn set_algebra() {
        let a = ScopeSet::of(&["ai.read", "ai.use", "services.define"]);
        let b = ScopeSet::of(&["ai.read", "flows.read"]);
        assert_eq!(a.intersection(&b), ScopeSet::of(&["ai.read"]));
        assert_eq!(a.union(&b).len(), 4);
        assert!(ScopeSet::of(&["ai.read"]).is_subset_of(&a));
        assert!(!b.is_subset_of(&a));
        assert!(ScopeSet::empty().is_subset_of(&ScopeSet::empty()));
        assert!(a.has_dangerous() && !a.without_dangerous().has_dangerous());
        assert_eq!(a.without_dangerous(), ScopeSet::of(&["ai.read", "ai.use"]));
        assert_eq!(ScopeSet::all().len(), 54);
        assert!(ScopeSet::all().contains("vault.kt") && ScopeSet::all().contains("system.read"));
        assert!(ScopeSet::all().extra.is_empty());
        // A subset check must look at resource scopes too.
        let with = ScopeSet::parse(["connector.p.c"]).unwrap();
        assert!(!with.is_subset_of(&ScopeSet::empty()));
        assert!(with.is_subset_of(&with));
    }

    #[test]
    fn names_come_out_in_table_order_and_the_resource_scopes_last() {
        let s =
            ScopeSet::parse(["connector.p.c", "runs.write", "system.read", "auth.revoke"]).unwrap();
        assert_eq!(
            s.names(),
            ["system.read", "runs.write", "auth.revoke", "connector.p.c"]
        );
        assert_eq!(s.resource_names(), ["connector.p.c"]);
    }
}
