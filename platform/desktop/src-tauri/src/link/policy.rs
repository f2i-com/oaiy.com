//! What the provider's website may run on this computer, through the command relay.
//!
//! The relay carries commands somebody queued on the provider's site for this
//! desktop to run. Anyone who acts as the account's owner there can queue one: a
//! signed-in browser, a leaked key. This side cannot tell them from the owner, so
//! what a relayed command may reach is decided HERE, from a list this build
//! ships (`resources/relay-policy.json`), and never from what a plugin says about
//! itself.
//!
//! * A connector with an entry gets exactly the verbs the entry names. There is
//!   no wildcard: a verb is reachable when it is written down.
//! * A connector with no entry gets its plugin's reads: the verbs it declares and
//!   does not journal. That is a weaker rule, because the plugin decides what it
//!   journals, which is why a plugin that does things in the world needs an entry.
//! * A policy that is missing, cannot be read, or is not understood allows
//!   nothing at all, reads included.
//!
//! Only commands that arrive through the relay are asked. A plugin's own screens,
//! the Agent's control tools and flows on this computer call the plugin host
//! directly and never come here.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::plugins::GateRefusal;

/// The policy this build ships. Embedded, like the provider's descriptor, so a
/// plugin (which runs as the same user) has no file to rewrite.
const SHIPPED: &str = include_str!("../../resources/relay-policy.json");

/// The one policy version this build reads. A newer file means rules this build
/// does not know how to apply, and guessing would mean guessing wide.
const VERSION: u32 = 1;

/// The policy file's shape. Strict: a misspelt field is a refusal, not a rule
/// that quietly does nothing.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileShape {
    version: u32,
    #[serde(default)]
    #[allow(dead_code)]
    about: Option<String>,
    connectors: BTreeMap<String, EntryShape>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EntryShape {
    #[serde(default)]
    #[allow(dead_code)]
    about: Option<String>,
    /// The verbs the website may run. Exact names.
    allow: Vec<String>,
    /// Where each verb is seen being queued. For the people who edit the list;
    /// nothing here reads it (a test checks it is kept).
    #[serde(default)]
    #[allow(dead_code)]
    evidence: BTreeMap<String, Vec<String>>,
}

/// What the plugin registry says about a command, for a connector the policy has
/// no entry for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Declared {
    /// No installed plugin serves the connector.
    NoPlugin,
    /// A plugin serves it: does it declare the command, and does it journal it?
    Plugin { declares: bool, journalled: bool },
}

/// The rules for commands that arrive through the relay.
#[derive(Debug, Clone)]
pub enum RelayPolicy {
    /// Connector id to the verbs the website may run.
    Listed(BTreeMap<String, BTreeSet<String>>),
    /// The policy could not be used, so nothing runs. Carries why.
    Closed(String),
}

impl RelayPolicy {
    /// The policy this build ships. If even that cannot be read, a closed one:
    /// the way to be wrong here is to refuse.
    pub fn shipped() -> RelayPolicy {
        RelayPolicy::from_text(Some(SHIPPED))
    }

    /// A policy from its text. `None` is a policy nobody could read (absent or
    /// unreadable). Anything that is not a complete, understood policy is a
    /// closed one, and never an empty one: an empty policy would hand every
    /// connector the default rule, which still lets reads through.
    pub fn from_text(text: Option<&str>) -> RelayPolicy {
        let Some(text) = text else {
            return RelayPolicy::closed("there is no policy to read");
        };
        if text.trim().is_empty() {
            return RelayPolicy::closed("the policy is empty");
        }
        match parse(text) {
            Ok(rules) => RelayPolicy::Listed(rules),
            Err(why) => RelayPolicy::closed(why),
        }
    }

    /// A policy that allows nothing.
    pub fn closed(why: impl Into<String>) -> RelayPolicy {
        RelayPolicy::Closed(why.into())
    }

    /// Why nothing runs, when nothing does.
    pub fn closed_because(&self) -> Option<&str> {
        match self {
            RelayPolicy::Closed(why) => Some(why),
            RelayPolicy::Listed(_) => None,
        }
    }

    /// May the website run `command` on `connector`?
    ///
    /// `command_id` is the relayed command's own id. The plugin's idempotency key
    /// is made from it, so a command without one has no key a retry could be
    /// recognised by.
    ///
    /// `plugin` says what the plugin registry knows, and is asked ONLY for a
    /// connector the policy has no entry for: an entry is the whole answer, and a
    /// plugin has no say in it.
    pub fn check(
        &self,
        connector: &str,
        command: &str,
        command_id: &str,
        plugin: impl FnOnce() -> Declared,
    ) -> Result<(), Refusal> {
        let RelayPolicy::Listed(rules) = self else {
            return Err(Refusal::PolicyUnreadable { command: command.to_string() });
        };
        if command_id.trim().is_empty() {
            return Err(Refusal::NoCommandId { command: command.to_string() });
        }
        let only_here = |why| Refusal::ThisComputerOnly {
            connector: connector.to_string(),
            command: command.to_string(),
            why,
        };
        match rules.get(connector) {
            Some(allow) if allow.contains(command) => Ok(()),
            Some(_) => Err(only_here(Why::NotListed)),
            None => match plugin() {
                Declared::NoPlugin => Err(Refusal::NotOffered {
                    connector: connector.to_string(),
                    command: command.to_string(),
                    plugin_installed: false,
                }),
                Declared::Plugin { declares: false, .. } => Err(Refusal::NotOffered {
                    connector: connector.to_string(),
                    command: command.to_string(),
                    plugin_installed: true,
                }),
                Declared::Plugin { journalled: true, .. } => Err(only_here(Why::Journalled)),
                Declared::Plugin { .. } => Ok(()),
            },
        }
    }
}

/// Why a command that is on this computer's side of the line stays there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// The connector has an entry and the verb is not on it.
    NotListed,
    /// The connector has no entry, and the plugin journals the verb: it changes
    /// something, so without an entry naming it, it stays here.
    Journalled,
}

/// Why a relayed command was refused, in the terms a caller can act on. Typed like
/// [`GateRefusal`], and worded for the person reading it on the provider's site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// A command that can only be run from OAIY on this computer.
    ThisComputerOnly { connector: String, command: String, why: Why },
    /// The plugin does not offer the command (or nothing offers the connector).
    /// Said in the gate's own words, since that is what it would have said.
    NotOffered { connector: String, command: String, plugin_installed: bool },
    /// The policy could not be used, so no command for a plugin runs.
    PolicyUnreadable { command: String },
    /// The command carried no id.
    NoCommandId { command: String },
}

impl Refusal {
    /// The closed-taxonomy code, the same ones the plugin gate uses.
    pub fn code(&self) -> &'static str {
        match self {
            Refusal::ThisComputerOnly { .. } => "capability_denied",
            Refusal::NotOffered { connector, command, plugin_installed } => {
                self.gate(connector, command, *plugin_installed).code()
            }
            Refusal::PolicyUnreadable { .. } => "capability_unavailable",
            Refusal::NoCommandId { .. } => "invalid_request",
        }
    }

    /// A short stable name for the audit log: which rule said no.
    pub fn reason(&self) -> &'static str {
        match self {
            Refusal::ThisComputerOnly { why: Why::NotListed, .. } => "not_listed",
            Refusal::ThisComputerOnly { why: Why::Journalled, .. } => "journalled",
            Refusal::NotOffered { .. } => "not_declared",
            Refusal::PolicyUnreadable { .. } => "policy_unreadable",
            Refusal::NoCommandId { .. } => "no_command_id",
        }
    }

    /// The sentence a person reads on the provider's site. Names the command, and
    /// says where it can be run.
    pub fn message(&self) -> String {
        match self {
            Refusal::ThisComputerOnly { connector, command, .. } => format!(
                "The \"{command}\" command of the \"{connector}\" plugin can only be run from OAIY \
                 on this computer, so it was refused here and never reached the plugin."
            ),
            Refusal::NotOffered { connector, command, plugin_installed } => {
                self.gate(connector, command, *plugin_installed).message()
            }
            Refusal::PolicyUnreadable { command } => format!(
                "OAIY could not read its list of commands the website may run, so \"{command}\" \
                 was refused, and so is every other command for a plugin until that is fixed. \
                 It can be run from OAIY on this computer."
            ),
            Refusal::NoCommandId { command } => format!(
                "The website sent \"{command}\" without a command id, so it was refused: every \
                 command needs one, so that a retry cannot run it twice."
            ),
        }
    }

    fn gate(&self, connector: &str, command: &str, plugin_installed: bool) -> GateRefusal {
        if plugin_installed {
            GateRefusal::CapabilityDenied {
                connector_id: connector.to_string(),
                command: command.to_string(),
            }
        } else {
            GateRefusal::ConnectorMissing { connector_id: connector.to_string() }
        }
    }
}

/// The rules in a policy file, or why they cannot be used.
fn parse(text: &str) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let file: FileShape = serde_json::from_str(text.trim_start_matches('\u{feff}'))
        .map_err(|e| format!("it is not a policy this build understands: {e}"))?;
    if file.version != VERSION {
        return Err(format!(
            "it is version {}, and this build reads version {VERSION}",
            file.version
        ));
    }
    let mut rules = BTreeMap::new();
    for (connector, entry) in file.connectors {
        if !plain(&connector) {
            return Err(format!("the connector id {connector:?} is not a plain id"));
        }
        if connector == super::ops::DESKTOP_CONNECTOR {
            return Err(format!(
                "{connector:?} is this app's own closed list of operations and takes no entry"
            ));
        }
        let mut verbs = BTreeSet::new();
        for verb in entry.allow {
            if !plain(&verb) {
                return Err(format!(
                    "{connector}: {verb:?} is not a plain verb (no wildcard, space or empty name)"
                ));
            }
            if !verbs.insert(verb.clone()) {
                return Err(format!("{connector}: {verb} is listed twice"));
            }
        }
        rules.insert(connector, verbs);
    }
    Ok(rules)
}

/// A name of letters, digits and `.` `_` `-`: no wildcard, no space, nothing that
/// looks like a name but is not the one the plugin declares.
fn plain(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 96
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::PluginManifest;

    /// Aokie's manifest as OAIY tests it. Its 34 declared commands and 15
    /// journalled ones are the same as the plugin's own `manifest.json`.
    const AOKIE_MANIFEST: &str = include_str!("../plugins/fixtures/aokie-v4.manifest.json");

    fn aokie() -> PluginManifest {
        serde_json::from_str(AOKIE_MANIFEST).expect("the Aokie fixture parses")
    }

    fn shipped() -> RelayPolicy {
        RelayPolicy::shipped()
    }

    /// What the plugin registry would say about `command` of `connector`, from a manifest.
    fn declared(manifest: &PluginManifest, connector: &str, command: &str) -> Declared {
        Declared::Plugin {
            declares: manifest.declares_command(connector, command),
            journalled: manifest.is_journalled(command),
        }
    }

    fn never() -> Declared {
        panic!("the plugin was asked about a connector the policy has an entry for")
    }

    fn one(connector: &str, verbs: &[&str]) -> RelayPolicy {
        let list = verbs.iter().map(|v| format!("\"{v}\"")).collect::<Vec<_>>().join(",");
        RelayPolicy::from_text(Some(&format!(
            "{{\"version\":1,\"connectors\":{{\"{connector}\":{{\"allow\":[{list}]}}}}}}"
        )))
    }

    // ---- what the shipped policy is -----------------------------------------------

    /// The verbs the shipped policy allows Aokie, and where FormLogic queues each
    /// one. Kept here as well as in the policy file: this is the list a change has
    /// to be made to on purpose, twice.
    const AOKIE_ALLOWED: [&str; 11] = [
        "call.current",
        "call.answer",
        "call.reject",
        "call.hangup",
        "call.operatorSpeak",
        "call.dial",
        "sms.send",
        "sms.threads",
        "sms.thread",
        "phone.status",
        "dongle.list",
    ];

    /// The reads among them. The rest change something on the phone line, and so
    /// must be journalled, which is what makes the plugin ask for a key.
    const AOKIE_READS: [&str; 5] =
        ["call.current", "sms.threads", "sms.thread", "phone.status", "dongle.list"];

    /// The verbs FormLogic's MCP `connector_command` tool tells an AI client to send
    /// to `aokie`, as its description lists them
    /// (`formlogic/backend/src/Services/ChatToolsService.php:1241`, FormLogic at
    /// 12275d2e). An MCP client is one of the three things the brief derives the list
    /// from, and the tool text is the only place it is written down.
    const MCP_TOOL_TEXT: [&str; 10] = [
        "call.answer",
        "call.reject",
        "call.hangup",
        "call.operatorSpeak",
        "sms.send",
        "sms.thread",
        "call.current",
        "phone.status",
        "dongle.list",
        "dongle.diagnostics",
    ];

    /// What that text names and the policy still leaves on this computer, each on
    /// purpose (the entry's `about` says why).
    const MCP_TOOL_TEXT_KEPT_HERE: [&str; 1] = ["dongle.diagnostics"];

    #[test]
    fn the_shipped_policy_is_understood() {
        // The one that ships is read by the same strict parser as any other. If it
        // stopped parsing, every command for a plugin would be refused, and the
        // provider's call console would be blank: so this fails here, not there.
        let RelayPolicy::Listed(rules) = shipped() else {
            panic!("the shipped policy is closed: {:?}", shipped().closed_because());
        };
        let aokie = rules.get("aokie").expect("Aokie has an entry");
        let listed: Vec<&str> = aokie.iter().map(String::as_str).collect();
        let mut expected = AOKIE_ALLOWED.to_vec();
        expected.sort_unstable();
        assert_eq!(listed, expected);
    }

    #[test]
    fn every_verb_the_policy_lists_has_evidence_and_no_evidence_is_left_over() {
        // Each verb the website may run says where FormLogic is seen queueing it,
        // as `path:line` from the formlogic.com repository's root. A verb nobody can
        // point at is one nobody should be able to reach.
        let file: serde_json::Value = serde_json::from_str(SHIPPED).unwrap();
        for (connector, entry) in file["connectors"].as_object().unwrap() {
            let allow: BTreeSet<&str> =
                entry["allow"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            let evidence = entry["evidence"].as_object().expect("an entry keeps its evidence");
            let cited: BTreeSet<&str> = evidence.keys().map(String::as_str).collect();
            assert_eq!(allow, cited, "{connector}: the verbs and their evidence must be the same set");
            for (verb, places) in evidence {
                let places = places.as_array().unwrap();
                assert!(!places.is_empty(), "{connector} {verb}: no evidence");
                for place in places {
                    let place = place.as_str().unwrap();
                    let (path, line) = place.rsplit_once(':').unwrap_or_else(|| panic!("{place} is not path:line"));
                    assert!(path.contains('/') && !path.contains(' '), "{place}: not a path");
                    assert!(
                        line.parse::<u32>().is_ok_and(|n| n > 0),
                        "{place}: the line must be a positive number"
                    );
                }
            }
        }
    }

    #[test]
    fn aokie_lists_only_commands_it_declares() {
        // A verb misspelt in the policy would be dead: listed, never declared, so
        // the gate would refuse it every time and the website would break with no
        // reason in sight.
        let manifest = aokie();
        for verb in AOKIE_ALLOWED {
            assert!(manifest.declares_command("aokie", verb), "{verb} is not a command Aokie declares");
        }
    }

    #[test]
    fn aokie_may_run_the_phone_line_from_the_website() {
        let policy = shipped();
        for verb in AOKIE_ALLOWED {
            assert_eq!(policy.check("aokie", verb, "cmd-1", never), Ok(()), "{verb}");
        }
    }

    #[test]
    fn aokie_keeps_the_computers_setup_on_the_computer() {
        // What the defect let through, and the rest of what FormLogic gives only its
        // Device Admin role. Every one is a real command Aokie declares.
        let refused = [
            // Drivers and certificates: install_driver takes a vid and pid from the
            // command and runs an elevated helper.
            "dongle.installDriver",
            "dongle.restoreDriver",
            "dongle.removeCerts",
            "dongle.reset",
            "dongle.setPreferred",
            // Pairing, and the phone's connection.
            "phone.startPairing",
            "phone.stopPairing",
            "phone.confirmPairing",
            "phone.removePaired",
            "phone.connect",
            "phone.disconnect",
            // Consent, and settings (which can send a call's audio elsewhere).
            "consent.set",
            "consent.revoke",
            "settings.set",
            // The outbox.
            "outbox.redrive",
            // Reads of those. `dongle.diagnostics` is one: FormLogic's MCP tool text
            // names it, but grants it to Device Admin only, it shows the dongle's and
            // the phone's Bluetooth addresses, and its `simulate` option plays a
            // scripted call in dev mode, which a list of verbs cannot tell from the
            // plain read.
            "dongle.getPreferred",
            "dongle.diagnostics",
            "phone.listPaired",
            "settings.get",
            "consent.get",
            // Verbs nothing on the provider's side queues through the relay.
            // `call.configureAgent` is FormLogic's Personalize Caller flow, which holds
            // logic and condition nodes, so it runs in the browser or on this computer
            // and never crosses the relay (its cloud runner refuses those nodes).
            "call.configureAgent",
            "call.switchboard",
            "call.activate",
        ];
        let manifest = aokie();
        let policy = shipped();
        for verb in refused {
            assert!(manifest.declares_command("aokie", verb), "{verb}: not a command Aokie declares, so this proves nothing");
            let refusal = policy.check("aokie", verb, "cmd-1", never).expect_err(verb);
            assert_eq!(
                refusal,
                Refusal::ThisComputerOnly {
                    connector: "aokie".into(),
                    command: verb.into(),
                    why: Why::NotListed
                },
                "{verb}"
            );
        }
    }

    #[test]
    fn what_formlogics_mcp_tool_tells_an_ai_to_send_is_allowed_unless_kept_here_on_purpose() {
        // The list is what FormLogic is seen queueing through its relay, and its MCP
        // `connector_command` tool is one of the three places it queues from (the call
        // console and the front-desk role's grants are the others). A verb the tool
        // text names and the policy refuses is a website feature that breaks with no
        // reason in sight, unless somebody decided it should: that decision is in
        // `MCP_TOOL_TEXT_KEPT_HERE`, and in the entry's `about`.
        let policy = shipped();
        for verb in MCP_TOOL_TEXT {
            let allowed = policy.check("aokie", verb, "cmd-1", never).is_ok();
            let kept = MCP_TOOL_TEXT_KEPT_HERE.contains(&verb);
            assert_eq!(allowed, !kept, "{verb}: allowed {allowed}, kept here on purpose {kept}");
        }
        for verb in MCP_TOOL_TEXT_KEPT_HERE {
            assert!(MCP_TOOL_TEXT.contains(&verb), "{verb} is kept for a reason the text does not give");
        }
    }

    #[test]
    fn what_aokie_declares_is_either_run_from_the_website_or_kept_on_the_computer() {
        // Every command in the manifest lands on one side or the other, so a verb
        // added to the plugin later is caught by this list before it is reachable.
        let manifest = aokie();
        let policy = shipped();
        let declared_verbs = &manifest.connectors[0].commands;
        assert_eq!(declared_verbs.len(), 34);
        let allowed = declared_verbs
            .iter()
            .filter(|v| policy.check("aokie", v, "cmd-1", never).is_ok())
            .count();
        assert_eq!(allowed, AOKIE_ALLOWED.len());
    }

    #[test]
    fn what_aokie_may_run_and_changes_something_carries_a_key() {
        // The plugin refuses a journalled command that arrives without an
        // idempotency key, so a retried "answer" or "send" cannot happen twice.
        // Every listed verb that is not a read has to be one it journals: an
        // effectful verb it forgot to journal would get no key.
        let manifest = aokie();
        for verb in AOKIE_ALLOWED {
            let read = AOKIE_READS.contains(&verb);
            assert_eq!(manifest.is_journalled(verb), !read, "{verb}");
        }
    }

    #[test]
    fn the_default_rule_alone_would_let_the_dangerous_verbs_through() {
        // Why Aokie has an entry at all. With no entry the plugin's own
        // `journalled` list is the only line, and Aokie does not journal these
        // eight, though each of them changes something.
        let manifest = aokie();
        let no_entry = RelayPolicy::from_text(Some("{\"version\":1,\"connectors\":{}}"));
        for verb in [
            "dongle.installDriver",
            "dongle.restoreDriver",
            "dongle.removeCerts",
            "dongle.setPreferred",
            "settings.set",
            "outbox.redrive",
            "consent.set",
            "consent.revoke",
        ] {
            assert_eq!(
                no_entry.check("aokie", verb, "cmd-1", || declared(&manifest, "aokie", verb)),
                Ok(()),
                "{verb}: if this is refused, the default rule got stricter and the entry can say so"
            );
            assert!(
                shipped().check("aokie", verb, "cmd-1", never).is_err(),
                "{verb} must be refused by Aokie's entry"
            );
        }
    }

    #[test]
    fn formlogics_own_private_commands_are_not_listed() {
        // FormLogic keeps these off its relay (DesktopCommandService.php:42-49). The
        // desktop does not rely on that: none of them is on the list.
        for verb in [
            "call.remoteStatus",
            "call.assistance.respond",
            "call.takeOver",
            "call.resumeBot",
            "call.endCaller",
            "call.declineWaiting",
            "remote.bootstrap",
            "remote.session.refresh",
        ] {
            assert!(shipped().check("aokie", verb, "cmd-1", never).is_err(), "{verb}");
        }
    }

    // ---- the rules ------------------------------------------------------------------

    #[test]
    fn a_listed_verb_is_allowed_and_an_unlisted_one_is_not() {
        let policy = one("printer", &["job.status", "job.cancel"]);
        assert_eq!(policy.check("printer", "job.status", "c1", never), Ok(()));
        assert_eq!(policy.check("printer", "job.cancel", "c1", never), Ok(()));
        assert_eq!(
            policy.check("printer", "job.purge", "c1", never),
            Err(Refusal::ThisComputerOnly {
                connector: "printer".into(),
                command: "job.purge".into(),
                why: Why::NotListed
            })
        );
    }

    #[test]
    fn a_plugin_cannot_widen_its_own_entry() {
        // The plugin is never asked (`never` panics if it is): however its manifest
        // declares a verb, and however it leaves it out of `journalled`, an entry is
        // the whole answer.
        let policy = one("printer", &["job.status"]);
        assert!(policy.check("printer", "job.purge", "c1", never).is_err());
        let generous = || Declared::Plugin { declares: true, journalled: false };
        assert!(policy.check("printer", "job.purge", "c1", generous).is_err());
    }

    #[test]
    fn a_connector_with_no_entry_gets_its_reads_and_nothing_that_is_journalled() {
        let policy = one("printer", &["job.status"]);
        let read = || Declared::Plugin { declares: true, journalled: false };
        let write = || Declared::Plugin { declares: true, journalled: true };
        assert_eq!(policy.check("scanner", "scan.status", "c1", read), Ok(()));
        assert_eq!(
            policy.check("scanner", "scan.start", "c1", write),
            Err(Refusal::ThisComputerOnly {
                connector: "scanner".into(),
                command: "scan.start".into(),
                why: Why::Journalled
            })
        );
    }

    #[test]
    fn a_connector_with_no_entry_gets_no_command_its_plugin_does_not_declare() {
        let policy = one("printer", &["job.status"]);
        let undeclared = || Declared::Plugin { declares: false, journalled: false };
        assert_eq!(
            policy.check("scanner", "scan.nothing", "c1", undeclared),
            Err(Refusal::NotOffered {
                connector: "scanner".into(),
                command: "scan.nothing".into(),
                plugin_installed: true
            })
        );
        assert_eq!(
            policy.check("scanner", "scan.status", "c1", || Declared::NoPlugin),
            Err(Refusal::NotOffered {
                connector: "scanner".into(),
                command: "scan.status".into(),
                plugin_installed: false
            })
        );
    }

    #[test]
    fn an_entry_with_nothing_listed_means_nothing_from_the_website() {
        let policy = one("printer", &[]);
        let read = || Declared::Plugin { declares: true, journalled: false };
        assert!(policy.check("printer", "job.status", "c1", read).is_err());
        // …and says nothing about any other connector.
        assert_eq!(policy.check("scanner", "scan.status", "c1", read), Ok(()));
    }

    #[test]
    fn a_name_that_is_not_exactly_a_listed_one_is_not_listed() {
        // No wildcard, no case folding, no trimming: the plugin's gate compares
        // exactly, so must this.
        let policy = one("aokie", &["call.answer"]);
        for verb in ["call.*", "call.", "Call.answer", "call.answer ", " call.answer", "call.answer\n", "call.answe"] {
            assert!(policy.check("aokie", verb, "c1", never).is_err(), "{verb:?}");
        }
        // A connector id that is not exactly `aokie` has no entry, so it is asked
        // of the plugin and, here, the plugin has nothing to say for it.
        for connector in ["Aokie", "AOKIE", "aokie ", " aokie", "aokie\n"] {
            let refusal = policy
                .check(connector, "call.answer", "c1", || Declared::NoPlugin)
                .expect_err(connector);
            assert_eq!(refusal.reason(), "not_declared", "{connector:?}");
        }
    }

    #[test]
    fn a_command_with_no_id_is_refused_whatever_it_is() {
        let policy = shipped();
        for id in ["", " ", "\t\n"] {
            assert_eq!(
                policy.check("aokie", "call.answer", id, never),
                Err(Refusal::NoCommandId { command: "call.answer".into() }),
                "{id:?}"
            );
        }
    }

    // ---- a policy that cannot be used ------------------------------------------------

    const UNUSABLE: [&str; 20] = [
        "",
        "   \n ",
        "not json at all",
        "null",
        "[]",
        "\"aokie\"",
        "{}",
        // A newer or older shape than this build reads.
        r#"{"version":2,"connectors":{}}"#,
        r#"{"version":0,"connectors":{}}"#,
        r#"{"connectors":{}}"#,
        // A field this build does not know: a misspelling is not a rule.
        r#"{"version":1,"connectors":{},"extra":1}"#,
        r#"{"version":1,"connectors":{"aokie":{"allow":[],"deny":["x"]}}}"#,
        // The wrong types.
        r#"{"version":1,"connectors":[]}"#,
        r#"{"version":1,"connectors":{"aokie":{"allow":"call.answer"}}}"#,
        r#"{"version":1,"connectors":{"aokie":{"allow":[7]}}}"#,
        // Names that are not plain.
        r#"{"version":1,"connectors":{"aokie":{"allow":["call.*"]}}}"#,
        r#"{"version":1,"connectors":{"aokie":{"allow":[""]}}}"#,
        r#"{"version":1,"connectors":{"":{"allow":[]}}}"#,
        // Twice, and an entry for what takes none.
        r#"{"version":1,"connectors":{"aokie":{"allow":["call.answer","call.answer"]}}}"#,
        r#"{"version":1,"connectors":{"desktop":{"allow":["services.list"]}}}"#,
    ];

    #[test]
    fn a_policy_that_cannot_be_used_allows_nothing_not_even_a_read() {
        // The way to be wrong is to refuse. An empty policy would be the wrong way
        // round: every connector would fall to the default rule, which allows reads.
        let read = || Declared::Plugin { declares: true, journalled: false };
        for text in UNUSABLE {
            let policy = RelayPolicy::from_text(Some(text));
            assert!(policy.closed_because().is_some(), "{text:?} was accepted");
            for connector in ["aokie", "printer", "anything"] {
                assert_eq!(
                    policy.check(connector, "call.current", "c1", read),
                    Err(Refusal::PolicyUnreadable { command: "call.current".into() }),
                    "{text:?} let {connector} through"
                );
            }
        }
    }

    #[test]
    fn a_policy_that_is_absent_allows_nothing() {
        let policy = RelayPolicy::from_text(None);
        assert!(policy.closed_because().is_some());
        assert_eq!(
            policy.check("aokie", "call.current", "c1", never),
            Err(Refusal::PolicyUnreadable { command: "call.current".into() })
        );
    }

    #[test]
    fn a_byte_order_mark_does_not_make_a_good_policy_unreadable() {
        // Windows editors write one; refusing everything for it would be a
        // failure of the wrong kind.
        let text = format!("\u{feff}{SHIPPED}");
        assert!(RelayPolicy::from_text(Some(&text)).closed_because().is_none());
    }

    // ---- what the person reads --------------------------------------------------------

    #[test]
    fn a_command_kept_on_the_computer_says_so_and_names_itself() {
        let refusal = shipped().check("aokie", "dongle.installDriver", "c1", never).unwrap_err();
        let message = refusal.message();
        assert!(message.contains("\"dongle.installDriver\""), "{message}");
        assert!(message.contains("\"aokie\""), "{message}");
        assert!(message.contains("can only be run from OAIY on this computer"), "{message}");
        assert!(message.contains("never reached the plugin"), "{message}");
        assert_eq!(refusal.code(), "capability_denied");
        assert_eq!(refusal.reason(), "not_listed");
    }

    #[test]
    fn a_command_no_plugin_offers_is_refused_in_the_gates_own_words() {
        let missing = Refusal::NotOffered {
            connector: "ghost".into(),
            command: "x.y".into(),
            plugin_installed: false,
        };
        let gate = GateRefusal::ConnectorMissing { connector_id: "ghost".into() };
        assert_eq!(missing.message(), gate.message());
        assert_eq!(missing.code(), gate.code());
        let undeclared = Refusal::NotOffered {
            connector: "aokie".into(),
            command: "x.y".into(),
            plugin_installed: true,
        };
        let gate = GateRefusal::CapabilityDenied { connector_id: "aokie".into(), command: "x.y".into() };
        assert_eq!(undeclared.message(), gate.message());
        assert_eq!(undeclared.code(), gate.code());
    }

    #[test]
    fn an_unusable_policy_and_a_missing_id_say_what_to_do() {
        let closed = Refusal::PolicyUnreadable { command: "call.answer".into() };
        assert!(closed.message().contains("\"call.answer\""));
        assert!(closed.message().contains("from OAIY on this computer"));
        assert_eq!(closed.code(), "capability_unavailable");
        let no_id = Refusal::NoCommandId { command: "call.answer".into() };
        assert!(no_id.message().contains("\"call.answer\""));
        assert_eq!(no_id.code(), "invalid_request");
    }
}
