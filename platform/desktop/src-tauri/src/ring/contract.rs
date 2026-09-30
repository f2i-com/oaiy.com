//! The `transfer_v1` contract as types: the frames on the call's stream and the requests between
//! the desktop and the phone plugin, as `docs/contracts/transfer/` says them.
//!
//! Nothing here runs a call. These types are what the fixtures are parsed with, so a change to
//! what is sent or read that the fixtures do not describe fails a test. The folder is the phone
//! plugin's own, copied byte for byte (its tests parse the same files with its types), and
//! `scripts/check-transfer-contract.mjs` compares the two.
//!
//! What this desktop reads from the plugin tolerates members it does not know (the plugin adds
//! members without a new version); what it writes has exactly the members the contract names.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::plan::{Decision, PlanReason, Reason};

/// The domain the reserved offer id is derived under.
pub const OFFER_DOMAIN: &str = "oaiy/transfer-offer/v1";

/// The id of the signed offer the plugin gives a device for a transfer request, and the id a ring hint would carry, so that a phone upgrades the
/// ring it already shows instead of ringing again. **This desktop posts no ring hint, launches no Companion and sends this to nobody**: the id is
/// computed here and held to the shared fixture (`reserved-offer-id`) so that the two programs agree on it, and nothing else uses it.
/// `"toffer_"` and the first 26 characters of the lower-case base32 (RFC 4648 alphabet, no padding)
/// of SHA-256 over the domain, the request id and the device's endpoint-key thumbprint, each
/// followed by a zero byte before the next; a retired offer takes the next `generation`, which
/// appends a zero byte and its decimal digits (generation 0 appends nothing).
pub fn offer_id(request_id: &str, holder_thumbprint: &str, generation: u32) -> String {
    let mut hash = Sha256::new();
    hash.update(OFFER_DOMAIN.as_bytes());
    hash.update([0]);
    hash.update(request_id.as_bytes());
    hash.update([0]);
    hash.update(holder_thumbprint.as_bytes());
    if generation > 0 {
        hash.update([0]);
        hash.update(generation.to_string().as_bytes());
    }
    format!("toffer_{}", &base32_lower(&hash.finalize())[..26])
}

/// Lower-case base32 (RFC 4648 alphabet), no padding.
fn base32_lower(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut out = String::with_capacity(bytes.len() * 8 / 5 + 1);
    let (mut bits, mut acc) = (0u32, 0u32);
    for byte in bytes {
        acc = (acc << 8) | u32::from(*byte);
        bits += 8;
        while bits >= 5 {
            out.push(char::from(ALPHABET[((acc >> (bits - 5)) & 31) as usize]));
            bits -= 5;
        }
    }
    if bits > 0 {
        out.push(char::from(ALPHABET[((acc << (5 - bits)) & 31) as usize]));
    }
    out
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// `formlogic.realtime.start`, as far as this contract goes: the members Aokie sends, and the two this contract adds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartFrame {
    #[serde(rename = "type")]
    pub kind: String,
    pub call_id: String,
    pub generation: u64,
    pub destination_origin: String,
    pub instructions: String,
    pub greeting: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub turn_detection: String,
    pub max_output_tokens: u32,
    pub input_format: String,
    pub output_format: String,
    pub sample_rate: u32,
    pub allow_business_lookup: bool,
    pub allow_request_appointment: bool,
    pub allow_finish_call: bool,
    /// New: the phone can put a caller through to the owner on this call (only ever true for an inbound call on this route).
    #[serde(default, skip_serializing_if = "is_false")]
    pub allow_transfer: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opening_line: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// New: the session that follows a handoff, for the same call id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume: Option<Resume>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResumeVia {
    /// The owner handed the call back.
    Return,
    /// The takeover failed, and the call is back with the receptionist.
    Failback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Resume {
    pub after_handoff: bool,
    pub handoff_seconds: u64,
    pub via: ResumeVia,
}

/// `formlogic.realtime.ready`: `features` lists `transfer_v1` when this desktop agrees to it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadyFrame {
    #[serde(rename = "type")]
    pub kind: String,
    pub call_id: String,
    pub generation: u64,
    pub destination_origin: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<String>,
}

/// `formlogic.realtime.tool_call`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    #[serde(rename = "type")]
    pub kind: String,
    pub call_id: String,
    pub generation: u64,
    pub tool_call_id: String,
    pub name: String,
    pub arguments: Value,
}

/// The arguments of `transfer_to_owner`: exactly `{reason}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferArguments {
    pub reason: Reason,
}

/// `formlogic.realtime.tool_result`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResult {
    #[serde(rename = "type")]
    pub kind: String,
    pub call_id: String,
    pub generation: u64,
    pub tool_call_id: String,
    pub name: String,
    pub ok: bool,
    pub output: Value,
    pub continue_response: bool,
}

/// What a transfer's tool result carries when the owner is being rung.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RingingOutput {
    pub status: RingingStatus,
    pub request_id: String,
    pub ring_seconds: u64,
    pub instruction: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RingingStatus {
    Ringing,
}

/// What it carries when the request is not made: `refused` (do not offer a person at all) or `unavailable` (offer a message).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefusedOutput {
    pub status: RefusedStatus,
    /// A plan reason, or one of the plugin's (`consent`, `pending_request`, `bad_arguments`, `plan_unavailable`,
    /// `call_changed`, ...), or one of this desktop's own (`not_offered`, `tool_limit`, `no_answer`).
    pub reason: String,
    pub instruction: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusedStatus {
    Refused,
    Unavailable,
}

/// `formlogic.realtime.transfer_outcome`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutcomeFrame {
    #[serde(rename = "type")]
    pub kind: String,
    pub call_id: String,
    pub generation: u64,
    pub request_id: String,
    pub outcome: crate::voice::transfer::Outcome,
    /// The owner's words for the caller (only with `declined`): untrusted text of at most 320 characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Unix epoch milliseconds.
    pub at_ms: u64,
}

/// `formlogic.realtime.transfer_cancel`: this desktop withdraws a request that rings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelFrame {
    #[serde(rename = "type")]
    pub kind: String,
    pub call_id: String,
    pub generation: u64,
    pub request_id: String,
    pub reason: crate::voice::transfer::CancelReason,
}

/// What a `formlogic.realtime.transfer_notice` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoticeKind {
    /// An owner device has already won the request: the takeover goes on.
    TooLate,
    /// The call has no open request with that id: nothing was changed.
    UnknownRequest,
}

impl NoticeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NoticeKind::TooLate => crate::voice::transfer::TOO_LATE,
            NoticeKind::UnknownRequest => crate::voice::transfer::UNKNOWN_REQUEST,
        }
    }
}

/// `formlogic.realtime.transfer_notice`: the plugin's answer to a withdrawal that changed nothing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NoticeFrame {
    #[serde(rename = "type")]
    pub kind: String,
    pub call_id: String,
    pub generation: u64,
    pub request_id: String,
    pub notice: NoticeKind,
    /// Unix epoch milliseconds.
    pub at_ms: u64,
}

/// `formlogic.realtime.stop`: a reason starting `handoff:` hands the call to the owner (the session ends, the call does not).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StopFrame {
    #[serde(rename = "type")]
    pub kind: String,
    pub call_id: String,
    pub generation: u64,
    pub reason: String,
}

/// The params of `oaiy.ring.plan`, the plugin asking the desktop who may be rung.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanParams {
    pub call_id: String,
    pub call_epoch: u64,
    pub owner_epoch: u64,
    pub reason: Reason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller_number: Option<String>,
    /// The caller's last turns, at most three of at most 300 characters. The desktop reads its own record of the call first.
    pub recent_caller_turns: Vec<String>,
}

/// The desktop's answer to `oaiy.ring.plan`. Written in full; read as the plugin reads it, where a member the answer leaves out
/// (a plan that does not ring names nobody) is empty or false.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanResult {
    /// Always a valid id, even when nobody is rung: the plugin loses the reason of a refusal it cannot tell apart.
    #[serde(default)]
    pub plan_id: String,
    pub decision: Decision,
    pub reason: PlanReason,
    #[serde(default)]
    pub ring_seconds: u32,
    #[serde(default)]
    pub phones: Vec<String>,
    #[serde(default)]
    pub wake: Vec<String>,
    #[serde(default)]
    pub desktop_toast: bool,
    #[serde(default)]
    pub desktop_companions: Vec<String>,
    /// This desktop itself vouches for the request's reason, so the plugin need not see the caller ask for a person. True only
    /// for `urgent`, when the owner allows it and one of their urgent phrases was heard; otherwise false.
    #[serde(default)]
    pub reason_allowed: bool,
}

/// The params of `oaiy.ring.opened`, the plugin telling the desktop the request is out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenedParams {
    pub plan_id: String,
    pub request_id: String,
    pub call_id: String,
    pub call_epoch: u64,
    pub owner_epoch: u64,
    /// Unix seconds.
    pub expires_at: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::transfer::{self, CancelReason, Outcome};
    use serde::de::DeserializeOwned;
    use serde_json::json;
    use std::collections::{BTreeMap, BTreeSet};

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../docs/contracts/transfer");

    /// The shared fixtures, by the name between `transfer-v1.` and `.fixture.json`: every one is handled by a test below, and a
    /// fixture that is in the folder and not here fails `every_fixture_in_the_folder_is_one_these_tests_handle`.
    const FIXTURES: [&str; 8] = ["caller-asked", "cancel", "outcome", "reserved-offer-id", "ring-plan", "start-ready", "tool-call", "tool-result"];

    fn read(name: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(DIR).join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    fn fixture(kind: &str) -> Value {
        let value: Value = serde_json::from_str(&read(&format!("transfer-v1.{kind}.fixture.json"))).unwrap_or_else(|e| panic!("{kind}: {e}"));
        assert_eq!(value["contract"], "transfer_v1", "{kind}");
        // The kind is the file's name in words (`transfer_cancel` for `cancel`, `tool_call` for `tool-call`).
        assert!(value["kind"].as_str().is_some_and(|k| k.ends_with(&kind.replace('-', "_"))), "{kind}: {}", value["kind"]);
        value
    }

    fn typed<T: DeserializeOwned>(what: &str, value: &Value) -> T {
        serde_json::from_value(value.clone()).unwrap_or_else(|e| panic!("{what}: {e}: {value}"))
    }

    fn strings(value: &Value) -> Vec<String> {
        value.as_array().unwrap_or_else(|| panic!("not a list: {value}")).iter().map(|v| v.as_str().unwrap_or_else(|| panic!("not a string: {v}")).to_string()).collect()
    }

    /// What the plugin does with a plan answer, as the fixture states it: a plan that rings needs an id it can use, every device is
    /// an identifier, and there are at most sixteen. (What it does with the reason is `plan_reason_is_known`.)
    fn plugin_can_use(result: &Value) -> bool {
        fn token(s: &str) -> bool {
            !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
        }
        let Some(plan) = result.as_object() else { return false };
        let decision = plan.get("decision").and_then(Value::as_str);
        if !matches!(decision, Some("ring" | "message_only" | "refused")) {
            return false;
        }
        if decision == Some("ring") && !plan.get("planId").and_then(Value::as_str).is_some_and(token) {
            return false;
        }
        let mut devices = 0;
        for key in ["phones", "desktopCompanions"] {
            match plan.get(key) {
                None => {}
                Some(Value::Array(list)) => {
                    for device in list {
                        if !device.as_str().is_some_and(token) {
                            return false;
                        }
                        devices += 1;
                    }
                }
                Some(_) => return false,
            }
        }
        devices <= 16
    }

    /// The reasons a plan may give: the closed set of the tool-result fixture.
    fn plan_reasons() -> BTreeSet<String> {
        strings(&fixture("tool-result")["planReasons"]).into_iter().collect()
    }

    /// The digest the check script and the phone's own check use: SHA-256 with CRLF folded to LF, so a checkout that adds carriage
    /// returns and the committed blob agree.
    fn digest(bytes: &[u8]) -> String {
        let mut folded = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
                i += 1;
                continue;
            }
            folded.push(bytes[i]);
            i += 1;
        }
        Sha256::digest(&folded).iter().map(|b| format!("{b:02x}")).collect()
    }

    /// `SHA256SUMS` in `dir` lists exactly the `.json` files there, each with its digest.
    fn check_sums(dir: &str) {
        let folder = std::path::Path::new(DIR).join(dir);
        let sums = std::fs::read_to_string(folder.join("SHA256SUMS")).unwrap_or_else(|e| panic!("{dir}/SHA256SUMS: {e}"));
        let mut listed = BTreeMap::new();
        for line in sums.lines() {
            let (sum, name) = line.split_once("  ").expect("a digest, two spaces, a name");
            assert_eq!(sum.len(), 64, "{line}");
            assert!(listed.insert(name.to_string(), sum.to_string()).is_none(), "{name} twice");
        }
        let mut on_disk: Vec<String> = std::fs::read_dir(&folder).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".json")).collect();
        on_disk.sort();
        assert_eq!(listed.keys().cloned().collect::<Vec<_>>(), on_disk, "{dir}/SHA256SUMS lists exactly the fixtures");
        for (name, sum) in &listed {
            assert_eq!(&digest(&std::fs::read(folder.join(name)).unwrap()), sum, "{dir}/{name}");
        }
    }

    #[test]
    fn every_fixture_in_the_folder_is_one_these_tests_handle_and_the_contract_names_them_all() {
        let mut on_disk: Vec<String> = std::fs::read_dir(DIR).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".json")).collect();
        on_disk.sort();
        let handled: Vec<String> = FIXTURES.iter().map(|k| format!("transfer-v1.{k}.fixture.json")).collect();
        assert_eq!(on_disk, handled, "a fixture was added or removed: give it a test, and add it to FIXTURES");
        // The prose contract lists every one of them.
        let doc = read("transfer-v1.md");
        for name in &handled {
            assert!(doc.contains(&format!("[{name}]({name})")), "transfer-v1.md does not name {name}");
        }
        // And each is its own kind, once.
        let kinds: BTreeSet<String> = FIXTURES.iter().map(|k| fixture(k)["kind"].as_str().unwrap().to_string()).collect();
        assert_eq!(kinds.len(), FIXTURES.len());
    }

    #[test]
    fn the_sums_list_every_fixture_and_each_sum_is_right() {
        check_sums("");
        check_sums("oaiy-only");
    }

    #[test]
    fn the_folder_is_lf_only_so_its_digests_are_the_same_on_every_checkout() {
        for dir in ["", "oaiy-only"] {
            let folder = std::path::Path::new(DIR).join(dir);
            let mut checked = 0;
            for entry in std::fs::read_dir(&folder).unwrap().flatten().filter(|e| e.path().is_file()) {
                let bytes = std::fs::read(entry.path()).unwrap();
                assert!(!bytes.contains(&b'\r'), "{:?} has a carriage return: .gitattributes keeps this folder LF", entry.file_name());
                checked += 1;
            }
            assert!(checked >= 4, "{dir}: {checked} files");
        }
    }

    #[test]
    fn the_reserved_offer_ids_of_both_test_phones_are_what_this_computes() {
        let f = fixture("reserved-offer-id");
        assert_eq!(f["domain"], OFFER_DOMAIN);
        assert_eq!(f["prefix"], "toffer_");
        assert_eq!(f["digestChars"], 26);
        assert_eq!(f["idLength"], 33);
        let alphabet = f["idAlphabet"].as_str().unwrap();
        let request = f["requestId"].as_str().unwrap();
        let phones = f["testPhones"].as_array().unwrap();
        assert_eq!(phones.len(), 2);
        let mut all = BTreeSet::new();
        for phone in phones {
            let thumbprint = phone["holderThumbprint"].as_str().unwrap();
            let ids = strings(&phone["offerIds"]);
            assert_eq!(ids.len(), 3, "generations 0, 1 and 2");
            for (generation, expected) in ids.iter().enumerate() {
                let id = offer_id(request, thumbprint, generation as u32);
                assert_eq!(&id, expected, "{} generation {generation}", phone["name"]);
                assert_eq!(id.len(), 33);
                assert!(id.strip_prefix("toffer_").unwrap().chars().all(|c| alphabet.contains(c)), "{id}");
                assert!(all.insert(id), "every offer id is its own");
            }
        }
        // Ids that are not reserved ones are not what this computes.
        for other in strings(&f["notReservedIds"]) {
            assert!(!all.contains(&other), "{other}");
        }
        // A different request, holder or generation is a different offer, and the parts cannot run into one another.
        let thumbprint = phones[0]["holderThumbprint"].as_str().unwrap();
        let base = offer_id(request, thumbprint, 0);
        assert_ne!(base, offer_id("assist_other", thumbprint, 0));
        assert_ne!(base, offer_id(request, phones[1]["holderThumbprint"].as_str().unwrap(), 0));
        assert_ne!(base, offer_id(request, thumbprint, 1));
        assert_ne!(offer_id("a", "bc", 0), offer_id("ab", "c", 0));
    }

    #[test]
    fn the_thumbprints_this_desktop_derives_for_the_fixtures_keys_are_the_fixtures() {
        // A plan names devices by endpoint-key thumbprint, so the thumbprint this desktop derives for a device's key must be the one the plugin
        // derives: the fixture's test phones are Ed25519 keys from a seed of one repeated byte, with the public key and thumbprint written down.
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        let f = fixture("reserved-offer-id");
        let seed_byte = regex::Regex::new(r"0x([0-9a-f]{2})").unwrap();
        for phone in f["testPhones"].as_array().unwrap() {
            let name = phone["name"].as_str().unwrap();
            let seed = u8::from_str_radix(&seed_byte.captures(name).unwrap_or_else(|| panic!("{name}"))[1], 16).unwrap();
            let public = ed25519_dalek::SigningKey::from_bytes(&[seed; 32]).verifying_key().to_bytes();
            let hex: String = public.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, phone["publicKeyHex"].as_str().unwrap(), "{name}: the key of that seed");
            let thumbprint = crate::companion::identity::thumbprint_for(&URL_SAFE_NO_PAD.encode(public));
            assert_eq!(thumbprint, phone["holderThumbprint"].as_str().unwrap(), "{name}");
        }
    }

    #[test]
    fn base32_is_the_lower_case_rfc_4648_alphabet_without_padding() {
        assert_eq!(base32_lower(b""), "");
        assert_eq!(base32_lower(b"f"), "my");
        assert_eq!(base32_lower(b"fo"), "mzxq");
        assert_eq!(base32_lower(b"foo"), "mzxw6");
        assert_eq!(base32_lower(b"foob"), "mzxw6yq");
        assert_eq!(base32_lower(b"fooba"), "mzxw6ytb");
        assert_eq!(base32_lower(b"foobar"), "mzxw6ytboi");
    }

    #[test]
    fn the_starts_the_phone_sends_and_the_ready_this_desktop_answers_are_the_fixtures() {
        let f = fixture("start-ready");
        // A whole start as this desktop's types read it (the fixture names only what the contract adds).
        let whole: Value = serde_json::from_str(&read("oaiy-only/start-allow-transfer.json")).unwrap();
        for case in f["start"]["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let mut start = whole.clone();
            for member in ["allowTransfer", "resume"] {
                start.as_object_mut().unwrap().remove(member);
            }
            // What the case says the start has (its `allowTransfer` is the case's own, and `startHas` names the rest).
            if case["allowTransfer"].as_bool().unwrap() {
                start["allowTransfer"] = json!(true);
            }
            for (member, value) in case.get("startHas").and_then(Value::as_object).into_iter().flatten() {
                start[member] = value.clone();
            }
            for member in case.get("startOmits").map(strings).unwrap_or_default() {
                assert!(start.get(&member).is_none(), "{name}: {member}");
            }
            let parsed: StartFrame = typed(name, &start);
            assert_eq!(parsed.allow_transfer, case["allowTransfer"].as_bool().unwrap(), "{name}");
            match case.get("resume") {
                Some(resume) => {
                    let resume: Resume = typed(name, resume);
                    assert_eq!(parsed.resume, Some(resume), "{name}");
                    assert!(resume.after_handoff && resume.handoff_seconds > 0, "{name}");
                }
                None => assert!(parsed.resume.is_none(), "{name}"),
            }
            // A start that offers nothing is written as it always was: neither member is there.
            if !parsed.allow_transfer {
                let written = serde_json::to_value(&parsed).unwrap();
                assert!(written.get("allowTransfer").is_none() && written.get("resume").is_none(), "{name}");
            }
        }
        // Both ways to come back are known.
        let vias: BTreeSet<String> = f["start"]["cases"].as_array().unwrap().iter().filter_map(|c| c["resume"]["via"].as_str().map(String::from)).collect();
        assert_eq!(vias, BTreeSet::from(["return".to_string(), "failback".to_string()]));
        assert_eq!(serde_json::to_value(ResumeVia::Return).unwrap(), "return");
        assert_eq!(serde_json::to_value(ResumeVia::Failback).unwrap(), "failback");

        // ready: this desktop lists transfer_v1 exactly when the fixture says a plugin negotiates it.
        for case in f["ready"]["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let ready: ReadyFrame = typed(name, &case["frame"]);
            assert_eq!(ready.features, strings(&case["features"]), "{name}");
            assert_eq!(ready.features.iter().any(|x| x == transfer::FEATURE), case["negotiatedWhenOffered"].as_bool().unwrap(), "{name}");
        }
        let ours = ReadyFrame { kind: "formlogic.realtime.ready".into(), call_id: "call_0123".into(), generation: 7, destination_origin: crate::voice::DESTINATION.into(), features: vec![transfer::FEATURE.into()] };
        assert_eq!(serde_json::to_value(&ours).unwrap(), f["ready"]["cases"][0]["frame"], "the ready this desktop writes is the first case");
        assert!(serde_json::to_value(ReadyFrame { features: vec![], ..ours }).unwrap().get("features").is_none(), "an older desktop's ready has no features member: this is the second case's shape");

        // stop: a handoff names itself, and the old free text does not.
        let stop = &f["stop"];
        assert_eq!(stop["handoffPrefix"], transfer::HANDOFF_PREFIX);
        let frame = StopFrame { kind: "formlogic.realtime.stop".into(), call_id: "call_0123".into(), generation: 7, reason: stop["handoffReason"].as_str().unwrap().into() };
        assert!(frame.reason.starts_with(transfer::HANDOFF_PREFIX));
        assert!(!stop["legacyReason"].as_str().unwrap().starts_with(transfer::HANDOFF_PREFIX));
        assert_eq!(typed::<StopFrame>("stop", &serde_json::to_value(&frame).unwrap()), frame);
        // The whole-frame fixtures of this desktop's own read back as themselves.
        let resume_start: StartFrame = typed("start-resume", &serde_json::from_str::<Value>(&read("oaiy-only/start-resume.json")).unwrap());
        assert_eq!(resume_start.resume, Some(Resume { after_handoff: true, handoff_seconds: 42, via: ResumeVia::Return }));
        assert_eq!(resume_start.generation, 2);
    }

    #[test]
    fn the_tool_call_is_exactly_a_reason_and_this_desktop_is_never_looser_than_the_fixture() {
        let f = fixture("tool-call");
        let call: ToolCall = typed("frame", &f["frame"]);
        assert_eq!(call.name, transfer::TOOL);
        assert_eq!(f["toolName"], transfer::TOOL);
        // Every tool name this desktop puts on the wire is one the plugin's rule accepts, and none of the invalid ones is.
        let rule = regex::Regex::new(f["toolNameRule"].as_str().unwrap()).unwrap();
        for name in [transfer::TOOL, "lookup_business_data", "request_appointment", "finish_call"] {
            assert!(rule.is_match(name), "{name}");
        }
        for name in strings(&f["validToolNames"]) {
            assert!(rule.is_match(&name), "{name:?}");
        }
        for name in strings(&f["invalidToolNames"]) {
            assert!(!rule.is_match(&name), "{name:?}");
        }
        // The arguments: what the plugin refuses this desktop never sends, and it sends only what it accepts, and no `policy_rule`
        // (the owner's own rule, which a model may not claim: this desktop is stricter here on purpose).
        let cases = f["arguments"]["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 14);
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let arguments = &case["arguments"];
            let accepted = case["accepted"].as_bool().unwrap();
            let ours = transfer::parse_arguments(arguments);
            match (accepted, case["reason"].as_str()) {
                (false, _) => {
                    assert!(ours.is_err(), "{name}: the plugin refuses it and this desktop must too");
                    assert!(serde_json::from_value::<TransferArguments>(arguments.clone()).is_err(), "{name}");
                }
                (true, Some("policy_rule")) => {
                    assert!(ours.is_err(), "{name}: not for a model to claim");
                    assert_eq!(typed::<TransferArguments>(name, arguments).reason, Reason::PolicyRule);
                }
                (true, Some(reason)) => {
                    assert_eq!(ours.unwrap().as_str(), reason, "{name}");
                    assert_eq!(typed::<TransferArguments>(name, arguments).reason.as_str(), reason, "{name}");
                }
                (true, None) => panic!("{name}: an accepted case names its reason"),
            }
        }
    }

    #[test]
    fn the_tool_results_say_ringing_or_why_not_in_words_this_desktop_knows() {
        let f = fixture("tool-result");
        let result: ToolResult = typed("frame", &f["frame"]);
        assert!(result.ok && result.continue_response);
        let out: RingingOutput = typed("ringing", &result.output);
        assert_eq!((out.ring_seconds, out.request_id.as_str()), (f["ringing"]["ringSeconds"].as_u64().unwrap(), f["ringing"]["requestId"].as_str().unwrap()));
        // The plan reasons are exactly the reasons a plan of this desktop can give, and no other.
        let ours: BTreeSet<String> = [
            PlanReason::Disabled,
            PlanReason::InitiativeOff,
            PlanReason::NotUrgent,
            PlanReason::CallerDidNotAsk,
            PlanReason::LimitCall,
            PlanReason::LimitGap,
            PlanReason::LimitCaller,
            PlanReason::LimitGlobal,
            PlanReason::QuietHours,
            PlanReason::AllDoNotDisturb,
            PlanReason::NoEndpoint,
        ]
        .map(|r| r.as_str().to_string())
        .into();
        assert_eq!(ours, plan_reasons(), "a plan reason the plugin does not know becomes plan_unavailable, so this desktop gives none but these");
        assert_eq!(PlanReason::Ok.as_str(), "ok");
        // Every refusal reads as a refusal, and this desktop refuses in the same shape with the same status and reason.
        let plugin_reasons = strings(&f["pluginReasons"]);
        let mut statuses = BTreeSet::new();
        for refusal in f["refusals"].as_array().unwrap() {
            let (status, reason) = (refusal["status"].as_str().unwrap(), refusal["reason"].as_str().unwrap());
            assert!(plan_reasons().contains(reason) || plugin_reasons.iter().any(|r| r == reason), "{reason}");
            let read: RefusedOutput = typed(reason, &json!({"status": status, "reason": reason, "instruction": "x", "somethingNew": 1}));
            assert_eq!((read.status, read.reason.as_str()), (if status == "refused" { RefusedStatus::Refused } else { RefusedStatus::Unavailable }, reason));
            let mine = transfer::refused(status, reason);
            assert_eq!((mine["ok"].clone(), mine["output"]["status"].clone(), mine["output"]["reason"].clone()), (json!(false), json!(status), json!(reason)));
            assert!(typed::<RefusedOutput>(reason, &mine["output"]).instruction.len() > 20, "{reason}: the model is told what to do");
            statuses.insert(status.to_string());
        }
        assert_eq!(statuses, BTreeSet::from(["refused".to_string(), "unavailable".to_string()]));
        // A reason this desktop's own plans give is refused or unavailable exactly as the plugin says it is: `refused` (do not offer a person)
        // for a decision of `refused`, `unavailable` (offer a message) for `message_only`. The plugin's table is the fixture's, and a plan
        // of the 33 vectors that gives a reason must land in the same row.
        let table: BTreeMap<String, String> = f["refusals"].as_array().unwrap().iter().map(|r| (r["reason"].as_str().unwrap().to_string(), r["status"].as_str().unwrap().to_string())).collect();
        let mut checked = BTreeSet::new();
        for (id, plan) in super::super::tests::vector_plans() {
            let (plan, _) = super::super::host::name_somebody(plan);
            if plan.rings() {
                continue;
            }
            let ours = if plan.decision == Decision::Refused { "refused" } else { "unavailable" };
            assert_eq!(table.get(plan.reason.as_str()).map(String::as_str), Some(ours), "{id}: {}", plan.reason.as_str());
            checked.insert(plan.reason.as_str());
        }
        assert!(checked.len() >= 5, "{checked:?}");
        // The tool intake errors carry no status: they are not a refusal of a transfer.
        for case in f["toolRefusals"]["cases"].as_array().unwrap() {
            assert!(case["output"]["error"].is_string() && case["output"].get("status").is_none(), "{}", case["name"]);
            assert!(serde_json::from_value::<RefusedOutput>(case["output"].clone()).is_err());
        }
        // The words this desktop adds are its own: none is the plugin's, and each says unavailable (offer a message).
        for own in ["not_offered", "tool_limit", "no_answer"] {
            assert!(!plan_reasons().contains(own) && !plugin_reasons.iter().any(|r| r == own), "{own}");
            let mine = transfer::refused("unavailable", own);
            assert_eq!((mine["output"]["status"].clone(), mine["output"]["reason"].clone()), (json!("unavailable"), json!(own)));
        }
    }

    #[test]
    fn every_outcome_is_read_by_this_desktop_and_the_owners_words_come_only_with_a_decline() {
        let f = fixture("outcome");
        assert_eq!(strings(&f["outcomes"]), [Outcome::Accepted, Outcome::Declined, Outcome::Unavailable, Outcome::Expired, Outcome::Cancelled].map(|o| o.as_str().to_string()));
        assert_eq!(f["maxMessageChars"], transfer::MAX_OWNER_MESSAGE);
        let mut seen = BTreeSet::new();
        for case in f["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let frame: OutcomeFrame = typed(name, &case["frame"]);
            assert_eq!(frame.kind, "formlogic.realtime.transfer_outcome", "{name}");
            assert_eq!(Outcome::parse(frame.outcome.as_str()), Some(frame.outcome), "{name}");
            assert!(frame.at_ms > 1_000_000_000_000, "{name}: atMs is Unix epoch milliseconds");
            if frame.outcome != Outcome::Declined {
                assert!(frame.message.is_none(), "{name}: the owner's words come with a decline only");
            }
            if let Some(words) = &frame.message {
                assert!(words.chars().count() <= transfer::MAX_OWNER_MESSAGE, "{name}");
                assert_eq!(transfer::owner_message(words).as_deref(), Some(words.as_str()), "{name}: plain words are kept as they are");
            }
            seen.insert(frame.outcome.as_str());
        }
        assert_eq!(seen.len(), 5, "a case for every outcome");
        // The takeover has the plugin's setup time and the grace it takes to record the result before this desktop gives up on it.
        let t = &f["timings"];
        assert_eq!(transfer::SETUP_LIMIT.as_secs(), t["mediaSetupSeconds"].as_u64().unwrap() + t["resolutionGraceSeconds"].as_u64().unwrap());
    }

    #[test]
    fn the_withdrawal_and_its_notices_are_the_fixtures_and_this_desktop_says_every_reason() {
        let f = fixture("cancel");
        let cancel = &f["cancel"];
        assert_eq!(cancel["frameType"], transfer::CANCEL_FRAME);
        let ours = [CancelReason::OwnerDeclined, CancelReason::MessageInstead, CancelReason::GaveUp];
        assert_eq!(strings(&cancel["reasons"]), ours.map(|r| r.as_str().to_string()));
        let mut seen = BTreeSet::new();
        for case in cancel["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let frame: CancelFrame = typed(name, &case["frame"]);
            assert_eq!(frame.kind, transfer::CANCEL_FRAME, "{name}");
            assert!(seen.insert(frame.reason.as_str()), "{name}");
            // What this desktop writes for it is this frame, member for member.
            assert_eq!(serde_json::to_value(&frame).unwrap(), case["frame"], "{name}");
        }
        assert_eq!(seen.len(), ours.len(), "a case for every reason");
        // What the plugin ignores this desktop does not send: a request id it can use, and a reason of the set.
        for case in cancel["ignored"]["cases"].as_array().unwrap() {
            assert!(serde_json::from_value::<CancelFrame>(case["frame"].clone()).is_err() || case["frame"]["requestId"].as_str().is_some_and(|id| id.is_empty() || id.len() > 128 || id.contains(|c: char| !(c.is_ascii_alphanumeric() || "-_.:".contains(c)))), "{}: not a frame this desktop's types would write", case["name"]);
        }
        // The notices.
        let notice = &f["notice"];
        assert_eq!(notice["frameType"], transfer::NOTICE_FRAME);
        assert_eq!(strings(&notice["notices"]), [NoticeKind::TooLate, NoticeKind::UnknownRequest].map(|n| n.as_str().to_string()));
        let mut kinds = BTreeSet::new();
        for case in notice["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let frame: NoticeFrame = typed(name, &case["frame"]);
            assert_eq!(frame.kind, transfer::NOTICE_FRAME, "{name}");
            assert!(frame.at_ms > 1_000_000_000_000, "{name}: atMs is Unix epoch milliseconds");
            kinds.insert(frame.notice.as_str());
        }
        assert_eq!(kinds.len(), 2, "a case for each notice");
    }

    #[test]
    fn the_plan_the_plugin_asks_for_and_the_answers_are_the_fixtures() {
        let f = fixture("ring-plan");
        // The host features: this desktop announces ringPlan, exactly as the fixture's host that answers the two requests does.
        let mut announced = false;
        for case in f["init"]["cases"].as_array().unwrap() {
            // A member that is not a string announces nothing (one case has a number in the list).
            let features: Vec<String> = case["params"].get("features").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect()).unwrap_or_default();
            assert_eq!(features.iter().any(|x| x == "ringPlan"), case["ringPlan"].as_bool().unwrap(), "{}", case["name"]);
            if features == crate::plugins::process::HOST_FEATURES {
                announced = true;
                assert_eq!(case["ringPlan"], true);
            }
        }
        assert!(announced, "the fixture has this desktop's own plugin.init features");

        // The question: at most the last three caller turns, each cut to 300 characters, as this desktop cuts what it is told.
        let plan = &f["plan"];
        assert_eq!(plan["method"], "oaiy.ring.plan");
        let params: PlanParams = typed("plan params", &plan["params"]);
        assert_eq!(params.reason, Reason::CallerAsked);
        assert_eq!(super::super::phrases::recent(&strings(&plan["input"]["recentCallerTurns"])), params.recent_caller_turns, "the turns kept are the last three");
        assert!(params.recent_caller_turns.len() <= 3 && params.recent_caller_turns.iter().all(|t| t.chars().count() <= 300));

        // The answers: every one reads, and says what the fixture says the plugin makes of it.
        let known = plan_reasons();
        for case in plan["results"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let (result, parsed) = (&case["result"], &case["parsed"]);
            assert!(plugin_can_use(result), "{name}");
            let read: PlanResult = typed(name, result);
            assert_eq!(serde_json::to_value(read.decision).unwrap(), parsed["decision"], "{name}");
            assert_eq!(read.reason_allowed, parsed["reasonAllowed"].as_bool().unwrap_or(false), "{name}");
            let planned = super::super::RingPlan { decision: read.decision, reason: read.reason, ring_seconds: read.ring_seconds, phones: read.phones, wake: read.wake, desktop_toast: read.desktop_toast, desktop_companions: read.desktop_companions };
            assert_eq!(planned.targets(), strings(&parsed["targets"]), "{name}: each device once, phones first");
            if read.decision != Decision::Ring {
                assert_eq!(parsed["reason"], read.reason.as_str(), "{name}");
                assert!(known.contains(read.reason.as_str()), "{name}");
                assert!(planned.targets().is_empty(), "{name}");
            } else {
                // A ring that names nobody is what this desktop never plans (it is a message_only / no_endpoint).
                assert_eq!(parsed["targetRule"], if planned.targets().is_empty() { "nobody" } else { "only" }, "{name}");
            }
        }
        // What the plugin cannot use, this desktop's own plans never are (below), and the fixture says so.
        for case in plan["unusableResults"].as_array().unwrap() {
            assert!(!plugin_can_use(&case["result"]), "{}", case["name"]);
        }
        // The other request.
        let opened = &f["opened"];
        assert_eq!(opened["method"], "oaiy.ring.opened");
        let params: OpenedParams = typed("opened", &opened["input"]);
        assert!(params.expires_at > 1_000_000_000 && params.expires_at < 100_000_000_000, "expiresAt is Unix seconds");
        assert_eq!(opened["result"], json!({"ok": true}));
    }

    #[test]
    fn every_plan_this_desktop_makes_is_one_the_plugin_can_use_with_a_reason_it_knows() {
        let known = plan_reasons();
        let mut rings = 0;
        let vectors = super::super::tests::vector_plans();
        assert_eq!(vectors.len(), 33);
        for (id, plan) in vectors {
            // The plan as this desktop answers with it (the reference's V01 rings only the toast, which is not somebody).
            let (plan, _) = super::super::host::name_somebody(plan);
            let result = super::super::host::plan_result(&super::super::Authorised { plan: plan.clone(), plan_id: "plan_0123456789abcdef".into(), reason_allowed: false, caller_number: String::new(), caller_name: String::new() });
            assert!(plugin_can_use(&result), "{id}: {result}");
            let read: PlanResult = typed(&id, &result);
            if plan.rings() {
                rings += 1;
                assert_eq!(result["reason"], "ok", "{id}");
                assert!(!plan.targets().is_empty(), "{id}: a plan that rings names somebody");
                assert!((20..=90).contains(&read.ring_seconds), "{id}: {}", read.ring_seconds);
            } else {
                assert!(known.contains(read.reason.as_str()), "{id}: {} is not a reason the plugin knows", read.reason.as_str());
            }
        }
        assert!((5..33).contains(&rings), "the vectors ring in {rings} cases and refuse in the rest");
        // And the plan of a call that has no record here, and one whose reason this desktop cannot say, is still a plan with an id.
        let unknown = super::super::host::plan_result(&super::super::Authorised {
            plan: super::super::RingPlan::refuse(PlanReason::CallerDidNotAsk, Decision::Refused),
            plan_id: "plan_x".into(),
            reason_allowed: false,
            caller_number: String::new(),
            caller_name: String::new(),
        });
        assert!(plugin_can_use(&unknown) && known.contains(unknown["reason"].as_str().unwrap()));
    }

    #[test]
    fn the_plan_this_desktop_writes_has_every_member_of_the_fixtures_ring_answer() {
        let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<BTreeSet<_>>();
        let f = fixture("ring-plan");
        let fixture_plan = &f["plan"]["results"][1]["result"];
        let ours = super::super::host::plan_result(&super::super::Authorised {
            plan: super::super::RingPlan { decision: Decision::Ring, reason: PlanReason::Ok, ring_seconds: 30, phones: vec![], wake: vec![], desktop_toast: true, desktop_companions: vec!["thumb_windows".into()] },
            plan_id: "plan_x".into(),
            reason_allowed: false,
            caller_number: String::new(),
            caller_name: String::new(),
        });
        // Everything but the member the fixture adds to show the plugin ignores what it does not know.
        let mut expected = keys(fixture_plan);
        expected.remove("somethingNew");
        expected.insert("reasonAllowed".into());
        assert_eq!(keys(&ours), expected);
        let read: PlanResult = typed("ours", &ours);
        assert_eq!((read.decision, read.plan_id.as_str(), read.reason_allowed), (Decision::Ring, "plan_x", false));
    }
}
