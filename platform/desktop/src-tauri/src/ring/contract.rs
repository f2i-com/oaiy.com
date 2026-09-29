//! The `transfer_v1` contract as types: the frames on the call's stream and the requests between
//! the desktop and the phone plugin, exactly as `docs/contracts/transfer/` says them.
//!
//! Nothing here runs a call. These types are what the fixtures are parsed with, so a change to
//! what is sent or read that the fixtures do not describe fails a test; the Aokie repository has
//! the same fixtures and a test of its own, and the two are compared by `SHA256SUMS`.
//!
//! The fixtures are canonical JSON: keys sorted at every level, no whitespace, one trailing line
//! feed ([`canonical`]).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::plan::{Decision, PlanReason, Reason};

/// The domain the reserved offer id is derived under.
pub const OFFER_DOMAIN: &str = "oaiy/transfer-offer/v1";

/// The id of the signed offer the plugin gives a device for a transfer request, and the id the ring
/// hint carries, so a phone upgrades the ring it already shows instead of ringing again:
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

/// `value` as canonical JSON: keys sorted at every level, no whitespace (a fixture is this and one line feed).
pub fn canonical(value: &Value) -> String {
    match value {
        Value::Array(items) => format!("[{}]", items.iter().map(canonical).collect::<Vec<_>>().join(",")),
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            format!("{{{}}}", keys.iter().map(|k| format!("{}:{}", Value::String((*k).clone()), canonical(&map[*k]))).collect::<Vec<_>>().join(","))
        }
        other => other.to_string(),
    }
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
#[serde(rename_all = "camelCase", deny_unknown_fields)]
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
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RefusedOutput {
    pub status: RefusedStatus,
    /// A plan reason, or `consent`, `pending_request`, `busy`, `bad_arguments`, `not_offered`, `tool_limit`.
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

/// `oaiy.ring.plan`, the plugin asking the desktop who may be rung.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanRequest {
    pub method: String,
    pub params: PlanParams,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
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

/// The desktop's answer to `oaiy.ring.plan`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlanResult {
    /// Always a valid id, even when nobody is rung: the plugin refuses a plan without one.
    pub plan_id: String,
    pub decision: Decision,
    pub reason: PlanReason,
    pub ring_seconds: u32,
    pub phones: Vec<String>,
    pub wake: Vec<String>,
    pub desktop_toast: bool,
    pub desktop_companions: Vec<String>,
    /// This desktop itself vouches for the request's reason, so the plugin need not see the caller ask for a person. True only
    /// for `urgent`, when the owner allows it and one of their urgent phrases was heard; otherwise false.
    #[serde(default)]
    pub reason_allowed: bool,
}

/// `oaiy.ring.opened`, the plugin telling the desktop the request is out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenedRequest {
    pub method: String,
    pub params: OpenedParams,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
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
    use std::collections::BTreeMap;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../docs/contracts/transfer");

    fn read(name: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(DIR).join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// The fixture parses as `T` and writes back as the very same bytes.
    fn round_trip<T: Serialize + for<'de> Deserialize<'de> + std::fmt::Debug + PartialEq>(name: &str) -> T {
        let text = read(name);
        assert!(text.ends_with('\n') && !text[..text.len() - 1].contains('\n') && !text.contains('\r'), "{name}: one line, one line feed");
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(format!("{}\n", canonical(&value)), text, "{name} is canonical JSON");
        let typed: T = serde_json::from_value(value.clone()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let again = serde_json::to_value(&typed).unwrap();
        assert_eq!(format!("{}\n", canonical(&again)), text, "{name}: what the types write is what the fixture says");
        typed
    }

    #[test]
    fn the_offer_ids_of_the_design_are_what_this_computes() {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Vector {
            generation: u32,
            holder_thumbprint: String,
            offer_id: String,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct File {
            domain: String,
            request_id: String,
            vectors: Vec<Vector>,
        }
        let file: File = serde_json::from_str(&read("offer-id.json")).unwrap();
        assert_eq!(file.domain, OFFER_DOMAIN);
        assert_eq!(file.vectors.len(), 4);
        for v in &file.vectors {
            let id = offer_id(&file.request_id, &v.holder_thumbprint, v.generation);
            assert_eq!(id, v.offer_id, "generation {}", v.generation);
            assert_eq!(id.len(), 33, "{id}");
            assert!(id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
        }
        // A different request, holder or generation is a different offer.
        let base = offer_id(&file.request_id, &file.vectors[0].holder_thumbprint, 0);
        assert_ne!(base, offer_id("assist_other", &file.vectors[0].holder_thumbprint, 0));
        assert_ne!(base, offer_id(&file.request_id, &file.vectors[1].holder_thumbprint, 0));
        assert_ne!(base, offer_id(&file.request_id, &file.vectors[0].holder_thumbprint, 1));
        assert_ne!(offer_id(&file.request_id, &file.vectors[0].holder_thumbprint, 1), offer_id(&file.request_id, &file.vectors[0].holder_thumbprint, 2));
        // The parts cannot run into one another: the zero bytes keep "a"+"bc" from being "ab"+"c".
        assert_ne!(offer_id("a", "bc", 0), offer_id("ab", "c", 0));
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
    fn a_start_that_allows_transfer_and_one_that_resumes_a_call_round_trip() {
        let start: StartFrame = round_trip("start-allow-transfer.json");
        assert!(start.allow_transfer && start.resume.is_none());
        assert_eq!((start.direction.as_deref(), start.from.as_deref()), (Some("inbound"), Some("+61491570006")));
        let resume: StartFrame = round_trip("start-resume.json");
        assert_eq!(resume.resume, Some(Resume { after_handoff: true, handoff_seconds: 42, via: ResumeVia::Return }));
        assert_eq!(resume.generation, 2);
        // A start without either (an older phone) is a start as it always was: neither member is written.
        let mut plain = start.clone();
        plain.allow_transfer = false;
        let written = serde_json::to_value(&plain).unwrap();
        assert!(written.get("allowTransfer").is_none() && written.get("resume").is_none());
    }

    #[test]
    fn ready_says_transfer_v1_only_when_it_agrees() {
        let ready: ReadyFrame = round_trip("ready-features.json");
        assert_eq!(ready.features, vec![crate::voice::transfer::FEATURE.to_string()]);
        let none = ReadyFrame { features: Vec::new(), ..ready };
        assert!(serde_json::to_value(none).unwrap().get("features").is_none(), "an older desktop's ready has no features member");
    }

    #[test]
    fn the_tool_call_is_exactly_a_reason() {
        let call: ToolCall = round_trip("tool-call.json");
        assert_eq!(call.name, crate::voice::transfer::TOOL);
        let args: TransferArguments = serde_json::from_value(call.arguments.clone()).unwrap();
        assert_eq!(args.reason, Reason::CallerAsked);
        assert!(serde_json::from_value::<TransferArguments>(serde_json::json!({"reason": "caller_asked", "note": "x"})).is_err());
        assert!(serde_json::from_value::<TransferArguments>(serde_json::json!({"reason": "anything"})).is_err());
        assert!(crate::voice::transfer::parse_arguments(&call.arguments).is_ok());
    }

    #[test]
    fn the_tool_results_say_ringing_or_why_not() {
        let ringing: ToolResult = round_trip("tool-result-ringing.json");
        assert!(ringing.ok && ringing.continue_response);
        let out: RingingOutput = serde_json::from_value(ringing.output).unwrap();
        assert_eq!((out.ring_seconds, out.request_id.as_str()), (40, "assist_0123456789abcdef0123456789abcdef"));
        for (name, status, reason) in [("tool-result-refused.json", RefusedStatus::Refused, "caller_did_not_ask"), ("tool-result-unavailable.json", RefusedStatus::Unavailable, "quiet_hours")] {
            let result: ToolResult = round_trip(name);
            assert!(!result.ok, "{name}");
            let out: RefusedOutput = serde_json::from_value(result.output).unwrap();
            assert_eq!((out.status, out.reason.as_str()), (status, reason), "{name}");
            // This desktop refuses in the same shape and with the same reason, in its own words (the plugin writes its own).
            let ours = crate::voice::transfer::refused(if status == RefusedStatus::Refused { "refused" } else { "unavailable" }, reason);
            assert_eq!(ours["output"]["reason"], out.reason, "{name}");
            assert_eq!(ours["output"]["status"], serde_json::to_value(status).unwrap(), "{name}");
            let mine: RefusedOutput = serde_json::from_value(ours["output"].clone()).unwrap();
            assert!(!mine.instruction.is_empty(), "{name}");
        }
    }

    #[test]
    fn every_outcome_round_trips_and_is_one_this_desktop_understands() {
        use crate::voice::transfer::Outcome;
        let mut seen = Vec::new();
        for (name, outcome) in [
            ("transfer-outcome-accepted.json", Outcome::Accepted),
            ("transfer-outcome-declined.json", Outcome::Declined),
            ("transfer-outcome-unavailable.json", Outcome::Unavailable),
            ("transfer-outcome-expired.json", Outcome::Expired),
            ("transfer-outcome-cancelled.json", Outcome::Cancelled),
        ] {
            let frame: OutcomeFrame = round_trip(name);
            assert_eq!(frame.outcome, outcome, "{name}");
            assert_eq!(Outcome::parse(outcome.as_str()), Some(outcome));
            assert_eq!(frame.message.is_some(), outcome == Outcome::Declined, "{name}: the owner's words come with a decline only");
            seen.push(outcome);
        }
        assert_eq!(seen.len(), 5);
        let declined: OutcomeFrame = round_trip("transfer-outcome-declined.json");
        assert_eq!(crate::voice::transfer::owner_message(&declined.message.unwrap()).as_deref(), Some("Back after three, please leave a message"));
    }

    #[test]
    fn a_stop_that_hands_the_call_over_names_a_handoff() {
        let stop: StopFrame = round_trip("stop-handoff.json");
        assert!(stop.reason.starts_with(crate::voice::transfer::HANDOFF_PREFIX));
    }

    #[test]
    fn the_desktop_and_the_plugin_ask_and_answer_in_these_shapes() {
        let request: PlanRequest = round_trip("plan-request.json");
        assert_eq!(request.method, "oaiy.ring.plan");
        assert_eq!(request.params.reason, Reason::CallerAsked);
        assert!(request.params.recent_caller_turns.len() <= 3 && request.params.recent_caller_turns.iter().all(|t| t.chars().count() <= 300));
        let ring: PlanResult = round_trip("plan-result-ring.json");
        assert_eq!((ring.decision, ring.reason, ring.ring_seconds, ring.desktop_toast), (Decision::Ring, PlanReason::Ok, 30, true));
        assert!(!ring.plan_id.is_empty());
        let message: PlanResult = round_trip("plan-result-message-only.json");
        assert_eq!((message.decision, message.reason, message.ring_seconds), (Decision::MessageOnly, PlanReason::QuietHours, 0));
        assert!(!message.plan_id.is_empty() && message.phones.is_empty() && !message.desktop_toast, "a plan that rings nobody still has an id");
        let opened: OpenedRequest = round_trip("ring-opened.json");
        assert_eq!(opened.method, "oaiy.ring.opened");
        assert_eq!(opened.params.plan_id, ring.plan_id);
    }

    #[test]
    fn the_plan_results_this_desktop_makes_have_the_fixtures_shape() {
        // What `plan_result` writes for a ring and for a refusal has exactly the members of the fixtures.
        let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<std::collections::BTreeSet<_>>();
        let fixture: Value = serde_json::from_str(&read("plan-result-ring.json")).unwrap();
        let ours = super::super::host::plan_result(&super::super::Authorised {
            plan: super::super::RingPlan { decision: Decision::Ring, reason: PlanReason::Ok, ring_seconds: 30, phones: vec![], wake: vec![], desktop_toast: true, desktop_companions: vec![] },
            plan_id: "plan_x".into(),
            reason_allowed: false,
            caller_number: String::new(),
            caller_name: String::new(),
        });
        assert_eq!(keys(&ours), keys(&fixture));
        let typed: PlanResult = serde_json::from_value(ours).unwrap();
        assert_eq!((typed.decision, typed.plan_id.as_str()), (Decision::Ring, "plan_x"));
    }

    #[test]
    fn the_sums_file_lists_every_fixture_and_each_sum_is_right() {
        let sums = read("SHA256SUMS");
        let mut listed = BTreeMap::new();
        for line in sums.lines() {
            let (sum, name) = line.split_once("  ").expect("sum, two spaces, name");
            assert!(listed.insert(name.to_string(), sum.to_string()).is_none(), "{name} twice");
        }
        let mut on_disk: Vec<String> = std::fs::read_dir(DIR).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".json")).collect();
        on_disk.sort();
        assert_eq!(listed.keys().cloned().collect::<Vec<_>>(), on_disk, "SHA256SUMS lists exactly the fixtures");
        for (name, sum) in &listed {
            let bytes = std::fs::read(std::path::Path::new(DIR).join(name)).unwrap();
            let got: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(&got, sum, "{name}");
        }
    }
}
