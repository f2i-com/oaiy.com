//! Firing the linked account's flows from events that happen on this desktop.
//!
//! A plugin here — Aokie taking a call, say — emits an event. This desktop
//! already matches that against its OWN local bindings. But the flows the user
//! actually built live in the provider's web app, and until now nothing carried
//! the event there: the event was dispatched locally, found nothing, and the
//! flow the user wrote never ran.
//!
//! So: fetch the account's bindings, match the event by name, and reserve a run
//! for each match. The run is reserved **queued** — this desktop does not
//! execute the provider's flow graph, it asks the account to. Whatever runtime
//! is already claiming queued runs picks it up, which is what makes this useful
//! without a second flow engine to keep in step with the first.
//!
//! # Two rules everything here follows
//!
//! **Fail towards not running.** A trigger that fires when it should not sends
//! the email, charges the card, answers the call. So a binding whose condition
//! this desktop cannot evaluate is SKIPPED and said so, never fired.
//!
//! **One run per event per binding.** The idempotency key is
//! `flow:<bindingId>:<eventIdempotencyKey>` — the exact string every other
//! runtime uses. Inventing a different prefix would not collide with theirs, so
//! the same event would reserve a second row and the flow would run twice.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use super::descriptor::FlowsSpec;
use super::LinkedAccount;

/// How long a fetched binding list is reused.
///
/// Short, because a user who just wired up a trigger expects the next call to
/// fire it — and long enough that a burst of events is not a burst of fetches.
const BINDINGS_TTL: Duration = Duration::from_secs(60);

/// Bindings one event may fire, mirroring the local dispatcher's bound: a
/// misconfigured account should not turn one call into hundreds of runs.
const MAX_BINDINGS_PER_EVENT: usize = 5;

/// One binding as the provider publishes it.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Binding {
    pub id: String,
    /// The event name to match, e.g. `aokie.call.incoming`.
    pub event: String,
    /// The flow's SLUG, which is what reserving a run names. The row also
    /// carries `flowDefinitionId`, which the reserve route does not accept.
    #[serde(rename = "flow")]
    pub flow_slug: Option<String>,
    #[serde(default)]
    pub app_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Present as JSON of whatever shape the author wrote. This desktop cannot
    /// evaluate it, so its PRESENCE alone is what matters here.
    #[serde(default)]
    pub condition: Option<Value>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct BindingsReply {
    #[serde(default)]
    bindings: Vec<Binding>,
}

/// Why a binding did not fire, for the operator rather than the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Skip {
    Disabled,
    /// Manual bindings render a button; they never receive events.
    ManualMode,
    /// No slug, so there is nothing to reserve against.
    NoFlow,
    /// The condition was understood and said no.
    ConditionFalse,
    /// The condition was NOT understood. Treated as false deliberately —
    /// guessing which way it would have gone is exactly the mistake that fires
    /// a trigger that should not have fired.
    ConditionUnknown(String),
    TooManyBindings,
}

impl Skip {
    pub fn message(&self) -> String {
        match self {
            Skip::Disabled => "the binding is disabled".into(),
            Skip::ManualMode => {
                "the binding is manual, and manual bindings never receive events".into()
            }
            Skip::NoFlow => "the binding names no flow slug".into(),
            Skip::ConditionFalse => "its condition evaluated false".into(),
            Skip::ConditionUnknown(why) => format!(
                "its condition could not be evaluated, so it was treated as false ({why})"
            ),
            Skip::TooManyBindings => "too many bindings matched this event".into(),
        }
    }
}

/// Bindings, cached briefly.
pub struct FlowBindings {
    inner: Mutex<Option<(Instant, Vec<Binding>)>>,
}

impl Default for FlowBindings {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowBindings {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    /// The account's enabled bindings, refetched when the cache is stale.
    ///
    /// A failed fetch returns the LAST known list rather than an empty one:
    /// treating "could not ask" as "no triggers exist" would silently stop
    /// every automation the moment the network hiccuped.
    pub fn load(
        &self,
        account: &LinkedAccount,
        spec: &FlowsSpec,
    ) -> Result<Vec<Binding>, String> {
        {
            let cached = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((at, list)) = cached.as_ref() {
                if at.elapsed() < BINDINGS_TTL {
                    return Ok(list.clone());
                }
            }
        }
        match fetch(account, spec) {
            Ok(list) => {
                *self.inner.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some((Instant::now(), list.clone()));
                Ok(list)
            }
            Err(e) => {
                let cached = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                match cached.as_ref() {
                    Some((_, list)) => Ok(list.clone()),
                    None => Err(e),
                }
            }
        }
    }

    /// Drop the cache, so the next event refetches.
    pub fn invalidate(&self) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

fn fetch(account: &LinkedAccount, spec: &FlowsSpec) -> Result<Vec<Binding>, String> {
    let http = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| format!("could not build the flows client: {e}"))?;
    let resp = http
        .get(super::oauth::join(&account.base_url, &spec.bindings_path))
        .bearer_auth(&account.credential)
        .send()
        .map_err(|e| format!("could not reach the flows lane: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body: Value = resp.json().unwrap_or(Value::Null);
        let message = body
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("the provider refused the binding list");
        if status.as_u16() == 403 {
            return Err(format!(
                "this desktop's key may not read flows ({message}) — link again to grant it"
            ));
        }
        return Err(format!("HTTP {}: {message}", status.as_u16()));
    }
    let reply: BindingsReply = resp
        .json()
        .map_err(|e| format!("the flows lane returned an unreadable binding list: {e}"))?;
    Ok(reply.bindings)
}

/// The bindings this event should fire, and why the others were skipped.
pub fn select<'a>(
    bindings: &'a [Binding],
    event_name: &str,
    envelope: &Value,
) -> (Vec<&'a Binding>, Vec<(&'a Binding, Skip)>) {
    let mut fire = Vec::new();
    let mut skipped = Vec::new();
    // Exact name equality, like the local dispatcher. No wildcards: a binding
    // that fires on more than its author named is the dangerous direction.
    for binding in bindings.iter().filter(|b| b.event == event_name) {
        let reason = if !binding.enabled {
            Some(Skip::Disabled)
        } else if binding.mode.as_deref() == Some("manual") {
            Some(Skip::ManualMode)
        } else if binding.flow_slug.as_deref().unwrap_or("").is_empty() {
            Some(Skip::NoFlow)
        } else {
            match condition_verdict(binding, envelope) {
                super::condition::Verdict::True => None,
                super::condition::Verdict::False => Some(Skip::ConditionFalse),
                super::condition::Verdict::Unknown(why) => Some(Skip::ConditionUnknown(why)),
            }
        };
        let reason = reason.or_else(|| {
            (fire.len() >= MAX_BINDINGS_PER_EVENT).then_some(Skip::TooManyBindings)
        });
        match reason {
            Some(r) => skipped.push((binding, r)),
            None => fire.push(binding),
        }
    }
    (fire, skipped)
}

/// What this binding's condition decides for this event.
///
/// The condition arrives as `{"type":"expression","expr":"…"}`, or as a bare
/// string, or absent. An absent one is not a condition and fires; a shape this
/// does not recognise is Unknown, which does not.
fn condition_verdict(binding: &Binding, envelope: &Value) -> super::condition::Verdict {
    let Some(raw) = binding.condition.as_ref() else {
        return super::condition::Verdict::True;
    };
    let expr = match raw {
        Value::Null => return super::condition::Verdict::True,
        Value::String(s) => s.clone(),
        Value::Object(o) if o.is_empty() => return super::condition::Verdict::True,
        Value::Object(o) => match o.get("expr").and_then(Value::as_str) {
            Some(e) => e.to_string(),
            None => {
                return super::condition::Verdict::Unknown(
                    "the condition names no expression".into(),
                )
            }
        },
        other => {
            return super::condition::Verdict::Unknown(format!(
                "the condition is a {} rather than an expression",
                match other {
                    Value::Bool(_) => "boolean",
                    Value::Number(_) => "number",
                    Value::Array(_) => "list",
                    _ => "value",
                }
            ))
        }
    };
    super::condition::evaluate(&expr, envelope)
}

/// The exact idempotency key every runtime uses for a binding-fired run.
///
/// The `flow:` prefix is not decoration. The uniqueness index is on
/// (flow, key), so a different prefix does not collide with the other
/// runtimes' — the same event would reserve a second row and the user's flow
/// would run twice.
pub fn idempotency_key(binding_id: &str, event_idempotency_key: &str) -> String {
    format!("flow:{binding_id}:{event_idempotency_key}")
}

/// Reserve one queued run for a binding.
///
/// Returns the run id, or `None` when the provider says it already had this
/// one — which is the idempotency gate doing its job, not a failure.
pub fn reserve(
    account: &LinkedAccount,
    spec: &FlowsSpec,
    binding: &Binding,
    event_name: &str,
    correlation_id: &str,
    event_idempotency_key: &str,
    envelope: &Value,
) -> Result<Option<String>, String> {
    let flow_slug = binding
        .flow_slug
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "the binding names no flow slug".to_string())?;

    let mut body = json!({
        "flowSlug": flow_slug,
        "bindingId": binding.id,
        "triggerEvent": event_name,
        "correlationId": correlation_id,
        "idempotencyKey": idempotency_key(&binding.id, event_idempotency_key),
        // The whole envelope, under `event` — the shape a flow's selectors
        // already expect, so an author's `$event.data.…` resolves unchanged.
        "inputSnapshot": { "event": envelope },
        // QUEUED, deliberately. This desktop does not execute the provider's
        // flow graph; it asks the account to, and whatever runtime already
        // claims queued runs picks it up. Reserving unqueued would insert a row
        // marked running that nothing runs, and it would be reaped as a failure
        // ten minutes later — success followed by a silent death.
        "queued": true,
    });
    // Omitted entirely for a workspace flow: an empty string is not "no app",
    // and sending one would look for the flow in the wrong scope.
    if let Some(app_id) = binding.app_id.as_deref().filter(|a| !a.is_empty()) {
        body["appId"] = json!(app_id);
    }

    let http = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| format!("could not build the flows client: {e}"))?;
    let resp = http
        .post(super::oauth::join(&account.base_url, &spec.reserve_path))
        .bearer_auth(&account.credential)
        .json(&body)
        .send()
        .map_err(|e| format!("could not reserve the run: {e}"))?;
    let status = resp.status();
    let payload: Value = resp.json().unwrap_or(Value::Null);
    if !status.is_success() {
        let message = payload
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("the provider refused the run");
        return Err(format!("HTTP {}: {message}", status.as_u16()));
    }
    // Already reserved by an earlier delivery of this same event.
    if payload.get("created").and_then(Value::as_bool) == Some(false) {
        return Ok(None);
    }
    Ok(payload
        .pointer("/run/runId")
        .or_else(|| payload.pointer("/run/id"))
        .and_then(Value::as_str)
        .map(str::to_string))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(id: &str, event: &str) -> Binding {
        Binding {
            id: id.into(),
            event: event.into(),
            flow_slug: Some("call-summary".into()),
            app_id: None,
            mode: Some("async".into()),
            enabled: true,
            condition: None,
        }
    }

    #[test]
    fn a_binding_list_parses_from_the_providers_actual_wire_shape() {
        // The flow's SLUG arrives under `flow`, not `flowSlug` — and the slug
        // is what reserving names. Reading flowDefinitionId instead would be
        // refused as an unknown flow on every single event.
        let raw = json!({
            "bindings": [{
                "id": "b1",
                "appId": null,
                "formId": null,
                "connectorId": "aokie",
                "flowDefinitionId": "119854a8-df5d-41c9-b070-d56b07a92993",
                "flow": "call-summary-follow-up",
                "event": "aokie.call.ended",
                "mode": "async",
                "condition": null,
                "inputMap": {},
                "outputActions": [],
                "timeoutMs": 30000,
                "retryPolicy": null,
                "fallbackPolicy": null,
                "enabled": true,
                "sortOrder": 0,
                "createdAt": "2026-07-30 08:00:00",
                "updatedAt": "2026-07-30 08:00:00"
            }]
        });
        let reply: BindingsReply = serde_json::from_value(raw).expect("the real shape must parse");
        assert_eq!(reply.bindings.len(), 1);
        let b = &reply.bindings[0];
        assert_eq!(b.flow_slug.as_deref(), Some("call-summary-follow-up"));
        assert_eq!(b.event, "aokie.call.ended");
        assert!(b.enabled);
        assert!(b.app_id.is_none());
    }

    #[test]
    fn the_idempotency_key_is_the_one_every_other_runtime_writes() {
        // The uniqueness index is on (flow, key). A different prefix does NOT
        // collide with the other runtimes', so the same event reserves a second
        // row and the user's flow runs twice.
        assert_eq!(idempotency_key("b1", "evt-7"), "flow:b1:evt-7");
        assert!(idempotency_key("b1", "evt-7").starts_with("flow:"));
        assert_ne!(idempotency_key("b1", "evt-7"), "binding:b1:evt-7");
    }

    fn envelope(data: Value) -> Value {
        json!({ "name": "aokie.call.ended", "data": data })
    }

    #[test]
    fn only_bindings_named_for_this_event_are_considered() {
        let all = vec![
            binding("b1", "aokie.call.ended"),
            binding("b2", "aokie.call.incoming"),
            binding("b3", "aokie.call.ended"),
        ];
        let e = envelope(json!({}));
        let (fire, skipped) = select(&all, "aokie.call.ended", &e);
        assert_eq!(fire.iter().map(|b| b.id.as_str()).collect::<Vec<_>>(), ["b1", "b3"]);
        assert!(skipped.is_empty());
        // No wildcards or prefixes: firing on more than the author named is the
        // dangerous direction.
        assert!(select(&all, "aokie.call", &e).0.is_empty());
        assert!(select(&all, "aokie.call.ended.extra", &e).0.is_empty());
    }

    #[test]
    fn a_condition_the_author_wrote_is_evaluated_rather_than_skipped() {
        // The live blocker this fixes: the flows that record a call's
        // transcript are bound with a condition, so refusing every condition
        // meant the console showed "no transcript recorded" for every call.
        let mut b = binding("b1", "aokie.call.transcript.settled");
        b.condition = Some(json!({
            "type": "expression",
            "expr": "event && event.data ? Number(event.data.durationSeconds || 0) > 5 : false",
        }));
        let all = vec![b];

        let long = envelope(json!({ "durationSeconds": 33 }));
        assert_eq!(select(&all, "aokie.call.transcript.settled", &long).0.len(), 1);

        let short = envelope(json!({ "durationSeconds": 2 }));
        let (fire, skipped) = select(&all, "aokie.call.transcript.settled", &short);
        assert!(fire.is_empty());
        // False is a DECISION, and reads differently from "not understood".
        assert_eq!(skipped[0].1, Skip::ConditionFalse);
        assert!(skipped[0].1.message().contains("evaluated false"));
    }

    #[test]
    fn a_condition_that_is_not_understood_still_refuses_to_fire() {
        // The safe direction is unchanged: understood-and-true is the only path
        // to firing, so a shape the evaluator cannot read never sends the SMS.
        let mut b = binding("b1", "aokie.call.ended");
        b.condition = Some(json!({ "type": "expression", "expr": "event.data.from.includes('+44')" }));
        let (fire, skipped) = select(std::slice::from_ref(&b), "aokie.call.ended", &envelope(json!({})));
        assert!(fire.is_empty());
        assert!(matches!(skipped[0].1, Skip::ConditionUnknown(_)));
        // …and it says WHY, so an author is not left guessing.
        assert!(skipped[0].1.message().contains("could not be evaluated"));

        // A condition of an unexpected SHAPE is equally refused.
        let mut odd = binding("b2", "aokie.call.ended");
        odd.condition = Some(json!([1, 2, 3]));
        let (fire, skipped) = select(std::slice::from_ref(&odd), "aokie.call.ended", &envelope(json!({})));
        assert!(fire.is_empty());
        assert!(matches!(skipped[0].1, Skip::ConditionUnknown(_)));
    }

    #[test]
    fn everything_uncertain_skips_rather_than_fires() {
        // A trigger that fires when it should not sends the email, charges the
        // card, answers the call. Each of these resolves to "do not run", and
        // each says why rather than vanishing.
        let mut disabled = binding("b1", "e");
        disabled.enabled = false;
        let mut manual = binding("b2", "e");
        manual.mode = Some("manual".into());
        let mut no_flow = binding("b3", "e");
        no_flow.flow_slug = None;
        let mut conditional = binding("b4", "e");
        conditional.condition = Some(json!("event.data.from.startsWith('+44')"));

        let all = vec![disabled, manual, no_flow, conditional];
        let (fire, skipped) = select(&all, "e", &envelope(json!({})));
        assert!(fire.is_empty(), "none of these may fire");
        assert!(matches!(skipped[0].1, Skip::Disabled));
        assert!(matches!(skipped[1].1, Skip::ManualMode));
        assert!(matches!(skipped[2].1, Skip::NoFlow));
        assert!(matches!(skipped[3].1, Skip::ConditionUnknown(_)));
        assert!(skipped.iter().all(|(_, s)| !s.message().is_empty()));
    }

    #[test]
    fn an_absent_condition_is_not_a_condition() {
        // null, "" and {} all mean the author wrote no condition. Reading any
        // of them as one would skip a binding that should have fired — the
        // opposite mistake, and just as invisible.
        for empty in [json!(null), json!(""), json!("   "), json!({})] {
            let mut b = binding("b", "e");
            b.condition = Some(empty.clone());
            assert_eq!(
                select(std::slice::from_ref(&b), "e", &envelope(json!({}))).0.len(),
                1,
                "{empty:?} is not a condition and must not block the binding"
            );
        }
    }

    #[test]
    fn one_event_cannot_become_an_unbounded_number_of_runs() {
        let all: Vec<Binding> = (0..9).map(|i| binding(&format!("b{i}"), "e")).collect();
        let (fire, skipped) = select(&all, "e", &envelope(json!({})));
        assert_eq!(fire.len(), MAX_BINDINGS_PER_EVENT);
        assert!(skipped.iter().all(|(_, s)| *s == Skip::TooManyBindings));
        assert_eq!(skipped.len(), 9 - MAX_BINDINGS_PER_EVENT);
    }
}
