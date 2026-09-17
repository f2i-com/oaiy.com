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
use crate::bridge::conditions::{self, ConditionJob, ConditionVerdict, ShadowContext, Verdict};
use crate::bridge::script_host::{Prelude, ScriptBatch};

/// How long a fetched binding list is reused.
///
/// Short, because a user who just wired up a trigger expects the next call to
/// fire it — and long enough that a burst of events is not a burst of fetches.
const BINDINGS_TTL: Duration = Duration::from_secs(60);

/// A failed refresh must not hold up every subsequent plugin event while the
/// provider is unavailable. Keep the last list and retry after a short pause.
const FETCH_RETRY_AFTER: Duration = Duration::from_secs(30);

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
    /// Everything else the provider publishes on a binding, kept as it arrived.
    ///
    /// The post-run actions live under a key this file must not know: the
    /// descriptor names it, because a second provider will call the same idea
    /// something else. Capturing the whole remainder means adding a field to
    /// the descriptor is enough to reach it — no struct change, and nothing
    /// here to keep in step.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl Binding {
    /// One of the provider's own fields, by the name its descriptor gives.
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.extra.get(name)
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct BindingsReply {
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
    inner: Mutex<BindingsState>,
}

#[derive(Default)]
struct BindingsState {
    // The runner's cache outlives a link. Never reuse another account's output
    // actions, even during the TTL or when the new provider cannot be reached.
    source: Option<(LinkedAccount, String)>,
    fetched: Option<(Instant, Vec<Binding>)>,
    retry_at: Option<Instant>,
    last_error: Option<String>,
}

impl Default for FlowBindings {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowBindings {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BindingsState::default()),
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
        // Serialize refreshes as well as cache access: an older in-flight
        // request must not overwrite the bindings for a newly linked account.
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if state.source.as_ref().map_or(true, |(held, path)| {
            held != account || path != &spec.bindings_path
        }) {
            *state = BindingsState {
                source: Some((account.clone(), spec.bindings_path.clone())),
                ..BindingsState::default()
            };
        }
        if let Some((at, list)) = state.fetched.as_ref() {
            if at.elapsed() < BINDINGS_TTL {
                return Ok(list.clone());
            }
        }
        if state.retry_at.is_some_and(|at| Instant::now() < at) {
            return match state.fetched.as_ref() {
                Some((_, list)) => Ok(list.clone()),
                None => Err(state.last_error.clone().unwrap_or_else(|| {
                    "the account's flow bindings could not be read; waiting to retry".into()
                })),
            };
        }
        match fetch(account, spec) {
            Ok(list) => {
                state.fetched = Some((Instant::now(), list.clone()));
                state.retry_at = None;
                state.last_error = None;
                Ok(list)
            }
            Err(e) => {
                state.retry_at = Some(Instant::now() + FETCH_RETRY_AFTER);
                state.last_error = Some(e.clone());
                log::warn!("could not refresh flow bindings: {e}");
                match state.fetched.as_ref() {
                    Some((_, list)) => Ok(list.clone()),
                    None => Err(e),
                }
            }
        }
    }

    /// Drop the cache, so the next event refetches.
    pub fn invalidate(&self) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = BindingsState::default();
    }
}

/// What the run is started with.
///
/// The whole envelope goes under `event`, because an author's `$event.data.…`
/// expects it there. The binding's INPUT MAP is applied on top: it names the
/// flow's own inputs — "callId" from "$event.data.callId" — and it is what
/// makes `$inputs.callId` mean anything once the flow is running.
///
/// Dropping it is not a small loss. Seventeen of this account's nineteen
/// bindings declare one, and while it was discarded every flow that named its
/// trigger's fields ran blind: the receptionist could not be told which call to
/// configure, so it answered and hung up.
///
/// The map cannot shadow `event` — a flow reading `$event` must get the
/// envelope whatever an input happens to be called.
fn input_snapshot(binding: &Binding, spec: &FlowsSpec, envelope: &Value) -> Value {
    let mapped = match (spec.selectors.as_ref(), spec.input_map_field.as_deref()) {
        (Some(selectors), Some(field)) => match binding.field(field) {
            Some(map) => super::result_actions::build_inputs(
                selectors,
                map,
                &super::result_actions::Scope {
                    event: Some(envelope),
                    ..super::result_actions::Scope::default()
                },
            ),
            None => json!({}),
        },
        // A provider that describes neither still gets the envelope, which is
        // what this desktop sent before input maps were applied at all.
        _ => json!({}),
    };
    let mut out = match mapped {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    out.insert("event".to_string(), envelope.clone());
    Value::Object(out)
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

/// One matched binding, on its way to a decision.
#[derive(Debug)]
pub enum Pending<'a> {
    /// Out already — disabled, manual, or no flow to run. No condition needed,
    /// and therefore no engine call.
    Skipped(&'a Binding, Skip),
    /// Fires unless its condition says otherwise.
    Candidate {
        binding: &'a Binding,
        /// The expression to decide, or `None` when the binding has no
        /// condition at all and simply fires.
        condition: Option<String>,
        /// Which job in the batch answers this one.
        job: Option<usize>,
    },
}

/// What this event matched, in binding order, before any condition was decided.
///
/// Split out of `select` so the conditions can be evaluated with no lock held.
/// The order of the returned list is the order the bindings appear, which is
/// what makes [`MAX_BINDINGS_PER_EVENT`] land on a stable set.
pub fn filter<'a>(bindings: &'a [Binding], event_name: &str) -> Vec<Pending<'a>> {
    let mut out = Vec::new();
    let mut next_job = 0usize;
    // Exact name equality, like the local dispatcher. No wildcards: a binding
    // that fires on more than its author named is the dangerous direction.
    for binding in bindings.iter().filter(|b| b.event == event_name) {
        if !binding.enabled {
            out.push(Pending::Skipped(binding, Skip::Disabled));
            continue;
        }
        if binding.mode.as_deref() == Some("manual") {
            out.push(Pending::Skipped(binding, Skip::ManualMode));
            continue;
        }
        if binding.flow_slug.as_deref().unwrap_or("").is_empty() {
            out.push(Pending::Skipped(binding, Skip::NoFlow));
            continue;
        }
        match condition_source(binding) {
            Err(skip) => out.push(Pending::Skipped(binding, skip)),
            Ok(None) => out.push(Pending::Candidate { binding, condition: None, job: None }),
            Ok(Some(expr)) => {
                out.push(Pending::Candidate {
                    binding,
                    condition: Some(expr),
                    job: Some(next_job),
                });
                next_job += 1;
            }
        }
    }
    out
}

/// The expression this binding's condition asks for, if any.
///
/// The condition arrives as `{"type":"expression","expr":"…"}`, or as a bare
/// string, or absent. An absent one — and a blank one — is not a condition: it
/// fires, and it costs no engine call. A shape this does not recognise is a
/// skip, because guessing which way it would have gone is the mistake this
/// whole lane exists to avoid.
fn condition_source(binding: &Binding) -> Result<Option<String>, Skip> {
    let Some(raw) = binding.condition.as_ref() else {
        return Ok(None);
    };
    let expr = match raw {
        Value::Null => return Ok(None),
        Value::String(s) => s.clone(),
        Value::Object(o) if o.is_empty() => return Ok(None),
        Value::Object(o) => match o.get("expr").and_then(Value::as_str) {
            Some(e) => e.to_string(),
            None => {
                return Err(Skip::ConditionUnknown("the condition names no expression".into()))
            }
        },
        other => {
            return Err(Skip::ConditionUnknown(format!(
                "the condition is a {} rather than an expression",
                match other {
                    Value::Bool(_) => "boolean",
                    Value::Number(_) => "number",
                    Value::Array(_) => "list",
                    _ => "value",
                }
            )))
        }
    };
    Ok(Some(expr.trim().to_string()).filter(|e| !e.is_empty()))
}

/// The batch this event sends: ONE request, one job per condition.
///
/// `event` is bound to the ENVELOPE, not to this desktop's internal event —
/// conditions are authored against `event.data.*` as it appears on the wire,
/// which is exactly what the Rust grammar read.
pub fn condition_jobs(pending: &[Pending<'_>], envelope: &Value) -> Vec<ConditionJob> {
    let mut jobs: Vec<Option<ConditionJob>> = Vec::new();
    for p in pending {
        let Pending::Candidate { condition: Some(expr), job: Some(n), .. } = p else {
            continue;
        };
        if jobs.len() <= *n {
            jobs.resize_with(n + 1, || None);
        }
        jobs[*n] = Some(ConditionJob {
            id: ConditionJob::id_for(*n),
            source: expr.clone(),
            globals: json!({ "event": envelope }),
        });
    }
    jobs.into_iter().flatten().collect()
}

/// The bindings this event should fire, why the others were skipped, and any
/// `condition-shadow` lines the two grammars produced.
#[derive(Debug, Default)]
pub struct Selection<'a> {
    pub fire: Vec<&'a Binding>,
    pub skipped: Vec<(&'a Binding, Skip)>,
    /// Already logged; kept so a test can read them without capturing the log.
    pub shadow: Vec<String>,
}

/// Fold ZIPP's answers back into the matched list.
///
/// Pure: no engine, no lock, no clock. `answers` is what
/// [`crate::bridge::conditions::decide`] returned for [`condition_jobs`], in the
/// same order; a candidate whose answer is missing is Unknown, which does not
/// fire.
pub fn decide<'a>(
    pending: Vec<Pending<'a>>,
    answers: &[ConditionVerdict],
    event_name: &str,
    source: &str,
    envelope: &Value,
) -> Selection<'a> {
    let idempotency_key = envelope
        .get("idempotencyKey")
        .and_then(Value::as_str)
        .unwrap_or("");
    let mut sel = Selection::default();
    for p in pending {
        let (binding, reason) = match p {
            Pending::Skipped(binding, skip) => (binding, Some(skip)),
            Pending::Candidate { binding, condition: None, .. } => (binding, None),
            Pending::Candidate { binding, condition: Some(expr), job } => {
                let answer = job.and_then(|n| answers.get(n));
                let verdict = match answer {
                    Some(a) => a.verdict.clone(),
                    None => Verdict::Unknown(
                        "its condition was not evaluated before this dispatch".into(),
                    ),
                };
                if answer.map(|a| a.answered).unwrap_or(false) {
                    let rust = super::condition::shadow_verdict(&expr, envelope);
                    let ctx = ShadowContext {
                        lane: "flows",
                        binding: &binding.id,
                        event: event_name,
                        source,
                        idempotency_key,
                        expr: &expr,
                    };
                    if let Some(line) = conditions::shadow_line(&ctx, &verdict, &rust) {
                        log::warn!("{line}");
                        sel.shadow.push(line);
                    }
                }
                let reason = match verdict {
                    Verdict::True => None,
                    Verdict::False => Some(Skip::ConditionFalse),
                    Verdict::Unknown(why) => Some(Skip::ConditionUnknown(why)),
                };
                (binding, reason)
            }
        };
        let reason = reason.or_else(|| {
            (sel.fire.len() >= MAX_BINDINGS_PER_EVENT).then_some(Skip::TooManyBindings)
        });
        match reason {
            Some(r) => sel.skipped.push((binding, r)),
            None => sel.fire.push(binding),
        }
    }
    sel
}

/// The bindings this event should fire, and why the others were skipped.
///
/// Conditions are decided on ZIPP, in ONE batch, before anything is reserved,
/// under the provider's own prelude — these are the PROVIDER's conditions,
/// authored in its editor against its standard library, and a `Missing`
/// prelude skips every one of them rather than deciding it in a language half
/// of which is absent.
pub fn select<'a>(
    host: &dyn ScriptBatch,
    bindings: &'a [Binding],
    event_name: &str,
    source: &str,
    envelope: &Value,
    prelude: Prelude<'_>,
) -> Selection<'a> {
    let pending = filter(bindings, event_name);
    let jobs = condition_jobs(&pending, envelope);
    let answers = conditions::decide(host, &jobs, prelude);
    decide(pending, &answers, event_name, source, envelope)
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
        "inputSnapshot": input_snapshot(binding, spec, envelope),
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
            extra: serde_json::Map::new(),
        }
    }

    /// The shipped provider, so the test reads the real names rather than ones
    /// invented here — the point of the descriptor is that these are data.
    fn shipped_flows() -> FlowsSpec {
        super::super::descriptor::builtin()
            .into_iter()
            .find_map(|c| c.flows)
            .expect("the built-in connector describes its flows lane")
    }

    fn account(base: String) -> LinkedAccount {
        LinkedAccount {
            connector_id: "formlogic".into(),
            base_url: base,
            credential: "test-key-one".into(),
            account_id: Some("account-one".into()),
            account_name: None,
            granted_scopes: None,
            linked_at: chrono::Utc::now(),
            instance_id: Some("test-desktop".into()),
        }
    }

    /// Serve only the expected reads. The notification channel lets recovery
    /// tests prove that events during backoff did not make more HTTP requests.
    fn binding_server(
        replies: Vec<(&'static str, &'static str)>,
    ) -> (String, std::sync::mpsc::Receiver<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for (status, body) in replies {
                let (mut stream, _) = listener.accept().unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut request = Vec::new();
                let mut buffer = [0; 1024];
                while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                    let length = stream.read(&mut buffer).unwrap();
                    assert_ne!(length, 0);
                    request.extend_from_slice(&buffer[..length]);
                }
                let _ = tx.send(());
                write!(stream, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        (format!("http://{address}"), rx)
    }

    const FIRST_BINDINGS: &str = r#"{"bindings":[{"id":"first","event":"aokie.call.ended","flow":"summary"}]}"#;
    const SECOND_BINDINGS: &str = r#"{"bindings":[{"id":"second","event":"aokie.call.ended","flow":"summary"}]}"#;

    #[test]
    fn reconnecting_with_a_new_key_never_reuses_the_previous_accounts_bindings() {
        let (base, requests) = binding_server(vec![("200 OK", FIRST_BINDINGS), ("200 OK", SECOND_BINDINGS)]);
        let mut account = account(base);
        let cache = FlowBindings::new();
        let spec = shipped_flows();
        assert_eq!(cache.load(&account, &spec).unwrap()[0].id, "first");
        // The same account id may be retained when credentials are reissued.
        // A reconnect still needs a fresh list and fresh permission checks.
        account.credential = "test-key-two".into();
        assert_eq!(cache.load(&account, &spec).unwrap()[0].id, "second");
        assert_eq!(requests.iter().count(), 2);
    }

    #[test]
    fn changing_providers_does_not_fall_back_to_the_previous_providers_actions() {
        let (first, _) = binding_server(vec![("200 OK", FIRST_BINDINGS)]);
        let (second, _) = binding_server(vec![("503 Service Unavailable", r#"{"message":"offline"}"#)]);
        let mut account = account(first);
        let cache = FlowBindings::new();
        let spec = shipped_flows();
        assert_eq!(cache.load(&account, &spec).unwrap()[0].id, "first");
        account.base_url = second;
        assert!(cache.load(&account, &spec).unwrap_err().contains("503"));
    }

    #[test]
    fn an_outage_reuses_known_bindings_without_blocking_every_event_and_recovers() {
        let (base, requests) = binding_server(vec![
            ("200 OK", FIRST_BINDINGS),
            ("503 Service Unavailable", r#"{"message":"offline"}"#),
            ("200 OK", SECOND_BINDINGS),
        ]);
        let account = account(base);
        let cache = FlowBindings::new();
        let spec = shipped_flows();
        assert_eq!(cache.load(&account, &spec).unwrap()[0].id, "first");
        requests.recv().unwrap();
        cache.inner.lock().unwrap().fetched.as_mut().unwrap().0 = Instant::now() - BINDINGS_TTL;
        assert_eq!(cache.load(&account, &spec).unwrap()[0].id, "first");
        requests.recv().unwrap();
        for _ in 0..5 {
            assert_eq!(cache.load(&account, &spec).unwrap()[0].id, "first");
        }
        assert!(requests.try_recv().is_err(), "events during backoff must not fetch again");
        cache.inner.lock().unwrap().retry_at = Some(Instant::now());
        assert_eq!(cache.load(&account, &spec).unwrap()[0].id, "second");
        requests.recv().unwrap();
        assert!(cache.inner.lock().unwrap().last_error.is_none());
    }

    #[test]
    fn an_initial_failure_backs_off_and_invalidation_allows_immediate_recovery() {
        let (base, requests) = binding_server(vec![
            ("503 Service Unavailable", r#"{"message":"offline"}"#),
            ("200 OK", FIRST_BINDINGS),
        ]);
        let account = account(base);
        let cache = FlowBindings::new();
        let spec = shipped_flows();
        let failure = cache.load(&account, &spec).unwrap_err();
        requests.recv().unwrap();
        assert_eq!(cache.load(&account, &spec).unwrap_err(), failure);
        assert!(requests.try_recv().is_err());
        cache.invalidate();
        assert_eq!(cache.load(&account, &spec).unwrap()[0].id, "first");
        requests.recv().unwrap();
    }

    #[test]
    fn an_unrecognized_successful_reply_does_not_erase_known_bindings() {
        let (base, requests) = binding_server(vec![("200 OK", FIRST_BINDINGS), ("200 OK", "{}")]);
        let account = account(base);
        let cache = FlowBindings::new();
        let spec = shipped_flows();
        cache.load(&account, &spec).unwrap();
        cache.inner.lock().unwrap().fetched.as_mut().unwrap().0 = Instant::now() - BINDINGS_TTL;
        assert_eq!(cache.load(&account, &spec).unwrap()[0].id, "first");
        assert!(cache.inner.lock().unwrap().last_error.as_ref().unwrap().contains("unreadable binding list"));
        assert_eq!(requests.iter().count(), 2);
        // An explicitly empty catalogue remains valid: all triggers may have
        // been removed by the owner.
        assert!(serde_json::from_value::<BindingsReply>(json!({"bindings": []})).unwrap().bindings.is_empty());
    }

    #[test]
    fn a_run_is_started_with_the_inputs_its_binding_maps() {
        // The receptionist answered and hung up because its flow asked the
        // plugin to configure `$inputs.callId` and nothing had ever built
        // `inputs`. Seventeen of nineteen live bindings declare a map.
        let spec = shipped_flows();
        let field = spec
            .input_map_field
            .clone()
            .expect("the provider names its input map");
        let mut b = binding("b1", "aokie.call.caller_id");
        b.extra.insert(
            field,
            json!({ "from": "$event.data.from", "callId": "$event.data.callId" }),
        );
        let envelope = json!({
            "name": "aokie.call.caller_id",
            "data": { "from": "0421285243", "callId": "call_5089e2ff" }
        });

        let snapshot = input_snapshot(&b, &spec, &envelope);
        assert_eq!(snapshot["callId"], json!("call_5089e2ff"));
        assert_eq!(snapshot["from"], json!("0421285243"));
        // The envelope still rides along: an author's `$event.data.…` must keep
        // resolving whatever the map is called.
        assert_eq!(snapshot["event"]["data"]["callId"], json!("call_5089e2ff"));

        // A binding with no map still gets the envelope, which is all this
        // desktop sent before maps were applied at all.
        let bare = binding("b2", "aokie.call.ended");
        let plain = input_snapshot(&bare, &spec, &envelope);
        assert_eq!(plain["event"]["data"]["from"], json!("0421285243"));
        assert_eq!(plain.as_object().map(|m| m.len()), Some(1));
    }

    #[test]
    fn an_input_named_event_cannot_hide_the_envelope() {
        // A flow reading `$event` must get the envelope whatever an input
        // happens to be called, or the map silently blinds the whole graph.
        let spec = shipped_flows();
        let field = spec.input_map_field.clone().unwrap();
        let mut b = binding("b3", "aokie.call.ended");
        b.extra.insert(field, json!({ "event": "$event.data.from" }));
        let envelope = json!({ "name": "aokie.call.ended", "data": { "from": "0421285243" } });
        let snapshot = input_snapshot(&b, &spec, &envelope);
        assert_eq!(snapshot["event"]["data"]["from"], json!("0421285243"));
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
        json!({ "name": "aokie.call.ended", "idempotencyKey": "evt-1", "data": data })
    }

    use crate::bridge::conditions::testing::{guest, FakeHost};

    /// Every condition answers `value`.
    fn answering(value: Value) -> FakeHost {
        FakeHost::always(value)
    }

    fn fire_ids<'a>(sel: &Selection<'a>) -> Vec<&'a str> {
        sel.fire.iter().map(|b| b.id.as_str()).collect()
    }

    #[test]
    fn only_bindings_named_for_this_event_are_considered() {
        let all = vec![
            binding("b1", "aokie.call.ended"),
            binding("b2", "aokie.call.incoming"),
            binding("b3", "aokie.call.ended"),
        ];
        let e = envelope(json!({}));
        let host = answering(json!(true));
        let sel = select(&host, &all, "aokie.call.ended", "aokie", &e, Prelude::None);
        assert_eq!(fire_ids(&sel), ["b1", "b3"]);
        assert!(sel.skipped.is_empty());
        // No wildcards or prefixes: firing on more than the author named is the
        // dangerous direction.
        assert!(select(&host, &all, "aokie.call", "aokie", &e, Prelude::None).fire.is_empty());
        assert!(select(&host, &all, "aokie.call.ended.extra", "aokie", &e, Prelude::None).fire.is_empty());
    }

    #[test]
    fn a_condition_the_author_wrote_is_evaluated_rather_than_skipped() {
        // The live blocker this fixes: the flows that record a call's
        // transcript are bound with a condition, so refusing every condition
        // meant the console showed "no transcript recorded" for every call.
        let expr = "event && event.data ? Number(event.data.durationSeconds || 0) > 5 : false";
        let mut b = binding("b1", "aokie.call.transcript.settled");
        b.condition = Some(json!({ "type": "expression", "expr": expr }));
        let all = vec![b];

        let long = answering(json!(true));
        assert_eq!(
            select(&long, &all, "aokie.call.transcript.settled", "aokie", &envelope(json!({ "durationSeconds": 33 })), Prelude::None).fire.len(),
            1
        );

        let short = answering(json!(false));
        let sel = select(&short, &all, "aokie.call.transcript.settled", "aokie", &envelope(json!({ "durationSeconds": 2 })), Prelude::None);
        assert!(sel.fire.is_empty());
        // False is a DECISION, and reads differently from "not understood".
        assert_eq!(sel.skipped[0].1, Skip::ConditionFalse);
        assert!(sel.skipped[0].1.message().contains("evaluated false"));
    }

    #[test]
    fn a_condition_that_is_not_understood_still_refuses_to_fire() {
        // The safe direction is unchanged: understood-and-true is the only path
        // to firing, so a condition that throws never sends the SMS.
        let mut b = binding("b1", "aokie.call.ended");
        b.condition = Some(json!({ "type": "expression", "expr": "event.data.from.includes('+44')" }));
        let throwing = FakeHost::by_source(|_| Err(guest("TypeError: cannot read 'includes' of undefined")));
        let sel = select(&throwing, std::slice::from_ref(&b), "aokie.call.ended", "aokie", &envelope(json!({})), Prelude::None);
        assert!(sel.fire.is_empty());
        assert!(matches!(sel.skipped[0].1, Skip::ConditionUnknown(_)));
        // …and it says WHY, so an author is not left guessing.
        assert!(sel.skipped[0].1.message().contains("could not be evaluated"));

        // A condition of an unexpected SHAPE is refused without an engine at
        // all: there is no expression to send.
        let mut odd = binding("b2", "aokie.call.ended");
        odd.condition = Some(json!([1, 2, 3]));
        let host = answering(json!(true));
        let sel = select(&host, std::slice::from_ref(&odd), "aokie.call.ended", "aokie", &envelope(json!({})), Prelude::None);
        assert!(sel.fire.is_empty());
        assert!(matches!(sel.skipped[0].1, Skip::ConditionUnknown(_)));
        assert_eq!(host.calls(), 0, "a shape with no expression is not a job");
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
        let throwing = FakeHost::by_source(|_| Err(guest("TypeError")));
        let sel = select(&throwing, &all, "e", "aokie", &envelope(json!({})), Prelude::None);
        assert!(sel.fire.is_empty(), "none of these may fire");
        assert!(matches!(sel.skipped[0].1, Skip::Disabled));
        assert!(matches!(sel.skipped[1].1, Skip::ManualMode));
        assert!(matches!(sel.skipped[2].1, Skip::NoFlow));
        assert!(matches!(sel.skipped[3].1, Skip::ConditionUnknown(_)));
        assert!(sel.skipped.iter().all(|(_, s)| !s.message().is_empty()));
    }

    #[test]
    fn an_absent_condition_is_not_a_condition_and_costs_no_engine() {
        // null, "", "   " and {} all mean the author wrote no condition. Reading
        // any of them as one would skip a binding that should have fired — the
        // opposite mistake, and just as invisible. And none of them may reach
        // the script host: the first conditioned event after boot spawns a
        // child, and an unconditioned workspace must never pay for one.
        for empty in [json!(null), json!(""), json!("   "), json!({})] {
            let mut b = binding("b", "e");
            b.condition = Some(empty.clone());
            let host = answering(json!(false));
            let sel = select(&host, std::slice::from_ref(&b), "e", "aokie", &envelope(json!({})), Prelude::None);
            assert_eq!(sel.fire.len(), 1, "{empty:?} is not a condition and must not block the binding");
            assert_eq!(host.calls(), 0, "{empty:?} must not become a job");
        }
    }

    #[test]
    fn one_event_cannot_become_an_unbounded_number_of_runs() {
        let all: Vec<Binding> = (0..9).map(|i| binding(&format!("b{i}"), "e")).collect();
        let host = answering(json!(true));
        let sel = select(&host, &all, "e", "aokie", &envelope(json!({})), Prelude::None);
        assert_eq!(sel.fire.len(), MAX_BINDINGS_PER_EVENT);
        assert!(sel.skipped.iter().all(|(_, s)| *s == Skip::TooManyBindings));
        assert_eq!(sel.skipped.len(), 9 - MAX_BINDINGS_PER_EVENT);
    }

    // -- on ZIPP ------------------------------------------------------------

    fn conditional(id: &str, expr: &str) -> Binding {
        let mut b = binding(id, "aokie.call.ended");
        b.condition = Some(json!({ "type": "expression", "expr": expr }));
        b
    }

    #[test]
    fn every_condition_of_one_event_travels_in_one_batch_bound_to_the_envelope() {
        let all = vec![
            conditional("b1", "event.data.n === 1"),
            binding("b2", "aokie.call.ended"),
            conditional("b3", "event.data.n === 2"),
        ];
        let host = answering(json!(true));
        let e = envelope(json!({ "n": 1 }));
        select(&host, &all, "aokie.call.ended", "aokie", &e, Prelude::None);

        let request = host.only_request();
        let jobs = request["jobs"].as_array().expect("jobs");
        assert_eq!(jobs.len(), 2, "the unconditioned binding is not a job");
        assert_eq!(jobs[0]["source"], "event.data.n === 1");
        assert_eq!(jobs[1]["source"], "event.data.n === 2");
        // The ENVELOPE, not this desktop's internal event: conditions are
        // authored against `event.data.*` as it appears on the wire, which is
        // exactly what the Rust grammar read.
        assert_eq!(jobs[0]["globals"], json!({ "event": e }));
    }

    #[test]
    fn zipps_verdict_is_the_one_used_not_the_rust_grammars() {
        // `event.data.n > 1` is a condition BOTH grammars understand, and the
        // Rust one says true for n = 5. ZIPP says false, and ZIPP decides.
        let all = vec![conditional("b1", "event.data.n > 1")];
        let e = envelope(json!({ "n": 5 }));

        let zipp_says_no = answering(json!(false));
        let sel = select(&zipp_says_no, &all, "aokie.call.ended", "aokie", &e, Prelude::None);
        assert!(sel.fire.is_empty(), "ZIPP said false, so the binding does not fire");
        assert_eq!(sel.skipped[0].1, Skip::ConditionFalse);
        assert_eq!(sel.shadow.len(), 1, "and the disagreement is on the record");
        assert!(sel.shadow[0].contains("zipp=false rust=true"), "{}", sel.shadow[0]);

        // And the other way: an expression the Rust grammar REFUSES — it knows
        // `String` and `Number` and no other call — which ZIPP evaluates
        // perfectly well, and which now fires.
        let method = vec![conditional("b1", "event.data.tags.includes('vip')")];
        let zipp_says_yes = answering(json!(true));
        let sel = select(&zipp_says_yes, &method, "aokie.call.ended", "aokie", &envelope(json!({ "tags": ["vip"] })), Prelude::None);
        assert_eq!(fire_ids(&sel), ["b1"], "a method call is legal JavaScript");
        assert_eq!(sel.shadow.len(), 1);
        assert!(sel.shadow[0].contains("refused=rust"), "{}", sel.shadow[0]);
    }


    #[test]
    fn the_providers_prelude_travels_with_the_binding_conditions() {
        let preamble = "var validators = { email: function (v) { return /@/.test(v); } };";
        let doc = json!({
            "v": 1,
            "preamble": preamble,
            "preambleSha256": crate::link::script_profile::sha256_hex(preamble),
        });
        let all = vec![conditional("b1", "validators.email(event.data.to)")];
        let host = answering(json!(true));
        let sel = select(
            &host,
            &all,
            "aokie.call.ended",
            "aokie",
            &envelope(json!({ "to": "a@b.co" })),
            Prelude::Profile(&doc),
        );
        assert_eq!(fire_ids(&sel), ["b1"]);
        assert_eq!(host.only_request()["profile"], doc);
    }

    #[test]
    fn without_the_providers_prelude_no_conditioned_binding_fires_and_nothing_is_sent() {
        // And an UNCONDITIONED binding still fires: it reads no helper, so
        // there is nothing about it the missing library could change.
        let all = vec![
            conditional("b1", "validators.email(event.data.to)"),
            binding("b2", "aokie.call.ended"),
        ];
        let host = answering(json!(true));
        let sel = select(
            &host,
            &all,
            "aokie.call.ended",
            "aokie",
            &envelope(json!({ "to": "a@b.co" })),
            Prelude::Missing("the provider's prelude could not be read"),
        );
        assert_eq!(host.calls(), 0);
        assert_eq!(fire_ids(&sel), ["b2"]);
        assert!(matches!(sel.skipped[0].1, Skip::ConditionUnknown(_)));
        assert!(sel.skipped[0].1.message().contains("could not be read"));
        assert!(sel.shadow.is_empty(), "an absent prelude is not a grammar disagreement");
    }

    #[test]
    fn agreement_leaves_no_shadow_line() {
        let all = vec![conditional("b1", "event.data.n > 1")];
        let host = answering(json!(true));
        let sel = select(&host, &all, "aokie.call.ended", "aokie", &envelope(json!({ "n": 5 })), Prelude::None);
        assert_eq!(fire_ids(&sel), ["b1"]);
        assert!(sel.shadow.is_empty(), "a shadow line means the two disagreed");
    }

    #[test]
    fn a_host_outage_skips_every_conditioned_binding_and_shadows_nothing() {
        // Fail closed: the Rust grammar does NOT take over, because a fallback
        // that decided would change semantics under the user mid-outage — in
        // the firing direction. And an outage is not a disagreement, so it must
        // not fill the shadow log that PR9's deletion gate reads.
        let all = vec![conditional("b1", "event.data.n > 1"), binding("b2", "aokie.call.ended")];
        let down = FakeHost::down("no engine on this machine");
        let sel = select(&down, &all, "aokie.call.ended", "aokie", &envelope(json!({ "n": 5 })), Prelude::None);

        assert_eq!(fire_ids(&sel), ["b2"], "an unconditioned binding is unaffected");
        assert!(matches!(sel.skipped[0].1, Skip::ConditionUnknown(_)));
        assert!(sel.skipped[0].1.message().contains("no engine on this machine"));
        assert!(sel.shadow.is_empty(), "an outage is an outage, not a semantic disagreement");
    }

    #[test]
    fn a_verdict_nobody_produced_does_not_fire() {
        // `decide` is pure and takes whatever answers it is given. A candidate
        // with no answer at all must read as unknown, never as true.
        let all = vec![conditional("b1", "event.data.n > 1")];
        let e = envelope(json!({ "n": 5 }));
        let pending = filter(&all, "aokie.call.ended");
        let sel = decide(pending, &[], "aokie.call.ended", "aokie", &e);
        assert!(sel.fire.is_empty());
        assert!(matches!(sel.skipped[0].1, Skip::ConditionUnknown(_)));
    }
}
