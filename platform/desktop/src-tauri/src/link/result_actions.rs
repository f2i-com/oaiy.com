//! What a binding asks for once its flow has finished.
//!
//! A binding says three things: which flow to run, when to run it, and what to
//! DO with the answer. This is the third. A finished run's result is fed to a
//! short list of actions — write a record, update one, say something, call a
//! connector — with the values in each pulled out of the result by reference.
//!
//! THIS IS A PORT, NOT A DESIGN. The provider already runs these actions in its
//! own browser, for runs its browser dispatches; a desktop that claimed a run
//! and then invented its own semantics would make the same binding behave two
//! ways depending on who happened to win the claim. Every rule here was read
//! out of that implementation. Where this file says "the reference", that is
//! what it means.
//!
//! Exactly one runtime performs a given run's actions, and that is not a
//! convention here — it is enforced by the provider: a run is reserved under a
//! UNIQUE idempotency key (a replay returns the existing run rather than a
//! second one) and claimed by an atomic `queued -> running` update, so a run
//! has one claimer. That matters more than usual: twelve of the actions live on
//! this account send an SMS or dial a phone number.

use serde_json::{json, Map, Value};

use super::descriptor::{
    ResultActionFields, ResultActionOperation, ResultActionsSpec, SelectorSource,
};

/// How deep a value template is walked before it is left alone.
///
/// An action is remote input — anyone who can build an app on the provider can
/// author one — so a self-similar payload must not be able to spend this
/// desktop's stack. The reference caps at the same depth.
const MAX_DEPTH: usize = 8;

/// The values a reference can address.
///
/// Borrowed rather than owned: a result can be large, and every action in a
/// binding reads the same one.
#[derive(Debug, Clone, Copy, Default)]
pub struct Scope<'a> {
    pub result: Option<&'a Value>,
    pub event: Option<&'a Value>,
    pub inputs: Option<&'a Value>,
    pub app: Option<&'a Value>,
}

impl<'a> Scope<'a> {
    fn for_source(&self, source: SelectorSource) -> Option<&'a Value> {
        match source {
            SelectorSource::RunResult => self.result,
            SelectorSource::RunEvent => self.event,
            SelectorSource::RunInputs => self.inputs,
            SelectorSource::AppContext => self.app,
            // Addressable and never populated out here. Declared so that a
            // reference to it resolves to NOTHING rather than being mistaken
            // for literal text — see `root_of`.
            SelectorSource::Unavailable => None,
        }
    }
}

/// What a reference resolved to.
///
/// `Missing` is not `Null`. A key whose value is missing must not appear in a
/// written record at all, exactly as the reference's `undefined` disappears
/// when its object is serialised — writing an explicit null instead would
/// overwrite a real stored value with nothing.
#[derive(Debug, Clone, PartialEq)]
enum Resolved {
    Found(Value),
    Missing,
}

/// Walk `path` into `root`. Own keys and array indices only.
///
/// A segment that is not an own key stops the walk. Rust has no prototype
/// chain to protect against, but matching the reference here matters for a
/// different reason: both sides must agree on what "not there" means, or the
/// same action writes different records depending on who ran it.
fn walk<'a>(root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cur = root;
    for segment in path {
        cur = match cur {
            Value::Object(map) => map.get(*segment)?,
            Value::Array(items) => {
                let idx: usize = segment.parse().ok()?;
                items.get(idx)?
            }
            _ => return None,
        };
    }
    Some(cur)
}

/// Split `"$result.a.b"` into its root name and path, if it is spelled like a
/// reference at all. The sigil alone does not make one — see `resolve_one`.
fn split_reference<'a>(text: &'a str, sigil: &str) -> Option<(&'a str, Vec<&'a str>)> {
    let rest = text.strip_prefix(sigil)?;
    let mut parts = rest.split('.');
    let root = parts.next()?;
    if root.is_empty() {
        return None;
    }
    Some((root, parts.collect()))
}

/// Is `name` a root this provider declares, and what does it address?
fn root_of(spec: &ResultActionsSpec, name: &str) -> Option<SelectorSource> {
    spec.selectors
        .roots
        .iter()
        .find(|r| r.name == name)
        .map(|r| r.source)
}

/// Resolve a whole-value reference, or report that the text is not one.
///
/// A string beginning with the sigil is only a reference when its root is
/// DECLARED. This is load-bearing: an SMS body of "$250 deposit received" is
/// text, and treating every `$` as a reference would send the customer an empty
/// message. The reference makes the same distinction, and for the same reason.
fn resolve_one(spec: &ResultActionsSpec, text: &str, scope: &Scope) -> Option<Resolved> {
    let (root, path) = split_reference(text, &spec.selectors.sigil)?;
    let source = root_of(spec, root)?;
    let Some(base) = scope.for_source(source) else {
        // A declared root with nothing behind it. Still a reference, and it
        // resolves to nothing — NOT to the text "$nodes.x", which would land in
        // somebody's record looking like data.
        return Some(Resolved::Missing);
    };
    Some(match walk(base, &path) {
        Some(v) => Resolved::Found(v.clone()),
        None => Resolved::Missing,
    })
}

/// Interpolate `{{ nodes.x.y }}` occurrences into text.
///
/// The sigil inside the braces is optional, because graphs are authored both
/// ways. A reference that resolves to nothing interpolates to nothing, and an
/// object is rendered as its JSON — a message is text, and the alternative is
/// the word "[object Object]" reaching a caller by SMS.
fn interpolate(spec: &ResultActionsSpec, text: &str, scope: &Scope) -> String {
    let open = &spec.selectors.open;
    let close = &spec.selectors.close;
    if open.is_empty() || close.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(open.as_str()) {
        let after = &rest[start + open.len()..];
        let Some(end) = after.find(close.as_str()) else {
            break;
        };
        out.push_str(&rest[..start]);
        let raw = after[..end].trim();
        let with_sigil = if raw.starts_with(&spec.selectors.sigil) {
            raw.to_string()
        } else {
            format!("{}{raw}", spec.selectors.sigil)
        };
        match resolve_one(spec, &with_sigil, scope) {
            Some(Resolved::Found(Value::String(s))) => out.push_str(&s),
            Some(Resolved::Found(Value::Null)) | Some(Resolved::Missing) => {}
            Some(Resolved::Found(v)) => out.push_str(&v.to_string()),
            // Not a declared root: the braces were not a reference. Kept
            // verbatim rather than silently deleted, so a stray brace in a
            // message survives instead of eating the text around it.
            None => {
                out.push_str(open);
                out.push_str(&after[..end]);
                out.push_str(close);
            }
        }
        rest = &after[end + close.len()..];
    }
    out.push_str(rest);
    out
}

/// Resolve every reference inside a value, recursively.
///
/// An object key whose value is missing is DROPPED; an array element becomes
/// null. That asymmetry is not arbitrary — it is what the reference's
/// `undefined` does when its structure is serialised, and a record written from
/// a resolved object must contain the same fields on either runtime.
fn resolve_deep(spec: &ResultActionsSpec, value: &Value, scope: &Scope, depth: usize) -> Resolved {
    if depth > MAX_DEPTH {
        return Resolved::Found(value.clone());
    }
    match value {
        Value::String(text) => match resolve_one(spec, text, scope) {
            Some(found) => found,
            // Not a reference: interpolate any braces, else leave it be.
            None => Resolved::Found(Value::String(interpolate(spec, text, scope))),
        },
        Value::Array(items) => Resolved::Found(Value::Array(
            items
                .iter()
                .map(|item| match resolve_deep(spec, item, scope, depth + 1) {
                    Resolved::Found(v) => v,
                    Resolved::Missing => Value::Null,
                })
                .collect(),
        )),
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, v) in map {
                if let Resolved::Found(resolved) = resolve_deep(spec, v, scope, depth + 1) {
                    out.insert(k.clone(), resolved);
                }
            }
            Resolved::Found(Value::Object(out))
        }
        other => Resolved::Found(other.clone()),
    }
}

/// Does an action's gate let it run?
///
/// Absent or blank passes. A leading negation flips it, repeatably. A reference
/// is judged on the truthiness of what it resolves to; anything else is judged
/// as its own text, so a gate of "yes" runs and a gate of "" does not.
fn gate_passes(spec: &ResultActionsSpec, gate: Option<&str>, scope: &Scope) -> bool {
    let Some(raw) = gate.map(str::trim).filter(|g| !g.is_empty()) else {
        return true;
    };
    let mut expr = raw;
    let mut negate = false;
    while let Some(stripped) = expr.strip_prefix(&spec.selectors.negate) {
        negate = !negate;
        expr = stripped.trim();
    }
    let truthy = match resolve_one(spec, expr, scope) {
        Some(Resolved::Found(v)) => truthiness(&v),
        Some(Resolved::Missing) => false,
        None => !expr.is_empty(),
    };
    if negate {
        !truthy
    } else {
        truthy
    }
}

/// JavaScript truthiness, because that is what the author wrote the gate
/// against. An empty array or object is TRUE there and would be false under a
/// more obvious reading — a gate of `$result.items` fires on `[]`.
fn truthiness(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// What one action turned out to be, once its gate and references were read.
#[derive(Debug, Clone, PartialEq)]
pub enum Planned {
    /// The gate said no. Not a failure, and not reported as one.
    Skipped,
    /// Create a record.
    Submit { form_id: String, record: Value },
    /// Update a record.
    Update {
        form_id: String,
        record_id: String,
        record: Value,
    },
    /// Say something at this machine.
    Notify { message: String },
    /// Call a plugin connector.
    Connector {
        connector_id: String,
        command: String,
        payload: Value,
    },
}

fn field<'a>(action: &'a Value, name: &str) -> Option<&'a Value> {
    action.get(name)
}

fn text_field(action: &Value, name: &str) -> Option<String> {
    field(action, name)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// A resolved reference that must be a non-empty string to be usable.
fn resolved_id(spec: &ResultActionsSpec, raw: Option<&Value>, scope: &Scope) -> Option<String> {
    match resolve_deep(spec, raw?, scope, 0) {
        Resolved::Found(Value::String(s)) if !s.trim().is_empty() => Some(s),
        _ => None,
    }
}

/// A resolved reference that must be an object to be written as a record.
fn resolved_record(spec: &ResultActionsSpec, raw: Option<&Value>, scope: &Scope) -> Result<Value, String> {
    let Some(raw) = raw else {
        return Err("names no values to write".to_string());
    };
    match resolve_deep(spec, raw, scope, 0) {
        Resolved::Found(v @ Value::Object(_)) => Ok(v),
        Resolved::Found(other) => Err(format!(
            "its values resolved to {}, not an object",
            kind_of(&other)
        )),
        Resolved::Missing => Err("its values resolved to nothing".to_string()),
    }
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "nothing",
        Value::Bool(_) => "a true/false",
        Value::Number(_) => "a number",
        Value::String(_) => "text",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

/// Read one action into what it would DO, without doing any of it.
///
/// Separated from performing on purpose: everything that can be decided from
/// the action and the result is decided here, where it is testable without a
/// network, a plugin, or a record store.
pub fn plan(
    spec: &ResultActionsSpec,
    action: &Value,
    scope: &Scope,
) -> Result<Planned, String> {
    let f: &ResultActionFields = &spec.fields;
    let kind = text_field(action, &f.kind).unwrap_or_default();
    let Some(operation) = spec
        .actions
        .iter()
        .find(|a| a.action_type == kind)
        .map(|a| a.operation)
    else {
        let known: Vec<&str> = spec.actions.iter().map(|a| a.action_type.as_str()).collect();
        return Err(format!(
            "the connector maps no such action, so it was not performed; it maps {}",
            known.join(", ")
        ));
    };

    if !gate_passes(spec, field(action, &f.gate).and_then(Value::as_str), scope) {
        return Ok(Planned::Skipped);
    }

    match operation {
        ResultActionOperation::SubmitRecord => {
            let form_id = text_field(action, &f.form_id)
                .ok_or_else(|| "names no form to write to".to_string())?;
            let record = resolved_record(spec, field(action, &f.record), scope)?;
            Ok(Planned::Submit { form_id, record })
        }
        ResultActionOperation::UpdateRecord => {
            let form_id = text_field(action, &f.form_id)
                .ok_or_else(|| "names no form to write to".to_string())?;
            let record_id = resolved_id(spec, field(action, &f.record_id), scope)
                .ok_or_else(|| "names no record to update".to_string())?;
            let record = resolved_record(spec, field(action, &f.record), scope)?;
            Ok(Planned::Update {
                form_id,
                record_id,
                record,
            })
        }
        ResultActionOperation::Notify => {
            let raw = field(action, &f.message)
                .and_then(Value::as_str)
                .unwrap_or_default();
            let message = interpolate(spec, raw, scope);
            if message.trim().is_empty() {
                return Err("names no message to show".to_string());
            }
            Ok(Planned::Notify { message })
        }
        ResultActionOperation::ConnectorRequest => {
            let connector_id = text_field(action, &f.connector_id)
                .ok_or_else(|| "names no connector to call".to_string())?;
            let command = text_field(action, &f.command)
                .ok_or_else(|| "names no command to run".to_string())?;
            let payload = match field(action, &f.payload) {
                Some(raw) => match resolve_deep(spec, raw, scope, 0) {
                    Resolved::Found(v) => v,
                    Resolved::Missing => json!({}),
                },
                None => json!({}),
            };
            Ok(Planned::Connector {
                connector_id,
                command,
                payload,
            })
        }
    }
}

/// Read a whole binding's list, in order, refusing an oversized one.
///
/// Over the declared maximum the list is TRUNCATED rather than dropped, and the
/// overflow is reported: a binding that grew past the cap should still perform
/// the work it can, and must not do so silently.
pub fn plan_all(
    spec: &ResultActionsSpec,
    actions: &[Value],
    scope: &Scope,
) -> (Vec<(usize, Planned)>, Vec<String>) {
    let mut planned = Vec::new();
    let mut errors = Vec::new();
    for (index, action) in actions.iter().take(spec.max_actions).enumerate() {
        match plan(spec, action, scope) {
            Ok(Planned::Skipped) => {}
            Ok(p) => planned.push((index, p)),
            Err(why) => errors.push(format!("{}: {why}", action_name(spec, action, index))),
        }
    }
    if actions.len() > spec.max_actions {
        errors.push(format!(
            "this binding carries {} actions and only the first {} were performed",
            actions.len(),
            spec.max_actions
        ));
    }
    (planned, errors)
}

/// How an action is named in a failure, for somebody reading a run.
fn action_name(spec: &ResultActionsSpec, action: &Value, index: usize) -> String {
    match text_field(action, &spec.fields.kind) {
        Some(kind) => kind,
        None => format!("action {}", index + 1),
    }
}

/// Attach the failures to a result so the provider's console can show them.
///
/// A result that is not an object is wrapped first: a flow answering with a
/// bare string has nowhere to carry a list, and the failures would vanish
/// behind a run that reads as wholly successful.
pub fn attach_errors(spec: &ResultActionsSpec, result: Value, errors: &[String]) -> Value {
    if errors.is_empty() {
        return result;
    }
    let mut map = match result {
        Value::Object(map) => map,
        Value::Null => Map::new(),
        other => {
            let mut m = Map::new();
            m.insert(spec.result_wrapper.clone(), other);
            m
        }
    };
    map.insert(spec.errors_field.clone(), json!(errors));
    Value::Object(map)
}

/// What performing an action needs from the outside world.
///
/// A trait rather than concrete calls so the ordering and reporting below can
/// be tested without a network or a plugin — the part most worth testing is
/// which actions run, in what order, and what happens when one fails.
pub trait Perform {
    /// Create a record; returns its id when the provider reports one.
    fn submit(&self, form_id: &str, record: &Value) -> Result<Option<String>, String>;
    /// Update a record.
    fn update(&self, form_id: &str, record_id: &str, record: &Value) -> Result<(), String>;
    /// Say something at this machine.
    fn notify(&self, message: &str);
    /// Call a plugin connector, under a key that makes a retry harmless.
    fn connector(
        &self,
        connector_id: &str,
        command: &str,
        payload: &Value,
        idempotency_key: &str,
    ) -> Result<Value, String>;
}

/// Perform a binding's actions against a finished run, in order.
///
/// Returns what to append to the run's reported result. ORDER IS THE CONTRACT:
/// actions are written to run in sequence and later ones legitimately depend on
/// earlier ones — a submit that creates the record a following update patches.
///
/// A failure does NOT stop the rest. That is the reference's behaviour and it
/// is the right one here: the actions in a binding are independent errands, and
/// abandoning the remaining four because the second one's form was deleted
/// loses work that would have succeeded. Each failure is collected and reported
/// with the run instead.
pub fn perform_all(
    spec: &ResultActionsSpec,
    actions: &[Value],
    scope: &Scope,
    run_key: &str,
    doer: &dyn Perform,
) -> Vec<String> {
    let (planned, mut errors) = plan_all(spec, actions, scope);
    for (index, action) in planned {
        let outcome = match &action {
            Planned::Skipped => Ok(()),
            Planned::Submit { form_id, record } => doer.submit(form_id, record).map(|_| ()),
            Planned::Update {
                form_id,
                record_id,
                record,
            } => doer.update(form_id, record_id, record),
            Planned::Notify { message } => {
                doer.notify(message);
                Ok(())
            }
            Planned::Connector {
                connector_id,
                command,
                payload,
            } => doer
                .connector(connector_id, command, payload, &action_key(run_key, index))
                .map(|_| ()),
        };
        if let Err(why) = outcome {
            errors.push(format!("{}: {why}", planned_name(&action, index)));
        }
    }
    errors
}

/// The key under which one action's connector call is journalled.
///
/// Built from the RUN's own key, which the provider derives from the binding
/// and the triggering event, plus this action's position. Stable across a
/// retry of the same logical event — a redelivered call.ended must not send a
/// second SMS — and distinct between two actions of one binding, which would
/// otherwise collapse into each other and silently perform only the first.
fn action_key(run_key: &str, index: usize) -> String {
    format!("{run_key}#action{}", index + 1)
}

fn planned_name(action: &Planned, index: usize) -> String {
    let what = match action {
        Planned::Skipped => "action",
        Planned::Submit { .. } => "record write",
        Planned::Update { .. } => "record update",
        Planned::Notify { .. } => "message",
        Planned::Connector { command, .. } => return format!("{command}"),
    };
    format!("{what} {}", index + 1)
}

/// Performing actions for real: records over the provider's API, connector
/// commands through the same gate the relay uses.
pub struct Live<'a> {
    pub account: &'a super::LinkedAccount,
    pub spec: &'a ResultActionsSpec,
    /// Absent on a runtime with no plugin host — a connector action then fails
    /// and says so, rather than being quietly dropped.
    pub connector: Option<&'a super::relay::Dispatcher>,
}

impl Live<'_> {
    fn client() -> Result<reqwest::blocking::Client, String> {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| format!("could not build the client: {e}"))
    }

    fn refusal(status: reqwest::StatusCode, payload: &Value, doing: &str) -> String {
        let message = payload
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("no reason given");
        format!(
            "the provider refused to {doing}: HTTP {}: {message}",
            status.as_u16()
        )
    }
}

impl Perform for Live<'_> {
    fn submit(&self, form_id: &str, record: &Value) -> Result<Option<String>, String> {
        let url = super::oauth::join(
            &self.account.base_url,
            &self.spec.submit_path.replace("{formId}", form_id),
        );
        let resp = Self::client()?
            .post(&url)
            .bearer_auth(&self.account.credential)
            .json(&json!({ &self.spec.fields.record: record }))
            .send()
            .map_err(|e| format!("could not reach the provider to write the record: {e}"))?;
        let status = resp.status();
        let payload: Value = resp.json().unwrap_or(Value::Null);
        if !status.is_success() {
            return Err(Self::refusal(status, &payload, "write the record"));
        }
        Ok(payload
            .get(&self.spec.fields.id)
            .or_else(|| payload.get("response").and_then(|r| r.get(&self.spec.fields.id)))
            .and_then(Value::as_str)
            .map(str::to_string))
    }

    fn update(&self, form_id: &str, record_id: &str, record: &Value) -> Result<(), String> {
        let url = super::oauth::join(
            &self.account.base_url,
            &self
                .spec
                .update_path
                .replace("{formId}", form_id)
                .replace("{id}", record_id),
        );
        let resp = Self::client()?
            .put(&url)
            .bearer_auth(&self.account.credential)
            .json(&json!({ &self.spec.fields.record: record }))
            .send()
            .map_err(|e| format!("could not reach the provider to update the record: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let payload: Value = resp.json().unwrap_or(Value::Null);
            return Err(Self::refusal(status, &payload, "update the record"));
        }
        Ok(())
    }

    fn notify(&self, message: &str) {
        // Reaches the log, not the user. Said plainly rather than pretended:
        // the toast lane a script would need does not exist on this side yet.
        log::info!("flow action says: {message}");
    }

    fn connector(
        &self,
        connector_id: &str,
        command: &str,
        payload: &Value,
        idempotency_key: &str,
    ) -> Result<Value, String> {
        let Some(dispatch) = self.connector else {
            return Err(
                "this runtime has no plugin host, so the command was not run".to_string()
            );
        };
        dispatch(connector_id, command, payload, idempotency_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::descriptor;

    fn spec() -> ResultActionsSpec {
        descriptor::builtin()
            .into_iter()
            .find(|c| c.flows.is_some())
            .and_then(|c| c.flows.and_then(|f| f.result_actions))
            .expect("the built-in connector describes the actions it performs")
    }

    fn result_scope(v: &Value) -> Scope<'_> {
        Scope {
            result: Some(v),
            ..Scope::default()
        }
    }

    #[test]
    fn the_builtin_connector_describes_every_action_shape_this_account_actually_uses() {
        // Read off the live account: these four types cover all 82 actions on
        // the 28 bindings that carry any. A type this desktop cannot map is
        // refused loudly by `plan`, so the list must stay complete.
        let s = spec();
        let mapped: Vec<&str> = s.actions.iter().map(|a| a.action_type.as_str()).collect();
        for used in [
            "formlogic.updateResponse",
            "formlogic.submitResponse",
            "formlogic.toast",
            "connector.request",
        ] {
            assert!(mapped.contains(&used), "{used} is unmapped; {mapped:?}");
        }
    }

    #[test]
    fn a_whole_value_reference_yields_the_referenced_value_itself() {
        // "answers": "$result.callUpdate" replaces the ENTIRE answers map, so a
        // reference must be able to produce an object, not just text.
        let s = spec();
        let result = json!({ "callUpdate": { "summary": "went well" }, "n": 0 });
        let scope = result_scope(&result);
        assert_eq!(
            resolve_deep(&s, &json!("$result.callUpdate"), &scope, 0),
            Resolved::Found(json!({ "summary": "went well" }))
        );
        // A falsy value is a value: 0 is an answer, not an absence.
        assert_eq!(
            resolve_deep(&s, &json!("$result.n"), &scope, 0),
            Resolved::Found(json!(0))
        );
    }

    #[test]
    fn a_missing_reference_is_absent_rather_than_null_or_its_own_text() {
        // Writing null would overwrite a stored value with nothing; writing the
        // literal "$result.nope" puts a selector into a customer's record.
        let s = spec();
        let result = json!({ "kept": 1 });
        let scope = result_scope(&result);
        let Resolved::Found(out) = resolve_deep(
            &s,
            &json!({ "kept": "$result.kept", "gone": "$result.nope" }),
            &scope,
            0,
        ) else {
            panic!("an object always resolves");
        };
        assert_eq!(out, json!({ "kept": 1 }));
        assert!(!out.to_string().contains("$result"));

        // In a LIST the element cannot simply vanish without renumbering the
        // rest, so it becomes null — the same as the reference.
        let Resolved::Found(list) = resolve_deep(&s, &json!(["$result.nope"]), &scope, 0) else {
            panic!("a list always resolves");
        };
        assert_eq!(list, json!([null]));
    }

    #[test]
    fn a_dollar_sign_in_ordinary_text_is_not_a_reference() {
        // An SMS body of "$250 deposit received" must arrive intact. Only a
        // DECLARED root makes a reference.
        let s = spec();
        let result = json!({});
        let scope = result_scope(&result);
        assert_eq!(
            resolve_deep(&s, &json!("$250 deposit received"), &scope, 0),
            Resolved::Found(json!("$250 deposit received"))
        );
    }

    #[test]
    fn a_reference_to_a_root_that_has_no_value_here_resolves_to_nothing() {
        // `$nodes` is addressable inside the engine and meaningless out here.
        // It must resolve to NOTHING — declaring it is what stops it being
        // mistaken for literal text and written into a record.
        let s = spec();
        let result = json!({});
        let scope = result_scope(&result);
        assert_eq!(
            resolve_deep(&s, &json!("$nodes.summary.content"), &scope, 0),
            Resolved::Missing
        );
    }

    #[test]
    fn templates_interpolate_and_leave_unknown_braces_alone() {
        let s = spec();
        let event = json!({ "data": { "to": "0400111222", "reason": "busy" } });
        let scope = Scope {
            event: Some(&event),
            ..Scope::default()
        };
        // The live toast, verbatim.
        assert_eq!(
            interpolate(&s, "SMS to {{event.data.to}} FAILED: {{event.data.reason}}", &scope),
            "SMS to 0400111222 FAILED: busy"
        );
        // The sigil inside braces is optional, and whitespace is tolerated.
        assert_eq!(interpolate(&s, "{{ $event.data.to }}", &scope), "0400111222");
        // A missing one interpolates to nothing rather than to its own text.
        assert_eq!(interpolate(&s, "x{{event.data.nope}}y", &scope), "xy");
        // Braces that name nothing addressable are left as written.
        assert_eq!(interpolate(&s, "{{not_a_root}}", &scope), "{{not_a_root}}");
    }

    #[test]
    fn the_gate_decides_from_the_result_and_negates() {
        let s = spec();
        let result = json!({ "hasTask": true, "hasSms": false, "count": 0, "list": [] });
        let scope = result_scope(&result);
        assert!(gate_passes(&s, None, &scope), "no gate runs");
        assert!(gate_passes(&s, Some(""), &scope), "a blank gate runs");
        assert!(gate_passes(&s, Some("$result.hasTask"), &scope));
        assert!(!gate_passes(&s, Some("$result.hasSms"), &scope));
        assert!(!gate_passes(&s, Some("$result.missing"), &scope));
        assert!(gate_passes(&s, Some("!$result.hasSms"), &scope));
        assert!(!gate_passes(&s, Some("!!$result.hasSms"), &scope));
        // 0 is false and an empty list is TRUE, because that is what the author
        // wrote the gate against.
        assert!(!gate_passes(&s, Some("$result.count"), &scope));
        assert!(gate_passes(&s, Some("$result.list"), &scope));
    }

    #[test]
    fn the_live_bindings_actions_plan_into_exactly_what_they_say() {
        // Verbatim from this account's bindings.
        let s = spec();
        let result = json!({
            "hasTask": true,
            "task": { "title": "Call back", "phone": "0421285243" },
            "hasSms": true,
            "sms": { "to": "0421285243", "body": "Thanks!" },
            "hasCall": true,
            "responseId": "resp-1",
            "callUpdate": { "summary": "went well" }
        });
        let scope = result_scope(&result);

        let submit = json!({
            "form": "62c2a2cc", "type": "formlogic.submitResponse",
            "when": "$result.hasTask", "answers": "$result.task"
        });
        assert_eq!(
            plan(&s, &submit, &scope).unwrap(),
            Planned::Submit {
                form_id: "62c2a2cc".into(),
                record: json!({ "title": "Call back", "phone": "0421285243" }),
            }
        );

        let update = json!({
            "form": "fde1f7f6", "type": "formlogic.updateResponse",
            "when": "$result.hasCall", "answers": "$result.callUpdate",
            "responseId": "$result.responseId"
        });
        assert_eq!(
            plan(&s, &update, &scope).unwrap(),
            Planned::Update {
                form_id: "fde1f7f6".into(),
                record_id: "resp-1".into(),
                record: json!({ "summary": "went well" }),
            }
        );

        // The one that reaches a real phone: nested references inside a payload.
        let sms = json!({
            "type": "connector.request", "when": "$result.hasSms", "command": "sms.send",
            "payload": { "to": "$result.sms.to", "body": "$result.sms.body" },
            "connectorId": "aokie"
        });
        assert_eq!(
            plan(&s, &sms, &scope).unwrap(),
            Planned::Connector {
                connector_id: "aokie".into(),
                command: "sms.send".into(),
                payload: json!({ "to": "0421285243", "body": "Thanks!" }),
            }
        );
    }

    #[test]
    fn a_gate_that_says_no_is_skipped_and_is_not_a_failure() {
        let s = spec();
        let result = json!({ "hasSms": false });
        let scope = result_scope(&result);
        let sms = json!({
            "type": "connector.request", "when": "$result.hasSms", "command": "sms.send",
            "payload": { "to": "$result.sms.to" }, "connectorId": "aokie"
        });
        assert_eq!(plan(&s, &sms, &scope).unwrap(), Planned::Skipped);
        let (planned, errors) = plan_all(&s, std::slice::from_ref(&sms), &scope);
        assert!(planned.is_empty());
        assert!(errors.is_empty(), "a gate saying no is not an error: {errors:?}");
    }

    #[test]
    fn a_write_whose_values_did_not_resolve_refuses_rather_than_writing_a_blank() {
        // The failure this most protects against: an ungated update whose
        // answers resolved to nothing would otherwise blank a real record.
        let s = spec();
        let result = json!({});
        let scope = result_scope(&result);
        let update = json!({
            "form": "fde1f7f6", "type": "formlogic.updateResponse",
            "answers": "$result.nope", "responseId": "$result.alsoNope"
        });
        let err = plan(&s, &update, &scope).expect_err("must refuse");
        assert!(err.contains("record"), "{err}");

        // And a submit with no id to update at all.
        let submit = json!({ "form": "x", "type": "formlogic.submitResponse", "answers": "$result.nope" });
        assert!(plan(&s, &submit, &scope).is_err());
    }

    #[test]
    fn an_action_this_desktop_cannot_map_is_refused_by_name() {
        let s = spec();
        let result = json!({});
        let scope = result_scope(&result);
        // Allowed by the provider, deliberately unmapped here for now.
        let speak = json!({ "type": "call.speak", "message": "hello" });
        let err = plan(&s, &speak, &scope).expect_err("an unmapped action must not be silently skipped");
        assert!(err.contains("maps no such action"), "{err}");
        // It is reported, not dropped.
        let (planned, errors) = plan_all(&s, &[speak], &scope);
        assert!(planned.is_empty());
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("call.speak"), "{:?}", errors);
    }

    #[test]
    fn an_oversized_binding_performs_what_it_can_and_says_what_it_did_not() {
        let s = spec();
        let result = json!({ "m": "hi" });
        let scope = result_scope(&result);
        let one = json!({ "type": "formlogic.toast", "message": "{{result.m}}" });
        let many: Vec<Value> = std::iter::repeat_n(one, s.max_actions + 3).collect();
        let (planned, errors) = plan_all(&s, &many, &scope);
        assert_eq!(planned.len(), s.max_actions);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("only the first"), "{:?}", errors);
    }

    #[test]
    fn a_failure_reaches_the_result_even_when_the_flow_answered_with_a_bare_value() {
        let s = spec();
        // An object result carries the list directly.
        let attached = attach_errors(&s, json!({ "ok": true }), &["toast: nope".into()]);
        assert_eq!(attached["ok"], json!(true));
        assert_eq!(attached[&s.errors_field][0], json!("toast: nope"));

        // A bare answer is wrapped, so the failure cannot vanish behind a run
        // that otherwise reads as wholly successful.
        let wrapped = attach_errors(&s, json!("just text"), &["toast: nope".into()]);
        assert_eq!(wrapped[&s.result_wrapper], json!("just text"));
        assert_eq!(wrapped[&s.errors_field][0], json!("toast: nope"));

        // Nothing failed, nothing added.
        assert_eq!(attach_errors(&s, json!({ "ok": true }), &[]), json!({ "ok": true }));
    }

    /// Records what was asked of it, and fails whichever calls it is told to.
    #[derive(Default)]
    struct Spy {
        done: std::cell::RefCell<Vec<String>>,
        keys: std::cell::RefCell<Vec<String>>,
        fail: Vec<&'static str>,
    }

    impl Perform for Spy {
        fn submit(&self, form_id: &str, _record: &Value) -> Result<Option<String>, String> {
            self.done.borrow_mut().push(format!("submit:{form_id}"));
            if self.fail.contains(&"submit") {
                return Err("the form is gone".into());
            }
            Ok(Some("new-1".into()))
        }
        fn update(&self, form_id: &str, record_id: &str, _record: &Value) -> Result<(), String> {
            self.done
                .borrow_mut()
                .push(format!("update:{form_id}/{record_id}"));
            if self.fail.contains(&"update") {
                return Err("no such record".into());
            }
            Ok(())
        }
        fn notify(&self, message: &str) {
            self.done.borrow_mut().push(format!("notify:{message}"));
        }
        fn connector(
            &self,
            connector_id: &str,
            command: &str,
            _payload: &Value,
            idempotency_key: &str,
        ) -> Result<Value, String> {
            self.done
                .borrow_mut()
                .push(format!("connector:{connector_id}/{command}"));
            self.keys.borrow_mut().push(idempotency_key.to_string());
            if self.fail.contains(&"connector") {
                return Err("the plugin refused".into());
            }
            Ok(json!({}))
        }
    }

    fn live_actions() -> Vec<Value> {
        vec![
            json!({ "form": "tasks", "type": "formlogic.submitResponse",
                    "when": "$result.hasTask", "answers": "$result.task" }),
            json!({ "type": "connector.request", "when": "$result.hasSms", "command": "sms.send",
                    "payload": { "to": "$result.sms.to" }, "connectorId": "aokie" }),
            json!({ "form": "calls", "type": "formlogic.updateResponse",
                    "when": "$result.hasCall", "answers": "$result.callUpdate",
                    "responseId": "$result.responseId" }),
            json!({ "type": "formlogic.toast", "message": "done {{result.responseId}}" }),
        ]
    }

    fn live_result() -> Value {
        json!({
            "hasTask": true, "task": { "title": "Call back" },
            "hasSms": true, "sms": { "to": "0421285243" },
            "hasCall": true, "callUpdate": { "summary": "ok" }, "responseId": "resp-1"
        })
    }

    #[test]
    fn actions_run_in_the_order_written_because_later_ones_depend_on_earlier_ones() {
        let s = spec();
        let result = live_result();
        let scope = result_scope(&result);
        let spy = Spy::default();
        let errors = perform_all(&s, &live_actions(), &scope, "flow:b1:call_x:ended:v1", &spy);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            *spy.done.borrow(),
            vec![
                "submit:tasks",
                "connector:aokie/sms.send",
                "update:calls/resp-1",
                "notify:done resp-1",
            ]
        );
    }

    #[test]
    fn one_failed_action_does_not_abandon_the_others_and_is_reported_by_name() {
        // Four independent errands. Losing three because the second's form was
        // deleted would throw away work that would have succeeded.
        let s = spec();
        let result = live_result();
        let scope = result_scope(&result);
        let spy = Spy {
            fail: vec!["submit"],
            ..Spy::default()
        };
        let errors = perform_all(&s, &live_actions(), &scope, "flow:b1:call_x:ended:v1", &spy);
        assert_eq!(spy.done.borrow().len(), 4, "every action still ran");
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("the form is gone"), "{:?}", errors);
    }

    #[test]
    fn a_connector_call_is_keyed_so_a_redelivered_event_cannot_send_a_second_sms() {
        // The run's key already identifies the binding and the triggering
        // event; the position distinguishes two actions of the same binding,
        // which would otherwise collapse and perform only the first.
        let s = spec();
        let result = json!({ "hasSms": true, "sms": { "to": "1" } });
        let scope = result_scope(&result);
        let two = vec![
            json!({ "type": "connector.request", "command": "sms.send",
                    "payload": {}, "connectorId": "aokie" }),
            json!({ "type": "connector.request", "command": "call.dial",
                    "payload": {}, "connectorId": "aokie" }),
        ];
        let run_key = "flow:b1:aokie:call_2a6b:ended:v1";

        let first = Spy::default();
        perform_all(&s, &two, &scope, run_key, &first);
        let second = Spy::default();
        perform_all(&s, &two, &scope, run_key, &second);

        // Same event replayed -> byte-identical keys, so the journal dedupes.
        assert_eq!(*first.keys.borrow(), *second.keys.borrow());
        // ...and the two actions of one binding are still told apart.
        let keys = first.keys.borrow().clone();
        assert_eq!(keys.len(), 2);
        assert_ne!(keys[0], keys[1]);
        assert!(keys[0].starts_with(run_key), "{}", keys[0]);
    }

    #[test]
    fn a_gated_off_action_never_reaches_the_outside_world() {
        let s = spec();
        // No SMS, no call: only the toast should happen.
        let result = json!({ "hasTask": false, "hasSms": false, "hasCall": false, "responseId": "r" });
        let scope = result_scope(&result);
        let spy = Spy::default();
        let errors = perform_all(&s, &live_actions(), &scope, "k", &spy);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(*spy.done.borrow(), vec!["notify:done r"]);
    }

    #[test]
    fn no_provider_vocabulary_is_written_into_this_module() {
        // Every action type, field name and root above is DESCRIPTOR DATA. A
        // name compiled in here would make this lane work for exactly one
        // provider while looking general. Serde renames at run time, so this
        // checks the source text, and a previous review found a leak that a
        // looser check had missed.
        let source = include_str!("result_actions.rs");
        for provider_word in [
            "formlogic.submitResponse",
            "formlogic.updateResponse",
            "formlogic.toast",
            "outputActions",
            "outputActionErrors",
            "responseId",
            "connectorId",
        ] {
            // The tests quote the provider's own data on purpose; only the
            // implementation above must stay free of it.
            let impl_end = source.find("#[cfg(test)]").expect("tests exist");
            assert!(
                !source[..impl_end].contains(provider_word),
                "{provider_word:?} is compiled into result_actions.rs — it belongs in the descriptor"
            );
        }
    }
}
