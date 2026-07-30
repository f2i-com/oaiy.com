//! The trigger dispatcher: event → binding → reserved run.
//!
//! Contract: `protocol/README.md` §Triggers and
//! `protocol/v1/trigger-binding.schema.json`.
//!
//! ```text
//!   event ──► match enabled bindings on `event`
//!               ──► evaluate `condition`   (FAIL-SAFE: false or error = skip)
//!                     ──► resolve `inputMap` selectors against the event
//!                           ──► ledger.reserve() with lineage
//!                                 ──► a queued run for a worker to claim
//! ```
//!
//! # Everything here fails towards not running
//!
//! A trigger that fires when it should not is worse than one that misses: it
//! sends the email, charges the card, answers the call. So every uncertainty
//! resolves to *skip*, and every skip is recorded with a reason rather than
//! silently dropped.
//!
//! # Conditions are evaluated in a restricted evaluator, not a JS engine
//!
//! FormLogic ran conditions in QuickJS. Embedding a JS engine here to evaluate
//! `event.data.flowId === 'abc'` would be a large amount of attack surface for
//! the shape of expression the trigger editor actually emits.
//!
//! So [`eval_condition`] understands a deliberately small language: selector
//! comparisons against literals, joined by `&&` / `||`. Anything it cannot parse
//! is **refused, not guessed** — and refusal means the binding does not fire.
//! That is the safe direction, and it is loud: the skip reason names the
//! expression, so an author sees "I cannot evaluate this" rather than a trigger
//! that mysteriously never runs.
//!
//! The one thing this must never do is treat an unparseable or erroring
//! condition as true, which would turn a typo into a binding that fires on
//! everything.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ledger::{Ledger, LineageRef, ReserveOutcome, RunRecord, RunRequest};

/// Maximum bindings one event may fire. Bounded so a misconfigured workspace
/// cannot turn a single event into hundreds of runs.
pub const MAX_BINDINGS_PER_EVENT: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BindingMode {
    Sync,
    Async,
    Background,
    /// Renders a button. **Never** receives events — binding a manual trigger to
    /// an event is an authoring mistake, and firing it anyway would make the mode
    /// meaningless.
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TriggerBinding {
    pub id: String,
    /// Event name to match, e.g. `aokie.call.incoming` or `flow.succeeded`.
    pub event: String,
    pub flow_id: String,
    pub mode: BindingMode,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub condition: Option<String>,
    /// Flow input name → selector, e.g. `callerPhone` → `$event.data.callerNumber`.
    #[serde(default)]
    pub input_map: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub sort_order: i32,
}

fn default_true() -> bool {
    true
}

/// Why a binding did not fire. Recorded rather than dropped — a trigger that
/// silently does nothing is the hardest kind of automation bug to find.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    Disabled,
    /// `mode: manual` never receives events.
    ManualMode,
    ConditionFalse,
    /// The condition could not be evaluated. Treated as false, deliberately.
    ConditionUnevaluatable { expression: String, why: String },
    /// A loop guard refused it.
    Guard(String),
    /// The same event already reserved this run.
    Duplicate,
    /// Past [`MAX_BINDINGS_PER_EVENT`].
    TooManyBindings,
}

impl SkipReason {
    pub fn message(&self) -> String {
        match self {
            SkipReason::Disabled => "the binding is disabled".into(),
            SkipReason::ManualMode => {
                "the binding is manual, and manual bindings never receive events".into()
            }
            SkipReason::ConditionFalse => "its condition evaluated false".into(),
            SkipReason::ConditionUnevaluatable { expression, why } => format!(
                "its condition could not be evaluated so it was treated as false ({why}): {expression}"
            ),
            SkipReason::Guard(r) => format!("a loop guard refused it: {r}"),
            SkipReason::Duplicate => "this event already reserved that run".into(),
            SkipReason::TooManyBindings => format!(
                "more than {MAX_BINDINGS_PER_EVENT} bindings matched this event"
            ),
        }
    }
}

#[derive(Debug)]
pub enum DispatchOutcome {
    Reserved { binding_id: String, run: RunRecord },
    Skipped { binding_id: String, reason: SkipReason },
}

/// An inbound event, already validated against its emitter's manifest.
#[derive(Debug, Clone)]
pub struct Event {
    pub name: String,
    pub source: String,
    pub correlation_id: String,
    pub idempotency_key: String,
    pub data: Value,
    /// Set when the event came from a flow's terminal outcome, so the loop guards
    /// can see the lineage.
    pub origin_run: Option<LineageRef>,
}

/// Match bindings against `event` and reserve a run for each that fires.
///
/// Takes `&mut Ledger` rather than the handle so the caller controls the lock
/// scope: dispatching several bindings under one lock keeps the guard bookkeeping
/// consistent, and it is the reserve calls themselves that must be serialized.
pub fn dispatch(
    ledger: &mut Ledger,
    bindings: &[TriggerBinding],
    event: &Event,
) -> Vec<DispatchOutcome> {
    let mut matched: Vec<&TriggerBinding> = bindings
        .iter()
        .filter(|b| b.event == event.name)
        .collect();
    // Deterministic order, so which bindings hit the cap is stable rather than
    // depending on storage order.
    matched.sort_by(|a, b| a.sort_order.cmp(&b.sort_order).then(a.id.cmp(&b.id)));

    let mut out = Vec::new();
    let mut fired = 0usize;

    for b in matched {
        if !b.enabled {
            out.push(DispatchOutcome::Skipped {
                binding_id: b.id.clone(),
                reason: SkipReason::Disabled,
            });
            continue;
        }
        if b.mode == BindingMode::Manual {
            out.push(DispatchOutcome::Skipped {
                binding_id: b.id.clone(),
                reason: SkipReason::ManualMode,
            });
            continue;
        }
        if fired >= MAX_BINDINGS_PER_EVENT {
            out.push(DispatchOutcome::Skipped {
                binding_id: b.id.clone(),
                reason: SkipReason::TooManyBindings,
            });
            continue;
        }

        if let Some(expr) = b.condition.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            match eval_condition(expr, event) {
                Ok(true) => {}
                Ok(false) => {
                    out.push(DispatchOutcome::Skipped {
                        binding_id: b.id.clone(),
                        reason: SkipReason::ConditionFalse,
                    });
                    continue;
                }
                Err(why) => {
                    // The important branch. An unevaluatable condition must never
                    // read as true — that turns a typo into a binding that fires
                    // on everything.
                    out.push(DispatchOutcome::Skipped {
                        binding_id: b.id.clone(),
                        reason: SkipReason::ConditionUnevaluatable {
                            expression: expr.to_string(),
                            why,
                        },
                    });
                    continue;
                }
            }
        }

        // Lineage: extend the origin run's chain so the guards can see it. A
        // dispatch with no origin is a root.
        let origin = event.origin_run.clone().unwrap_or_default();
        let root = origin
            .root_run_id
            .clone()
            .or_else(|| origin.parent_run_id.clone());
        let lineage = LineageRef {
            root_run_id: root,
            parent_run_id: origin.parent_run_id.clone(),
            binding_id: Some(b.id.clone()),
            depth: origin.depth + 1,
        };

        let req = RunRequest {
            caller_product: format!("oaiy-trigger:{}", event.source),
            flow_id: Some(b.flow_id.clone()),
            inline_graph: false,
            correlation_id: event.correlation_id.clone(),
            // The dedupe key: one binding handles one event occurrence once,
            // however many times that event is delivered.
            idempotency_key: format!("binding:{}:{}", b.id, event.idempotency_key),
            lineage,
            trigger_event: Some(event.name.clone()),
        };

        match ledger.reserve(&req) {
            ReserveOutcome::Reserved(run) => {
                fired += 1;
                out.push(DispatchOutcome::Reserved {
                    binding_id: b.id.clone(),
                    run,
                });
            }
            // Not counted against the cap: nothing new was created.
            ReserveOutcome::Duplicate(_) => out.push(DispatchOutcome::Skipped {
                binding_id: b.id.clone(),
                reason: SkipReason::Duplicate,
            }),
            ReserveOutcome::Refused { reason } => out.push(DispatchOutcome::Skipped {
                binding_id: b.id.clone(),
                reason: SkipReason::Guard(reason),
            }),
        }
    }

    out
}

/// Resolve a binding's `inputMap` against an event.
///
/// A selector that resolves to nothing yields JSON `null` rather than being
/// omitted: a flow declaring an input should receive it as explicitly absent,
/// not have the key vanish and read as "the binding never mapped this".
pub fn resolve_inputs(
    input_map: &std::collections::BTreeMap<String, String>,
    event: &Event,
) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    for (name, selector) in input_map {
        out.insert(
            name.clone(),
            resolve_selector(selector, event).unwrap_or(Value::Null),
        );
    }
    out
}

/// Resolve `$event.data.x`, `$event.name`, `$event.correlationId`, or a literal.
///
/// A string that does not start with `$` is a literal, matching the browser
/// executor's convention — otherwise every constant in an input map would need
/// escaping.
pub fn resolve_selector(selector: &str, event: &Event) -> Option<Value> {
    let s = selector.trim();
    if !s.starts_with('$') {
        return Some(Value::String(s.to_string()));
    }
    let mut parts = s.trim_start_matches('$').split('.');
    match parts.next()? {
        "event" => {
            let mut cur = match parts.clone().next() {
                Some("name") => Value::String(event.name.clone()),
                Some("source") => Value::String(event.source.clone()),
                Some("correlationId") => Value::String(event.correlation_id.clone()),
                Some("idempotencyKey") => Value::String(event.idempotency_key.clone()),
                Some("data") => event.data.clone(),
                // `$event` alone.
                None => return Some(event.data.clone()),
                Some(_) => return None,
            };
            // Walk any remaining path segments (skip the one just consumed).
            for seg in parts.skip(1) {
                cur = match &cur {
                    Value::Object(o) => o.get(seg).cloned()?,
                    Value::Array(a) => seg.parse::<usize>().ok().and_then(|i| a.get(i).cloned())?,
                    _ => return None,
                };
            }
            Some(cur)
        }
        _ => None,
    }
}

/// Evaluate a restricted condition expression.
///
/// Supported: `<selector-or-literal> <op> <selector-or-literal>` where op is one
/// of `===`, `!==`, `==`, `!=`, joined by `&&` or `||`. A bare selector is
/// truthy-tested. Everything else is an error, and an error means do not fire.
///
/// `&&` binds tighter than `||`, as in JS, so `a || b && c` is `a || (b && c)`.
pub fn eval_condition(expr: &str, event: &Event) -> Result<bool, String> {
    let expr = expr.trim();
    if expr.is_empty() {
        return Err("empty expression".into());
    }
    // Parentheses would need a real parser; refusing is safer than mis-grouping
    // and silently changing which events match.
    if expr.contains('(') || expr.contains(')') {
        return Err("parentheses are not supported by the restricted evaluator".into());
    }

    // `||` first (lowest precedence), then `&&`.
    //
    // Branch on whether `split_top` actually SPLIT, not on whether the separator
    // appears in the string. `event.data.msg === 'a && b'` contains `&&` inside a
    // quote, so a `find()` check passes while the quote-aware split correctly
    // returns one part — and recursing on that same single part is infinite
    // recursion. It blew the stack rather than misbehaving, which at least made
    // it obvious, but a deeply-nested expression could have found it in
    // production instead.
    for sep in ["||", "&&"] {
        let parts = split_top(expr, sep);
        if parts.len() < 2 {
            continue;
        }
        let is_or = sep == "||";
        for part in parts {
            let v = eval_condition(&part, event)?;
            if is_or && v {
                return Ok(true);
            }
            if !is_or && !v {
                return Ok(false);
            }
        }
        return Ok(!is_or);
    }

    for op in ["===", "!==", "==", "!="] {
        if let Some(idx) = expr.find(op) {
            let lhs = expr[..idx].trim();
            let rhs = expr[idx + op.len()..].trim();
            if lhs.is_empty() || rhs.is_empty() {
                return Err(format!("operator {op} is missing an operand"));
            }
            let l = operand(lhs, event)?;
            let r = operand(rhs, event)?;
            // Loose and strict compare the same way here: both sides are already
            // JSON values, and JS's coercion rules are exactly the source of
            // surprise a restricted evaluator exists to avoid.
            let eq = values_equal(&l, &r);
            return Ok(if op.starts_with('!') { !eq } else { eq });
        }
    }

    // A bare operand: truthy test.
    match operand(expr, event)? {
        Value::Bool(b) => Ok(b),
        Value::Null => Ok(false),
        Value::String(s) => Ok(!s.is_empty()),
        Value::Number(n) => Ok(n.as_f64().map(|f| f != 0.0).unwrap_or(false)),
        Value::Array(a) => Ok(!a.is_empty()),
        Value::Object(_) => Ok(true),
    }
}

/// Compare two JSON values for condition purposes.
///
/// Numbers compare by numeric value, not by `serde_json::Number` identity. This
/// matters more than it looks: an event carrying `{"attempts": 2}` parses as an
/// integer, while the literal `2` in a condition parses through `f64` into
/// `2.0` — and `Number(2) != Number(2.0)`. Raw `Value` equality therefore made
/// `event.data.attempts === 2` evaluate FALSE, which is the fail-safe direction
/// and so would never have thrown an error; the trigger would simply never have
/// fired. Found by a test comparing a count.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(xf), Some(yf)) => xf == yf,
            // Neither converts (a u64 past f64 precision): fall back to identity.
            _ => x == y,
        },
        _ => a == b,
    }
}

/// Split on `sep`, ignoring occurrences inside quotes.
fn split_top(expr: &str, sep: &str) -> Vec<String> {
    let bytes = expr.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let b = bytes[i];
        match quote {
            Some(q) if b == q => quote = None,
            Some(_) => {}
            None if b == b'\'' || b == b'"' => quote = Some(b),
            None if expr[i..].starts_with(sep) => {
                parts.push(expr[start..i].to_string());
                i += sep.len();
                start = i;
                continue;
            }
            None => {}
        }
        i += 1;
    }
    parts.push(expr[start..].to_string());
    parts
}

/// A selector, a quoted string, a number, or a boolean/null keyword.
fn operand(text: &str, event: &Event) -> Result<Value, String> {
    let t = text.trim();
    if t.is_empty() {
        return Err("empty operand".into());
    }
    // Quoted literal.
    if (t.starts_with('\'') && t.ends_with('\'') && t.len() >= 2)
        || (t.starts_with('"') && t.ends_with('"') && t.len() >= 2)
    {
        return Ok(Value::String(t[1..t.len() - 1].to_string()));
    }
    match t {
        "true" => return Ok(Value::Bool(true)),
        "false" => return Ok(Value::Bool(false)),
        "null" | "undefined" => return Ok(Value::Null),
        _ => {}
    }
    if let Ok(n) = t.parse::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(n) {
            return Ok(Value::Number(num));
        }
    }
    // Selectors may be written with or without the leading `$` in conditions —
    // the trigger editor emits `event.data.flowId === '...'`.
    let selector = if t.starts_with('$') {
        t.to_string()
    } else if t.starts_with("event.") || t == "event" {
        format!("${t}")
    } else {
        return Err(format!(
            "{t:?} is neither a literal nor an `event.*` selector"
        ));
    };
    // A selector resolving to nothing is null, not an error: comparing a missing
    // field to a value is a legitimate false, not a broken expression.
    Ok(resolve_selector(&selector, event).unwrap_or(Value::Null))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event() -> Event {
        Event {
            name: "aokie.call.incoming".into(),
            source: "aokie".into(),
            correlation_id: "call_1".into(),
            idempotency_key: "aokie:call_1:incoming:v1".into(),
            data: json!({
                "callerNumber": "+61400000000",
                "known": true,
                "attempts": 2,
                "flowId": "abc",
                "nested": { "deep": { "value": "found" } },
                "list": ["a", "b"]
            }),
            origin_run: None,
        }
    }

    fn binding(id: &str, cond: Option<&str>) -> TriggerBinding {
        TriggerBinding {
            id: id.into(),
            event: "aokie.call.incoming".into(),
            flow_id: "caller-lookup".into(),
            mode: BindingMode::Async,
            enabled: true,
            condition: cond.map(str::to_string),
            input_map: Default::default(),
            sort_order: 0,
        }
    }

    // --- selectors --------------------------------------------------------

    #[test]
    fn selectors_read_the_event() {
        let e = event();
        assert_eq!(
            resolve_selector("$event.data.callerNumber", &e),
            Some(json!("+61400000000"))
        );
        assert_eq!(resolve_selector("$event.name", &e), Some(json!("aokie.call.incoming")));
        assert_eq!(resolve_selector("$event.correlationId", &e), Some(json!("call_1")));
        assert_eq!(
            resolve_selector("$event.data.nested.deep.value", &e),
            Some(json!("found"))
        );
        assert_eq!(resolve_selector("$event.data.list.1", &e), Some(json!("b")));
    }

    #[test]
    fn a_non_dollar_string_is_a_literal() {
        // Otherwise every constant in an input map would need escaping.
        assert_eq!(
            resolve_selector("just text", &event()),
            Some(json!("just text"))
        );
    }

    #[test]
    fn a_missing_path_resolves_to_nothing() {
        assert_eq!(resolve_selector("$event.data.nope", &event()), None);
        assert_eq!(resolve_selector("$event.nope", &event()), None);
        assert_eq!(resolve_selector("$other.thing", &event()), None);
    }

    #[test]
    fn input_maps_produce_explicit_nulls() {
        // A declared input should arrive as explicitly absent rather than having
        // its key vanish, which reads as "the binding never mapped this".
        let mut m = std::collections::BTreeMap::new();
        m.insert("phone".to_string(), "$event.data.callerNumber".to_string());
        m.insert("missing".to_string(), "$event.data.nope".to_string());
        let out = resolve_inputs(&m, &event());
        assert_eq!(out.get("phone"), Some(&json!("+61400000000")));
        assert_eq!(out.get("missing"), Some(&Value::Null));
    }

    // --- conditions: the fail-safe direction ------------------------------

    #[test]
    fn a_simple_equality_evaluates() {
        let e = event();
        assert_eq!(eval_condition("event.data.flowId === 'abc'", &e), Ok(true));
        assert_eq!(eval_condition("event.data.flowId === 'xyz'", &e), Ok(false));
        assert_eq!(eval_condition("event.data.flowId !== 'xyz'", &e), Ok(true));
        // The `$`-prefixed form works too.
        assert_eq!(eval_condition("$event.data.flowId == 'abc'", &e), Ok(true));
    }

    #[test]
    fn the_outcome_trigger_editors_shape_evaluates() {
        // This exact expression is what a source-flow picker generates.
        let mut e = event();
        e.data = json!({ "flowId": "flow_77", "status": "succeeded" });
        assert_eq!(eval_condition("event.data.flowId === 'flow_77'", &e), Ok(true));
        assert_eq!(eval_condition("event.data.flowId === 'flow_78'", &e), Ok(false));
    }

    #[test]
    fn numbers_and_booleans_compare() {
        let e = event();
        assert_eq!(eval_condition("event.data.attempts === 2", &e), Ok(true));
        assert_eq!(eval_condition("event.data.attempts === 3", &e), Ok(false));
        assert_eq!(eval_condition("event.data.known === true", &e), Ok(true));
    }

    #[test]
    fn a_bare_selector_is_truthy_tested() {
        let e = event();
        assert_eq!(eval_condition("event.data.known", &e), Ok(true));
        assert_eq!(eval_condition("event.data.nope", &e), Ok(false));
        assert_eq!(eval_condition("event.data.callerNumber", &e), Ok(true));
    }

    #[test]
    fn and_or_combine_with_js_precedence() {
        let e = event();
        assert_eq!(
            eval_condition("event.data.flowId === 'abc' && event.data.known === true", &e),
            Ok(true)
        );
        assert_eq!(
            eval_condition("event.data.flowId === 'xyz' && event.data.known === true", &e),
            Ok(false)
        );
        assert_eq!(
            eval_condition("event.data.flowId === 'xyz' || event.data.known === true", &e),
            Ok(true)
        );
        // `&&` binds tighter, so this is `false || (true && true)`.
        assert_eq!(
            eval_condition(
                "event.data.flowId === 'xyz' || event.data.known === true && event.data.attempts === 2",
                &e
            ),
            Ok(true)
        );
    }

    #[test]
    fn a_missing_field_compares_false_rather_than_erroring() {
        // Comparing an absent field to a value is a legitimate false, not a
        // broken expression — erroring would make every optional field a skip.
        assert_eq!(
            eval_condition("event.data.absent === 'x'", &event()),
            Ok(false)
        );
    }

    #[test]
    fn an_unparseable_condition_is_an_error_not_true() {
        // The single most important property in this module.
        for expr in [
            "someFunction(event)",
            "event.data.x.map(v => v)",
            "1 + 1",
            "event.data.x > 5",
            "!!event.data.x",
            "",
            "=== 'abc'",
            "event.data.x ===",
            "window.location",
            "process.exit(1)",
        ] {
            let r = eval_condition(expr, &event());
            assert!(
                r.is_err() || r == Ok(false),
                "{expr:?} must not evaluate true; got {r:?}"
            );
            // And specifically: never Ok(true).
            assert_ne!(r, Ok(true), "{expr:?} evaluated TRUE");
        }
    }

    #[test]
    fn quoted_separators_are_not_split() {
        let mut e = event();
        e.data = json!({ "msg": "a && b" });
        assert_eq!(eval_condition("event.data.msg === 'a && b'", &e), Ok(true));
    }

    #[test]
    fn separators_inside_quotes_do_not_recurse_forever() {
        // This was a stack overflow: `find("&&")` succeeded while the quote-aware
        // split returned one part, so the evaluator recursed on the same string.
        // Every case here must terminate — the assertion is really "does not hang
        // or blow the stack".
        let mut e = event();
        e.data = json!({ "a": "x && y", "b": "p || q", "c": "&&", "d": "||||" });
        let cases = [
            ("event.data.a === 'x && y'", Ok(true)),
            ("event.data.b === 'p || q'", Ok(true)),
            ("event.data.c === '&&'", Ok(true)),
            ("event.data.d === '||||'", Ok(true)),
            ("event.data.a === 'nope && nope'", Ok(false)),
            // A real split still works alongside a quoted one.
            ("event.data.a === 'x && y' && event.data.b === 'p || q'", Ok(true)),
            ("event.data.a === 'x && y' || event.data.b === 'wrong'", Ok(true)),
        ];
        for (expr, want) in cases {
            assert_eq!(eval_condition(expr, &e), want, "{expr}");
        }
    }

    #[test]
    fn a_dangling_separator_does_not_hang() {
        let e = event();
        for expr in ["event.data.known &&", "&& event.data.known", "||", "&&", "a || || b"] {
            let r = eval_condition(expr, &e);
            // Whatever it decides, it must decide — and never decide TRUE from a
            // malformed expression.
            assert_ne!(r, Ok(true), "{expr:?} evaluated TRUE");
        }
    }

    // --- dispatch ---------------------------------------------------------

    #[test]
    fn a_matching_binding_reserves_a_run() {
        let mut l = Ledger::new();
        let out = dispatch(&mut l, &[binding("b1", None)], &event());
        assert_eq!(out.len(), 1);
        match &out[0] {
            DispatchOutcome::Reserved { binding_id, run } => {
                assert_eq!(binding_id, "b1");
                assert_eq!(run.flow_id.as_deref(), Some("caller-lookup"));
                assert_eq!(run.trigger_event.as_deref(), Some("aokie.call.incoming"));
                assert_eq!(run.lineage.binding_id.as_deref(), Some("b1"));
                assert_eq!(run.lineage.depth, 1);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_non_matching_event_fires_nothing() {
        let mut l = Ledger::new();
        let mut e = event();
        e.name = "aokie.call.ended".into();
        assert!(dispatch(&mut l, &[binding("b1", None)], &e).is_empty());
    }

    #[test]
    fn the_same_event_delivered_twice_reserves_once() {
        // At-least-once delivery is the norm; without this a redelivered call
        // event runs the flow again.
        let mut l = Ledger::new();
        let bs = [binding("b1", None)];
        let first = dispatch(&mut l, &bs, &event());
        let second = dispatch(&mut l, &bs, &event());
        assert!(matches!(first[0], DispatchOutcome::Reserved { .. }));
        assert!(matches!(
            second[0],
            DispatchOutcome::Skipped { reason: SkipReason::Duplicate, .. }
        ));
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn a_disabled_binding_is_skipped_with_a_reason() {
        let mut l = Ledger::new();
        let mut b = binding("b1", None);
        b.enabled = false;
        match &dispatch(&mut l, &[b], &event())[0] {
            DispatchOutcome::Skipped { reason, .. } => assert_eq!(*reason, SkipReason::Disabled),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_manual_binding_never_receives_events() {
        let mut l = Ledger::new();
        let mut b = binding("b1", None);
        b.mode = BindingMode::Manual;
        match &dispatch(&mut l, &[b], &event())[0] {
            DispatchOutcome::Skipped { reason, .. } => assert_eq!(*reason, SkipReason::ManualMode),
            other => panic!("{other:?}"),
        }
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn a_false_condition_skips() {
        let mut l = Ledger::new();
        let b = binding("b1", Some("event.data.flowId === 'nope'"));
        match &dispatch(&mut l, &[b], &event())[0] {
            DispatchOutcome::Skipped { reason, .. } => {
                assert_eq!(*reason, SkipReason::ConditionFalse)
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn an_unevaluatable_condition_skips_and_names_the_expression() {
        // Fail-safe, and loud: an author must be able to see why nothing fired.
        let mut l = Ledger::new();
        let b = binding("b1", Some("someFunction(event) && whatever"));
        match &dispatch(&mut l, &[b], &event())[0] {
            DispatchOutcome::Skipped {
                reason: SkipReason::ConditionUnevaluatable { expression, .. },
                ..
            } => assert!(expression.contains("someFunction"), "{expression}"),
            other => panic!("expected an unevaluatable skip, got {other:?}"),
        }
        assert_eq!(l.len(), 0, "an unevaluatable condition must not fire");
    }

    #[test]
    fn a_blank_condition_is_treated_as_absent() {
        let mut l = Ledger::new();
        // Distinct ids: the idempotency key is derived from the binding id, so
        // reusing one would make the second case a legitimate Duplicate and the
        // test would be asserting the wrong thing.
        for (i, c) in [Some(""), Some("   "), None].into_iter().enumerate() {
            let mut b = binding("b1", c);
            b.id = format!("blank-{i}");
            let out = dispatch(&mut l, &[b], &event());
            assert!(
                matches!(out[0], DispatchOutcome::Reserved { .. }),
                "{c:?} -> {out:?}"
            );
        }
    }

    #[test]
    fn the_bindings_per_event_cap_is_enforced_deterministically() {
        let mut l = Ledger::new();
        let bs: Vec<TriggerBinding> = (0..8)
            .map(|i| {
                let mut b = binding(&format!("b{i}"), None);
                b.sort_order = i;
                b
            })
            .collect();
        let out = dispatch(&mut l, &bs, &event());
        let reserved: Vec<&str> = out
            .iter()
            .filter_map(|o| match o {
                DispatchOutcome::Reserved { binding_id, .. } => Some(binding_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reserved.len(), MAX_BINDINGS_PER_EVENT);
        // Deterministic: the first five by sort_order, not whichever the map
        // happened to yield.
        assert_eq!(reserved, vec!["b0", "b1", "b2", "b3", "b4"]);
        assert!(out.iter().any(|o| matches!(
            o,
            DispatchOutcome::Skipped { reason: SkipReason::TooManyBindings, .. }
        )));
    }

    // --- the loop guards, through the dispatcher --------------------------

    #[test]
    fn a_binding_cannot_handle_the_outcome_of_the_run_it_started() {
        let mut l = Ledger::new();
        let bs = [TriggerBinding {
            event: "flow.succeeded".into(),
            ..binding("b1", None)
        }];

        // b1 starts a run from the call event...
        let mut call = event();
        call.name = "aokie.call.incoming".into();
        let started = match &dispatch(&mut l, &[binding("b1", None)], &call)[0] {
            DispatchOutcome::Reserved { run, .. } => run.clone(),
            other => panic!("{other:?}"),
        };

        // ...and that run's success event comes back to b1.
        let mut outcome = event();
        outcome.name = "flow.succeeded".into();
        outcome.idempotency_key = "flow:succeeded:1".into();
        outcome.origin_run = Some(LineageRef {
            root_run_id: Some(started.run_id.clone()),
            parent_run_id: Some(started.run_id.clone()),
            binding_id: Some("b1".into()),
            depth: 1,
        });

        match &dispatch(&mut l, &bs, &outcome)[0] {
            DispatchOutcome::Skipped { reason: SkipReason::Guard(r), .. } => {
                assert!(r.contains("b1"), "{r}")
            }
            other => panic!("expected a guard refusal, got {other:?}"),
        }
    }

    #[test]
    fn depth_accumulates_so_a_chain_is_bounded() {
        let mut l = Ledger::new();
        let mut e = event();
        e.origin_run = Some(LineageRef {
            root_run_id: Some("root".into()),
            parent_run_id: Some("parent".into()),
            binding_id: Some("upstream".into()),
            depth: 7,
        });
        match &dispatch(&mut l, &[binding("b1", None)], &e)[0] {
            DispatchOutcome::Reserved { run, .. } => {
                assert_eq!(run.lineage.depth, 8, "each hop must increment");
                assert_eq!(run.lineage.root_run_id.as_deref(), Some("root"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_cycle_stops_at_the_depth_cap() {
        // The end-to-end property: an unguarded cycle is a fork bomb, so drive
        // one and assert it terminates.
        let mut l = Ledger::new();
        let bs = [TriggerBinding {
            event: "flow.succeeded".into(),
            ..binding("looper", None)
        }];
        let mut depth = 0u32;
        let mut reserved = 0usize;
        loop {
            let mut e = event();
            e.name = "flow.succeeded".into();
            e.idempotency_key = format!("cycle:{depth}");
            e.origin_run = Some(LineageRef {
                root_run_id: Some(format!("root{depth}")),
                parent_run_id: Some(format!("p{depth}")),
                binding_id: Some("other".into()),
                depth,
            });
            let out = dispatch(&mut l, &bs, &e);
            match &out[0] {
                DispatchOutcome::Reserved { .. } => reserved += 1,
                DispatchOutcome::Skipped { reason: SkipReason::Guard(_), .. } => break,
                other => panic!("unexpected {other:?}"),
            }
            depth += 1;
            assert!(depth < 100, "the cycle did not terminate");
        }
        assert!(reserved > 0 && reserved <= 17, "reserved {reserved}");
    }

    #[test]
    fn every_skip_reason_explains_itself() {
        for r in [
            SkipReason::Disabled,
            SkipReason::ManualMode,
            SkipReason::ConditionFalse,
            SkipReason::ConditionUnevaluatable {
                expression: "f(x)".into(),
                why: "unsupported".into(),
            },
            SkipReason::Guard("depth".into()),
            SkipReason::Duplicate,
            SkipReason::TooManyBindings,
        ] {
            let m = r.message();
            assert!(m.len() > 15, "too terse: {m:?}");
        }
    }
}
