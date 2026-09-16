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
//! # Conditions are evaluated on ZIPP, not in a grammar of our own
//!
//! [`eval_condition`] used to decide them: a deliberately small language of
//! selector comparisons joined by `&&` / `||`, which **refused what it could not
//! parse**. Refusing is the safe direction, but the author wrote JavaScript and
//! the browser runs JavaScript — so a pair of parentheses was enough to make
//! this desktop disagree with the web app about the same binding.
//!
//! The condition now goes to ZIPP through the warm script host
//! ([`super::conditions`]), and ZIPP's answer is the one used. `eval_condition`
//! runs alongside it for ONE release as a shadow, so every disagreement is
//! logged as a `condition-shadow` line and classified; PR9 deletes it.
//!
//! The one thing this must never do is treat an unparseable, erroring or
//! UNEVALUATED condition as true, which would turn a typo — or an engine that
//! is not running — into a binding that fires on everything.
//!
//! # Verdicts are decided BEFORE the ledger is locked
//!
//! [`dispatch`] runs under the ledger lock, because the reserve calls must be
//! serialised. Evaluating a condition there would put the whole plugin system
//! behind a script host that may be starting a child process. So the caller
//! decides every condition first — [`evaluate_conditions`], one batch, no locks
//! held — and hands [`dispatch`] the answers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::conditions::{self, ConditionJob, ShadowContext, Verdict};
use super::ledger::{Ledger, LineageRef, ReserveOutcome, RunRecord, RunRequest};
use super::script_host::ScriptBatch;

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
///
/// `verdicts` are the conditions, ALREADY DECIDED — see [`evaluate_conditions`].
/// This function evaluates nothing: it holds the ledger lock, and the script
/// host takes a process-wide lock of its own and may spawn a child, so asking it
/// anything from in here would serialise every plugin event in the process
/// behind a script host that is starting. A binding whose condition is missing
/// from `verdicts` is skipped, not fired — an unevaluated condition is exactly
/// as unsafe as an unevaluatable one.
pub fn dispatch(
    ledger: &mut Ledger,
    bindings: &[TriggerBinding],
    event: &Event,
    verdicts: &Verdicts,
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

        if let Some(expr) = condition_of(b) {
            let verdict = verdicts.get(&b.id).cloned().unwrap_or_else(|| {
                // Nobody decided this one. Fail towards not running: the
                // alternative is firing on a condition nothing has read.
                Verdict::Unknown("its condition was not evaluated before this dispatch".into())
            });
            match verdict {
                Verdict::True => {}
                Verdict::False => {
                    out.push(DispatchOutcome::Skipped {
                        binding_id: b.id.clone(),
                        reason: SkipReason::ConditionFalse,
                    });
                    continue;
                }
                Verdict::Unknown(why) => {
                    // The important branch. An unevaluatable condition must never
                    // read as true — that turns a typo, or an engine that is not
                    // running, into a binding that fires on everything.
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

        // Inputs resolve at dispatch, not at claim: the worker holds a record,
        // not the event, so the mapping must already be applied.
        let inputs = resolve_inputs(&b.input_map, event);
        let req = RunRequest {
            caller_product: format!("oaiy-trigger:{}", event.source),
            flow_id: Some(b.flow_id.clone()),
            inline_graph: false,
            input: if inputs.is_empty() { None } else { Some(Value::Object(inputs)) },
            timeout_ms: None,
            mode: "async".into(),
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

/// The trimmed, non-empty condition of a binding, or `None` when it has none.
///
/// A blank condition is not a condition: it must fire, and it must cost no
/// engine call at all.
pub fn condition_of(b: &TriggerBinding) -> Option<&str> {
    b.condition.as_deref().map(str::trim).filter(|s| !s.is_empty())
}

/// Every condition of one dispatch, decided.
///
/// Built by [`evaluate_conditions`] with no lock held, then handed to
/// [`dispatch`], which holds the ledger lock and asks nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Verdicts {
    by_binding: BTreeMap<String, Verdict>,
    /// `condition-shadow` lines this dispatch produced. Already logged; kept so
    /// a test can read them without capturing the log.
    pub shadow: Vec<String>,
}

impl Verdicts {
    /// Verdicts from `(binding id, verdict)` pairs.
    pub fn from_pairs<I: IntoIterator<Item = (String, Verdict)>>(pairs: I) -> Self {
        Verdicts { by_binding: pairs.into_iter().collect(), shadow: Vec::new() }
    }

    pub fn get(&self, binding_id: &str) -> Option<&Verdict> {
        self.by_binding.get(binding_id)
    }

    pub fn len(&self) -> usize {
        self.by_binding.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_binding.is_empty()
    }
}

/// The names a trigger condition may read.
///
/// `event` and `$event` are the SAME object. The old grammar accepted both
/// spellings ([`operand`] maps a bare `event.…` to `$event.…`), and the trigger
/// editor emits the undollared one, so both have to exist or half the stored
/// conditions stop resolving.
pub fn event_globals(event: &Event) -> Value {
    let e = json!({
        "name": event.name,
        "source": event.source,
        "correlationId": event.correlation_id,
        "idempotencyKey": event.idempotency_key,
        "data": event.data,
    });
    json!({ "event": e, "$event": e })
}

/// The bindings whose conditions this event has to decide, in dispatch order.
///
/// A superset of what the old in-line evaluation reached: [`dispatch`] checked
/// [`MAX_BINDINGS_PER_EVENT`] before the condition, so once five had RESERVED it
/// stopped evaluating. Nothing here knows yet which will reserve — a duplicate
/// or a guard refusal does not count against the cap — so every matching,
/// enabled, non-manual, conditioned binding is evaluated. That is already the
/// old code's worst case (an event where nothing reserves evaluated all of
/// them), and it buys the property that matters: `dispatch` asks no engine.
fn conditioned<'a>(bindings: &'a [TriggerBinding], event: &Event) -> Vec<&'a TriggerBinding> {
    let mut matched: Vec<&TriggerBinding> = bindings
        .iter()
        .filter(|b| b.event == event.name && b.enabled && b.mode != BindingMode::Manual)
        .filter(|b| condition_of(b).is_some())
        .collect();
    matched.sort_by(|a, b| a.sort_order.cmp(&b.sort_order).then(a.id.cmp(&b.id)));
    matched
}

/// The batch this event sends: ONE request, one job per condition.
pub fn condition_jobs(bindings: &[TriggerBinding], event: &Event) -> Vec<ConditionJob> {
    jobs_for(&conditioned(bindings, event), event)
}

/// One job per binding in `matched`, in that order — so the `n`th answer is the
/// `n`th binding's, structurally rather than by two functions agreeing.
fn jobs_for(matched: &[&TriggerBinding], event: &Event) -> Vec<ConditionJob> {
    if matched.is_empty() {
        return Vec::new();
    }
    let globals = event_globals(event);
    matched
        .iter()
        .enumerate()
        .map(|(i, b)| ConditionJob {
            id: ConditionJob::id_for(i),
            source: condition_of(b).unwrap_or_default().to_string(),
            globals: globals.clone(),
        })
        .collect()
}

/// Decide every condition of this event on ZIPP, and shadow each answer against
/// the Rust grammar.
///
/// Call this with NO lock held. It takes the script host's process-wide batch
/// lock and may spawn a child; [`dispatch`] then runs under the ledger lock with
/// the answers already in hand.
pub fn evaluate_conditions(
    host: &dyn ScriptBatch,
    bindings: &[TriggerBinding],
    event: &Event,
) -> Verdicts {
    let matched = conditioned(bindings, event);
    let jobs = jobs_for(&matched, event);
    let answers = conditions::decide(host, &jobs);
    let mut out = Verdicts::default();
    for (b, answer) in matched.iter().zip(answers) {
        let expr = condition_of(b).unwrap_or_default();
        if answer.answered {
            let rust = rust_shadow(expr, event);
            let ctx = ShadowContext {
                lane: "triggers",
                binding: &b.id,
                event: &event.name,
                source: &event.source,
                idempotency_key: &event.idempotency_key,
                expr,
            };
            if let Some(line) = conditions::shadow_line(&ctx, &answer.verdict, &rust) {
                log::warn!("{line}");
                out.shadow.push(line);
            }
        }
        out.by_binding.insert(b.id.clone(), answer.verdict);
    }
    out
}

/// The Rust grammar's reading, as a [`Verdict`] — for the shadow log only.
fn rust_shadow(expr: &str, event: &Event) -> Verdict {
    match eval_condition(expr, event) {
        Ok(true) => Verdict::True,
        Ok(false) => Verdict::False,
        Err(why) => Verdict::Unknown(why),
    }
}

// ---------------------------------------------------------------------------
// The one-time `$event` migration
// ---------------------------------------------------------------------------

/// One condition rewritten by [`migrate_bindings`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionRewrite {
    pub binding_id: String,
    pub before: String,
    pub after: String,
}

/// The tokens that meant `event.data` to the Rust grammar and mean the whole
/// envelope to ZIPP. Longest first, so `$event` is recognised as one token.
const BARE_EVENT_TOKENS: [&str; 2] = ["$event", "event"];

fn ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// Rewrite a bare `event` / `$event` operand to `<token>.data`.
///
/// [`resolve_selector`] answers `$event` alone with `event.data` (the `None`
/// arm), and [`operand`] routes a bare `event` to that same arm. Under ZIPP the
/// `event` global is the whole envelope, which is truthy even when `data` is not
/// — the FIRING direction, so this is not a difference that can be left to
/// settle. The stored text is rewritten once, on load, and the old reading is
/// kept nowhere.
///
/// Returns `None` when there is nothing to change, which is what leaves an
/// already-migrated file byte-identical: no rewrite, no write, no `.bak`.
///
/// **Idempotent by construction.** The rewritten token is followed by `.`, and a
/// token followed by `.` is not bare, so a second pass matches nothing.
///
/// Quoting follows [`split_top`] exactly — a `'` or `"` opens a span the scan
/// steps over, with no escape handling — so this and the evaluator always agree
/// about which bytes are code.
pub fn migrate_condition(expr: &str) -> Option<String> {
    let bytes = expr.as_bytes();
    let mut out = String::new();
    let mut last = 0usize;
    let mut i = 0usize;
    let mut quote: Option<u8> = None;
    let mut rewrote = false;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = quote {
            if b == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        if b == b'\'' || b == b'"' {
            quote = Some(b);
            i += 1;
            continue;
        }
        let mut end = None;
        for token in BARE_EVENT_TOKENS {
            let t = token.as_bytes();
            if !bytes[i..].starts_with(t) {
                continue;
            }
            // Bare means: not part of a longer name, and not a property read.
            let before_ok = i == 0 || !(ident_byte(bytes[i - 1]) || bytes[i - 1] == b'.');
            let after = i + t.len();
            let after_ok = after >= bytes.len() || !(ident_byte(bytes[after]) || bytes[after] == b'.');
            if before_ok && after_ok {
                end = Some(after);
                break;
            }
        }
        match end {
            Some(end) => {
                out.push_str(&expr[last..end]);
                out.push_str(".data");
                last = end;
                i = end;
                rewrote = true;
            }
            None => i += 1,
        }
    }
    if !rewrote {
        return None;
    }
    out.push_str(&expr[last..]);
    Some(out)
}

/// Apply [`migrate_condition`] to every binding, in place.
///
/// Only the CONDITION text moves. `inputMap` selectors keep the Rust semantics
/// (`$event` alone is `event.data`) because they are resolved by
/// [`resolve_selector`] and never by an engine — rewriting them would change
/// what a flow receives for no reason at all.
pub fn migrate_bindings(bindings: &mut [TriggerBinding]) -> Vec<ConditionRewrite> {
    let mut rewrites = Vec::new();
    for b in bindings.iter_mut() {
        let Some(before) = b.condition.clone() else {
            continue;
        };
        // The stored string as stored, untrimmed: a migration must not also
        // reformat, or "already migrated" stops meaning "byte-identical".
        let Some(after) = migrate_condition(&before) else {
            continue;
        };
        b.condition = Some(after.clone());
        rewrites.push(ConditionRewrite { binding_id: b.id.clone(), before, after });
    }
    rewrites
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

/// The Rust grammar's reading of a condition — a SHADOW, for one release.
///
/// Supported: `<selector-or-literal> <op> <selector-or-literal>` where op is one
/// of `===`, `!==`, `==`, `!=`, joined by `&&` or `||`. A bare selector is
/// truthy-tested. Everything else is an error.
///
/// `&&` binds tighter than `||`, as in JS, so `a || b && c` is `a || (b && c)`.
///
/// Nothing decides anything on this any more: [`evaluate_conditions`] calls it
/// only to compare with ZIPP's answer. PR9 deletes it, together with
/// [`split_top`], [`operand`] and [`values_equal`], once every
/// `condition-shadow` line from the shadow release has been classified.
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

    // --- fixtures ---------------------------------------------------------

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

    /// Dispatch with the conditions read the way the RUST grammar reads them.
    ///
    /// The ledger, lineage, guard and cap tests below are about `dispatch`, not
    /// about where its verdicts came from, so this keeps every one of their
    /// expectations exactly as it was.
    fn dispatch_rust(
        l: &mut Ledger,
        bs: &[TriggerBinding],
        e: &Event,
    ) -> Vec<DispatchOutcome> {
        let verdicts = Verdicts::from_pairs(
            bs.iter()
                .filter_map(|b| condition_of(b).map(|expr| (b.id.clone(), rust_shadow(expr, e)))),
        );
        dispatch(l, bs, e, &verdicts)
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
        let out = dispatch_rust(&mut l, &[binding("b1", None)], &event());
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
        assert!(dispatch_rust(&mut l, &[binding("b1", None)], &e).is_empty());
    }

    #[test]
    fn the_same_event_delivered_twice_reserves_once() {
        // At-least-once delivery is the norm; without this a redelivered call
        // event runs the flow again.
        let mut l = Ledger::new();
        let bs = [binding("b1", None)];
        let first = dispatch_rust(&mut l, &bs, &event());
        let second = dispatch_rust(&mut l, &bs, &event());
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
        match &dispatch_rust(&mut l, &[b], &event())[0] {
            DispatchOutcome::Skipped { reason, .. } => assert_eq!(*reason, SkipReason::Disabled),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_manual_binding_never_receives_events() {
        let mut l = Ledger::new();
        let mut b = binding("b1", None);
        b.mode = BindingMode::Manual;
        match &dispatch_rust(&mut l, &[b], &event())[0] {
            DispatchOutcome::Skipped { reason, .. } => assert_eq!(*reason, SkipReason::ManualMode),
            other => panic!("{other:?}"),
        }
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn a_false_condition_skips() {
        let mut l = Ledger::new();
        let b = binding("b1", Some("event.data.flowId === 'nope'"));
        match &dispatch_rust(&mut l, &[b], &event())[0] {
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
        match &dispatch_rust(&mut l, &[b], &event())[0] {
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
            let out = dispatch_rust(&mut l, &[b], &event());
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
        let out = dispatch_rust(&mut l, &bs, &event());
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
        let started = match &dispatch_rust(&mut l, &[binding("b1", None)], &call)[0] {
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

        match &dispatch_rust(&mut l, &bs, &outcome)[0] {
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
        match &dispatch_rust(&mut l, &[binding("b1", None)], &e)[0] {
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
            let out = dispatch_rust(&mut l, &bs, &e);
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

    // --- conditions on ZIPP -----------------------------------------------

    use crate::bridge::conditions::testing::{guest, FakeHost};

    fn cond(id: &str, expr: &str) -> TriggerBinding {
        binding(id, Some(expr))
    }

    #[test]
    fn every_condition_of_one_event_travels_in_one_batch() {
        // The host serialises one batch at a time process-wide, so sending a
        // batch per condition would queue this event's own conditions behind
        // each other — for conditions that take microseconds.
        let bs = vec![
            cond("b1", "event.data.known === true"),
            binding("b2", None),
            cond("b3", "event.data.attempts === 2"),
        ];
        let host = FakeHost::always(json!(true));
        evaluate_conditions(&host, &bs, &event());

        let request = host.only_request();
        assert_eq!(request["v"], 1);
        let jobs = request["jobs"].as_array().expect("jobs");
        assert_eq!(jobs.len(), 2, "the unconditioned binding is not a job");
        assert_eq!(jobs[0]["id"], "c0");
        assert_eq!(jobs[0]["mode"], "program");
        assert_eq!(jobs[0]["source"], "event.data.known === true");
        assert_eq!(jobs[0]["budgetMs"], 250);
        assert_eq!(jobs[1]["source"], "event.data.attempts === 2");
    }

    #[test]
    fn the_condition_sees_the_event_under_both_of_its_names() {
        // The old grammar accepted `event.…` and `$event.…` alike, and the
        // trigger editor emits the undollared one. Both names must exist, and
        // they must be the SAME object, or half the stored conditions stop
        // resolving and the other half read something different.
        let host = FakeHost::always(json!(true));
        let e = event();
        evaluate_conditions(&host, &[cond("b1", "event.data.known")], &e);

        let globals = host.only_request()["jobs"][0]["globals"].clone();
        assert_eq!(globals["event"], globals["$event"]);
        assert_eq!(
            globals["event"],
            json!({
                "name": "aokie.call.incoming",
                "source": "aokie",
                "correlationId": "call_1",
                "idempotencyKey": "aokie:call_1:incoming:v1",
                "data": e.data,
            })
        );
    }

    #[test]
    fn only_a_binding_that_could_still_fire_is_evaluated() {
        let mut disabled = cond("b1", "event.data.known");
        disabled.enabled = false;
        let mut manual = cond("b2", "event.data.known");
        manual.mode = BindingMode::Manual;
        let mut other_event = cond("b3", "event.data.known");
        other_event.event = "aokie.call.ended".into();
        let blank = binding("b4", Some("   "));

        let host = FakeHost::always(json!(true));
        let verdicts = evaluate_conditions(&host, &[disabled, manual, other_event, blank], &event());
        assert!(verdicts.is_empty());
        assert_eq!(host.calls(), 0, "none of these can fire, so none of them costs an engine");
    }

    #[test]
    fn zipps_verdict_is_the_one_used_not_the_rust_grammars() {
        let mut l = Ledger::new();
        let bs = [cond("b1", "event.data.known === true")];

        // The Rust grammar reads this as TRUE. ZIPP says false, and ZIPP decides.
        let no = FakeHost::always(json!(false));
        let verdicts = evaluate_conditions(&no, &bs, &event());
        match &dispatch(&mut l, &bs, &event(), &verdicts)[0] {
            DispatchOutcome::Skipped { reason, .. } => assert_eq!(*reason, SkipReason::ConditionFalse),
            other => panic!("{other:?}"),
        }
        assert_eq!(l.len(), 0);

        // And an expression the Rust grammar REFUSES outright — parentheses —
        // which ZIPP evaluates, now fires.
        let parens = [cond("b2", "(event.data.known === true) && event.data.attempts === 2")];
        let yes = FakeHost::always(json!(true));
        let verdicts = evaluate_conditions(&yes, &parens, &event());
        assert!(matches!(
            dispatch(&mut l, &parens, &event(), &verdicts)[0],
            DispatchOutcome::Reserved { .. }
        ));
    }

    #[test]
    fn agreement_leaves_no_shadow_line() {
        let host = FakeHost::always(json!(true));
        let verdicts = evaluate_conditions(&host, &[cond("b1", "event.data.known === true")], &event());
        assert_eq!(verdicts.get("b1"), Some(&Verdict::True));
        assert!(verdicts.shadow.is_empty(), "a shadow line means the two disagreed");
    }

    #[test]
    fn each_class_of_disagreement_produces_its_own_line() {
        let e = event();

        // (i)/(ii) — a Rust-grammar REFUSAL that ZIPP evaluates. Parentheses are
        // the commonest: the old evaluator would not group, so it refused.
        let host = FakeHost::always(json!(true));
        let v = evaluate_conditions(&host, &[cond("b1", "(event.data.known === true)")], &e);
        assert_eq!(v.shadow.len(), 1);
        let line = &v.shadow[0];
        assert!(line.starts_with("condition-shadow: lane=triggers "), "{line}");
        assert!(line.contains(r#"binding="b1""#), "{line}");
        assert!(line.contains("zipp=true rust=unknown refused=rust"), "{line}");
        assert!(line.contains(r#"expr="(event.data.known === true)""#), "{line}");
        assert!(line.contains("parentheses"), "the reason has to be classifiable: {line}");
        assert!(
            line.contains(r#"key="aokie:call_1:incoming:v1""#),
            "and the occurrence findable: {line}"
        );

        // (iii) — a real semantic change. `==` coerced in JavaScript; the Rust
        // grammar compared JSON values, so `2 == '2'` was FALSE.
        let v = evaluate_conditions(&host, &[cond("b2", "event.data.attempts == '2'")], &e);
        assert_eq!(v.shadow.len(), 1);
        assert!(v.shadow[0].contains("zipp=true rust=false refused=none"), "{}", v.shadow[0]);

        // A condition ZIPP itself refuses, which the Rust grammar read.
        let throwing = FakeHost::by_source(|_| Err(guest("ReferenceError: x is not defined")));
        let v = evaluate_conditions(&throwing, &[cond("b3", "event.data.known === true")], &e);
        assert_eq!(v.shadow.len(), 1);
        assert!(v.shadow[0].contains("zipp=unknown rust=true refused=zipp"), "{}", v.shadow[0]);
    }

    #[test]
    fn a_host_outage_skips_every_conditioned_binding_and_shadows_nothing() {
        // Fail closed. The Rust grammar is NOT the fallback: it would change
        // semantics under the user mid-outage, in the firing direction, and it
        // would fill the shadow log with lines nobody can classify — and any
        // unclassified line blocks PR9 from deleting the grammar at all.
        let mut l = Ledger::new();
        let bs = [cond("b1", "event.data.known === true"), binding("b2", None)];
        let down = FakeHost::down("the CLI is not installed");
        let verdicts = evaluate_conditions(&down, &bs, &event());

        assert!(verdicts.shadow.is_empty(), "an outage is an outage, not a disagreement");
        let out = dispatch(&mut l, &bs, &event(), &verdicts);
        match &out[0] {
            DispatchOutcome::Skipped {
                reason: SkipReason::ConditionUnevaluatable { why, .. },
                ..
            } => assert!(why.contains("the CLI is not installed"), "{why}"),
            other => panic!("a conditioned binding must not fire during an outage: {other:?}"),
        }
        assert!(
            matches!(out[1], DispatchOutcome::Reserved { .. }),
            "a binding with no condition is unaffected — it never needed the engine"
        );
    }

    #[test]
    fn a_condition_nobody_decided_does_not_fire() {
        // `dispatch` evaluates nothing, by construction — it holds the ledger
        // lock. A verdict that never arrived must read as unknown, never true.
        let mut l = Ledger::new();
        let bs = [cond("b1", "event.data.known === true")];
        let out = dispatch(&mut l, &bs, &event(), &Verdicts::default());
        assert!(matches!(
            &out[0],
            DispatchOutcome::Skipped { reason: SkipReason::ConditionUnevaluatable { .. }, .. }
        ));
        assert_eq!(l.len(), 0);
    }

    // --- the one-time `$event` migration ----------------------------------

    #[test]
    fn a_bare_event_operand_gains_the_data_it_always_meant() {
        // `resolve_selector` answers `$event` alone with `event.data`, and
        // `operand` routes a bare `event` to the same arm. ZIPP binds `event` to
        // the whole envelope, which is truthy even when `data` is not — so
        // leaving this alone would fire bindings that used to skip.
        assert_eq!(migrate_condition("$event").as_deref(), Some("$event.data"));
        assert_eq!(migrate_condition("event").as_deref(), Some("event.data"));
        assert_eq!(
            migrate_condition("event && event.data.known === true").as_deref(),
            Some("event.data && event.data.known === true")
        );
        assert_eq!(
            migrate_condition("$event.data.x === 1 || $event").as_deref(),
            Some("$event.data.x === 1 || $event.data")
        );
    }

    #[test]
    fn a_condition_already_in_the_new_form_is_not_touched() {
        // This is what leaves an already-migrated file byte-identical: nothing
        // changes, so nothing is written and no `.bak` is made.
        for untouched in [
            "$event.data",
            "event.data.known === true",
            "$event.data.x === 1 && $event.name === 'y'",
            "$eventually === 1",
            "event.data.msg === 'event'",
            "event.data.msg === \"$event\"",
            "eventCount === 2",
            "x.event === 1",
        ] {
            assert_eq!(migrate_condition(untouched), None, "{untouched:?} must be left alone");
        }
    }

    #[test]
    fn the_migration_is_idempotent() {
        // It runs on every load. Running it twice must change nothing the
        // second time, or a user's condition grows a `.data` per boot.
        for before in [
            "$event",
            "event",
            "event && event.data.known",
            "$event === null || event",
            "  event  ",
            "!event",
        ] {
            let once = migrate_condition(before).expect("the first pass rewrites");
            assert_eq!(migrate_condition(&once), None, "{before:?} -> {once:?} is not stable");
        }
    }

    #[test]
    fn a_quoted_event_is_text_not_an_operand() {
        // The scan follows `split_top`'s quoting exactly, so the migration and
        // the evaluator can never disagree about which bytes are code.
        assert_eq!(migrate_condition("event.data.msg === 'event'"), None);
        assert_eq!(migrate_condition("event.data.msg === '$event'"), None);
        assert_eq!(
            migrate_condition("event === 'event'").as_deref(),
            Some("event.data === 'event'"),
            "the operand moves and the literal does not"
        );
    }

    #[test]
    fn migrate_bindings_reports_what_changed_and_leaves_the_rest_alone() {
        let mut bs = vec![
            cond("b1", "event && event.data.known"),
            cond("b2", "event.data.known === true"),
            binding("b3", None),
        ];
        let rewrites = migrate_bindings(&mut bs);
        assert_eq!(rewrites.len(), 1, "only the binding that changed is reported");
        assert_eq!(rewrites[0].binding_id, "b1");
        assert_eq!(rewrites[0].before, "event && event.data.known");
        assert_eq!(rewrites[0].after, "event.data && event.data.known");
        assert_eq!(bs[0].condition.as_deref(), Some("event.data && event.data.known"));
        assert_eq!(bs[1].condition.as_deref(), Some("event.data.known === true"));
        assert!(bs[2].condition.is_none());
        // Idempotent at the collection level too.
        assert!(migrate_bindings(&mut bs).is_empty());
    }

    #[test]
    fn input_map_selectors_keep_their_meaning() {
        // `$event` in an inputMap is resolved by `resolve_selector`, never by an
        // engine, so `$event` alone still means the data. Rewriting it would
        // change what a flow receives for no reason at all.
        let e = event();
        assert_eq!(resolve_selector("$event", &e), Some(e.data.clone()));
        let mut map = std::collections::BTreeMap::new();
        map.insert("all".to_string(), "$event".to_string());
        assert_eq!(resolve_inputs(&map, &e)["all"], e.data);
    }
}
