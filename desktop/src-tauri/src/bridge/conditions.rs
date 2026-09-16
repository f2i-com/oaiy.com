//! Deciding trigger and binding conditions on ZIPP.
//!
//! A condition is the one place a user's own JavaScript decides whether an
//! automation fires. Two hand-written Rust grammars used to answer that —
//! [`crate::link::condition`] for the linked account's flow bindings and
//! [`super::triggers::eval_condition`] for local triggers — each a deliberately
//! small language that **refused what it could not parse**. Refusing is the safe
//! direction, but it is still wrong: the author wrote JavaScript, the browser
//! runs JavaScript, and a pair of parentheses was enough to make this desktop
//! disagree with the web app about the same binding.
//!
//! This module sends the expression to ZIPP instead, through the warm script
//! host, and reads the answer back. The grammars stay for ONE release as a
//! shadow: both answer, ZIPP's answer is the one used, and a disagreement is
//! logged as a `condition-shadow` line for review. PR9 deletes the grammars once
//! those lines are all classified.
//!
//! # One batch per dispatch, not one per condition
//!
//! [`super::script_host::ScriptHost::evaluate`] serialises one batch at a time
//! process-wide. Sending a batch per condition would queue one event's own
//! conditions behind each other for no reason, so every condition of one
//! dispatch goes in ONE batch, and each job carries a small
//! [`CONDITION_BUDGET_MS`] — a condition that needs a quarter of a second is not
//! a condition.
//!
//! # A host outage is an outage, not a disagreement
//!
//! When the host cannot serve, every condition is [`Verdict::Unknown`] and every
//! conditioned binding is skipped. The Rust grammar does **not** quietly take
//! over: it is a shadow for one release, and a fallback that decided would both
//! change semantics under the user mid-outage (in the firing direction, which is
//! the dangerous one) and poison the very log the shadow release exists to
//! collect. For the same reason an outage produces **no** `condition-shadow`
//! line — those lines mean "the two grammars disagreed", and PR9's deletion gate
//! is blocked by any line nobody can classify.

use serde_json::{json, Value};

use super::script_host::{batch_request, HostError, JobResult, ScriptBatch};

/// Per-condition wall-clock budget. Conditions are property reads and
/// comparisons; anything that needs longer has already gone wrong, and the
/// host's batch deadline is the sum of these plus a grace each.
pub const CONDITION_BUDGET_MS: u64 = 250;

/// How long the save-time `parse` check may take. Same reasoning; it compiles
/// one expression and never runs it.
pub const CHECK_BUDGET_MS: u64 = 250;

/// What a condition decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    True,
    False,
    /// Not decided. Never fires, and says why.
    Unknown(String),
}

impl Verdict {
    pub fn fires(&self) -> bool {
        matches!(self, Verdict::True)
    }

    /// The word a `condition-shadow` line uses.
    pub fn label(&self) -> &'static str {
        match self {
            Verdict::True => "true",
            Verdict::False => "false",
            Verdict::Unknown(_) => "unknown",
        }
    }

    /// The reason, for an `Unknown`.
    pub fn why(&self) -> Option<&str> {
        match self {
            Verdict::Unknown(why) => Some(why.as_str()),
            _ => None,
        }
    }

    /// Do these two DECIDE the same thing?
    ///
    /// Compares the decision, not the prose: two `Unknown`s skip the binding
    /// identically, and their reasons are written by two different grammars, so
    /// requiring the text to match would log every unparseable condition as a
    /// disagreement forever.
    pub fn agrees_with(&self, other: &Verdict) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

/// One condition to decide, and the data it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionJob {
    /// Unique within the batch. Synthetic (`c0`, `c1`, …) rather than the
    /// binding id: a binding id is the provider's string, and the request
    /// schema bounds a job id at 1..128 characters.
    pub id: String,
    /// The author's expression, verbatim. Never rewritten on the way out.
    pub source: String,
    /// The names the expression may read, as a JSON object. It crosses into the
    /// guest as a JSON literal, never as code.
    pub globals: Value,
}

impl ConditionJob {
    /// The batch id for the `n`th condition of a dispatch.
    pub fn id_for(n: usize) -> String {
        format!("c{n}")
    }

    /// This job as the `script-request` schema describes it.
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "mode": "program",
            "source": self.source,
            "globals": self.globals,
            "budgetMs": CONDITION_BUDGET_MS,
        })
    }
}

/// One condition's answer from ZIPP.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionVerdict {
    pub verdict: Verdict,
    /// Did ZIPP answer THIS condition?
    ///
    /// True when the job ran — whether it produced a value or threw. False when
    /// the host never served the batch, timed out, ran out of resources, or left
    /// this job unanswered. Only an answered condition may produce a
    /// `condition-shadow` line: an outage is not a grammar disagreement, and a
    /// log full of outages would block PR9's deletion gate forever.
    pub answered: bool,
}

impl ConditionVerdict {
    fn outage(why: String) -> Self {
        ConditionVerdict { verdict: Verdict::Unknown(why), answered: false }
    }
}

/// The request this dispatch sends: ONE batch, one job per condition.
pub fn build_batch(jobs: &[ConditionJob]) -> Value {
    batch_request(jobs.iter().map(ConditionJob::to_json).collect())
}

/// Decide every condition of one dispatch, in one batch.
///
/// Never panics and never returns short: there is exactly one
/// [`ConditionVerdict`] per input job, in the same order, whatever the host did.
pub fn decide(host: &dyn ScriptBatch, jobs: &[ConditionJob]) -> Vec<ConditionVerdict> {
    if jobs.is_empty() {
        // The load-bearing case. An event whose bindings carry no conditions
        // must not cost a script host — which, on the first such event, would
        // mean spawning one.
        return Vec::new();
    }
    let request = build_batch(jobs);
    let response = match host.run(&request) {
        Ok(r) => r,
        Err(e) => {
            // ONE line for the outage, not one per binding: the bindings did not
            // disagree about anything, the engine was not there.
            log::warn!(
                "condition-host: {} condition(s) could not be evaluated ({e}); every one of them is unknown, so nothing fires",
                jobs.len()
            );
            return jobs.iter().map(|_| ConditionVerdict::outage(e.to_string())).collect();
        }
    };
    jobs.iter()
        .map(|job| match response.result(&job.id) {
            Some(result) => verdict_from(result),
            None => ConditionVerdict::outage(
                "the script host answered the batch without this condition".to_string(),
            ),
        })
        .collect()
}

/// One job's result as a verdict.
///
/// `ok` with a JS-truthy value fires; `ok` with a falsy one does not; anything
/// else is unknown, which also does not fire.
pub fn verdict_from(result: &JobResult) -> ConditionVerdict {
    match &result.outcome {
        Ok(value) => ConditionVerdict {
            verdict: if truthy(value) { Verdict::True } else { Verdict::False },
            answered: true,
        },
        Err(e) => ConditionVerdict {
            verdict: Verdict::Unknown(format!("{}: {}", e.kind, e.message)),
            // `guest` is the condition's own fault — it threw, or it did not
            // compile — so ZIPP HAS answered, and the Rust grammar may be
            // compared against it. Every other kind (`timeout`, `resource`,
            // `host`, `source`, `unsupported`, `prepare`) is the engine or the
            // wiring rather than the expression, and comparing against it would
            // file an outage as a semantic disagreement.
            answered: e.kind == "guest",
        },
    }
}

/// JavaScript truthiness of a value that came back through JSON.
///
/// `NaN` and `undefined` cannot survive JSON and arrive as `null`, which is
/// falsy — the same answer JavaScript gives for both. An empty array or object
/// is TRUTHY, which is the one rule people get wrong.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0 && !f.is_nan()).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

// ---------------------------------------------------------------------------
// The shadow log
// ---------------------------------------------------------------------------

/// Which lane a condition belongs to, and which occurrence of which binding.
///
/// Everything a reviewer needs to classify a disagreement WITHOUT re-running
/// it: the expression verbatim, both verdicts with both reasons, and enough
/// identity to find the event in the ring or the dead-letter queue.
#[derive(Debug, Clone, Copy)]
pub struct ShadowContext<'a> {
    /// `triggers` (this desktop's own bindings) or `flows` (the linked
    /// account's). Different grammars, different globals, different fix.
    pub lane: &'a str,
    pub binding: &'a str,
    pub event: &'a str,
    /// The plugin or account the event came from.
    pub source: &'a str,
    /// The event's idempotency key — the one field that pins the occurrence.
    pub idempotency_key: &'a str,
    pub expr: &'a str,
}

/// The `condition-shadow` line, or `None` when the two grammars agree.
///
/// Emitted only for an ANSWERED condition (see [`ConditionVerdict::answered`]).
/// One line per disagreement, with every free-text field JSON-quoted, so a whole
/// release's shadow log can be read by machine and every line classified without
/// reconstructing a single event.
pub fn shadow_line(ctx: &ShadowContext, zipp: &Verdict, rust: &Verdict) -> Option<String> {
    if zipp.agrees_with(rust) {
        return None;
    }
    let refused = match (zipp.why().is_some(), rust.why().is_some()) {
        (false, true) => "rust",
        (true, false) => "zipp",
        // Both refusing is not a disagreement (`agrees_with` already returned),
        // and neither refusing is the plain true-against-false case.
        _ => "none",
    };
    Some(format!(
        "condition-shadow: lane={} binding={} event={} source={} key={} zipp={} rust={} refused={} expr={} zipp_why={} rust_why={}",
        ctx.lane,
        quote(ctx.binding),
        quote(ctx.event),
        quote(ctx.source),
        quote(ctx.idempotency_key),
        zipp.label(),
        rust.label(),
        refused,
        quote(ctx.expr),
        why_field(zipp),
        why_field(rust),
    ))
}

/// A free-text field, JSON-quoted so a newline or a quote inside a condition
/// cannot break one line into two.
fn quote(s: &str) -> String {
    Value::String(s.to_string()).to_string()
}

fn why_field(v: &Verdict) -> String {
    match v.why() {
        Some(why) => quote(why),
        None => "-".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Save time
// ---------------------------------------------------------------------------

/// What the save-time check decided about one expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckOutcome {
    /// ZIPP compiled it as one expression.
    Parses,
    /// ZIPP refused it, and this is why — a 400 at the point the author wrote
    /// it, rather than a binding that looks right and never fires.
    Rejected(String),
    /// Nobody could ask. Save it and say so; dispatch still fails closed.
    Unchecked(String),
}

/// The request the save-time check sends: one `parse`-mode job.
///
/// `parse` compiles the source and never invokes it, and the CLI runs a real
/// single-expression parse (acorn, `cli/src/zipp/script-worker.ts`) before the
/// engine sees it — so `1); x = (1` is refused rather than quietly compiling
/// into two perfectly legal statements.
pub fn check_request(source: &str) -> Value {
    batch_request(vec![json!({
        "id": "check",
        "mode": "parse",
        "source": source,
        "budgetMs": CHECK_BUDGET_MS,
    })])
}

/// Can this expression be evaluated at all?
pub fn check(host: &dyn ScriptBatch, source: &str) -> CheckOutcome {
    let request = check_request(source);
    match host.run(&request) {
        Ok(response) => match response.result("check") {
            Some(JobResult { outcome: Ok(_), .. }) => CheckOutcome::Parses,
            Some(JobResult { outcome: Err(e), .. }) if e.kind == "guest" => {
                CheckOutcome::Rejected(e.message.clone())
            }
            Some(JobResult { outcome: Err(e), .. }) => {
                CheckOutcome::Unchecked(format!("{}: {}", e.kind, e.message))
            }
            None => CheckOutcome::Unchecked(
                "the script host answered without checking the condition".to_string(),
            ),
        },
        // The CLI refused the whole request — deterministic, this caller's
        // fault, nothing ran — so it is a 400 like any other bad expression.
        Err(HostError::Refused { message }) => CheckOutcome::Rejected(message),
        Err(e) => CheckOutcome::Unchecked(e.to_string()),
    }
}

/// A `ScriptBatch` for tests: answers from a closure and records every request.
///
/// Not behind `#[cfg(test)]` on the module, because the trigger, flow, plugin
/// and route tests all need the same one and a `#[cfg(test)]` module is visible
/// across the crate's test build.
#[cfg(test)]
pub mod testing {
    use super::*;
    use crate::bridge::script_host::{JobError, ScriptResponse};
    use std::sync::Mutex;

    type Answer = Box<dyn Fn(&Value) -> Result<ScriptResponse, HostError> + Send + Sync>;

    pub struct FakeHost {
        answer: Answer,
        seen: Mutex<Vec<Value>>,
    }

    impl FakeHost {
        /// Answer each job from its SOURCE.
        pub fn by_source<F>(f: F) -> Self
        where
            F: Fn(&str) -> Result<Value, JobError> + Send + Sync + 'static,
        {
            FakeHost::raw(move |request| {
                let results = request
                    .get("jobs")
                    .and_then(Value::as_array)
                    .map(|jobs| {
                        jobs.iter()
                            .map(|job| JobResult {
                                id: job["id"].as_str().unwrap_or_default().to_string(),
                                outcome: f(job["source"].as_str().unwrap_or_default()),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(ScriptResponse { engine: json!({ "name": "zipp" }), results })
            })
        }

        /// Answer every job with the same value.
        pub fn always(value: Value) -> Self {
            FakeHost::by_source(move |_| Ok(value.clone()))
        }

        /// Never answer: the host is down.
        pub fn down(reason: &str) -> Self {
            let reason = reason.to_string();
            FakeHost::raw(move |_| {
                Err(HostError::Unavailable { reason: reason.clone(), retry_after: None })
            })
        }

        /// Full control over the response.
        pub fn raw<F>(f: F) -> Self
        where
            F: Fn(&Value) -> Result<ScriptResponse, HostError> + Send + Sync + 'static,
        {
            FakeHost { answer: Box::new(f), seen: Mutex::new(Vec::new()) }
        }

        /// Every request this host was handed, in order.
        pub fn requests(&self) -> Vec<Value> {
            self.seen.lock().unwrap_or_else(|e| e.into_inner()).clone()
        }

        pub fn calls(&self) -> usize {
            self.requests().len()
        }

        /// The one request it was handed. Panics on any other count — which is
        /// the assertion, for "one dispatch is one batch".
        pub fn only_request(&self) -> Value {
            let seen = self.requests();
            assert_eq!(seen.len(), 1, "expected exactly one batch, got {}", seen.len());
            seen.into_iter().next().unwrap()
        }
    }

    impl ScriptBatch for FakeHost {
        fn run(&self, request: &Value) -> Result<ScriptResponse, HostError> {
            self.seen.lock().unwrap_or_else(|e| e.into_inner()).push(request.clone());
            (self.answer)(request)
        }
    }

    /// A `guest` failure — the condition's own fault, so ZIPP has answered.
    pub fn guest(message: &str) -> JobError {
        JobError { kind: "guest".into(), message: message.into() }
    }

    /// A per-job failure that is the ENGINE's, not the condition's.
    pub fn operational(kind: &str, message: &str) -> JobError {
        JobError { kind: kind.into(), message: message.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use crate::bridge::script_host::ScriptResponse;

    fn job(id: &str, source: &str) -> ConditionJob {
        ConditionJob {
            id: id.to_string(),
            source: source.to_string(),
            globals: json!({ "event": { "data": { "n": 1 } } }),
        }
    }

    // -- the batch shape ----------------------------------------------------

    #[test]
    fn one_dispatch_is_one_batch_carrying_every_condition() {
        // The host serialises one batch at a time process-wide, so a batch per
        // condition would queue an event's own conditions behind each other.
        let host = FakeHost::always(json!(true));
        let jobs = vec![job("c0", "a"), job("c1", "b"), job("c2", "c")];
        let out = decide(&host, &jobs);

        assert_eq!(out.len(), 3);
        let request = host.only_request();
        assert_eq!(request["v"], 1);
        let sent = request["jobs"].as_array().expect("jobs");
        assert_eq!(sent.len(), 3, "every condition of the dispatch travels together");
        assert_eq!(
            sent.iter().map(|j| j["source"].as_str().unwrap()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn a_job_is_a_program_over_json_globals_with_a_small_budget() {
        let request = build_batch(&[job("c0", "event.data.n === 1")]);
        assert_eq!(
            request["jobs"][0],
            json!({
                "id": "c0",
                "mode": "program",
                "source": "event.data.n === 1",
                "globals": { "event": { "data": { "n": 1 } } },
                "budgetMs": 250,
            }),
            "the job shape is the contract PR6c and the CLI both read"
        );
        assert_eq!(CONDITION_BUDGET_MS, 250, "a condition that needs longer is not a condition");
    }

    #[test]
    fn no_conditions_means_no_batch_at_all() {
        // Load-bearing: the first conditioned event after boot spawns a child.
        // An event whose bindings carry no conditions must never pay for one.
        let host = FakeHost::always(json!(true));
        assert!(decide(&host, &[]).is_empty());
        assert_eq!(host.calls(), 0, "an empty job list must not reach the host");
    }

    // -- verdicts -----------------------------------------------------------

    #[test]
    fn javascript_truthiness_decides_and_an_empty_object_is_true() {
        for falsy in [json!(false), json!(0), json!(-0.0), json!(""), json!(null)] {
            assert!(!truthy(&falsy), "{falsy} must be falsy");
        }
        for t in [json!(true), json!(1), json!(-1), json!("0"), json!([]), json!({})] {
            assert!(truthy(&t), "{t} must be truthy — an empty array or object is TRUE in JS");
        }
    }

    #[test]
    fn a_value_decides_and_a_guest_error_is_unknown() {
        let host = FakeHost::by_source(|src| match src {
            "yes" => Ok(json!(1)),
            "no" => Ok(json!("")),
            _ => Err(guest("ReferenceError: nope is not defined")),
        });
        let out = decide(&host, &[job("c0", "yes"), job("c1", "no"), job("c2", "boom")]);
        assert_eq!(out[0].verdict, Verdict::True);
        assert_eq!(out[1].verdict, Verdict::False);
        assert!(matches!(out[2].verdict, Verdict::Unknown(_)));
        assert!(out.iter().all(|v| v.answered), "all three ran: ZIPP answered each one");
    }

    #[test]
    fn an_outage_is_unknown_and_answers_nothing() {
        // The whole point: a host that cannot serve must not read as agreement,
        // and must not read as a disagreement either.
        let host = FakeHost::down("no engine");
        let out = decide(&host, &[job("c0", "a"), job("c1", "b")]);
        assert_eq!(out.len(), 2);
        for v in &out {
            assert!(matches!(v.verdict, Verdict::Unknown(_)));
            assert!(!v.answered, "an outage is not an answer");
        }
    }

    #[test]
    fn a_job_the_host_did_not_answer_is_unknown_and_unanswered() {
        // A protocol hole, not a disagreement.
        let host = FakeHost::raw(|_| {
            Ok(ScriptResponse {
                engine: json!({}),
                results: vec![JobResult { id: "c0".into(), outcome: Ok(json!(true)) }],
            })
        });
        let out = decide(&host, &[job("c0", "a"), job("c1", "b")]);
        assert_eq!(out[0].verdict, Verdict::True);
        assert!(matches!(out[1].verdict, Verdict::Unknown(_)));
        assert!(!out[1].answered);
    }

    #[test]
    fn only_the_conditions_own_failure_counts_as_an_answer() {
        // `guest` is the expression's fault, so the Rust grammar may be compared
        // against it. A timeout or a resource cap is the engine's, and filing
        // that as a semantic disagreement would block PR9 on an outage.
        let answered = verdict_from(&JobResult {
            id: "c0".into(),
            outcome: Err(guest("SyntaxError")),
        });
        assert!(answered.answered);
        for kind in ["timeout", "resource", "host", "source", "unsupported", "prepare"] {
            let v = verdict_from(&JobResult {
                id: "c0".into(),
                outcome: Err(operational(kind, "…")),
            });
            assert!(matches!(v.verdict, Verdict::Unknown(_)), "{kind}");
            assert!(!v.answered, "{kind} is the engine, not the condition");
        }
    }

    // -- the shadow log -----------------------------------------------------

    fn ctx() -> ShadowContext<'static> {
        ShadowContext {
            lane: "triggers",
            binding: "b1",
            event: "aokie.call.ended",
            source: "aokie",
            idempotency_key: "evt-1",
            expr: "event.data.n > 1",
        }
    }

    #[test]
    fn agreement_is_silent() {
        assert_eq!(shadow_line(&ctx(), &Verdict::True, &Verdict::True), None);
        assert_eq!(shadow_line(&ctx(), &Verdict::False, &Verdict::False), None);
        // Two refusals decide the same thing — skip — and their prose is written
        // by two different grammars. Comparing the text would log every
        // unparseable condition forever and block PR9 on noise.
        assert_eq!(
            shadow_line(
                &ctx(),
                &Verdict::Unknown("guest: SyntaxError".into()),
                &Verdict::Unknown("parentheses are not supported".into())
            ),
            None
        );
    }

    #[test]
    fn a_rust_refusal_zipp_evaluates_names_the_refusing_side() {
        // Classification (i)/(ii): parentheses and coercions the old grammar
        // refused. The line has to say which side refused and why, or the
        // reviewer has to re-run it to find out.
        let line = shadow_line(
            &ctx(),
            &Verdict::True,
            &Verdict::Unknown("parentheses are not supported by the restricted evaluator".into()),
        )
        .expect("a disagreement");
        assert_eq!(
            line,
            r#"condition-shadow: lane=triggers binding="b1" event="aokie.call.ended" source="aokie" key="evt-1" zipp=true rust=unknown refused=rust expr="event.data.n > 1" zipp_why=- rust_why="parentheses are not supported by the restricted evaluator""#
        );
    }

    #[test]
    fn a_real_semantic_change_carries_both_verdicts_and_the_expression() {
        // Classification (iii). Everything needed to judge it is on the line.
        let line = shadow_line(&ctx(), &Verdict::False, &Verdict::True).expect("a disagreement");
        assert!(line.contains("zipp=false rust=true refused=none"), "{line}");
        assert!(line.contains(r#"expr="event.data.n > 1""#), "{line}");
        assert!(line.contains(r#"key="evt-1""#), "the occurrence must be findable: {line}");
    }

    #[test]
    fn free_text_is_quoted_so_one_disagreement_is_one_line() {
        let mut c = ctx();
        let expr = "event.data.msg === 'a\nb'";
        c.expr = expr;
        let line = shadow_line(&c, &Verdict::True, &Verdict::False).expect("a disagreement");
        assert!(!line.contains('\n'), "a newline in a condition must not split the line: {line}");
        assert!(line.contains(r#"expr="event.data.msg === 'a\nb'""#), "{line}");
    }

    // -- save time ----------------------------------------------------------

    #[test]
    fn the_save_time_check_is_a_parse_job_that_never_runs_the_source() {
        let request = check_request("event.data.n > 1");
        assert_eq!(request["jobs"][0]["mode"], "parse");
        assert_eq!(request["jobs"][0]["source"], "event.data.n > 1");
        assert!(request["jobs"][0].get("globals").is_none(), "a parse needs no data");
    }

    #[test]
    fn a_syntax_error_is_rejected_and_an_outage_is_not() {
        let good = FakeHost::always(json!(null));
        assert_eq!(check(&good, "1 === 1"), CheckOutcome::Parses);

        let bad = FakeHost::by_source(|_| Err(guest("expected one expression")));
        assert_eq!(check(&bad, "1) ; ("), CheckOutcome::Rejected("expected one expression".into()));

        let down = FakeHost::down("no engine");
        assert!(matches!(check(&down, "1 === 1"), CheckOutcome::Unchecked(_)));

        // The CLI refusing the whole request is deterministic and ours to fix,
        // so it is a 400 rather than a saved binding nobody checked.
        let refused = FakeHost::raw(|_| Err(HostError::Refused { message: "bad job".into() }));
        assert_eq!(check(&refused, "x"), CheckOutcome::Rejected("bad job".into()));

        // An engine-side failure on the check is an outage, not the author's.
        let slow = FakeHost::by_source(|_| Err(operational("timeout", "…")));
        assert!(matches!(check(&slow, "x"), CheckOutcome::Unchecked(_)));
    }
}


