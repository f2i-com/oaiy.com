//! Executing the linked account's queued flow runs on this machine.
//!
//! The other half of `flows.rs`. That module RESERVES a run when something
//! happens here — a call arrives, a plugin fires an event — and leaves it
//! queued for whatever runtime the provider has. This module is that runtime:
//! it takes a queued run, fetches the graph, and hands it to the bundled OAIY
//! CLI, which is the same engine the browser uses.
//!
//! ```text
//!   GET {queuedPath} ──► POST {claimPath}  (409 = somebody else got it)
//!                          └─► GET {graphPath} ──► graph.json
//!                                                  inputs.json
//!                                                  connector.json ──► oaiy run … --connector
//!                                                                       └─► PATCH {completePath}
//! ```
//!
//! # Three rules everything here follows
//!
//! **Claim before working.** The claim is the exactly-once gate: two desktops
//! under one account both see the same queued run, and only one wins the POST.
//! Doing the work first and claiming afterwards would run the flow twice, which
//! for a flow that sends the email or charges the card is not recoverable.
//!
//! **Every exit path reports.** A run claimed and never completed is worse than
//! one left queued: the provider sees `running`, no other runtime will touch it,
//! and it sits there until somebody's reaper decides it is stale. So once the
//! claim succeeds, every branch below ends in a completion — including the ones
//! where this desktop is what failed.
//!
//! **The provider is data.** Paths come from [`FlowsSpec`], and the node types
//! a graph is written in come from the connector's `nodes` block and are copied
//! into the config file the CLI reads. Nothing here knows a provider's name or
//! any of its node-type names, and nothing here should ever learn one.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::descriptor::{self, FlowsSpec};
use super::result_actions;
use super::{LinkHandle, LinkedAccount};
use crate::bridge::worker::{run_flow_cli, CliOutcome, CliRequest};

/// How long to wait after finding the queue empty.
///
/// The queue is a plain GET, not a long poll, so the interval IS the latency a
/// user waiting on their own trigger sees. Short enough to feel immediate,
/// long enough that an idle desktop is not a meaningful load on the provider.
const IDLE_POLL: Duration = Duration::from_secs(3);

/// Pause after a failed poll, so a provider that is down — or a key that was
/// revoked — does not become a hot loop against it.
const ERROR_BACKOFF: Duration = Duration::from_secs(30);

/// Wall-clock budget for one run.
///
/// The provider's queue row carries no timeout, and an unbounded run would hold
/// this desktop's single execution slot forever. The CLI gets the same number
/// and times out first, so the error names the node that stuck.
const RUN_TIMEOUT: Duration = Duration::from_secs(300);

/// How many times a completion is attempted before giving up on it.
///
/// The one leg worth retrying: dropping it strands the run at `running`, where
/// nothing else will pick it up.
const COMPLETE_ATTEMPTS: usize = 3;

/// Cap on the result handed back to the provider.
///
/// Not decoration — a provider rejects an oversized completion, and a rejected
/// completion is a stranded run. Better to report a truncated result than to
/// lose the fact that the flow succeeded at all.
const MAX_RESULT_BYTES: usize = 192 * 1024;

/// Providers cap the message on a failure; a longer one is refused, and a
/// refused completion strands the run.
///
/// BYTES, not characters. The cap on the other side is a byte length, and these
/// messages are full of multi-byte punctuation (they quote this crate's own
/// prose and a CLI stderr tail) — a character budget would let a 900-character
/// message weigh well over a thousand bytes, be refused as too long, and strand
/// exactly the run whose failure it was trying to report. Sized under the
/// smallest cap seen so a header or two of slack costs nothing.
const MAX_MESSAGE_BYTES: usize = 900;

/// A run the provider has queued and nobody has taken.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QueuedRun {
    /// `runId` on the wire, not `id`.
    #[serde(rename = "runId")]
    id: String,
    /// The flow's SLUG. Present on every row, but it is not unique on its own —
    /// a workspace flow and an app flow may share one — so it is the fallback.
    #[serde(default)]
    flow: Option<String>,
    /// The flow's stable id, which IS unique. Preferred for exactly that reason.
    #[serde(default)]
    flow_definition_id: Option<String>,
    /// Which scope the flow lives in; `None` is the account's own workspace.
    #[serde(default)]
    app_id: Option<String>,
    /// Which BINDING queued this run, when one did.
    ///
    /// The binding is where the post-run actions live — the row itself carries
    /// a same-named field meaning something else (the actions a completion
    /// RECORDED), and it is empty while a run is queued. Absent for a run
    /// started directly rather than by a trigger, which simply has no actions.
    #[serde(default)]
    binding_id: Option<String>,
    /// The provider's own key for this run, derived from the binding and the
    /// triggering event. Reused to key the actions' side effects, so a
    /// redelivered event cannot send a second SMS.
    #[serde(default)]
    idempotency_key: Option<String>,
    /// The inputs the reserving side captured, which become the run's entry
    /// node. Absent is an empty object, not a failure: a flow triggered with
    /// nothing to say still runs.
    #[serde(default)]
    input_snapshot: Value,
}

#[derive(Debug, Deserialize)]
struct QueuedReply {
    #[serde(default)]
    runs: Vec<QueuedRun>,
}

/// One of the account's flows, as the provider publishes it.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FlowDoc {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    slug: Option<String>,
    #[serde(default)]
    app_id: Option<String>,
    /// The graph itself.
    #[serde(default)]
    flow_json: Value,
}

#[derive(Debug, Deserialize)]
struct FlowsReply {
    #[serde(default)]
    flows: Vec<FlowDoc>,
}

/// Why a run did not succeed, in the closed taxonomy a run ledger accepts.
///
/// Closed on purpose and small. An unknown code is refused by the provider,
/// which turns a reported failure into a STRANDED run — the worst outcome this
/// module has — so the set is fixed at compile time and every failure below
/// maps onto one of these rather than inventing a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum FailureCode {
    /// The flow ran and something in it failed.
    NodeFailed,
    /// It exceeded its budget.
    Timeout,
    /// Stopped rather than failed.
    Cancelled,
    /// The graph could not be read or was not a graph.
    InvalidFlow,
    /// This desktop could not run it at all — no CLI, or it would not start.
    RunnerUnavailable,
}

impl FailureCode {
    /// The terminal status that goes with this code.
    ///
    /// A timeout and a cancellation are their own states rather than flavours of
    /// error: a run history that calls them all "error" cannot tell an operator
    /// whether to raise the budget or fix the flow.
    fn status(self) -> &'static str {
        match self {
            FailureCode::Timeout => "timeout",
            FailureCode::Cancelled => "cancelled",
            _ => "error",
        }
    }
}

#[derive(Debug, Clone)]
struct Failure {
    code: FailureCode,
    message: String,
}

impl Failure {
    fn new(code: FailureCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// The four paths that together make a claim lane, resolved once.
///
/// Held as a type so the worker cannot get half way in and discover the
/// completion path is missing — which would be a claimed run it has no way to
/// report on.
struct Lane<'a> {
    queued: &'a str,
    claim: &'a str,
    complete: &'a str,
    graph: &'a str,
}

impl<'a> Lane<'a> {
    fn of(spec: &'a FlowsSpec) -> Option<Self> {
        Some(Lane {
            queued: spec.queued_path.as_deref()?,
            claim: spec.claim_path.as_deref()?,
            complete: spec.complete_path.as_deref()?,
            graph: spec.graph_path.as_deref()?,
        })
    }
}

/// Poll, claim, run, report — forever, while a link with a claimable flow lane
/// exists.
/// How a binding's connector actions reach this desktop's plugins.
///
/// One per process, because there is one plugin host per process. Set by
/// [`spawn`]; a runtime started without one (the headless server) leaves it
/// empty and a connector action then fails with a reason rather than being
/// silently dropped.
static CONNECTOR: std::sync::OnceLock<super::relay::Dispatcher> = std::sync::OnceLock::new();

/// The binding list, cached for the same reason the event lane caches it: the
/// actions for every run of one account come from the same short list.
fn bindings_cache() -> &'static super::flows::FlowBindings {
    static BINDINGS: std::sync::OnceLock<super::flows::FlowBindings> = std::sync::OnceLock::new();
    BINDINGS.get_or_init(super::flows::FlowBindings::new)
}

/// Run the account's queued flows, performing each binding's post-run actions
/// with `connector` when one is given.
pub fn spawn(
    store: LinkHandle,
    node: Option<crate::services::node_runtime::NodeHandle>,
    connector: Option<super::relay::Dispatcher>,
) {
    if let Some(dispatcher) = connector {
        let _ = CONNECTOR.set(dispatcher);
    }
    spawn_inner(store, node)
}

fn spawn_inner(store: LinkHandle, node: Option<crate::services::node_runtime::NodeHandle>) {
    std::thread::spawn(move || loop {
        let Some(account) = store.account() else {
            // Not linked. Sleep rather than spin; a link is a human action and
            // will not appear in the next millisecond.
            std::thread::sleep(Duration::from_secs(5));
            continue;
        };
        let Some(spec) =
            descriptor::find(store.data_dir(), &account.connector_id).and_then(|d| d.flows)
        else {
            std::thread::sleep(Duration::from_secs(30));
            continue;
        };
        // A provider that only wants runs QUEUED names no claim lane, and this
        // desktop then does nothing here — which is correct, not an error.
        let Some(lane) = Lane::of(&spec) else {
            std::thread::sleep(Duration::from_secs(30));
            continue;
        };
        let instance = store.instance_id();

        match poll_once(&account, &spec, &lane, &instance, node.as_ref()) {
            Ok((handled, trouble)) => {
                let stumbled = trouble.is_some();
                store.note_flow_run(trouble);
                if stumbled {
                    // The run is still at the head of the queue, so going
                    // straight back would retry the same failing claim as fast
                    // as the provider can refuse it.
                    std::thread::sleep(ERROR_BACKOFF);
                } else if handled == 0 {
                    // Nothing to do. A queue that HAD work probably has more, so
                    // that case goes straight back round.
                    std::thread::sleep(IDLE_POLL);
                }
            }
            Err(e) => {
                store.note_flow_run(Some(e));
                std::thread::sleep(ERROR_BACKOFF);
            }
        }
    });
}

fn client(timeout: Duration) -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| format!("could not build the flow runner client: {e}"))
}

/// One look at the queue, and at most one run.
///
/// One at a time deliberately: a flow run is a whole process with a five minute
/// budget, so taking a batch would mean claiming runs this desktop cannot start
/// for minutes — and a claimed run is one no other runtime will touch.
///
/// Returns how many runs were taken, and the first thing that went wrong at the
/// LANE level (the claim, the fetch, the report) — never the flow failing,
/// which is an answer rather than a fault.
fn poll_once(
    account: &LinkedAccount,
    spec: &FlowsSpec,
    lane: &Lane,
    instance: &str,
    node: Option<&crate::services::node_runtime::NodeHandle>,
) -> Result<(usize, Option<String>), String> {
    let http = client(Duration::from_secs(30))?;
    let resp = http
        .get(super::oauth::join(&account.base_url, lane.queued))
        .bearer_auth(&account.credential)
        .send()
        .map_err(|e| format!("could not reach the flow queue: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let body: Value = resp.json().unwrap_or(Value::Null);
        let message = body
            .get("message")
            .or_else(|| body.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("the provider refused the queue");
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(format!(
                "the provider no longer accepts this desktop's key ({message}) — link again"
            ));
        }
        return Err(format!("HTTP {}: {message}", status.as_u16()));
    }

    let reply: QueuedReply = resp
        .json()
        .map_err(|e| format!("the flow queue returned an unreadable list: {e}"))?;
    let Some(run) = reply.runs.into_iter().next() else {
        return Ok((0, None));
    };

    match serve(account, spec, lane, instance, node, &run) {
        Ok(taken) => Ok((usize::from(taken), None)),
        Err(e) => {
            log::warn!("flow run {} could not be served: {e}", run.id);
            Ok((1, Some(e)))
        }
    }
}

/// Claim one run, execute it, and report what happened.
///
/// Returns whether this desktop actually took the run — `false` when the claim
/// was lost, which is the ordinary outcome with two desktops on one account.
fn serve(
    account: &LinkedAccount,
    spec: &FlowsSpec,
    lane: &Lane,
    instance: &str,
    node: Option<&crate::services::node_runtime::NodeHandle>,
    run: &QueuedRun,
) -> Result<bool, String> {
    let http = client(Duration::from_secs(30))?;

    // Claim FIRST — before fetching the graph, before writing a file, before
    // anything with a side effect. Another runtime under the same account may
    // already have taken this run, and work done before the claim is work done
    // twice.
    let claim_url = super::oauth::join(&account.base_url, &lane.claim.replace("{id}", &run.id));
    let claimed = match http
        .post(&claim_url)
        .bearer_auth(&account.credential)
        // `runtime` says which kind of runtime took it, `instanceId` says which
        // machine. The provider binds the completion to the same instance, so
        // the two legs must present the same id.
        .json(&json!({ "runtime": "desktop", "instanceId": instance }))
        .send()
    {
        Ok(r) => r,
        Err(e) => {
            // AMBIGUOUS, not failed. The claim may have been committed and only
            // its answer lost — a read timeout, a reset, a proxy hiccup — and
            // then the run is marked running with nobody coming back for it.
            // The queue only lists QUEUED runs, so no later poll would find it
            // either. See release_if_claimed for why saying so is safe.
            release_if_claimed(account, lane, instance, &run.id, &format!("{e}"));
            return Err(format!("could not claim the run: {e}"));
        }
    };
    if !claimed.status().is_success() {
        let code = claimed.status().as_u16();
        // 409 is "somebody else got there first", which is the exactly-once
        // gate working, not a fault worth showing anyone.
        if code == 409 {
            return Ok(false);
        }
        // A 5xx is the same ambiguity as a lost connection: the provider may
        // have transitioned the row and then failed on its way back. A 4xx did
        // not — the request was rejected before anything moved.
        if code >= 500 {
            release_if_claimed(account, lane, instance, &run.id, &format!("HTTP {code}"));
        }
        return Err(format!("claim refused: HTTP {code}"));
    }

    // From here the run is OURS and the provider has it marked running. Every
    // path below ends at report().
    let outcome = execute(account, spec, lane, node, run);
    if let Err(f) = &outcome {
        log::info!("flow run {} failed ({:?}): {}", run.id, f.code, f.message);
    }
    // What the BINDING asked for with the answer, before the run is reported.
    //
    // Inside the claimed window on purpose. Reporting first and acting after
    // would leave a crash in between with the run already terminal, so nothing
    // would ever retry it and the writes would be lost for good; acting first
    // leaves the run claimed, which the provider's own reaper returns to the
    // queue. The provider's exactly-once gates are what make the retry safe —
    // a run is reserved under a unique key and claimed atomically — and each
    // side effect additionally carries a key of its own.
    let outcome = match outcome {
        Ok(result) => Ok(apply_result_actions(account, spec, run, result)),
        other => other,
    };
    report(account, lane, instance, &run.id, outcome)?;
    Ok(true)
}

/// Perform the binding's post-run actions and fold their failures into the
/// result the provider is about to be told.
///
/// A run with no binding, a provider that declares no such actions, or a
/// binding carrying none: nothing happens, which is the common case and not a
/// fault. Failures never turn a successful run into a failed one — the flow DID
/// run — they ride along in the result so the provider's console can show a run
/// that succeeded while its side effects did not.
fn apply_result_actions(
    account: &LinkedAccount,
    spec: &FlowsSpec,
    run: &QueuedRun,
    result: Value,
) -> Value {
    let bindings = bindings_cache();
    let connector = CONNECTOR.get();
    let Some(actions_spec) = spec.result_actions.as_ref() else {
        return result;
    };
    let Some(binding_id) = run.binding_id.as_deref().filter(|s| !s.is_empty()) else {
        return result;
    };
    let list = match bindings.load(account, spec) {
        Ok(list) => list,
        Err(e) => {
            // The actions exist and could not be read. Said in the result
            // rather than dropped, because silence here is indistinguishable
            // from a binding that asked for nothing.
            return result_actions::attach_errors(
                actions_spec,
                result,
                &[format!("could not read this run's binding: {e}")],
            );
        }
    };
    let Some(binding) = list.iter().find(|b| b.id == binding_id) else {
        return result;
    };
    let actions: Vec<Value> = match binding.field(&actions_spec.actions_field) {
        Some(Value::Array(items)) => items.clone(),
        _ => return result,
    };
    if actions.is_empty() {
        return result;
    }

    // The roots an action can address. `inputs` is what the run was started
    // with; the triggering event rides inside it under its own name, which is
    // how the provider's own runtime sees it too.
    let inputs = &run.input_snapshot;
    let event = inputs.get("event");
    let scope = result_actions::Scope {
        result: Some(&result),
        event,
        inputs: Some(inputs),
        app: None,
    };
    // The provider's key for this run already identifies the binding and the
    // triggering event, so a redelivery reproduces it exactly. Falling back to
    // the run id keeps actions keyed at all for a provider that sends none —
    // distinct per run, which is weaker but never collides.
    let run_key = run
        .idempotency_key
        .as_deref()
        .filter(|k| !k.is_empty())
        .unwrap_or(&run.id);
    let doer = result_actions::Live {
        account,
        spec: actions_spec,
        connector,
    };
    let errors = result_actions::perform_all(actions_spec, &actions, &scope, run_key, &doer);
    for e in &errors {
        log::info!("flow run {} binding {binding_id}: {e}", run.id);
    }
    result_actions::attach_errors(actions_spec, result, &errors)
}

/// Hand a possibly-claimed run back, after a claim whose answer never arrived.
///
/// Safe to call when the claim did NOT land, which is the point: a completion
/// only transitions a run that is actually `running` and bound to this instance,
/// so a run still sitting in the queue refuses the report and stays queued for
/// whoever takes it next — that refusal is a 409, which [`report`] already
/// treats as "nothing left to say". If the claim DID land, this is the only
/// thing standing between the run and a permanent `running`.
///
/// Best effort by construction: its own failure is logged, never propagated,
/// because the caller is already returning the real error.
fn release_if_claimed(
    account: &LinkedAccount,
    lane: &Lane,
    instance: &str,
    run_id: &str,
    why: &str,
) {
    let outcome = Err(Failure::new(
        FailureCode::RunnerUnavailable,
        format!(
            "this desktop's claim was sent but its answer was lost ({why}), so the run was not \
             started here; it is being released rather than left marked running"
        ),
    ));
    if let Err(e) = report(account, lane, instance, run_id, outcome) {
        log::warn!("flow run {run_id}: could not release an ambiguous claim: {e}");
    }
}

/// Fetch the graph, hand it to the CLI, and turn the result into an outcome.
fn execute(
    account: &LinkedAccount,
    spec: &FlowsSpec,
    lane: &Lane,
    node: Option<&crate::services::node_runtime::NodeHandle>,
    run: &QueuedRun,
) -> Result<Value, Failure> {
    let graph = fetch_graph(account, lane, run)?;

    // Everything travels by file. Values can be large and the connector config
    // carries this desktop's credential — argv is visible to every process
    // lister on the machine.
    let scratch = std::env::temp_dir().join(format!("oaiy-flow-run-{}", sanitize(&run.id)));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).map_err(|e| {
        Failure::new(
            FailureCode::RunnerUnavailable,
            format!("could not create a working directory for the run: {e}"),
        )
    })?;

    let result = run_in(&scratch, account, spec, node, run, &graph);
    // Removed on every path: the connector config in here holds the credential.
    let _ = std::fs::remove_dir_all(&scratch);
    result
}

fn run_in(
    scratch: &Path,
    account: &LinkedAccount,
    spec: &FlowsSpec,
    node: Option<&crate::services::node_runtime::NodeHandle>,
    run: &QueuedRun,
    graph: &Value,
) -> Result<Value, Failure> {
    let graph_path = scratch.join("graph.json");
    let inputs_path = scratch.join("inputs.json");
    let connector_path = scratch.join("connector.json");
    let out_path = scratch.join("result.json");

    let inputs = match &run.input_snapshot {
        Value::Null => json!({}),
        other => other.clone(),
    };
    write_file(&graph_path, &graph.to_string())?;
    write_file(&inputs_path, &inputs.to_string())?;
    write_file(&connector_path, &connector_config(account, spec).to_string())?;
    // The credential is in that file. Same treatment as the stored link.
    super::restrict_to_owner(&connector_path);

    let outcome = run_flow_cli(
        CliRequest {
            flow_path: &graph_path,
            inputs_path: &inputs_path,
            out_path: &out_path,
            connector_path: Some(&connector_path),
            timeout: RUN_TIMEOUT,
            node,
        },
        // Nothing cancels a provider's run from this side; the budget does.
        &|| false,
    );

    match outcome {
        CliOutcome::Succeeded(v) => flow_answer(v),
        CliOutcome::Unreadable(why) => Err(Failure::new(FailureCode::NodeFailed, why)),
        CliOutcome::Failed { exit_code, detail } => Err(Failure::new(
            FailureCode::NodeFailed,
            if detail.is_empty() {
                format!("the flow failed (CLI exit {exit_code})")
            } else {
                format!("the flow failed (CLI exit {exit_code}): {detail}")
            },
        )),
        CliOutcome::TimedOut => Err(Failure::new(
            FailureCode::Timeout,
            format!(
                "the flow exceeded its {}s budget on this desktop and was killed",
                RUN_TIMEOUT.as_secs()
            ),
        )),
        CliOutcome::Cancelled => Err(Failure::new(
            FailureCode::Cancelled,
            "cancelled while running; side effects already performed stay performed",
        )),
        // Reported, not swallowed: an unanswered run looks to the provider like
        // no runtime exists, which sends the user hunting for a link problem
        // instead of reading "install the CLI".
        CliOutcome::Unavailable {
            message, detail, ..
        } => Err(Failure::new(
            FailureCode::RunnerUnavailable,
            match detail {
                Some(d) => format!("{message}. {d}"),
                None => message,
            },
        )),
    }
}

/// What the CLI wrote, read as the flow's own report of itself.
///
/// The CLI writing a result file means the CLI ran, NOT that the flow worked.
/// Its report carries the flow's own verdict, and a run that reports `success:
/// false` with an error is a failed run — reporting it as done tells the person
/// who queued it that their automation ran, when it did not, and leaves the
/// error sitting in a payload nobody reads. Sixteen of the last sixty runs on
/// this machine said `success: false` and every one was reported done.
///
/// On success the answer is the flow's OUTPUT, not the CLI's envelope. The
/// envelope is this desktop's business — a job id and a map of every node's
/// working — and handing it to the provider both leaks a private shape into
/// their data model and puts the wrong object where their automations look for
/// the flow's result. An older CLI that reports no `output` at all still has
/// its envelope passed through, so a stale binary keeps working.
fn flow_answer(v: Value) -> Result<Value, Failure> {
    // Only an EXPLICIT denial counts. A report that simply does not carry the
    // field says nothing about the flow, and treating silence as failure would
    // fail every run made by a CLI older than this convention.
    if v.get("success") == Some(&Value::Bool(false)) {
        let message = v
            .get("error")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| match v.get("status").and_then(Value::as_str) {
                Some(s) => format!("the flow reported {s:?} without saying why"),
                None => "the flow reported failure without saying why".to_string(),
            });
        return Err(Failure::new(FailureCode::NodeFailed, message));
    }
    match v.get("output") {
        // `null` IS an answer — a flow may legitimately answer with nothing —
        // so presence of the key decides, not truthiness.
        Some(output) => Ok(output.clone()),
        None => Ok(v),
    }
}

/// The config the CLI reads with `--connector`.
///
/// This is the entire bridge between the provider's vocabulary and what this
/// desktop can do. The node TYPE names are the provider's and come straight out
/// of its descriptor; the OPERATIONS are the closed set the CLI implements. No
/// name of any provider's node type is written here, and adding a provider whose
/// graphs call the same idea something else is a line in a JSON file.
fn connector_config(account: &LinkedAccount, spec: &FlowsSpec) -> Value {
    json!({
        "baseUrl": account.base_url,
        "credential": account.credential,
        "nodes": spec.nodes,
    })
}

fn write_file(path: &Path, body: &str) -> Result<(), Failure> {
    std::fs::write(path, body).map_err(|e| {
        Failure::new(
            FailureCode::RunnerUnavailable,
            format!("could not write {}: {e}", path.display()),
        )
    })
}

/// The graph for this run.
///
/// A claimed run names its flow but does not carry it, so the graphs are read
/// from the account and the right one picked out. By stable id first: a slug is
/// unique only within its scope, and running the wrong flow because two of them
/// share a name would be silent and wrong rather than loud and wrong.
fn fetch_graph(
    account: &LinkedAccount,
    lane: &Lane,
    run: &QueuedRun,
) -> Result<Value, Failure> {
    let http = client(Duration::from_secs(30)).map_err(|e| {
        Failure::new(FailureCode::RunnerUnavailable, e)
    })?;
    let resp = http
        .get(super::oauth::join(&account.base_url, lane.graph))
        .bearer_auth(&account.credential)
        .send()
        .map_err(|e| {
            Failure::new(
                FailureCode::RunnerUnavailable,
                format!("could not fetch the flow graph: {e}"),
            )
        })?;
    if !resp.status().is_success() {
        return Err(Failure::new(
            FailureCode::RunnerUnavailable,
            format!(
                "the provider refused the flow graph: HTTP {}",
                resp.status().as_u16()
            ),
        ));
    }
    let reply: FlowsReply = resp.json().map_err(|e| {
        Failure::new(
            FailureCode::InvalidFlow,
            format!("the provider returned an unreadable flow list: {e}"),
        )
    })?;

    let found = select_flow(&reply.flows, run).ok_or_else(|| {
        Failure::new(
            FailureCode::InvalidFlow,
            format!(
                "the account has no flow matching this run ({})",
                run.flow_definition_id
                    .as_deref()
                    .or(run.flow.as_deref())
                    .unwrap_or("unnamed")
            ),
        )
    })?;

    // The CLI reads `{ nodes, edges }`. Anything else is a flow this desktop
    // cannot execute, and saying so beats handing the engine a shape it will
    // silently make nothing of.
    if !found.flow_json.get("nodes").is_some_and(Value::is_array) {
        return Err(Failure::new(
            FailureCode::InvalidFlow,
            "the flow carries no node list, so there is nothing to execute",
        ));
    }
    Ok(found.flow_json.clone())
}

/// The flow this run belongs to: by stable id, else by slug within the same
/// scope.
fn select_flow<'a>(flows: &'a [FlowDoc], run: &QueuedRun) -> Option<&'a FlowDoc> {
    if let Some(id) = run.flow_definition_id.as_deref().filter(|s| !s.is_empty()) {
        if let Some(f) = flows.iter().find(|f| f.id.as_deref() == Some(id)) {
            return Some(f);
        }
    }
    let slug = run.flow.as_deref().filter(|s| !s.is_empty())?;
    // The scope has to match too. A workspace flow and an app flow may share a
    // slug, and picking the wrong one runs the wrong automation.
    let scope = run.app_id.as_deref().filter(|s| !s.is_empty());
    flows
        .iter()
        .find(|f| f.slug.as_deref() == Some(slug) && f.app_id.as_deref().filter(|s| !s.is_empty()) == scope)
}

/// Tell the provider how the run ended.
///
/// Retried, because this is the one leg whose loss is unrecoverable from here:
/// the run stays `running`, no other runtime will claim it, and it waits for
/// somebody's reaper. A 409 is not retried — it means the run was already
/// finalised, so there is nothing left to say.
fn report(
    account: &LinkedAccount,
    lane: &Lane,
    instance: &str,
    run_id: &str,
    outcome: Result<Value, Failure>,
) -> Result<(), String> {
    let body = match outcome {
        Ok(result) => json!({
            "instanceId": instance,
            "status": "done",
            "result": cap_result(result),
        }),
        Err(f) => json!({
            "instanceId": instance,
            "status": f.code.status(),
            "error": { "code": f.code, "message": clip(&f.message, MAX_MESSAGE_BYTES) },
        }),
    };
    let url = super::oauth::join(&account.base_url, &lane.complete.replace("{id}", run_id));

    let mut last = String::new();
    for attempt in 0..COMPLETE_ATTEMPTS {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(2));
        }
        let http = match client(Duration::from_secs(30)) {
            Ok(c) => c,
            Err(e) => {
                last = e;
                continue;
            }
        };
        match http
            .patch(&url)
            .bearer_auth(&account.credential)
            .json(&body)
            .send()
        {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            // Already finalised, or claimed by someone else after all. Nothing
            // more to report, and retrying would only repeat the refusal.
            Ok(resp) if resp.status().as_u16() == 409 => {
                log::info!("flow run {run_id} was already finalised by another runtime");
                return Ok(());
            }
            Ok(resp) => {
                let code = resp.status().as_u16();
                let payload: Value = resp.json().unwrap_or(Value::Null);
                let message = payload
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("the provider refused the outcome");
                last = format!("could not report the outcome: HTTP {code}: {message}");
            }
            Err(e) => last = format!("could not report the outcome: {e}"),
        }
    }
    // Loud, because this is the state the whole module exists to avoid: a run
    // this desktop claimed and left looking as though it were still working.
    Err(format!(
        "{last} — run {run_id} was claimed by this desktop and may be stuck at running"
    ))
}

/// Keep a result under the size a completion can carry.
///
/// An oversized body is refused, and a refused completion is a stranded run —
/// so a truncated answer is strictly better than the truth nobody receives.
fn cap_result(result: Value) -> Value {
    let encoded = result.to_string();
    if encoded.len() <= MAX_RESULT_BYTES {
        return result;
    }
    json!({
        "truncated": true,
        "bytes": encoded.len(),
        "note": "the flow's result was too large to report and was dropped; \
                 the run itself succeeded",
    })
}

/// Trim to at most `max_bytes` BYTES, never splitting a character.
///
/// Byte-wise because that is how the far side measures it; char-boundary-safe
/// because slicing UTF-8 anywhere else panics, and a panic here would take the
/// whole runner down while holding a claimed run.
fn clip(s: &str, max_bytes: usize) -> String {
    let t = s.trim();
    if t.len() <= max_bytes {
        return t.to_string();
    }
    // The ellipsis is itself three bytes and has to fit inside the budget.
    let mut end = max_bytes.saturating_sub('…'.len_utf8()).min(t.len());
    while end > 0 && !t.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &t[..end])
}

/// A run id becomes a directory name. Strict charset rather than escaping: the
/// id is remote input, and `..` in a path component is not a theoretical worry.
fn sanitize(id: &str) -> String {
    let cleaned: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(64)
        .collect();
    if cleaned.is_empty() {
        "run".to_string()
    } else {
        cleaned
    }
}

/// Where the scratch directory for a run would go, for tests.
#[cfg(test)]
fn scratch_for(id: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("oaiy-flow-run-{}", sanitize(id)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- what the provider actually sends ---------------------------------

    #[test]
    fn a_queued_run_parses_from_the_providers_actual_wire_shape() {
        // Captured live. The id arrives as `runId`, the flow as its SLUG under
        // `flow`, and the graph is NOT included — which is the whole reason a
        // graph path exists. Reading `id` instead of `runId` would not fail
        // loudly: the batch simply would not deserialize and every queued run
        // would sit there while this lane reported a parse error.
        let raw = json!({
            "runs": [{
                "runId": "9f1c8a2e-2f5d-4a70-9a11-1f0d5c2b8e44",
                "appId": null,
                "formId": null,
                "responseId": null,
                "bindingId": "b1",
                "flowDefinitionId": "119854a8-df5d-41c9-b070-d56b07a92993",
                "flow": "call-summary-follow-up",
                "triggerEvent": "aokie.call.ended",
                "correlationId": "corr-1",
                "idempotencyKey": "flow:b1:evt-7",
                "status": "queued",
                "runtime": null,
                "claimedBy": null,
                "inputSnapshot": { "event": { "name": "aokie.call.ended" } },
                "result": null,
                "error": null,
                "startedAt": null,
                "finishedAt": null,
                "createdAt": "2026-08-01 09:00:00"
            }]
        });
        let reply: QueuedReply = serde_json::from_value(raw).expect("the real shape must parse");
        assert_eq!(reply.runs.len(), 1);
        let r = &reply.runs[0];
        assert_eq!(r.id, "9f1c8a2e-2f5d-4a70-9a11-1f0d5c2b8e44");
        assert_eq!(r.flow.as_deref(), Some("call-summary-follow-up"));
        assert_eq!(
            r.flow_definition_id.as_deref(),
            Some("119854a8-df5d-41c9-b070-d56b07a92993")
        );
        assert!(r.app_id.is_none());
        assert_eq!(r.input_snapshot["event"]["name"], "aokie.call.ended");

        // An empty queue is the ordinary case, not an error.
        let empty: QueuedReply = serde_json::from_value(json!({})).unwrap();
        assert!(empty.runs.is_empty());
    }

    #[test]
    fn the_right_flow_is_chosen_by_id_before_slug() {
        // Two flows can share a slug across scopes. Matching the slug alone
        // would run somebody's other automation, silently and successfully.
        let flows = vec![
            FlowDoc {
                id: Some("f-workspace".into()),
                slug: Some("summary".into()),
                app_id: None,
                flow_json: json!({ "nodes": [], "edges": [] }),
            },
            FlowDoc {
                id: Some("f-app".into()),
                slug: Some("summary".into()),
                app_id: Some("app-1".into()),
                flow_json: json!({ "nodes": [], "edges": [] }),
            },
        ];

        let by_id = QueuedRun {
            id: "r1".into(),
            flow: Some("summary".into()),
            flow_definition_id: Some("f-app".into()),
            app_id: None,
            binding_id: None,
            idempotency_key: None,
            input_snapshot: Value::Null,
        };
        assert_eq!(select_flow(&flows, &by_id).unwrap().id.as_deref(), Some("f-app"));

        // No id: the slug still has to match the SCOPE.
        let workspace = QueuedRun {
            id: "r2".into(),
            flow: Some("summary".into()),
            flow_definition_id: None,
            app_id: None,
            binding_id: None,
            idempotency_key: None,
            input_snapshot: Value::Null,
        };
        assert_eq!(
            select_flow(&flows, &workspace).unwrap().id.as_deref(),
            Some("f-workspace")
        );
        let scoped = QueuedRun {
            app_id: Some("app-1".into()),
            ..workspace.clone()
        };
        assert_eq!(select_flow(&flows, &scoped).unwrap().id.as_deref(), Some("f-app"));

        // A scope with no flow in it matches nothing rather than falling back.
        let elsewhere = QueuedRun {
            app_id: Some("app-2".into()),
            ..workspace.clone()
        };
        assert!(select_flow(&flows, &elsewhere).is_none());
    }

    // --- the connector config the CLI reads -------------------------------

    #[test]
    fn the_connector_config_carries_the_providers_vocabulary_from_the_descriptor() {
        // The point of the whole design. The node TYPE names come out of the
        // connector file; the OPERATIONS are the closed set. If this test ever
        // needs a node type spelled in this crate's source, the design failed.
        let d = descriptor::find(std::path::Path::new("/nonexistent"), "formlogic").unwrap();
        let spec = d.flows.unwrap();
        let account = account("http://provider.test".into());
        let config = connector_config(&account, &spec);

        assert_eq!(config["baseUrl"], "http://provider.test");
        assert_eq!(config["credential"], "flk_secret");
        // Exactly the three keys the CLI half is written against. An extra one
        // is not harmless: the two halves are built separately and only this
        // file joins them.
        let keys: Vec<&str> = config.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, ["baseUrl", "credential", "nodes"]);
        let nodes = config["nodes"].as_array().expect("nodes must be a list");
        assert_eq!(nodes.len(), spec.nodes.len());
        // camelCase on the wire, and a path only where the operation needs one.
        assert!(nodes.iter().all(|n| n["nodeType"].is_string() && n["operation"].is_string()));
        assert!(nodes
            .iter()
            .any(|n| n["operation"] == "listRecords" && n["path"].as_str().is_some_and(|p| p.starts_with('/'))));
        assert!(nodes.iter().any(|n| n["operation"] == "runInput" && n.get("path").is_none()));

        // Every declared operation is one the CLI was told about — the closed
        // set, serialized in the form the contract fixes.
        let ops: Vec<&str> = nodes.iter().filter_map(|n| n["operation"].as_str()).collect();
        for op in &ops {
            assert!(
                [
                    "runInput",
                    "chat",
                    "listRecords",
                    "createRecord",
                    "updateRecord",
                    "connectorRequest",
                    "serviceControl",
                ]
                .contains(op),
                "unknown operation {op}"
            );
        }
    }

    #[test]
    fn no_provider_name_or_node_type_is_written_into_this_module() {
        // The single most important constraint, asserted against the source
        // rather than trusted. Test fixtures are allowed a provider name; the
        // executing code is not.
        let source = include_str!("flow_runner.rs");
        let code_only: String = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .unwrap()
            .to_string();
        for forbidden in ["formlogic", "FormLogic", "formlogic_list_responses", "llm_chat"] {
            assert!(
                !code_only.contains(forbidden),
                "{forbidden:?} must not appear outside the tests"
            );
        }
    }

    // --- failure vocabulary ------------------------------------------------

    #[test]
    fn a_timeout_and_a_cancellation_are_not_flattened_into_error() {
        // A history that calls all three "error" cannot tell an operator
        // whether to raise the budget or fix the flow.
        assert_eq!(FailureCode::Timeout.status(), "timeout");
        assert_eq!(FailureCode::Cancelled.status(), "cancelled");
        assert_eq!(FailureCode::NodeFailed.status(), "error");
        assert_eq!(FailureCode::InvalidFlow.status(), "error");
        assert_eq!(FailureCode::RunnerUnavailable.status(), "error");
        // Serialized in the taxonomy a run ledger accepts. An unknown code is
        // refused, and a refused completion is a stranded run.
        assert_eq!(json!(FailureCode::NodeFailed), json!("node_failed"));
        assert_eq!(json!(FailureCode::RunnerUnavailable), json!("runner_unavailable"));
        assert_eq!(json!(FailureCode::InvalidFlow), json!("invalid_flow"));
    }

    #[test]
    fn a_flow_that_reports_its_own_failure_is_not_reported_as_done() {
        // The live shape, verbatim: the CLI ran fine and wrote a result; the
        // FLOW inside it did not. Reported done, this tells someone their
        // automation ran and buries the reason it did not.
        let failed = json!({
            "jobId": "e80cc135",
            "status": "failed",
            "success": false,
            "results": {},
            "error": "llm_chat node \"summary\": returned HTTP 403: origin not allowed",
        });
        let err = flow_answer(failed).expect_err("a failed flow must not report done");
        assert_eq!(err.code, FailureCode::NodeFailed);
        assert!(err.message.contains("403"), "the reason must survive: {}", err.message);

        // Failure with nothing said is still failure, and says so.
        let mute = json!({ "success": false, "status": "aborted" });
        let err = flow_answer(mute).expect_err("silence is not success");
        assert!(err.message.contains("aborted"), "{}", err.message);
    }

    #[test]
    fn a_successful_run_answers_with_the_flows_output_not_this_desktops_envelope() {
        // What the provider's automations read as the flow's result is the
        // OUTPUT node's declared value. Handing back the envelope instead puts
        // a job id and every node's working where they look for the answer.
        let ok = json!({
            "jobId": "bbbe630b",
            "status": "completed",
            "success": true,
            "results": { "in": { "event": {} }, "out": { "hasCall": true } },
            "output": { "hasCall": true, "responseId": "resp-1" },
        });
        assert_eq!(
            flow_answer(ok).expect("a successful flow reports its output"),
            json!({ "hasCall": true, "responseId": "resp-1" })
        );

        // `null` is an answer a flow may legitimately give; presence decides.
        let empty = json!({ "success": true, "output": null });
        assert_eq!(flow_answer(empty).expect("null is an answer"), Value::Null);

        // A CLI older than this convention carries no `output`. Its envelope is
        // passed through rather than turned into an empty result.
        let legacy = json!({ "success": true, "results": { "out": 1 } });
        assert_eq!(
            flow_answer(legacy.clone()).expect("a legacy report still succeeds"),
            legacy
        );
    }

    #[test]
    fn an_oversized_result_still_reports_success() {
        // A body the provider refuses is a completion that never lands, which
        // is the one failure this module cannot recover from.
        let small = json!({ "ok": true });
        assert_eq!(cap_result(small.clone()), small);

        let huge = json!({ "text": "x".repeat(MAX_RESULT_BYTES + 10) });
        let capped = cap_result(huge);
        assert_eq!(capped["truncated"], true);
        assert!(capped.to_string().len() < 1000);
    }

    #[test]
    fn a_long_failure_message_is_clipped_rather_than_refused() {
        let long = "stack trace ".repeat(500);
        let clipped = clip(&long, MAX_MESSAGE_BYTES);
        assert!(clipped.len() <= MAX_MESSAGE_BYTES);
        assert!(clipped.ends_with('…'));
        assert_eq!(clip("  short  ", 100), "short");
    }

    #[test]
    fn the_message_budget_is_bytes_because_the_provider_counts_bytes() {
        // The failure this guards: these messages quote this crate's own prose
        // (full of — and …) and a CLI stderr tail, so a CHARACTER budget of 900
        // can weigh 2700 bytes, be refused as too long, and strand the very run
        // whose failure it was reporting.
        let wide = "— ".repeat(2000);
        let clipped = clip(&wide, MAX_MESSAGE_BYTES);
        assert!(
            clipped.len() <= MAX_MESSAGE_BYTES,
            "{} bytes for a {}-byte budget",
            clipped.len(),
            MAX_MESSAGE_BYTES
        );
        // Still under the smallest cap the wire has been seen to impose.
        assert!(clipped.len() <= 1000);
        // And it is still valid UTF-8 text, i.e. no character was cut in half.
        assert!(clipped.ends_with('…'));
        assert!(std::str::from_utf8(clipped.as_bytes()).is_ok());

        // A budget too small for even the ellipsis must not panic or overflow.
        assert!(clip("——————", 2).len() <= 3);
    }

    #[test]
    fn a_run_id_cannot_escape_the_scratch_directory() {
        // The id is remote input and becomes a path component.
        let escaped = scratch_for("../../evil");
        assert!(escaped.to_string_lossy().contains("oaiy-flow-run-evil"));
        assert!(!escaped.to_string_lossy().contains(".."));
        assert!(scratch_for("").to_string_lossy().ends_with("oaiy-flow-run-run"));
    }

    // --- the lane must be complete before anything is claimed --------------

    #[test]
    fn a_provider_that_only_queues_runs_is_never_claimed_from() {
        // Reserving is the minimum a flow lane can be. A provider that names no
        // claim/complete/graph paths must leave this worker idle rather than
        // half-executing something it cannot report on.
        let mut spec = FlowsSpec {
            bindings_path: "/bindings".into(),
            reserve_path: "/runs".into(),
            queued_path: None,
            claim_path: None,
            complete_path: None,
            graph_path: None,
            nodes: Vec::new(),
            result_actions: None,
        };
        assert!(Lane::of(&spec).is_none());

        spec.queued_path = Some("/runs/queued".into());
        spec.claim_path = Some("/runs/{id}/claim".into());
        spec.complete_path = Some("/runs/{id}".into());
        // Still no graph path: a run could be claimed and never executed.
        assert!(Lane::of(&spec).is_none());

        spec.graph_path = Some("/flows".into());
        assert!(Lane::of(&spec).is_some());
    }

    #[test]
    fn the_builtin_connector_describes_the_claim_lane_it_needs() {
        let d = descriptor::find(std::path::Path::new("/nonexistent"), "formlogic").unwrap();
        let f = d.flows.expect("the connector must declare a flow lane");
        let lane = Lane::of(&f).expect("the shipped connector must be claimable");
        assert!(lane.claim.contains("{id}"), "{}", lane.claim);
        assert!(lane.complete.contains("{id}"), "{}", lane.complete);
        // The queue and the graph list are not per-run; a placeholder there
        // would be sent to the provider literally.
        assert!(!lane.queued.contains("{id}"));
        assert!(!lane.graph.contains("{id}"));
    }

    // --- the claim/execute/report sequence ---------------------------------

    /// A stub provider: one queued run, a flow list, then claim and complete.
    ///
    /// Hand-rolled HTTP like the relay's tests. The assertions are about the
    /// sequence and what it reports, not about a server.
    ///
    /// Returns the base URL and a channel of `(request line, whole request)` in
    /// arrival order, so a test can assert on what was NOT sent too.
    fn stub_provider(
        queue: &'static str,
        flows: &'static str,
        claim_status: &'static str,
        complete_status: &'static str,
    ) -> (String, std::sync::mpsc::Receiver<(String, String)>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let mut served_queue = false;
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut raw = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let Ok(n) = stream.read(&mut buf) else { return };
                    if n == 0 {
                        break;
                    }
                    raw.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&raw).to_string();
                    if let Some(head_end) = text.find("\r\n\r\n") {
                        let len: usize = text
                            .lines()
                            .find_map(|l| {
                                l.strip_prefix("content-length: ")
                                    .or_else(|| l.strip_prefix("Content-Length: "))
                            })
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                        if raw.len() >= head_end + 4 + len {
                            break;
                        }
                    }
                }
                let text = String::from_utf8_lossy(&raw).to_string();
                let line = text.lines().next().unwrap_or_default().to_string();
                let (status, reply) = if line.contains("/runs/queued") {
                    // Once. A second poll must not re-serve work already taken.
                    if std::mem::replace(&mut served_queue, true) {
                        ("200 OK", r#"{"runs":[]}"#.to_string())
                    } else {
                        ("200 OK", queue.to_string())
                    }
                } else if line.contains("/claim") {
                    (claim_status, "{}".to_string())
                } else if line.starts_with("GET /flows") {
                    ("200 OK", flows.to_string())
                } else if line.starts_with("PATCH") {
                    (complete_status, "{}".to_string())
                } else {
                    ("200 OK", "{}".to_string())
                };
                let _ = tx.send((line, text));
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = stream.flush();
            }
        });
        (format!("http://127.0.0.1:{port}"), rx)
    }

    fn spec() -> FlowsSpec {
        FlowsSpec {
            bindings_path: "/bindings".into(),
            reserve_path: "/runs".into(),
            queued_path: Some("/runs/queued".into()),
            claim_path: Some("/runs/{id}/claim".into()),
            complete_path: Some("/runs/{id}".into()),
            graph_path: Some("/flows".into()),
            nodes: Vec::new(),
            result_actions: None,
        }
    }

    fn account(base: String) -> LinkedAccount {
        LinkedAccount {
            connector_id: "formlogic".into(),
            base_url: base,
            credential: "flk_secret".into(),
            account_id: None,
            account_name: None,
            granted_scopes: None,
            linked_at: chrono::Utc::now(),
            instance_id: Some("oaiy-test".into()),
        }
    }

    const ONE_RUN: &str = r#"{"runs":[{"runId":"r1","flow":"summary","flowDefinitionId":"f1","inputSnapshot":{"event":{"name":"x"}}}]}"#;
    /// A flow list with no matching flow, so the run fails BEFORE the CLI is
    /// ever reached — which is what these sequence tests are about.
    const NO_MATCHING_FLOW: &str = r#"{"flows":[{"id":"other","slug":"other","flowJson":{"nodes":[],"edges":[]}}]}"#;

    #[test]
    fn a_run_is_claimed_before_any_work_and_is_always_reported() {
        // The two rules of the module in one sequence: the claim comes first,
        // and the run is answered even when this desktop is what failed —
        // silence would leave it at `running` for everyone else.
        let (base, rx) = stub_provider(ONE_RUN, NO_MATCHING_FLOW, "200 OK", "200 OK");
        let (handled, trouble) =
            poll_once(&account(base), &spec(), &Lane::of(&spec()).unwrap(), "oaiy-test", None)
                .unwrap();
        assert_eq!(handled, 1);
        assert_eq!(trouble, None, "a flow that cannot run is not a failing lane");

        let (queue, _) = rx.recv().unwrap();
        assert!(queue.starts_with("GET /runs/queued "), "{queue}");
        // The CLAIM, before the graph is even fetched.
        let (claim, claim_raw) = rx.recv().unwrap();
        assert!(claim.starts_with("POST /runs/r1/claim "), "{claim}");
        assert!(claim_raw.contains("\"runtime\":\"desktop\""), "{claim_raw}");
        assert!(claim_raw.contains("oaiy-test"), "the claim must name this machine");
        let (graph, _) = rx.recv().unwrap();
        assert!(graph.starts_with("GET /flows "), "{graph}");
        // …and a completion, carrying the real reason.
        let (complete, complete_raw) = rx.recv().unwrap();
        assert!(complete.starts_with("PATCH /runs/r1 "), "{complete}");
        assert!(complete_raw.contains("\"status\":\"error\""), "{complete_raw}");
        assert!(complete_raw.contains("invalid_flow"), "{complete_raw}");
        assert!(
            complete_raw.contains("oaiy-test"),
            "the completion must present the same instance as the claim: {complete_raw}"
        );
    }

    #[test]
    fn losing_the_claim_does_no_work_and_reports_nothing() {
        // 409 is the ordinary outcome when an account has two runtimes: the
        // other one got there first. Fetching the graph or — far worse —
        // running the flow anyway would execute it twice.
        let (base, rx) = stub_provider(ONE_RUN, NO_MATCHING_FLOW, "409 Conflict", "200 OK");
        let (handled, trouble) =
            poll_once(&account(base), &spec(), &Lane::of(&spec()).unwrap(), "oaiy-test", None)
                .unwrap();
        assert_eq!(handled, 0, "a run we did not take was not handled");
        assert_eq!(trouble, None, "losing a race is not a fault");

        let _ = rx.recv().unwrap(); // the queue
        let (claim, _) = rx.recv().unwrap();
        assert!(claim.contains("/claim"), "{claim}");
        assert!(
            rx.recv_timeout(Duration::from_millis(400)).is_err(),
            "nothing may follow a lost claim — no graph fetch, no run, no completion"
        );
    }

    #[test]
    fn a_refused_claim_becomes_something_the_panel_can_show() {
        // The poll succeeded, so nothing here looks wrong, while on the
        // provider's site the run sits queued forever.
        let (base, _rx) = stub_provider(ONE_RUN, NO_MATCHING_FLOW, "500 Internal Server Error", "200 OK");
        let (handled, trouble) =
            poll_once(&account(base), &spec(), &Lane::of(&spec()).unwrap(), "oaiy-test", None)
                .unwrap();
        assert_eq!(handled, 1);
        let reported = trouble.expect("a refused claim must reach the status");
        assert!(reported.contains("claim refused"), "{reported}");
        assert!(reported.contains("500"), "the code is the diagnosis: {reported}");
    }

    #[test]
    fn a_claim_whose_answer_is_lost_releases_the_run_instead_of_stranding_it() {
        // The nastiest state in the module. The provider transitions the run to
        // `running` and THEN the connection dies, so this desktop believes it
        // failed to claim while the provider believes the run is being executed.
        // The queue only lists QUEUED runs, so nothing here would ever see it
        // again and no other runtime would touch it.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let mut served_queue = false;
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let line = text.lines().next().unwrap_or_default().to_string();
                let _ = tx.send((line.clone(), text));
                if line.contains("/claim") {
                    // The claim landed; the ANSWER is what is lost.
                    drop(stream);
                    continue;
                }
                let reply = if line.contains("/runs/queued") {
                    if std::mem::replace(&mut served_queue, true) {
                        r#"{"runs":[]}"#.to_string()
                    } else {
                        ONE_RUN.to_string()
                    }
                } else {
                    "{}".to_string()
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = stream.flush();
            }
        });

        let (handled, trouble) = poll_once(
            &account(format!("http://127.0.0.1:{port}")),
            &spec(),
            &Lane::of(&spec()).unwrap(),
            "oaiy-test",
            None,
        )
        .unwrap();
        assert_eq!(handled, 1);
        assert!(
            trouble.is_some_and(|t| t.contains("could not claim")),
            "the lane still reports the real failure"
        );

        let (queue, _) = rx.recv().unwrap();
        assert!(queue.starts_with("GET /runs/queued "), "{queue}");
        let (claim, _) = rx.recv().unwrap();
        assert!(claim.starts_with("POST /runs/r1/claim "), "{claim}");
        // The whole point: a completion follows anyway, so a run that WAS
        // claimed comes back rather than sitting at `running`.
        let (release, release_raw) = rx.recv().unwrap();
        assert!(release.starts_with("PATCH /runs/r1 "), "{release}");
        assert!(release_raw.contains("runner_unavailable"), "{release_raw}");
        assert!(
            release_raw.contains("oaiy-test"),
            "the release must present the claiming instance, or the provider will not honour it"
        );
        // And no graph fetch and no run: the work still has to wait for a claim
        // this desktop actually holds.
        assert!(
            rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "nothing may execute off an ambiguous claim"
        );
    }

    #[test]
    fn a_claim_the_provider_rejected_outright_is_not_completed() {
        // A 4xx says the transition never happened, so the run is still queued
        // for somebody. Reporting an outcome for it would be this desktop
        // finalising work it never took.
        let (base, rx) = stub_provider(ONE_RUN, NO_MATCHING_FLOW, "400 Bad Request", "200 OK");
        let (_, trouble) =
            poll_once(&account(base), &spec(), &Lane::of(&spec()).unwrap(), "oaiy-test", None)
                .unwrap();
        assert!(trouble.is_some_and(|t| t.contains("400")));

        let mut patches = 0;
        while let Ok((line, _)) = rx.recv_timeout(Duration::from_millis(400)) {
            if line.starts_with("PATCH") {
                patches += 1;
            }
        }
        assert_eq!(patches, 0, "a rejected claim leaves the run queued, untouched");
    }

    #[test]
    fn a_claim_that_five_hundreds_is_treated_as_ambiguous_too() {
        // Same shape as a lost connection: the provider may have moved the row
        // and then failed on its way back out.
        let (base, rx) = stub_provider(
            ONE_RUN,
            NO_MATCHING_FLOW,
            "500 Internal Server Error",
            "200 OK",
        );
        let _ = poll_once(&account(base), &spec(), &Lane::of(&spec()).unwrap(), "oaiy-test", None)
            .unwrap();

        let mut released = false;
        while let Ok((line, raw)) = rx.recv_timeout(Duration::from_millis(400)) {
            if line.starts_with("PATCH") {
                assert!(raw.contains("runner_unavailable"), "{raw}");
                released = true;
            }
            assert!(!line.starts_with("GET /flows"), "a 5xx claim is not a claim");
        }
        assert!(released, "an ambiguous claim must be released");
    }

    #[test]
    fn a_completion_the_provider_refuses_is_retried_and_then_shouted_about() {
        // The unrecoverable one. The run is claimed, so it is marked running;
        // if the completion never lands nothing else will ever touch it.
        let (base, rx) = stub_provider(
            ONE_RUN,
            NO_MATCHING_FLOW,
            "200 OK",
            "500 Internal Server Error",
        );
        let (handled, trouble) =
            poll_once(&account(base), &spec(), &Lane::of(&spec()).unwrap(), "oaiy-test", None)
                .unwrap();
        assert_eq!(handled, 1);
        let reported = trouble.expect("a lost completion must reach the status");
        assert!(reported.contains("stuck at running"), "{reported}");
        assert!(reported.contains("r1"), "it must name the run: {reported}");

        let mut patches = 0;
        while let Ok((line, _)) = rx.recv_timeout(Duration::from_millis(500)) {
            if line.starts_with("PATCH") {
                patches += 1;
            }
        }
        assert_eq!(patches, COMPLETE_ATTEMPTS, "the completion must be retried");
    }

    #[test]
    fn a_completion_that_lost_the_run_is_not_retried() {
        // 409 means somebody else finalised it. Repeating the PATCH would only
        // repeat the refusal, and this desktop has nothing left to say.
        let (base, rx) = stub_provider(ONE_RUN, NO_MATCHING_FLOW, "200 OK", "409 Conflict");
        let (_, trouble) =
            poll_once(&account(base), &spec(), &Lane::of(&spec()).unwrap(), "oaiy-test", None)
                .unwrap();
        assert_eq!(trouble, None, "a run finalised elsewhere is not a fault");

        let mut patches = 0;
        while let Ok((line, _)) = rx.recv_timeout(Duration::from_millis(500)) {
            if line.starts_with("PATCH") {
                patches += 1;
            }
        }
        assert_eq!(patches, 1);
    }

    #[test]
    fn the_credential_is_presented_on_every_leg_and_never_in_the_url() {
        // Four separate requests; a bearer missing on any one turns into a 401
        // that reads as a dead link.
        let (base, rx) = stub_provider(ONE_RUN, NO_MATCHING_FLOW, "200 OK", "200 OK");
        poll_once(&account(base), &spec(), &Lane::of(&spec()).unwrap(), "oaiy-test", None).unwrap();

        for leg in ["queue", "claim", "graph", "complete"] {
            let (line, raw) = rx.recv().unwrap();
            assert!(raw.contains("flk_secret"), "the {leg} leg carried no bearer");
            // URLs end up in access logs and error messages; headers do not.
            assert!(!line.contains("flk_secret"), "the {leg} leg put it in the URL: {line}");
        }
    }

    #[test]
    fn a_revoked_key_says_so_instead_of_reporting_an_empty_queue() {
        // An empty queue and a rejected one look identical from the panel
        // otherwise, and the fix for the second one is to link again.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let body = r#"{"message":"Invalid API key"}"#;
                let _ = write!(
                    stream,
                    "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        let err = poll_once(
            &account(format!("http://127.0.0.1:{port}")),
            &spec(),
            &Lane::of(&spec()).unwrap(),
            "oaiy-test",
            None,
        )
        .unwrap_err();
        assert!(err.contains("link again"), "{err}");
        assert!(err.contains("Invalid API key"), "the provider's reason must survive: {err}");
    }
}
