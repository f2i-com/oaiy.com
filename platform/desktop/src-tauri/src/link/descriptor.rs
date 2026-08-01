//! What a linkable provider looks like, as DATA.
//!
//! The point of this file is that no code below it names FormLogic. A provider
//! is a descriptor — endpoints, a client id, scopes, and where to find the
//! credential in the token response — so adding a second provider is a JSON
//! file, not a branch in the link flow.
//!
//! Descriptors load from two places, exactly like service templates: built-ins
//! compiled into the binary, and user files under `<data>/connectors/*.json`
//! which override a built-in of the same id.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Descriptors shipped with the app. One today; the list is the only place a
/// specific provider is named.
const BUILTIN: &[&str] = &[include_str!("../../resources/connectors/formlogic.json")];

/// How a provider authenticates a desktop.
///
/// An enum with one variant today, tagged in JSON, so a provider using a
/// different ceremony (device code, paste-a-key) is a new variant rather than a
/// reinterpretation of these fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AuthSpec {
    /// OAuth 2.1 authorization code + PKCE S256, with the redirect landing on a
    /// loopback listener this app binds for the duration of the ceremony.
    Oauth2Pkce(Oauth2Pkce),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Oauth2Pkce {
    /// Sent as `client_id`. A public client — there is no secret, because a
    /// secret shipped in a desktop binary is not a secret.
    pub client_id: String,
    /// Joined to the base URL the user supplies. Relative so one descriptor
    /// serves every deployment of that provider.
    pub authorize_path: String,
    pub token_path: String,
    pub scopes: Vec<String>,
    /// Path on our loopback listener. The provider must have registered a
    /// matching loopback redirect.
    #[serde(default = "default_callback_path")]
    pub callback_path: String,
    /// Query parameter carrying a human label for this machine, if the provider
    /// shows one on its consent screen. Omitted when it does not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_param: Option<String>,
    /// Anything else the provider requires on the authorize URL.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_authorize_params: BTreeMap<String, String>,
    /// Where the interesting values live in the token response.
    pub token_response: TokenResponseSpec,
}

fn default_callback_path() -> String {
    "/callback".to_string()
}

/// Which fields of the token response mean what.
///
/// Providers disagree here — one returns `access_token`, another a
/// product-specific key name — and that disagreement is data, not a reason for
/// the exchange code to know about any of them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TokenResponseSpec {
    /// The credential to store. Tried in order; the first present non-empty
    /// string wins.
    pub credential_fields: Vec<String>,
    /// A stable id for the connection the provider created, if it makes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id_field: Option<String>,
    /// A human label to show ("Reception PC").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_name_field: Option<String>,
    /// The scopes actually granted, which may be narrower than requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_field: Option<String>,
}

/// A provider this desktop can link to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConnectorDescriptor {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub docs_url: Option<String>,
    /// Prefilled in the UI. The user can still point at their own deployment,
    /// which is why every path above is relative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_base_url: Option<String>,
    pub auth: AuthSpec,
    /// Optional GET that proves the stored credential still works.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health_path: Option<String>,
    /// How this provider is told the desktop is still reachable. Omitted for a
    /// provider that tracks presence some other way, or not at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heartbeat: Option<HeartbeatSpec>,
    /// How this provider queues remote-control commands for the desktop.
    /// Omitted for a provider that has no such lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay: Option<RelaySpec>,
    /// How this provider tunnels end-to-end sealed AI turns to the desktop.
    /// Omitted for a provider whose web app has no such feature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desktop_ai: Option<DesktopAiSpec>,
    /// How this desktop enrols as a storage node the account can approve.
    /// Omitted for a provider with no such notion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_node: Option<DataNodeSpec>,
    /// How an event on this desktop reaches the account's own flows. Omitted
    /// for a provider with no flows, whose events then stay local.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flows: Option<FlowsSpec>,
    /// How an event on this desktop reaches the account's own LOGIC SCRIPTS —
    /// a lane separate from flows. Omitted for a provider with no such notion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_logic: Option<AppLogicSpec>,
}

/// The account's app-logic lane: scripts the provider's apps carry, which turn
/// one event into a list of EFFECTS this desktop performs.
///
/// A third mechanism, next to the local trigger dispatch and the flows lane.
/// Flows are graphs the account runs; these are small scripts an app ships with
/// itself, and they are how a transcript gets written at all — nothing else in
/// this app implements that.
///
/// The whole point of this block is the same as [`FlowsSpec::nodes`]: a script
/// returns effects named in ITS OWN vocabulary, and neither those names nor the
/// field names inside them belong in this app. The provider declares both; the
/// OPERATIONS are a small closed set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AppLogicSpec {
    /// Where the account's apps, their forms and their scripts are read from.
    pub path: String,
    /// The value of a script's hook that means "an event happened". A provider
    /// may carry scripts for other moments entirely, and running one of those
    /// on an event would perform work nobody asked for.
    pub event_hook: String,
    /// Create a record. `{formId}` is substituted.
    pub submit_path: String,
    /// Read records back, so an effect that names a record by a FIELD rather
    /// than an id can find it. `{formId}` is substituted.
    pub list_path: String,
    /// Update a record. `{formId}` and `{id}` are substituted.
    pub update_path: String,
    /// How many records one match scans at most.
    ///
    /// The listing is not filterable, so a match is resolved by reading the
    /// newest page and looking. Too small silently stops matching old records;
    /// too large turns every correction into a large download.
    #[serde(default = "default_app_logic_scan")]
    pub match_scan_limit: u32,
    /// What the fields of an effect — and of a record on the wire — are called.
    pub fields: AppLogicFields,
    /// What the parts of the fetched app list are called.
    pub catalogue: AppLogicCatalogue,
    /// The effect types this provider's scripts emit, and what each one means
    /// in terms this desktop can actually perform.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<AppLogicEffectSpec>,
}

fn default_app_logic_scan() -> u32 {
    200
}

/// One effect type a provider's scripts emit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AppLogicEffectSpec {
    /// The name the provider's scripts use for it.
    pub effect_type: String,
    /// What it does, from a closed set this desktop knows how to perform.
    pub operation: AppLogicOperation,
}

/// The operations a provider may map its effect types onto.
///
/// Closed on purpose, and small, for the same reason [`FlowNodeOperation`] is:
/// a script is remote input to this desktop — anyone who can publish an app can
/// author one — so the set of things an effect can DO is fixed here, and the
/// provider only chooses which of them its vocabulary maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AppLogicOperation {
    /// Create a record.
    SubmitRecord,
    /// Update a record, found by id or by one of its own fields.
    UpdateRecord,
    /// Remember something under a key, so a redelivery can be recognised.
    SetStorage,
    /// Say something to the person at this machine.
    Notify,
    /// Call a plugin connector on this desktop — the same gate the relay uses.
    ConnectorRequest,
}

/// What the parts of an effect are called.
///
/// Field names are data here for exactly the reason they are in
/// [`TokenResponseSpec`]: providers disagree about them, and that disagreement
/// must not become a branch in the code that performs the effect. Every name
/// below is required — a blank one would read an effect field called "" and
/// find nothing, on every event, silently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AppLogicFields {
    /// On an effect: which of the declared effect types this one is.
    pub kind: String,
    /// On an effect: which form the record belongs to, in the app's own
    /// portable naming rather than the account's record ids.
    pub form_key: String,
    /// The record's values. Used both inside an effect and as the wrapper the
    /// write body puts them in, because they are one idea on both sides.
    pub record: String,
    /// On an update effect: the record's id, when the script already knows it.
    pub record_id: String,
    /// On an update effect: the block naming a field to find the record by.
    pub match_block: String,
    /// Inside that block: which field, and what it must equal.
    pub match_field: String,
    pub match_value: String,
    /// On an update effect: create the record when the match finds none.
    pub upsert: String,
    /// On a storage effect: the key, and the value stored under it.
    pub storage_key: String,
    pub storage_value: String,
    /// On a notify effect: what to say, and how loudly.
    pub message: String,
    pub level: String,
    /// On a connector effect: whose command, which command, and its body.
    pub connector_id: String,
    pub command: String,
    pub payload: String,
    /// In a listing reply: the array of records, and a record's own id.
    pub items: String,
    pub id: String,
    /// On what a script RETURNS: the list of effects. One provider's scripts
    /// answer `{ effects: [...] }`, another's might say `{ actions: [...] }`,
    /// and a name spelled in the code would read a key that is not there and
    /// perform nothing at all, on every event.
    pub effects_list: String,
}

/// What the parts of the fetched app list are called.
///
/// The catalogue is as much the provider's vocabulary as an effect is: the
/// wrapper around the apps, the block a set of scripts lives in, the key a form
/// is named by. Spelling any of them in the code would make this lane work for
/// exactly one provider while looking general — and the way it would fail for a
/// second one is silence, because a key that is not there simply reads as an
/// app with no scripts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AppLogicCatalogue {
    /// In the reply: the array of installed apps.
    pub apps: String,
    /// On one of those: the block identifying the app, and the two names it
    /// may be identified by. The id is preferred; the slug is the fallback.
    pub app_block: String,
    pub app_id: String,
    pub app_slug: String,
    /// On one of those: the block the scripts live in, and the array inside it.
    pub logic_block: String,
    pub scripts: String,
    /// On one script: its name, WHEN it runs, and the JavaScript itself.
    pub script_id: String,
    pub script_hook: String,
    pub script_source: String,
    /// On an app: its forms, a form's own id, and the portable key the app's
    /// scripts name that form by.
    pub forms: String,
    pub form_id: String,
    pub form_key: String,
}

impl AppLogicCatalogue {
    /// Every name, labelled, for the "nothing may be blank" rule.
    fn all(&self) -> [(&'static str, &str); 12] {
        [
            ("apps", self.apps.as_str()),
            ("appBlock", self.app_block.as_str()),
            ("appId", self.app_id.as_str()),
            ("appSlug", self.app_slug.as_str()),
            ("logicBlock", self.logic_block.as_str()),
            ("scripts", self.scripts.as_str()),
            ("scriptId", self.script_id.as_str()),
            ("scriptHook", self.script_hook.as_str()),
            ("scriptSource", self.script_source.as_str()),
            ("forms", self.forms.as_str()),
            ("formId", self.form_id.as_str()),
            ("formKey", self.form_key.as_str()),
        ]
    }
}

impl AppLogicFields {
    /// Every name, labelled, for validation and for the "nothing may be blank"
    /// rule. Listed once so a new field cannot be added and left unchecked.
    fn all(&self) -> [(&'static str, &str); 18] {
        [
            ("kind", self.kind.as_str()),
            ("formKey", self.form_key.as_str()),
            ("record", self.record.as_str()),
            ("recordId", self.record_id.as_str()),
            ("matchBlock", self.match_block.as_str()),
            ("matchField", self.match_field.as_str()),
            ("matchValue", self.match_value.as_str()),
            ("upsert", self.upsert.as_str()),
            ("storageKey", self.storage_key.as_str()),
            ("storageValue", self.storage_value.as_str()),
            ("message", self.message.as_str()),
            ("level", self.level.as_str()),
            ("connectorId", self.connector_id.as_str()),
            ("command", self.command.as_str()),
            ("payload", self.payload.as_str()),
            ("items", self.items.as_str()),
            ("id", self.id.as_str()),
            ("effectsList", self.effects_list.as_str()),
        ]
    }
}

/// The account's flow lane: read the bindings, reserve a run, and — when the
/// provider offers the rest of the paths — claim a queued run and execute it.
///
/// Reserving is the minimum. A provider that names only `bindingsPath` and
/// `reservePath` gets runs queued for whatever runtime it already has; one that
/// also names the queue, the claim, the completion and where its graphs live
/// lets this desktop be that runtime. There is still no second flow engine —
/// the graph is handed to the same CLI the browser's engine is built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FlowsSpec {
    /// The account's enabled trigger bindings.
    pub bindings_path: String,
    /// Reserve one run.
    pub reserve_path: String,
    /// Queued runs waiting for a runtime to take them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queued_path: Option<String>,
    /// Take one queued run, exactly-once. `{id}` is substituted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_path: Option<String>,
    /// Report a run's terminal status. `{id}` is substituted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub complete_path: Option<String>,
    /// Where the account's flow GRAPHS are read from.
    ///
    /// A claimed run names a flow but does not carry it, and this desktop
    /// cannot execute what it cannot read — so a provider that wants its runs
    /// executed here says where the graphs are. Omitted by a provider that only
    /// wants runs QUEUED, which leaves the claim lane unusable and the runs to
    /// whatever runtime the provider already has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_path: Option<String>,
    /// The node types this provider's graphs are written in, and what each one
    /// means in terms this desktop can actually perform.
    ///
    /// The whole point of this block: a graph names nodes in ITS OWN
    /// vocabulary — one provider calls a node `formlogic_list_responses`,
    /// another might call the same idea `records.query`. Neither name belongs
    /// in this app. The provider says what its nodes are called and which
    /// operation each performs; the operations are a small closed set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<FlowNodeSpec>,
    /// How a value reference is spelled, for this whole lane.
    ///
    /// The same language appears in three places — the map from a binding's
    /// trigger to its flow's inputs, the fields of a node, and what a binding
    /// does with the answer — so it is described once here rather than per
    /// block, where the copies would drift.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selectors: Option<SelectorSpec>,
    /// What a binding calls the map from flow input names to value references.
    ///
    /// Absent means this desktop does not build a flow's inputs at all, and a
    /// flow reading `$inputs.x` finds nothing — which is what happened before
    /// this existed: seventeen of nineteen bindings declared one and every one
    /// was discarded, so every flow that named its trigger's fields ran blind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_map_field: Option<String>,
    /// What a BINDING asks for once its flow has finished.
    ///
    /// A third thing a binding says, next to which flow to run and when: what to
    /// DO with the answer. Omitted by a provider whose bindings carry no such
    /// list, whose runs then simply finish.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_actions: Option<ResultActionsSpec>,
}

/// Actions a binding performs with a finished run's result.
///
/// Every name here is author-facing and therefore the provider's — including
/// the key the list itself arrives under, for the same reason
/// [`AppLogicFields::effects_list`] is data: a name spelled in the code would
/// read a key that is not there and perform nothing at all, silently, on every
/// run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResultActionsSpec {
    /// The key on a BINDING carrying the list.
    pub actions_field: String,
    /// The key, inside a reported result, listing actions that failed. The
    /// provider's console reads it to mark a run that succeeded while its side
    /// effects did not.
    pub errors_field: String,
    /// The key a non-object result is wrapped under before that list can be
    /// attached — a flow answering with a bare string has nowhere to carry one,
    /// and the failure would vanish behind a clean success.
    pub result_wrapper: String,
    /// Most actions one binding may perform. A binding is remote input; a
    /// misconfigured one must not turn a single run into unbounded work.
    #[serde(default = "default_max_result_actions")]
    pub max_actions: usize,
    /// Create a record. `{formId}` is substituted.
    pub submit_path: String,
    /// Update a record. `{formId}` and `{id}` are substituted.
    pub update_path: String,
    /// What the parts of one action are called.
    pub fields: ResultActionFields,
    /// The action types this provider's bindings carry, and what each one means
    /// in terms this desktop can actually perform.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<ResultActionSpec>,
}

fn default_max_result_actions() -> usize {
    16
}

/// How a provider spells a value reference.
///
/// Two spellings of one language: a selector standing alone as a whole value,
/// and the same path interpolated inside free text. Both are the provider's
/// notation, so both are described rather than assumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SelectorSpec {
    /// What marks a string as a reference rather than text.
    pub sigil: String,
    /// What negates a gate, repeatable.
    pub negate: String,
    /// The delimiters around an interpolated path.
    pub open: String,
    pub close: String,
    /// The roots a reference may address. A root that is DECLARED but has no
    /// value here still resolves to nothing — which is the point of declaring
    /// it. Leaving it out instead would make the author's reference a literal,
    /// and write the text "$nodes.x" into somebody's record as if it were data.
    pub roots: Vec<SelectorRoot>,
}

/// One addressable root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SelectorRoot {
    /// The name an author writes after the sigil.
    pub name: String,
    /// Which value it addresses, from a closed set.
    pub source: SelectorSource,
}

/// The values a selector root may address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SelectorSource {
    /// What the finished flow answered.
    RunResult,
    /// The event that triggered the run.
    RunEvent,
    /// The inputs the run was started with.
    RunInputs,
    /// Which app the flow belongs to.
    AppContext,
    /// Addressable, and never has a value out here. Declared so an author's
    /// reference to it resolves to nothing rather than passing through as text.
    Unavailable,
}

/// One action type a provider's bindings carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResultActionSpec {
    /// The name the provider's bindings use for it.
    pub action_type: String,
    /// What it does, from a closed set this desktop knows how to perform.
    pub operation: ResultActionOperation,
}

/// The operations a provider may map its action types onto.
///
/// Closed for the same reason [`FlowNodeOperation`] is: a binding is remote
/// input, authored by anyone who can build an app on the provider, so what an
/// action can DO is fixed here and the provider only chooses which of them its
/// vocabulary names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ResultActionOperation {
    /// Create a record.
    SubmitRecord,
    /// Update a record named by its id.
    UpdateRecord,
    /// Say something to the person at this machine.
    Notify,
    /// Call a plugin connector on this desktop — the same gate the relay uses.
    ConnectorRequest,
}

/// What the parts of one action are called.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResultActionFields {
    /// Which of the declared action types this one is.
    pub kind: String,
    /// The gate deciding whether it runs at all.
    pub gate: String,
    /// Which form the record belongs to. An id, not a portable key — a binding
    /// names forms the way the account does.
    pub form_id: String,
    /// The record's values, and the id of the one being updated.
    pub record: String,
    pub record_id: String,
    /// On a notify action: what to say.
    pub message: String,
    /// On a connector action: whose command, which command, and its body.
    pub connector_id: String,
    pub command: String,
    pub payload: String,
    /// In a write's reply: the record's own id.
    pub id: String,
}

/// One node type a provider's graphs use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FlowNodeSpec {
    /// The name the provider's graphs use for it.
    pub node_type: String,
    /// What it does, from a closed set this desktop knows how to perform.
    pub operation: FlowNodeOperation,
    /// The path the operation acts on, when it needs one. `{id}` and other
    /// placeholders are substituted from the node's own data at run time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// The operations a provider may map its node types onto.
///
/// Closed on purpose, and small. A node type is remote input to this desktop —
/// whatever is reachable here is reachable by anyone who can author a flow on
/// the provider — so the set of things a node can DO is fixed here, and the
/// provider only chooses which of them its vocabulary maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FlowNodeOperation {
    /// Read records back from the provider.
    ListRecords,
    /// Create a record.
    CreateRecord,
    /// Update a record.
    UpdateRecord,
    /// Call a plugin connector on this desktop — the same gate the relay uses.
    ConnectorRequest,
    /// Read or control this desktop's services.
    ServiceControl,
    /// The run's inputs, as the graph's entry node.
    RunInput,
    /// A chat completion from this machine's own AI.
    Chat,
}

/// Enrolment of this machine as a node the account owner approves by name and
/// by key. Registration only — holding data is a separate surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DataNodeSpec {
    /// Register or refresh this desktop's signing identity. Idempotent.
    pub register_path: String,
    /// Read back this desktop's own node record.
    pub self_path: String,
    /// How often to refresh. Enrolment is not presence — the provider tracks
    /// that separately — so this is measured in hours, not seconds.
    #[serde(default = "default_enrolment_interval")]
    pub interval_seconds: u64,
    /// What this node offers to do. `storage` is the only one defined so far.
    #[serde(default = "default_node_capabilities")]
    pub capabilities: Vec<String>,
    #[serde(default = "default_protocol")]
    pub protocol_min: u32,
    #[serde(default = "default_protocol")]
    pub protocol_max: u32,
}

fn default_enrolment_interval() -> u64 {
    3600
}

fn default_node_capabilities() -> Vec<String> {
    vec!["storage".to_string()]
}

fn default_protocol() -> u32 {
    1
}

/// The sealed AI lane: the provider's web app relays a chat turn the backend
/// cannot read, and this desktop answers it with the account's own model.
///
/// Separate from [`RelaySpec`] because it is a different lane with a different
/// queue and different semantics — a long chat turn must not block service
/// control, and the provider treats the two independently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DesktopAiSpec {
    /// Where this desktop publishes the public half of its sealing key. Until
    /// it does, the provider's web app cannot encrypt anything to it and says
    /// so — which is the whole visible symptom of this lane being absent.
    pub pubkey_path: String,
    /// Long-poll for sealed turns addressed to this instance.
    pub pending_path: String,
    /// Take one turn, exactly-once. `{id}` is substituted.
    pub claim_path: String,
    /// Append one sealed frame. `{id}` is substituted.
    pub frames_path: String,
    /// Report the terminal status. `{id}` is substituted.
    pub complete_path: String,
    /// Poll the sealed channel BACK from the web app — how a user's answer to
    /// an approval reaches this desktop. `{id}` is substituted.
    pub input_path: String,
    /// The account's tool catalogue, and where to run one. Omitted for a
    /// provider whose chat has no tools, which simply never offers any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_catalog_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_execute_path: Option<String>,
    #[serde(default = "default_relay_wait")]
    pub wait_seconds: u64,
    #[serde(default = "default_ai_batch")]
    pub batch_limit: u32,
    #[serde(default = "default_relay_backoff")]
    pub error_backoff_seconds: u64,
}

fn default_ai_batch() -> u32 {
    8
}

/// The long-poll / claim / complete lane a provider's web app uses to act on
/// this desktop from anywhere.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RelaySpec {
    /// Long-poll for work addressed to this instance.
    pub pending_path: String,
    /// Take one command, exactly-once. `{id}` is substituted.
    pub claim_path: String,
    /// Report the outcome. `{id}` is substituted.
    pub complete_path: String,
    /// How long the server may hold the poll open.
    #[serde(default = "default_relay_wait")]
    pub wait_seconds: u64,
    #[serde(default = "default_relay_batch")]
    pub batch_limit: u32,
    /// Pause after a failed poll, so a provider that is down does not become a
    /// hot loop against it.
    #[serde(default = "default_relay_backoff")]
    pub error_backoff_seconds: u64,
}

fn default_relay_wait() -> u64 {
    25
}

fn default_relay_batch() -> u32 {
    20
}

fn default_relay_backoff() -> u64 {
    10
}

/// The periodic ping that keeps a desktop looking online.
///
/// Providers decide reachability from how recently the desktop last spoke, so
/// the interval has to sit comfortably inside their window — FormLogic's is 90
/// seconds and its own desktop beats every 45.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HeartbeatSpec {
    pub path: String,
    #[serde(default = "default_heartbeat_interval")]
    pub interval_seconds: u64,
    /// Body field carrying this install's stable id.
    pub instance_id_field: String,
    /// Body field carrying a human label, when the provider shows one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name_field: Option<String>,
}

fn default_heartbeat_interval() -> u64 {
    45
}

impl ConnectorDescriptor {
    fn validate(&self) -> Result<(), String> {
        if self.id.is_empty()
            || self.id.len() > 64
            || !self
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        {
            return Err(format!("connector id {:?} is not a safe identifier", self.id));
        }
        if self.name.trim().is_empty() {
            return Err(format!("connector {:?} has no name", self.id));
        }
        let AuthSpec::Oauth2Pkce(o) = &self.auth;
        if o.client_id.trim().is_empty() {
            return Err(format!("connector {:?} has no clientId", self.id));
        }
        for (label, path) in [
            ("authorizePath", &o.authorize_path),
            ("tokenPath", &o.token_path),
            ("callbackPath", &o.callback_path),
        ] {
            if !path.starts_with('/') {
                return Err(format!(
                    "connector {:?} {label} must be a path beginning with '/', got {path:?}",
                    self.id
                ));
            }
        }
        if o.scopes.is_empty() {
            return Err(format!("connector {:?} requests no scopes", self.id));
        }
        if let Some(h) = &self.heartbeat {
            if !h.path.starts_with('/') {
                return Err(format!(
                    "connector {:?} heartbeat path must begin with '/', got {:?}",
                    self.id, h.path
                ));
            }
            if h.instance_id_field.trim().is_empty() {
                return Err(format!("connector {:?} heartbeat names no instance id field", self.id));
            }
            // A zero interval would spin; one longer than any plausible presence
            // window would beat too rarely to keep the desktop online at all.
            if h.interval_seconds == 0 || h.interval_seconds > 3600 {
                return Err(format!(
                    "connector {:?} heartbeat interval {} is out of range (1..3600s)",
                    self.id, h.interval_seconds
                ));
            }
        }
        if let Some(r) = &self.relay {
            for (label, path) in [
                ("pendingPath", &r.pending_path),
                ("claimPath", &r.claim_path),
                ("completePath", &r.complete_path),
            ] {
                if !path.starts_with('/') {
                    return Err(format!(
                        "connector {:?} relay {label} must begin with '/', got {path:?}",
                        self.id
                    ));
                }
            }
            // Without the placeholder every claim would hit one fixed URL and
            // quietly do nothing useful.
            for (label, path) in [("claimPath", &r.claim_path), ("completePath", &r.complete_path)] {
                if !path.contains("{id}") {
                    return Err(format!(
                        "connector {:?} relay {label} must contain the {{id}} placeholder",
                        self.id
                    ));
                }
            }
            if r.wait_seconds == 0 || r.wait_seconds > 300 {
                return Err(format!(
                    "connector {:?} relay wait {} is out of range (1..300s)",
                    self.id, r.wait_seconds
                ));
            }
        }
        if let Some(a) = &self.desktop_ai {
            for (label, path) in [
                ("pubkeyPath", &a.pubkey_path),
                ("pendingPath", &a.pending_path),
                ("claimPath", &a.claim_path),
                ("framesPath", &a.frames_path),
                ("completePath", &a.complete_path),
                ("inputPath", &a.input_path),
            ] {
                if !path.starts_with('/') {
                    return Err(format!(
                        "connector {:?} desktopAi {label} must begin with '/', got {path:?}",
                        self.id
                    ));
                }
            }
            for (label, path) in [
                ("toolsCatalogPath", &a.tools_catalog_path),
                ("toolsExecutePath", &a.tools_execute_path),
            ] {
                if let Some(p) = path {
                    if !p.starts_with('/') {
                        return Err(format!(
                            "connector {:?} desktopAi {label} must begin with '/', got {p:?}",
                            self.id
                        ));
                    }
                }
            }
            // Both or neither: a catalogue with nowhere to run a tool would
            // offer the model things it can only fail to call.
            if a.tools_catalog_path.is_some() != a.tools_execute_path.is_some() {
                return Err(format!(
                    "connector {:?} desktopAi names one of toolsCatalogPath/toolsExecutePath \
                     without the other",
                    self.id
                ));
            }
            for (label, path) in [
                ("claimPath", &a.claim_path),
                ("framesPath", &a.frames_path),
                ("completePath", &a.complete_path),
                ("inputPath", &a.input_path),
            ] {
                if !path.contains("{id}") {
                    return Err(format!(
                        "connector {:?} desktopAi {label} must contain the {{id}} placeholder",
                        self.id
                    ));
                }
            }
            // The publish and the poll are not per-turn, so a placeholder there
            // would be sent to the provider literally.
            for (label, path) in [("pubkeyPath", &a.pubkey_path), ("pendingPath", &a.pending_path)] {
                if path.contains("{id}") {
                    return Err(format!(
                        "connector {:?} desktopAi {label} takes no {{id}} placeholder",
                        self.id
                    ));
                }
            }
            if a.wait_seconds == 0 || a.wait_seconds > 300 {
                return Err(format!(
                    "connector {:?} desktopAi wait {} is out of range (1..300s)",
                    self.id, a.wait_seconds
                ));
            }
        }
        if let Some(n) = &self.data_node {
            for (label, path) in [
                ("registerPath", &n.register_path),
                ("selfPath", &n.self_path),
            ] {
                if !path.starts_with('/') {
                    return Err(format!(
                        "connector {:?} dataNode {label} must begin with '/', got {path:?}",
                        self.id
                    ));
                }
            }
            // A zero interval would hammer the provider with a registration it
            // treats as a heartbeat; a day is already generous for enrolment.
            if n.interval_seconds < 60 || n.interval_seconds > 86_400 {
                return Err(format!(
                    "connector {:?} dataNode interval {} is out of range (60..86400s)",
                    self.id, n.interval_seconds
                ));
            }
            if n.capabilities.is_empty() {
                return Err(format!("connector {:?} dataNode claims no capabilities", self.id));
            }
        }
        if let Some(f) = &self.flows {
            for (label, path) in [
                ("bindingsPath", &f.bindings_path),
                ("reservePath", &f.reserve_path),
            ] {
                if !path.starts_with('/') {
                    return Err(format!(
                        "connector {:?} flows {label} must begin with '/', got {path:?}",
                        self.id
                    ));
                }
            }
            for (label, path) in [
                ("queuedPath", &f.queued_path),
                ("claimPath", &f.claim_path),
                ("completePath", &f.complete_path),
                ("graphPath", &f.graph_path),
            ] {
                if let Some(p) = path {
                    if !p.starts_with('/') {
                        return Err(format!(
                            "connector {:?} flows {label} must begin with '/', got {p:?}",
                            self.id
                        ));
                    }
                }
            }
            // Claiming is only meaningful with somewhere to report the outcome:
            // a run taken and never completed is worse than one left queued,
            // because it looks to everyone else like it is being worked on.
            if f.claim_path.is_some() != f.complete_path.is_some() {
                return Err(format!(
                    "connector {:?} flows names one of claimPath/completePath without the other",
                    self.id
                ));
            }
            // Without the placeholder every claim and every completion would hit
            // one fixed URL — the failure the relay lane already learned.
            for (label, path) in [("claimPath", &f.claim_path), ("completePath", &f.complete_path)] {
                if let Some(p) = path {
                    if !p.contains("{id}") {
                        return Err(format!(
                            "connector {:?} flows {label} must contain the {{id}} placeholder",
                            self.id
                        ));
                    }
                }
            }
            // These two are not per-run, so a placeholder in them would be sent
            // to the provider literally and fetch nothing.
            for (label, path) in [("queuedPath", &f.queued_path), ("graphPath", &f.graph_path)] {
                if let Some(p) = path {
                    if p.contains("{id}") {
                        return Err(format!(
                            "connector {:?} flows {label} takes no {{id}} placeholder",
                            self.id
                        ));
                    }
                }
            }
            let mut seen = std::collections::BTreeSet::new();
            for node in &f.nodes {
                if node.node_type.trim().is_empty() {
                    return Err(format!("connector {:?} flows has a node with no type", self.id));
                }
                // A duplicate would make which operation wins depend on file
                // order, and the loser would fail in a way nothing explains.
                if !seen.insert(node.node_type.as_str()) {
                    return Err(format!(
                        "connector {:?} flows declares the node type {:?} twice",
                        self.id, node.node_type
                    ));
                }
                if let Some(p) = &node.path {
                    if !p.starts_with('/') {
                        return Err(format!(
                            "connector {:?} flows node {:?} path must begin with '/', got {p:?}",
                            self.id, node.node_type
                        ));
                    }
                }
            }
        }
        if let Some(a) = &self.app_logic {
            for (label, path) in [
                ("path", &a.path),
                ("submitPath", &a.submit_path),
                ("listPath", &a.list_path),
                ("updatePath", &a.update_path),
            ] {
                if !path.starts_with('/') {
                    return Err(format!(
                        "connector {:?} appLogic {label} must begin with '/', got {path:?}",
                        self.id
                    ));
                }
            }
            // A write path with no form placeholder would send every record of
            // every form to one URL — which either fails on all of them or,
            // worse, succeeds on the wrong form.
            for (label, path) in [
                ("submitPath", &a.submit_path),
                ("listPath", &a.list_path),
                ("updatePath", &a.update_path),
            ] {
                if !path.contains("{formId}") {
                    return Err(format!(
                        "connector {:?} appLogic {label} must contain the {{formId}} placeholder",
                        self.id
                    ));
                }
            }
            // An update with no record placeholder would address the whole
            // collection; a create or a listing WITH one would ask the provider
            // for a record literally called "{id}".
            if !a.update_path.contains("{id}") {
                return Err(format!(
                    "connector {:?} appLogic updatePath must contain the {{id}} placeholder",
                    self.id
                ));
            }
            for (label, path) in [
                ("path", &a.path),
                ("submitPath", &a.submit_path),
                ("listPath", &a.list_path),
            ] {
                if path.contains("{id}") {
                    return Err(format!(
                        "connector {:?} appLogic {label} takes no {{id}} placeholder",
                        self.id
                    ));
                }
            }
            if a.path.contains("{formId}") {
                return Err(format!(
                    "connector {:?} appLogic path takes no {{formId}} placeholder",
                    self.id
                ));
            }
            if a.event_hook.trim().is_empty() {
                return Err(format!(
                    "connector {:?} appLogic names no eventHook, so it could not tell an \
                     event script from any other kind",
                    self.id
                ));
            }
            // Zero would match nothing and make every update look like a fresh
            // record; a listing large enough to be a download is its own fault.
            if a.match_scan_limit == 0 || a.match_scan_limit > 5000 {
                return Err(format!(
                    "connector {:?} appLogic matchScanLimit {} is out of range (1..5000)",
                    self.id, a.match_scan_limit
                ));
            }
            for (label, name) in a.fields.all() {
                if name.trim().is_empty() {
                    return Err(format!(
                        "connector {:?} appLogic fields.{label} is blank, which would read a \
                         field called \"\" on every effect",
                        self.id
                    ));
                }
            }
            // Same rule one level up: a blank catalogue name reads a key that
            // is not there, which is indistinguishable from an account with no
            // apps — the lane would run nothing and say nothing.
            for (label, name) in a.catalogue.all() {
                if name.trim().is_empty() {
                    return Err(format!(
                        "connector {:?} appLogic catalogue.{label} is blank, which would read a \
                         key called \"\" in the app list",
                        self.id
                    ));
                }
            }
            let mut seen = std::collections::BTreeSet::new();
            for effect in &a.effects {
                if effect.effect_type.trim().is_empty() {
                    return Err(format!(
                        "connector {:?} appLogic has an effect with no type",
                        self.id
                    ));
                }
                // A duplicate would make which operation wins depend on file
                // order, and the loser would fail in a way nothing explains.
                if !seen.insert(effect.effect_type.as_str()) {
                    return Err(format!(
                        "connector {:?} appLogic declares the effect type {:?} twice",
                        self.id, effect.effect_type
                    ));
                }
            }
            // A lane that maps nothing would run every script and then refuse
            // every effect it produced — all the cost, none of the result.
            if a.effects.is_empty() {
                return Err(format!(
                    "connector {:?} appLogic maps no effect types, so no script could do \
                     anything",
                    self.id
                ));
            }
        }
        if o.token_response.credential_fields.is_empty() {
            return Err(format!(
                "connector {:?} names no credential field, so a successful \
                 exchange could not be read",
                self.id
            ));
        }
        Ok(())
    }
}

/// The descriptors compiled into this binary, parsed and validated.
///
/// Separate from [`load_all`] so a caller that wants the SHIPPED provider — a
/// test, or anything that must not depend on what happens to be in the user's
/// data directory — can have it without reading the disk.
pub fn builtin() -> Vec<ConnectorDescriptor> {
    let mut out: Vec<ConnectorDescriptor> = Vec::new();
    for raw in BUILTIN {
        match serde_json::from_str::<ConnectorDescriptor>(raw) {
            Ok(d) => match d.validate() {
                Ok(()) => out.push(d),
                // A broken built-in is our bug, not the user's; loud in dev.
                Err(e) => debug_assert!(false, "built-in connector is invalid: {e}"),
            },
            Err(e) => debug_assert!(false, "built-in connector does not parse: {e}"),
        }
    }
    out
}

/// Every descriptor available, built-ins first, user files overriding by id.
///
/// A malformed user file is skipped with a log rather than failing the load —
/// one bad file must not take away every provider, including the working ones.
pub fn load_all(data_dir: &Path) -> Vec<ConnectorDescriptor> {
    let mut out: Vec<ConnectorDescriptor> = builtin();

    let dir = data_dir.join("connectors");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(&path) else { continue };
        // Notepad and PowerShell's `-Encoding utf8` both prepend a byte order
        // mark, and serde stops at it with "expected value at line 1 column 1".
        // Since the file is then skipped with only a log line, the user is told
        // to drop a descriptor in a folder and gets no provider and no reason.
        match serde_json::from_str::<ConnectorDescriptor>(raw.trim_start_matches('\u{feff}')) {
            Ok(d) => match d.validate() {
                Ok(()) => {
                    if let Some(slot) = out.iter_mut().find(|e| e.id == d.id) {
                        *slot = d;
                    } else {
                        out.push(d);
                    }
                }
                Err(e) => log::warn!("ignoring connector {}: {e}", path.display()),
            },
            Err(e) => log::warn!("ignoring connector {}: {e}", path.display()),
        }
    }
    out
}

pub fn find(data_dir: &Path, id: &str) -> Option<ConnectorDescriptor> {
    load_all(data_dir).into_iter().find(|d| d.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("oaiy-conn-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(p.join("connectors")).unwrap();
        p
    }

    #[test]
    fn the_builtin_descriptor_parses_and_validates() {
        // It is compiled in, so a mistake here ships. The debug_assert in
        // load_all would fire in dev, but this says so with a name.
        let all = load_all(std::path::Path::new("/nonexistent"));
        assert_eq!(all.len(), 1, "expected exactly the shipped built-in");
        let d = &all[0];
        assert_eq!(d.id, "formlogic");
        d.validate().unwrap();
        let AuthSpec::Oauth2Pkce(o) = &d.auth;
        assert!(!o.scopes.is_empty());
        assert!(
            o.token_response.credential_fields.iter().any(|f| f == "formlogic_api_key"),
            "the provider's own key field must be first-class"
        );
    }

    #[test]
    fn a_user_file_overrides_a_builtin_of_the_same_id() {
        // How someone points at a fork or a staging deployment without waiting
        // for a release.
        let d = dir("override");
        let mut custom: ConnectorDescriptor =
            serde_json::from_str(BUILTIN[0]).unwrap();
        custom.name = "My FormLogic".into();
        std::fs::write(
            d.join("connectors").join("formlogic.json"),
            serde_json::to_string(&custom).unwrap(),
        )
        .unwrap();

        let all = load_all(&d);
        assert_eq!(all.len(), 1, "an override must replace, not duplicate");
        assert_eq!(all[0].name, "My FormLogic");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_provider_names_its_own_flow_node_vocabulary() {
        // The point of the block. A graph names nodes in ITS vocabulary — one
        // provider says `formlogic_list_responses`, another might say
        // `records.query` for the same idea. Neither name belongs in this app,
        // so the provider declares the mapping and the OPERATIONS are the
        // closed set this desktop knows how to perform.
        let d = find(std::path::Path::new("/nonexistent"), "formlogic").unwrap();
        let f = d.flows.expect("the connector must declare its flow lane");
        assert!(!f.nodes.is_empty(), "a provider with flows declares its node types");

        let by_op = |op: FlowNodeOperation| -> usize {
            f.nodes.iter().filter(|n| n.operation == op).count()
        };
        assert_eq!(by_op(FlowNodeOperation::RunInput), 1);
        assert_eq!(by_op(FlowNodeOperation::Chat), 1);
        assert!(by_op(FlowNodeOperation::ListRecords) >= 1);
        assert!(by_op(FlowNodeOperation::ConnectorRequest) >= 1);

        // Nothing in the CODE names a provider's node type — every one of them
        // comes from the file.
        assert!(f.nodes.iter().any(|n| n.node_type == "formlogic_list_responses"));
        // …and a record operation says WHERE, because the operation alone
        // cannot know the provider's URL shape.
        let list = f
            .nodes
            .iter()
            .find(|n| n.operation == FlowNodeOperation::ListRecords)
            .unwrap();
        assert!(list.path.as_deref().is_some_and(|p| p.starts_with('/')));
    }

    #[test]
    fn a_flow_lane_that_can_claim_must_be_able_to_report() {
        // A run taken and never completed is worse than one left queued: it
        // looks to every other runtime like it is being worked on.
        let mut d: ConnectorDescriptor = serde_json::from_str(BUILTIN[0]).unwrap();
        let flows = d.flows.as_mut().unwrap();
        flows.complete_path = None;
        let err = d.validate().unwrap_err();
        assert!(err.contains("claimPath/completePath"), "{err}");
    }

    #[test]
    fn a_flow_lane_that_can_claim_says_where_the_graphs_are() {
        // A claimed run names its flow but does not carry it. Without somewhere
        // to read the graph this desktop would claim runs it cannot execute —
        // the worst state of all, since a claimed run is one no other runtime
        // will take either.
        let d = find(std::path::Path::new("/nonexistent"), "formlogic").unwrap();
        let f = d.flows.unwrap();
        assert!(f.queued_path.is_some());
        assert!(f.graph_path.as_deref().is_some_and(|p| p.starts_with('/')));
    }

    #[test]
    fn the_per_run_paths_take_a_placeholder_and_the_list_paths_do_not() {
        // Both directions are silent failures: a claim path without {id} sends
        // every claim to one URL, and a queue path WITH one asks the provider
        // for a run literally called "{id}".
        let base: ConnectorDescriptor = serde_json::from_str(BUILTIN[0]).unwrap();

        let mut no_placeholder = base.clone();
        no_placeholder.flows.as_mut().unwrap().claim_path = Some("/api/v1/flow-runs/claim".into());
        let err = no_placeholder.validate().unwrap_err();
        assert!(err.contains("claimPath"), "{err}");

        let mut stray = base.clone();
        stray.flows.as_mut().unwrap().graph_path = Some("/api/v1/flows/{id}".into());
        let err = stray.validate().unwrap_err();
        assert!(err.contains("graphPath"), "{err}");
    }

    #[test]
    fn a_duplicated_node_type_is_refused_rather_than_resolved_by_file_order() {
        let mut d: ConnectorDescriptor = serde_json::from_str(BUILTIN[0]).unwrap();
        let flows = d.flows.as_mut().unwrap();
        let first = flows.nodes[0].clone();
        flows.nodes.push(first);
        let err = d.validate().unwrap_err();
        assert!(err.contains("twice"), "{err}");
    }

    #[test]
    fn a_provider_names_its_own_effect_vocabulary_and_its_own_field_names() {
        // The same point as the flow node block, one level down. A script
        // returns effects named in ITS vocabulary, with ITS field names inside
        // them — one provider says `formlogic.submitResponse` with `answers`,
        // another might say `records.create` with `values`. None of those names
        // belongs in this app, so they are all declared here and the OPERATIONS
        // are the closed set.
        let d = find(std::path::Path::new("/nonexistent"), "formlogic").unwrap();
        let a = d.app_logic.expect("the connector must declare its app-logic lane");
        assert!(!a.effects.is_empty());

        let by_op = |op: AppLogicOperation| a.effects.iter().filter(|e| e.operation == op).count();
        assert!(by_op(AppLogicOperation::SubmitRecord) >= 1);
        assert!(by_op(AppLogicOperation::UpdateRecord) >= 1);
        assert_eq!(by_op(AppLogicOperation::SetStorage), 1);
        assert!(by_op(AppLogicOperation::Notify) >= 1);
        assert_eq!(by_op(AppLogicOperation::ConnectorRequest), 1);

        // Nothing in the CODE names a provider's effect type or field.
        assert!(a.effects.iter().any(|e| e.effect_type == "formlogic.submitResponse"));
        assert_eq!(a.fields.record, "answers");
        assert_eq!(a.fields.kind, "type");
        // …and every name is filled in, because a blank one reads a field
        // called "" on every effect and finds nothing, silently.
        assert!(a.fields.all().iter().all(|(_, v)| !v.trim().is_empty()));
    }

    #[test]
    fn an_app_logic_lane_that_maps_nothing_is_refused() {
        // It would run every script and then refuse every effect they produced:
        // all of the cost, none of the result.
        let mut d: ConnectorDescriptor = serde_json::from_str(BUILTIN[0]).unwrap();
        d.app_logic.as_mut().unwrap().effects.clear();
        let err = d.validate().unwrap_err();
        assert!(err.contains("maps no effect types"), "{err}");
    }

    #[test]
    fn the_record_paths_take_the_placeholders_they_need_and_no_others() {
        // Every one of these is a silent failure: a write path with no form
        // placeholder addresses one fixed form, an update with no record
        // placeholder addresses the whole collection, and a catalogue path WITH
        // a placeholder asks the provider for an app literally called "{id}".
        let base: ConnectorDescriptor = serde_json::from_str(BUILTIN[0]).unwrap();

        let mut no_form = base.clone();
        no_form.app_logic.as_mut().unwrap().submit_path = "/api/v1/responses".into();
        assert!(no_form.validate().unwrap_err().contains("submitPath"));

        let mut no_record = base.clone();
        no_record.app_logic.as_mut().unwrap().update_path = "/api/v1/forms/{formId}/responses".into();
        assert!(no_record.validate().unwrap_err().contains("updatePath"));

        let mut stray = base.clone();
        stray.app_logic.as_mut().unwrap().path = "/api/v1/app-logic/{id}".into();
        assert!(stray.validate().unwrap_err().contains("path"));

        let mut blank = base.clone();
        blank.app_logic.as_mut().unwrap().fields.record = "  ".into();
        assert!(blank.validate().unwrap_err().contains("fields.record"));

        let mut duplicated = base.clone();
        let first = duplicated.app_logic.as_ref().unwrap().effects[0].clone();
        duplicated.app_logic.as_mut().unwrap().effects.push(first);
        assert!(duplicated.validate().unwrap_err().contains("twice"));
    }

    #[test]
    fn a_descriptor_saved_by_a_windows_editor_still_loads() {
        // Found the hard way: PowerShell's `-Encoding utf8` and Notepad both
        // write a BOM, serde refuses it at column 1, and the file is skipped
        // with nothing but a log line. The panel tells people to drop a file in
        // this folder, so the most likely way to write one must work.
        let d = dir("bom");
        let mut custom: ConnectorDescriptor = serde_json::from_str(BUILTIN[0]).unwrap();
        custom.name = "BOM'd FormLogic".into();
        std::fs::write(
            d.join("connectors").join("formlogic.json"),
            format!("\u{feff}{}", serde_json::to_string(&custom).unwrap()),
        )
        .unwrap();

        let all = load_all(&d);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "BOM'd FormLogic", "the BOM must not hide the file");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_new_provider_is_a_file_not_a_code_change() {
        // The whole point of the descriptor. If this test needs code to change,
        // the design has failed.
        let d = dir("newprovider");
        let other = serde_json::json!({
            "id": "acme",
            "name": "Acme Cloud",
            "auth": {
                "kind": "oauth2_pkce",
                "clientId": "acme-desktop",
                "authorizePath": "/authorize",
                "tokenPath": "/oauth/token",
                "scopes": ["data:read"],
                "tokenResponse": { "credentialFields": ["access_token"] }
            }
        });
        std::fs::write(
            d.join("connectors").join("acme.json"),
            serde_json::to_string(&other).unwrap(),
        )
        .unwrap();

        let all = load_all(&d);
        assert_eq!(all.len(), 2);
        let acme = find(&d, "acme").expect("the new provider is available");
        assert_eq!(acme.name, "Acme Cloud");
        let AuthSpec::Oauth2Pkce(o) = &acme.auth;
        assert_eq!(o.callback_path, "/callback", "defaulted, not required");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_malformed_user_file_does_not_remove_the_working_providers() {
        // One bad file must cost only itself.
        let d = dir("malformed");
        std::fs::write(d.join("connectors").join("broken.json"), "{ not json").unwrap();
        std::fs::write(
            d.join("connectors").join("nocreds.json"),
            serde_json::json!({
                "id": "nocreds", "name": "No Creds",
                "auth": {
                    "kind": "oauth2_pkce", "clientId": "x",
                    "authorizePath": "/a", "tokenPath": "/t", "scopes": ["s"],
                    "tokenResponse": { "credentialFields": [] }
                }
            })
            .to_string(),
        )
        .unwrap();

        let all = load_all(&d);
        assert_eq!(all.len(), 1, "the built-in must survive its neighbours");
        assert_eq!(all[0].id, "formlogic");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn paths_must_be_paths_so_a_descriptor_cannot_redirect_the_ceremony() {
        // An absolute URL in authorizePath would send the user — and the
        // authorization code — to a host the base URL never named.
        let mut d: ConnectorDescriptor = serde_json::from_str(BUILTIN[0]).unwrap();
        let AuthSpec::Oauth2Pkce(o) = &mut d.auth;
        o.authorize_path = "https://evil.example/authorize".into();
        let err = d.validate().unwrap_err();
        assert!(err.contains("authorizePath"), "{err}");
    }
}
