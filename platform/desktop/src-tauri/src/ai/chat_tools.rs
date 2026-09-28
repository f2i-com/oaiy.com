//! Letting a tunneled chat turn actually DO things in the linked account.
//!
//! A grant-less turn never reaches this module — it takes the plain path
//! unchanged. A turn whose sealed body carries a `toolGrant` gets a bounded
//! loop: offer the account's tool catalogue to the model, run what it asks for
//! as the granting user, feed the result back, and stop.
//!
//! Two transports, because not every model can be handed tool schemas:
//!
//! - **Native** — OpenAI-dialect providers take a `tools` array and answer with
//!   `tool_calls`, so the wire carries everything.
//! - **Prompted** — the managed ChatGPT account is prompt-only and has no place
//!   to put a schema. The catalogue is taught in a preamble instead, with one
//!   pinned convention: to call a tool the whole reply must be a single fenced
//!   ` ```tool_call ` block. The parse rule is well-formed-or-text — only a
//!   parseable block naming a catalogued tool counts as a call, and anything
//!   else is the answer. A guessed call would run something the model never
//!   asked for.
//!
//! The loop is bounded and TERMINATES BY CONSTRUCTION: the last allowed round
//! is offered no tools at all, so it can only answer. Without that, a model
//! that keeps calling tools would run until the request's TTL and the user
//! would see it expire.
//!
//! Confirm mode pauses before each call: a sealed proposal goes out, and the
//! turn waits for a sealed approval to come back the other way. Silence is a
//! denial, not an approval, and a denial is a failed tool result fed honestly
//! back to the model — never a failed turn.

use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::tunnel::Lane;

/// The loop is bounded at this many provider rounds.
pub const MAX_TOOL_ROUNDS: usize = 6;
/// Tool calls honoured within ONE reply. Extras get an honest refusal rather
/// than being dropped silently.
const MAX_CALLS_PER_ROUND: usize = 8;
/// A result larger than this is replaced by a note in the sealed frame — the
/// envelope has a hard plaintext cap and a huge result would fail the seal.
const MAX_RESULT_FRAME_BYTES: usize = 192 * 1024;
/// Bound on the result text fed back to the model.
const MAX_RESULT_MODEL_BYTES: usize = 32 * 1024;
/// How long a proposal waits for its approval before auto-denying.
const CONFIRM_DEADLINE: Duration = Duration::from_secs(110);
/// Cadence of the approval poll while paused.
const CONFIRM_POLL: Duration = Duration::from_millis(1500);

/// Grant refusals that are terminal for this turn: the provider will refuse
/// every later call identically, so retrying would be noise. After one, the
/// remaining rounds are offered no tools.
const TERMINAL_GRANT_CODES: [&str; 3] =
    ["grant_expired", "grant_invalid", "grant_instance_mismatch"];

/// The fence the preamble teaches and the parser accepts. One string, so the
/// two can never drift.
const FENCE: &str = "```tool_call";

/// The instruction that replaces the preamble once tools are gone.
///
/// It has to say what TO do, not only what not to. An earlier version said
/// only that tool calling was unavailable, and the model dutifully answered
/// "Tool calling is unavailable for this reply." — restating the instruction
/// instead of using the results it had already been given.
const PLAIN_ANSWER: &str = "This reply must be your final answer to the user. Use the tool \
results already in this conversation to answer them directly and completely, in plain text. Do \
not emit a tool_call block, and do not mention tools or their availability — just answer.";

/// One catalogued tool, in the shape the provider publishes it.
#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Fetch the account's tool catalogue.
///
/// An empty or unreachable catalogue is not an error: the caller falls back to
/// the plain chat path, and the model simply has no tools. Refusing the turn
/// instead would take chat away over a feature the user may not be using.
pub async fn catalog(
    http: &reqwest::Client,
    base_url: &str,
    credential: &str,
    path: &str,
) -> Vec<Tool> {
    let url = crate::link::oauth::join(base_url, path);
    let Ok(resp) = http.get(&url).bearer_auth(credential).send().await else {
        return Vec::new();
    };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let Ok(body) = resp.json::<Value>().await else {
        return Vec::new();
    };
    let rows = body
        .get("tools")
        .and_then(Value::as_array)
        .or_else(|| body.pointer("/data/tools").and_then(Value::as_array));
    let Some(rows) = rows else {
        return Vec::new();
    };
    rows.iter()
        .filter_map(|t| {
            let name = t.get("name").and_then(Value::as_str)?;
            Some(Tool {
                name: name.to_string(),
                description: t
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                input_schema: t
                    .get("inputSchema")
                    .or_else(|| t.get("parameters"))
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object" })),
            })
        })
        .collect()
}

/// The `tools` array an OpenAI-dialect provider takes.
pub fn as_openai_tools(tools: &[Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            })
        })
        .collect()
}

/// The preamble that teaches a prompt-only model the catalogue and the one
/// reply convention the parser accepts.
pub fn preamble(tools: &[Tool]) -> String {
    let mut s = String::from("You can use tools from the linked account in this conversation.\n\nAvailable tools:\n");
    for t in tools {
        s.push_str(&format!(
            "- {}: {}\n  input schema: {}\n",
            t.name, t.description, t.input_schema
        ));
    }
    s.push_str(concat!(
        "\nTo call a tool, reply with ONLY one fenced block and nothing else — ",
        "no prose before or after it:\n\n",
        "```tool_call\n",
        "{\"tool\":\"<name>\",\"input\":{...}}\n",
        "```\n\n",
        "One tool call per reply. Each result arrives as a message starting with ",
        "\"tool_result\"; then call another tool or answer the user. When no tool ",
        "is needed, answer the user in plain text.",
    ));
    s
}

/// A tool call the model asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// Read a prompt-only reply: either ONE well-formed fenced call, or the text.
///
/// Deliberately strict. Prose around the fence, malformed JSON, or a tool that
/// is not in the catalogue all mean "this is the answer" — never a guessed
/// call. Guessing here would run something in the user's account that the model
/// never actually asked for.
pub fn parse_prompted(reply: &str, tools: &[Tool]) -> Option<ToolCall> {
    let trimmed = reply.trim();
    let rest = trimmed.strip_prefix(FENCE)?;
    let body = rest.strip_suffix("```")?;
    let parsed: Value = serde_json::from_str(body.trim()).ok()?;
    let name = parsed.get("tool").and_then(Value::as_str)?;
    if !tools.iter().any(|t| t.name == name) {
        return None;
    }
    Some(ToolCall {
        // Prompt-only replies carry no call id, so one is derived from the
        // name. It only has to be stable within the turn for the browser to
        // match a result frame to its proposal.
        id: format!("prompted-{name}"),
        name: name.to_string(),
        input: parsed
            .get("input")
            .cloned()
            .unwrap_or_else(|| json!({})),
    })
}

/// Read an OpenAI-dialect reply's `tool_calls`.
pub fn parse_native(message: &Value) -> Vec<ToolCall> {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|c| {
                    let name = c.pointer("/function/name").and_then(Value::as_str)?;
                    // Arguments arrive as a JSON STRING, not an object. Passing
                    // the string through would make every call fail validation
                    // with a message about the wrong type.
                    let input = c
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .and_then(|a| serde_json::from_str::<Value>(a).ok())
                        .unwrap_or_else(|| json!({}));
                    Some(ToolCall {
                        id: c
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or(name)
                            .to_string(),
                        name: name.to_string(),
                        input,
                    })
                })
                .take(MAX_CALLS_PER_ROUND)
                .collect()
        })
        .unwrap_or_default()
}

/// What running one call produced.
pub enum Executed {
    Ok(Value),
    /// Failed, and whether tool use is now dead for the whole turn.
    Failed { error: String, terminal: bool },
}

/// Run one catalogued tool as the granting user.
pub async fn execute(
    http: &reqwest::Client,
    base_url: &str,
    credential: &str,
    path: &str,
    grant: &str,
    call: &ToolCall,
) -> Executed {
    let url = crate::link::oauth::join(base_url, path);
    let body = json!({ "grantToken": grant, "tool": call.name, "input": call.input });
    let resp = match http.post(&url).bearer_auth(credential).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            return Executed::Failed {
                error: format!("could not reach the tool service: {e}"),
                terminal: false,
            }
        }
    };
    let status = resp.status();
    let value: Value = resp.json().await.unwrap_or(Value::Null);
    if status.is_success() {
        // Unwrap the envelope, tolerating a bare body.
        return Executed::Ok(value.get("data").cloned().unwrap_or(value));
    }
    let code = value
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| value.pointer("/error/code").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string();
    let message = value
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("the tool service refused the call")
        .to_string();
    Executed::Failed {
        terminal: TERMINAL_GRANT_CODES.contains(&code.as_str()),
        error: if code.is_empty() {
            message
        } else {
            format!("{code}: {message}")
        },
    }
}

/// The sealed frame announcing a call is running.
pub fn call_frame(call: &ToolCall) -> Value {
    json!({
        "v": 1,
        "type": "tool_call",
        "kind": "tool_call",
        "id": call.id,
        "name": call.name,
        "status": "running",
    })
}

/// The sealed frame reporting what a call produced.
///
/// A result too large for one envelope is replaced by a note rather than
/// failing the seal: the work already happened, and killing the turn over the
/// size of its receipt would hide that.
pub fn result_frame(call: &ToolCall, outcome: &Executed) -> Value {
    match outcome {
        Executed::Ok(value) => {
            let rendered = value.to_string();
            let result = if rendered.len() > MAX_RESULT_FRAME_BYTES {
                json!({ "truncated": true, "note": "the tool result was too large to relay" })
            } else {
                value.clone()
            };
            json!({
                "v": 1,
                "type": "tool_result",
                "kind": "tool_result",
                "id": call.id,
                "name": call.name,
                "status": "done",
                "result": result,
            })
        }
        Executed::Failed { error, .. } => json!({
            "v": 1,
            "type": "tool_result",
            "kind": "tool_result",
            "id": call.id,
            "name": call.name,
            "status": "failed",
            "error": error,
        }),
    }
}

/// The sealed frame asking the user to approve a call.
pub fn proposal_frame(call: &ToolCall, request_id: &str) -> Value {
    json!({
        "v": 1,
        "type": "tool_proposal",
        "kind": "tool_proposal",
        "callId": call.id,
        "requestId": request_id,
        "tool": call.name,
        "input": call.input,
    })
}

/// What a tool result looks like fed back to the model.
pub fn result_for_model(call: &ToolCall, outcome: &Executed) -> String {
    let mut rendered = match outcome {
        Executed::Ok(value) => value.to_string(),
        Executed::Failed { error, .. } => json!({ "error": error }).to_string(),
    };
    if rendered.len() > MAX_RESULT_MODEL_BYTES {
        // On a char boundary: slicing mid-codepoint would panic.
        let mut cut = MAX_RESULT_MODEL_BYTES;
        while cut > 0 && !rendered.is_char_boundary(cut) {
            cut -= 1;
        }
        rendered.truncate(cut);
        rendered.push_str(" … [truncated]");
    }
    format!("tool_result {}: {rendered}", call.name)
}

/// The outcome of waiting for an approval.
#[derive(Debug, PartialEq)]
pub enum Approval {
    Approved,
    Denied(&'static str),
}

/// Wait for the user's answer to a proposal on the sealed inbound channel.
///
/// Silence is a DENIAL. A turn that treated a timeout as approval would run
/// something in the user's account precisely when they were not there to say
/// no — which is the one case confirm mode exists for.
pub async fn wait_for_approval(
    lane: &Lane<'_>,
    sessions: &crate::ai::e2e::E2eSessions,
    identity: &crate::ai::e2e::E2eIdentity,
    eph_pub: &str,
    call_id: &str,
    cursor: &mut u64,
) -> Approval {
    let deadline = Instant::now() + CONFIRM_DEADLINE;
    while Instant::now() < deadline {
        tokio::time::sleep(CONFIRM_POLL).await;
        let Ok(body) = lane.fetch_input(*cursor).await else {
            continue;
        };
        // The request leaving the claimed window means nobody is going to
        // answer — waiting out the deadline would just delay the denial.
        if let Some(status) = body.get("status").and_then(Value::as_str) {
            if matches!(status, "done" | "failed" | "expired") {
                return Approval::Denied("the approval channel closed");
            }
        }
        for frame in body.get("frames").and_then(Value::as_array).into_iter().flatten() {
            // The cursor advances even for a frame that cannot be opened, so
            // one bad frame can never wedge the poll.
            if let Some(seq) = frame.get("seq").and_then(Value::as_u64) {
                if seq > *cursor {
                    *cursor = seq;
                }
            }
            let Some(envelope) = frame.get("envelope").and_then(Value::as_str) else {
                continue;
            };
            let Ok(plain) = sessions.open_inbound(identity, lane.id(), eph_pub, envelope) else {
                continue;
            };
            let Ok(value) = serde_json::from_slice::<Value>(&plain) else {
                continue;
            };
            let is_approval = value.get("type").and_then(Value::as_str) == Some("tool_approval")
                || value.get("kind").and_then(Value::as_str) == Some("tool_approval");
            if !is_approval {
                continue;
            }
            if value.get("callId").and_then(Value::as_str) != Some(call_id) {
                continue;
            }
            return if value.get("approved").and_then(Value::as_bool) == Some(true) {
                Approval::Approved
            } else {
                Approval::Denied("denied by you")
            };
        }
    }
    Approval::Denied("the approval timed out")
}

/// Whether this round may still offer tools.
///
/// The LAST allowed round never does, which is what makes the loop terminate:
/// a model that only ever calls tools still has to answer at the end.
pub fn tools_available(round: usize, grant_dead: bool) -> bool {
    !grant_dead && round + 1 < MAX_TOOL_ROUNDS
}

/// The instruction a prompt-only round carries when tools are gone.
pub fn plain_answer_instruction() -> &'static str {
    PLAIN_ANSWER
}

/// Render a message list as plain conversation text — what a prompt-only model
/// receives in place of a structured history.
pub fn render_conversation(messages: &[Value]) -> String {
    messages
        .iter()
        .map(|m| {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            let text = message_text(m.get("content").unwrap_or(&Value::Null));
            format!("{role}: {text}")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The text of a message's content, whether a plain string or OpenAI parts.
pub fn message_text(content: &Value) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    let Some(parts) = content.as_array() else {
        return String::new();
    };
    parts
        .iter()
        .filter_map(|p| match p.get("type").and_then(Value::as_str) {
            Some("text") => p.get("text").and_then(Value::as_str).map(str::to_string),
            // Kept as a marker rather than dropped, so the transcript stays
            // honest about where an image sat in the conversation.
            Some("image_url") => Some("[image attached]".to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "create_form".into(),
                description: "Create a form".into(),
                input_schema: json!({ "type": "object", "properties": { "title": { "type": "string" } } }),
            },
            Tool {
                name: "list_apps".into(),
                description: "List apps".into(),
                input_schema: json!({ "type": "object" }),
            },
        ]
    }

    #[test]
    fn a_prompted_reply_is_a_call_only_when_it_is_exactly_the_taught_shape() {
        let call = parse_prompted(
            "```tool_call\n{\"tool\":\"create_form\",\"input\":{\"title\":\"Contact\"}}\n```",
            &tools(),
        )
        .expect("the taught shape must parse");
        assert_eq!(call.name, "create_form");
        assert_eq!(call.input["title"], "Contact");

        // Everything else is the ANSWER, never a guessed call — guessing would
        // run something in the user's account the model never asked for.
        for not_a_call in [
            "Sure! ```tool_call\n{\"tool\":\"list_apps\"}\n```",       // prose before
            "```tool_call\n{\"tool\":\"list_apps\"}\n``` anything",     // prose after
            "```tool_call\n{not json}\n```",                            // malformed
            "```tool_call\n{\"tool\":\"rm_rf\",\"input\":{}}\n```",     // not catalogued
            "```json\n{\"tool\":\"list_apps\"}\n```",                   // wrong fence
            "I'll create a form for you.",                              // a plain answer
            "",
        ] {
            assert!(parse_prompted(not_a_call, &tools()).is_none(), "{not_a_call:?}");
        }
    }

    #[test]
    fn the_preamble_teaches_exactly_the_fence_the_parser_accepts() {
        // If these drift the model emits a block nothing recognises, and every
        // tool call silently becomes the assistant's answer instead.
        let text = preamble(&tools());
        assert!(text.contains(FENCE), "{text}");
        assert!(text.contains("create_form: Create a form"));
        assert!(text.contains("list_apps"));
        // …and a block written to the taught shape round-trips through it.
        let call = parse_prompted(
            "```tool_call\n{\"tool\":\"list_apps\",\"input\":{}}\n```",
            &tools(),
        );
        assert_eq!(call.map(|c| c.name).as_deref(), Some("list_apps"));
    }

    #[test]
    fn native_tool_call_arguments_are_parsed_out_of_their_json_string() {
        // They arrive as a STRING. Passing it through as-is makes every call
        // fail validation with a complaint about the wrong type.
        let message = json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": { "name": "create_form", "arguments": "{\"title\":\"Contact\"}" },
            }],
        });
        let calls = parse_native(&message);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].input["title"], "Contact");

        // A reply with no tool_calls is an answer, not an empty call list bug.
        assert!(parse_native(&json!({ "content": "hi" })).is_empty());
        // Unparseable arguments become an empty object rather than failing the
        // whole round — the tool's own validation gives a better message.
        let bad = json!({ "tool_calls": [{ "id": "c", "function": { "name": "x", "arguments": "{" } }] });
        assert_eq!(parse_native(&bad)[0].input, json!({}));
    }

    #[test]
    fn the_final_round_asks_for_an_answer_rather_than_announcing_a_restriction() {
        // Observed live: an instruction that only said tools were unavailable
        // got back "Tool calling is unavailable for this reply." — the model
        // restated it instead of using results it already had.
        let instruction = plain_answer_instruction();
        assert!(instruction.contains("final answer"));
        assert!(instruction.contains("tool results already in this conversation"));
        assert!(instruction.contains("just answer"));
        // …and it still has to forbid the fence, or the round never terminates.
        assert!(instruction.contains("emit a tool_call block"));
    }

    #[test]
    fn the_loop_always_reaches_a_round_that_cannot_call_a_tool() {
        // What makes it terminate. Without it a model that only ever calls
        // tools runs until the request's TTL and the user is told it expired.
        assert!(tools_available(0, false));
        assert!(!tools_available(MAX_TOOL_ROUNDS - 1, false), "the last round must answer");
        // …and a dead grant ends tool use immediately rather than retrying a
        // refusal the provider will repeat identically.
        assert!(!tools_available(0, true));
    }

    #[test]
    fn a_terminal_grant_refusal_is_told_apart_from_an_ordinary_tool_error() {
        for code in TERMINAL_GRANT_CODES {
            assert!(TERMINAL_GRANT_CODES.contains(&code));
        }
        assert!(!TERMINAL_GRANT_CODES.contains(&"unknown_tool"));
        assert!(!TERMINAL_GRANT_CODES.contains(&"validation_failed"));
    }

    #[test]
    fn an_oversized_result_is_noted_rather_than_failing_the_seal() {
        let call = ToolCall { id: "c1".into(), name: "list_apps".into(), input: json!({}) };
        let huge = Executed::Ok(json!({ "rows": "x".repeat(MAX_RESULT_FRAME_BYTES + 10) }));
        let frame = result_frame(&call, &huge);
        assert_eq!(frame["status"], "done");
        assert_eq!(frame["result"]["truncated"], true);

        // The model's copy is clipped on a char boundary, not sliced blindly.
        let wide = Executed::Ok(json!({ "t": "é".repeat(MAX_RESULT_MODEL_BYTES) }));
        let fed = result_for_model(&call, &wide);
        assert!(fed.starts_with("tool_result list_apps: "));
        assert!(fed.ends_with(" … [truncated]"));
    }

    #[test]
    fn a_failure_is_reported_as_a_failed_result_not_a_silent_gap() {
        let call = ToolCall { id: "c2".into(), name: "create_form".into(), input: json!({}) };
        let failed = Executed::Failed { error: "unknown_tool: no such tool".into(), terminal: false };
        let frame = result_frame(&call, &failed);
        assert_eq!(frame["status"], "failed");
        assert_eq!(frame["error"], "unknown_tool: no such tool");
        // …and the model is told, so it can retry or answer without the tool.
        assert!(result_for_model(&call, &failed).contains("no such tool"));
    }

    #[test]
    fn a_conversation_renders_with_roles_and_image_markers() {
        let messages = vec![
            json!({ "role": "user", "content": "hello" }),
            json!({ "role": "assistant", "content": [
                { "type": "text", "text": "hi" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,AA" } },
            ]}),
        ];
        let text = render_conversation(&messages);
        assert!(text.contains("user: hello"));
        assert!(text.contains("assistant: hi"));
        // The marker keeps the transcript honest about where the image was.
        assert!(text.contains("[image attached]"));
        // …and never leaks the data URI into the prompt.
        assert!(!text.contains("base64"));
    }

    #[test]
    fn the_proposal_and_activity_frames_carry_what_the_reader_matches_on() {
        let call = ToolCall { id: "c3".into(), name: "create_form".into(), input: json!({ "title": "T" }) };
        let proposal = proposal_frame(&call, "req-1");
        assert_eq!(proposal["callId"], "c3");
        assert_eq!(proposal["tool"], "create_form");
        assert_eq!(proposal["requestId"], "req-1");
        // Both spellings on every frame: the reader tries `type` then `kind`.
        for frame in [call_frame(&call), proposal] {
            assert!(frame["type"].is_string() && frame["kind"].is_string());
            assert_eq!(frame["v"], 1);
        }
    }
}
