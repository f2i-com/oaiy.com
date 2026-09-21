//! Chat-template helpers for instruction-tuned models.
//!
//! Each supported architecture has its own role-marker convention. This module
//! formats `[ChatMessage]` into the raw string that the model's tokenizer can
//! then encode. We hard-code the standard format per arch family rather than
//! parse the GGUF's Jinja `tokenizer.chat_template` — the per-arch formats are
//! stable enough that hard-coding gives us reliable results without dragging
//! in a Jinja interpreter.
//!
//! For unrecognized architectures, falls back to a simple `role: content`
//! concatenation that won't match any specific instruct format but is at least
//! deterministic.

use crate::Architecture;

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role:    Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(&self) -> &'static str {
        match self {
            Self::System    => "system",
            Self::User      => "user",
            Self::Assistant => "assistant",
        }
    }
}

impl ChatMessage {
    pub fn system(content: impl Into<String>)    -> Self { Self { role: Role::System,    content: content.into() } }
    pub fn user(content: impl Into<String>)      -> Self { Self { role: Role::User,      content: content.into() } }
    pub fn assistant(content: impl Into<String>) -> Self { Self { role: Role::Assistant, content: content.into() } }
}

/// Format `messages` into the raw prompt string for `arch`. When `add_assistant`
/// is true (the typical case for inference), appends an open assistant turn so
/// the model continues from there.
///
/// The returned string is meant to be passed straight to the model's tokenizer.
/// BOS handling: each per-arch format includes the BOS marker as a string the
/// tokenizer will recognize. Pass `add_bos=false` when calling the encoder
/// (or strip the leading BOS yourself) to avoid double-BOS.
pub fn apply_chat_template(arch: &Architecture, messages: &[ChatMessage], add_assistant: bool) -> String {
    match arch {
        // Gemma 3 and 3n share the `<start_of_turn>` / `<end_of_turn>` format.
        Architecture::Gemma3 | Architecture::Gemma3n => gemma3_template(messages, add_assistant),
        // Gemma 4 introduced new `<|turn>` / `<turn|>` markers.
        Architecture::Gemma4 => gemma4_template(messages, add_assistant),
        Architecture::Qwen2 => chatml_template(messages, add_assistant, false),
        // VENDORED-LOCAL: GLM-5.3-Flash.
        Architecture::Glm5Next => glm5next_template(messages, add_assistant),
        // Qwen3 ships with "thinking mode" enabled by default. To get a normal
        // (non-thinking) chat response, the official template appends
        // `<think>\n\n</think>\n\n` after the assistant turn opener — that
        // satisfies the model's expectation that thinking has already happened.
        Architecture::Qwen3 => chatml_template(messages, add_assistant, true),
        // Qwen3.5 reuses ChatML (`<|im_start|>...<|im_end|>`) but introduced
        // a thinking-channel wrapper that we don't emit by default, so the
        // simple no-thinking ChatML form works as the default chat template.
        Architecture::Qwen35 | Architecture::Qwen3Moe | Architecture::Qwen3VlMoe |
        Architecture::Qwen35Moe | Architecture::Qwen36MoeVl =>
            chatml_template(messages, add_assistant, false),
        Architecture::Llama => llama3_template(messages, add_assistant),
        Architecture::Mistral => mistral_template(messages, add_assistant),
        Architecture::Unsupported(_) => fallback_template(messages, add_assistant),
    }
}

/// Gemma 3 / Gemma 2 turn-based format. Notes:
///   * Gemma uses "model" instead of "assistant" for the model role.
///   * No system role; system messages get prepended to the first user turn.
///   * `<bos>` at the start (the tokenizer's BOS will tokenize this).
fn gemma3_template(messages: &[ChatMessage], add_assistant: bool) -> String {
    let mut out = String::with_capacity(messages.iter().map(|m| m.content.len()).sum::<usize>() + 64);
    out.push_str("<bos>");

    // Coalesce a leading system message into the first user message, since
    // Gemma's chat template doesn't have a separate system turn.
    let mut pending_system: Option<&str> = None;
    for m in messages {
        match m.role {
            Role::System => pending_system = Some(&m.content),
            Role::User => {
                out.push_str("<start_of_turn>user\n");
                if let Some(sys) = pending_system.take() {
                    out.push_str(sys);
                    out.push_str("\n\n");
                }
                out.push_str(&m.content);
                out.push_str("<end_of_turn>\n");
            }
            Role::Assistant => {
                out.push_str("<start_of_turn>model\n");
                out.push_str(&m.content);
                out.push_str("<end_of_turn>\n");
            }
        }
    }
    if add_assistant {
        out.push_str("<start_of_turn>model\n");
    }
    out
}

/// Gemma 4 chat format. New marker scheme vs Gemma 3:
///   * `<|turn>{role}\n{content}<turn|>\n` per message.
///   * Role for assistant is `model`. System messages get their own turn (not
///     coalesced into the user turn like Gemma 3 / 3n).
///   * `<bos>` at the start.
fn gemma4_template(messages: &[ChatMessage], add_assistant: bool) -> String {
    let mut out = String::with_capacity(messages.iter().map(|m| m.content.len()).sum::<usize>() + 128);
    out.push_str("<bos>");
    for m in messages {
        let role = match m.role {
            Role::System    => "system",
            Role::User      => "user",
            Role::Assistant => "model",
        };
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<turn|>\n");
    }
    if add_assistant {
        out.push_str("<|turn>model\n");
    }
    out
}

// VENDORED-LOCAL: GLM-5.3-Flash.
/// GLM-5.3-Flash (`glm5next`). `[gMASK]<sop>` opens the sequence, then each turn
/// is a role marker followed **directly** by its content — no separator.
///
/// Transcribed from the GGUF `tokenizer.chat_template`, after a generation run
/// showed the model emitting a bare `</think>`. Three things that template does
/// which are easy to get wrong:
///
///   * **The generation prompt pre-fills the thinking block.** It ends
///     `<|assistant|>{{- '<think>' -}}`, so the assistant turn opens *inside*
///     `<think>`. Omitting it makes a reasoning model close a block nobody
///     opened, which is exactly the stray `</think>` that found this.
///   * **No newline after a role marker.** The template is
///     `<|user|>{{ visible_text(m.content) }}`, not `<|user|>''...`.
///   * **A reasoning-effort preamble.** `effective_reasoning_effort` defaults to
///     `max` when the caller does not set it, and is emitted as
///     `<|system|>Reasoning Effort: Max`.
///
/// Token ids, from this model own vocab: `[gMASK]` 154822, `<sop>` 154824,
/// `<|system|>` 154826, `<|user|>` 154827, `<|assistant|>` 154828,
/// `<|observation|>` 154829, `<think>` 154841, `</think>` 154842.
///
/// **Not yet emitted**: the `# Tools` system block with `<tools>` signatures, and
/// a caller-selectable reasoning effort (this always requests the default,
/// `Max`). A prior assistant turn is given an empty `<think></think>` pair, which
/// is what the template does when that turn carries no `reasoning_content`.
// VENDORED-LOCAL: GLM-5.3-Flash.
/// How much the model is asked to think before answering.
///
/// The reference Jinja reads `reasoning_effort`, accepts `'low'` and `'high'`, and
/// falls back to `'max'` for anything else including the unset case -- so `Max` is
/// the default the shipped template gives, and it is the most verbose one. A short
/// factual question answered at Max spends hundreds of tokens deliberating before it
/// closes `</think>`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    High,
    #[default]
    Max,
}

impl ReasoningEffort {
    /// The word the template capitalises into the system line.
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::High => "High",
            Self::Max => "Max",
        }
    }

    /// Parse `low` / `high` / `max`, case-insensitively. Anything else is `Max`,
    /// which is what the Jinja does with an unrecognised value.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Self::Low,
            "high" => Self::High,
            _ => Self::Max,
        }
    }
}

fn glm5next_template(messages: &[ChatMessage], add_assistant: bool) -> String {
    glm5next_template_with(messages, add_assistant, ReasoningEffort::default())
}

/// [`glm5next_template`] at a chosen reasoning effort.
pub fn glm5next_template_with(
    messages: &[ChatMessage],
    add_assistant: bool,
    effort: ReasoningEffort,
) -> String {
    glm5next_template_full(messages, add_assistant, effort, true)
}

// VENDORED-LOCAL: GLM-5.3-Flash.
/// [`glm5next_template_with`], and `thinking` chooses which generation prompt.
///
/// The reference Jinja always opens `<think>` for `add_generation_prompt`, so
/// `thinking = true` is what the shipped template does and what the golden tests
/// check. But the same Jinja writes `<think></think>` -- opened and immediately
/// closed -- for an assistant turn that did not reason, so the closed form is a
/// shape this model was trained on. Prefilling it asks for a direct answer instead
/// of a reasoning block, which is how GLM deployments usually expose a
/// "non-thinking" mode.
///
/// Measured, not assumed: `print_both_generation_prompts` in
/// `glm5next::device`'s tests runs one prompt through both forms. They agree --
/// short prompts answer correctly under either, long ones under either. So the
/// closed form is a deviation from the reference (which emits only `<think>` for
/// `add_generation_prompt`, `glm5next.cpp` line 255) but a sound one, and keeping
/// it is a latency decision: at ~13 tok/s a forced reasoning block costs seconds
/// on every coder-cli turn for a question that does not need one.
///
/// That 2x2 is worth keeping for a second reason. This pair was the prime suspect
/// for a day, because `nrob-server` defaults to `thinking = false` and so every
/// request a harness made took the off-reference branch -- which made the closed
/// form correlate perfectly with output that degenerated. It was not the cause;
/// the KDA decay axis was (see `glm5next::kda`). A correlation with the one thing
/// that differs from the reference is not evidence that it is the fault.
///
/// Worth knowing rather than guessing at: at `ReasoningEffort::Max` a one-line
/// factual question spends hundreds of tokens deliberating first.
pub fn glm5next_template_full(
    messages: &[ChatMessage],
    add_assistant: bool,
    effort: ReasoningEffort,
    thinking: bool,
) -> String {
    let mut out = String::with_capacity(
        messages.iter().map(|m| m.content.len()).sum::<usize>() + 128,
    );
    out.push_str("[gMASK]<sop>");
    // The template emits this whenever an effort is set, and an unset or unknown
    // value means max.
    out.push_str("<|system|>Reasoning Effort: ");
    out.push_str(effort.as_str());
    for m in messages {
        match m.role {
            Role::System => out.push_str("<|system|>"),
            Role::User => out.push_str("<|user|>"),
            Role::Assistant => {
                out.push_str("<|assistant|>");
                // A history turn with no recorded reasoning gets an empty pair.
                out.push_str("<think></think>");
            }
        }
        out.push_str(&m.content);
    }
    if add_assistant {
        // The opener the template pre-fills, so the model continues inside the
        // thinking block instead of closing one that was never opened.
        //
        // The closed form is the same shape the Jinja writes for an assistant turn
        // that did not reason, so it is trained; pre-filling it asks for a direct
        // answer. See `glm5next_template_full`.
        out.push_str(if thinking {
            "<|assistant|><think>"
        } else {
            "<|assistant|><think></think>"
        });
    }
    out
}

/// ChatML format used by Qwen 2 / Qwen 3 / many other modern instruct models.
///   * System messages get their own turn.
///   * `<|im_start|>{role}\n{content}<|im_end|>\n` per message.
///
/// `skip_thinking`: appends `<think>\n\n</think>\n\n` after the assistant turn
/// opener — required for Qwen3 (which has thinking mode on by default) to
/// produce a normal direct response.
fn chatml_template(messages: &[ChatMessage], add_assistant: bool, skip_thinking: bool) -> String {
    let mut out = String::with_capacity(messages.iter().map(|m| m.content.len()).sum::<usize>() + 64);
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(m.role.as_str());
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    if add_assistant {
        out.push_str("<|im_start|>assistant\n");
        if skip_thinking {
            out.push_str("<think>\n\n</think>\n\n");
        }
    }
    out
}

/// Llama 3 instruct format with header_id markers.
///   * `<|begin_of_text|>` at the start.
///   * `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>` per message.
fn llama3_template(messages: &[ChatMessage], add_assistant: bool) -> String {
    let mut out = String::with_capacity(messages.iter().map(|m| m.content.len()).sum::<usize>() + 128);
    out.push_str("<|begin_of_text|>");
    for m in messages {
        out.push_str("<|start_header_id|>");
        out.push_str(m.role.as_str());
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(&m.content);
        out.push_str("<|eot_id|>");
    }
    if add_assistant {
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    out
}

/// Mistral / Llama 2 `[INST] ... [/INST]` format.
///   * No system role; system messages get prepended to the first user turn.
///   * `<s>[INST] {user} [/INST] {assistant}</s>` per turn pair.
fn mistral_template(messages: &[ChatMessage], add_assistant: bool) -> String {
    let mut out = String::with_capacity(messages.iter().map(|m| m.content.len()).sum::<usize>() + 64);
    out.push_str("<s>");

    let mut pending_system: Option<&str> = None;
    let mut i = 0;
    while i < messages.len() {
        let m = &messages[i];
        match m.role {
            Role::System => { pending_system = Some(&m.content); i += 1; }
            Role::User => {
                out.push_str("[INST] ");
                if let Some(sys) = pending_system.take() {
                    out.push_str(sys);
                    out.push_str("\n\n");
                }
                out.push_str(&m.content);
                out.push_str(" [/INST]");
                i += 1;
            }
            Role::Assistant => {
                out.push(' ');
                out.push_str(&m.content);
                out.push_str("</s>");
                if i + 1 < messages.len() && matches!(messages[i + 1].role, Role::User) {
                    out.push_str("<s>");
                }
                i += 1;
            }
        }
    }
    let _ = add_assistant; // open turn is implicit after [/INST] in this format
    out
}

/// Stop tokens that signal "the assistant turn is over" for each arch's chat
/// format. These are in addition to the tokenizer's regular EOS — many chat
/// models emit a turn marker (`<end_of_turn>`, `<|im_end|>`, `<|eot_id|>`)
/// rather than the global EOS to end an assistant response.
///
/// Returns the literal token strings; pair with `Tokenizer::token_id()` to
/// resolve them in the model's vocabulary.
pub fn chat_stop_tokens(arch: &Architecture) -> &'static [&'static str] {
    match arch {
        Architecture::Gemma3 | Architecture::Gemma3n => &["<end_of_turn>"],
        Architecture::Gemma4 => &["<turn|>"],
        Architecture::Qwen2 | Architecture::Qwen3 | Architecture::Qwen35 |
        Architecture::Qwen3Moe | Architecture::Qwen3VlMoe |
        Architecture::Qwen35Moe | Architecture::Qwen36MoeVl => &["<|im_end|>"],
        // VENDORED-LOCAL: GLM-5.3-Flash. Resolved from the GGUF vocab, not guessed:
        // eos 154820 = `<|endoftext|>`, eot 154827 = `<|user|>`, eom 154829 =
        // `<|observation|>`. GLM ends an assistant turn by emitting the *next*
        // role marker, so `<|user|>` is a stop token rather than a prompt-only one.
        Architecture::Glm5Next => &["<|endoftext|>", "<|user|>", "<|observation|>"],
        Architecture::Llama => &["<|eot_id|>", "<|end_of_text|>"],
        Architecture::Mistral => &["</s>"],
        Architecture::Unsupported(_) => &[],
    }
}

/// Generic fallback for unknown architectures. Just dumps role: content per line.
fn fallback_template(messages: &[ChatMessage], add_assistant: bool) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(m.role.as_str());
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    if add_assistant {
        out.push_str("assistant: ");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // VENDORED-LOCAL: GLM-5.3-Flash.
    /// Golden strings for the glm5next chat format.
    ///
    /// **These were rendered from the GGUF own `tokenizer.chat_template`** with
    /// Jinja (`jinja2`, with the `loopcontrols` extension the template needs for
    /// its `break`) and compared byte for byte, so this is a check against the
    /// reference rather than against itself.
    ///
    /// A generation run is what prompted it: the model emitted a bare `</think>`,
    /// which turned out to be three separate errors -- a missing pre-filled
    /// `<think>` opener, a spurious newline after each role marker, and a missing
    /// reasoning-effort preamble.
    #[test]
    fn glm5next_template_matches_the_reference_jinja() {
        let u = |c: &str| ChatMessage { role: Role::User, content: c.into() };
        let a = |c: &str| ChatMessage { role: Role::Assistant, content: c.into() };
        let sy = |c: &str| ChatMessage { role: Role::System, content: c.into() };

        // One user turn, asking for a generation prompt.
        assert_eq!(
            apply_chat_template(&Architecture::Glm5Next, &[u("a")], true),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>a<|assistant|><think>"
        );
        // No generation prompt: no assistant opener at all.
        assert_eq!(
            apply_chat_template(&Architecture::Glm5Next, &[u("a")], false),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>a"
        );
        // A system turn comes after the effort preamble, not instead of it.
        assert_eq!(
            apply_chat_template(&Architecture::Glm5Next, &[sy("S"), u("a")], true),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>S<|user|>a<|assistant|><think>"
        );
        // A history assistant turn carries an empty think pair.
        assert_eq!(
            apply_chat_template(&Architecture::Glm5Next, &[u("a"), a("b"), u("c")], true),
            concat!(
                "[gMASK]<sop><|system|>Reasoning Effort: Max",
                "<|user|>a<|assistant|><think></think>b<|user|>c<|assistant|><think>"
            )
        );
    }

    #[test]
    fn gemma3_simple_user_assistant() {
        let msgs = [ChatMessage::user("Hi"), ChatMessage::assistant("Hello!")];
        let out = apply_chat_template(&Architecture::Gemma3, &msgs, true);
        assert_eq!(
            out,
            "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn gemma3_coalesces_system_into_user() {
        let msgs = [ChatMessage::system("You are helpful."), ChatMessage::user("Hi")];
        let out = apply_chat_template(&Architecture::Gemma3, &msgs, true);
        // System content appears before the user content in the same turn.
        assert!(out.contains("<start_of_turn>user\nYou are helpful.\n\nHi<end_of_turn>"));
    }

    #[test]
    fn chatml_qwen2_basic() {
        let msgs = [ChatMessage::system("S"), ChatMessage::user("U")];
        let out = apply_chat_template(&Architecture::Qwen2, &msgs, true);
        assert_eq!(
            out,
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// Qwen3 needs `<think>\n\n</think>\n\n` after the assistant opener so the
    /// model treats this as a direct response (skip thinking mode).
    #[test]
    fn chatml_qwen3_skips_thinking() {
        let msgs = [ChatMessage::user("U")];
        let out = apply_chat_template(&Architecture::Qwen3, &msgs, true);
        assert!(
            out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "got: {out:?}"
        );
    }

    #[test]
    fn llama3_header_ids() {
        let msgs = [ChatMessage::user("Hello")];
        let out = apply_chat_template(&Architecture::Llama, &msgs, true);
        assert_eq!(
            out,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn mistral_inst_brackets() {
        let msgs = [ChatMessage::user("Hi")];
        let out = apply_chat_template(&Architecture::Mistral, &msgs, true);
        assert_eq!(out, "<s>[INST] Hi [/INST]");
    }

    #[test]
    fn no_assistant_marker_when_disabled() {
        let msgs = [ChatMessage::user("Hi")];
        let out = apply_chat_template(&Architecture::Qwen2, &msgs, false);
        assert!(!out.ends_with("<|im_start|>assistant\n"));
        // Also: the Qwen3 thinking-skip block must not appear when add_assistant=false.
        let out3 = apply_chat_template(&Architecture::Qwen3, &msgs, false);
        assert!(!out3.contains("<think>"));
    }
}
