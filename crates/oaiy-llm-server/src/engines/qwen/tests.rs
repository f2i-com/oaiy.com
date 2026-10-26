//! `qwen`'s tests.

use super::*;
#[test]
fn native_file_call_round_trip_preserves_literal_content() {
    let tools = vec![Json::parse(br#"{"function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}"#).unwrap()];
    let content = "\n<style>\n  :root { --bg: #000; }\n  body::before { background: var(--bg); content: 'λ 😀'; }\n</style>\n<script>const x = a < b && b > 0;</script>\n\n";
    let raw = format!("<tool_call>\n<function=write_file>\n<parameter=path>\nsite/index.html\n</parameter>\n<parameter=content>\n{content}\n</parameter>\n</function>\n</tool_call>");
    let mut stream = NativeStream::default();
    let mut parser = crate::dsv41::chat::StreamParser::new(Mode::Chat);
    for end in raw.char_indices().map(|(at,c)| at+c.len_utf8()) {
        parser.push(&stream.push(&raw[..end], &tools, false).unwrap());
        assert!(!parser.tool_calls_ready());
    }
    parser.push(&stream.push(&raw, &tools, true).unwrap());
    assert!(parser.tool_call_error().is_none());
    let (_, calls) = parser.finish();
    assert_eq!(calls.len(), 1);
    let args = Json::parse(calls[0].arguments.as_bytes()).unwrap();
    assert_eq!(args.get("content").and_then(Json::as_str), Some(content));
    assert_eq!(args.get("path").and_then(Json::as_str), Some("site/index.html"));
}

#[test]
fn native_stream_releases_reasoning_and_prose_before_completion() {
    let mut stream = NativeStream::default();
    let mut parser = crate::dsv41::chat::StreamParser::new(Mode::Thinking);
    let first = stream.push("Let me check", &[], false).unwrap();
    assert!(matches!(&parser.push(&first)[..], [crate::dsv41::chat::Delta::Reasoning(t)] if t == "Let me check"));
    let next = stream.push("Let me check</think>Hello", &[], false).unwrap();
    assert!(parser.push(&next).iter().any(|d| matches!(d,crate::dsv41::chat::Delta::Content(t) if t == "Hello")));
    assert!(stream.push("Let me check</think>Hello", &[], true).unwrap().is_empty());
    let mut utf8 = NativeStream::default();
    assert_eq!(utf8.push("Hi \u{fffd}", &[], false).unwrap(), "Hi ");
    assert_eq!(utf8.push("Hi 😀", &[], false).unwrap(), "😀");
}

#[test]
fn native_stream_never_exposes_partial_or_invalid_tool_calls() {
    let raw = "Checking. <tool_call>\n<function=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n<parameter=keys>\n[]\n</parameter>\n</function>\n</tool_call>";
    let mut stream = NativeStream::default();
    let mut emitted = String::new();
    for end in 1..=raw.len() { emitted.push_str(&stream.push(&raw[..end], &tools(), false).unwrap()); }
    assert_eq!(emitted, "Checking. ");
    let (preview,start) = stream.tool_preview(raw).unwrap().unwrap();
    assert!(start && preview.starts_with("<tool_call>"));
    assert!(stream.tool_preview(raw).unwrap().is_none());
    emitted.push_str(&stream.push(raw, &tools(), true).unwrap());
    assert_eq!(emitted, normalize(raw, &tools()).unwrap());
    for invalid in [raw.replace("function=computer", "function=unknown"), raw.replace("</tool_call>", "")] {
        let mut stream = NativeStream::default();
        assert_eq!(stream.push(&invalid, &tools(), false).unwrap(), "Checking. ");
        assert!(stream.push(&invalid, &tools(), true).is_err());
    }
}

#[test]
fn checkpoints_survive_changed_assistant_suffix_and_leave_logits_token() {
    // 1 is im_start; later reasoning and tool XML need not match.
    let first = [1, 10, 11, 1, 20, 21, 1, 30, 31, 32];
    let next = [1, 10, 11, 1, 20, 21, 1, 30, 40, 41, 1, 50];
    let stops = checkpoint_positions(&first, Some(1));
    assert_eq!(stops, [3, 6, 9]);
    assert_eq!(stops.iter().copied().filter(|&p| next.starts_with(&first[..p])).max(), Some(6));
    assert_eq!(checkpoint_positions(&[1], Some(1)), []);
    assert_eq!(checkpoint_positions(&[7, 8, 9], None), [2]);
}

#[test]
fn the_disk_keeps_a_turns_checkpoints_but_the_one_a_few_tokens_after_the_assistants_header() {
    // a conversation's: before its first user message, before the assistant's header, a token short of the prompt
    assert_eq!(disk_positions(&[120, 3432, 3438], 3439), [120, 3432]);
    // the last one alone, or far from the one before it: kept
    assert_eq!(disk_positions(&[2], 3), [2]);
    assert_eq!(disk_positions(&[120, 3438], 3439), [120, 3438]);
    assert_eq!(disk_positions(&[120, 3000], 3439), [120, 3000]);
    assert_eq!(disk_positions(&[120, 3000, 3438], 3439), [120, 3000, 3438]);
}
fn tools() -> Vec<Json> { vec![Json::parse(br#"{"function":{"name":"computer","parameters":{"type":"object","properties":{"action":{"type":"string"},"keys":{"type":"array"}},"required":["action"]}}}"#).unwrap()] }
#[test]
fn native_calls_preserve_types_and_reject_partial_or_unknown_calls() {
    let call = "<tool_call>\n<function=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n<parameter=keys>\n[{\"x\":42,\"key\":\"h\"}]\n</parameter>\n</function>\n</tool_call>";
    let converted=normalize(call,&tools()).unwrap();
    let mut parser=crate::dsv41::chat::StreamParser::new(Mode::Chat); parser.push(&converted);
    assert!(parser.tool_call_error().is_none()); let (_,calls)=parser.finish();
    let args=Json::parse(calls[0].arguments.as_bytes()).unwrap(); assert!(args.get("keys").unwrap().as_array().is_some());
    assert!(normalize(&call.replace("</tool_call>",""),&tools()).is_err());
    assert!(normalize(&call.replace("function=computer","function=unknown"),&tools()).is_err());
    assert!(normalize(&format!("{call} unwanted suffix"),&tools()).is_err());
}
/// Qwen3-VL calls in Qwen3's own form, JSON in the call's tags (its arguments an object, or that object as a string):
/// read as the XML form is, each argument's type kept, and an undeclared function still refused.
#[test]
fn json_calls_are_read_as_the_xml_ones() {
    for call in [
        "<tool_call>\n{\"name\": \"computer\", \"arguments\": {\"action\": \"keyboard_sequence\", \"keys\": [{\"x\": 42, \"key\": \"h\"}]}}\n</tool_call>",
        "<tool_call>\n{\"name\": \"computer\", \"arguments\": \"{\\\"action\\\": \\\"keyboard_sequence\\\", \\\"keys\\\": [{\\\"x\\\": 42, \\\"key\\\": \\\"h\\\"}]}\"}\n</tool_call>",
    ] {
        let converted = normalize(call, &tools()).unwrap();
        let mut parser = crate::dsv41::chat::StreamParser::new(Mode::Chat);
        parser.push(&converted);
        assert!(parser.tool_call_error().is_none(), "{converted}");
        let (_, calls) = parser.finish();
        let args = Json::parse(calls[0].arguments.as_bytes()).unwrap();
        assert_eq!(args.get("action").and_then(Json::as_str), Some("keyboard_sequence"));
        assert!(args.get("keys").unwrap().as_array().is_some(), "{converted}");
    }
    assert!(normalize("<tool_call>{\"name\": \"unknown\", \"arguments\": {}}</tool_call>", &tools()).is_err());
    assert!(normalize("<tool_call>{\"name\": \"computer\", \"arguments\": [1]}</tool_call>", &tools()).is_err());
}
#[test]
fn a_function_named_with_the_parameter_tag_is_read_as_the_function() {
    let call = "<tool_call>\n<parameter=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n</parameter>\n</tool_call>";
    let fixed = "<tool_call>\n<function=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n</function>\n</tool_call>";
    assert_eq!(normalize(call, &tools()).unwrap(), normalize(fixed, &tools()).unwrap());
    // No stray closing tag, or a proper </function>: read the same.
    assert!(normalize(&call.replace("</parameter>\n</parameter>", "</parameter>"), &tools()).is_ok());
    assert!(normalize(&call.replace("</parameter>\n</parameter>", "</parameter>\n</function>"), &tools()).is_ok());
    // Only for a declared function, and only when the tags balance.
    assert!(normalize(&call.replace("parameter=computer", "parameter=unknown"), &tools()).is_err());
    assert!(normalize(&call.replace("</parameter>\n</parameter>", "</parameter>\n</parameter>\n</parameter>"), &tools()).is_err());
}
#[test]
fn a_function_tag_behind_a_stray_template_token_is_read_as_the_function() {
    let fixed = "<tool_call>\n<function=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n</function>\n</tool_call>";
    let want = normalize(fixed, &tools()).unwrap();
    // `<|im_start|>` where the `<` belongs, or no `<` at all.
    assert_eq!(normalize(&fixed.replace("<function=", "<|im_start|>function="), &tools()).unwrap(), want);
    assert_eq!(normalize(&fixed.replace("<function=", "function="), &tools()).unwrap(), want);
    assert_eq!(normalize(&fixed.replace("<function=", "<|im_start|>function=").replace("\n</function>", ""), &tools()).unwrap(), want);
    // Still only a declared function with balanced tags, and only a template-like token.
    assert!(normalize(&fixed.replace("<function=computer", "<|im_start|>function=unknown"), &tools()).is_err());
    assert!(normalize(&fixed.replace("<function=", "<|im_start|>function=").replace("</parameter>", ""), &tools()).is_err());
    assert!(normalize(&fixed.replace("<function=", "<|a b|>function="), &tools()).is_err());
}
#[test]
fn rejected_calls_are_quoted_from_their_start() {
    assert_eq!(call_excerpt("prose <tool_call>\n{\"name\":\"x\"}"), "\"<tool_call>\\n{\\\"name\\\":\\\"x\\\"}\"");
    let long = format!("<tool_call>{}", "é".repeat(400));
    assert!(call_excerpt(&long).ends_with('…'));
}
#[test]
fn template_keeps_tool_images_and_non_thinking_prefix() {
    let msgs=Json::parse(br#"[{"role":"user","content":"look"},{"role":"tool","content":[{"type":"text","text":"screen"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]"#).unwrap();
    let encoded=chat_prompt(msgs.as_array().unwrap(),&Options {mode:Mode::Chat,effort:25,drop_thinking:true}).unwrap();
    assert_eq!(encoded.images.len(),1); assert!(encoded.prompt.contains(IMAGE));
    assert!(encoded.prompt.ends_with("<think>\n\n</think>\n\n"));
}

// ---- Two conversations taking turns on Qwen3.8-Flash-Next: what the owner's call test did
// between 11:22 and 11:26, a runner of about 21,000 tokens and a call's sub-agent of about
// 5,000, sharing a 520-token system prompt. Every switch read the whole prompt again (36 s).

use crate::engines::qwen::park::fixtures::{dump, kv_like_flash, ROWS};
use crate::engines::qwen::park::Checkpoint;

const IM: u32 = 1;

fn mix(t: u32, pos: u32, salt: u32, j: u32) -> f32 {
    let mut h = ((t as u64) << 32) ^ (pos as u64);
    h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (((salt as u64) << 8) | j as u64);
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    (h % 2000) as f32 / 16.0 - 62.5
}

/// A fake model: what it holds is a pure function of the token sequence, as a transformer's
/// cache is (a row for each token and position, recurrent states that fold every token in), so
/// a cache that holds a prefix and one built from nothing agree to the last bit.
fn forward(kv: &mut KvCache, tokens: &[u32]) {
    for &t in tokens {
        let pos = kv.len as u32;
        for (slot, heads, dim) in ROWS {
            let k: Vec<f32> = (0..heads * dim).map(|j| mix(t, pos, slot as u32, j as u32)).collect();
            let v: Vec<f32> = k.iter().map(|x| -x * 0.5).collect();
            let backend = kv.layer_backends[slot].clone();
            kv.append(backend.as_ref(), slot, &Tensor::from_vec(k, vec![1, heads, dim]), &Tensor::from_vec(v, vec![1, heads, dim]));
        }
        kv.commit(1);
        let fold = |old: Option<Tensor>, shape: Vec<usize>, salt: u32| -> Tensor {
            let mut data = old.map_or_else(|| vec![0.0; shape.iter().product()], |o| o.data().to_vec());
            for (j, x) in data.iter_mut().enumerate() { *x = *x * 0.75 + mix(t, 0, salt, j as u32); }
            Tensor::from_vec(data, shape)
        };
        kv.ssm_state[1] = Some(fold(kv.ssm_state[1].take(), vec![2, 3, 3], 11));
        kv.ssm_conv[1] = Some(fold(kv.ssm_conv[1].take(), vec![3, 7], 12));
        kv.ssm_state[3] = Some(fold(kv.ssm_state[3].take(), vec![2], 13));
        kv.ssm_conv[3] = Some(fold(kv.ssm_conv[3].take(), vec![4, 5], 14));
    }
}

/// What a cache holds after `prompt` and `reply`, read from nothing.
fn cold(prompt: &[u32], reply: &[u32]) -> Vec<Option<Vec<u32>>> {
    let mut kv = kv_like_flash(65536);
    forward(&mut kv, prompt);
    forward(&mut kv, reply);
    dump(&kv)
}

struct Served { source: &'static str, start: usize, swapped: bool, read: usize }

/// The engine's state, and `QwenEngine::generate`'s cache logic for Flash-Next as written there
/// (no disk, no Qwen3.5 checkpoint restore), with the fake model's prefill and decode.
struct Mini { kv: KvCache, covered: Vec<u64>, checkpoints: Vec<Checkpoint>, parking: Parking }

impl Mini {
    fn new(budget: usize, log: bool) -> Self {
        Mini { kv: kv_like_flash(65536), covered: Vec::new(), checkpoints: Vec::new(), parking: Parking::new(budget, log) }
    }
    fn serve(&mut self, prompt: &[u32], reply: &[u32], forget: bool) -> Served {
        let keys: Vec<u64> = prompt.iter().map(|&t| t as u64).collect();
        let stops = checkpoint_positions(prompt, Some(IM));
        let swapped = !forget && self.parking.swap_in(Held { kv: &mut self.kv, covered: &mut self.covered, checkpoints: &mut self.checkpoints, private: false }, &keys);
        if !forget {
            self.parking.park_displaced(Held { kv: &mut self.kv, covered: &mut self.covered, checkpoints: &mut self.checkpoints, private: false }, &keys);
        }
        let common = self.covered.iter().zip(&keys).take_while(|(a, b)| a == b).count();
        self.checkpoints.retain(|(saved, _, _)| saved.len() <= common && keys.starts_with(saved));
        let mut start = if common == self.covered.len() && common < keys.len() { common } else { 0 };
        let mut source = if start > 0 { "memory" } else { "none" };
        if let Some((saved, snap, _)) = self.checkpoints.iter().filter(|(saved, _, _)| saved.len() > start && saved.len() < keys.len() && keys.starts_with(saved)).max_by_key(|(saved, _, _)| saved.len()) {
            snap.restore_slots(&mut self.kv, common).unwrap();
            self.covered = saved.clone(); start = saved.len(); source = "checkpoint";
            self.checkpoints.retain(|(saved, _, _)| saved.len() <= start);
        }
        if swapped && start > 0 { source = "ram"; }
        if start == 0 { self.kv.reset(); self.covered.clear(); self.checkpoints.clear(); }
        let mut pos = start;
        while pos < keys.len() {
            let end = (pos + PREFILL_CHUNK).min(keys.len()).min(stops.iter().copied().find(|&s| s > pos).unwrap_or(keys.len()));
            forward(&mut self.kv, &prompt[pos..end]);
            self.covered.extend_from_slice(&keys[pos..end]);
            pos = end;
            if stops.contains(&pos) && !self.checkpoints.iter().any(|(saved, _, _)| saved == &keys[..pos]) {
                let base = stops.first() == Some(&pos);
                self.checkpoints.push((keys[..pos].to_vec(), RecurrentSnapshot::capture(&self.kv), base));
                trim_checkpoints(&mut self.checkpoints);
            }
        }
        for &t in reply { forward(&mut self.kv, &[t]); self.covered.push(t as u64); }
        Served { source, start, swapped, read: keys.len() - start }
    }
}

/// A conversation: the system prompt every conversation starts with (520 tokens), a system
/// prompt of its own after it (300 tokens, so that the first message boundary lies past what
/// they share), and the messages that follow.
#[derive(Clone)]
struct Chat { id: u32, messages: Vec<Vec<u32>>, shared_system: bool }

impl Chat {
    fn new(id: u32, first: usize) -> Self {
        Chat { id, messages: vec![Self::message(id, 1, first)], shared_system: false }
    }
    /// As `new`, but its own part of the system prompt is the same for every conversation: all
    /// of them share it up to the first message boundary.
    fn sharing_its_system_prompt(id: u32, first: usize) -> Self {
        Chat { shared_system: true, ..Self::new(id, first) }
    }
    /// A message of `n` tokens of its own; it begins at a message boundary.
    fn message(id: u32, turn: u32, n: usize) -> Vec<u32> {
        std::iter::once(IM).chain((1..n as u32).map(|i| (id << 20) + (turn << 16) + i)).collect()
    }
    /// The assistant's turn as a later prompt renders it: its header, the reply, its end.
    fn answered(&mut self, reply: &[u32], rendered_with: Option<u32>, next: usize) {
        let mut turn = vec![IM, 2];
        turn.extend(rendered_with);
        turn.extend(reply);
        turn.push(3);
        self.messages.push(turn);
        let n = self.messages.len() as u32;
        self.messages.push(Self::message(self.id, n, next));
    }
    /// What a request sends: the system prompts, the messages and the assistant's header.
    fn prompt(&self) -> Vec<u32> {
        let mut p: Vec<u32> = std::iter::once(IM).chain(5..524).collect();
        assert_eq!(p.len(), 520);
        let own = if self.shared_system { 0 } else { self.id };
        p.extend((0..300).map(|i| (own << 20) + (1 << 19) + i));
        for m in &self.messages { p.extend(m); }
        p.extend([IM, 2]);
        p
    }
}

fn lengths(m: &Mini) -> Vec<usize> {
    m.parking.keys_held().iter().map(Vec::len).collect()
}

#[test]
fn two_conversations_taking_turns_swap_in_with_full_reuse_after_the_first_round() {
    let reply = |id: u32, n: u32| -> Vec<u32> { (0..n).map(|i| 900_000 + id * 1_000 + i).collect() };
    let (ra1, ra2, ra3, rb1, rb2, rb3) = (reply(1, 40), reply(2, 25), reply(3, 60), reply(4, 30), reply(5, 20), reply(6, 35));
    let mut m = Mini::new(1 << 40, true);
    // The runner (A) and the call's sub-agent (B).
    let (mut a, mut b) = (Chat::new(10, 16_180), Chat::new(20, 3_680));

    // The first round reads everything: nothing was set aside yet.
    let a1 = a.prompt();
    let s = m.serve(&a1, &ra1, false);
    assert_eq!((s.source, s.start, s.swapped, s.read), ("none", 0, false, a1.len()));
    let b1 = b.prompt();
    let s = m.serve(&b1, &rb1, false);
    assert_eq!((s.source, s.start, s.swapped, s.read), ("none", 0, false, b1.len()), "520 tokens in common are no checkpoint");
    // A1 was big and mostly unrelated: set aside as B1 displaced it.
    assert_eq!(lengths(&m), [a1.len() + ra1.len()]);

    // From here on every request is a swap-in that reads only what is new.
    a.answered(&ra1, None, 2_000);
    let a2 = a.prompt();
    let s = m.serve(&a2, &ra2, false);
    assert_eq!((s.source, s.swapped), ("ram", true));
    assert_eq!(s.start, a1.len() + ra1.len(), "all of A1 and its reply");
    assert_eq!(s.read, a2.len() - (a1.len() + ra1.len()));
    assert!(s.read < 2_100, "only the new message, the end of the reply and the header");
    assert_eq!(dump(&m.kv), cold(&a2, &ra2), "the restored cache goes on bit for bit as a cold read of the whole prompt would");
    assert_eq!(lengths(&m), [b1.len() + rb1.len()], "B1 waits now");

    b.answered(&rb1, None, 400);
    let b2 = b.prompt();
    let s = m.serve(&b2, &rb2, false);
    assert_eq!((s.source, s.swapped, s.start), ("ram", true, b1.len() + rb1.len()));
    assert_eq!(s.read, b2.len() - (b1.len() + rb1.len()));
    assert_eq!(dump(&m.kv), cold(&b2, &rb2));
    assert_eq!(lengths(&m), [a2.len() + ra2.len()]);

    // The runner's next prompt renders its last reply with something more in it: it goes back
    // to the checkpoint taken before the reply, which was set aside with the state.
    a.answered(&ra2, Some(9), 1_500);
    let a3 = a.prompt();
    let s = m.serve(&a3, &ra3, false);
    assert_eq!((s.source, s.swapped), ("ram", true));
    assert_eq!(s.start, a2.len() - 1, "the checkpoint one token short of A2's prompt");
    assert_eq!(s.read, a3.len() - (a2.len() - 1));
    assert_eq!(dump(&m.kv), cold(&a3, &ra3));

    b.answered(&rb2, None, 300);
    let b3 = b.prompt();
    let s = m.serve(&b3, &rb3, false);
    assert_eq!((s.source, s.swapped, s.start), ("ram", true, b2.len() + rb2.len()));
    assert_eq!(dump(&m.kv), cold(&b3, &rb3));
}

#[test]
fn a_reply_rendered_otherwise_after_a_tools_round_trip_goes_back_to_its_own_checkpoint() {
    // A call's agent: a reply, the tool's answer and the agent's next request, then a prompt that renders that reply
    // otherwise (the tool's call written out). Its checkpoints are the request before last's: still kept.
    let (r1, r2, r3) = (vec![905_000; 40], vec![906_000; 20], vec![907_000; 30]);
    let mut m = Mini::new(1 << 40, false);
    let mut c = Chat::new(30, 2_600);
    let p1 = c.prompt();
    m.serve(&p1, &r1, false);
    let mut later = c.clone();
    c.answered(&r1, None, 120);
    let p2 = c.prompt();
    let s = m.serve(&p2, &r2, false);
    assert_eq!(s.source, "memory");
    later.answered(&r1, Some(9), 150);
    let p3 = later.prompt();
    let s = m.serve(&p3, &r3, false);
    assert_eq!((s.source, s.start), ("checkpoint", p1.len() - 1), "the checkpoint a token short of the first prompt, not the base");
    assert!(s.read < 250, "{} read again", s.read);
    assert_eq!(dump(&m.kv), cold(&p3, &r3));
}

#[test]
fn a_reply_rendered_differently_still_finds_its_checkpoint_after_a_conversation_was_discarded() {
    // The runner's A1 is set aside as B1 displaces it (nothing of it is shared but 520 tokens),
    // and its next prompt renders the last reply with something more in it, so that it
    // can only go back to a checkpoint of A1: the checkpoints went into the stash with it,
    // though the engine drops the ones a prompt does not share as soon as B1 is read.
    let (ra1, ra2, rb1) = (vec![902_000; 40], vec![903_000; 25], vec![904_000; 30]);
    let mut m = Mini::new(1 << 40, false);
    let (mut a, b) = (Chat::new(10, 16_180), Chat::new(20, 3_680));
    let a1 = a.prompt();
    m.serve(&a1, &ra1, false);
    m.serve(&b.prompt(), &rb1, false);
    assert_eq!(lengths(&m), [a1.len() + ra1.len()]);
    a.answered(&ra1, Some(9), 2_000);
    let a2 = a.prompt();
    let s = m.serve(&a2, &ra2, false);
    assert_eq!((s.source, s.swapped), ("ram", true));
    assert_eq!(s.start, a1.len() - 1, "the checkpoint one token short of A1's prompt");
    assert_eq!(s.read, a2.len() - (a1.len() - 1));
    assert_eq!(dump(&m.kv), cold(&a2, &ra2));
}

#[test]
fn a_conversation_rolled_back_to_a_shared_system_prompt_is_set_aside_as_a_copy() {
    // The two conversations share their whole system prompt, which ends at a message boundary:
    // B1 goes back to A1's checkpoint there and writes over the rest of A1. A1 has to be set
    // aside before that, and the checkpoint it leaves B1 is still B1's to use.
    let (ra1, ra2, rb1, rb2) = (vec![902_000; 40], vec![903_000; 25], vec![904_000; 30], vec![905_000; 20]);
    let mut m = Mini::new(1 << 40, false);
    let (mut a, mut b) = (Chat::sharing_its_system_prompt(10, 16_180), Chat::sharing_its_system_prompt(20, 3_680));
    let a1 = a.prompt();
    m.serve(&a1, &ra1, false);
    let b1 = b.prompt();
    let s = m.serve(&b1, &rb1, false);
    assert_eq!((s.source, s.start, s.swapped), ("checkpoint", 820, false), "B1 reads on from A1's checkpoint at the end of the system prompt");
    assert_eq!(dump(&m.kv), cold(&b1, &rb1));
    assert_eq!(lengths(&m), [a1.len() + ra1.len()], "and A1 waits");
    a.answered(&ra1, None, 2_000);
    let a2 = a.prompt();
    let s = m.serve(&a2, &ra2, false);
    assert_eq!((s.source, s.swapped, s.start), ("ram", true, a1.len() + ra1.len()));
    assert_eq!(dump(&m.kv), cold(&a2, &ra2));
    b.answered(&rb1, None, 400);
    let b2 = b.prompt();
    let s = m.serve(&b2, &rb2, false);
    assert_eq!((s.source, s.swapped, s.start), ("ram", true, b1.len() + rb1.len()));
    assert_eq!(dump(&m.kv), cold(&b2, &rb2));
}

#[test]
fn without_it_every_switch_reads_the_whole_prompt_again_and_with_it_only_what_is_new() {
    let reply = |id: u32| -> Vec<u32> { (0..30).map(|i| 900_000 + id * 1_000 + i).collect() };
    // (prompt length, tokens read) of each request, three rounds of the two conversations.
    let rounds = |budget: usize| -> Vec<(usize, usize)> {
        let mut m = Mini::new(budget, false);
        let (mut a, mut b) = (Chat::new(10, 16_180), Chat::new(20, 3_680));
        let mut out = Vec::new();
        for round in 0..3u32 {
            for (chat, id) in [(&mut a, 1u32), (&mut b, 2)] {
                if round > 0 { chat.answered(&reply(id * 10 + round - 1), None, 500); }
                let p = chat.prompt();
                out.push((p.len(), m.serve(&p, &reply(id * 10 + round), false).read));
            }
        }
        out
    };
    let plain = rounds(0);
    assert!(plain.iter().all(|&(len, read)| read == len), "what ran on 1 October: {plain:?}");
    let parked = rounds(1 << 40);
    assert_eq!(parked[..2], plain[..2]);
    // One token (the end of the reply), the message of 500 and the header: 503.
    assert!(parked[2..].iter().all(|&(_, read)| read == 503), "{parked:?}");
    // A budget that holds one conversation and not both keeps the one displaced last, and the
    // other is read whole.
    let tight = rounds(2_000_000);
    let total = |r: &[(usize, usize)]| r.iter().map(|x| x.1).sum::<usize>();
    assert!(total(&parked) < total(&tight) && total(&tight) <= total(&plain), "{} {} {}", total(&parked), total(&tight), total(&plain));
}

#[test]
fn an_incognito_request_neither_takes_from_the_stash_nor_adds_to_it() {
    let mut m = Mini::new(1 << 40, false);
    let (a, b) = (Chat::new(10, 6_000), Chat::new(20, 3_000));
    m.serve(&a.prompt(), &[7, 8], false);
    m.serve(&b.prompt(), &[7, 8], false);
    assert_eq!(m.parking.held().0, 1, "A waits");
    // An incognito request that continues A takes nothing from the stash, and B, which it
    // displaces, is not set aside for it.
    let mut again = Chat { id: 10, messages: a.messages.clone(), shared_system: false };
    again.answered(&[7, 8], None, 100);
    let s = m.serve(&again.prompt(), &[], true);
    assert_eq!((s.source, s.start, s.swapped), ("none", 0, false));
    assert_eq!(lengths(&m).len(), 1, "A is still the only one");
    // And the engine forgets everything when it ends (QwenEngine::forget_state).
    m.kv.reset(); m.covered.clear(); m.checkpoints.clear(); m.parking.clear();
    assert_eq!(m.parking.held(), (0, 0));
}

/// MiniCPM-V 4.7's prompt for a picture and a question, made here (its template, its tokenizer, the picture cut and
/// laid out as its processor lays it out), is transformers' own (`OAIY_MINICPMV_REFERENCE`: tools/minicpmv/reference.py
/// with `OAIY_MINICPMV_PROMPT` on `OAIY_MINICPMV_PICTURE`; `OAIY_MINICPMV_MODE` 4x where it ran so), token for token;
/// its pictures' tiles the reference's grids.
#[test]
#[ignore = "needs MiniCPM-V 4.7 (OAIY_MINICPMV_GGUF), a picture (OAIY_MINICPMV_PICTURE) and transformers' prompt for it (OAIY_MINICPMV_REFERENCE, OAIY_MINICPMV_PROMPT)"]
fn a_minicpmv_prompt_is_transformers() {
    let var = |k: &str| std::env::var(k).ok();
    let (Some(model), Some(picture), Some(reference), Some(question)) = (var("OAIY_MINICPMV_GGUF"), var("OAIY_MINICPMV_PICTURE"), var("OAIY_MINICPMV_REFERENCE"), var("OAIY_MINICPMV_PROMPT")) else { return };
    let tok = tokenizer::Tokenizer::from_gguf(&gguf::GgufFile::open(&model).unwrap()).unwrap();
    let msgs = Json::Arr(vec![Json::obj([("role", Json::str("user")), ("content", Json::Arr(vec![
        Json::obj([("type", Json::str("image_url")), ("image_url", Json::obj([("url", Json::str(&picture))]))]),
        Json::obj([("type", Json::str("text")), ("text", Json::str(&question))]),
    ]))])]);
    let encoded = chat_prompt_as(msgs.as_array().unwrap(), &Options { mode: Mode::Chat, effort: 50, drop_thinking: true }, Pictures::MiniCpmV).unwrap();
    let id = |s: &str| tok.token_id(s).unwrap();
    let m = MiniCpmV {
        markers: crate::llm::vision::mmproj::minicpmv::Markers { image_start: id("<image>"), image_end: id("</image>"), slice_start: id("<slice>"), slice_end: id("</slice>"), newline: 198 },
        merge: if var("OAIY_MINICPMV_MODE").as_deref() == Some("4x") { 2 } else { 4 },
        max_slices: 9,
        mean: [0.5; 3],
        std: [0.5; 3],
    };
    let prompt = tok.encode(&encoded.prompt, false).unwrap();
    let (ids, images) = prepare_minicpmv(&encoded.images, prompt, id("<|image_pad|>"), &m, |s| tok.encode(s, false).unwrap(), true, 1 << 20).unwrap();
    // the reference's tokens (an I64 tensor of a safetensors file)
    let bytes = std::fs::read(&reference).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = Json::parse(&bytes[8..8 + n]).unwrap();
    let span = header.get("input_ids").unwrap().get("data_offsets").unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as usize).collect::<Vec<_>>();
    let theirs: Vec<u32> = bytes[8 + n + span[0]..8 + n + span[1]].chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    if ids != theirs {
        let first = ids.iter().zip(&theirs).position(|(a, b)| a != b).unwrap_or(ids.len().min(theirs.len()));
        panic!("{} tokens here, {} transformers'; the first difference at {first}: {:?} against {:?}\n{}", ids.len(), theirs.len(), &ids[first.saturating_sub(3)..(first + 8).min(ids.len())], &theirs[first.saturating_sub(3)..(first + 8).min(theirs.len())], tok.decode(&ids[first.saturating_sub(10)..(first + 10).min(ids.len())]));
    }
    eprintln!("{} tokens, the same as transformers'; {} tiles", ids.len(), images.len());
}

/// A safetensors file's F32 and I64 tensors (`tools/minicpmv/reference.py`'s), each its shape and values as f32.
fn minicpmv_reference(path: &str) -> std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = Json::parse(&bytes[8..8 + n]).unwrap();
    let mut out = std::collections::BTreeMap::new();
    for (name, t) in header.as_object().unwrap().iter().filter(|(k, _)| k != "__metadata__") {
        let shape = t.get("shape").unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as usize).collect();
        let span: Vec<usize> = t.get("data_offsets").unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as usize).collect();
        let raw = &bytes[8 + n + span[0]..8 + n + span[1]];
        let values = match t.get("dtype").and_then(Json::as_str) {
            Some("F32") => raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            _ => raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        };
        out.insert(name.clone(), (shape, values));
    }
    out
}

/// MiniCPM-V 4.7's tower chained on the first WebGPU adapter (`OAIY_MINICPMV_MMPROJ`) against transformers' own
/// (`OAIY_MINICPMV_REFERENCE`): each of transformers' tiles to the same tokens, and how long a tile takes.
#[test]
#[cfg(feature = "webgpu")]
#[ignore = "needs MiniCPM-V 4.7's projector (OAIY_MINICPMV_MMPROJ), transformers' run on a picture (OAIY_MINICPMV_REFERENCE) and a WebGPU adapter"]
fn the_minicpmv_tower_on_webgpu_is_transformers() {
    let var = |k: &str| std::env::var(k).ok();
    let (Some(mmproj), Some(reference)) = (var("OAIY_MINICPMV_MMPROJ"), var("OAIY_MINICPMV_REFERENCE")) else { return };
    let Ok(gpu) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
    let want = minicpmv_reference(&reference);
    let file = gguf::GgufFile::open(&mmproj).unwrap();
    let mut mm = crate::llm::MmProj::from_gguf(&file, Arc::new(gpu)).unwrap();
    let mut before = Vec::new();
    for k in 0.. {
        let Some((shape, pixels)) = want.get(&format!("tile.{k}")) else { break };
        let clock = std::time::Instant::now();
        let got = mm.forward(&Tensor::from_vec(pixels.clone(), shape.clone())).unwrap().to_host();
        let took = clock.elapsed().as_secs_f64();
        let (fshape, theirs) = &want[&format!("features.{k}")];
        assert_eq!(got.shape(), fshape.as_slice(), "tile {k}");
        let dot: f64 = got.data().iter().zip(theirs).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm(got.data()) * norm(theirs));
        let worst = got.data().iter().zip(theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("tile {k} {shape:?}: cosine {cosine:.6}, the largest difference {worst:.4}, in {took:.2} s");
        assert!(cosine > 0.999, "tile {k}: {cosine}");
        before.push(got);
    }
    // the host's copy of its matrices let go: the device's tower gives every tile's tokens as it did
    let freed = mm.keep_on_device();
    eprintln!("the host's copy let go: {:.2} GB", freed as f64 / 1e9);
    assert!(freed > 1_000_000_000, "{freed} bytes let go");
    for (k, was) in before.iter().enumerate() {
        let (shape, pixels) = &want[&format!("tile.{k}")];
        let got = mm.forward(&Tensor::from_vec(pixels.clone(), shape.clone())).unwrap().to_host();
        assert_eq!(got.data(), was.data(), "tile {k} after the host's copy went");
    }
}

/// A Qwen3-VL tower (Qwen3.8 27B's projector: OAIY_QWEN3VL_MMPROJ) on the GPU gives a picture's tokens as it did once
/// the host's copy of its matrices is let go (`MmProj::keep_on_device`: the GPU's f16 copy the one that runs), and that
/// copy is the most of its weights.
#[cfg(feature = "webgpu")]
#[test]
#[ignore = "needs a WebGPU adapter and a Qwen3-VL mmproj (OAIY_QWEN3VL_MMPROJ)"]
fn a_qwen3vl_tower_gives_its_tokens_with_the_hosts_copy_let_go() {
    let Ok(mmproj) = std::env::var("OAIY_QWEN3VL_MMPROJ") else { return };
    let Ok(gpu) = ggml_rs_wgpu::WgpuBackend::new(None) else { return };
    let file = gguf::GgufFile::open(&mmproj).unwrap();
    let mut mm = crate::llm::MmProj::from_gguf(&file, Arc::new(gpu)).unwrap();
    let crate::llm::MmProj::Qwen3Vl(q) = &mm else { panic!("a Qwen3-VL projector") };
    let side = q.config.image_size;
    let picture: Vec<f32> = (0..3 * side * side).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
    let image = Tensor::from_vec(picture, vec![3, side, side]);
    let clock = std::time::Instant::now();
    let before = mm.forward(&image).unwrap().to_host();
    let took = clock.elapsed().as_secs_f64();
    let freed = mm.keep_on_device();
    let after = mm.forward(&image).unwrap().to_host();
    eprintln!("{:?} tokens in {took:.2} s; the host's copy let go: {:.2} GB", before.shape(), freed as f64 / 1e9);
    assert!(freed > 1_000_000_000, "{freed} bytes let go");
    assert_eq!(after.data(), before.data());
}

/// Qwen3-VL's turns carry no thought: an instruct model's template opens the reply straight away, and an earlier reply
/// stands as it was said; its pictures stand as Qwen3.5's.
#[test]
fn qwen3_vl_prompts_open_no_thought() {
    let msgs = Json::Arr(vec![
        Json::obj([("role", Json::str("user")), ("content", Json::str("hi"))]),
        Json::obj([("role", Json::str("assistant")), ("content", Json::str("hello"))]),
        Json::obj([("role", Json::str("user")), ("content", Json::Arr(vec![
            Json::obj([("type", Json::str("image_url")), ("image_url", Json::obj([("url", Json::str("data:image/png;base64,iVBORw0KGgo="))]))]),
            Json::obj([("type", Json::str("text")), ("text", Json::str("what is it?"))]),
        ]))]),
    ]);
    let encoded = chat_prompt_as(msgs.as_array().unwrap(), &Options { mode: Mode::Chat, effort: 50, drop_thinking: true }, Pictures::Qwen3Vl).unwrap();
    assert!(!encoded.prompt.contains("<think>"), "{}", encoded.prompt);
    assert!(encoded.prompt.contains("<|im_start|>assistant\nhello<|im_end|>"), "{}", encoded.prompt);
    assert!(encoded.prompt.contains("<|vision_start|><|image_pad|><|vision_end|>what is it?"), "{}", encoded.prompt);
    assert!(encoded.prompt.ends_with("<|im_end|>\n<|im_start|>assistant\n"), "{}", encoded.prompt);
    assert_eq!(encoded.images.len(), 1);
}

/// Qwen3-VL is told of its tools in its own template's words (the tools as JSON, each call asked for as JSON in the
/// call's tags), after the system's text; a past call is shown in that form.
#[test]
fn qwen3_vl_tools_are_offered_in_its_own_form() {
    let msgs = Json::parse(br#"[
        {"role": "system", "content": "Be brief.", "tools": [{"type": "function", "function": {"name": "computer", "parameters": {"type": "object"}}}]},
        {"role": "user", "content": "press h"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "computer", "arguments": "{\"action\": \"key\"}"}}]},
        {"role": "tool", "content": "done"}
    ]"#).unwrap();
    let encoded = chat_prompt_as(msgs.as_array().unwrap(), &Options { mode: Mode::Chat, effort: 50, drop_thinking: true }, Pictures::Qwen3Vl).unwrap();
    let p = &encoded.prompt;
    assert!(p.starts_with("<|im_start|>system\nBe brief.\n\n# Tools\n\nYou may call one or more functions"), "{p}");
    assert!(p.contains("<tools>\n{\"type\":\"function\""), "{p}");
    assert!(p.contains("{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"), "{p}");
    assert!(p.contains("<|im_start|>assistant\n<tool_call>\n{\"name\": \"computer\", \"arguments\": {\"action\": \"key\"}}\n</tool_call><|im_end|>\n"), "{p}");
    assert!(p.contains("<|im_start|>user\n<tool_response>\ndone\n</tool_response><|im_end|>\n"), "{p}");
    assert!(!p.contains("<function="), "{p}");
}

/// A dense model of another family whose template is ChatML with thought blocks and Qwen's XML calls is told of its
/// tools as that template tells it (the system's text, then Qwen3's preamble, the tools and an example call), and shown
/// a past reply as it writes one (an empty thought, its text, its first call a blank line on and the next a line on).
#[test]
fn a_chatml_template_with_qwens_calls_is_written_as_it_writes_itself() {
    let msgs = Json::parse(br#"[
        {"role": "system", "content": "Be brief.", "tools": [{"type": "function", "function": {"name": "computer", "parameters": {"type": "object"}}}]},
        {"role": "user", "content": "press h"},
        {"role": "assistant", "content": "Pressing it.", "tool_calls": [{"function": {"name": "computer", "arguments": "{\"action\": \"key\", \"count\": 2}"}}, {"function": {"name": "wait", "arguments": "{}"}}]},
        {"role": "tool", "content": "done"},
        {"role": "tool", "content": "ok"}
    ]"#).unwrap();
    let encoded = chat_prompt_as(msgs.as_array().unwrap(), &Options { mode: Mode::Chat, effort: 50, drop_thinking: true }, Pictures::None).unwrap();
    let tool = msgs.as_array().unwrap()[0].get("tools").and_then(Json::as_array).unwrap()[0].to_python_json();
    let want = format!(
        "<|im_start|>system\nBe brief.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tool}\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
         <|im_start|>user\npress h<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\nPressing it.\n\n<tool_call>\n<function=computer>\n<parameter=action>\nkey\n</parameter>\n<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=wait>\n</function>\n</tool_call><|im_end|>\n\
         <|im_start|>user\n<tool_response>\ndone\n</tool_response>\n<tool_response>\nok\n</tool_response><|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(encoded.prompt, want);
    // thinking: the reply opens its thought
    let thinking = chat_prompt_as(msgs.as_array().unwrap(), &Options { mode: Mode::Thinking, effort: 50, drop_thinking: true }, Pictures::None).unwrap();
    assert!(thinking.prompt.ends_with("<|im_start|>assistant\n<think>\n"), "{}", thinking.prompt);
    // a picture is refused (the model reads text alone), in words an agent knows to go on in text by
    let picture = Json::parse(br#"[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]"#).unwrap();
    let refused = chat_prompt_as(picture.as_array().unwrap(), &Options { mode: Mode::Chat, effort: 50, drop_thinking: true }, Pictures::None).unwrap_err();
    assert!(refused.contains("image"), "{refused}");
}

/// Such a template's own system message, where a conversation has none, is read from it (the whole literal, not one
/// the message's text is added to) and written first, as the template writes it; a conversation with its own keeps it.
#[test]
fn a_chatml_template_s_own_system_message_is_written_where_there_is_none() {
    // as Nanbeige4.3's: the system's text where there is one, its own where not
    let template = r#"{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}{%- else %} {{- '<|im_start|>system\nYou are Example, a model made by an example lab.<|im_end|>\n' }} {%- endif %}"#;
    assert_eq!(default_system(template).as_deref(), Some("You are Example, a model made by an example lab."));
    assert_eq!(default_system(&template.replace('\'', "\"")).as_deref(), Some("You are Example, a model made by an example lab."));
    assert_eq!(default_system("{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"), None);
    let options = Options { mode: Mode::Thinking, effort: 50, drop_thinking: true };
    let asked = Json::parse(br#"[{"role": "user", "content": "hi"}]"#).unwrap();
    let encoded = chatml_prompt(asked.as_array().unwrap(), &options, Some("You are Example.")).unwrap();
    assert_eq!(encoded.prompt, "<|im_start|>system\nYou are Example.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n");
    let own = Json::parse(br#"[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]"#).unwrap();
    let encoded = chatml_prompt(own.as_array().unwrap(), &options, Some("You are Example.")).unwrap();
    assert!(encoded.prompt.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhi"), "{}", encoded.prompt);
    assert!(!encoded.prompt.contains("Example"), "{}", encoded.prompt);
}

/// Which templates are served so: ChatML that writes Qwen's XML calls and opens thought blocks, not Qwen3-VL's JSON
/// calls nor another family's turns.
#[test]
fn a_template_that_writes_qwens_calls_is_known() {
    let chatml_xml = "{{- '<|im_start|>system\n' }}<tools></tools>{{- '<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>' }}{{- '<|im_start|>assistant\n<think>\n' }}";
    let chatml_json = "{{- '<|im_start|>system\n' }}<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
    let llama3 = "{{- '<|start_header_id|>system<|end_header_id|>\n\n' }}";
    assert!(writes_qwen_calls(chatml_xml));
    assert!(!writes_qwen_calls(chatml_json));
    assert!(!writes_qwen_calls(llama3));
}
