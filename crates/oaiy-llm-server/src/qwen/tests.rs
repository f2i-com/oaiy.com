//! `qwen`'s tests.

use super::*;
#[test]
fn native_file_call_round_trip_preserves_literal_content() {
    let tools = vec![Json::parse(br#"{"function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}"#).unwrap()];
    let content = "\n<style>\n  :root { --bg: #000; }\n  body::before { background: var(--bg); content: 'λ 😀'; }\n</style>\n<script>const x = a < b && b > 0;</script>\n\n";
    let raw = format!("<tool_call>\n<function=write_file>\n<parameter=path>\nsite/index.html\n</parameter>\n<parameter=content>\n{content}\n</parameter>\n</function>\n</tool_call>");
    let mut stream = NativeStream::default();
    let mut parser = dsv41::chat::StreamParser::new(Mode::Chat);
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
    let mut parser = dsv41::chat::StreamParser::new(Mode::Thinking);
    let first = stream.push("Let me check", &[], false).unwrap();
    assert!(matches!(&parser.push(&first)[..], [dsv41::chat::Delta::Reasoning(t)] if t == "Let me check"));
    let next = stream.push("Let me check</think>Hello", &[], false).unwrap();
    assert!(parser.push(&next).iter().any(|d| matches!(d,dsv41::chat::Delta::Content(t) if t == "Hello")));
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
fn tools() -> Vec<Json> { vec![Json::parse(br#"{"function":{"name":"computer","parameters":{"type":"object","properties":{"action":{"type":"string"},"keys":{"type":"array"}},"required":["action"]}}}"#).unwrap()] }
#[test]
fn native_calls_preserve_types_and_reject_partial_or_unknown_calls() {
    let call = "<tool_call>\n<function=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n<parameter=keys>\n[{\"x\":42,\"key\":\"h\"}]\n</parameter>\n</function>\n</tool_call>";
    let converted=normalize(call,&tools()).unwrap();
    let mut parser=dsv41::chat::StreamParser::new(Mode::Chat); parser.push(&converted);
    assert!(parser.tool_call_error().is_none()); let (_,calls)=parser.finish();
    let args=Json::parse(calls[0].arguments.as_bytes()).unwrap(); assert!(args.get("keys").unwrap().as_array().is_some());
    assert!(normalize(&call.replace("</tool_call>",""),&tools()).is_err());
    assert!(normalize(&call.replace("function=computer","function=unknown"),&tools()).is_err());
    assert!(normalize(&format!("{call} unwanted suffix"),&tools()).is_err());
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

use crate::qwen_park::fixtures::{dump, kv_like_flash, ROWS};
use crate::qwen_park::Checkpoint;

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
