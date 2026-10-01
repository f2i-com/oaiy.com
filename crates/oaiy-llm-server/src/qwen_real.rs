//! TEST ONLY (review branch): the REAL `QwenEngine::run` / `generate` driven on the CPU, with a fake
//! `Hybrid` whose cache is a pure function of the token sequence and whose logits depend on every row
//! and every recurrent tensor the cache holds. So a restore that is off by a row, a slot, a tensor, or a
//! cache that is wrongly claimed changes what the engine answers, and the cache logic that is under
//! test is the code in `generate`, not a copy of it.
use crate::engine::{Event, Job, Sampling};
use crate::qwen::{Hybrid, QwenEngine};
use crate::qwen_park::fixtures::{kv_like_flash, ROWS};
use ggml_rs::Tensor;
use llama_rs::KvCache;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use tokenizer::Tokenizer;

pub(crate) const IM: u32 = 1;
pub(crate) const END: u32 = 2;
const FIRST: u32 = 4;
const WORDS: u32 = 100;

pub(crate) struct FakeModel {
    pub tok: Tokenizer,
    pub width: usize,
    /// A token that makes `forward` panic (a CUDA failure in the middle of a request).
    pub panic_on: Option<u32>,
    /// Tokens run through the model (prefill and decode).
    pub forwarded: Arc<AtomicUsize>,
}

impl FakeModel {
    pub fn new(panic_on: Option<u32>, forwarded: Arc<AtomicUsize>) -> Self {
        let mut tokens = vec!["<pad>".to_string(), "<|im_start|>".into(), "<|im_end|>".into(), "</think>".into()];
        for i in FIRST..FIRST + WORDS { tokens.push(format!("{i}.")); }
        let tok = Tokenizer::from_qwen3_bpe_parts(tokens, vec![], vec![1, 2, 3], END).unwrap();
        FakeModel { tok, width: 4, panic_on, forwarded }
    }

    fn mix(t: u32, pos: u32, salt: u32, j: u32) -> f32 {
        let mut h = ((t as u64) << 32) ^ (pos as u64);
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (((salt as u64) << 8) | j as u64);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 29;
        (h % 2000) as f32 / 16.0 - 62.5
    }

    pub fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Tensor, String> {
        if self.panic_on.is_some_and(|p| tokens.contains(&p)) { panic!("injected: the device failed in the middle of a request"); }
        self.forwarded.fetch_add(tokens.len(), Ordering::Relaxed);
        for &t in tokens {
            let pos = kv.len as u32;
            for (slot, heads, dim) in ROWS {
                let k: Vec<f32> = (0..heads * dim).map(|j| Self::mix(t, pos, slot as u32, j as u32)).collect();
                let v: Vec<f32> = k.iter().map(|x| -x * 0.5).collect();
                let backend = kv.layer_backends[slot].clone();
                kv.append(backend.as_ref(), slot, &Tensor::from_vec(k, vec![1, heads, dim]), &Tensor::from_vec(v, vec![1, heads, dim]));
            }
            kv.commit(1);
            let fold = |old: Option<Tensor>, shape: Vec<usize>, salt: u32| -> Tensor {
                let mut data = old.map_or_else(|| vec![0.0; shape.iter().product()], |o| o.data().to_vec());
                for (j, x) in data.iter_mut().enumerate() { *x = *x * 0.75 + Self::mix(t, 0, salt, j as u32); }
                Tensor::from_vec(data, shape)
            };
            kv.ssm_state[1] = Some(fold(kv.ssm_state[1].take(), vec![2, 3, 3], 11));
            kv.ssm_conv[1] = Some(fold(kv.ssm_conv[1].take(), vec![3, 7], 12));
            kv.ssm_state[3] = Some(fold(kv.ssm_state[3].take(), vec![2], 13));
            kv.ssm_conv[3] = Some(fold(kv.ssm_conv[3].take(), vec![4, 5], 14));
        }
        // The next word depends on everything the cache holds.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bits: u32| { h = (h ^ bits as u64).wrapping_mul(0x0000_0100_0000_01b3); };
        eat(kv.len as u32);
        for (slot, heads, dim) in ROWS {
            let n = kv.len * heads * dim;
            for x in &kv.k[slot].data()[..n] { eat(x.to_bits()); }
            for x in &kv.v[slot].data()[..n] { eat(x.to_bits()); }
        }
        for t in kv.ssm_state.iter().chain(&kv.ssm_conv).flatten() { for x in t.data() { eat(x.to_bits()); } }
        let mut logits = vec![0.0f32; (FIRST + WORDS) as usize];
        logits[(FIRST as u64 + h % WORDS as u64) as usize] = 1.0;
        Ok(Tensor::from_vec(logits, vec![(FIRST + WORDS) as usize]))
    }
}

pub(crate) fn new_kv(max: usize) -> KvCache { kv_like_flash(max) }

/// What one request came back with.
#[derive(Debug, Clone, Default)]
pub(crate) struct Reply {
    pub words: Vec<u32>,
    /// (cached tokens, source, common) as the engine reported them.
    pub reuse: Option<(usize, &'static str, usize)>,
    pub error: Option<String>,
    pub done: bool,
}

pub(crate) struct Engine {
    tx: Option<Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    pub forwarded: Arc<AtomicUsize>,
}

impl Engine {
    /// `budget` bytes of stash (0: parking off).
    pub fn start(budget: usize, panic_on: Option<u32>) -> Self {
        let forwarded = Arc::new(AtomicUsize::new(0));
        let engine = QwenEngine::new(Hybrid::Fake(Box::new(FakeModel::new(panic_on, forwarded.clone()))), None, 65536, false).park_up_to(budget);
        let (tx, rx) = channel();
        let thread = std::thread::spawn(move || engine.run(rx));
        Engine { tx: Some(tx), thread: Some(thread), forwarded }
    }

    fn send(&self, prompt: Vec<u32>, forget: bool, session: Option<&str>, wipe: bool, cancel: bool, max_tokens: usize) -> Reply {
        let (events, rx) = channel();
        let job = Job {
            tools: vec![], tool_precision: false, observer_context: String::new(), prompt, images: vec![], max_tokens, think_budget: None,
            sampling: Sampling { temperature: 0.0, reasoning_repeat_penalty: 1.0, reasoning_repeat_last_n: 0, repeat_penalty: 1.0, repeat_last_n: 0,
                presence_penalty: 0.0, frequency_penalty: 0.0, top_p: 1.0, top_k: 0, seed: 1 },
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(cancel)), events, forget, session: session.map(str::to_owned), wipe,
        };
        self.tx.as_ref().unwrap().send(job).unwrap();
        let mut reply = Reply::default();
        let mut text = String::new();
        // The engine drops the job (and so the sender) when it is done with it: the loop ends.
        for event in rx {
            match event {
                Event::CacheReuse { cached, source, common } => reply.reuse = Some((cached, source, common)),
                Event::Text(t) => text.push_str(&t),
                Event::Error(e) => reply.error = Some(e),
                Event::Done { .. } => reply.done = true,
                _ => {}
            }
        }
        reply.words = text.split('.').filter(|w| !w.is_empty()).map(|w| w.parse().unwrap()).collect();
        reply
    }

    pub fn ask(&self, prompt: &[u32], max_tokens: usize) -> Reply { self.send(prompt.to_vec(), false, None, false, false, max_tokens) }
    pub fn ask_private(&self, prompt: &[u32], session: Option<&str>, max_tokens: usize) -> Reply { self.send(prompt.to_vec(), true, session, false, false, max_tokens) }
    pub fn ask_cancelled(&self, prompt: &[u32]) -> Reply { self.send(prompt.to_vec(), false, None, false, true, 4) }
    pub fn wipe(&self, session: &str) { self.send(vec![9], true, Some(session), true, false, 1); }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(t) = self.thread.take() { let _ = t.join(); }
    }
}

/// A conversation: the 520-token preamble every conversation starts with, 300 tokens of its own, then
/// messages that begin at message boundaries.
pub(crate) struct Chat { pub id: u32, pub shared_system: bool, pub messages: Vec<Vec<u32>> }

impl Chat {
    pub fn new(id: u32, first: usize) -> Self { Chat { id, shared_system: false, messages: vec![Self::message(id, 1, first)] } }
    pub fn sharing(id: u32, first: usize) -> Self { Chat { shared_system: true, ..Self::new(id, first) } }
    fn message(id: u32, turn: u32, n: usize) -> Vec<u32> {
        std::iter::once(IM).chain((1..n as u32).map(|i| 1_000 + ((id << 20) + (turn << 16) + i) % 90_000)).collect()
    }
    /// The assistant's turn as a later prompt renders it, and the user's next message.
    pub fn answered(&mut self, reply: &[u32], extra: Option<u32>, next: usize) {
        let mut turn = vec![IM, 2];
        turn.extend(extra);
        turn.extend(reply);
        turn.push(3);
        self.messages.push(turn);
        let n = self.messages.len() as u32;
        self.messages.push(Self::message(self.id, n, next));
    }
    pub fn prompt(&self) -> Vec<u32> {
        let mut p: Vec<u32> = std::iter::once(IM).chain(5..524).collect();
        let own = if self.shared_system { 0 } else { self.id };
        p.extend((0..300).map(|i| 1_000 + ((own << 20) + (1 << 19) + i) % 90_000));
        for m in &self.messages { p.extend(m); }
        p.extend([IM, 2]);
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX: usize = 8;
    const BIG: usize = 1 << 40;

    /// The prompt for the conversation's next turn, rendered as the engine's own reply would be (the
    /// reply's tokens, then the end of the turn).
    fn next_turn(chat: &mut Chat, reply: &Reply, next: usize) -> Vec<u32> {
        chat.answered(&reply.words, None, next);
        chat.prompt()
    }

    /// The same conversations on an engine that sets nothing aside: what the answers must be.
    fn without_parking(script: &dyn Fn(&Engine) -> Vec<Reply>) -> Vec<Reply> {
        script(&Engine::start(0, None))
    }

    fn two_conversations(e: &Engine) -> Vec<Reply> {
        let (mut a, mut b) = (Chat::new(10, 3_000), Chat::new(20, 1_500));
        let mut out = Vec::new();
        for round in 0..3 {
            for chat in [&mut a, &mut b] {
                let prompt = chat.prompt();
                let reply = e.ask(&prompt, MAX);
                if round < 2 { chat.answered(&reply.words, None, 200); }
                out.push(reply);
            }
        }
        out
    }

    #[test]
    fn the_real_engine_brings_a_conversation_back_from_ram_and_answers_as_it_would_have() {
        let want = without_parking(&two_conversations);
        let e = Engine::start(BIG, None);
        let got = two_conversations(&e);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.words, w.words, "request {i}: what the engine answers must not depend on the stash");
            assert!(g.error.is_none() && g.done);
        }
        // Round 0 reads everything (the first conversation is parked as the second displaces it); rounds 1 and 2 are swap-ins.
        let sources: Vec<_> = got.iter().map(|r| r.reuse.unwrap().1).collect();
        assert_eq!(sources, ["none", "none", "ram", "ram", "ram", "ram"], "{got:?}");
        assert!(want.iter().all(|r| r.reuse.unwrap().1 != "ram"));
        // Only what is new is read: the end of the reply, the next message, the header.
        for r in &got[2..] { assert!(r.reuse.unwrap().0 > 1_000, "{r:?}"); }
    }

    #[test]
    fn shared_system_prompt_ending_at_a_boundary_is_set_aside_as_a_copy_in_the_real_engine() {
        let script = |e: &Engine| {
            let (mut a, mut b) = (Chat::sharing(10, 3_000), Chat::sharing(20, 1_500));
            let mut out = Vec::new();
            for round in 0..3 {
                for chat in [&mut a, &mut b] {
                    let r = e.ask(&chat.prompt(), MAX);
                    if round < 2 { chat.answered(&r.words, None, 200); }
                    out.push(r);
                }
            }
            out
        };
        let want = without_parking(&script);
        let e = Engine::start(BIG, None);
        let got = script(&e);
        for (g, w) in got.iter().zip(&want) { assert_eq!(g.words, w.words); }
        let sources: Vec<_> = got.iter().map(|r| r.reuse.unwrap().1).collect();
        assert_eq!(sources, ["none", "checkpoint", "ram", "ram", "ram", "ram"], "{got:?}");
    }

    #[test]
    fn an_incognito_request_takes_nothing_from_the_stash_and_empties_it() {
        let e = Engine::start(BIG, None);
        let (mut a, mut b) = (Chat::new(10, 3_000), Chat::new(20, 1_500));
        let ra = e.ask(&a.prompt(), MAX);
        let rb = e.ask(&b.prompt(), MAX);
        assert_eq!((ra.reuse.unwrap().1, rb.reuse.unwrap().1), ("none", "none"));
        let next_a = next_turn(&mut a, &ra, 200);
        // The incognito request that continues A gets nothing from the stash...
        let p = e.ask_private(&next_a, None, MAX);
        assert_eq!((p.reuse.unwrap().0, p.reuse.unwrap().1), (0, "none"), "{p:?}");
        // ...and the engine holds nothing afterwards: A is read from nothing.
        let again = e.ask(&next_a, MAX);
        assert_eq!(again.reuse.unwrap().1, "none", "{again:?}");
        assert_eq!(again.words, p.words);
        let _ = (&mut b, rb);
    }

    #[test]
    fn a_private_session_is_never_set_aside_nor_restored_into_a_normal_request() {
        let e = Engine::start(BIG, None);
        let mut p = Chat::new(30, 3_000);
        let r1 = e.ask_private(&p.prompt(), Some("s"), MAX);
        let p2 = next_turn(&mut p, &r1, 200);
        // The session continues from its own state in memory...
        let r2 = e.ask_private(&p2, Some("s"), MAX);
        assert_eq!(r2.reuse.unwrap().1, "memory", "{r2:?}");
        // ...a big normal request displaces it (and wipes it: nothing of it is set aside)...
        let big = Chat::new(40, 3_000);
        e.ask(&big.prompt(), MAX);
        // ...so that neither the session nor anyone sending its prompt finds it again.
        let p3 = next_turn(&mut p, &r2, 200);
        let n = e.ask(&p3, MAX);
        assert_eq!(n.reuse.unwrap().1, "none", "{n:?}");
    }

    #[test]
    fn a_request_the_device_fails_on_leaves_the_stash_and_the_next_requests_sound() {
        const BOOM: u32 = 77_777;
        let e = Engine::start(BIG, Some(BOOM));
        let (mut a, b) = (Chat::new(10, 3_000), Chat::new(20, 1_500));
        let ra = e.ask(&a.prompt(), MAX);
        e.ask(&b.prompt(), MAX);
        // A is set aside. A request that dies in the middle of its prefill...
        let mut poison = Chat::new(50, 2_000);
        poison.messages[0][1_500] = BOOM;
        let dead = e.ask(&poison.prompt(), MAX);
        assert!(dead.error.is_some() && !dead.done, "{dead:?}");
        // ...does not take the stash with it: A comes back from RAM.
        let next_a = next_turn(&mut a, &ra, 200);
        let back = e.ask(&next_a, MAX);
        assert_eq!(back.reuse.unwrap().1, "ram", "{back:?}");
        let cold = Engine::start(0, None).ask(&next_a, MAX);
        assert_eq!(back.words, cold.words);
    }

    #[test]
    fn a_cancelled_request_leaves_a_state_that_is_sound() {
        let e = Engine::start(BIG, None);
        let (mut a, b) = (Chat::new(10, 3_000), Chat::new(20, 1_500));
        let ra = e.ask(&a.prompt(), MAX);
        e.ask(&b.prompt(), MAX);
        let next_a = next_turn(&mut a, &ra, 200);
        // A's request is cancelled before it starts: it may swap A in and stop.
        e.ask_cancelled(&next_a);
        let back = e.ask(&next_a, MAX);
        let cold = Engine::start(0, None).ask(&next_a, MAX);
        assert_eq!(back.words, cold.words);
        assert_ne!(back.reuse.unwrap().1, "none", "{back:?}");
    }

    fn fuzz_seeds(default: u64) -> u64 { std::env::var("PARK_FUZZ_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(default) }

    /// A small deterministic generator.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
            (self.0 >> 11) % n
        }
    }

    /// Random requests over a few conversations (continued, re-rendered, retried, edited, tiny, incognito,
    /// wiped, cancelled, failing): what the engine answers must be what an engine that sets nothing aside
    /// answers, whatever the stash held and whatever it did.
    fn random_script(seed: u64, steps: usize, e: &Engine, log: &mut Vec<String>, boom: u32) -> Vec<Vec<u32>> {
        let mut rng = Rng(seed * 0x9E37_79B9 + 1);
        let sizes = [2_800usize, 1_300, 2_100, 1_700];
        let mut chats: Vec<Chat> = (0..4).map(|i| if i % 2 == 0 { Chat::new(10 + i as u32, sizes[i]) } else { Chat::sharing(10 + i as u32, sizes[i]) }).collect();
        let mut last: Vec<Option<Reply>> = vec![None; 4];
        let mut answers = Vec::new();
        for step in 0..steps {
            let c = rng.next(4) as usize;
            let what = rng.next(100);
            let chat = &mut chats[c];
            let (prompt, kind) = match (what, last[c].clone()) {
                (0..=44, Some(r)) => { chat.answered(&r.words, None, 150 + rng.next(300) as usize); (chat.prompt(), "continue") }
                (45..=54, Some(r)) => { chat.answered(&r.words, Some(9), 150 + rng.next(300) as usize); (chat.prompt(), "rerender") }
                (55..=62, Some(_)) => (chat.prompt(), "retry"),
                (63..=70, Some(_)) => {
                    // edit a message somewhere in the middle
                    let m = rng.next(chat.messages.len() as u64) as usize;
                    let at = 1 + rng.next(chat.messages[m].len() as u64 - 1) as usize;
                    chat.messages[m][at] = 2_000 + rng.next(1_000) as u32;
                    (chat.prompt(), "edit")
                }
                (71..=78, _) => ((0..300 + rng.next(600) as u32).map(|i| if i == 0 { IM } else { 50_000 + i }).collect(), "tiny"),
                (79..=84, _) => (chat.prompt(), "private"),
                (85..=87, _) => (chat.prompt(), "private-session"),
                (88..=89, _) => (vec![], "wipe"),
                (90..=92, _) => (chat.prompt(), "cancelled"),
                (93..=94, _) => { let mut p = chat.prompt(); let n = p.len(); p[n / 2] = boom; (p, "boom") }
                _ => (chat.prompt(), "again"),
            };
            let reply = match kind {
                "private" => e.ask_private(&prompt, None, MAX),
                "private-session" => e.ask_private(&prompt, Some("s1"), MAX),
                "wipe" => { e.wipe("s1"); Reply::default() }
                "cancelled" => e.ask_cancelled(&prompt),
                _ => e.ask(&prompt, MAX),
            };
            log.push(format!("{step}: chat {c} {kind} len {} -> {:?} err={:?}", prompt.len(), reply.reuse, reply.error));
            if kind == "boom" || kind == "wipe" || kind == "cancelled" { answers.push(vec![reply.error.is_some() as u32, reply.done as u32]); continue; }
            if reply.error.is_none() && reply.done { last[c] = Some(reply.clone()); }
            answers.push(reply.words);
        }
        answers
    }

    #[test]
    fn what_the_real_engine_answers_does_not_depend_on_what_it_set_aside() {
        const BOOM: u32 = 77_777;
        let mut sources = std::collections::BTreeMap::<&str, usize>::new();
        for seed in 1..=fuzz_seeds(24) {
            let (mut plain_log, mut log) = (Vec::new(), Vec::new());
            let plain = random_script(seed, 70, &Engine::start(0, Some(BOOM)), &mut plain_log, BOOM);
            let e = Engine::start(BIG, Some(BOOM));
            let parked = random_script(seed, 70, &e, &mut log, BOOM);
            for (i, (p, q)) in plain.iter().zip(&parked).enumerate() {
                assert_eq!(p, q, "seed {seed} step {i}\nplain:  {}\nparked: {}\n{}", plain_log[i], log[i], log.join("\n"));
            }
            for l in &log { if let Some(s) = l.split("-> Some((").nth(1) { let src = s.split('"').nth(1).unwrap_or("?"); *sources.entry(match src { "ram" => "ram", "memory" => "memory", "checkpoint" => "checkpoint", _ => "none" }).or_default() += 1; } }
        }
        eprintln!("sources over the random scripts: {sources:?}");
        assert!(sources["ram"] > 20, "{sources:?}: the scripts never exercised the stash");
    }

    #[test]
    fn a_small_budget_evicts_without_changing_any_answer() {
        const BOOM: u32 = 77_777;
        for seed in 1..=fuzz_seeds(8) {
            let (mut plain_log, mut log) = (Vec::new(), Vec::new());
            let plain = random_script(seed, 70, &Engine::start(0, Some(BOOM)), &mut plain_log, BOOM);
            // Room for about two of the conversations (each is 1.5-3k tokens, 3-4 MB in the fake's small cache).
            let tight = random_script(seed, 70, &Engine::start(300_000, Some(BOOM)), &mut log, BOOM);
            assert_eq!(plain, tight, "seed {seed}\n{}", log.join("\n"));
        }
    }
}
