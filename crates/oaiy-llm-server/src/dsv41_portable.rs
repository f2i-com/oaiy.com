//! DeepSeek-V4.1 without CUDA: the CPU model (`dsv41::model`, the reference the CUDA path is tested against) with its
//! dense trunk on the WebGPU adapter. Measured on the CPU alone, a prompt is compute-bound and its attention (the fp8
//! trunk's projections among it) costs more than its experts, and a warm decode step spends a flat second there
//! (docs/DEEPSEEK_V41.md, "The CPU model, measured"): so the trunk goes to the GPU, every fp8 and bf16 matrix the
//! budget holds (`ggml_rs_wgpu::dense`), the activation still quantized and the result still rounded by the CPU model
//! as the reference does. The routed experts stay on the CPU's tiers.

use std::sync::Arc;

use dsv41::linear::{DenseKernel, Weight};
use ggml_rs_wgpu::dense::{DenseData, DenseGpu};
use ggml_rs_wgpu::WgpuBackend;

/// A dense weight on the WebGPU adapter, as the CPU model's [`DenseKernel`].
struct WgpuDense(Arc<DenseGpu>);

impl DenseKernel for WgpuDense {
    fn forward_rows(&self, x: &[f32], t: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
        self.0.forward(x, t, rows)
    }
}

/// The tokens one forward pass reads of a prompt. A pass reads each layer's experts once for all its tokens, so a
/// prompt is read whole when it fits (an Agent's is 1.5-3k tokens): in pieces it would read them once a piece.
const PREFILL_CHUNK: usize = 4096;

/// The portable server's DeepSeek worker: jobs in, events out, the contract of the CUDA engine (`crate::engine`) and
/// GLM's (`crate::glm`). A request that extends what the model's state covers reads only the new tail, in one chunk
/// (`dsv41`'s attention continues a sequence by a chunk as it would token by token). That state has moved on through
/// the last reply, which the next turn's prompt writes again its own way (its chat template), so the state as the last
/// prompt left it is kept too, a token short of its end (so the same prompt again can still use it): a conversation's
/// next turn continues from there. A prompt that shares neither starts again. A reply ends at the end-of-sentence
/// token, at a complete tool call, or at `max_tokens`.
pub(crate) struct Engine {
    model: dsv41::model::Model,
    tok: Arc<dsv41::tokenizer::Tokenizer>,
    /// The token ids the model's state covers, in order.
    covered: Vec<u32>,
    /// The state after the last prompt's tokens but its last one, and those tokens.
    checkpoint: Option<(Vec<u32>, dsv41::model::Checkpoint)>,
    eos: u32,
    log: bool,
}

/// How much of `prompt` a state covering `tokens` can serve: all of `tokens`, when the prompt starts with them and goes
/// on past them (a state cannot be rewound, and the next token's logits need a forward pass); else none.
fn reusable(tokens: &[u32], prompt: &[u32]) -> usize {
    let n = tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    if n == tokens.len() && n < prompt.len() {
        n
    } else {
        0
    }
}

impl Engine {
    pub(crate) fn new(model: dsv41::model::Model, tok: Arc<dsv41::tokenizer::Tokenizer>, log: bool) -> Engine {
        let eos = model.cfg.eos_token_id;
        Engine { model, tok, covered: Vec::new(), checkpoint: None, eos, log }
    }

    pub(crate) fn run(mut self, jobs: std::sync::mpsc::Receiver<crate::engine::Job>) {
        use crate::engine::{Event, Finish};
        for job in jobs {
            if job.wipe {
                self.covered.clear();
                self.checkpoint = None;
                let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                continue;
            }
            if let Err(e) = self.generate(&job) {
                // The state's extent is unknown after a failure: start clean.
                self.covered.clear();
                self.checkpoint = None;
                let _ = job.events.send(Event::Error(e.to_string()));
            }
            // Incognito: nothing of it is kept for the next request.
            if job.forget {
                self.covered.clear();
                self.checkpoint = None;
            }
        }
    }

    fn generate(&mut self, job: &crate::engine::Job) -> oaiy_engine::Result<()> {
        use crate::engine::{sample, Event, Finish};
        use std::sync::atomic::Ordering;
        if !job.images.is_empty() {
            return Err(oaiy_engine::Error::Arg("DeepSeek-V4.1 takes images in the CUDA build only".into()));
        }
        let prompt = &job.prompt;
        let max_seq = self.model.max_seq();
        if prompt.is_empty() || prompt.len() >= max_seq {
            return Err(oaiy_engine::Error::Arg(format!("a prompt of {} tokens does not fit a context of {max_seq}", prompt.len())));
        }
        let started = std::time::Instant::now();
        let common = self.covered.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        // The live state, or the last prompt's checkpoint: whichever serves more of this prompt.
        let live = reusable(&self.covered, prompt);
        let kept = self.checkpoint.as_ref().map_or(0, |(tokens, _)| reusable(tokens, prompt));
        let start = if kept > live {
            let (tokens, state) = self.checkpoint.as_ref().expect("a checkpoint serves it");
            self.model.restore(state);
            self.covered.clone_from(tokens);
            kept
        } else {
            live
        };
        if start == 0 {
            self.covered.clear();
        }
        let _ = job.events.send(Event::CacheReuse { cached: start, source: if start > 0 { "memory" } else { "none" }, common });
        let total = prompt.len() - start;
        let _ = job.events.send(Event::Progress { done: 0, total });
        // All but the last token, then the checkpoint, then the last one (whose logits the reply starts from).
        let last = prompt.len() - 1;
        let mut pos = start;
        while pos < last {
            if job.cancel.load(Ordering::Relaxed) {
                let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                return Ok(());
            }
            let take = PREFILL_CHUNK.min(last - pos);
            self.model.forward(&prompt[pos..pos + take], pos, &mut |_, _| {})?;
            self.covered.extend_from_slice(&prompt[pos..pos + take]);
            pos += take;
            let _ = job.events.send(Event::Progress { done: pos - start, total });
        }
        if !job.forget {
            self.checkpoint = Some((self.covered.clone(), self.model.checkpoint()));
        }
        let mut logits = self.model.forward(&prompt[last..], last, &mut |_, _| {})?;
        self.covered.push(prompt[last]);
        pos = prompt.len();
        let _ = job.events.send(Event::Progress { done: total, total });
        let prefill_s = started.elapsed().as_secs_f64();
        let _ = job.events.send(Event::Prefilled { cached: start });

        let think_start = self.tok.special(dsv41::chat::THINK_START);
        let mut budget = crate::glm::ThinkBudget::new(think_start.is_some() && prompt.last().copied() == think_start, self.tok.special(dsv41::chat::THINK_END), job.think_budget);
        let mut parser = dsv41::chat::StreamParser::new(if budget.thinking { dsv41::chat::Mode::Thinking } else { dsv41::chat::Mode::Chat });
        let mut rng = job.sampling.seed ^ 0x9E37_79B9_7F4A_7C15;
        let (mut pending, mut n) = (Vec::<u8>::new(), 0usize);
        let decode = std::time::Instant::now();
        let finish = loop {
            let was_thinking = budget.thinking;
            let (next, _forced) = budget.pass(sample(&logits, &job.sampling, &mut rng));
            if was_thinking && (!budget.thinking || budget.used.is_multiple_of(crate::glm::THINKING_EVERY)) {
                let _ = job.events.send(Event::Thinking { used: budget.used, budget: budget.budget, done: !budget.thinking });
            }
            if next == self.eos {
                break Finish::Stop;
            }
            n += 1;
            // Whole UTF-8 characters only: a token can be part of one.
            pending.extend_from_slice(self.tok.token_bytes(next));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                parser.push(&text);
                if job.events.send(Event::Text(text)).is_err() {
                    break Finish::Stop; // nobody is listening
                }
            }
            // A complete tool call is the reply: the caller runs it and comes back with its result.
            if parser.tool_calls_ready() {
                break Finish::Stop;
            }
            if n >= job.max_tokens || pos + 1 >= max_seq {
                break Finish::Length;
            }
            if job.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            logits = self.model.forward(&[next], pos, &mut |_, _| {})?;
            self.covered.push(next);
            pos += 1;
        };
        if !pending.is_empty() {
            let _ = job.events.send(Event::Text(String::from_utf8_lossy(&pending).into_owned()));
        }
        if self.log {
            let decode_s = decode.elapsed().as_secs_f64();
            eprintln!("  {} prompt tokens ({start} reused) in {prefill_s:.1}s; {n} generated in {decode_s:.1}s ({:.2} tok/s)", prompt.len(), n as f64 / decode_s.max(1e-9));
        }
        let _ = job.events.send(Event::Done { finish, completion_tokens: n });
        Ok(())
    }
}

/// A prompt's busy routed experts on the WebGPU adapter (`dsv41::moe::Experts::gpu`): a group's records uploaded for
/// the call (MXFP4, their scales as f32), every expert's gate and up in one submit, the SwiGLU on the host as dsv41
/// takes it, then every down in another. An expert the budget does not hold is computed on the CPU instead. Measured on
/// the CPU model, a prompt's MoE was its matmuls far more than its reads (docs/DEEPSEEK_V41.md).
pub(crate) struct WgpuExperts(pub(crate) Arc<dyn ggml_rs::Backend>);

impl dsv41::expert::ExpertsKernel for WgpuExperts {
    fn forward(&self, jobs: &[dsv41::expert::ExpertJob<'_>], swiglu_limit: f32) -> Vec<Vec<f32>> {
        use dsv41::expert::{BLOCK, DIM, INTER, S1, S2, S3, W1, W2, W3};
        use dsv41::formats::{e8m0_to_f32, fake_quant_fp8, to_bf16};
        let Some(b) = self.0.as_any().downcast_ref::<WgpuBackend>() else {
            return jobs.iter().map(|j| dsv41::expert::expert_forward_batch(j.record, j.x, Some(j.weights), swiglu_limit)).collect();
        };
        let mx = |rec: &[u8], w: std::ops::Range<usize>, s: std::ops::Range<usize>, n: usize, k: usize| DenseData::Mxfp4 {
            w: rec[w].to_vec(),
            scales: rec[s].iter().map(|&x| e8m0_to_f32(x)).collect(),
            n,
            k,
        };
        let rows: Vec<usize> = jobs.iter().map(|j| j.x.len() / DIM).collect();
        let mut out: Vec<Option<Vec<f32>>> = (0..jobs.len()).map(|_| None).collect();
        // Gate and up, every expert's in one submit.
        let gate_up: Vec<Option<(Arc<DenseGpu>, Arc<DenseGpu>)>> = jobs
            .iter()
            .map(|j| Some((b.dense(mx(j.record, W1, S1, INTER, DIM)).ok().flatten()?, b.dense(mx(j.record, W3, S3, INTER, DIM)).ok().flatten()?)))
            .collect();
        let xq: Vec<Vec<f32>> = jobs.iter().zip(&gate_up).map(|(j, g)| if g.is_some() { fake_quant_fp8(j.x, BLOCK) } else { Vec::new() }).collect();
        let mut items = Vec::new();
        for (i, g) in gate_up.iter().enumerate() {
            if let Some((gate, up)) = g {
                items.push((&**gate, xq[i].as_slice(), rows[i], 0..INTER));
                items.push((&**up, xq[i].as_slice(), rows[i], 0..INTER));
            }
        }
        let mut sums = ggml_rs_wgpu::dense::forward_batch(&items).into_iter();
        let mut hq: Vec<Option<Vec<f32>>> = (0..jobs.len()).map(|_| None).collect();
        for (i, g) in gate_up.iter().enumerate() {
            if g.is_some() {
                let (gate, up) = (sums.next().expect("a gate"), sums.next().expect("an up"));
                hq[i] = Some(fake_quant_fp8(&dsv41::expert::swiglu(&gate, &up, Some(jobs[i].weights), swiglu_limit), BLOCK));
            }
        }
        drop(items);
        drop(gate_up);
        // Then every down, in another.
        let downs: Vec<Option<Arc<DenseGpu>>> = jobs.iter().zip(&hq).map(|(j, h)| h.as_ref().and_then(|_| b.dense(mx(j.record, W2, S2, DIM, INTER)).ok().flatten())).collect();
        let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> =
            downs.iter().enumerate().filter_map(|(i, d)| Some((&**d.as_ref()?, hq[i].as_deref()?, rows[i], 0..DIM))).collect();
        let mut sums = ggml_rs_wgpu::dense::forward_batch(&items).into_iter();
        for (i, d) in downs.iter().enumerate() {
            if d.is_some() {
                out[i] = Some(sums.next().expect("a down").into_iter().map(to_bf16).collect());
            }
        }
        // What the budget did not hold, on the CPU.
        out.into_iter()
            .zip(jobs)
            .map(|(o, j)| o.unwrap_or_else(|| dsv41::expert::expert_forward_batch(j.record, j.x, Some(j.weights), swiglu_limit)))
            .collect()
    }
}

/// Put `model`'s dense trunk on `gpu` while its budget holds it: how many matrices went, and their stored bytes.
pub(crate) fn offload(model: &mut dsv41::model::Model, gpu: &WgpuBackend) -> (usize, u64) {
    model.offload(|_name, w| {
        let (data, n, k, fp8) = match w {
            Weight::Fp8 { w, s, n, k } => {
                let scales = s.iter().map(|&b| dsv41::formats::e8m0_to_f32(b)).collect();
                (DenseData::Fp8 { w: w.clone(), scales, n: *n, k: *k }, *n, *k, true)
            }
            Weight::Bf16 { w, n, k } => (DenseData::Bf16 { w: w.clone(), n: *n, k: *k }, *n, *k, false),
            _ => return None,
        };
        let placed = gpu.dense(data).ok().flatten()?;
        Some(Weight::Device { kernel: Arc::new(WgpuDense(placed)), n, k, fp8 })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsv41::model::{Model, ModelOptions};
    use std::path::PathBuf;
    use std::time::Instant;

    fn checkpoint() -> PathBuf {
        std::env::var_os("DSV41_MODEL").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"D:\deepseek\model"))
    }

    fn prompt(dir: &std::path::Path, tokens: usize) -> Vec<u32> {
        let tok = dsv41::tokenizer::Tokenizer::load(dir).unwrap();
        let docs = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs");
        let text: String = ["WEBGPU.md", "STUDIO.md", "FLASHNEXT.md", "ORCASAQ.md"].iter().filter_map(|f| std::fs::read_to_string(format!("{docs}/{f}")).ok()).collect();
        let mut ids = vec![0u32];
        ids.extend(tok.encode(&text));
        ids.truncate(tokens);
        ids
    }

    /// An expert on the GPU gives the CPU's output but for the order of its sums: on a random record (realistic scales)
    /// and 33 rows, nearly every output is the same bf16 value, and the rest a few bf16 steps away (an fp8 rounding of
    /// the SwiGLU's output can go the other way), never a different answer.
    #[test]
    fn an_expert_on_the_gpu_gives_the_cpus_output_but_for_its_sums_order() {
        use dsv41::expert::{expert_forward_batch, ExpertJob, ExpertsKernel, DIM, RECORD_BYTES, S1, W1};
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut record = vec![0u8; RECORD_BYTES];
        for (i, byte) in record.iter_mut().enumerate() {
            // Weights: any nibbles. Scales (e8m0): 2^-9 .. 2^-6, as a trained expert's are.
            *byte = if i < S1.start { (next() & 255) as u8 } else { 118 + (next() % 4) as u8 };
        }
        assert!(W1.start == 0);
        let rows = 33;
        let x: Vec<f32> = (0..rows * DIM).map(|_| (((next() % 2001) as f32 / 1000.0) - 1.0) * 0.5).collect();
        let weights: Vec<f32> = (0..rows).map(|i| 0.05 + 0.01 * i as f32).collect();
        let cpu = expert_forward_batch(&record, &x, Some(&weights), 10.0);
        let gpu = WgpuExperts(Arc::new(b)).forward(&[ExpertJob { record: &record, x: &x, weights: &weights }], 10.0).pop().unwrap();
        assert_eq!(cpu.len(), gpu.len());
        let same = cpu.iter().zip(&gpu).filter(|(a, b)| a == b).count();
        let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
        eprintln!("{same} of {} outputs the same; the largest difference {worst:.2e} of the largest output", cpu.len());
        assert!(same as f64 >= 0.9 * cpu.len() as f64, "{same} of {}", cpu.len());
        assert!(worst < 0.05, "{worst}");
    }

    fn gpu() -> WgpuBackend {
        let gb: u64 = std::env::var("DSV41_WEBGPU_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(27);
        WgpuBackend::new(Some(gb << 30)).expect("a WebGPU adapter")
    }

    /// The trunk and a prompt's busy experts on the GPU give the CPU model's answer: on a prompt long enough for experts
    /// to be busy (400 tokens) the last position's logits agree closely, and the next greedy tokens are the same.
    #[test]
    #[ignore = "loads the 510 GB checkpoint twice; needs a WebGPU adapter"]
    fn the_trunk_and_the_experts_on_the_gpu_answer_as_the_cpu_model_does() {
        let dir = checkpoint();
        let meta = dir.join("engram_meta.safetensors");
        let ids = prompt(&dir, 400);
        let opts = ModelOptions { max_seq: 512, expert_cache_bytes: 24 << 30, direct_io: true };
        let run = |model: &mut Model| {
            let mut logits = model.forward(&ids, 0, &mut |_, _| {}).unwrap();
            let first = logits.clone();
            let mut tokens = Vec::new();
            for step in 0..6 {
                let next = dsv41::model::argmax(&logits);
                tokens.push(next);
                logits = model.forward(&[next], ids.len() + step, &mut |_, _| {}).unwrap();
            }
            (first, tokens)
        };
        let (cpu_logits, cpu_tokens) = run(&mut Model::load(&dir, &meta, &opts).unwrap());
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(gpu());
        let mut model = Model::load(&dir, &meta, &opts).unwrap();
        let (count, bytes) = offload(&mut model, backend.as_any().downcast_ref::<WgpuBackend>().unwrap());
        model.set_experts_kernel(Some(Arc::new(WgpuExperts(Arc::clone(&backend)))));
        eprintln!("{count} dense matrices on the GPU, {:.2} GB, and a prompt's busy experts", bytes as f64 / 1e9);
        let (gpu_logits, gpu_tokens) = run(&mut model);
        let dot: f64 = cpu_logits.iter().zip(&gpu_logits).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm(&cpu_logits) * norm(&gpu_logits));
        let max_diff = cpu_logits.iter().zip(&gpu_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("logits: cosine {cosine:.6}, largest difference {max_diff:.4}; tokens CPU {cpu_tokens:?}, GPU {gpu_tokens:?}");
        assert!(count > 0);
        // The sums' order differs on the GPU: an expert's outputs are the CPU's to the bit but for about one in 170,000
        // (`an_expert_on_the_gpu_gives_the_cpus_output_but_for_its_sums_order`), and those few, through the fp8
        // roundings of 40 layers and a 400-token prompt's experts, leave the logits close, not equal (measured 2026-10-05:
        // cosine 0.99882, the same six tokens; the trunk alone, cosine 1.000000).
        assert!(cosine > 0.998, "cosine {cosine}");
        assert_eq!(cpu_tokens, gpu_tokens);
    }

    /// The CPU model's measurement (dsv41/tests/cpu_speed.rs) with the trunk on the GPU.
    #[test]
    #[ignore = "loads the 510 GB checkpoint and prefills a 2,000-token prompt; needs a WebGPU adapter"]
    fn measure_prefill_and_warm_decode_with_the_trunk_on_the_gpu() {
        use std::cell::RefCell;
        use std::collections::BTreeMap;
        let dir = checkpoint();
        let meta = dir.join("engram_meta.safetensors");
        let tokens: usize = std::env::var("DSV41_PROMPT_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
        let steps: usize = std::env::var("DSV41_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        let gb: usize = std::env::var("DSV41_CACHE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(96);
        let ids = prompt(&dir, tokens);
        let opts = ModelOptions { max_seq: ids.len() + steps + 8, expert_cache_bytes: gb << 30, direct_io: true };
        let mut model = Model::load(&dir, &meta, &opts).unwrap();
        let backend: Arc<dyn ggml_rs::Backend> = Arc::new(gpu());
        let t = Instant::now();
        let (count, bytes) = offload(&mut model, backend.as_any().downcast_ref::<WgpuBackend>().unwrap());
        eprintln!("{count} dense matrices on the GPU, {:.2} GB, in {:.1} s", bytes as f64 / 1e9, t.elapsed().as_secs_f64());
        if std::env::var("DSV41_GPU_EXPERTS").map_or(true, |v| v != "0") {
            model.set_experts_kernel(Some(Arc::new(WgpuExperts(Arc::clone(&backend)))));
            eprintln!("a prompt's busy experts on the GPU");
        }
        let spans: RefCell<BTreeMap<&'static str, f64>> = RefCell::new(BTreeMap::new());
        let last = RefCell::new(Instant::now());
        let mut trace = |name: &str, _: &[f32]| {
            let now = Instant::now();
            let what = if name.ends_with(".attn_out") { "attention" } else if name.ends_with(".moe_out") { "moe" } else { "other" };
            *spans.borrow_mut().entry(what).or_default() += now.duration_since(*last.borrow()).as_secs_f64();
            *last.borrow_mut() = now;
        };
        let report = |label: &str, secs: f64, before: oaiy_engine::ecache::CacheStats, after: oaiy_engine::ecache::CacheStats, spans: &BTreeMap<&'static str, f64>| {
            let read = after.bytes_read - before.bytes_read;
            eprintln!(
                "{label}: {secs:.2} s; read {:.2} GB, {} hits / {} misses; {}
    {}",
                read as f64 / 1e9,
                after.hits - before.hits,
                after.misses - before.misses,
                spans.iter().map(|(k, v)| format!("{k} {v:.2} s")).collect::<Vec<_>>().join(", "),
                dsv41::profile::take_line()
            );
        };
        let before = model.expert_cache().stats();
        dsv41::profile::take();
        *last.borrow_mut() = Instant::now();
        let t = Instant::now();
        let mut logits = model.forward(&ids, 0, &mut trace).unwrap();
        report(&format!("prompt of {} tokens", ids.len()), t.elapsed().as_secs_f64(), before, model.expert_cache().stats(), &spans.borrow());
        let mut times = Vec::new();
        for step in 0..steps {
            let next = dsv41::model::argmax(&logits);
            spans.borrow_mut().clear();
            let before = model.expert_cache().stats();
            *last.borrow_mut() = Instant::now();
            let t = Instant::now();
            logits = model.forward(&[next], ids.len() + step, &mut trace).unwrap();
            let secs = t.elapsed().as_secs_f64();
            times.push(secs);
            report(&format!("step {step} (token {next})"), secs, before, model.expert_cache().stats(), &spans.borrow());
        }
        let tail = &times[times.len() / 2..];
        eprintln!("last {} steps: {:.2} s a token on average", tail.len(), tail.iter().sum::<f64>() / tail.len() as f64);
    }
}
