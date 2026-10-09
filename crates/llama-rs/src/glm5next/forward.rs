// VENDORED-LOCAL: whole module. GLM-5.3-Flash host reference forward pass.
//! The layer loop: a host-f32 reference that sequences the stages in
//! [`super::hc`], [`super::kda`], [`super::mla`], [`super::indexer`],
//! [`super::kpool`] and [`super::routing`] into a whole forward pass.
//!
//! Reference: llama.cpp `llama_model_glm5next::graph::graph` (the trunk loop),
//! `build_kda_layer`, `build_dsa_layer` and `build_layer_ffn` (PR #27754, pinned
//! at `86ebfef`), plus `llm_graph_context::build_ffn`'s clamped-SwiGLU branch.
//!
//! This is the same methodology `dsv41` used: a CPU reference first, then the
//! device path validated against it. It is **not** fast — one token at a time,
//! dense f32 matvecs — and it is not meant to be. It is the thing a greedy-token
//! comparison against llama.cpp can be run on, and the thing a future
//! `Tensor`/`Backend` implementation gets checked against.
//!
//! ## Per-token, deliberately
//!
//! Every stage module is per-token, the KDA recurrence is definitionally
//! sequential, and the reference's own autoregressive path is what its chunked
//! prefill is validated against. Looping this over a prompt gives prefill for
//! free — slowly, but with the same arithmetic. Batched prefill is an
//! optimisation for later.
//!
//! ## The trunk loop
//!
//! ```text
//! stream = hc::init(embd[token])                  // HC exact copies
//! for il in 0..n_layer:
//!     residual = stream
//!     mix  = hc::mixes(stream, hc_attn)
//!     cur  = rms_norm(hc::collapse(stream, mix.pre), attn_norm)
//!     cur  = kda_layer(cur) | mla_layer(cur)
//!     stream = hc::combine(cur, residual, mix)
//!
//!     residual = stream
//!     mix  = hc::mixes(stream, hc_ffn)
//!     cur  = rms_norm(hc::collapse(stream, mix.pre), ffn_norm)
//!     cur  = dense_ffn(cur) | moe_ffn(cur)
//!     stream = hc::combine(cur, residual, mix)
//!
//! logits = output @ rms_norm(hc::mean(stream), output_norm)
//! ```
//!
//! Note the mixes are derived from the **un-normed** stream — `build_hc_pre`
//! does its own RMSNorm over the flattened `hc * n_embd` vector internally, and
//! `attn_norm` / `ffn_norm` apply only to the collapsed result.
//!
//! The NextN/MTP block (`blk.45`) is not run: it has no `hc_*` mixer and is a
//! draft head, not part of a plain decode.
//!
//! ## Weight layout
//!
//! Every weight is `&[f32]` in the **loader's reversed layout**, so `shape()[0]`
//! is the output dimension — the same convention `Glm5NextModel`'s `want_shape`
//! checks. A 2-D weight `[out, in]` is row-major, so row `o` is
//! `w[o*in..(o+1)*in]` and a matvec is a dot product per row. Dequantisation is
//! the caller's problem; keeping it out makes this module pure and testable on
//! synthetic weights.
//!
//! **Runs on real weights** through [`super::bridge::HostModel`], which loads
//! the non-expert tensors resident as f32 and serves routed experts from the
//! `.gguf` per dispatch. `Glm5NextModel::forward` itself is still the stub: that
//! path wants the device implementation, not this one.

use std::sync::Arc;
use ggml_rs::{Backend, Tensor};

use super::{hc, indexer, kda, kpool, mla, routing, LayerKind};
use crate::loader::Weight;
use crate::{LlamaError, Result};

mod layers;
mod math;
mod mats;
pub mod prof;
mod state;
#[cfg(test)]
mod tests;
mod weights;

// (what this file gave the crate and its users, and what the files make for each other)
pub use {mats::*, math::*, state::*, weights::*};
pub use layers::*;

/// Run one token through the trunk and return its logits, advancing `state`.
///
/// `state.len` must be the token's position, and grows by one on success.
pub fn forward_token(
    sh: &Shape,
    w: &ModelW<'_>,
    state: &mut State,
    token: u32,
) -> Result<Vec<f32>> {
    if hc::HC != sh.hc_count {
        return Err(LlamaError::Config(format!(
            "forward: hyper_connection.count is {} but this build fixes HC at {}",
            sh.hc_count,
            hc::HC
        )));
    }
    if w.layers.len() != sh.n_layer {
        return Err(LlamaError::Config(format!(
            "forward: {} layer weights for an {}-layer trunk",
            w.layers.len(),
            sh.n_layer
        )));
    }
    let pos = state.len;
    if pos >= state.max_len {
        return Err(LlamaError::Config(format!(
            "forward: position {pos} past the cache capacity {}",
            state.max_len
        )));
    }
    let tok = token as usize;
    if tok >= sh.n_vocab {
        return Err(LlamaError::Config(format!(
            "forward: token {tok} outside the {}-entry vocabulary",
            sh.n_vocab
        )));
    }

    let n_embd = sh.n_embd;
    // The residual stream opens as HC exact copies of the embedding: no scaling
    // and no one-hot into stream 0.
    let embd = &w.tok_embd[tok * n_embd..(tok + 1) * n_embd];
    let mut stream = hc::init(embd);

    let mut sub_out = vec![0.0f32; n_embd];
    // Reused across all 45 layers and both halves: the hyper-connection path used
    // to allocate a collapsed vector, a combined stream and a clone of the stream
    // twice a layer -- 270 allocations a token, on the critical path.
    let mut cur = vec![0.0f32; n_embd];
    let mut residual = vec![0.0f32; hc::HC * n_embd];
    let mut next = vec![0.0f32; hc::HC * n_embd];

    for il in 0..sh.n_layer {
        let lw = &w.layers[il];

        // --- attention half -------------------------------------------------
        let t_hc = std::time::Instant::now();
        residual.copy_from_slice(&stream);
        let mix = hc_mixes(&stream, &lw.hc_attn, sh)?;
        hc::collapse_into(&stream, &mix.pre, &mut cur);
        rms_norm(&mut cur, lw.attn_norm, sh.rms_eps)?;
        prof::add(&prof::HC, t_hc);

        match &lw.attn {
            AttnW::Kda(kw) => {
                let t = std::time::Instant::now();
                let ord = sh.kda_ordinal(il);
                // Split the borrow: conv is host, the recurrent state may not be.
                let State { conv, kda, kda_dev, .. } = &mut *state;
                let cst = &mut conv[ord];
                let kst = match kda_dev {
                    Some((dev, backend)) => KdaSt::Device {
                        t: &mut dev[ord],
                        backend: &**backend,
                    },
                    None => KdaSt::Host(&mut kda[ord]),
                };
                kda_layer(sh, kw, kst, cst, &cur, &mut sub_out)?;
                prof::add(&prof::KDA, t);
            }
            AttnW::Mla(mw) => {
                let t = std::time::Instant::now();
                let ord = sh.mla_ordinal(il);
                // Split the borrow: latents and its mirror are per-layer, kpool is
                // shared, and the mirror's backend comes out of the same field.
                let State { latents, latents_dev, kpool, .. } = &mut *state;
                let lat = &mut latents[ord];
                let dev = latents_dev
                    .as_mut()
                    .map(|(t, be)| (&mut t[ord], &**be as &dyn Backend));
                mla_layer(sh, mw, lat, dev, kpool, ord, pos, &cur, &mut sub_out)?;
                prof::add(&prof::MLA, t);
            }
        }
        let t_hc = std::time::Instant::now();
        hc::combine_into(&sub_out, &residual, &mix, &mut next);
        std::mem::swap(&mut stream, &mut next);

        // --- FFN half -------------------------------------------------------
        residual.copy_from_slice(&stream);
        let mix = hc_mixes(&stream, &lw.hc_ffn, sh)?;
        hc::collapse_into(&stream, &mix.pre, &mut cur);
        rms_norm(&mut cur, lw.ffn_norm, sh.rms_eps)?;
        prof::add(&prof::HC, t_hc);

        let t = std::time::Instant::now();
        ffn_layer(sh, &lw.ffn, il, &cur, &mut sub_out)?;
        prof::add(&prof::FFN, t);
        let t_hc = std::time::Instant::now();
        hc::combine_into(&sub_out, &residual, &mix, &mut next);
        std::mem::swap(&mut stream, &mut next);
        prof::add(&prof::HC, t_hc);
    }

    // The trunk collapses with an UNWEIGHTED mean, not DeepSeek-V4.1's learned
    // gated head; glm5next ships no hc_head_* tensors.
    let t = std::time::Instant::now();
    let mut x = hc::mean(&stream);
    rms_norm(&mut x, w.output_norm, sh.rms_eps)?;

    let mut logits = vec![0.0f32; sh.n_vocab];
    w.output.apply(&x, &mut logits)?;
    prof::add(&prof::HEAD, t);

    state.len = pos + 1;
    Ok(logits)
}

/// VENDORED-LOCAL: GLM-5.3-Flash. How many tokens a batched prefill pass holds.
///
/// The chunk decides what is amortised. Bigger means more tokens sharing each
/// expert read, and `chunk * hc::HC * n_embd * 4` bytes of hidden state held at
/// once -- 8 MB at 128 tokens for this model, so the ceiling is not memory. It is
/// that a chunk's routes are a union: past a few hundred tokens nearly all 288
/// experts of every layer are in it, and reading all of them costs more than
/// reading eight per token did.
///
/// GLM5_PREFILL_CHUNK overrides it; 1 turns batching off.
pub fn prefill_chunk() -> usize {
    std::env::var("GLM5_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(64)
}

/// VENDORED-LOCAL: GLM-5.3-Flash. A chunk of tokens, layer by layer.
///
/// [`forward_token`] walks the layers for one token, so a prompt of T tokens walks
/// them T times and reads every layer's routed experts T times over. Prefill
/// therefore costs the same per token as decode -- measured, ~100 ms -- and a
/// coder-cli system prompt of a few thousand tokens takes minutes before the first
/// reply.
///
/// This walks the layers once for the whole chunk. Two things follow:
///
///   * The FFN sees every token's route together, so a layer's distinct experts are
///     resolved and read once for all of them: see [`ExpertFfn::apply_batch`].
///   * The attention stays per token, because it has to. KDA's delta rule is a
///     recurrence -- token t+1's state update needs token t's -- and MLA appends to
///     the latent cache in position order. So they run in order inside the chunk,
///     against the same state the one-token path uses, which is what makes this
///     produce identical logits rather than merely similar ones.
///
/// Returns the **last** token's logits, like [`forward_prompt`], and advances the
/// state by `tokens.len()`.
pub fn forward_chunk(
    sh: &Shape,
    w: &ModelW<'_>,
    state: &mut State,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    if hc::HC != sh.hc_count {
        return Err(LlamaError::Config(format!(
            "forward: hyper_connection.count is {} but this build fixes HC at {}",
            sh.hc_count,
            hc::HC
        )));
    }
    if w.layers.len() != sh.n_layer {
        return Err(LlamaError::Config(format!(
            "forward: {} layer weights for an {}-layer trunk",
            w.layers.len(),
            sh.n_layer
        )));
    }
    if tokens.is_empty() {
        return Err(LlamaError::Config("forward: empty chunk".into()));
    }
    let n = tokens.len();
    let base = state.len;
    if base + n > state.max_len {
        return Err(LlamaError::Config(format!(
            "forward: positions {base}..{} past the cache capacity {}",
            base + n,
            state.max_len
        )));
    }
    let n_embd = sh.n_embd;

    // One hyper-connection stream per token, seeded from its embedding exactly as
    // the one-token path seeds its own.
    let mut streams: Vec<Vec<f32>> = Vec::with_capacity(n);
    for &token in tokens {
        let tok = token as usize;
        if tok >= sh.n_vocab {
            return Err(LlamaError::Config(format!(
                "forward: token {tok} is outside the {}-entry vocabulary",
                sh.n_vocab
            )));
        }
        streams.push(hc::init(&w.tok_embd[tok * n_embd..(tok + 1) * n_embd]));
    }

    let mut cur = vec![0.0f32; n * n_embd];
    let mut sub_out = vec![0.0f32; n * n_embd];
    let mut residual = vec![0.0f32; n * hc::HC * n_embd];
    let mut mixes: Vec<hc::Mix> = Vec::with_capacity(n);
    let mut next = vec![0.0f32; hc::HC * n_embd];

    for (il, lw) in w.layers.iter().enumerate() {
        // --- attention half ----------------------------------------------------
        let t_hc = std::time::Instant::now();
        mixes.clear();
        for t in 0..n {
            let (rlo, clo) = (t * hc::HC * n_embd, t * n_embd);
            residual[rlo..rlo + hc::HC * n_embd].copy_from_slice(&streams[t]);
            let mix = hc_mixes(&streams[t], &lw.hc_attn, sh)?;
            hc::collapse_into(&streams[t], &mix.pre, &mut cur[clo..clo + n_embd]);
            rms_norm(&mut cur[clo..clo + n_embd], lw.attn_norm, sh.rms_eps)?;
            mixes.push(mix);
        }
        prof::add(&prof::HC, t_hc);

        // In position order: both mixers carry state forward from one token to the
        // next, so this is the one part of a chunk that cannot be reordered.
        for t in 0..n {
            let clo = t * n_embd;
            let (x, out) = split_at_chunk(&cur, &mut sub_out, clo, n_embd);
            match &lw.attn {
                AttnW::Kda(kw) => {
                    let tm = std::time::Instant::now();
                    let ord = sh.kda_ordinal(il);
                    let State { conv, kda, kda_dev, .. } = &mut *state;
                    let cst = &mut conv[ord];
                    let kst = match kda_dev {
                        Some((dev, backend)) => KdaSt::Device {
                            t: &mut dev[ord],
                            backend: &**backend,
                        },
                        None => KdaSt::Host(&mut kda[ord]),
                    };
                    kda_layer(sh, kw, kst, cst, x, out)?;
                    prof::add(&prof::KDA, tm);
                }
                AttnW::Mla(mw) => {
                    let tm = std::time::Instant::now();
                    let ord = sh.mla_ordinal(il);
                    let State { latents, latents_dev, kpool, .. } = &mut *state;
                    let lat = &mut latents[ord];
                    let dev = latents_dev
                        .as_mut()
                        .map(|(dt, be)| (&mut dt[ord], &**be as &dyn Backend));
                    mla_layer(sh, mw, lat, dev, kpool, ord, base + t, x, out)?;
                    prof::add(&prof::MLA, tm);
                }
            }
        }

        let t_hc = std::time::Instant::now();
        for t in 0..n {
            let (rlo, clo) = (t * hc::HC * n_embd, t * n_embd);
            hc::combine_into(
                &sub_out[clo..clo + n_embd],
                &residual[rlo..rlo + hc::HC * n_embd],
                &mixes[t],
                &mut next,
            );
            streams[t].copy_from_slice(&next);
        }

        // --- FFN half ----------------------------------------------------------
        mixes.clear();
        for t in 0..n {
            let (rlo, clo) = (t * hc::HC * n_embd, t * n_embd);
            residual[rlo..rlo + hc::HC * n_embd].copy_from_slice(&streams[t]);
            let mix = hc_mixes(&streams[t], &lw.hc_ffn, sh)?;
            hc::collapse_into(&streams[t], &mix.pre, &mut cur[clo..clo + n_embd]);
            rms_norm(&mut cur[clo..clo + n_embd], lw.ffn_norm, sh.rms_eps)?;
            mixes.push(mix);
        }
        prof::add(&prof::HC, t_hc);

        let t = std::time::Instant::now();
        ffn_chunk(sh, &lw.ffn, il, n, &cur, &mut sub_out)?;
        prof::add(&prof::FFN, t);

        let t_hc = std::time::Instant::now();
        for t in 0..n {
            let (rlo, clo) = (t * hc::HC * n_embd, t * n_embd);
            hc::combine_into(
                &sub_out[clo..clo + n_embd],
                &residual[rlo..rlo + hc::HC * n_embd],
                &mixes[t],
                &mut next,
            );
            streams[t].copy_from_slice(&next);
        }
        prof::add(&prof::HC, t_hc);
    }

    // Only the last token's logits are wanted: the head is the single largest
    // matmul there is (vocab x n_embd) and a prompt needs none of the others.
    let t = std::time::Instant::now();
    let mut x = hc::mean(&streams[n - 1]);
    rms_norm(&mut x, w.output_norm, sh.rms_eps)?;
    let mut logits = vec![0.0f32; sh.n_vocab];
    w.output.apply(&x, &mut logits)?;
    prof::add(&prof::HEAD, t);

    state.len = base + n;
    Ok(logits)
}

/// One token's slice of the chunk's input and output buffers.
///
/// A free function because the borrow checker will not take `&cur[..]` and
/// `&mut sub_out[..]` from inside one expression that also borrows `state`.
fn split_at_chunk<'a>(
    cur: &'a [f32],
    sub_out: &'a mut [f32],
    lo: usize,
    n_embd: usize,
) -> (&'a [f32], &'a mut [f32]) {
    (&cur[lo..lo + n_embd], &mut sub_out[lo..lo + n_embd])
}

/// One FFN layer for a chunk of tokens.
///
/// Dense layers are per token: there is nothing to share, the weights are resident
/// and every token reads the same ones. A MoE layer batches, which is the whole
/// point of a chunk -- see [`ExpertFfn::apply_batch`].
fn ffn_chunk(
    sh: &Shape,
    w: &FfnW<'_>,
    il: usize,
    n: usize,
    xs: &[f32],
    outs: &mut [f32],
) -> Result<()> {
    let n_embd = sh.n_embd;
    match w {
        FfnW::Dense { .. } => {
            for t in 0..n {
                let lo = t * n_embd;
                let (x, out) = split_at_chunk(xs, outs, lo, n_embd);
                // The borrow split hands back a shared x; the dense arm needs the
                // same thing the one-token path passes it.
                ffn_layer(sh, w, il, x, out)?;
            }
            Ok(())
        }
        FfnW::Moe(m) => {
            // Route every token first, so the layer's experts are known as a set.
            let ne = sh.n_expert;
            let mut routes: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n);
            let t_router = std::time::Instant::now();
            for t in 0..n {
                let lo = t * n_embd;
                let mut logits = vec![0.0f32; ne];
                m.router.apply(&xs[lo..lo + n_embd], &mut logits)?;
                let (ids, weights) = routing::route_token(
                    &logits,
                    Some(m.probs_b),
                    sh.n_expert_used,
                    sh.expert_weights_norm,
                    sh.expert_weights_scale,
                );
                routes.push(ids.into_iter().zip(weights).collect());
            }
            prof::add(&prof::FFN_ROUTER, t_router);

            let t_routed = std::time::Instant::now();
            m.experts
                .apply_batch(m.ord, &routes, xs, n_embd, sh.swiglu_clamp_exp[il], outs)?;
            prof::add(&prof::FFN_ROUTED, t_routed);

            // The shared expert runs for every token and is unscaled.
            let t_shared = std::time::Instant::now();
            let sff = sh.n_ff_shexp;
            for t in 0..n {
                let lo = t * n_embd;
                let mut sgu = vec![0.0f32; 2 * sff];
                m.sh_gate_up.apply(&xs[lo..lo + n_embd], sff, &mut sgu)?;
                let (sg, su) = sgu.split_at(sff);
                let mut sh_h = vec![0.0f32; sff];
                swiglu_clamped(sg, su, sh.swiglu_clamp_shexp[il], &mut sh_h)?;
                let mut s_out = vec![0.0f32; n_embd];
                m.sh_down.apply(&sh_h, &mut s_out)?;
                for (o, &sv) in outs[lo..lo + n_embd].iter_mut().zip(s_out.iter()) {
                    *o += sv;
                }
            }
            prof::add(&prof::FFN_SHARED, t_shared);
            Ok(())
        }
    }
}

/// Run a whole prompt, returning the last token's logits. Prefill is this loop —
/// the reference's own autoregressive path, one token at a time.
pub fn forward_prompt(
    sh: &Shape,
    w: &ModelW<'_>,
    state: &mut State,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    if tokens.is_empty() {
        return Err(LlamaError::Config("forward: empty prompt".into()));
    }
    let mut last = Vec::new();
    for &t in tokens {
        last = forward_token(sh, w, state, t)?;
    }
    Ok(last)
}
