//! A layer's stages for one token: the mixes of the streams, the KDA and MLA attention layers, and the
//! feed-forward (dense, or routed experts on the host or a device).

use super::*;

/// The hyper-connection coefficients for one token.
///
/// The `[hc_mix, hc * n_embd]` projection goes through [`Mat`] like any other
/// matrix; the RMS reciprocal and the Sinkhorn tail are 24 numbers and a 4x4, so
/// they stay scalar wherever the matmul ran.
pub(super) fn hc_mixes(stream: &[f32], w: &HcW<'_>, sh: &Shape) -> Result<hc::Mix> {
    let mut proj = [0.0f32; hc::MIX];
    w.fn_.apply(stream, &mut proj)?;
    let sumsq: f32 = stream.iter().map(|v| v * v).sum();
    let r = 1.0 / (sumsq / stream.len() as f32 + sh.rms_eps).sqrt();
    Ok(hc::mixes_from_projection(
        &proj,
        r,
        w.base,
        w.scale,
        sh.hc_sinkhorn_iters,
        sh.hc_eps,
    ))
}

/// One KDA linear-attention layer. `x` is the post-`attn_norm` input; every
/// projection here reads it, including `f`, `g` and `beta` — the reference is
/// explicit that those read the layer input and **not** the convolved `q‖k‖v`.
/// One KDA layer's recurrent state, on whichever side it lives.
///
/// The [`Mat`] seam for the recurrence: `Host` runs [`kda::step`], the oracle,
/// and `Device` runs [`Backend::kda_delta_step`] against a state that never
/// leaves the card.
pub enum KdaSt<'a> {
    Host(&'a mut [f32]),
    Device {
        t: &'a mut Tensor,
        backend: &'a dyn Backend,
    },
}

pub(super) fn kda_layer(
    sh: &Shape,
    w: &KdaW<'_>,
    state: KdaSt<'_>,
    conv_state: &mut [f32],
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let (nh, hd, d_conv) = (sh.n_head, sh.kda_head_dim, sh.d_conv);
    let di = sh.d_inner();
    let cd = 3 * di;

    // q‖k‖v from the layer input.
    let t = std::time::Instant::now();
    let mut qkv = vec![0.0f32; cd];
    w.qk.apply(x, di, &mut qkv[0..2 * di])?;
    w.v.apply(x, &mut qkv[2 * di..cd])?;
    prof::add(&prof::KDA_PROJ, t);

    // One depthwise conv over the whole concatenation, SiLU on its output (not
    // on the projections). The three weights concatenate in q, k, v order.
    let t = std::time::Instant::now();
    let mut conv_out = vec![0.0f32; cd];
    for c in 0..cd {
        let (third, ch) = (c / di, c % di);
        let cw = match third {
            0 => w.conv_q,
            1 => w.conv_k,
            _ => w.conv_v,
        };
        let cw = &cw[ch * d_conv..(ch + 1) * d_conv];
        let mut acc = 0.0f32;
        for k in 0..d_conv - 1 {
            acc += conv_state[k * cd + c] * cw[k];
        }
        acc += qkv[c] * cw[d_conv - 1];
        conv_out[c] = silu(acc);
    }
    // Shift the pre-conv inputs through the state.
    for c in 0..cd {
        for k in 0..d_conv.saturating_sub(2) {
            conv_state[k * cd + c] = conv_state[(k + 1) * cd + c];
        }
        if d_conv >= 2 {
            conv_state[(d_conv - 2) * cd + c] = qkv[c];
        }
    }

    prof::add(&prof::KDA_CONV, t);

    let mut q: Vec<f32> = conv_out[0..di].to_vec();
    let mut k: Vec<f32> = conv_out[di..2 * di].to_vec();
    let v: Vec<f32> = conv_out[2 * di..cd].to_vec();

    // Per-head L2 norm at the reference's own constant.
    for h in 0..nh {
        l2_norm(&mut q[h * hd..(h + 1) * hd], 1e-6);
        l2_norm(&mut k[h * hd..(h + 1) * hd], 1e-6);
    }

    // g = lower_bound * sigmoid(-(ssm_a * (f_b(f_a(x)) + dt_bias))), per channel.
    // `ssm_a` holds -exp(A_log), so the negation inside the sigmoid recovers the
    // reference's `sigmoid(exp(A_log) * ...)`.
    let t = std::time::Instant::now();
    // f_a and g_a share x, so one matmul gives both roots; g_a's half is used by
    // the output gate further down.
    let mut roots = vec![0.0f32; 2 * hd];
    w.fga.apply(x, hd, &mut roots)?;
    let mut g_log = vec![0.0f32; di];
    w.f_b.apply(&roots[..hd], &mut g_log)?;
    for h in 0..nh {
        for i in 0..hd {
            let idx = h * hd + i;
            let t = (g_log[idx] + w.dt_bias[idx]) * w.a[h];
            g_log[idx] = sh.kda_gate_lower_bound * sigmoid(-t);
        }
    }

    let mut beta = vec![0.0f32; nh];
    w.beta.apply(x, &mut beta)?;
    for b in beta.iter_mut() {
        *b = sigmoid(*b);
    }

    prof::add(&prof::KDA_GATE, t);

    let t = std::time::Instant::now();
    let mut scan = vec![0.0f32; di];
    match state {
        KdaSt::Host(st) => kda::step(st, &q, &k, &v, &g_log, &beta, nh, hd, &mut scan)?,
        KdaSt::Device { t: st, backend } => {
            // q, k, v and g_log go up as one [4, di] tensor rather than four, so a
            // layer costs two uploads and one download instead of six crossings.
            let mut packed = Vec::with_capacity(4 * di);
            packed.extend_from_slice(&q);
            packed.extend_from_slice(&k);
            packed.extend_from_slice(&v);
            packed.extend_from_slice(&g_log);
            let qkvg = backend.to_device(Tensor::from_vec(packed, vec![4, di]));
            let bt = backend.to_device(Tensor::from_vec(beta.clone(), vec![nh]));
            let o = backend.kda_delta_step(st, &qkvg, &bt, nh, hd);
            let oh = backend.to_host(o);
            if oh.data().len() != scan.len() {
                return Err(LlamaError::Config(format!(
                    "forward: kda step returned {} values, expected {}",
                    oh.data().len(),
                    scan.len()
                )));
            }
            scan.copy_from_slice(oh.data());
        }
    }
    prof::add(&prof::KDA_STEP, t);

    // Per-head RMSNorm by ssm_norm, gated by a PLAIN sigmoid of g_b(g_a(x)) --
    // not the SiLU a FusedRMSNormGated would default to.
    let mut gate = vec![0.0f32; di];
    w.g_b.apply(&roots[hd..], &mut gate)?;
    for h in 0..nh {
        let head = &mut scan[h * hd..(h + 1) * hd];
        rms_norm(head, w.o_norm, sh.rms_eps)?;
        for (i, s) in head.iter_mut().enumerate() {
            *s *= sigmoid(gate[h * hd + i]);
        }
    }

    w.out.apply(&scan, out)
}

/// One full-attention layer: absorbed MLA plus the sparse indexer.
#[allow(clippy::too_many_arguments)]
pub(super) fn mla_layer(
    sh: &Shape,
    w: &MlaW<'_>,
    latents: &mut [f32],
    // VENDORED-LOCAL: GLM-5.3-Flash. The latent cache on a card, when there is one,
    // so the attention over it runs there: see `mla::attend_latent_device`.
    latents_dev: Option<(&mut Tensor, &dyn Backend)>,
    kp: &mut kpool::KpoolCache,
    kp_layer: usize,
    pos: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let (nh, r, d_idx) = (sh.n_head, sh.kpool, sh.d_idx);
    let len = pos + 1;

    // (a) the shared query root.
    let t = std::time::Instant::now();
    let mut qr = vec![0.0f32; sh.q_lora];
    w.q_a.apply(x, &mut qr)?;
    rms_norm(&mut qr, w.q_a_norm, sh.rms_eps)?;
    prof::add(&prof::MLA_PROJ, t);
    let t_ix = std::time::Instant::now();

    // (b) store this cell's indexer key and compressor gate. Unconditional: the
    // reference stores on the dense path too, or cells written below the
    // scoring threshold would have no indexer state once a later batch crosses
    // it.
    let mut ik = vec![0.0f32; d_idx];
    w.indexer.attn_k.apply(x, &mut ik)?;
    layer_norm(
        &mut ik,
        w.indexer.k_norm,
        w.indexer.k_norm_bias,
        sh.norm_eps,
    )?;
    let mut ig = vec![0.0f32; d_idx];
    w.indexer.comp_gate.apply(x, &mut ig)?;
    kp.store(kp_layer, pos, &ik, &ig)?;

    // (c) this token may have just completed a pool.
    if kpool::slot_of(pos, r) == r - 1 {
        let pool = kpool::pool_of(pos, r);
        let mut keys = vec![0.0f32; r * d_idx];
        let mut gates = vec![0.0f32; r * d_idx];
        for (s, cell) in kpool::pool_members(pool, r).enumerate() {
            keys[s * d_idx..(s + 1) * d_idx].copy_from_slice(kp.key(kp_layer, cell)?);
            gates[s * d_idx..(s + 1) * d_idx].copy_from_slice(kp.gate(kp_layer, cell)?);
        }
        let mut pooled = vec![0.0f32; d_idx];
        indexer::pooled_key(&keys, &gates, w.indexer.comp_ape, r, d_idx, &mut pooled)?;
        kp.set_pooled(kp_layer, pool, r, &pooled)?;
    }

    // (d) score and select, or take the dense path.
    let selected: Option<Vec<u32>> =
        if kpool::indexer_scores(sh.max_len, sh.indexer_top_k, r) {
            let mut iq = vec![0.0f32; sh.n_ihead * d_idx];
            w.indexer.attn_q_b.apply(&qr, &mut iq)?;
            let mut hw = vec![0.0f32; sh.n_ihead];
            w.indexer.proj.apply(x, &mut hw)?;

            let n_pools = kpool::n_pools(len, r);
            let mut pool_keys = vec![0.0f32; n_pools * d_idx];
            let complete = kpool::n_complete_pools(len, r);
            for p in 0..complete {
                pool_keys[p * d_idx..(p + 1) * d_idx]
                    .copy_from_slice(kp.pooled(kp_layer, p, r)?);
            }
            let mut bias = vec![0.0f32; n_pools];
            kpool::pool_bias_row(pos, len, r, &mut bias)?;

            Some(indexer::candidates(
                &iq,
                &hw,
                &pool_keys,
                &bias,
                sh.n_ihead,
                d_idx,
                len,
                r,
                sh.indexer_top_k,
            )?)
        } else {
            None
        };

    prof::add(&prof::MLA_INDEX, t_ix);

    // (e) the latent, cached before attending so the token sees itself.
    let t = std::time::Instant::now();
    let mut latent = vec![0.0f32; sh.kv_lora];
    w.kv_a_mqa.apply(x, &mut latent)?;
    rms_norm(&mut latent, w.kv_a_norm, sh.rms_eps)?;
    latents[pos * sh.kv_lora..(pos + 1) * sh.kv_lora].copy_from_slice(&latent);
    // And into the mirror, if there is one: a device-side copy of `kv_lora` floats
    // into row `pos`, which is what lets the attention read the whole cache without
    // it crossing PCIe every token.
    let latents_dev = match latents_dev {
        Some((cache, be)) => {
            let row = be.to_device(Tensor::from_vec(latent.clone(), vec![1, 1, sh.kv_lora]));
            be.copy_axis0_into(cache, pos, &row);
            Some((&*cache, be))
        }
        None => None,
    };

    // (f) absorbed attention over the candidate set.
    let mut q = vec![0.0f32; nh * sh.qk_head];
    w.q_b.apply(&qr, &mut q)?;
    prof::add(&prof::MLA_PROJ, t);
    let t = std::time::Instant::now();
    let mut q_abs = vec![0.0f32; nh * sh.kv_lora];
    w.k_b
        .apply(&q, nh, sh.kv_lora, sh.qk_head, &mut q_abs)?;
    prof::add(&prof::MLA_ABSORB, t);

    let mut mask = vec![0.0f32; len];
    mla::attn_mask(pos, len, r, selected.as_deref(), &mut mask)?;

    // On the card when the cache is there, and on the host otherwise -- which is
    // the host oracle's path, and the one `bridge::HostModel` is checked against.
    //
    // This used to say that the softmax and the weighted sum "stay on the host: they
    // only touch the latents, which are at most len * kv_lora". That reasoning was
    // wrong about which part is expensive: `len * kv_lora` grows with the prompt, so
    // it is O(len) work a token and O(n^2) over a prompt, and it measured 124.25 ms
    // a token at position 1740 against the absorb's 3.76.
    let t = std::time::Instant::now();
    let mut ctx = vec![0.0f32; nh * sh.kv_lora];
    match latents_dev {
        Some((cache, be)) => mla::attend_latent_device(
            be,
            &q_abs,
            cache,
            &mask,
            mla::kq_scale(sh.qk_head),
            nh,
            sh.kv_lora,
            len,
            &mut ctx,
        )?,
        None => mla::attend_latent(
            &q_abs,
            &latents[..len * sh.kv_lora],
            &mask,
            mla::kq_scale(sh.qk_head),
            nh,
            sh.kv_lora,
            len,
            &mut ctx,
        )?,
    }
    let mut attn = vec![0.0f32; nh * sh.v_head];
    w.v_b
        .apply(&ctx, nh, sh.v_head, sh.kv_lora, &mut attn)?;
    prof::add(&prof::MLA_ATTEND, t);

    w.out.apply(&attn, out)
}

/// The FFN half of a layer: a plain clamped SwiGLU on the leading dense blocks,
/// VENDORED-LOCAL: PERF. One MoE layer with the activations kept on the card.
///
/// The host arm of [`ffn_layer`] costs eight round trips a layer: the router, the
/// routed experts, the shared expert's two halves, and a host-side SwiGLU between
/// them, each an upload, a launch and a synchronising download. This costs three,
/// and two of those are inside the expert tier's own seam.
///
/// What is left is one genuine hop: the 288 router logits. Picking the top eight
/// with a bias, a normalisation and a scale is a host decision, and the expert
/// cache is indexed on the host, so the ids have to exist here. That is the floor
/// for this layer, not an omission.
///
/// The shared expert never touches the host: gate||up, the clamped SwiGLU
/// (`swiglu_clamped_split`, which CUDA does in one kernel) and down chain on the
/// card. Its result is added unscaled, because `expert_weights_scale` applies to
/// the routed weights only.
fn ffn_moe_device(
    sh: &Shape,
    m: &MoeW<'_>,
    il: usize,
    be: &dyn Backend,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    // One upload for the layer: the router and the shared expert read the same x.
    let xd = be.to_device(Tensor::from_vec(x.to_vec(), vec![1, sh.n_embd]));

    let t_router = std::time::Instant::now();
    let logits_d = m
        .router
        .linear_dev(&xd)
        .ok_or_else(|| LlamaError::Config("forward: device router lost its device".into()))?;
    let logits_h = be.to_host(logits_d);
    if logits_h.numel() != sh.n_expert {
        return Err(LlamaError::Config(format!(
            "forward: router produced {} logits, expected {}",
            logits_h.numel(),
            sh.n_expert
        )));
    }
    let (ids, weights) = routing::route_token(
        logits_h.data(),
        Some(m.probs_b),
        sh.n_expert_used,
        sh.expert_weights_norm,
        sh.expert_weights_scale,
    );
    prof::add(&prof::FFN_ROUTER, t_router);

    // The routed experts keep the host seam: it is where the three tiers live, and
    // a miss may be computed on the CPU from a record in RAM.
    let t_routed = std::time::Instant::now();
    let route: Vec<(u32, f32)> = ids.iter().copied().zip(weights.iter().copied()).collect();
    m.experts
        .apply_layer(m.ord, &route, x, sh.swiglu_clamp_exp[il], out)?;
    prof::add(&prof::FFN_ROUTED, t_routed);

    let t_shared = std::time::Instant::now();
    let sff = sh.n_ff_shexp;
    let gu = m
        .sh_gate_up
        .linear_dev(&xd)
        .ok_or_else(|| LlamaError::Config("forward: device shared pair lost its device".into()))?;
    if gu.numel() != 2 * sff {
        return Err(LlamaError::Config(format!(
            "forward: shared gate||up produced {} values, expected {}",
            gu.numel(),
            2 * sff
        )));
    }
    let h = be.swiglu_clamped_split(&gu, sff, sh.swiglu_clamp_shexp[il], true);
    let s_out = m
        .sh_down
        .linear_dev(&h)
        .ok_or_else(|| LlamaError::Config("forward: device shared down lost its device".into()))?;
    let s_host = be.to_host(s_out);
    if s_host.numel() != sh.n_embd {
        return Err(LlamaError::Config(format!(
            "forward: shared expert produced {} values, expected {}",
            s_host.numel(),
            sh.n_embd
        )));
    }
    for (o, &sv) in out.iter_mut().zip(s_host.data()) {
        *o += sv;
    }
    prof::add(&prof::FFN_SHARED, t_shared);
    Ok(())
}

/// or routed experts plus an **unscaled** shared expert after them.
pub(super) fn ffn_layer(sh: &Shape, w: &FfnW<'_>, il: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    match w {
        FfnW::Dense { gate, up, down } => {
            let ff = sh.n_ff_dense;
            let mut g = vec![0.0f32; ff];
            let mut u = vec![0.0f32; ff];
            gate.apply(x, &mut g)?;
            up.apply(x, &mut u)?;
            let mut h = vec![0.0f32; ff];
            // build_ffn reads the *shexp* clamp array for every call it makes,
            // so the dense blocks are clamped by it too.
            swiglu_clamped(&g, &u, sh.swiglu_clamp_shexp[il], &mut h)?;
            down.apply(&h, out)
        }
        FfnW::Moe(m) => {
            // VENDORED-LOCAL: PERF. With the weights on a card, this layer's chain
            // runs there: see `ffn_moe_device`. The host arm below stays because
            // `bridge::HostModel` is the oracle both are checked against.
            if let (Some(be), Some(_)) = (m.router.device(), m.sh_gate_up.device()) {
                return ffn_moe_device(sh, m, il, be, x, out);
            }
            let ne = sh.n_expert;
            let mut logits = vec![0.0f32; ne];
            let t_router = std::time::Instant::now();
            m.router.apply(x, &mut logits)?;
            let (ids, weights) = routing::route_token(
                &logits,
                Some(m.probs_b),
                sh.n_expert_used,
                sh.expert_weights_norm,
                sh.expert_weights_scale,
            );

            // One call for the layer, not one per expert: see
            // [`ExpertFfn::apply_layer`].
            let route: Vec<(u32, f32)> =
                ids.iter().copied().zip(weights.iter().copied()).collect();
            prof::add(&prof::FFN_ROUTER, t_router);
            let t_routed = std::time::Instant::now();
            m.experts
                .apply_layer(m.ord, &route, x, sh.swiglu_clamp_exp[il], out)?;
            prof::add(&prof::FFN_ROUTED, t_routed);
            let t_shared = std::time::Instant::now();

            // The shared expert is added unscaled: expert_weights_scale applies
            // to the routed weights only.
            let sff = sh.n_ff_shexp;
            let mut sgu = vec![0.0f32; 2 * sff];
            m.sh_gate_up.apply(x, sff, &mut sgu)?;
            let (sg, su) = sgu.split_at(sff);
            let mut sh_h = vec![0.0f32; sff];
            swiglu_clamped(sg, su, sh.swiglu_clamp_shexp[il], &mut sh_h)?;
            let mut s_out = vec![0.0f32; sh.n_embd];
            m.sh_down.apply(&sh_h, &mut s_out)?;
            for (o, &sv) in out.iter_mut().zip(s_out.iter()) {
                *o += sv;
            }
            prof::add(&prof::FFN_SHARED, t_shared);
            Ok(())
        }
    }
}
