use candle_core::{DType, Result, Tensor, D};

pub fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?)?
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(x.dtype())
}

pub fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    let f = f.broadcast_sub(&f.mean_keepdim(D::Minus1)?)?;
    f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?
        .to_dtype(x.dtype())
}

/// [batch, heads, sequence, dimension], with a causal text prefix and a
/// bidirectional image suffix. Query chunking bounds the score allocation.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, text_len: usize) -> Result<Tensor> {
    let (_, _, nq, dim) = q.dims4()?;
    let nk = k.dim(2)?;
    let kt = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;
    let mut chunks = Vec::new();
    for start in (0..nq).step_by(128) {
        let len = 128.min(nq - start);
        let mut scores = (q.narrow(2, start, len)?.contiguous()?.matmul(&kt)?
            / (dim as f64).sqrt())?
        .to_dtype(DType::F32)?;
        if start < text_len {
            let mask: Vec<f32> = (start..start + len)
                .flat_map(|i| {
                    (0..nk).map(move |j| {
                        if i < text_len && j > i {
                            f32::NEG_INFINITY
                        } else {
                            0.
                        }
                    })
                })
                .collect();
            scores = scores.broadcast_add(&Tensor::from_vec(mask, (1, 1, len, nk), q.device())?)?;
        }
        chunks.push(
            candle_nn::ops::softmax_last_dim(&scores)?
                .to_dtype(q.dtype())?
                .matmul(&v)?,
        );
    }
    Tensor::cat(&chunks, 2)
}

pub fn heads(x: &Tensor, n: usize) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    x.reshape((b, s, n, d / n))?.transpose(1, 2)?.contiguous()
}

pub fn unheads(x: &Tensor) -> Result<Tensor> {
    let (b, n, s, d) = x.dims4()?;
    x.transpose(1, 2)?.contiguous()?.reshape((b, s, n * d))
}

/// Causal text segments and bidirectional reference blocks can see their prefix,
/// never a later segment. Flash's causal mask is bottom-right aligned.
pub fn block_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    segments: &[(usize, bool)],
) -> Result<Tensor> {
    let mut parts = Vec::new();
    let mut start = 0;
    for &(len, causal) in segments {
        if len == 0 {
            continue;
        }
        let end = start + len;
        parts.push(prefix_attention(
            &q.narrow(2, start, len)?,
            &k.narrow(2, 0, end)?,
            &v.narrow(2, 0, end)?,
            causal,
        )?);
        start = end;
    }
    Tensor::cat(&parts, 2)
}
pub fn prefix_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    if !causal {
        return attention(q, k, v, 0);
    }
    let (_, _, nq, d) = q.dims4()?;
    let nk = k.dim(2)?;
    let mut parts = Vec::new();
    for start in (0..nq).step_by(128) {
        let n = 128.min(nq - start);
        let scores = (q
            .narrow(2, start, n)?
            .contiguous()?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            / (d as f64).sqrt())?
        .to_dtype(DType::F32)?;
        let mask = (start..start + n)
            .flat_map(|i| {
                (0..nk).map(move |j| {
                    if j > nk - nq + i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect::<Vec<_>>();
        let scores = scores.broadcast_add(&Tensor::from_vec(mask, (1, 1, n, nk), q.device())?)?;
        parts.push(
            candle_nn::ops::softmax_last_dim(&scores)?
                .to_dtype(q.dtype())?
                .matmul(&v.contiguous()?)?,
        );
    }
    Tensor::cat(&parts, 2)
}
#[test]
fn references_are_bidirectional_with_causal_text_between() -> Result<()> {
    let q = Tensor::zeros((1, 1, 7, 1), DType::F32, &candle_core::Device::Cpu)?;
    let v = Tensor::from_vec(
        vec![0f32, 2., 4., 6., 8., 10., 12.],
        (1, 1, 7, 1),
        q.device(),
    )?;
    let y = block_attention(
        &q,
        &q,
        &v,
        &[(1, true), (2, false), (1, true), (2, false), (1, true)],
    )?
    .flatten_all()?
    .to_vec1::<f32>()?;
    for (a, b) in y.iter().zip([0., 2., 2., 3., 5., 5., 6.]) {
        assert!((a - b).abs() < 1e-5);
    }
    Ok(())
}

pub fn swiglu(
    x: &Tensor,
    gate: &crate::weights::Linear,
    up: &crate::weights::Linear,
    down: &crate::weights::Linear,
) -> Result<Tensor> {
    down.forward(&(candle_nn::ops::silu(&gate.forward(x)?)? * up.forward(x)?)?)
}

#[test]
fn text_is_causal_but_image_is_bidirectional() -> Result<()> {
    let dev = candle_core::Device::Cpu;
    let q = Tensor::zeros((1, 1, 4, 1), DType::F32, &dev)?;
    let v = Tensor::from_vec(vec![2f32, 4., 10., 20.], (1, 1, 4, 1), &dev)?;
    let out = attention(&q, &q, &v, 2)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(out, vec![2., 3., 9., 9.]);
    Ok(())
}

#[test]
fn norms_use_the_channel_axis() -> Result<()> {
    let dev = candle_core::Device::Cpu;
    let x = Tensor::new(&[[1f32, 3.], [4., 4.]], &dev)?;
    let norm = layer_norm(&x)?.to_vec2::<f32>()?;
    assert!((norm[0][0] + 1.).abs() < 1e-5);
    assert!((norm[0][1] - 1.).abs() < 1e-5);
    assert_eq!(norm[1], vec![0., 0.]);
    let r = rms(&x, &Tensor::ones(2, DType::F32, &dev)?, 1e-6)?.to_vec2::<f32>()?;
    assert!((r[1][0] - 1.).abs() < 1e-6);
    Ok(())
}

#[test]
fn causal_mask_and_image_suffix_cross_chunk_boundary() -> Result<()> {
    let dev = candle_core::Device::Cpu;
    let n = 1031;
    let q = Tensor::zeros((1, 1, n, 1), DType::F32, &dev)?;
    let v = Tensor::from_vec((0..n).map(|i| i as f32).collect(), (1, 1, n, 1), &dev)?;
    let out = attention(&q, &q, &v, 1027)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (i, actual) in out.into_iter().enumerate() {
        let expected = if i < 1027 {
            i as f32 / 2.
        } else {
            (n - 1) as f32 / 2.
        };
        assert!(
            (actual - expected).abs() < 0.002,
            "query {i}: {actual} vs {expected}"
        );
    }
    Ok(())
}

