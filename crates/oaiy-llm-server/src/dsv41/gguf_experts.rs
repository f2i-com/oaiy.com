//! Routed experts of a GGUF checkpoint on the CPU ([`ExpertFormat::Gguf`](crate::dsv41::expert::ExpertFormat)): each
//! projection's rows decoded from the record's blocks as they lie (`ggml_quants`' decoders, every GGML type), and between
//! them the routed expert's own SwiGLU ([`swiglu`]: each sum rounded to bf16, clamped at the limit, `silu(gate) * up`
//! times the route weight, rounded again), its output rounded to bf16 as the other formats' is. The activation is taken
//! as it is (llama.cpp quantizes it to 8 bits for its dot products; here the sums are of the decoded weights in f32).
//! A decode step's experts go over the host's cores together, a projection's rows in runs ([`forward_step`]); a
//! prompt's one an expert on a worker of its own ([`forward_rows`]). A reference's speed, a few milliseconds an expert.

use std::sync::Mutex;

use oaiy_engine::backend::parallel_rows;
use oaiy_engine::{Error, Result};

use crate::dsv41::expert::{swiglu, GgufProj, DIM};
use crate::dsv41::formats::to_bf16;

/// Rows of a projection a run decodes and multiplies.
const RUN_ROWS: usize = 64;

/// Whether `record` holds a layer's experts as `layout` says.
pub fn check(record: &[u8], layout: &[GgufProj; 3]) -> Result<()> {
    for p in layout {
        if p.at + p.bytes() > record.len() {
            return Err(Error::Format(format!("a GGUF expert record ({} bytes) does not hold its {}x{} projection as its layer's layout says", record.len(), p.k, p.n)));
        }
    }
    Ok(())
}

/// `a . b` in sixteen lanes (which the compiler makes vector sums), the lanes then added in order.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 16;
    let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ca.zip(cb) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Row `r` of `p` in `record`, decoded into `w` (`[k]`).
fn decode_row(record: &[u8], p: &GgufProj, r: usize, w: &mut [f32]) {
    let row_bytes = p.row_bytes();
    let at = p.at + r * row_bytes;
    ggml_quants::dequantize(p.ggml(), &record[at..at + row_bytes], w).expect("a GGUF expert's rows are whole blocks, checked when its store opened");
}

/// `p`'s sums of `m` rows `x` (`[m, k]`): `[m, n]`, each of its rows decoded once, on the calling thread.
fn project(record: &[u8], p: &GgufProj, x: &[f32], m: usize) -> Vec<f32> {
    let (k, n) = (p.k, p.n);
    let mut y = vec![0.0f32; m * n];
    let mut w = vec![0.0f32; k];
    for r in 0..n {
        decode_row(record, p, r, &mut w);
        for (i, xr) in x.chunks_exact(k).enumerate() {
            y[i * n + r] = dot(xr, &w);
        }
    }
    y
}

/// One expert's rows (`x`, `[m, DIM]`) through its `record` on the calling thread (a prompt's workers are one a
/// core): gate and up, the SwiGLU (the rows' route `weights` in it), then down: `[m, DIM]`, rounded to bf16.
pub fn forward_rows(record: &[u8], layout: &[GgufProj; 3], x: &[f32], weights: Option<&[f32]>, limit: f32) -> Result<Vec<f32>> {
    check(record, layout)?;
    let m = x.len() / DIM;
    let [g, u, d] = layout;
    let (gate, up) = (project(record, g, x, m), project(record, u, x, m));
    let h = swiglu(&gate, &up, weights, limit);
    Ok(project(record, d, &h, m).into_iter().map(to_bf16).collect())
}

/// `p`'s sums for each expert `records[i]` on its one row `xs[i]` (`[k]`): `[n]` each, every expert's rows in one set
/// of runs on the host's cores (a run never crossing from one expert into the next).
fn project_each(records: &[&[u8]], p: &GgufProj, xs: &[&[f32]]) -> Vec<Vec<f32>> {
    let n = p.n;
    let out: Vec<Mutex<Vec<f32>>> = records.iter().map(|_| Mutex::new(vec![0.0f32; n])).collect();
    parallel_rows(records.len() * n, RUN_ROWS, &|b, e| {
        let mut w = vec![0.0f32; p.k];
        let mut r = b;
        while r < e {
            let (i, r0) = (r / n, r % n);
            let end = r0 + (e - r).min(n - r0);
            let part: Vec<f32> = (r0..end)
                .map(|row| {
                    decode_row(records[i], p, row, &mut w);
                    dot(xs[i], &w)
                })
                .collect();
            out[i].lock().unwrap_or_else(|p| p.into_inner())[r0..end].copy_from_slice(&part);
            r += end - r0;
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap_or_else(|p| p.into_inner())).collect()
}

/// A decode step's experts (one row, `x`), each `records[i]` with its route weight `weights[i]`, at once on the host's
/// cores: a projection's rows of every expert in one set of runs, a token's six experts being too few tasks for the
/// cores alone. Each expert's output, `[DIM]`, rounded to bf16: the same as [`forward_rows`] gives it.
pub fn forward_step(records: &[&[u8]], layout: &[GgufProj; 3], x: &[f32], weights: &[f32], limit: f32) -> Result<Vec<Vec<f32>>> {
    for r in records {
        check(r, layout)?;
    }
    let [g, u, d] = layout;
    let xs: Vec<&[f32]> = vec![x; records.len()];
    let (gate, up) = (project_each(records, g, &xs), project_each(records, u, &xs));
    let h: Vec<Vec<f32>> = (0..records.len()).map(|i| swiglu(&gate[i], &up[i], Some(&weights[i..i + 1]), limit)).collect();
    let hs: Vec<&[f32]> = h.iter().map(|v| &v[..]).collect();
    Ok(project_each(records, d, &hs).into_iter().map(|y| y.into_iter().map(to_bf16).collect()).collect())
}
